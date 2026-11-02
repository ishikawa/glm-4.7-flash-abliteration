#!/usr/bin/env python3
"""Fetch selected BF16 tensors with validated, resumable HTTP Range requests."""

import argparse
import hashlib
import json
import math
import os
import struct
import urllib.request
from pathlib import Path

from direction_profiles import GLM_KINDS, PROFILES, glm_targets, resolve_index

KINDS = GLM_KINDS


def layers(value):
    result = set()
    for item in value.split(","):
        ends = item.split("-")
        lo, hi = int(ends[0]), int(ends[-1])
        if len(ends) > 2 or not 0 <= lo <= hi < 47:
            raise ValueError("layers must be in 0..46")
        result.update(range(lo, hi + 1))
    return sorted(result)


def request(url, start=None, end=None):
    headers = {}
    if os.environ.get("HF_TOKEN"):
        headers["Authorization"] = "Bearer " + os.environ["HF_TOKEN"]
    if start is not None:
        headers["Range"] = f"bytes={start}-{end}"
    with urllib.request.urlopen(
        urllib.request.Request(url, headers=headers), timeout=60
    ) as r:
        if start is not None:
            expected = f"bytes {start}-{end}/"
            if r.status != 206 or not r.headers.get("Content-Range", "").startswith(
                expected
            ):
                raise ValueError("server did not honor Range")
        data = r.read((end - start + 2) if start is not None else 64 * 1024 * 1024)
    if start is not None and len(data) != end - start + 1:
        raise ValueError("Range size mismatch")
    return data


def read_header(read):
    size = struct.unpack("<Q", read(0, 7))[0]
    if not 2 <= size <= 64 * 1024 * 1024:
        raise ValueError("invalid safetensors header size")
    raw = read(8, 7 + size)
    return json.loads(raw), 8 + size


def validate_entry(entry):
    shape = entry["shape"]
    start, end = entry["data_offsets"]
    if (
        entry["dtype"] != "BF16"
        or not shape
        or any(not isinstance(x, int) or x <= 0 for x in shape)
    ):
        raise ValueError("expected BF16 tensor with positive shape")
    if start < 0 or end - start != math.prod(shape) * 2:
        raise ValueError("tensor size mismatch")
    return start, end


def fetch_tensor(url, name, output, http=request, chunk_size=4 * 1024 * 1024):
    header, base = read_header(lambda a, b: http(url, a, b))
    entry = header[name]
    start, end = validate_entry(entry)
    output = Path(output)
    part = output.with_suffix(output.suffix + ".part")
    state = output.with_suffix(output.suffix + ".part.json")
    identity = dict(
        url=url,
        name=name,
        entry=entry,
        header_sha256=hashlib.sha256(
            json.dumps(header, sort_keys=True).encode()
        ).hexdigest(),
    )
    if part.exists() or state.exists():
        if not part.exists() or not state.exists():
            raise ValueError("orphan partial download")
        previous = json.loads(state.read_text())
        committed = previous["downloaded"]
        if (
            previous["identity"] != identity
            or not isinstance(committed, int)
            or not 0 <= committed <= part.stat().st_size <= end - start
        ):
            raise ValueError("partial header/size mismatch")
        # A crash after payload fsync but before state rename may leave a tail.
        with part.open("r+b") as f:
            f.truncate(committed)
    else:
        part.touch()
        state.write_text(json.dumps(dict(identity=identity, downloaded=0)))
    done = part.stat().st_size
    if done > end - start:
        raise ValueError("oversized partial")
    with part.open("ab") as f:
        while done < end - start:
            count = min(chunk_size, end - start - done)
            data = http(url, base + start + done, base + start + done + count - 1)
            if len(data) != count:
                raise ValueError("Range size mismatch")
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
            done += count
            tmp = state.with_suffix(".tmp")
            tmp.write_text(json.dumps(dict(identity=identity, downloaded=done)))
            tmp.replace(state)
    local = {
        name: dict(dtype="BF16", shape=entry["shape"], data_offsets=[0, end - start])
    }
    raw = json.dumps(local).encode()
    raw += b" " * (-len(raw) % 8)
    temp = output.with_suffix(output.suffix + ".tmp")
    with temp.open("wb") as f, part.open("rb") as src:
        f.write(struct.pack("<Q", len(raw)) + raw)
        while data := src.read(chunk_size):
            f.write(data)
    temp.replace(output)
    part.unlink()
    state.unlink()


def selected_targets(profile, config, selected_layers, kinds):
    if profile != "glm47":
        raise ValueError("unsupported profile")
    if config is None:
        raise ValueError("GLM requires --config-json")
    return glm_targets(config, selected_layers or [], kinds)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--profile", choices=PROFILES, default="glm47")
    p.add_argument("--config-json", type=Path)
    p.add_argument("--index-json", type=Path)
    p.add_argument("--list-only", action="store_true")
    p.add_argument("--repo")
    p.add_argument("--revision", default="main")
    p.add_argument("--layers", type=layers)
    p.add_argument("--kinds")
    p.add_argument("--output-dir", type=Path)
    a = p.parse_args()
    kinds = (a.kinds or ",".join(GLM_KINDS)).split(",")
    if set(kinds) - set(KINDS):
        p.error("unknown kind")
    if not a.layers and any(k != "token_embd" for k in kinds):
        p.error("--layers required for layer kinds")
    if not a.list_only and a.output_dir is None:
        p.error("--output-dir required")
    repo = a.repo or "zai-org/GLM-4.7-Flash"
    root = f"https://huggingface.co/{repo}/resolve/{a.revision}/"
    index = json.loads(
        a.index_json.read_text()
        if a.index_json
        else request(root + "model.safetensors.index.json")
    )["weight_map"]
    config = json.loads(a.config_json.read_text()) if a.config_json else None
    targets = selected_targets(a.profile, config, a.layers, kinds)
    listing = resolve_index(targets, index)
    if a.list_only:
        print(
            json.dumps(
                dict(
                    profile=a.profile,
                    gguf_count=len(listing),
                    hf_count=sum(len(t["hf"]) for t in listing),
                    tensors=listing,
                ),
                indent=2,
            )
        )
        return
    a.output_dir.mkdir(parents=True, exist_ok=True)
    for _, _, gg, names in targets:
        for expert, hf in enumerate(names):
            # Read individual files in numeric expert order in direction_edit.
            suffix = f".expert-{expert}" if len(names) > 1 else ""
            output = a.output_dir / (gg + suffix + ".safetensors")
            if not output.exists():
                fetch_tensor(root + index[hf], hf, output)


if __name__ == "__main__":
    main()
