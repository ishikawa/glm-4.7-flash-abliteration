#!/usr/bin/env python3
"""Validate a release and display its upload plan; --yes invokes the hf CLI."""

import argparse
import hashlib
import re
import shlex
import subprocess
import sys
from pathlib import Path

DEFAULT_REPO = "takanori-ishikawa/GLM-4.7-Flash-Abliterated-UD-Q4_K_XL-GGUF"
FILENAME = "GLM-4.7-Flash-Abliterated-UD-Q4_K_XL.gguf"
HF_DIR = Path(__file__).resolve().parents[1] / "hf"
DOCUMENTS = ("README.md", "LICENSE", "NOTICE")


def release_metadata(directory):
    for name in DOCUMENTS:
        if not (directory / name).is_file():
            raise ValueError(f"missing release document: {name}")
    text = (directory / "README.md").read_text(encoding="utf-8")
    rows = re.findall(
        r"^\| `([^`]+\.gguf)` \| ([0-9,]+) \| `([a-f0-9]{64})` \|$",
        text,
        re.MULTILINE,
    )
    if len(rows) != 1 or rows[0][0] != FILENAME:
        raise ValueError("model card must contain one release file/size/SHA-256 row")
    _, size, digest = rows[0]
    return int(size.replace(",", "")), digest


def validate_gguf(path, size, digest, *, required):
    if not path.exists():
        if required:
            raise ValueError(f"GGUF does not exist: {path}")
        print(
            f"Warning: GGUF missing; size and SHA-256 not verified: {path}",
            file=sys.stderr,
        )
        return False
    if not path.is_file() or path.stat().st_size != size:
        raise ValueError("GGUF size differs from hf/README.md")
    with path.open("rb") as stream:
        actual = hashlib.file_digest(stream, "sha256").hexdigest()
    if actual != digest:
        raise ValueError("GGUF SHA-256 differs from hf/README.md")
    return True


def upload_plan(repo_id, gguf, directory):
    commands = [["hf", "repos", "create", repo_id, "--type", "model", "--exist-ok"]]
    for local, remote in [(gguf, FILENAME), *((directory / n, n) for n in DOCUMENTS)]:
        commands.append(
            ["hf", "upload", repo_id, str(local), remote, "--type", "model"]
        )
    return commands


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("gguf", type=Path)
    parser.add_argument("--repo-id", default=DEFAULT_REPO)
    parser.add_argument("--yes", action="store_true", help="Create the repo and upload")
    args = parser.parse_args(argv)
    if not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*", args.repo_id
    ):
        parser.error("--repo-id must be namespace/name")
    gguf = args.gguf.expanduser().absolute()
    try:
        size, digest = release_metadata(HF_DIR)
        verified = validate_gguf(gguf, size, digest, required=args.yes)
        commands = upload_plan(args.repo_id, gguf, HF_DIR)
        print(f"Repository: {args.repo_id}")
        print(f"Mode: {'upload' if args.yes else 'dry-run'}")
        status = "verified" if verified else "expected"
        print(f"{gguf} -> {FILENAME}: {size} bytes ({status})")
        for name in DOCUMENTS:
            path = HF_DIR / name
            print(f"{path} -> {name}: {path.stat().st_size} bytes")
        for command in commands:
            print(shlex.join(command))
        if args.yes:
            for command in commands:
                subprocess.run(command, check=True)
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        parser.exit(1, f"Error: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
