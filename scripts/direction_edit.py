#!/usr/bin/env python3
"""Clone GGUF shards and project output weights, streaming fused experts."""

import argparse
import ctypes
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from bf16_fetch import KINDS, layers, read_header, selected_targets, validate_entry
from direction_profiles import PROFILES
from gguf_header_scan import GGML_TYPES, parse

try:
    from gguf import GGMLQuantizationType
    from gguf.quants import dequantize
except ModuleNotFoundError:
    if not os.environ.get("LLAMA_CPP_DIR"):
        raise
    sys.path.insert(0, str(Path(os.environ["LLAMA_CPP_DIR"]) / "gguf-py"))
    from gguf import GGMLQuantizationType
    from gguf.quants import dequantize


class Quantizer:
    def __init__(self, path=None):
        library = path or os.environ.get("GGML_BASE_LIB")
        if not library:
            raise ValueError("set --ggml-lib or GGML_BASE_LIB")
        self.lib = ctypes.CDLL(str(library))
        self.lib.ggml_quantize_init.argtypes = [ctypes.c_int]
        self.lib.ggml_quantize_init.restype = None
        self.lib.ggml_quantize_chunk.argtypes = [
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_void_p,
        ]
        self.lib.ggml_quantize_chunk.restype = ctypes.c_size_t

    def quantize(self, values, typ):
        values = np.ascontiguousarray(values, dtype=np.float32)
        _, block, size = GGML_TYPES[int(typ)]
        if values.ndim != 2 or values.shape[1] % block:
            raise ValueError("quantization row alignment")
        result = np.empty(
            (values.shape[0], values.shape[1] // block * size), dtype=np.uint8
        )
        self.lib.ggml_quantize_init(int(typ))
        written = self.lib.ggml_quantize_chunk(
            int(typ),
            values.ctypes.data,
            result.ctypes.data,
            0,
            values.shape[0],
            values.shape[1],
            None,
        )
        if written != result.nbytes:
            raise ValueError(f"quantized size mismatch: {written} != {result.nbytes}")
        return result


def cosine(x, y):
    x, y = x.astype(np.float64).ravel(), y.astype(np.float64).ravel()
    den = np.linalg.norm(x) * np.linalg.norm(y)
    return float(x @ y / den) if den else 0.0


def layer_k(value):
    """Parse disjoint layer/range strengths, leaving unspecified layers to --k."""
    result = {}
    try:
        for item in value.split(","):
            span, strength = item.split(":")
            strength = float(strength)
            if not np.isfinite(strength):
                raise ValueError
            for layer in layers(span):
                if layer in result:
                    raise ValueError
                result[layer] = strength
    except (ValueError, TypeError) as exc:
        raise argparse.ArgumentTypeError("invalid/overlapping layer-k ranges") from exc
    return result


def biproject(u, benign):
    """Two-pass Gram-Schmidt against a benign mean, followed by unit scaling."""
    u, benign = np.asarray(u, dtype=np.float64), np.asarray(benign, dtype=np.float64)
    if (
        u.ndim != 1
        or benign.shape != u.shape
        or not np.isfinite(u).all()
        or not np.isfinite(benign).all()
        or np.linalg.norm(u) == 0
        or np.linalg.norm(benign) == 0
    ):
        raise ValueError("invalid biprojection direction/benign mean")
    original = u / np.linalg.norm(u)
    b = benign / np.linalg.norm(benign)
    v = original.copy()
    for _ in range(2):
        v -= (v @ b) * b
    norm = np.linalg.norm(v)
    if norm <= 1e-12:
        raise ValueError("biprojection leaves no direction")
    v /= norm
    return v.astype(np.float32), float(original @ v)


def orthonormalize(directions):
    """Build a row basis in input order, rejecting dependent directions."""
    basis = []
    for direction in directions:
        v = np.asarray(direction, dtype=np.float64).copy()
        if v.ndim != 1 or not np.isfinite(v).all() or np.linalg.norm(v) == 0:
            raise ValueError("invalid subspace direction")
        v /= np.linalg.norm(v)
        for _ in range(2):
            for previous in basis:
                v -= (v @ previous) * previous
        norm = np.linalg.norm(v)
        if norm <= 1e-7:
            raise ValueError("linearly dependent directions")
        basis.append(v / norm)
    return np.asarray(basis, dtype=np.float32)


def direction_shape(u, width):
    return u.shape == (width,) or (
        u.ndim == 2 and 1 < u.shape[0] <= width and u.shape[1] == width
    )


def component_functions(u, row_space):
    if u.ndim == 2:
        if row_space:
            return (lambda m: m @ u.T), (lambda v: v @ u)
        return (lambda m: u @ m), (lambda v: u.T @ v)
    component = (lambda m: m @ u) if row_space else (lambda m: u @ m)
    expand = (lambda v: np.outer(v, u)) if row_space else (lambda v: np.outer(u, v))
    return component, expand


def vector_projection(v, u):
    return (v @ u.T) @ u if u.ndim == 2 else (v @ u)[..., None] * u


def norm_preserved_target(w, u, k=1.0, row_space=False):
    """Return a fixed FP32 target with norms in u's space preserved.

    Linear W=(out,in): preserve columns (axis -2); embeddings=(vocab,hidden):
    preserve rows (axis -1). Leading expert axes are independent. For k != 1,
    normalize W-k*u*u.T*W back to each original vector norm, preserving partial
    removal's meaning. For k=1 use the reference's normalized-vector projection,
    normalization, second full projection, normalization and original magnitude.
    Nonzero vectors that collapse have no norm-preserving direction: reject.
    """
    w, u = np.asarray(w, dtype=np.float32), np.asarray(u, dtype=np.float32)
    axis = -1 if row_space else -2
    if (
        w.ndim not in (2, 3)
        or not direction_shape(u, w.shape[axis])
        or not np.isfinite(w).all()
        or not np.isfinite(u).all()
        or np.linalg.norm(u) == 0
        or not np.isfinite(k)
    ):
        raise ValueError("invalid norm-preserving weight/direction/k")
    u = orthonormalize(u) if u.ndim == 2 else u / np.linalg.norm(u)
    vectors = np.moveaxis(w, axis, -1)
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    v = vectors / np.maximum(norms, 1e-12)
    v = v - (k * vector_projection(v, u) if u.ndim == 2 else k * (v @ u)[..., None] * u)
    if k == 1:
        if np.any(
            (norms > 0)
            & (
                np.linalg.norm(v, axis=-1, keepdims=True)
                <= 4 * np.finfo(np.float32).eps
            )
        ):
            raise ValueError("norm preservation collapsed a nonzero vector")
        v = v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12)
        v = v - vector_projection(v, u)
    after = np.linalg.norm(v, axis=-1, keepdims=True)
    if np.any((norms > 0) & (after <= 1e-12)):
        raise ValueError("norm preservation collapsed a nonzero vector")
    v = v / np.maximum(after, 1e-12) * norms
    return np.moveaxis(v, -1, axis)


def norm_drift(w, original, row_space):
    axis = -1 if row_space else -2
    before = np.linalg.norm(original.astype(np.float64), axis=axis)
    after = np.linalg.norm(w.astype(np.float64), axis=axis)
    # Zero source vectors use absolute drift (unit denominator).
    relative = np.abs(after - before) / np.where(before > 0, before, 1)
    return dict(max=float(relative.max()), mean=float(relative.mean()))


def edit_matrix(
    w, u, typ, quantizer, iterations=4, k=1.0, row_space=False, norm_preserve=False
):
    w = np.asarray(w, dtype=np.float32)
    u = np.asarray(u, dtype=np.float32)
    if (
        not direction_shape(u, w.shape[1] if row_space else w.shape[0])
        or not np.isfinite(w).all()
        or not np.isfinite(u).all()
        or np.linalg.norm(u) == 0
    ):
        raise ValueError("invalid weight/direction")
    u = orthonormalize(u) if u.ndim == 2 else u / np.linalg.norm(u)
    if norm_preserve:
        return edit_norm_preserved(w, u, typ, quantizer, iterations, k, row_space)
    component, expand = component_functions(u, row_space)
    den = float(np.linalg.norm(component(w)))
    # Keep the target as a vector, including for the large token embedding table.
    # Each iteration restores (1-k) times the original component, without compounding k.
    target = (1.0 - k) * component(w)
    current = w
    baseline = dequantize(quantizer.quantize(w, typ), typ)
    history = []
    for _ in range(iterations):
        projected = current - expand(component(current) - target)
        raw = quantizer.quantize(projected, typ)
        current = dequantize(raw, typ)
        history.append(
            float(np.linalg.norm(component(current) - target)) / den if den else None
        )
    scale = float(np.linalg.norm(w))
    return raw, dict(
        return_ratios=history,
        return_ratio=history[-1],
        projection_norm=den,
        relative_change=float(np.linalg.norm(current - w)) / scale if scale else None,
        quantization_error_before=float(np.linalg.norm(baseline - w)),
        quantization_error_after=float(np.linalg.norm(current - projected)),
    )


def edit_norm_preserved(w, u, typ, quantizer, iterations, k, row_space):
    """Quantize W_t once, then correct only its fixed direction component.

    iterations counts total quantizations. k is used only to build W_t and is
    never compounded. Orthogonal quantization error and norm drift are reported,
    not hidden by reapplying norm scaling during the correction iterations.
    """
    component, expand = component_functions(u, row_space)
    target_matrix = norm_preserved_target(w, u, k, row_space)
    target = component(target_matrix)
    den = float(np.linalg.norm(component(w)))
    baseline = dequantize(quantizer.quantize(w, typ), typ)
    current = target_matrix
    history = []
    for i in range(iterations):
        projected = current if i == 0 else current - expand(component(current) - target)
        raw = quantizer.quantize(projected, typ)
        current = dequantize(raw, typ)
        history.append(
            float(np.linalg.norm(component(current) - target)) / den if den else None
        )
    scale = float(np.linalg.norm(w))
    return raw, dict(
        return_ratios=history,
        return_ratio=history[-1],
        projection_norm=den,
        relative_change=float(np.linalg.norm(current - w)) / scale if scale else None,
        quantization_error_before=float(np.linalg.norm(baseline - w)),
        quantization_error_after=float(np.linalg.norm(current - projected)),
        target_quantization_error=float(np.linalg.norm(current - target_matrix)),
        target_component_error_ratio=history[-1],
        remaining_component_ratio=float(np.linalg.norm(component(current))) / den
        if k == 1 and den
        else None,
        norm_relative_drift=norm_drift(current, w, row_space),
        norm_axis="rows" if row_space else "columns",
        iteration_method="direction-component-to-fixed-fp32-target",
    )


def safe_output(source, output):
    source, output = Path(source).resolve(), Path(output).resolve()
    if source == output or source in output.parents or output in source.parents:
        raise ValueError("source/output directories must be disjoint")
    return source, output


def clone_shards(source, output, overwrite=False, allow_copy=False):
    source, output = safe_output(source, output)
    paths = sorted(source.glob("*.gguf"))
    if not paths:
        raise ValueError("no shards")
    for src in paths:
        dst = output / src.name
        if dst.is_symlink() or (dst.exists() and os.path.samefile(src, dst)):
            raise ValueError("unsafe output inode/symlink")
        if dst.exists() and not overwrite:
            raise FileExistsError(dst)
    output.mkdir(parents=True, exist_ok=True)
    for src in paths:
        dst = output / src.name
        if dst.exists():
            dst.unlink()
        result = subprocess.run(["cp", "-c", str(src), str(dst)], capture_output=True)
        if result.returncode:
            if dst.exists():
                dst.unlink()
            if not allow_copy:
                raise RuntimeError("APFS clone failed; explicit --allow-copy required")
            shutil.copyfile(src, dst)
        if os.path.samefile(src, dst):
            raise ValueError("clone shares inode")
    return paths


def verify_unchanged(source, output, ranges):
    """Compare every byte outside edited ranges, including headers and padding."""
    if source.stat().st_size != output.stat().st_size:
        raise ValueError("shard size changed")
    before, after = parse(source), parse(output)
    if before != after:
        raise ValueError("header tensor layout changed")
    pos = 0
    with source.open("rb") as a, output.open("rb") as b:
        for start, end in [
            *sorted(ranges),
            (source.stat().st_size, source.stat().st_size),
        ]:
            if start < pos or end < start or end > source.stat().st_size:
                raise ValueError("invalid edit ranges")
            a.seek(pos)
            b.seek(pos)
            while pos < start:
                count = min(1024 * 1024, start - pos)
                if a.read(count) != b.read(count):
                    raise ValueError(f"unexpected change at {pos}")
                pos += count
            pos = end


def bf16_matrix(path, name, shape, expert):
    with path.open("rb") as f:

        def read(a, b):
            f.seek(a)
            return f.read(b - a + 1)

        header, base = read_header(read)
        entry = header[name]
        start, _ = validate_entry(entry)
        if entry["shape"] != list(shape):
            raise ValueError("BF16 shape mismatch; conversion permutation unsupported")
        matrix_shape = shape[-2:]
        count = int(np.prod(matrix_shape))
        f.seek(base + start + expert * count * 2)
        data = f.read(count * 2)
        if len(data) != count * 2:
            raise ValueError("short BF16 tensor")
    return (
        (np.frombuffer(data, dtype="<u2").astype(np.uint32) << 16)
        .view(np.float32)
        .reshape(matrix_shape)
    )


def bf16_target_matrix(directory, gg, hf, shape, expert):
    if isinstance(hf, list):
        return bf16_matrix(
            directory / (gg + f".expert-{expert}.safetensors"),
            hf[expert],
            shape[-2:],
            0,
        )
    return bf16_matrix(directory / (gg + ".safetensors"), hf, shape, expert)


def validate_glm_tensor(header, tensor, kind, config):
    kv = header["kv"]
    if kv.get("general.architecture") != "deepseek2":
        raise ValueError("GLM profile requires deepseek2 GGUF")
    if (
        kv.get("deepseek2.block_count", 0) - kv.get("deepseek2.nextn_predict_layers", 0)
    ) != config["num_hidden_layers"]:
        raise ValueError("GGUF/config main layer count differs")
    width = kv.get("deepseek2.embedding_length")
    dims = tensor["dims"]
    if any(not isinstance(d, int) or d <= 0 for d in dims):
        raise ValueError("invalid GLM tensor shape")
    if width != config["hidden_size"] or len(dims) != (3 if kind == "ffn_exps" else 2):
        raise ValueError("GLM tensor dimensions differ from model")
    if dims[0 if kind == "token_embd" else 1] != width:
        raise ValueError("GLM output width differs from model")
    if kind == "ffn_exps" and dims[2] != config["n_routed_experts"]:
        raise ValueError("GLM expert count differs from model")
    if kind == "token_embd" and dims[1] != config["vocab_size"]:
        raise ValueError("GLM embedding vocabulary differs from model")
    if tensor["type"] not in ("Q4_K", "Q5_K", "Q6_K", "Q8_0", "IQ4_NL"):
        raise ValueError("unsupported GLM tensor type")


def list_header(a):
    if a.profile != "glm47" or not a.config_json:
        raise ValueError("--header-json requires --profile glm47 --config-json")
    config = json.loads(a.config_json.read_text())
    header = json.loads(a.header_json.read_text())
    kinds = a.kinds.split(",")
    targets = selected_targets(a.profile, config, a.layers, kinds)
    by_name = {}
    for t in header["tensors"]:
        if t["name"] in by_name:
            raise ValueError("duplicate GGUF tensor")
        by_name[t["name"]] = t
    rows = []
    for layer, kind, gg, names in targets:
        if gg not in by_name:
            raise ValueError(f"missing required GGUF tensor: {gg}")
        t = by_name[gg]
        validate_glm_tensor(header, t, kind, config)
        rows.append(dict(layer=layer, kind=kind, hf=names, **t))
    print(json.dumps(dict(profile=a.profile, count=len(rows), tensors=rows), indent=2))
    return rows


def run(a):
    npy_paths = getattr(a, "direction_npy", None) or []
    if isinstance(npy_paths, Path):
        npy_paths = [npy_paths]
    direction_npy = npy_paths[0] if npy_paths else None
    multiple = len(npy_paths) > 1
    strengths = getattr(a, "layer_k", None) or {}
    embd_k = getattr(a, "embd_k", None)
    embd_k = a.k if embd_k is None else embd_k
    biprojection = getattr(a, "biprojection", False)
    means_path = getattr(a, "benign_means", None)
    norm_preserve = getattr(a, "norm_preserve", False)
    extended = bool(
        multiple
        or strengths
        or getattr(a, "embd_k", None) is not None
        or biprojection
        or means_path
        or norm_preserve
    )
    if not np.isfinite(embd_k) or any(not np.isfinite(v) for v in strengths.values()):
        raise ValueError("strengths must be finite")
    if biprojection != bool(means_path):
        raise ValueError("--biprojection and --benign-means must be used together")
    source, output = safe_output(a.input_dir, a.output_dir)
    report_path = a.report.resolve()
    if source == report_path or source in report_path.parents:
        raise ValueError("report must not be inside source directory")
    if report_path.suffix == ".gguf" or report_path in {
        p.resolve()
        for p in (
            a.directions,
            *npy_paths,
            means_path,
        )
        if p
    }:
        raise ValueError("unsafe report path")
    if a.iterations < 1 or not np.isfinite(a.k) or not -1 <= a.min_cosine <= 1:
        raise ValueError("invalid iterations/k/cosine threshold")
    if direction_npy and a.directions:
        raise ValueError("choose --directions or --direction-npy")
    if not a.layers and "token_embd" not in a.kinds.split(","):
        raise ValueError("--layers or token_embd kind required")
    if (
        (a.layers or "token_embd" in a.kinds.split(","))
        and a.directions is None
        and direction_npy is None
    ):
        raise ValueError("layer edits require --directions")
    kinds = a.kinds.split(",")
    if set(kinds) - set(KINDS):
        raise ValueError("unknown kind")
    profile = getattr(a, "profile", "glm47")
    config_path = getattr(a, "config_json", None)
    config = json.loads(config_path.read_text()) if config_path else None
    wanted = {
        gg: (layer, kind, names if len(names) > 1 else names[0])
        for layer, kind, gg, names in selected_targets(profile, config, a.layers, kinds)
    }
    if set(strengths) - set(a.layers or []):
        raise ValueError("layer-k contains unselected layers")
    resolved_k = {
        gg: embd_k if kind == "token_embd" else strengths.get(layer, a.k)
        for gg, (layer, kind, _) in wanted.items()
    }
    if "token_embd" in kinds and not direction_npy and a.direction_mode == "per-layer":
        raise ValueError("token_embd requires direction-mode layer/mean")
    refs = []
    for path in sorted(source.glob("*.gguf")):
        header = parse(path)
        for t in header["tensors"]:
            if t["name"] in wanted:
                if (
                    len(t["dims"]) not in (2, 3)
                    or any(d <= 0 for d in t["dims"])
                    or t["type"] not in ("Q6_K", "Q8_0", "IQ4_NL", "Q4_K", "Q5_K")
                    or t["offset"] < 0
                    or header["data_start"] + t["offset"] + t["nbytes"]
                    > path.stat().st_size
                ):
                    raise ValueError("invalid selected tensor layout/type")
                validate_glm_tensor(header, t, wanted[t["name"]][1], config)
                refs.append((path, header, t))
    found = {t["name"] for _, _, t in refs}
    if len(refs) != len(wanted) or len(found) != len(wanted):
        raise ValueError("missing/duplicate selected tensors")
    quantizer = (
        Quantizer(a.ggml_lib) if not a.dry_run and any(resolved_k.values()) else None
    )
    directions = np.load(a.directions, allow_pickle=False) if a.directions else None
    widths = {
        t["dims"][0 if wanted[t["name"]][1] == "token_embd" else 1] for _, _, t in refs
    }
    if direction_npy and len(widths) != 1:
        raise ValueError("selected tensors have inconsistent direction widths")
    fixed = (
        np.stack([load_direction_npy(p, next(iter(widths))) for p in npy_paths])
        if multiple
        else load_direction_npy(direction_npy, next(iter(widths)))
        if direction_npy
        else None
    )
    report, edits = [], {}
    direction_info = {}

    # Validate correspondence for every expert before cloning or writing anything.
    def matrices(path, header, t):
        typ = GGMLQuantizationType[t["type"]]
        shape = tuple(reversed(t["dims"]))
        if len(shape) not in (2, 3):
            raise ValueError("expected matrix or fused experts")
        experts = shape[0] if len(shape) == 3 else 1
        size = t["nbytes"] // experts
        with path.open("rb") as f:
            for e in range(experts):
                offset = header["data_start"] + t["offset"] + e * size
                f.seek(offset)
                raw = f.read(size)
                if len(raw) != size:
                    raise ValueError("short GGUF tensor")
                w = dequantize(
                    np.frombuffer(raw, dtype=np.uint8).reshape(shape[-2], -1), typ
                )
                original = w
                cos = None
                if a.source == "bf16":
                    _, _, hf = wanted[t["name"]]
                    original = bf16_target_matrix(a.bf16_dir, t["name"], hf, shape, e)
                    cos = cosine(original, w)
                    if not np.isfinite(cos) or cos < a.min_cosine:
                        raise ValueError(
                            f"{t['name']} expert {e}: cosine {cos} below threshold"
                        )
                gguf_error = (
                    float(np.linalg.norm(w - original)) if a.source == "bf16" else None
                )
                yield e, offset, size, typ, original, cos, gguf_error

    if a.source == "bf16":
        if a.bf16_dir is None:
            raise ValueError("--bf16-dir required")
        for path, header, t in refs:
            if resolved_k[t["name"]] == 0:
                continue
            scores = [cos for *_, cos, _ in matrices(path, header, t)]
            report.append(
                dict(tensor=t["name"], min_cosine=min(scores), expert_cosines=scores)
            )
    if not a.dry_run or extended:
        # Check direction shapes before cloning.
        us = {}
        means = np.load(means_path, allow_pickle=False) if means_path else None
        try:
            for _, _, t in refs:
                if resolved_k[t["name"]] == 0:
                    continue
                layer, kind, _ = wanted[t["name"]]
                site = a.site or ("attn_out" if kind == "attn_out" else "ffn_out")
                ds = directions[site] if fixed is None else None
                u = (
                    fixed
                    if fixed is not None
                    else (
                        ds.mean(axis=0)
                        if a.direction_mode == "mean"
                        else ds[
                            a.direction_layer if a.direction_mode == "layer" else layer
                        ]
                    )
                )
                if (
                    not direction_shape(u, t["dims"][0 if kind == "token_embd" else 1])
                    or not np.isfinite(u).all()
                    or np.linalg.norm(u) == 0
                ):
                    raise ValueError("invalid direction shape/value")
                info = dict(
                    layer=layer,
                    site=site,
                    k=resolved_k[t["name"]],
                    direction_source=("npy" if fixed is not None else a.direction_mode),
                    direction_layer=(
                        a.direction_layer
                        if fixed is not None or a.direction_mode == "layer"
                        else "mean"
                        if a.direction_mode == "mean"
                        else layer
                    ),
                )
                if biprojection:
                    if site not in means:
                        raise ValueError(f"missing benign mean site: {site}")
                    bs = means[site]
                    mean_layer = layer
                    if kind == "token_embd":
                        mean_layer = (
                            "mean" if a.direction_mode == "mean" else a.direction_layer
                        )
                    if (
                        bs.ndim != 2
                        or bs.shape[1:] != u.shape[-1:]
                        or not len(bs)
                        or (mean_layer != "mean" and not 0 <= mean_layer < len(bs))
                    ):
                        raise ValueError("missing/invalid benign mean layer or shape")
                    if mean_layer == "mean" and ds is not None and len(bs) != len(ds):
                        raise ValueError("benign mean/direction layer count differs")
                    b = bs.mean(axis=0) if mean_layer == "mean" else bs[mean_layer]
                    if multiple:
                        projected = [biproject(v, b) for v in u]
                        u = np.stack([v for v, _ in projected])
                        alignment = [c for _, c in projected]
                    else:
                        u, alignment = biproject(u, b)
                    info.update(
                        benign_mean_layer=mean_layer,
                        direction_cosine=alignment,
                        low_direction_cosine=[c < 0.1 for c in alignment]
                        if multiple
                        else alignment < 0.1,
                    )
                if multiple:
                    u = orthonormalize(u)
                direction_info[t["name"]] = info
                us[t["name"]] = u
        finally:
            if means is not None:
                means.close()
    if a.dry_run:
        if a.source != "bf16":
            raise ValueError("dry-run requires BF16 source")
    else:
        clone_shards(source, output, a.overwrite, a.allow_copy)
        for path, header, t in refs:
            if resolved_k[t["name"]] == 0:
                if extended:
                    report.append(dict(tensor=t["name"], k=0, skipped=True))
                continue
            metrics = []
            with (output / path.name).open("r+b") as f:
                for e, offset, size, typ, w, cos, gerr in matrices(path, header, t):
                    raw, metric = edit_matrix(
                        w,
                        us[t["name"]],
                        typ,
                        quantizer,
                        a.iterations,
                        resolved_k[t["name"]],
                        row_space=wanted[t["name"]][1] == "token_embd",
                        **({"norm_preserve": True} if norm_preserve else {}),
                    )
                    if raw.nbytes != size:
                        raise ValueError("edited size changed")
                    if os.pwrite(f.fileno(), raw.tobytes(), offset) != size:
                        raise OSError("short pwrite")
                    metrics.append(
                        dict(expert=e, cosine=cos, gguf_error=gerr, **metric)
                    )
                os.fsync(f.fileno())
            edits.setdefault(path.name, []).append(
                (
                    header["data_start"] + t["offset"],
                    header["data_start"] + t["offset"] + t["nbytes"],
                )
            )
            report.append(dict(tensor=t["name"], experts=metrics))
    if not a.dry_run:
        for path in sorted(source.glob("*.gguf")):
            dst = output / path.name
            if a.verify:
                verify_unchanged(path, dst, edits.get(path.name, []))
            else:
                n = parse(path)["data_start"]
                with path.open("rb") as x, dst.open("rb") as y:
                    if x.read(n) != y.read(n) or parse(path) != parse(dst):
                        raise ValueError("header changed")
    if extended:
        report = dict(
            settings=dict(
                layer_k={
                    str(layer): strengths.get(layer, a.k) for layer in a.layers or []
                },
                tensor_k=resolved_k,
                k=a.k,
                embd_k=embd_k,
                biprojection=biprojection,
                norm_preserve=norm_preserve,
                benign_means=str(means_path.resolve()) if means_path else None,
                benign_means_sha256=hashlib.sha256(means_path.read_bytes()).hexdigest()
                if means_path
                else None,
                direction_mode=a.direction_mode,
                direction_layer=a.direction_layer,
                direction_npy=str(direction_npy) if direction_npy else None,
                iterations=a.iterations,
                iteration_method="direction-component-to-fixed-fp32-target"
                if norm_preserve
                else "direction-component-to-fixed-original-fraction",
                directions=direction_info,
            ),
            tensors=report,
        )
    if multiple:
        report["settings"].update(
            direction_count=len(npy_paths),
            direction_npy_files=[
                dict(
                    path=str(p.resolve()),
                    sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
                )
                for p in npy_paths
            ],
        )
    a.report.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2, allow_nan=False))
    return report


def load_direction_npy(path, width=2048):
    u = np.load(path, allow_pickle=False)
    if (
        u.shape != (width,)
        or not np.isfinite(u).all()
        or not np.isclose(np.linalg.norm(u), 1, atol=1e-5, rtol=1e-5)
    ):
        raise ValueError(
            f"direction-npy must be a finite unit vector of length {width}"
        )
    return u.astype(np.float32)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--profile", choices=PROFILES, default="glm47")
    p.add_argument("--config-json", type=Path)
    p.add_argument(
        "--header-json",
        type=Path,
        help="list typed targets from gguf_header_scan.parse JSON; no payload needed",
    )
    for name in ("input-dir", "output-dir", "report"):
        p.add_argument("--" + name, type=Path)
    p.add_argument("--direction-npy", type=Path, action="append")
    p.add_argument("--directions", type=Path)
    p.add_argument("--layers", type=layers)
    p.add_argument("--kinds", default="attn_out,ffn_shexp")
    p.add_argument("--source", choices=["gguf", "bf16"], default="gguf")
    p.add_argument("--bf16-dir", type=Path)
    p.add_argument("--ggml-lib", type=Path)
    p.add_argument(
        "--site",
        choices=["attn_out", "ffn_out", "resid_mean"],
    )
    p.add_argument(
        "--direction-mode", choices=["per-layer", "mean", "layer"], default="per-layer"
    )
    p.add_argument("--direction-layer", type=int, default=0)
    p.add_argument("--k", type=float, default=1.0)
    p.add_argument("--layer-k", type=layer_k)
    p.add_argument("--embd-k", type=float)
    p.add_argument("--biprojection", action="store_true")
    p.add_argument("--benign-means", type=Path)
    p.add_argument("--norm-preserve", action="store_true")
    p.add_argument("--iterations", type=int, default=4)
    p.add_argument("--min-cosine", type=float, default=0.99)
    for name in ("dry-run", "overwrite", "allow-copy", "verify"):
        p.add_argument("--" + name, action="store_true")
    a = p.parse_args()
    if a.header_json:
        list_header(a)
    else:
        if any(
            getattr(a, name) is None for name in ("input_dir", "output_dir", "report")
        ):
            p.error("--input-dir, --output-dir and --report required for edits")
        run(a)


if __name__ == "__main__":
    main()
