"""Synthetic-only direction fitting, Range downloads and quantized editing."""

import argparse
import importlib
import json
import os
import struct
import sys
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
f = importlib.import_module("direction_fit")
b = importlib.import_module("bf16_fetch")
e = importlib.import_module("direction_edit")
gguf = pytest.importorskip("gguf")


def test_fit(tmp_path):
    rng = np.random.default_rng(9)
    for name, shift in [("h", 4), ("b", -4)]:
        d = tmp_path / name
        d.mkdir()
        (d / "manifest.json").write_text(json.dumps({"records": [0] * 20}))
        for site in ["attn_out", "ffn_out", "resid"]:
            shape = (21, 2, 1, 8) if site == "resid" else (21, 2, 8)
            x = rng.normal(size=shape).astype(np.float32)
            x[..., 0] += shift
            x[-1] = np.nan  # unused capacity must not influence fitting
            np.save(d / f"{site}.npy", x)
    rows = f.fit(
        tmp_path / "h",
        tmp_path / "b",
        tmp_path / "u.npz",
        tmp_path / "r.json",
        tmp_path / "h",
        tmp_path / "b",
    )
    assert len(rows) == 6
    assert all(r["auc"] == 1 for r in rows)
    assert np.all(np.load(tmp_path / "u.npz")["attn_out"][:, 0] > 0.95)
    assert f.auc(np.array([1, 2]), np.array([1, 2])) == 0.5


def safefile(shape=(4, 256), dtype="BF16"):
    x = np.random.default_rng(3).normal(size=shape).astype(np.float32)
    raw = (x.view(np.uint32) >> 16).astype("<u2").tobytes()
    header = json.dumps(
        {"tensor": dict(dtype=dtype, shape=list(shape), data_offsets=[0, len(raw)])}
    ).encode()
    return struct.pack("<Q", len(header)) + header + raw


def test_download_resume(tmp_path):
    data = safefile()
    output = tmp_path / "tensor.safetensors"
    calls = []

    def http(url, start, end):
        calls.append((start, end))
        if len(calls) == 4:
            raise OSError("interrupted")
        return data[start : end + 1]

    with pytest.raises(OSError):
        b.fetch_tensor("mock", "tensor", output, http, chunk_size=100)
    assert output.with_suffix(".safetensors.part").stat().st_size == 100
    b.fetch_tensor(
        "mock",
        "tensor",
        output,
        lambda url, start, end: data[start : end + 1],
        chunk_size=100,
    )
    actual = e.bf16_matrix(output, "tensor", (4, 256), 0)
    assert actual.shape == (4, 256)
    assert not output.with_suffix(".safetensors.part").exists()
    bad = safefile(dtype="F16")
    with pytest.raises(ValueError, match="BF16"):
        b.fetch_tensor(
            "mock", "tensor", output, lambda url, start, end: bad[start : end + 1]
        )
    with pytest.raises(ValueError, match="size"):
        b.validate_entry(dict(dtype="BF16", shape=[2], data_offsets=[0, 3]))


@pytest.fixture
def quantizer():
    if not os.environ.get("GGML_BASE_LIB"):
        pytest.skip("GGML_BASE_LIB is not set")
    return e.Quantizer()


@pytest.mark.parametrize("typ", ["Q6_K", "IQ4_NL", "Q8_0", "Q4_K", "Q5_K"])
def test_quantization(quantizer, typ):
    qt = gguf.GGMLQuantizationType[typ]
    rng = np.random.default_rng(0)
    w = rng.normal(size=(8, 256)).astype(np.float32)
    u = rng.normal(size=8).astype(np.float32)
    u /= np.linalg.norm(u)
    raw = quantizer.quantize(w, qt)
    if typ == "Q8_0":
        assert np.array_equal(
            raw, importlib.import_module("gguf.quants").quantize(w, qt)
        )
    _, one = e.edit_matrix(w, u, qt, quantizer, 1)
    _, many = e.edit_matrix(w, u, qt, quantizer, 6)
    assert many["return_ratio"] < one["return_ratio"]


def test_edit_scale_does_not_compound(quantizer):
    qt = gguf.GGMLQuantizationType["Q8_0"]
    rng = np.random.default_rng(1)
    w = rng.normal(size=(8, 256)).astype(np.float32)
    u = rng.normal(size=8).astype(np.float32)
    u /= np.linalg.norm(u)
    raw, report = e.edit_matrix(w, u, qt, quantizer, 6, k=2.0)
    edited = e.dequantize(raw, qt)
    assert np.allclose(u @ edited, -(u @ w), atol=0.05 * np.linalg.norm(u @ w))
    assert report["return_ratio"] < 0.05


def test_resume_rejects_changed_header_and_short_range(tmp_path):
    data = safefile()
    out = tmp_path / "download.safetensors"
    calls = 0

    def interrupt(url, start, end):
        nonlocal calls
        calls += 1
        if calls == 4:
            raise OSError("stop")
        return data[start : end + 1]

    with pytest.raises(OSError):
        b.fetch_tensor("mock", "tensor", out, interrupt, chunk_size=100)
    changed = data.replace(b"BF16", b"F16 ", 1)
    with pytest.raises(ValueError):
        b.fetch_tensor("mock", "tensor", out, lambda url, a, z: changed[a : z + 1])
    state = out.with_suffix(".safetensors.part.json")
    record = json.loads(state.read_text())
    record["identity"]["header_sha256"] = "wrong"
    state.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="header/size"):
        b.fetch_tensor("mock", "tensor", out, lambda url, a, z: data[a : z + 1])

    def short(url, start, end):
        if start > 8:
            return b""
        return data[start : end + 1]

    with pytest.raises(ValueError, match="Range size"):
        b.fetch_tensor("mock", "tensor", tmp_path / "short", short)


def test_clone_rejects_same_inode(tmp_path):
    src, dst = tmp_path / "src", tmp_path / "dst"
    src.mkdir()
    dst.mkdir()
    (src / "x.gguf").write_bytes(b"not a model")
    (dst / "x.gguf").hardlink_to(src / "x.gguf")
    with pytest.raises(ValueError, match="inode"):
        e.clone_shards(src, dst, overwrite=True)


def test_direction_npy(tmp_path):
    path = tmp_path / "u.npy"
    u = np.zeros(2048, dtype=np.float32)
    u[7] = 1
    np.save(path, u)
    assert np.array_equal(e.load_direction_npy(path), u)
    for invalid in (u * 2, u[:20], u + np.nan):
        np.save(path, invalid)
        with pytest.raises(ValueError, match="unit vector"):
            e.load_direction_npy(path)


def test_biprojection():
    u = np.array([1, 2, 3], dtype=np.float32)
    b = np.array([3, -1, 2], dtype=np.float32)
    projected, alignment = e.biproject(u, b)
    assert np.linalg.norm(projected) == pytest.approx(1, abs=1e-7)
    assert projected @ b == pytest.approx(0, abs=2e-7)
    assert alignment == pytest.approx(e.cosine(u, projected))
    _, tiny = e.biproject(np.array([1, 0.001, 0]), np.array([1, 0, 0]))
    assert tiny < 0.1
    for invalid in (np.zeros(3), u, np.ones(2), np.full(3, np.nan)):
        with pytest.raises(ValueError):
            e.biproject(u, invalid)


@pytest.mark.parametrize(
    "shape,row_space", [((7, 256), False), ((7, 256), True), ((3, 7, 256), False)]
)
def test_norm_preserving_reference(shape, row_space):
    rng = np.random.default_rng(413)
    w = rng.normal(size=shape).astype(np.float32)
    axis = -1 if row_space else -2
    u = rng.normal(size=w.shape[axis]).astype(np.float32)
    u /= np.linalg.norm(u)
    target = e.norm_preserved_target(w, u, row_space=row_space)
    np.testing.assert_allclose(
        np.linalg.norm(target, axis=axis), np.linalg.norm(w, axis=axis), rtol=3e-7
    )
    vectors = np.moveaxis(target, axis, -1)
    np.testing.assert_allclose(vectors @ u, 0, atol=6e-7)
    # jim-plus ca6e223, sharded_ablate.py L136-179: output space last,
    # normalize, project, normalize, project again, normalize, restore magnitude.
    v = np.moveaxis(w, axis, -1)
    magnitude = np.linalg.norm(v, axis=-1, keepdims=True)
    v = v / magnitude
    v -= (v @ u)[..., None] * u
    v /= np.linalg.norm(v, axis=-1, keepdims=True)
    v -= (v @ u)[..., None] * u
    v /= np.linalg.norm(v, axis=-1, keepdims=True)
    reference = np.moveaxis(v * magnitude, -1, axis)
    np.testing.assert_allclose(target, reference, rtol=1e-6, atol=5e-7)
    partial = e.norm_preserved_target(w, u, 0.5, row_space)
    raw = np.moveaxis(w, axis, -1)
    raw = raw - 0.5 * (raw @ u)[..., None] * u
    expected = raw / np.linalg.norm(raw, axis=-1, keepdims=True) * magnitude
    np.testing.assert_allclose(
        np.moveaxis(partial, axis, -1), expected, rtol=2e-6, atol=5e-7
    )
    assert np.linalg.norm(np.moveaxis(partial, axis, -1) @ u) > 0.01
    np.testing.assert_allclose(
        e.norm_preserved_target(w, u, 0, row_space), w, atol=5e-7, rtol=3e-7
    )


def test_norm_preserving_degenerate():
    with pytest.raises(ValueError, match="collapsed"):
        e.norm_preserved_target(np.ones((2, 3)), np.ones(2))
    assert not e.norm_preserved_target(np.zeros((2, 3)), np.ones(2)).any()


@pytest.mark.parametrize("row_space", [False, True])
@pytest.mark.parametrize("k", [0.5, 1.0])
def test_norm_quantized_fixed_target(quantizer, row_space, k):
    rng = np.random.default_rng(99)
    w = rng.normal(size=(32, 256)).astype(np.float32)
    u = rng.normal(size=w.shape[1 if row_space else 0]).astype(np.float32)
    u /= np.linalg.norm(u)
    typ = gguf.GGMLQuantizationType.Q8_0
    raw, metric = e.edit_matrix(w, u, typ, quantizer, 4, k, row_space, True)
    final = e.dequantize(raw, typ)
    target = e.norm_preserved_target(w, u, k, row_space)
    component = (lambda m: m @ u) if row_space else (lambda m: u @ m)
    den = np.linalg.norm(component(w))
    assert metric["target_component_error_ratio"] == pytest.approx(
        np.linalg.norm(component(final) - component(target)) / den, abs=5e-8
    )
    assert metric["target_component_error_ratio"] < 0.02
    assert metric["target_quantization_error"] == pytest.approx(
        np.linalg.norm(final - target)
    )
    assert metric["norm_relative_drift"]["max"] < 0.02
    assert metric["iteration_method"] == "direction-component-to-fixed-fp32-target"
    if k == 1:
        assert metric["remaining_component_ratio"] < 0.02
    else:
        assert metric["remaining_component_ratio"] is None
        assert np.linalg.norm(component(final)) > 0.4 * den


def test_layer_k_parser():
    assert e.layer_k("0-2:0,3:0.5,4-6:1") == {
        0: 0,
        1: 0,
        2: 0,
        3: 0.5,
        4: 1,
        5: 1,
        6: 1,
    }
    for invalid in ("0-2:1,2:0", "3-1:1", "48:1", "-1:0", "0:nan", "0:inf", ""):
        with pytest.raises(argparse.ArgumentTypeError):
            e.layer_k(invalid)


def test_optional_benign_means(tmp_path):
    for name, shift in (("h", 2), ("b", -2)):
        directory = tmp_path / name
        directory.mkdir()
        (directory / "manifest.json").write_text(json.dumps(dict(records=[0, 1, 2])))
        for site in ("attn_out", "ffn_out", "resid"):
            shape = (4, 2, 1, 8) if site == "resid" else (4, 2, 8)
            x = np.random.default_rng(12).normal(size=shape).astype(np.float32)
            x[:3, ..., 1] += shift
            x[:3, ..., 0] = 10000 + shift
            x[3] = np.nan  # uncommitted capacity must not enter means
            np.save(directory / f"{site}.npy", x)
    opts = dict(harmful=tmp_path / "h", benign=tmp_path / "b", massive_ratio=10)
    old = f.fit(**opts, output=tmp_path / "old.npz", report=tmp_path / "old.json")
    new = f.fit(
        **opts,
        output=tmp_path / "new.npz",
        report=tmp_path / "new.json",
        benign_means_out=tmp_path / "means.npz",
    )
    assert old == new
    assert (tmp_path / "old.json").read_bytes() == (tmp_path / "new.json").read_bytes()
    with (
        np.load(tmp_path / "old.npz") as before,
        np.load(tmp_path / "new.npz") as after,
    ):
        assert before.files == after.files
        for site in before.files:
            assert np.array_equal(before[site], after[site])
    with np.load(tmp_path / "means.npz") as means:
        assert set(means.files) == set(f.SITES)
        for site in means.files:
            expected = np.stack(
                [
                    f.layer_values(f.activations(tmp_path / "b"), site, i).mean(axis=0)
                    for i in range(2)
                ]
            )
            expected[:, 0] = 0
            np.testing.assert_allclose(means[site], expected, rtol=1e-7, atol=1e-7)
            assert means[site].dtype == np.float32


def legacy_edit_matrix(w, u, typ, quantizer, iterations=4, k=1.0, row_space=False):
    """Frozen pre-option calculation for byte/report compatibility checks."""
    w, u = np.asarray(w, dtype=np.float32), np.asarray(u, dtype=np.float32)
    u = u / np.linalg.norm(u)
    component = (lambda m: m @ u) if row_space else (lambda m: u @ m)
    expand = (lambda v: np.outer(v, u)) if row_space else (lambda v: np.outer(u, v))
    den = float(np.linalg.norm(component(w)))
    target = (1.0 - k) * component(w)
    current = w
    baseline = e.dequantize(quantizer.quantize(w, typ), typ)
    history = []
    for _ in range(iterations):
        projected = current - expand(component(current) - target)
        raw = quantizer.quantize(projected, typ)
        current = e.dequantize(raw, typ)
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


@pytest.mark.parametrize("row_space", [False, True])
def test_default_edit_compatibility(quantizer, row_space):
    rng = np.random.default_rng(88)
    w = rng.normal(size=(32, 256)).astype(np.float32)
    u = rng.normal(size=w.shape[1 if row_space else 0]).astype(np.float32)
    typ = gguf.GGMLQuantizationType.Q8_0
    old_raw, old_report = legacy_edit_matrix(w, u, typ, quantizer, row_space=row_space)
    raw, report = e.edit_matrix(w, u, typ, quantizer, row_space=row_space)
    assert np.array_equal(raw, old_raw)
    assert report == old_report


@pytest.mark.parametrize("row_space", [False, True])
@pytest.mark.parametrize("nonorthogonal", [False, True])
@pytest.mark.parametrize("k", [0.5, 1.0])
def test_subspace_target(row_space, nonorthogonal, k):
    w = np.random.default_rng(413).normal(size=(8, 256)).astype(np.float32)
    width = w.shape[1 if row_space else 0]
    directions = np.eye(width, dtype=np.float32)[:2].copy()
    if nonorthogonal:
        directions[1] += directions[0]
        directions[1] /= np.linalg.norm(directions[1])
    u = e.orthonormalize(directions)
    np.testing.assert_allclose(u @ u.T, np.eye(2), atol=2e-7)
    component, expand = e.component_functions(u, row_space)
    projected = w - expand(component(w))
    np.testing.assert_allclose(component(projected), 0, atol=5e-7)
    target = e.norm_preserved_target(w, u, k, row_space)
    axis = -1 if row_space else -2
    np.testing.assert_allclose(
        np.linalg.norm(target, axis=axis), np.linalg.norm(w, axis=axis), rtol=3e-7
    )
    if k == 1:
        np.testing.assert_allclose(component(target), 0, atol=5e-7)
    else:
        expected = w - k * expand(component(w))
        expected *= np.expand_dims(
            np.linalg.norm(w, axis=axis) / np.linalg.norm(expected, axis=axis), axis
        )
        np.testing.assert_allclose(target, expected, rtol=2e-6, atol=5e-7)


def test_subspace_dependent():
    u = np.array([1, 2, 3], dtype=np.float32)
    for second in (u, -u, 2 * u):
        with pytest.raises(ValueError, match="dependent"):
            e.orthonormalize([u, second])
    with pytest.raises(ValueError, match="dependent"):
        e.orthonormalize([e.biproject(u, [0, 1, 0])[0]] * 2)


@pytest.mark.parametrize("row_space", [False, True])
@pytest.mark.parametrize("norm_preserve", [False, True])
@pytest.mark.parametrize("k", [0.5, 1.0])
def test_subspace_quantized(quantizer, row_space, norm_preserve, k):
    w = np.random.default_rng(808).normal(size=(8, 256)).astype(np.float32)
    width = w.shape[1 if row_space else 0]
    u = e.orthonormalize(np.random.default_rng(810).normal(size=(2, width)))
    qt = gguf.GGMLQuantizationType.Q8_0
    raw, report = e.edit_matrix(w, u, qt, quantizer, 4, k, row_space, norm_preserve)
    final = e.dequantize(raw, qt)
    component, _ = e.component_functions(u, row_space)
    target = (
        component(e.norm_preserved_target(w, u, k, row_space))
        if norm_preserve
        else (1 - k) * component(w)
    )
    ratio = np.linalg.norm(component(final) - target) / np.linalg.norm(component(w))
    assert report["return_ratio"] == pytest.approx(ratio, abs=1e-7)
    assert ratio < 0.03
