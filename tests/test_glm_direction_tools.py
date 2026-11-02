"""Model-free GLM profile coverage, including header/index-only resolution."""

import argparse
import importlib
import json
import os
import struct
import sys
from collections import Counter
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
b = importlib.import_module("bf16_fetch")
e = importlib.import_module("direction_edit")
f = importlib.import_module("direction_fit")
p = importlib.import_module("direction_profiles")


@pytest.fixture
def config():
    return dict(
        num_hidden_layers=47,
        hidden_size=2048,
        n_routed_experts=64,
        first_k_dense_replace=1,
        moe_layer_freq=1,
        vocab_size=154880,
    )


def test_glm_targets(config):
    targets = b.selected_targets("glm47", config, list(range(47)), p.GLM_KINDS)
    assert len(targets) == 141
    assert Counter(k for _, k, _, _ in targets) == dict(
        attn_out=47, ffn_dense=1, ffn_shexp=46, ffn_exps=46, token_embd=1
    )
    names = [name for _, _, _, names in targets for name in names]
    assert len(names) == len(set(names)) == 3039
    assert not any("layers.47." in n for n in names)
    assert "model.layers.0.mlp.down_proj.weight" in names
    assert "model.layers.46.mlp.shared_experts.down_proj.weight" in names
    assert "model.layers.0.self_attn.o_proj.weight" in names
    assert "model.embed_tokens.weight" in names
    index = dict.fromkeys(reversed(names), "shard.safetensors")
    rows = p.resolve_index(targets, index)
    exps = next(r for r in rows if r["gguf"] == "blk.1.ffn_down_exps.weight")
    assert exps["hf"] == [
        f"model.layers.1.mlp.experts.{i}.down_proj.weight" for i in range(64)
    ]
    del index[exps["hf"][2]]
    with pytest.raises(ValueError, match="missing 1 required HF"):
        p.resolve_index(targets, index)
    with pytest.raises(ValueError, match="MTP"):
        p.glm_targets(config, [47], p.GLM_KINDS)
    # Config controls dense/MoE selection.
    config["first_k_dense_replace"] = 2
    kinds = [kind for layer, kind, _, _ in p.glm_targets(config, [1], p.GLM_KINDS)]
    assert kinds == ["attn_out", "ffn_dense", "token_embd"]


def test_header_listing(config, tmp_path, monkeypatch, capsys):
    targets = p.glm_targets(config, range(47), p.GLM_KINDS)
    tensors = []
    for _, kind, gg, _ in targets:
        dims = [2048, config["vocab_size"]] if kind == "token_embd" else [256, 2048]
        if kind == "ffn_exps":
            dims.append(64)
        tensors.append(
            dict(name=gg, dims=dims, type="Q5_K" if kind == "token_embd" else "Q4_K")
        )
    # MTP may be present in GGUF but must not be selected.
    tensors.append(dict(name="blk.47.nextn.foo.weight", dims=[1], type="F32"))
    header = dict(
        kv={
            "general.architecture": "deepseek2",
            "deepseek2.block_count": 48,
            "deepseek2.nextn_predict_layers": 1,
            "deepseek2.embedding_length": 2048,
        },
        tensors=tensors,
    )
    cp, hp = tmp_path / "config.json", tmp_path / "header.json"
    cp.write_text(json.dumps(config))
    hp.write_text(json.dumps(header))
    args = argparse.Namespace(
        profile="glm47",
        config_json=cp,
        header_json=hp,
        layers=list(range(47)),
        kinds=",".join(p.GLM_KINDS),
    )
    rows = e.list_header(args)
    assert len(rows) == 141
    assert rows[-1]["type"] == "Q5_K"
    capsys.readouterr()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "direction_edit.py",
            "--profile",
            "glm47",
            "--config-json",
            str(cp),
            "--header-json",
            str(hp),
            "--layers",
            "0-46",
            "--kinds",
            ",".join(p.GLM_KINDS),
        ],
    )
    e.main()
    assert json.loads(capsys.readouterr().out)["count"] == 141
    ip = tmp_path / "index.json"
    index = {name: "shard.safetensors" for _, _, _, names in targets for name in names}
    ip.write_text(json.dumps(dict(weight_map=index)))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "bf16_fetch.py",
            "--profile",
            "glm47",
            "--config-json",
            str(cp),
            "--index-json",
            str(ip),
            "--layers",
            "0-46",
            "--list-only",
        ],
    )
    b.main()
    listing = json.loads(capsys.readouterr().out)
    assert listing["hf_count"] == 3039 and listing["gguf_count"] == 141
    header["tensors"].pop(0)
    hp.write_text(json.dumps(header))
    with pytest.raises(ValueError, match="missing required GGUF"):
        e.list_header(args)


def test_expert_stack_order_and_no_permutation(tmp_path):
    names = [f"model.layers.1.mlp.experts.{i}.down_proj.weight" for i in range(64)]
    gg = "blk.1.ffn_down_exps.weight"
    for expert in reversed(range(64)):
        raw = (
            (np.full((2, 256), expert, dtype=np.float32).view(np.uint32) >> 16)
            .astype("<u2")
            .tobytes()
        )
        header = json.dumps(
            {
                names[expert]: dict(
                    dtype="BF16", shape=[2, 256], data_offsets=[0, len(raw)]
                )
            }
        ).encode()
        (tmp_path / (gg + f".expert-{expert}.safetensors")).write_bytes(
            struct.pack("<Q", len(header)) + header + raw
        )
    stack = np.stack(
        [e.bf16_target_matrix(tmp_path, gg, names, (64, 2, 256), i) for i in range(64)]
    )
    assert np.array_equal(stack[:, 0, 0], np.arange(64))
    # GLM o_proj stays unchanged even for other attention input widths.
    hf = "model.layers.0.self_attn.o_proj.weight"
    values = np.arange(6144, dtype=np.float32).reshape(1, 6144)
    raw = (values.view(np.uint32) >> 16).astype("<u2").tobytes()
    header = json.dumps(
        {hf: dict(dtype="BF16", shape=[1, 6144], data_offsets=[0, len(raw)])}
    ).encode()
    path = tmp_path / "attn.safetensors"
    path.write_bytes(struct.pack("<Q", len(header)) + header + raw)
    expected = (
        (np.frombuffer(raw, dtype="<u2").astype(np.uint32) << 16)
        .view(np.float32)
        .reshape(1, 6144)
    )
    assert np.array_equal(e.bf16_matrix(path, hf, (1, 6144), 0), expected)


def test_glm_direction_width_and_fit(tmp_path):
    u = np.zeros(2048, dtype=np.float32)
    u[0] = 1
    path = tmp_path / "direction.npy"
    np.save(path, u)
    assert np.array_equal(e.load_direction_npy(path, 2048), u)
    with pytest.raises(ValueError, match="length 2560"):
        e.load_direction_npy(path, 2560)
    for name, shift in (("h", 1), ("b", -1)):
        d = tmp_path / name
        d.mkdir()
        (d / "manifest.json").write_text(json.dumps(dict(records=[0, 1])))
        for site in ("attn_out", "ffn_out", "resid"):
            shape = (2, 47, 1, 2048) if site == "resid" else (2, 47, 2048)
            np.save(d / f"{site}.npy", np.full(shape, shift, dtype=np.float32))
    rows = f.fit(
        tmp_path / "h", tmp_path / "b", tmp_path / "u.npz", tmp_path / "r.json"
    )
    assert len(rows) == 141
    assert np.load(tmp_path / "u.npz")["resid_mean"].shape == (47, 2048)


def test_glm_edit_roundtrip(config, tmp_path):
    """Check source types, embedding dimensions and pre-clone width checks."""
    if not os.environ.get("GGML_BASE_LIB"):
        pytest.skip("GGML_BASE_LIB is not set")
    lib = Path(os.environ["GGML_BASE_LIB"])
    gguf = importlib.import_module("gguf")
    quantizer = e.Quantizer(lib)
    src = tmp_path / "src"
    src.mkdir()
    config["vocab_size"] = 2
    cp = tmp_path / "config.json"
    cp.write_text(json.dumps(config))
    writer = gguf.GGUFWriter(src / "tiny.gguf", "deepseek2")
    writer.add_block_count(48)
    writer.add_nextn_predict_layers(1)
    writer.add_embedding_length(2048)
    rng = np.random.default_rng(413)
    weights = {}
    for name, typ, shape in (
        ("blk.0.ffn_down.weight", "Q4_K", (2048, 256)),
        ("token_embd.weight", "Q5_K", (2, 2048)),
    ):
        w = rng.normal(size=shape).astype(np.float32)
        qt = gguf.GGMLQuantizationType[typ]
        weights[name] = quantizer.quantize(w, qt)
        writer.add_tensor(name, weights[name], raw_dtype=qt)
    writer.add_tensor(
        "blk.47.nextn.untouched.weight", np.ones((2, 2), dtype=np.float32)
    )
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    u = np.zeros(2048, dtype=np.float32)
    u[5] = 1
    up = tmp_path / "u.npy"
    np.save(up, u)
    args = argparse.Namespace(
        profile="glm47",
        config_json=cp,
        input_dir=src,
        output_dir=tmp_path / "out",
        report=tmp_path / "report.json",
        direction_npy=up,
        directions=None,
        layers=[0],
        kinds="ffn_dense,ffn_shexp,ffn_exps,token_embd",
        source="gguf",
        iterations=2,
        k=1.0,
        min_cosine=0.99,
        ggml_lib=lib,
        dry_run=False,
        site=None,
        direction_mode="per-layer",
        direction_layer=0,
        overwrite=False,
        allow_copy=True,
        verify=True,
    )
    original = (src / "tiny.gguf").read_bytes()
    # Rejection precedes cloning and never leaves an output model.
    np.save(up, np.ones(2560, dtype=np.float32) / np.sqrt(2560))
    with pytest.raises(ValueError, match="length 2048"):
        e.run(args)
    assert not args.output_dir.exists()
    np.save(up, u)
    report = e.run(args)
    assert len(report) == 2
    assert all(r["experts"][0]["return_ratio"] < 0.1 for r in report)
    assert (src / "tiny.gguf").read_bytes() == original
    headers = e.parse(args.output_dir / "tiny.gguf")
    assert {t["name"]: t["type"] for t in headers["tensors"]} == {
        "blk.0.ffn_down.weight": "Q4_K",
        "token_embd.weight": "Q5_K",
        "blk.47.nextn.untouched.weight": "F32",
    }


def option_fixture(tmp_path):
    """Synthetic GLM matrices, including routed experts and token rows."""
    gguf = importlib.import_module("gguf")
    src = tmp_path / "src"
    src.mkdir()
    config = dict(
        num_hidden_layers=47,
        hidden_size=2048,
        n_routed_experts=64,
        first_k_dense_replace=1,
        moe_layer_freq=1,
        vocab_size=2,
    )
    cp = tmp_path / "config.json"
    cp.write_text(json.dumps(config))
    writer = gguf.GGUFWriter(src / "tiny.gguf", "deepseek2")
    writer.add_block_count(48)
    writer.add_nextn_predict_layers(1)
    writer.add_embedding_length(2048)
    rng = np.random.default_rng(7)
    for name, shape in (
        ("blk.0.attn_output.weight", (2048, 32)),
        ("blk.1.attn_output.weight", (2048, 32)),
        ("blk.2.attn_output.weight", (2048, 32)),
        ("token_embd.weight", (2, 2048)),
    ):
        w = rng.normal(size=shape).astype(np.float32)
        raw = gguf.quants.quantize(w, gguf.GGMLQuantizationType.Q8_0)
        writer.add_tensor(name, raw, raw_dtype=gguf.GGMLQuantizationType.Q8_0)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    ds = np.zeros((47, 2048), dtype=np.float32)
    ds[:, :2] = [1, 0.001]
    ds /= np.linalg.norm(ds, axis=1, keepdims=True)
    dp, bp = tmp_path / "directions.npz", tmp_path / "means.npz"
    np.savez(dp, attn_out=ds, ffn_out=ds)
    means = np.zeros_like(ds)
    means[:, 0] = 1
    means[2, :2] = [0, 1]
    np.savez(bp, attn_out=means, ffn_out=means)
    args = argparse.Namespace(
        profile="glm47",
        config_json=cp,
        input_dir=src,
        output_dir=tmp_path / "out",
        report=tmp_path / "report.json",
        directions=dp,
        direction_npy=None,
        layers=[0, 1, 2],
        kinds="attn_out,token_embd",
        source="gguf",
        iterations=3,
        k=0.75,
        min_cosine=0.99,
        ggml_lib=None,
        dry_run=False,
        site=None,
        direction_mode="layer",
        direction_layer=7,
        overwrite=False,
        allow_copy=True,
        verify=True,
        bf16_dir=None,
    )
    return args, bp


class CPUQuantizer:
    def __init__(self, path=None):
        self.gguf = importlib.import_module("gguf")

    def quantize(self, w, typ):
        return self.gguf.quants.quantize(w, typ)


def test_glm_option_run(tmp_path, monkeypatch):
    import hashlib

    args, means = option_fixture(tmp_path)
    monkeypatch.setattr(e, "Quantizer", CPUQuantizer)
    args.layer_k = e.layer_k("0:0,1:1")
    args.embd_k = 0
    args.biprojection = True
    args.benign_means = means
    args.norm_preserve = True
    before = (args.input_dir / "tiny.gguf").read_bytes()
    report = e.run(args)
    settings = report["settings"]
    assert settings["layer_k"] == {"0": 0, "1": 1, "2": 0.75}
    assert settings["embd_k"] == 0
    assert (
        settings["benign_means_sha256"]
        == hashlib.sha256(means.read_bytes()).hexdigest()
    )
    assert settings["directions"]["blk.1.attn_output.weight"]["benign_mean_layer"] == 1
    assert settings["directions"]["blk.2.attn_output.weight"]["benign_mean_layer"] == 2
    assert settings["directions"]["blk.1.attn_output.weight"]["low_direction_cosine"]
    assert settings["directions"]["blk.2.attn_output.weight"]["direction_cosine"] > 0.9
    after = (args.output_dir / "tiny.gguf").read_bytes()
    header = e.parse(args.input_dir / "tiny.gguf")
    for tensor in header["tensors"]:
        start = header["data_start"] + tensor["offset"]
        end = start + tensor["nbytes"]
        if tensor["name"] in ("blk.0.attn_output.weight", "token_embd.weight"):
            assert before[start:end] == after[start:end]
        else:
            assert before[start:end] != after[start:end]
    assert (args.input_dir / "tiny.gguf").read_bytes() == before
    assert json.loads(args.report.read_text()) == report


@pytest.mark.parametrize("mode,expected", [("layer", 7), ("mean", "mean")])
def test_embedding_biprojection_mean_source(tmp_path, monkeypatch, mode, expected):
    args, means = option_fixture(tmp_path)
    monkeypatch.setattr(e, "Quantizer", CPUQuantizer)
    args.direction_mode = mode
    args.biprojection, args.benign_means = True, means
    args.layer_k = {0: 0, 1: 0, 2: 0}
    report = e.run(args)
    info = report["settings"]["directions"]["token_embd.weight"]
    assert info["benign_mean_layer"] == expected
    assert report["settings"]["embd_k"] == args.k


def test_biprojection_rejects_missing_before_clone(tmp_path, monkeypatch):
    args, means = option_fixture(tmp_path)
    monkeypatch.setattr(e, "Quantizer", CPUQuantizer)
    args.biprojection, args.benign_means = True, means
    np.savez(means, attn_out=np.ones((1, 2048)))
    with pytest.raises(ValueError, match="benign mean layer"):
        e.run(args)
    assert not args.output_dir.exists()
    np.savez(means, resid_mean=np.ones((47, 2048)))
    with pytest.raises(ValueError, match="benign mean site"):
        e.run(args)
    args.direction_mode = "per-layer"
    with pytest.raises(ValueError, match="token_embd requires"):
        e.run(args)


def test_default_run_compatibility(tmp_path, monkeypatch):
    from test_direction_tools import legacy_edit_matrix

    args, _ = option_fixture(tmp_path)
    monkeypatch.setattr(e, "Quantizer", CPUQuantizer)
    actual = e.run(args)
    actual_report_bytes = args.report.read_bytes()
    actual_bytes = (args.output_dir / "tiny.gguf").read_bytes()
    args.output_dir, args.report = tmp_path / "legacy", tmp_path / "legacy.json"
    monkeypatch.setattr(e, "edit_matrix", legacy_edit_matrix)
    reference = e.run(args)
    assert isinstance(actual, list)
    assert actual == reference
    assert actual_report_bytes == args.report.read_bytes()
    assert actual_bytes == (args.output_dir / "tiny.gguf").read_bytes()


def test_zero_strength_never_quantizes(tmp_path, monkeypatch):
    args, _ = option_fixture(tmp_path)
    args.layer_k = {0: 0, 1: 0, 2: 0}
    args.embd_k = 0

    def forbidden(*args, **kwargs):
        raise AssertionError("zero-strength tensors must not initialize a quantizer")

    monkeypatch.setattr(e, "Quantizer", forbidden)
    report = e.run(args)
    assert all(r["skipped"] for r in report["tensors"])
    assert (args.input_dir / "tiny.gguf").read_bytes() == (
        args.output_dir / "tiny.gguf"
    ).read_bytes()


def test_editor_option_cli(tmp_path, monkeypatch):
    captured = []
    monkeypatch.setattr(e, "run", captured.append)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "direction_edit.py",
            "--input-dir",
            str(tmp_path / "src"),
            "--output-dir",
            str(tmp_path / "dst"),
            "--report",
            str(tmp_path / "report.json"),
            "--layers",
            "0-2",
            "--layer-k",
            "0:0,1-2:0.5",
            "--embd-k",
            "0.25",
            "--biprojection",
            "--benign-means",
            str(tmp_path / "means.npz"),
            "--norm-preserve",
        ],
    )
    e.main()
    assert captured[0].layer_k == {0: 0, 1: 0.5, 2: 0.5}
    assert captured[0].embd_k == 0.25
    assert captured[0].biprojection and captured[0].norm_preserve


def test_extended_dry_run_records_projection(tmp_path, monkeypatch):
    args, means = option_fixture(tmp_path)
    args.dry_run, args.source, args.bf16_dir = True, "bf16", tmp_path
    args.biprojection, args.benign_means = True, means
    gguf = importlib.import_module("gguf")
    path = args.input_dir / "tiny.gguf"
    header = e.parse(path)
    weights = {}
    data = path.read_bytes()
    for tensor in header["tensors"]:
        start = header["data_start"] + tensor["offset"]
        raw = np.frombuffer(data[start : start + tensor["nbytes"]], dtype=np.uint8)
        weights[tensor["name"]] = e.dequantize(
            raw.reshape(tensor["dims"][1], -1), gguf.GGMLQuantizationType.Q8_0
        )
    monkeypatch.setattr(
        e, "bf16_target_matrix", lambda directory, gg, *args: weights[gg]
    )
    report = e.run(args)
    assert "direction_cosine" in report["settings"]["directions"]["token_embd.weight"]
    assert not args.output_dir.exists()
