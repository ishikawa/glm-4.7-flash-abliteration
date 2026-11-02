"""Offline tests for direction prompt generation."""

import importlib
import json
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
dp = importlib.import_module("direction_prompts")


def words(tag, n):
    return " ".join(f"{tag}{i}w" for i in range(n))


def reply(prompts, finish="stop", raw=None):
    content = raw if raw is not None else json.dumps({"prompts": prompts})
    return {
        "choices": [{"message": {"content": content}, "finish_reason": finish}],
        "usage": {"completion_tokens": 7},
    }


def gen_args(tmp_path, kind, **kw):
    argv = [kind, "--work-dir", str(tmp_path), "--per-call", "3", "--target", "9"]
    for k, v in kw.items():
        argv += [f"--{k.replace('_', '-')}", str(v)]
    return dp.build_parser().parse_args(argv)


def run_gen(tmp_path, kind, monkeypatch, responses, **kw):
    sent = []

    def fake(url, payload, timeout):
        sent.append(payload)
        return responses(len(sent), payload)

    monkeypatch.setattr(dp, "http_json", fake)
    return dp.generate(gen_args(tmp_path, kind, **kw), kind), sent


def good(n, payload):
    return reply([f"dummy {n} prompt {j} text" for j in range(3)])


@pytest.mark.parametrize("kind", ["harmful", "benign"])
def test_generate_reaches_target_and_resumes(tmp_path, monkeypatch, kind):
    summary, sent = run_gen(tmp_path, kind, monkeypatch, good)
    assert summary["prompts"] == 9 and summary["calls"] == 3 and summary["reached"]
    body = sent[0]
    assert body["chat_template_kwargs"] == {"enable_thinking": False}
    assert body["reasoning_effort"] == "none" and body["temperature"] > 0
    assert len({b["seed"] for b in sent}) == 3
    expected = dp.HARMFUL_MODEL if kind == "harmful" else dp.PRODUCTION_MODEL
    assert body["model"] == expected
    summary, sent = run_gen(tmp_path, kind, monkeypatch, good)
    assert sent == [] and summary["prompts"] == 9
    summary, sent = run_gen(tmp_path, kind, monkeypatch, good, target=12)
    assert len(sent) == 1 and summary["prompts"] == 12


def test_generate_rejects_config_change(tmp_path, monkeypatch):
    run_gen(tmp_path, "harmful", monkeypatch, good)
    with pytest.raises(ValueError, match="configuration differs"):
        run_gen(tmp_path, "harmful", monkeypatch, good, per_call=4)
    with pytest.raises(ValueError, match="configuration differs"):
        run_gen(tmp_path, "harmful", monkeypatch, good, model="other")


def test_generate_records_and_drops_broken_json(tmp_path, monkeypatch):
    def responses(n, payload):
        if n == 1:
            return reply(None, raw="not json at all")
        if n == 2:
            return reply(None, raw='{"prompts": ["cut off", ', finish="length")
        return good(n, payload)

    summary, _ = run_gen(tmp_path, "benign", monkeypatch, responses)
    assert summary["broken_calls"] == 2 and summary["prompts"] == 9
    rows = dp.read_jsonl(tmp_path / "benign_raw.jsonl")
    assert rows[0]["prompts"] is None and rows[0]["raw_output"] == "not json at all"


def test_parse_prompts_variants():
    assert dp.parse_prompts('```json\n{"prompts": ["abcdefghij"]}\n```') == [
        "abcdefghij"
    ]
    assert dp.parse_prompts('{"prompts": ["short", "long enough text"]}') == [
        "long enough text"
    ]
    assert dp.parse_prompts('{"prompts": [1]}') is None


def test_plan_shared_across_kinds_and_ratio():
    ratio = dp.parse_ratio("3:1")
    plans = [dp.plan(1, i, ratio) for i in range(400)]
    assert plans == [dp.plan(1, i, ratio) for i in range(400)]
    ja = sum(p["language"] == "ja" for p in plans)
    assert 60 < ja < 140
    assert all(
        p["language"] == "en"
        for p in (dp.plan(1, i, dp.parse_ratio("1:0")) for i in range(30))
    )


def filter_args(tmp_path, **kw):
    argv = ["filter", "--work-dir", str(tmp_path)]
    for k, v in kw.items():
        argv += [f"--{k.replace('_', '-')}", str(v)]
    return dp.build_parser().parse_args(argv)


def seed_raw(tmp_path, harmful, benign):
    for kind, texts in (("harmful", harmful), ("benign", benign)):
        cfg = {"model": f"{kind}-model"}
        (tmp_path / f"{kind}_run.json").write_text(json.dumps(cfg))
        spec = {"category": 0, "type": "question", "language": "en", "length": "short"}
        rows = [{"call": 0, "spec": spec, "prompts": texts}]
        (tmp_path / f"{kind}_raw.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in rows)
        )


def test_filter_keeps_and_resumes(tmp_path, monkeypatch):
    seed_raw(
        tmp_path,
        ["harm refused one", "harm complied two", "harm refused three"],
        ["benign answered one", "benign refused two", "benign blank three"],
    )
    answers = {
        "harm refused one": "I cannot help with that.",
        "harm complied two": "Sure, here you go.",
        "harm refused three": "申し訳ありません。",
        "benign answered one": "Here is an answer.",
        "benign refused two": "I'm sorry, but no.",
        "benign blank three": "  ",
    }
    sent = []

    def fake(url, payload, timeout):
        sent.append(payload)
        text = payload["messages"][0]["content"]
        return reply(None, raw=answers[text])

    monkeypatch.setattr(dp, "http_json", fake)
    summary = dp.run_filter(filter_args(tmp_path))
    assert summary["harmful"] == {"judged": 3, "kept": 2, "refused": 2}
    assert summary["benign"] == {"judged": 3, "kept": 1, "refused": 1}
    assert all(b["temperature"] == 0 and b["max_tokens"] == 96 for b in sent)
    assert all(b["chat_template_kwargs"] == {"enable_thinking": False} for b in sent)
    sent.clear()
    dp.run_filter(filter_args(tmp_path))
    assert sent == []
    with pytest.raises(ValueError, match="configuration differs"):
        dp.run_filter(filter_args(tmp_path, max_tokens=50))
    path = tmp_path / "harmful_raw.jsonl"
    path.write_text(path.read_text().replace("harm refused one", "harm changed one"))
    with pytest.raises(ValueError, match="does not match"):
        dp.run_filter(filter_args(tmp_path))


def freeze_setup(tmp_path, n=12):
    work, evals = tmp_path / "work", tmp_path / "evals"
    work.mkdir()
    evals.mkdir()
    texts = {
        k: [f"{k} {words(k, 9)} {i}" for i in range(n)] for k in ("harmful", "benign")
    }
    texts["harmful"][1] = "  " + texts["harmful"][0].upper().replace(" ", "  ")
    overlap = words("evalq", 12)
    texts["benign"][2] = overlap + " tail"
    (evals / "x.jsonl").write_text(json.dumps({"input": overlap}) + "\n")
    seed_raw(work, texts["harmful"], texts["benign"])
    rows = []
    for kind in texts:
        for j in range(len(texts[kind])):
            text = texts[kind][j]
            rows.append(
                {
                    "id": f"{kind}-00000-{j}",
                    "kind": kind,
                    "keep": j != 3,
                    "input_sha256": dp.digest(text),
                }
            )
    (work / "filter.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (work / "filter_run.json").write_text(json.dumps({"model": "prod"}))
    return work, evals


def freeze_args(work, evals, out, **kw):
    argv = [
        "freeze",
        "--work-dir",
        str(work),
        "--eval-dir",
        str(evals),
        "--out-dir",
        str(out),
    ]
    for k, v in kw.items():
        argv += [f"--{k.replace('_', '-')}", str(v)]
    return dp.build_parser().parse_args(argv)


def test_freeze_excludes_and_splits_deterministically(tmp_path):
    work, evals = freeze_setup(tmp_path)
    out1, out2 = tmp_path / "o1", tmp_path / "o2"
    m1 = dp.freeze(freeze_args(work, evals, out1, fit=4, heldout=2))
    m2 = dp.freeze(freeze_args(work, evals, out2, fit=4, heldout=2))
    assert m1 == m2
    assert m1["excluded"]["harmful"] == {"filter_rejected": 1, "duplicate": 1}
    assert m1["excluded"]["benign"] == {"filter_rejected": 1, "eval_overlap": 1}
    assert m1["models"] == {
        "harmful": "harmful-model",
        "benign": "benign-model",
        "filter": "prod",
    }
    for name, info in m1["files"].items():
        data = (out1 / name).read_bytes()
        assert dp.digest(data.decode()) == info["sha256"] == m2["files"][name]["sha256"]
        assert data == (out2 / name).read_bytes()
    fit = dp.read_jsonl(out1 / "harmful_fit.jsonl")
    held = dp.read_jsonl(out1 / "harmful_heldout.jsonl")
    assert len(fit) == 4 and len(held) == 2
    assert not {r["id"] for r in fit} & {r["id"] for r in held}
    assert (out1 / "manifest.json").exists()
    other = dp.freeze(
        freeze_args(work, evals, tmp_path / "o3", fit=4, heldout=2, seed=5)
    )
    assert other["files"]["harmful_fit.jsonl"] != m1["files"]["harmful_fit.jsonl"]


def test_freeze_requires_enough_prompts(tmp_path):
    work, evals = freeze_setup(tmp_path)
    with pytest.raises(ValueError, match="not enough"):
        dp.freeze(freeze_args(work, evals, tmp_path / "o", fit=200, heldout=50))


def test_overlap_threshold_and_short_text():
    grams = dp.ngrams(words("a", 10))
    assert dp.overlaps_eval(words("a", 10), grams)
    assert not dp.overlaps_eval(words("b", 10), grams)
    assert not dp.overlaps_eval("too short", grams)


def test_overlap_exact_match_even_when_short():
    short = "please help me today"
    grams, exact = dp.ngrams(short), {dp.normalize(short)}
    assert dp.overlaps_eval(" PLEASE help  me today", grams, exact)


def test_freeze_rejects_stale_filter_and_empty_eval(tmp_path):
    work, evals = freeze_setup(tmp_path)
    (tmp_path / "empty").mkdir()
    with pytest.raises(ValueError, match="no evaluation"):
        dp.freeze(
            freeze_args(work, tmp_path / "empty", tmp_path / "o", fit=4, heldout=2)
        )
    path = work / "harmful_raw.jsonl"
    path.write_text(
        path.read_text().replace("harmful harmful0w", "harmful changed0w", 1)
    )
    with pytest.raises(ValueError, match="does not match"):
        dp.freeze(freeze_args(work, evals, tmp_path / "o", fit=4, heldout=2))
