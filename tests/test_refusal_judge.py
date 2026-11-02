"""Offline tests for the HarmBench classifier."""

import argparse
import hashlib
import importlib
import json
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
judge = importlib.import_module("refusal_judge")


@pytest.mark.parametrize(
    ("raw", "label"),
    [
        (" YES\n", "yes"),
        ("No", "no"),
        ("yes.", "unparsed"),
        ("answer: no", "unparsed"),
        ("", "unparsed"),
    ],
)
def test_cls_label(raw, label):
    assert judge.parse_cls_label(raw) == label


@pytest.mark.parametrize("count", [0, 511, 512, 800])
@pytest.mark.parametrize(
    "base", ["http://offline.invalid/v1/", "http://offline.invalid"]
)
def test_cls_token_limit(count, base, tmp_path, monkeypatch):
    args = setup_run(tmp_path, [])
    args.base_url = base
    calls = []
    original = "original {behavior}"

    def http(url, body, timeout):
        assert timeout == 10
        calls.append(url)
        if url.endswith("/tokenize"):
            assert body == {"content": original, "add_special": False}
            return {"tokens": list(range(count))}
        if url.endswith("/detokenize"):
            assert body == {"tokens": list(range(511))}
            return {"content": "clipped {generation}"}
        assert url == "http://offline.invalid/completion"
        generation = "clipped {generation}" if count > 511 else original
        assert body == {
            "prompt": judge.HARMBENCH_PROMPT.format(
                behavior="question {generation}", generation=generation
            ),
            "n_predict": 1,
            "temperature": 0,
            "cache_prompt": False,
        }
        return {"content": " YES "}

    monkeypatch.setattr(judge, "http_json", http)
    result = judge.classify_harmbench(args, "question {generation}", original)
    assert result == {
        "label": "yes",
        "raw_output": " YES ",
        "generation_tokens": count,
        "generation_truncated": count > 511,
    }
    assert len(calls) == (3 if count > 511 else 2)


def test_cls_resume_summary(tmp_path, monkeypatch, capsys):
    rows = [
        source_row("a"),
        source_row("b", score=1),
        source_row("c"),
        source_row("d", truncated=True),
        source_row("e", empty=True),
        source_row("f", truncated=True, empty=True),
        {"bench": "xstest_safe", "id": "a"},
    ]
    args = setup_run(tmp_path, rows)
    # Existing model-mode artifacts must coexist without being read or changed.
    for name in ("judge_run.json", "judge.jsonl", "judge_summary.json"):
        (tmp_path / name).write_text("untouched")
    outputs = iter(["no", OSError("interrupted"), "yes", "yes."])
    calls = []

    def http(url, body, timeout):
        calls.append(url)
        if url.endswith("/tokenize"):
            return {"tokens": [1]}
        result = next(outputs)
        if isinstance(result, Exception):
            raise result
        return {"content": result}

    monkeypatch.setattr(judge, "http_json", http)
    with pytest.raises(OSError, match="interrupted"):
        judge.run_cls(args)
    assert len(judge.read_jsonl(tmp_path / "judge_cls.jsonl")) == 1
    summary = judge.run_cls(args)
    assert summary == {
        "harmbench": {
            "total": 6,
            "truncated": 2,
            "empty": 2,
            "unparsed": 1,
            "scored": 2,
            "yes": 1,
            "no": 1,
            "disagreements": 2,
        }
    }
    count = len(calls)
    assert judge.run_cls(args) == summary and len(calls) == count
    assert "harmbench\t6\t2\t2\t1\t2\t1\t1\t2" in capsys.readouterr().out
    assert json.loads((tmp_path / "judge_cls_summary.json").read_text()) == summary
    for name in ("judge_run.json", "judge.jsonl", "judge_summary.json"):
        assert (tmp_path / name).read_text() == "untouched"
    config = json.loads((tmp_path / "judge_cls_run.json").read_text())
    assert config["prompt_sha256"] == judge.digest(judge.HARMBENCH_PROMPT)
    assert config["generation_token_limit"] == 511
    args.model = "other"
    with pytest.raises(ValueError, match="configuration differs"):
        judge.run_cls(args)
    args.model = "production"
    rows[0]["message"]["content"] = "modified"
    (tmp_path / "results.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows)
    )
    with pytest.raises(ValueError, match="source result differs"):
        judge.run_cls(args)


def test_classifier_cli(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(judge, "run_cls", lambda args: calls.append(args.classifier))
    argv = ["--run-dir", str(tmp_path), "--model", "cls"]
    assert judge.main(argv) == 0
    assert judge.main([*argv, "--classifier", "harmbench-cls"]) == 0
    assert calls == ["harmbench-cls", "harmbench-cls"]
    with pytest.raises(SystemExit):
        judge.main([*argv, "--classifier", "harmbench-cls", "--caveat"])


def test_heldout_classifier_and_fallback(tmp_path, monkeypatch):
    from test_refusal_eval import synthetic_heldout

    eval_mod = importlib.import_module("refusal_eval")
    path, _ = synthetic_heldout(tmp_path, monkeypatch)
    rows = [
        source_row("synthetic-a", "harmbench", input="Count the blue squares."),
        source_row("synthetic-b", "harmbench", input="Name a primary color."),
    ]
    for row in rows:
        row["dataset"] = eval_mod.HELDOUT_DATASET
    args = setup_run(tmp_path, rows)
    config = dict(
        dataset=eval_mod.HELDOUT_DATASET,
        dataset_path=str(path),
        data_sha256={
            eval_mod.HELDOUT_DATASET: hashlib.sha256(path.read_bytes()).hexdigest()
        },
    )
    (tmp_path / "run.json").write_text(json.dumps(config))
    calls = []

    def http(url, body, timeout):
        assert url.startswith("http://offline.invalid/")
        if url.endswith("/tokenize"):
            return dict(tokens=[1, 2])
        assert url.endswith("/completion")
        calls.append(body)
        question = rows[len(calls) - 1]["input"]
        assert (
            body["prompt"]
            == judge.build_messages(
                "harmbench", question, rows[len(calls) - 1]["message"]["content"]
            )[0]["content"]
        )
        return dict(content="yes")

    monkeypatch.setattr(judge, "http_json", http)
    summary = judge.run_cls(args)
    assert summary["harmbench"]["dataset"] == eval_mod.HELDOUT_DATASET
    assert len(calls) == 2
    assert all(
        r["dataset"] == eval_mod.HELDOUT_DATASET
        for r in judge.read_jsonl(tmp_path / "judge_cls.jsonl")
    )
    assert judge.run_cls(args) == summary and len(calls) == 2
    # Legacy rows without input must resolve this dataset, never harmbench.jsonl.
    for row in rows:
        del row["input"]
    resolved = judge.load_questions(rows, config)
    assert resolved == {
        "synthetic-a": "Count the blue squares.",
        "synthetic-b": "Name a primary color.",
    }
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="held-out data hash"):
        judge.load_questions(rows, config)


def source_row(id_, bench="harmbench", **extra):
    return {
        "id": id_,
        "bench": bench,
        "input": "Original question",
        "message": {"content": "Final answer", "reasoning_content": "Secret"},
        "truncated": False,
        "empty": False,
        "score": 0,
        "refusal_matches": [],
        **extra,
    }


def setup_run(tmp_path, rows):
    (tmp_path / "run.json").write_text(json.dumps({"model": "candidate"}))
    (tmp_path / "results.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows)
    )
    return argparse.Namespace(
        run_dir=tmp_path,
        model="production",
        base_url="http://offline.invalid/v1",
        timeout=10,
        sets=None,
    )
