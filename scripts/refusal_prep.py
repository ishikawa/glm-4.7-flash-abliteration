#!/usr/bin/env python3
"""Select public evaluation samples with a fixed seed and freeze normalized JSONL."""

import argparse
import csv
import hashlib
import io
import json
import random
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parents[1] / "data/refusal"
SOURCES = {
    "harmbench": (
        "https://raw.githubusercontent.com/centerforaisafety/HarmBench/main/"
        "data/behavior_datasets/harmbench_behaviors_text_all.csv",
        "MIT",
    ),
    "xstest": (
        "https://raw.githubusercontent.com/paul-rottger/xstest/main/xstest_prompts.csv",
        "CC-BY-4.0",
    ),
    "gsm8k": (
        "https://raw.githubusercontent.com/openai/grade-school-math/master/"
        "grade_school_math/data/test.jsonl",
        "MIT",
    ),
    "mmlu_pro": (
        "https://datasets-server.huggingface.co/rows?"
        "dataset=TIGER-Lab/MMLU-Pro&config=default&split=test",
        "MIT",
    ),
}


def fetch_text(url, timeout=120, retries=6):
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as response:
                return response.read().decode("utf-8-sig")
        except urllib.error.HTTPError as e:
            if e.code != 429 or attempt == retries:
                raise
            time.sleep(int(e.headers.get("Retry-After") or 0) or 2**attempt * 5)


def stratified_sample(rows, key, count, seed):
    """Allocate by largest remainder with seeded tie breaking."""
    if not 0 <= count <= len(rows):
        raise ValueError(f"sample size {count} exceeds population {len(rows)}")
    groups = defaultdict(list)
    for row in rows:
        groups[row[key]].append(row)
    rng = random.Random(seed)
    keys = sorted(groups)
    rng.shuffle(keys)
    quotas = {k: count * len(groups[k]) // len(rows) for k in keys}
    ranked = sorted(keys, key=lambda k: -(count * len(groups[k]) % len(rows)))
    for k in ranked[: count - sum(quotas.values())]:
        quotas[k] += 1
    selected = []
    for k in keys:
        selected.extend(rng.sample(groups[k], quotas[k]))
    return selected


def record(bench, source_id, text, reference=None, **meta):
    return {
        "id": f"{bench}-{source_id}",
        "bench": bench,
        "input": text,
        "reference": reference,
        "meta": meta,
    }


def prepare(seed=407):
    harm = list(csv.DictReader(io.StringIO(fetch_text(SOURCES["harmbench"][0]))))
    harm = [r for r in harm if r["FunctionalCategory"] == "standard"]
    sets = {
        "harmbench": [
            record(
                "harmbench",
                r["BehaviorID"],
                r["Behavior"],
                category=r["SemanticCategory"],
                tags=r["Tags"],
            )
            for r in stratified_sample(harm, "SemanticCategory", 100, seed)
        ]
    }
    xs = list(csv.DictReader(io.StringIO(fetch_text(SOURCES["xstest"][0]))))
    for label, count in (("safe", 100), ("unsafe", 50)):
        name = f"xstest_{label}"
        sets[name] = [
            record(name, r["id"], r["prompt"], category=r["type"])
            for r in stratified_sample(
                [r for r in xs if r["label"] == label], "type", count, seed
            )
        ]
    gsm = [
        json.loads(line)
        for line in fetch_text(SOURCES["gsm8k"][0]).splitlines()
        if line.strip()
    ]
    sets["gsm8k"] = [
        record(
            "gsm8k",
            i,
            f"{gsm[i]['question']}\n\nGive the final answer as Answer: N.",
            gsm[i]["answer"].rsplit("####", 1)[1].strip(),
        )
        for i in sorted(random.Random(seed).sample(range(len(gsm)), 50))
    ]
    url = SOURCES["mmlu_pro"][0]
    first = json.loads(fetch_text(f"{url}&offset=0&length=1"))
    indices = sorted(random.Random(seed).sample(range(first["num_rows_total"]), 50))
    sets["mmlu_pro"] = []
    for i in indices:
        page = first if i == 0 else json.loads(fetch_text(f"{url}&offset={i}&length=1"))
        r = page["rows"][0]["row"]
        options = "\n".join(f"{chr(65 + j)}. {v}" for j, v in enumerate(r["options"]))
        text = f"{r['question']}\n\n{options}\n\nGive the final answer as Answer: X."
        sets["mmlu_pro"].append(
            record(
                "mmlu_pro",
                r["question_id"],
                text,
                r["answer"],
                category=r["category"],
                upstream_index=i,
            )
        )
    return sets


def freeze(sets, out_dir, seed):
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "select_seed": seed,
        "retrieved_at": datetime.now(UTC).date().isoformat(),
        "usage": "Evaluation only; not for direction measurement or tuning.",
        "sources": {k: {"url": u, "license": lic} for k, (u, lic) in SOURCES.items()},
        "sampling": {
            "stratified": "proportional largest remainder; seeded ties",
            "mmlu_pro": "uniform seeded row indices; not category stratified",
        },
        "files": {},
        "samples": {},
    }
    for name, rows in sets.items():
        rows = sorted(rows, key=lambda r: r["id"])
        if len({r["id"] for r in rows}) != len(rows):
            raise ValueError(f"duplicate IDs: {name}")
        data = "".join(
            json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in rows
        ).encode()
        (out_dir / f"{name}.jsonl").write_bytes(data)
        manifest["files"][f"{name}.jsonl"] = {
            "count": len(rows),
            "sha256": hashlib.sha256(data).hexdigest(),
            "source": "xstest" if name.startswith("xstest_") else name,
        }
        for r in rows:
            manifest["samples"][r["id"]] = r["meta"]
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def freeze_rest(out_dir):
    """Freeze the standard behaviors absent from the original selection."""
    original = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    selected = {i for i in original["samples"] if i.startswith("harmbench-")}
    source = list(csv.DictReader(io.StringIO(fetch_text(SOURCES["harmbench"][0]))))
    standard = [r for r in source if r["FunctionalCategory"] == "standard"]
    ids = [f"harmbench-{r['BehaviorID']}" for r in standard]
    if len(ids) != 200 or len(set(ids)) != 200 or len(selected) != 100:
        raise ValueError("expected 200 unique standard behaviors and 100 selected IDs")
    if not selected <= set(ids):
        raise ValueError("selected IDs absent from standard behaviors")
    rows = [
        record(
            "harmbench_rest",
            r["BehaviorID"],
            r["Behavior"],
            category=r["SemanticCategory"],
            tags=r["Tags"],
        )
        for r in standard
        if f"harmbench-{r['BehaviorID']}" not in selected
    ]
    remaining = {r["id"].replace("harmbench_rest-", "harmbench-", 1) for r in rows}
    if len(rows) != 100 or len(remaining) != 100 or remaining & selected:
        raise ValueError("invalid standard behavior complement")
    data = "".join(
        json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n"
        for r in sorted(rows, key=lambda r: r["id"])
    ).encode()
    manifest = {
        "retrieved_at": datetime.now(UTC).date().isoformat(),
        "usage": (
            "Untouched test set: not used to fit directions, compare direction "
            "layers or strengths, or select editing methods."
        ),
        "sources": {"harmbench": original["sources"]["harmbench"]},
        "selection_manifest_sha256": hashlib.sha256(
            (out_dir / "manifest.json").read_bytes()
        ).hexdigest(),
        "files": {
            "harmbench_rest.jsonl": {
                "count": len(rows),
                "sha256": hashlib.sha256(data).hexdigest(),
                "source": "harmbench",
            }
        },
        "samples": {r["id"]: r["meta"] for r in sorted(rows, key=lambda r: r["id"])},
    }
    (out_dir / "harmbench_rest.jsonl").write_bytes(data)
    (out_dir / "manifest_rest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=407)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--harmbench-rest", action="store_true")
    args = parser.parse_args(argv)
    if args.harmbench_rest:
        freeze_rest(args.out_dir)
        return 0
    freeze(prepare(args.seed), args.out_dir, args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
