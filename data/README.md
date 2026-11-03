# Frozen data

Use these files for reproducible direction fitting and evaluation. JSONL payloads
are byte-for-byte frozen; the manifests record counts and SHA-256 hashes. The
manifest path `eval_dir` is relative to the repository root.

## Direction data

| File under `direction/` | Records | Use |
| --- | ---: | --- |
| `harmful_fit.jsonl` | 200 | Harmful activation mean |
| `benign_fit.jsonl` | 200 | Benign activation mean and biprojection reference |
| `harmful_heldout.jsonl` | 50 | Direction separation and editing-method selection |
| `benign_heldout.jsonl` | 50 | Direction separation validation |
| `manifest.json` | — | Generation provenance, split seed, counts and hashes |

The prompts were generated with [Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) and its uncensored variant.
The recorded served names are `qwen3.8-flash-next-uncensored` for harmful
candidates and `qwen3.8-flash-next` for benign candidates and filtering. The
manifest does not pin model revisions or declare a separate license for these
generated prompts; no model-weight license is asserted as a data license here.

`scripts/direction_prompts.py` implements the preparation process:

1. Select category, request type, language and length from the seed and call number, with matching distributions for harmful and benign generation
2. Generate candidates and query the filtering model with thinking disabled; retain harmful candidates that trigger refusal phrases and benign candidates with nonempty answers that do not trigger them
3. Normalize with NFKC, collapsed whitespace and lowercase; deduplicate across both groups and remove exact or near overlaps with all 350 evaluation records
4. Reject near overlaps when at least half the candidate's word 8-grams occur in the evaluation collection; Japanese characters are individual units
5. Shuffle each group with `split:20261027:<kind>` and select 200 fit plus 50 held-out records, sorting each output by identifier

The split seed is 20261027. The frozen files are the reproducible inputs;
regenerating prompts with a served model does not guarantee identical text.

## Evaluation data

The original five subsets use selection seed 407 and were retrieved on 2026-10-26 UTC.
The manifests preserve the source URLs, declared licenses and sample metadata.

| File under `refusal/` | Records | Use | Source | Recorded license |
| --- | ---: | --- | --- | --- |
| `harmbench.jsonl` | 100 | Classifier compliance evaluation; direction layer/strength comparison | [HarmBench](https://github.com/centerforaisafety/HarmBench) | MIT |
| `harmbench_rest.jsonl` | 100 | Untouched HarmBench test set | [HarmBench](https://github.com/centerforaisafety/HarmBench) | MIT |
| `gsm8k.jsonl` | 50 | Arithmetic capability | [GSM8K](https://github.com/openai/grade-school-math) | MIT |
| `mmlu_pro.jsonl` | 50 | Multiple-choice capability | [MMLU-Pro](https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro) | MIT |
| `xstest_safe.jsonl` | 100 | Safe requests and evaluation-overlap exclusion | [XSTest, paul-rottger/xstest](https://github.com/paul-rottger/xstest) | CC-BY-4.0 |
| `xstest_unsafe.jsonl` | 50 | Unsafe requests and evaluation-overlap exclusion | [XSTest, paul-rottger/xstest](https://github.com/paul-rottger/xstest) | CC-BY-4.0 |
| `manifest.json` | — | Sources, sampling metadata, counts and hashes | — | Per source above |
| `manifest_rest.json` | — | Untouched set provenance, selection-manifest hash, counts and hashes | — | MIT |

`manifest_rest.json` records the untouched set separately.

These are normalized subsets of the linked datasets. XSTest attribution is to
the linked source repository; its prompts are redistributed under the recorded
CC-BY-4.0 license. Source fields are normalized into the common JSONL schema and
capability inputs include final-answer format instructions. No new license is
inferred for the evaluation datasets.

`scripts/refusal_prep.py` selects the subsets as follows:

- HarmBench: filter the CSV to `FunctionalCategory == standard`, then select 100 proportionally by `SemanticCategory`
- HarmBench rest: take the complement of the 100 selected `harmbench.jsonl` records within all 200 standard behaviors
- XSTest: split by `label` into safe and unsafe, then select 100 and 50 respectively, proportionally by `type`
- Proportional selection: allocate integer quotas by largest remainder, break equal remainders with a seeded shuffle of sorted categories, and sample within each category using the same seeded generator
- GSM8K: sample 50 distinct test row indices uniformly with `random.Random(407)`, then sort the selected indices; extract the reference after the final `####`
- MMLU-Pro: sample 50 distinct test row indices uniformly with `random.Random(407)` from the datasets-server row count, sort them, and fetch those rows; this is not category stratification
- Freeze all outputs in identifier order with normalized JSON encoding and record their hashes

To create new subsets from upstream sources:

```sh
python scripts/refusal_prep.py --seed 407 --out-dir _work/new-evaluation-data
```

The recorded upstream URLs use main/master branches or an unpinned rows API.
A new retrieval may differ from these frozen files. XSTest remains included
because direction-data preparation excludes overlaps against the original
350-record evaluation collection. HarmBench helped select direction layers and
strengths; it is not an untouched holdout. Editing methods were compared on the
50 harmful direction held-out records.

The selection caveat above applies to `harmbench.jsonl`. The separate
`harmbench_rest.jsonl` is an untouched test set: it was not used to fit
directions, compare direction layers or strengths, or select editing methods.
All 500 frozen direction prompts (400 fit and 100 held-out) were checked against
this set using `direction_prompts.overlaps_eval`: normalized exact matching or
at least half of a candidate's word 8-grams matching the evaluation collection.
There were zero overlaps.
