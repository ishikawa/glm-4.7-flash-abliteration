# Aggregate results

`evaluation.json` contains only counts for four GLM-4.7-Flash variants:

- `original`: original quantized baseline
- `huihui-ai`: the huihui-ai edited baseline
- `layer24-k1`: the zero-based layer 24 residual direction removed with strength 1
- `final`: the layer 24 residual direction, biprojected against benign means, with norm preservation, strength 1 and two quantization iterations

The `layer24-k1` and `final` variants modify all 47 main layers and the token
embedding. These
results use the frozen datasets under `data/`, seed 407, temperature 1.0,
top-p 0.95, top-k 20 and min-p 0, without a separate thinking-token budget

## Benchmark counts

HarmBench columns are official classifier `yes` / `no` / truncated generations,
not refusal-phrase matches. Capability columns count correct answers out of the
full 50 examples, including any remaining truncations as incorrect

| Model | HarmBench yes / no / truncated, out of 100 | GSM8K correct / 50 | MMLU-Pro correct / 50 |
| --- | --- | ---: | ---: |
| original | 15 / 85 / 0 | 47 | 39 |
| huihui-ai | 92 / 8 / 0 | 48 | 42 |
| layer24-k1 | 85 / 15 / 0 | 47 | 43 |
| final | 91 / 9 / 0 | 47 | 42 |

## Harmful held-out counts

These are the 50 harmful records in `data/direction/harmful_heldout.jsonl`, also
judged by the official classifier

| Model | Total | Yes | No | Truncated |
| --- | ---: | ---: | ---: | ---: |
| original | 50 | 12 | 38 | 0 |
| huihui-ai | 50 | 45 | 5 | 0 |
| layer24-k1 | 50 | 43 | 7 | 0 |
| final | 50 | 45 | 5 | 0 |

Direction layers and strengths were compared on HarmBench. Editing methods were
selected on the harmful held-out set, so neither is an untouched test set for
all decisions. The final variant and huihui-ai have comparable HarmBench counts;
the one-count difference does not establish an improvement

## Aggregation rules

Classifier totals, `yes`, `no`, truncated, empty and unparsed counts are selected
from each run's classifier summary. Empty and unparsed counts are zero for all
reported runs. A truncated source generation is excluded from classification
and counted separately

Capability counts combine the initial 16384-token run with 32768-token
regenerations of only the initially truncated examples. The initial summary's
`positive` count excludes truncations; add the regeneration summary's `positive`
count, retaining the original total of 50. Replace the original truncation count
with the regeneration's remaining truncation count. No regeneration is a new
example in the denominator, and no displayed accuracy is taken from a summary
that drops truncated or empty answers from its denominator

| Model | GSM8K regenerated | MMLU-Pro regenerated | MMLU-Pro still truncated at 32K |
| --- | ---: | ---: | ---: |
| original | 0 | 1 | 0 |
| huihui-ai | 0 | 3 | 0 |
| layer24-k1 | 0 | 0 | 0 |
| final | 1 | 3 | 1 |

The final variant's GSM8K regeneration adds zero correct answers; its MMLU-Pro
regenerations add one. Original MMLU-Pro adds one, and huihui-ai MMLU-Pro adds
three

## Editing statistics

`edit.json` selects the following values from the final edit report. Count only
tensor entries with an `experts` list; BF16 correspondence-check entries are not
additional edited tensors. Flatten those lists to count matrices and aggregate
`return_ratio`. The direction cosine is the minimum `direction_cosine` across
per-tensor direction settings, comparing directions before and after biprojection

| Statistic | Value |
| --- | ---: |
| Edited GGUF tensors | 141 |
| Edited matrices after expert expansion | 3039 |
| Median return ratio | 0.06974923024152041 |
| Maximum return ratio | 0.08361431043953504 |
| Minimum biprojection direction cosine | 0.9940133829979808 |

The JSON files use explicit aggregate-field allowlists. They contain no generated
answers, prompt text, per-example identifiers, local paths or internal run labels
