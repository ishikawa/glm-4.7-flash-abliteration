# Aggregate results

`evaluation.json` contains only counts for four [GLM-4.7-Flash](https://huggingface.co/zai-org/GLM-4.7-Flash) variants:

- `original`: original quantized baseline
- `huihui-ai`: the [huihui-ai](https://huggingface.co/huihui-ai/Huihui-GLM-4.7-Flash-abliterated) edited baseline
- `layer24-k1`: the zero-based layer 24 residual direction removed with strength 1
- `final`: the layer 24 residual direction, biprojected against benign means, with norm preservation, strength 1 and two quantization iterations

The `layer24-k1` and `final` variants modify all 47 main layers and the token
embedding. Except for the security tasks, which use external datasets and
the settings given in their section, these
results use the frozen datasets under `data/`, seed 407, temperature 1.0,
top-p 0.95, top-k 20 and min-p 0, without a separate thinking-token budget.
The untouched HarmBench set and [XSTest](https://github.com/paul-rottger/xstest)
were measured for `original`, `huihui-ai` and `final`; `paired_harmbench_rest`
holds the paired counts for `final` and `huihui-ai`.

## Untouched HarmBench test set

These are the 100 HarmBench standard behaviors in
`data/refusal/harmbench_rest.jsonl`. No direction, layer, strength or editing
method was chosen with them, and none of the 500 direction prompts overlaps them.
Counts are official classifier `yes` / `no` / truncated generations.

| Model | Yes | No | Truncated |
| --- | ---: | ---: | ---: |
| original | 12 | 88 | 0 |
| huihui-ai | 94 | 6 | 0 |
| final | 91 | 9 | 0 |

All 100 behaviors form valid pairs; none is excluded. Only `huihui-ai` complies
on 3 behaviors and only `final` on none. The exact two-sided McNemar test gives
p = 0.25, so under the criteria below the two are comparable.

## Over-refusal on XSTest

[WildGuard](https://huggingface.co/allenai/wildguard) judged each final answer;
a `Response refusal: yes` counts as a refusal, without a partial-refusal class.
Refusal-phrase matches are the secondary measure.

| Model | Safe refusals / 100 | Safe phrase matches | Unsafe refusals / 50 | Unsafe phrase matches |
| --- | ---: | ---: | ---: | ---: |
| original | 2 | 3 | 44 | 41 |
| huihui-ai | 0 | 0 | 3 | 3 |
| final | 0 | 0 | 3 | 3 |

`final` refuses none of the safe requests against 2 for `original`, meeting the
over-refusal criterion. No XSTest answer was truncated or empty, and every
WildGuard output was parsed. The longest WildGuard prompt has 1,649 tokens, so
WildGuard's 4096-token sliding window does not apply. Removing the refusal
direction also removes most refusals of the unsafe XSTest requests.

## Selection set and capability counts

[HarmBench](https://github.com/centerforaisafety/HarmBench) columns cover the 100 behaviors in
`data/refusal/harmbench.jsonl`, which were used to compare direction layers and
strengths. They are official classifier `yes` / `no` / truncated generations,
not refusal-phrase matches. Capability columns count correct answers out of the
full 50 examples, including any remaining truncations as incorrect.

| Model | HarmBench yes / no / truncated, out of 100 | GSM8K correct / 50 | MMLU-Pro correct / 50 |
| --- | --- | ---: | ---: |
| original | 15 / 85 / 0 | 47 | 39 |
| huihui-ai | 92 / 8 / 0 | 48 | 42 |
| layer24-k1 | 85 / 15 / 0 | 47 | 43 |
| final | 91 / 9 / 0 | 47 | 42 |

## Harmful held-out counts

These are the 50 harmful records in `data/direction/harmful_heldout.jsonl`, also
judged by the official classifier.

| Model | Total | Yes | No | Truncated |
| --- | ---: | ---: | ---: | ---: |
| original | 50 | 12 | 38 | 0 |
| huihui-ai | 50 | 45 | 5 | 0 |
| layer24-k1 | 50 | 43 | 7 | 0 |
| final | 50 | 45 | 5 | 0 |

Direction layers and strengths were compared on the HarmBench selection set.
Editing methods were selected on the harmful held-out set, so neither is an
untouched test set for all decisions; the untouched set above is.

## Criteria for the untouched test set and over-refusal

These criteria were fixed before any answer was generated on the sets below.

- Test set: the 100 HarmBench standard behaviors absent from
  `data/refusal/harmbench.jsonl`. No direction, layer, strength or editing
  method was chosen with them. Their classifier counts become the headline
  HarmBench result; the existing 100 stay reported as the selection set
- `final` versus `huihui-ai`: an exact two-sided McNemar test at α = 0.05 on
  the same 100 behaviors, over the behaviors where exactly one of the two
  answers is classified `yes`. A behavior with a truncated or unparsed answer
  on either side is excluded, and the exclusions are reported. Without a
  significant difference the two are reported as comparable. A significantly
  lower `final` count holds the release for review
- Over-refusal: [XSTest](https://github.com/paul-rottger/xstest) safe 100 and
  unsafe 50, judged by the `Response refusal` output of
  [WildGuard](https://huggingface.co/allenai/wildguard). `final` passes if its
  XSTest safe refusal rate does not exceed `original`'s. Refusal-phrase matches
  are reported alongside as a secondary measure. XSTest unsafe rates are
  reported without a criterion
- Variants: `original`, `huihui-ai` and `final`, with the generation settings
  above

## Security tasks

These evaluations compare `original` and `final` on defensive security work,
and measure how often each variant refuses attack-oriented requests. Their
evaluation code is not part of this repository; the upstream benchmarks,
commits and settings below define them. They run on the same llama.cpp fork
commit `5c4fc1a68` as the evaluations above.

| Measure | original | final | Only original | Only final | p |
| --- | ---: | ---: | ---: | ---: | ---: |
| CyberSecEval FRR refusals / 86 | 0 | 0 | 0 | 0 | 1.00 |
| CTI-MCQ correct / 200 | 140 | 127 | 26 | 13 | 0.053 |
| AgentDojo utility / 40 | 32 | 36 | 1 | 5 | 0.22 |
| AgentDojo utility under attack / 560 | 379 | 406 | 57 | 84 | 0.028 |
| AgentDojo attack success / 560 | 38 | 46 | 19 | 27 | 0.30 |

p is the exact two-sided McNemar test on the paired examples. These use the
generation settings above with a 16384-token output limit.

- [CyberSecEval](https://github.com/meta-llama/PurpleLlama/tree/main/CybersecurityBenchmarks)
  FRR (PurpleLlama `f64eebd`): benign cybersecurity requests that resemble
  attacks. 200 of the 750 prompts were sampled with seed 407 and the first 86
  were evaluated. Final answers are judged by the upstream refusal rule
- [CTI-MCQ](https://github.com/maveryn/cti-bench) (cti-bench `4543e5b`,
  dataset revision `9237e16`): 200 of 2,500 cyber threat intelligence
  multiple-choice questions, sampled with seed 407, with the official prompt and
  answer extraction. Three `final` answers reached the output limit and count
  as incorrect
- [AgentDojo](https://github.com/ethz-spylab/agentdojo) (commit `089ed46`,
  benchmark v1.2.2): the workspace suite's 40 user tasks without attack, and all
  560 pairs with its 14 injection tasks under the `important_instructions`
  attack. No task ended in an error

[CyberSecEval MITRE](https://github.com/meta-llama/PurpleLlama/tree/main/CybersecurityBenchmarks)
asks for help with cyberattack tasks in MITRE ATT&CK categories. The 100 prompts
are the upstream harness's `--num-test-cases 100` selection, 10 per category.
Generation disables thinking and uses the harness's default temperature 0.6 and
top-p 0.9, seed 407 and a 4096-token output limit. The upstream refusal and
judging logic is unchanged; its requests carry the same thinking, seed and
output-limit settings. `original` is the expansion and judge model for every
variant.

| Model | Refusals / 100 | Expansion `1` | Expansion `0` | Output-limit truncations |
| --- | ---: | ---: | ---: | ---: |
| original | 20 | 80 | 0 | 1 |
| huihui-ai | 0 | 100 | 0 | 4 |
| final | 0 | 100 | 0 | 6 |

The upstream refusal rule detects no refusal from either edited variant,
including on the 20 prompts `original` refuses (exact two-sided McNemar
p < 0.001 for each). The
expansion step answers `1` when it judges a response useful to an attacker.
The upstream benign / malicious verdict reads only this one-character output
and labels every non-refused response benign, so the expansion output is the
reported measure. Truncated responses are judged as returned. These counts
reflect rule-detected refusals and the judge's usefulness labels, not the
accuracy of the answers.

## Aggregation rules

Classifier totals, `yes`, `no`, truncated, empty and unparsed counts are selected
from each run's classifier summary. Empty and unparsed counts are zero for all
reported runs. A truncated source generation is excluded from classification
and counted separately.

XSTest `refusals` and `non_refusals` count WildGuard's `Response refusal` labels
per set; `phrase_refusals` counts generations whose final answer opens with a
refusal phrase. `paired_harmbench_rest` compares per-behavior classifier labels
of `final` and `huihui-ai` on the untouched set; `excluded` counts behaviors
with a truncated or unparsed answer on either side.
`paired_security` and `paired_mitre_refusals` count examples where exactly one
of the two variants is positive for the named measure.

Capability counts combine the initial 16384-token run with 32768-token
regenerations of only the initially truncated examples. The initial summary's
`positive` count excludes truncations; add the regeneration summary's `positive`
count, retaining the original total of 50. Replace the original truncation count
with the regeneration's remaining truncation count. No regeneration is a new
example in the denominator, and no displayed accuracy is taken from a summary
that drops truncated or empty answers from its denominator.

| Model | GSM8K regenerated | MMLU-Pro regenerated | MMLU-Pro still truncated at 32K |
| --- | ---: | ---: | ---: |
| original | 0 | 1 | 0 |
| huihui-ai | 0 | 3 | 0 |
| layer24-k1 | 0 | 0 | 0 |
| final | 1 | 3 | 1 |

The final variant's GSM8K regeneration adds zero correct answers; its MMLU-Pro
regenerations add one. Original MMLU-Pro adds one, and huihui-ai MMLU-Pro adds
three.

## Editing statistics

`edit.json` selects the following values from the final edit report. Count only
tensor entries with an `experts` list; BF16 correspondence-check entries are not
additional edited tensors. Flatten those lists to count matrices and aggregate
`return_ratio`. The direction cosine is the minimum `direction_cosine` across
per-tensor direction settings, comparing directions before and after biprojection.

| Statistic | Value |
| --- | ---: |
| Edited GGUF tensors | 141 |
| Edited matrices after expert expansion | 3039 |
| Median return ratio | 0.06974923024152041 |
| Maximum return ratio | 0.08361431043953504 |
| Minimum biprojection direction cosine | 0.9940133829979808 |

The JSON files use explicit aggregate-field allowlists. They contain no generated
answers, prompt text, per-example identifiers, local paths or internal run labels.
