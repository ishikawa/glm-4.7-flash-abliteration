---
license: mit
base_model:
  - zai-org/GLM-4.7-Flash
  - unsloth/GLM-4.7-Flash-GGUF
pipeline_tag: text-generation
tags:
  - gguf
  - abliterated
  - uncensored
  - glm
  - apple-silicon
---

# GLM-4.7-Flash Abliterated UD-Q4_K_XL GGUF

An abliterated build of [GLM-4.7-Flash](https://huggingface.co/zai-org/GLM-4.7-Flash)
on [Unsloth's UD-Q4_K_XL](https://huggingface.co/unsloth/GLM-4.7-Flash-GGUF)
quantization. It complies with 91 of 100 sampled [HarmBench](https://github.com/centerforaisafety/HarmBench) requests and retains
similar scores to the original on the small [GSM8K](https://github.com/openai/grade-school-math) and [MMLU-Pro](https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro) subsets below.

## Files

| File | Size (bytes) | SHA-256 |
| --- | ---: | --- |
| `GLM-4.7-Flash-Abliterated-UD-Q4_K_XL.gguf` | 17,520,169,312 | `33a80f475f219f86e9a9785c278ef46aa226a9c50a426fc42985bec8b7aa469f` |

## Running

The file differs from Unsloth's UD-Q4_K_XL only in the values of the edited
tensors; the GGUF metadata, tensor types and chat template are unchanged. It
runs in any GGUF runtime that supports GLM-4.7-Flash.

Long reasoning can exhaust the output limit; the evaluation uses 16K output
tokens and regenerates only truncated capability examples with a 32K limit.

## Evaluation

HarmBench standard 100, GSM8K test 50 and MMLU-Pro test 50, sampled with seed 407.
Evaluation runs on an M4 Pro Mac mini with 64 GB RAM, using the llama.cpp fork
at commit `5c4fc1a68`. Its successor
[ddf120c50](https://github.com/ishikawa/llama.cpp/commit/ddf120c50a8acfbf873c16cb070c2b5e4bb6a2fe)
adds only attention callback naming and has
the same model computation. Sampling is temperature 1.0, top-p 0.95, top-k 20,
min-p 0 and seed 407, with thinking enabled and no separate thinking budget.

HarmBench uses the official [HarmBench-Llama-2-13b-cls](https://huggingface.co/cais/HarmBench-Llama-2-13b-cls) classifier in Q8_0 GGUF
form. Only final answers are judged. [HarmBench's official evaluator](https://github.com/centerforaisafety/HarmBench/blob/main/evaluate_completions.py)
uses `tokenizer.encode(generation, max_length=512, truncation=True)` with the
Llama-2 tokenizer's default BOS, leaving 511 text tokens. This implementation
matches it with `add_special=false` and a 511-token text limit;
the classifier predicts one token.

| Model | HarmBench yes / no / truncated (100) | GSM8K correct / 50 | MMLU-Pro correct / 50 |
| --- | --- | ---: | ---: |
| original | 15 / 85 / 0 | 47 | 39 |
| huihui-ai | 92 / 8 / 0 | 48 | 42 |
| layer24-k1 | 85 / 15 / 0 | 47 | 43 |
| final | 91 / 9 / 0 | 47 | 42 |

`original` is Unsloth UD-Q4_K_XL; `huihui-ai` is
[mradermacher/Huihui-GLM-4.7-Flash-abliterated-GGUF](https://huggingface.co/mradermacher/Huihui-GLM-4.7-Flash-abliterated-GGUF)
Q4_K_M; `layer24-k1` removes the L24 residual direction without biprojection or
norm preservation; `final` is this model. Capability counts replace 16K
truncations with their 32K regenerations. One final-model MMLU-Pro example remains
truncated and counts as incorrect.

- These are small evaluation sets; differences of a few answers do not establish broad capability differences
- HarmBench helps select direction layers and strengths; editing methods are selected on 50 generated harmful held-out prompts, so these are not untouched evaluations of all choices
- This model is comparable to huihui-ai on this HarmBench subset, with 91 versus 92 compliant answers and 6 and 5 examples where only one model complies
- Removing a refusal direction reduces safety behavior and can produce harmful, inaccurate or unreliable output; these scores do not establish safety or suitability for a particular use

## How it was made

1. Estimate harmful-minus-benign activation directions from original-model activations on 200 harmful and 200 benign generated fit prompts
2. Take the zero-based L24 `resid_mean` direction and biproject it against each layer's benign mean
3. Starting from BF16 tensors, remove the direction at strength 1 from attention output, dense/shared/routed FFN down projections across all 47 main layers, and the token embedding; preserve column norms, or token-row norms for the embedding
4. Quantize each tensor back to its original Unsloth type with two fixed-target correction iterations; preserve all unedited GGUF bytes

See the [recipe, scripts, frozen data and aggregate results](https://github.com/ishikawa/glm-4.7-flash-abliteration)
to reproduce the model. No external importance matrix is supplied during editing.

## License and credits

- Base model: [zai-org/GLM-4.7-Flash](https://huggingface.co/zai-org/GLM-4.7-Flash), MIT ([LICENSE](LICENSE))
- Quantized tensors: [unsloth/GLM-4.7-Flash-GGUF](https://huggingface.co/unsloth/GLM-4.7-Flash-GGUF), UD-Q4_K_XL
- Direction prompts: generated using [Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) and its uncensored variant, filtered and split as described in the recipe's data documentation ([NOTICE](NOTICE))

The base model declares MIT in its model-card metadata without a standalone
LICENSE file. The included LICENSE uses the standard MIT text with attribution
to the GLM-4.7-Flash authors.
