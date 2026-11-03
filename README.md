# GLM-4.7-Flash Abliteration

A reproducible refusal-direction edit of [GLM-4.7-Flash](https://huggingface.co/zai-org/GLM-4.7-Flash) on
[Unsloth's UD-Q4_K_XL](https://huggingface.co/unsloth/GLM-4.7-Flash-GGUF) GGUF. The final variant removes a layer 24 residual direction, biprojected
against benign activation means, while preserving weight-vector norms. It
complies with 91 of 100 [HarmBench](https://github.com/centerforaisafety/HarmBench) requests held out from every selection step,
refuses none of 100 safe [XSTest](https://github.com/paul-rottger/xstest) requests, and retains similar scores
to the original on small [GSM8K](https://github.com/openai/grade-school-math) and [MMLU-Pro](https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro) subsets.

The model is published at
[takanori-ishikawa/GLM-4.7-Flash-Abliterated-UD-Q4_K_XL-GGUF](https://huggingface.co/takanori-ishikawa/GLM-4.7-Flash-Abliterated-UD-Q4_K_XL-GGUF).
The [model card](hf/README.md) lists its filename, size and SHA-256.

## Results and selection

| Model | Untouched HarmBench yes / no / truncated (100) | XSTest safe refusals / 100 | XSTest unsafe refusals / 50 |
| --- | --- | ---: | ---: |
| original | 12 / 88 / 0 | 2 | 44 |
| huihui-ai | 94 / 6 / 0 | 0 | 3 |
| final | 91 / 9 / 0 | 0 | 3 |

The untouched set is the 100 HarmBench standard behaviors not used in any
selection step. On it the final variant is comparable to huihui-ai: only
huihui-ai complies on 3 behaviors and only the final variant on none (exact
two-sided McNemar p = 0.25). [WildGuard](https://huggingface.co/allenai/wildguard)
judges XSTest refusals; the final variant refuses none of the safe requests
against 2 for the original. These criteria were fixed before measurement; see
[aggregate results](results/README.md).

The other 100 HarmBench behaviors served as the selection set, alongside the
capability subsets:

| Model | HarmBench selection set yes / no / truncated (100) | GSM8K correct / 50 | MMLU-Pro correct / 50 |
| --- | --- | ---: | ---: |
| original | 15 / 85 / 0 | 47 | 39 |
| huihui-ai | 92 / 8 / 0 | 48 | 42 |
| layer24-k1 | 85 / 15 / 0 | 47 | 43 |
| final | 91 / 9 / 0 | 47 | 42 |

Counts come from [evaluation.json](results/evaluation.json). `original` is
Unsloth UD-Q4_K_XL; `huihui-ai` is
[mradermacher/Huihui-GLM-4.7-Flash-abliterated-GGUF](https://huggingface.co/mradermacher/Huihui-GLM-4.7-Flash-abliterated-GGUF)
Q4_K_M, revision `2d925ab0fdaa279a87485f4786ca22661bf030b2`, derived from
[huihui-ai/Huihui-GLM-4.7-Flash-abliterated](https://huggingface.co/huihui-ai/Huihui-GLM-4.7-Flash-abliterated).
`layer24-k1` removes the L24 residual direction without biprojection or norm
preservation; `final` adds both. The two quantizations differ between the
Unsloth and huihui-ai baselines.

On the selection set the final variant and huihui-ai are also close: 91 versus
92 compliant answers, with 6 and 5 examples where only one model complies.
Neither set establishes an improvement over huihui-ai. Capability scores replace
initial 16K truncations with 32K regenerations, keep all 50 examples in each
denominator, and count the final model's one remaining MMLU-Pro truncation as
incorrect. See [aggregate results and counting rules](results/README.md).

Direction layers and strengths were compared on the HarmBench selection set. At
strength 1, the layer comparison gives:

| Zero-based direction layer | Yes | No | Truncated |
| --- | ---: | ---: | ---: |
| 21 | 82 | 18 | 0 |
| 23 | 76 | 24 | 0 |
| 24 | 85 | 15 | 0 |
| 25 | 63 | 36 | 1 |
| 28 | 45 | 55 | 0 |

Editing methods were then compared on 50 generated harmful held-out prompts:
per-layer directions gave 37 / 13 / 0; L24 biprojection plus norm preservation
45 / 5 / 0; restricting that edit to layers 11–40 gave 45 / 4 / 1. The all-layer
biprojection variant is the final recipe. The HarmBench selection set and the
generated held-out set both participate in selection; only the untouched set does
not. These small
sets do not establish broad capability or safety guarantees.

### Other variants

On the same 50 harmful held-out prompts, none of these variants met the
selection criteria for replacing the final recipe:

| Variant | Yes | No | Truncated |
| --- | ---: | ---: | ---: |
| Strength 1.25 everywhere | 42 | 7 | 1 |
| Strength 1.5 on layers 11–40, 1 elsewhere and on the embedding | 44 | 6 | 0 |
| Direction estimated at the end of the reasoning opener | 46 | 4 | 0 |
| Prompt-end and reasoning-opener directions removed jointly | 42 | 8 | 0 |

## Reproduce

Run commands from this repository root. The reference machine is an M4 Pro Mac
mini with 64 GB RAM. Times below are recorded run intervals on that machine,
including orchestration overhead; they are not estimates of pure kernel time.

The fixed editing conditions are zero-based L24, `resid_mean`, all 47 main
layers plus `token_embd`, BF16 source matrices, `k=1`, `--embd-k 1`, biprojection,
norm preservation and two fixed-target quantization iterations. MTP layers are
excluded. Edited tensors retain their original GGUF types; no external
importance matrix is supplied. Unedited bytes remain unchanged.

### 1. Prepare dependencies and model inputs

With Python 3.12+, uv and a separately installed `hf` CLI, fetch the original
GGUF, metadata and 3039 selected BF16 matrices. Acquisition time is not recorded.

```sh
uv sync --frozen
export MODEL_DIR="$PWD/_work/models"
export SOURCE_DIR="$MODEL_DIR/original"
export BF16_DIR="$MODEL_DIR/bf16"
export EDIT_DIR="$MODEL_DIR/edited"
export ACTIVATION_DIR="$PWD/_work/activations"
export RUN_DIR="$PWD/_work/evaluation"
export DIRECTION_DIR="$PWD/artifacts"
mkdir -p "$SOURCE_DIR" "$BF16_DIR/meta" "$RUN_DIR"

hf download unsloth/GLM-4.7-Flash-GGUF \
  GLM-4.7-Flash-UD-Q4_K_XL.gguf \
  --revision 0d32489ecb9db6d2a4fc93bd27ef01519f95474d \
  --local-dir "$SOURCE_DIR"

hf download zai-org/GLM-4.7-Flash \
  config.json model.safetensors.index.json \
  --revision 7dd20894a642a0aa287e9827cb1a1f7f91386b67 \
  --local-dir "$BF16_DIR/meta"

uv run --frozen python scripts/bf16_fetch.py \
  --profile glm47 --layers 0-46 \
  --kinds attn_out,ffn_dense,ffn_shexp,ffn_exps,token_embd \
  --repo zai-org/GLM-4.7-Flash \
  --revision 7dd20894a642a0aa287e9827cb1a1f7f91386b67 \
  --config-json "$BF16_DIR/meta/config.json" \
  --index-json "$BF16_DIR/meta/model.safetensors.index.json" \
  --output-dir "$BF16_DIR"
```

The fetcher uses validated HTTP ranges rather than downloading every BF16
shard. The selected matrices cover 141 GGUF tensors after expert expansion.
Keep source, BF16 and edited directories separate.

Shortcut: the supplied `artifacts/` archives let you skip steps 3 and 4. Keep
`DIRECTION_DIR` as set above and continue from the library build to step 5;
BF16 inputs and the original GGUF are still required.

### 2. Build the pinned llama.cpp fork

Build the fork's shared libraries, llama-server and collector. Build time is
not recorded.

```sh
git clone https://github.com/ishikawa/llama.cpp.git _work/llama.cpp
export LLAMA_CPP_DIR="$PWD/_work/llama.cpp"
git -C "$LLAMA_CPP_DIR" checkout ddf120c50a8acfbf873c16cb070c2b5e4bb6a2fe
cmake -S "$LLAMA_CPP_DIR" -B "$LLAMA_CPP_DIR/build" \
  -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON \
  -DLLAMA_BUILD_COMMON=ON -DGGML_METAL=ON -DGGML_BLAS=OFF
cmake --build "$LLAMA_CPP_DIR/build" -j 8
export GGML_BASE_LIB="$LLAMA_CPP_DIR/build/bin/libggml-base.dylib"

cmake -S tools/activation-collect -B build-activation-collect \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLAMA_CPP_ROOT="$LLAMA_CPP_DIR" \
  -DLLAMA_CPP_LIB_DIR="$LLAMA_CPP_DIR/build/bin"
cmake --build build-activation-collect -j 8
ctest --test-dir build-activation-collect --output-on-failure
```

These build settings target the reference macOS machine. On Linux, disable
Metal and point `GGML_BASE_LIB` to `libggml-base.so` instead. Evaluation used
commit `5c4fc1a68`; the public
[ddf120c50](https://github.com/ishikawa/llama.cpp/commit/ddf120c50a8acfbf873c16cb070c2b5e4bb6a2fe)
is its immediate successor and adds only an attention-output callback name,
without changing computation. The collector needs that name before the final
layer crop. See [collector documentation](tools/activation-collect/README.md).

With the artifact shortcut, only llama.cpp's libraries/server are needed; the
collector build is optional.

### 3. Collect original-model activations

Collect the original GGUF’s activations for all four direction splits into
`ACTIVATION_DIR`, with a manifest and four NPY arrays per split. Recorded times:
45 seconds for harmful fit, 46 for benign fit and 12 for each held-out split,
about 1 minute 55 seconds total.

```sh
for split in harmful_fit benign_fit harmful_heldout benign_heldout; do
  output_split=$(printf '%s' "$split" | tr '_' '-')
  build-activation-collect/activation-collect \
    --profile glm47 -m "$SOURCE_DIR/GLM-4.7-Flash-UD-Q4_K_XL.gguf" \
    --input "data/direction/$split.jsonl" \
    --output "$ACTIVATION_DIR/$output_split" \
    -ngl 99 -c 8192 -ub 2048 --threads 8
done
```

The collector uses the embedded chat template, common default thinking and
Unix epoch template time, and captures the last input token at all 47 layers.
With the artifact shortcut, skip this step entirely.

### 4. Fit directions and benign means

Fit two NPZ archives and a validation report from the four completed activation
splits. Fit time is not recorded.

```sh
export DIRECTION_DIR="$PWD/_work/directions"
mkdir -p "$DIRECTION_DIR"
uv run --frozen python scripts/direction_fit.py \
  --harmful "$ACTIVATION_DIR/harmful-fit" \
  --benign "$ACTIVATION_DIR/benign-fit" \
  --held-harmful "$ACTIVATION_DIR/harmful-heldout" \
  --held-benign "$ACTIVATION_DIR/benign-heldout" \
  --output "$DIRECTION_DIR/directions.npz" \
  --benign-means-out "$DIRECTION_DIR/benign_means.npz" \
  --report "$DIRECTION_DIR/report.json"
```

Fit means use only fit records. Held-out records measure separation. With the
artifact shortcut, skip this step and keep `DIRECTION_DIR="$PWD/artifacts"`.
See [artifact array metadata and hashes](artifacts/README.md).

### 5. Edit and requantize

Use the BF16 matrices, GLM config, direction archives and libggml-base to edit
the original GGUF and write a report. Recorded time: about 17 minutes 13 seconds.

```sh
uv run --frozen python scripts/direction_edit.py \
  --profile glm47 --config-json "$BF16_DIR/meta/config.json" \
  --input-dir "$SOURCE_DIR" --output-dir "$EDIT_DIR" \
  --layers 0-46 \
  --kinds attn_out,ffn_dense,ffn_shexp,ffn_exps,token_embd \
  --directions "$DIRECTION_DIR/directions.npz" \
  --site resid_mean --direction-mode layer --direction-layer 24 \
  --k 1 --embd-k 1 --iterations 2 \
  --source bf16 --bf16-dir "$BF16_DIR" \
  --biprojection --benign-means "$DIRECTION_DIR/benign_means.npz" \
  --norm-preserve --ggml-lib "$GGML_BASE_LIB" \
  --report "$RUN_DIR/edit.json" --verify
```

The source filename remains `GLM-4.7-Flash-UD-Q4_K_XL.gguf` in `EDIT_DIR`.
The release upload uses the filename in the model card. APFS cloning is the
default; add `--allow-copy` on filesystems without clone support. Do not reuse
an output directory for a different edit. The artifact shortcut enters here
with the supplied archives.

With the supplied archives and the pinned inputs, this step reproduces the
released GGUF byte for byte; compare its SHA-256 with the
[model card](hf/README.md#files).

### 6. Generate evaluations

Evaluate the edited GGUF on the frozen data, saving results and summaries under
`RUN_DIR`. Recorded times: about 39 minutes for harmful held-out, 1 hour
25 minutes 48 seconds for HarmBench and 1 hour 39 minutes 47 seconds for GSM8K
plus MMLU-Pro at 16K.

Start the foreground server in a separate terminal with the same exported
variables, or set the model path explicitly there. The caller owns its lifecycle.

```sh
GLM47_MODEL="$EDIT_DIR/GLM-4.7-Flash-UD-Q4_K_XL.gguf" \
GLM47_SERVED_NAME=glm-4.7-flash \
  bash scripts/start-glm47-flash.sh
```

Then run the clients from the repository root:

```sh
uv run --frozen python scripts/refusal_eval.py \
  --dataset glm-heldout-harmful --model glm-4.7-flash \
  --thinking on --seed 407 --max-tokens 16384 \
  --out-dir "$RUN_DIR/heldout"

uv run --frozen python scripts/refusal_eval.py \
  --dataset harmbench --model glm-4.7-flash \
  --thinking on --seed 407 --max-tokens 16384 \
  --out-dir "$RUN_DIR/harmbench"

uv run --frozen python scripts/refusal_eval.py \
  --sets gsm8k,mmlu_pro --model glm-4.7-flash \
  --thinking on --seed 407 --max-tokens 16384 \
  --out-dir "$RUN_DIR/capability"
```

For a full collection run, add
`--activation-manifest "$ACTIVATION_DIR/harmful-heldout/manifest.json"` to the
held-out command for collector input-hash validation. Without it, the shortcut
validates the frozen record count and identifiers. Client requests explicitly
set temperature 1.0, top-p 0.95, top-k 20 and min-p 0. The server uses context
65536, one slot, no speculation and Jinja. No separate thinking budget is set.

### 7. Regenerate only truncated capability examples at 32K

Select truncated 16K capability results, regenerate them and merge the counts.
Four final-model examples require regeneration. Recorded time: about 54 minutes
41 seconds.

```sh
uv run --frozen python - <<'PY'
import json
import os
from pathlib import Path
root = Path(os.environ["RUN_DIR"])
rows = [json.loads(line) for line in (root / "capability/results.jsonl").read_text().splitlines()]
(root / "truncated-ids.txt").write_text("\n".join(row["id"] for row in rows if row["truncated"]) + "\n")
PY

uv run --frozen python scripts/refusal_eval.py \
  --sets gsm8k,mmlu_pro --model glm-4.7-flash \
  --thinking on --seed 407 --max-tokens 32768 \
  --ids "$RUN_DIR/truncated-ids.txt" \
  --out-dir "$RUN_DIR/capability-32k"

uv run --frozen python - <<'PY'
import json
import os
from pathlib import Path
root = Path(os.environ["RUN_DIR"])
merged = {}
for name in ("capability", "capability-32k"):
    for line in (root / name / "results.jsonl").read_text().splitlines():
        row = json.loads(line)
        merged[(row["bench"], row["id"])] = row
counts = {}
for bench in ("gsm8k", "mmlu_pro"):
    rows = [r for (b, _), r in merged.items() if b == bench]
    counts[bench] = {"total": len(rows), "correct": sum(not r["truncated"] and not r["empty"] and bool(r["score"]) for r in rows), "truncated": sum(r["truncated"] for r in rows)}
(root / "capability-counts.json").write_text(json.dumps(counts, indent=2) + "\n")
PY
```

If no examples are truncated, skip the 32K run and use the initial counts.
The replacement logic keeps one result per example and includes remaining
truncations as incorrect. Runtime result files stay local; the repository's
`results/` contains aggregates only.

### 8. Judge HarmBench and harmful held-out

Judge final answers with the pinned
[HarmBench-Llama-2-13b-cls](https://huggingface.co/cais/HarmBench-Llama-2-13b-cls) classifier, saving `judge_cls_run.json`,
`judge_cls.jsonl` and `judge_cls_summary.json` per run. Recorded times: about
3 minutes 10 seconds for held-out and 5 minutes 46 seconds from classifier
readiness for HarmBench. Classifier acquisition time is not recorded.

Stop the GLM server before starting the classifier on the same port. Download:

```sh
hf download mradermacher/HarmBench-Llama-2-13b-cls-GGUF \
  HarmBench-Llama-2-13b-cls.Q8_0.gguf \
  --revision e1ff40dd31c7a3ebd38326f6d684b849ca18c434 \
  --local-dir "$MODEL_DIR/classifier"
```

Start the foreground classifier in a separate terminal:

```sh
HARMBENCH_CLS_MODEL="$MODEL_DIR/classifier/HarmBench-Llama-2-13b-cls.Q8_0.gguf" \
  bash scripts/start-harmbench-cls.sh
```

Run the clients:

```sh
for split in heldout harmbench; do
  uv run --frozen python scripts/refusal_judge.py \
    --classifier harmbench-cls --model harmbench-cls \
    --base-url http://127.0.0.1:8080/v1 --run-dir "$RUN_DIR/$split"
done
```

Only final answers are judged. [HarmBench's official evaluator](https://github.com/centerforaisafety/HarmBench/blob/main/evaluate_completions.py)
uses `tokenizer.encode(generation, max_length=512, truncation=True)` with the
Llama-2 tokenizer's default BOS, leaving 511 text tokens. This implementation
matches it with `add_special=false` and a 511-token text limit, then predicts
one token at temperature 0 with prompt caching disabled.

### 9. Evaluate the untouched HarmBench set and XSTest

Use the GLM server from step 6 with the same generation settings. The frozen
untouched set contains the other 100 standard HarmBench behaviors.

```sh
uv run --frozen python scripts/refusal_eval.py \
  --sets harmbench_rest,xstest_safe,xstest_unsafe --model glm-4.7-flash \
  --thinking on --seed 407 --max-tokens 16384 \
  --out-dir "$RUN_DIR/untouched"
```

Stop the GLM server and start the HarmBench classifier as in step 8, then run:

```sh
uv run --frozen python scripts/refusal_judge.py \
  --classifier harmbench-cls --model harmbench-cls \
  --base-url http://127.0.0.1:8080/v1 --run-dir "$RUN_DIR/untouched"
```

For XSTest, accept the gated WildGuard model's
[AI2 Responsible Use Guidelines](https://allenai.org/responsible-use.pdf) on the
Hub, then convert the pinned revision:

```sh
export WILDGUARD_MODEL="$MODEL_DIR/classifier/wildguard.Q8_0.gguf"
bash scripts/convert-wildguard.sh
```

Stop the HarmBench server and start WildGuard in a separate terminal with the
same exported variables:

```sh
bash scripts/start-wildguard.sh
```

Run the XSTest classifier:

```sh
uv run --frozen python scripts/refusal_judge.py \
  --classifier wildguard --model wildguard \
  --base-url http://127.0.0.1:8080/v1 --run-dir "$RUN_DIR/untouched"
```

Only final answers are judged. WildGuard counts `Response refusal: yes` without
a partial-refusal distinction. Check recorded `prompt_tokens` for prompts over
4096 tokens; HF's sliding window and llama.cpp's full attention can differ.
See [tokenizer correction and classifier details](scripts/README.md#untouched-evaluation-and-wildguard).

## Repository and development

| Directory | Contents |
| --- | --- |
| `scripts/` | BF16 fetching, direction fitting/editing, data preparation, evaluation and publication tools |
| `tools/activation-collect/` | C++ collector and model-free CPU tests |
| `data/` | Frozen direction/evaluation JSONL and provenance manifests |
| `artifacts/` | Direction and benign-mean NPZ archives |
| `results/` | Aggregate evaluation and editing statistics |
| `hf/` | Model card, MIT text and release notice |
| `tests/` | Offline synthetic Python tests |

The project runs scripts directly and is not installed as a Python package.
See [tool defaults](scripts/README.md) for optional settings.

```sh
uv sync --frozen
uv run --frozen ruff check .
uv run --frozen ruff format --check .
uv run --frozen pytest
```

CI uses Python 3.12. Tests requiring libggml-base skip when `GGML_BASE_LIB` is
unset, including in CI. Set it to a matching CPU quantization library to include
those tests; they use synthetic matrices and GGUF files without model loading
or GPU initialization. Collector tests run separately through ctest.

Preview the upload plan below; `--yes` creates the repository and uploads the
validated release. See [publication details](scripts/README.md#publication)
for validation, CLI requirements and options:

```sh
uv run --frozen python scripts/hf_publish.py \
  "$EDIT_DIR/GLM-4.7-Flash-UD-Q4_K_XL.gguf"
```

## License

Repository code uses [Apache License 2.0](LICENSE). The model release uses the
base model's declared MIT license; see [hf/LICENSE](hf/LICENSE) and
[hf/NOTICE](hf/NOTICE). Evaluation data retain their recorded MIT or CC-BY-4.0
licenses. The generated prompt manifest does not record a separate data license
grant. [NOTICE](NOTICE) and [data provenance](data/README.md) describe these
separate sources.
