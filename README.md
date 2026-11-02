# GLM-4.7-Flash Abliteration

A reproducible refusal-direction edit of GLM-4.7-Flash on Unsloth's UD-Q4_K_XL
GGUF. The final variant removes a layer 24 residual direction, biprojected
against benign activation means, while preserving weight-vector norms. It
complies with 91 of 100 sampled HarmBench requests and retains similar scores
to the original on the small capability subsets below

The release is
[takanori-ishikawa/GLM-4.7-Flash-Abliterated-UD-Q4_K_XL-GGUF](https://huggingface.co/takanori-ishikawa/GLM-4.7-Flash-Abliterated-UD-Q4_K_XL-GGUF).
The [model card](hf/README.md) lists its filename, size and SHA-256

## Results and selection

| Model | HarmBench yes / no / truncated (100) | GSM8K correct / 50 | MMLU-Pro correct / 50 |
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
Unsloth and huihui-ai baselines

The final variant is comparable to huihui-ai on this HarmBench subset: 91 versus
92 compliant answers, with 6 and 5 examples where only one model complies.
This does not establish an improvement over huihui-ai. Capability scores replace
initial 16K truncations with 32K regenerations, keep all 50 examples in each
denominator, and count the final model's one remaining MMLU-Pro truncation as
incorrect. See [aggregate results and counting rules](results/README.md)

Direction layers and strengths were compared on HarmBench. At strength 1, the
layer comparison gives:

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
biprojection variant is the final recipe. HarmBench is not an untouched holdout,
and the generated held-out set participates in method selection. These small
sets do not establish broad capability or safety guarantees

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
including orchestration overhead; they are not estimates of pure kernel time

The fixed editing conditions are zero-based L24, `resid_mean`, all 47 main
layers plus `token_embd`, BF16 source matrices, `k=1`, `--embd-k 1`, biprojection,
norm preservation and two fixed-target quantization iterations. MTP layers are
excluded. Edited tensors retain their original GGUF types; no external
importance matrix is supplied. Unedited bytes remain unchanged

### 1. Prepare dependencies and model inputs

Input: Python 3.12+, uv, a separately installed `hf` CLI, and the pinned model
repositories below. Output: a Python environment, the original GGUF and 3039
selected BF16 matrices plus metadata. Acquisition time is not recorded

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
Keep source, BF16 and edited directories separate

Shortcut: the supplied `artifacts/` archives let you skip steps 3 and 4. Keep
`DIRECTION_DIR` as set above and continue from the library build to step 5;
BF16 inputs and the original GGUF are still required

### 2. Build the pinned llama.cpp fork

Input: the public fork source. Output: shared libraries, llama-server and the
collector binary. Build time is not recorded

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
layer crop. See [collector documentation](tools/activation-collect/README.md)

With the artifact shortcut, only llama.cpp's libraries/server are needed; the
collector build is optional

### 3. Collect original-model activations

Input: the original GGUF and four frozen direction splits. Output: one manifest
and four NPY arrays per split under `ACTIVATION_DIR`. Recorded intervals are
45 seconds for harmful fit, 46 seconds for benign fit, and 12 seconds for each
held-out split, about 1 minute 55 seconds total

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
With the artifact shortcut, skip this step entirely

### 4. Fit directions and benign means

Input: completed activations for the four splits. Output: two NPZ archives and
a validation report. Fit time is not recorded

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
See [artifact array metadata and hashes](artifacts/README.md)

### 5. Edit and requantize

Input: the original GGUF, selected BF16 matrices, GLM config, direction and benign
mean archives, and libggml-base. Output: the edited GGUF and an edit report.
The recorded interval is about 17 minutes 13 seconds

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
with the supplied archives

### 6. Generate evaluations

Input: the edited GGUF and frozen evaluation data. Output: per-run results and
summaries under `RUN_DIR`. Recorded intervals are about 39 minutes for harmful
held-out, 1 hour 25 minutes 48 seconds for HarmBench, and 1 hour 39 minutes
47 seconds for GSM8K plus MMLU-Pro at 16K

Start the foreground server in a separate terminal with the same exported
variables, or set the model path explicitly there. The caller owns its lifecycle

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
65536, one slot, no speculation and Jinja. No separate thinking budget is set

### 7. Regenerate only truncated capability examples at 32K

Input: the 16K capability results. Output: a local identifier file, replacement
results and aggregate capability counts. Four final-model examples require
regeneration; the recorded interval is about 54 minutes 41 seconds

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
`results/` contains aggregates only

### 8. Judge HarmBench and harmful held-out

Input: final-answer generations and the pinned classifier. Output:
`judge_cls_run.json`, `judge_cls.jsonl` and `judge_cls_summary.json` in each run.
Recorded judging intervals are about 3 minutes 10 seconds for held-out and
5 minutes 46 seconds from classifier readiness for HarmBench; classifier
acquisition time is not recorded

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

Only final answers are passed to the classifier. Generation text is clipped to
511 classifier-tokenizer tokens with `add_special=false`: this is the
implementation's upper limit. The request predicts one token at temperature 0
with prompt caching disabled. No claim about a 512-token total context limit
is implied. The supplied artifacts change only steps 3–4; evaluation and judging
remain the same

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
See [tool defaults](scripts/README.md) for optional settings

```sh
uv sync --frozen
uv run --frozen ruff check .
uv run --frozen ruff format --check .
uv run --frozen pytest
```

CI uses Python 3.12. Tests requiring libggml-base skip when `GGML_BASE_LIB` is
unset, including in CI. Set it to a matching CPU quantization library to include
those tests; they use synthetic matrices and GGUF files without model loading
or GPU initialization. Collector tests run separately through ctest

The publication script requires a separately installed `hf` CLI, with its upload
and repos-create commands. It does not add a Python Hub dependency to this
project. Preview the exact four-file upload plan without invoking `hf` or reading
credentials:

```sh
uv run --frozen python scripts/hf_publish.py \
  "$EDIT_DIR/GLM-4.7-Flash-UD-Q4_K_XL.gguf"
```

It checks the local GGUF size and SHA-256 against `hf/README.md`. A missing GGUF
only warns in dry-run; a present mismatched file fails. `--yes` first requires a
verified file, then creates the model repository if needed and uploads that file
plus `hf/README.md`, `hf/LICENSE` and `hf/NOTICE`. `--repo-id` selects a different
destination. Authentication is delegated to the installed CLI

## License

Repository code uses [Apache License 2.0](LICENSE). The model release uses the
base model's declared MIT license; see [hf/LICENSE](hf/LICENSE) and
[hf/NOTICE](hf/NOTICE). Evaluation data retain their recorded MIT or CC-BY-4.0
licenses. The generated prompt manifest does not record a separate data license
grant. [NOTICE](NOTICE) and [data provenance](data/README.md) describe these
separate sources
