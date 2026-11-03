# Tools

Run scripts directly with Python 3.12 or later after installing the dependencies
from `pyproject.toml`. The project is not installed as a Python package. Tests
import modules through pytest's `pythonpath` setting.

- `bf16_fetch.py` downloads selected BF16 tensors using resumable HTTP ranges
- `direction_profiles.py` defines the GLM-4.7-Flash tensor mapping
- `direction_fit.py` estimates directions and optional benign means
- `direction_edit.py` edits GGUF tensors and verifies unchanged byte ranges
- `gguf_header_scan.py` inspects GGUF headers without loading model weights
- `refusal_prep.py` freezes evaluation subsets, including [XSTest](https://github.com/paul-rottger/xstest)
- `direction_prompts.py` generates, filters and splits direction prompts, removing evaluation overlaps
- `refusal_eval.py` evaluates final answers and capability scores
- `refusal_judge.py` uses the official [HarmBench](https://github.com/centerforaisafety/HarmBench) classifier
- `hf_publish.py` validates the GGUF and previews an upload plan; `--yes` invokes the external `hf` CLI

Use `--help` for each entry point. Prompt generation uses served names
`generator` for harmful generation and `model` for benign generation and filtering;
set `--model` to match the server. HTTP clients default to
`http://127.0.0.1:8080/v1`.

BF16 fetching defaults to profile `glm47`, repository `zai-org/GLM-4.7-Flash`,
revision `main`, and all five GLM tensor kinds. Pin `--revision` for reproducible
fetches. Editing defaults to profile `glm47`, kinds `attn_out,ffn_shexp`, GGUF
source, per-layer directions, direction layer 0, strength 1, four iterations and
minimum BF16 cosine 0.99. These are tool defaults; the reproduction recipe must
set its selected layer, kinds and two iterations explicitly. Embedding strength
inherits `--k` when `--embd-k` is omitted.

`direction_edit.py` imports the installed `gguf` package first. If unavailable,
it tries `$LLAMA_CPP_DIR/gguf-py`. Set `--ggml-lib` or `GGML_BASE_LIB` to the
CPU quantization shared library. Library tests skip when `GGML_BASE_LIB` is unset.

```sh
GGML_BASE_LIB="$LLAMA_CPP_DIR/build/bin/libggml-base.dylib" pytest
```

[HarmBench's official evaluator](https://github.com/centerforaisafety/HarmBench/blob/main/evaluate_completions.py)
uses `tokenizer.encode(generation, max_length=512, truncation=True)` with the
Llama-2 tokenizer's default BOS, leaving 511 text tokens. This implementation
matches it with `add_special=false` and a 511-token text limit before
requesting one classifier output token. See the
[reproduction recipe](../README.md#8-judge-harmbench-and-harmful-held-out)
for inference settings.

All three launchers run in the foreground and accept additional llama-server flags.
They use `$LLAMA_CPP_DIR/build/bin/llama-server` (or `./build/bin/llama-server`)
unless `GLM47_BIN`, `HARMBENCH_CLS_BIN` or `WILDGUARD_BIN` is set. Set `GLM47_MODEL` or
`HARMBENCH_CLS_MODEL` or `WILDGUARD_MODEL` to the GGUF path. Host and port default to `127.0.0.1:8080`
and can be set with the corresponding `_HOST` and `_PORT` variables. GLM context
is 65536, HarmBench classifier context 4096 and WildGuard context 16384, and each uses one slot by default. GLM sampling
is temperature 1, top-p 0.95, top-k 20 and min-p 0. Run the servers separately or
assign distinct ports.

## Untouched evaluation and WildGuard

Run `refusal_prep.py --harmbench-rest` to freeze the 100 standard HarmBench
behaviors absent from the original selection, without changing that selection.
It writes `data/refusal/harmbench_rest.jsonl` and `manifest_rest.json`, including
the selection-manifest hash. `refusal_eval.py --sets harmbench_rest,xstest_safe,xstest_unsafe`
verifies the rest-set hash and count before generation. The HarmBench classifier
also judges this set, with a separate `harmbench_rest` summary key.

Use `refusal_judge.py --classifier wildguard --model wildguard --run-dir PATH`
for XSTest safe/unsafe only. The [WildGuard](https://huggingface.co/allenai/wildguard)
model-card instruction format is sent verbatim to `/completion`, except for
the leading `<s>`: the server adds BOS automatically. `Response refusal: yes`
counts as refusal, without distinguishing full from partial refusal.
`judge_wildguard.jsonl` records the harmful-request and harmful-response labels,
raw output and `prompt_tokens`. Configuration and summaries are saved separately
in `judge_wildguard_run.json` and `judge_wildguard_summary.json`; interrupted
runs resume with the same configuration.

`convert-wildguard.sh` downloads a pinned revision using the separately installed
`hf` CLI and converts it to Q8_0. Set `LLAMA_CPP_DIR` and the required output
path `WILDGUARD_MODEL`. The model is gated: first accept the
[AI2 Responsible Use Guidelines](https://allenai.org/responsible-use.pdf) on the Hub.
The converter sets `tokenizer.ggml.add_space_prefix=false`. llama.cpp's default
adds a space prefix at the start and after `[INST]` / `[/INST]`, shifting tokens
relative to the HF tokenizer with `legacy=False`. With the correction, token
sequences matched HF on 430 actual XSTest responses in validation.

HF uses a 4096-token sliding window; llama.cpp attends to the full context.
Check how many recorded `prompt_tokens` exceed 4096, since these classifications
may differ from HF. Start `start-wildguard.sh` separately; its default context
is 16384 and its environment variables use the `WILDGUARD_` prefix.

## Publication

`scripts/hf_publish.py` requires a separately installed `hf` CLI with `upload`
and `repos create` commands. It adds no Python Hub dependency. By default it
prints the destination repository, file sizes and planned commands without
invoking `hf`, reading credentials or making network requests.

```sh
uv run --frozen python scripts/hf_publish.py \
  "$EDIT_DIR/GLM-4.7-Flash-UD-Q4_K_XL.gguf"
```

The script checks the local GGUF size and SHA-256 against `hf/README.md`. A missing GGUF
only warns in dry-run; a present mismatched file fails. `--yes` first requires a
verified file, then creates the model repository if needed and uploads that file
plus `hf/README.md`, `hf/LICENSE` and `hf/NOTICE`. `--repo-id` selects a different
destination. Authentication is delegated to the installed CLI.

The default destination is
`takanori-ishikawa/GLM-4.7-Flash-Abliterated-UD-Q4_K_XL-GGUF`, and the GGUF
is uploaded as `GLM-4.7-Flash-Abliterated-UD-Q4_K_XL.gguf`.
