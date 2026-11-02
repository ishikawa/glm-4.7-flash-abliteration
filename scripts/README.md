# Tools

Run scripts directly with Python 3.12 or later after installing the dependencies
from `pyproject.toml`. The project is not installed as a Python package. Tests
import modules through pytest's `pythonpath` setting

- `bf16_fetch.py` downloads selected BF16 tensors using resumable HTTP ranges
- `direction_profiles.py` defines the GLM-4.7-Flash tensor mapping
- `direction_fit.py` estimates directions and optional benign means
- `direction_edit.py` edits GGUF tensors and verifies unchanged byte ranges
- `gguf_header_scan.py` inspects GGUF headers without loading model weights
- `refusal_prep.py` freezes evaluation subsets, including XSTest
- `direction_prompts.py` generates, filters and splits direction prompts, removing evaluation overlaps
- `refusal_eval.py` evaluates final answers and capability scores
- `refusal_judge.py` uses the official HarmBench classifier
- `hf_publish.py` validates the GGUF and previews an upload plan; `--yes` invokes the external `hf` CLI

Use `--help` for each entry point. Prompt generation uses served names
`generator` for harmful generation and `model` for benign generation and filtering;
set `--model` to match the server. HTTP clients default to
`http://127.0.0.1:8080/v1`

BF16 fetching defaults to profile `glm47`, repository `zai-org/GLM-4.7-Flash`,
revision `main`, and all five GLM tensor kinds. Pin `--revision` for reproducible
fetches. Editing defaults to profile `glm47`, kinds `attn_out,ffn_shexp`, GGUF
source, per-layer directions, direction layer 0, strength 1, four iterations and
minimum BF16 cosine 0.99. These are tool defaults; the reproduction recipe must
set its selected layer, kinds and two iterations explicitly. Embedding strength
inherits `--k` when `--embd-k` is omitted

`direction_edit.py` imports the installed `gguf` package first. If unavailable,
it tries `$LLAMA_CPP_DIR/gguf-py`. Set `--ggml-lib` or `GGML_BASE_LIB` to the
CPU quantization shared library. Library tests skip when `GGML_BASE_LIB` is unset

```sh
GGML_BASE_LIB="$LLAMA_CPP_DIR/build/bin/libggml-base.dylib" pytest
```

The [reproduction recipe](../README.md#8-judge-harmbench-and-harmful-held-out)
defines classifier clipping and inference settings. The
[publication instructions](../README.md#repository-and-development) describe
release validation and dry-run behavior

Both launchers run in the foreground and accept additional llama-server flags.
They use `$LLAMA_CPP_DIR/build/bin/llama-server` (or `./build/bin/llama-server`)
unless `GLM47_BIN` or `HARMBENCH_CLS_BIN` is set. Set `GLM47_MODEL` or
`HARMBENCH_CLS_MODEL` to the GGUF path. Host and port default to `127.0.0.1:8080`
and can be set with the corresponding `_HOST` and `_PORT` variables. GLM context
is 65536, classifier context 4096, and each uses one slot by default. GLM sampling
is temperature 1, top-p 0.95, top-k 20 and min-p 0. Run the servers separately or
assign distinct ports
