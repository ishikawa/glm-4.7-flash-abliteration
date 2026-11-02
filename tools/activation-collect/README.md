# Activation collector

Collect the last input token's attention output, FFN output, residual and final
normalized residual from GLM-4.7-Flash. The only profile is `glm47`: architecture
`deepseek2`, 47 main layers, hidden width 2048 and one residual stream. Layer
indices are zero based; only main layers 0 through 46 are collected.

## Build and test

Use CMake 3.20 or later, a C++17 compiler and a POSIX little-endian host. Both
cache variables below are required; omitting either produces a CMake fatal error.
Point them to matching llama.cpp sources and prebuilt shared libraries.

```sh
cmake -S tools/activation-collect -B build-activation-collect \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLAMA_CPP_ROOT="$LLAMA_CPP_DIR" \
  -DLLAMA_CPP_LIB_DIR="$LLAMA_CPP_DIR/build/bin"
cmake --build build-activation-collect -j 8
ctest --test-dir build-activation-collect --output-on-failure
```

The executable links `llama`, `llama-common`, `ggml` and `ggml-base`; the source
tree supplies common chat/Jinja headers, nlohmann JSON and SHA-256 code. CMake
records the source tree's Git HEAD in the executable. Build against the public
[fork commit ddf120c50](https://github.com/ishikawa/llama.cpp/commit/ddf120c50a8acfbf873c16cb070c2b5e4bb6a2fe).
Its `src/models/deepseek2.cpp` exposes the attention output before the final
layer's token crop with `cb(cur, "attn_output", il)`. This names an existing graph
node without changing the model computation. The collector also observes
`ffn_out-L`, `l_out-L` and `result_norm`.

`collect-core` uses synthetic memory buffers and temporary files. It links only
the collector core and SHA-256 implementation, loads no model, and initializes
no inference backend. It covers strided reads, final-layer cropping, finite-value
checks, duplicate/missing captures, JSONL validation, NPY I/O and resumable writes.

## Usage

```sh
build-activation-collect/activation-collect \
  --profile glm47 -m "$MODEL" \
  --input data/direction/harmful_fit.jsonl \
  --output _work/activations/harmful-fit \
  -ngl 99 -c 8192 -ub 2048 --threads 8
```

Repeat for benign fit and both held-out splits with separate output directories.
Collection loads a model and may use a GPU when `-ngl` is positive.

| Argument | Meaning | Default |
| --- | --- | --- |
| `-m`, `--model` | GGUF model path | Required |
| `--input` | Input JSONL path | Required except template dump |
| `--output` | Resumable output directory | Required for collection |
| `--profile` | Model profile | `glm47` |
| `-ngl`, `--n-gpu-layers` | GPU layer count | 99 |
| `-c`, `--ctx-size` | Context size | 8192 |
| `-ub`, `--ubatch-size` | Batch and microbatch size | 2048 |
| `-t`, `--threads` | CPU thread count | Hardware concurrency, minimum 1 |
| `--list-tensors` | Inspect layouts using the first input; no output directory needed | Off |
| `--dump-template` | Format dummy user text with a vocabulary-only model load | Off |
| `-h`, `--help` | Show usage and exit before loading a model | Off |

Each JSONL record needs a nonempty unique string `id` and nonempty `input`.
A `text` field takes precedence and is treated as an already formatted prompt;
otherwise the embedded Jinja chat template supplies a user message and generation
prompt. Template time is fixed to Unix epoch and thinking follows common library
defaults. Special tokens are parsed and BOS handling follows the vocabulary.

The collector clears model memory between inputs, processes microbatches, and
captures only the last microbatch. The final main layer's attention output still
contains all microbatch tokens; its FFN and residual outputs, and `result_norm`,
contain only the final token. Unexpected layouts fail rather than silently
changing which activation is collected.

## Output and resume

All arrays are C-order little-endian float32 NPY files. `N` is the total input
record count, including rows not yet committed during an interrupted collection.

| File | Shape | Content |
| --- | --- | --- |
| `attn_out.npy` | `[N, 47, 2048]` | Attention output |
| `ffn_out.npy` | `[N, 47, 2048]` | FFN output |
| `resid.npy` | `[N, 47, 1, 2048]` | Residual after each main layer |
| `result_norm.npy` | `[N, 2048]` | Final normalized residual |
| `manifest.json` | JSON | Configuration, input hashes and committed records |

The manifest records model identity, library commit, template hash and settings,
input identifiers and hashes, formatting mode, token counts and formatted-input
hashes. It contains no input text. Model identity includes the canonical local
path, size and modification time; these runtime manifests are local provenance
and should be sanitized before sharing.

An exclusive directory lock prevents concurrent writers. Arrays are synced before
an atomic manifest update. Only the manifest's `records` prefix is committed;
resume overwrites uncommitted rows. Configuration, input order, hashes, formatting
and token counts must match, and saved rows are checked for finite values.
