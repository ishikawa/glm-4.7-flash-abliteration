#!/usr/bin/env bash
set -euo pipefail

# allenai/wildguard is gated: accept its terms on the Hub before running.
LLAMA_CPP_ROOT="${LLAMA_CPP_DIR:?Set LLAMA_CPP_DIR to a llama.cpp checkout}"
REVISION="${WILDGUARD_REVISION:-cbba4823f3e8020e5a74a5e29bf85072def6f2ff}"
OUT="${WILDGUARD_MODEL:?Set WILDGUARD_MODEL to the output GGUF path}"
PY=(uv run --no-project --python 3.12
    --with torch==2.11.0 --with 'numpy~=2.2.6' --with 'sentencepiece>=0.1.98,<0.3.0'
    --with transformers==4.57.6 --with 'protobuf>=4.21.0,<5.0.0' --with safetensors
    --with pyyaml --with tqdm python)

[[ ! -e "$OUT" ]] || { printf 'output exists: %s\n' "$OUT" >&2; exit 1; }
SNAPSHOT="$(hf download allenai/wildguard --revision "$REVISION" \
    --include '*.safetensors' --include '*.json' --include tokenizer.model --quiet)"
mkdir -p "$(dirname "$OUT")"
"${PY[@]}" "$LLAMA_CPP_ROOT/convert_hf_to_gguf.py" "$SNAPSHOT" --outtype q8_0 --outfile "$OUT"
# The HF tokenizer (legacy=False) adds no space prefix after special tokens;
# llama.cpp's SPM default does, which shifts the official prompt's tokens.
# gguf_set_metadata parses booleans with np.bool_, so only '' means false.
"${PY[@]}" "$LLAMA_CPP_ROOT/gguf-py/gguf/scripts/gguf_set_metadata.py" --force \
    "$OUT" tokenizer.ggml.add_space_prefix ''
