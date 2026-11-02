#!/usr/bin/env bash
set -euo pipefail

# Foreground launcher. The caller owns server lifecycle.
LLAMA_CPP_ROOT="${LLAMA_CPP_DIR:-.}"
BIN="${HARMBENCH_CLS_BIN:-$LLAMA_CPP_ROOT/build/bin/llama-server}"
MODEL="${HARMBENCH_CLS_MODEL:?Set HARMBENCH_CLS_MODEL to a GGUF path}"
HOST="${HARMBENCH_CLS_HOST:-127.0.0.1}"
PORT="${HARMBENCH_CLS_PORT:-8080}"
CTX="${HARMBENCH_CLS_CONTEXT:-4096}"
SLOTS="${HARMBENCH_CLS_SLOTS:-1}"

[[ -x "$BIN" ]] || { printf 'llama-server not found: %s\n' "$BIN" >&2; exit 1; }
[[ -f "$MODEL" ]] || { printf 'HarmBench classifier model not found: %s\n' "$MODEL" >&2; exit 1; }

exec "$BIN" \
    -m "$MODEL" \
    --alias harmbench-cls \
    -ngl 99 \
    -c "$CTX" \
    -np "$SLOTS" \
    --host "$HOST" \
    --port "$PORT" \
    "$@"
