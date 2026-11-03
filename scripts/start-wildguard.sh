#!/usr/bin/env bash
set -euo pipefail

# Foreground launcher. The caller owns server lifecycle.
LLAMA_CPP_ROOT="${LLAMA_CPP_DIR:-.}"
BIN="${WILDGUARD_BIN:-$LLAMA_CPP_ROOT/build/bin/llama-server}"
MODEL="${WILDGUARD_MODEL:?Set WILDGUARD_MODEL to a GGUF path}"
HOST="${WILDGUARD_HOST:-127.0.0.1}"
PORT="${WILDGUARD_PORT:-8080}"
CTX="${WILDGUARD_CONTEXT:-16384}"
SLOTS="${WILDGUARD_SLOTS:-1}"

[[ -x "$BIN" ]] || { printf 'llama-server not found: %s\n' "$BIN" >&2; exit 1; }
[[ -f "$MODEL" ]] || { printf 'WildGuard model not found: %s\n' "$MODEL" >&2; exit 1; }

exec "$BIN" \
    -m "$MODEL" \
    --alias wildguard \
    -ngl 99 \
    -c "$CTX" \
    -np "$SLOTS" \
    --host "$HOST" \
    --port "$PORT" \
    "$@"
