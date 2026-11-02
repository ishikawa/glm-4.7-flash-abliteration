#!/usr/bin/env bash
set -euo pipefail

# Foreground llama-server launcher for GLM-4.7-Flash, used to evaluate the original and edited GGUFs.
# The caller owns server lifecycle; this script does not stop another server.

LLAMA_CPP_ROOT="${LLAMA_CPP_DIR:-.}"
BIN="${GLM47_BIN:-$LLAMA_CPP_ROOT/build/bin/llama-server}"
MODEL="${GLM47_MODEL:?Set GLM47_MODEL to a GGUF path}"
SERVED_NAME="${GLM47_SERVED_NAME:-glm-4.7-flash}"
HOST="${GLM47_HOST:-127.0.0.1}"
PORT="${GLM47_PORT:-8080}"

# The evaluation generates up to 32K tokens after a short prompt, so 64K leaves room.
CTX="${GLM47_CONTEXT:-65536}"

[[ -x "$BIN" ]] || { printf 'llama-server not found: %s\n' "$BIN" >&2; exit 1; }
[[ -f "$MODEL" ]] || { printf 'GLM-4.7-Flash model not found: %s\n' "$MODEL" >&2; exit 1; }

printf 'glm47 config: binary=%s model=%s served=%s host=%s port=%s ctx=%s\n' \
    "$BIN" "$MODEL" "$SERVED_NAME" "$HOST" "$PORT" "$CTX" >&2

# Sampling follows the model card's default settings (temperature 1.0, top-p 0.95).
exec "$BIN" \
    -m "$MODEL" \
    -a "$SERVED_NAME" \
    -ngl 99 \
    -c "$CTX" \
    -np 1 \
    --spec-type none \
    --temp 1.0 \
    --top-p 0.95 \
    --top-k 20 \
    --min-p 0 \
    --jinja \
    --host "$HOST" \
    --port "$PORT" \
    "$@"
