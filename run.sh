#!/usr/bin/env bash
# vLLM server for empero-ai/Qwen3.8-35B-A3B-Distill, quantized locally by
# quantize.sh (NVFP4 W4A16 experts + FP8-dynamic attention, BF16 lm_head).
# Same architecture and flags as the nvidia/Qwen3.6-35B-A3B-NVFP4 production server.
# GB10 unified-memory box: keep --gpu-memory-utilization <= 0.70 or concurrency collapses,
# and don't run this alongside the Qwen3.6 container (it already holds 0.70).
set -euo pipefail

IMAGE="vllm-qwen38-a3b:local"   # NVIDIA vLLM 26.07 + xgrammar 0.2.4 tool-calling fix (see Dockerfile)
NAME="vllm-qwen38-a3b"
PORT="${PORT:-8002}"
# Host addresses to publish on. Default is all interfaces. On a box with the
# metering gateway, use BIND_ADDRS="127.0.0.1 172.17.0.1" so only the gateway
# (localhost) and Prometheus (docker bridge) can reach vLLM directly.
BIND_ADDRS="${BIND_ADDRS:-0.0.0.0}"
PUBLISH=()
for addr in $BIND_ADDRS; do PUBLISH+=(-p "$addr:$PORT:8000"); done
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_DIR="$SCRIPT_DIR/models/Qwen3.8-35B-A3B-Distill-NVFP4"

docker build -t "$IMAGE" "$SCRIPT_DIR"

docker rm -f "$NAME" 2>/dev/null || true

exec docker run -d \
  --name "$NAME" \
  --restart "${RESTART:-unless-stopped}" \
  --gpus all \
  --ipc host \
  "${PUBLISH[@]}" \
  -v "$MODEL_DIR":/models/qwen3.8-35b-a3b:ro \
  "$IMAGE" \
  python3 -m vllm.entrypoints.openai.api_server \
    --model /models/qwen3.8-35b-a3b \
    --served-model-name qwen3.8-35b-a3b \
    --gpu-memory-utilization 0.70 \
    --max-model-len 262144 \
    --max-num-seqs 16 \
    --max-num-batched-tokens 8192 \
    --enable-prefix-caching \
    --kv-cache-dtype fp8 \
    --moe-backend marlin \
    --language-model-only \
    --reasoning-parser qwen3 \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder \
    --host 0.0.0.0 \
    --port 8000
