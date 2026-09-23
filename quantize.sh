#!/usr/bin/env bash
# Build models/Qwen3.8-35B-A3B-Distill-NVFP4 from the BF16 download (see quantize.py).
set -euo pipefail
cd "$(dirname "$0")"
docker build -q -t vllm-qwen38-a3b:local .
docker build -q -t qwen38-quant:local -f Dockerfile.quant .
exec docker run --rm --gpus all --ipc host -v "$PWD":/work -w /work qwen38-quant:local \
  python3 quantize.py models/Qwen3.8-35B-A3B-Distill models/Qwen3.8-35B-A3B-Distill-NVFP4
