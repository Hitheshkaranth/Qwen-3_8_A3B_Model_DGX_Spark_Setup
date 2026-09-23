#!/usr/bin/env bash
# One-line installer for Qwen3.8-35B-A3B (empero-ai distill, NVFP4/FP8) on a
# DGX Spark (GB10) or any single Blackwell GPU box.
#
#   curl -fsSL https://raw.githubusercontent.com/Hitheshkaranth/Qwen-3_8_A3B_Model_DGX_Spark_Setup/main/install.sh | bash
#
# Clones this repo (or updates it if already present in the current
# directory), downloads the BF16 weights, quantizes them, and starts the
# server via run.sh. Every step skips work that's already done, so it's safe
# to re-run.
set -euo pipefail

REPO_URL="${QWEN38_REPO_URL:-https://github.com/Hitheshkaranth/Qwen-3_8_A3B_Model_DGX_Spark_Setup.git}"
DIR="Qwen-3_8_A3B_Model_DGX_Spark_Setup"

echo "==> Checking prerequisites..."

if ! command -v git >/dev/null 2>&1; then
  echo "git is required. Install it first (e.g. 'sudo apt-get install -y git')." >&2
  exit 1
fi

if ! command -v docker >/dev/null 2>&1; then
  echo "Docker is required. Install it first: https://docs.docker.com/engine/install/" >&2
  exit 1
fi

if ! docker info >/dev/null 2>&1; then
  echo "Docker daemon isn't reachable (is it running, and do you have permission to use it?)." >&2
  exit 1
fi

if ! docker run --rm --gpus all nvidia/cuda:12.6.0-base-ubuntu22.04 nvidia-smi >/dev/null 2>&1; then
  echo "GPU not visible to Docker. Install the NVIDIA Container Toolkit, then re-run this script:" >&2
  echo "  https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html" >&2
  exit 1
fi

if ! command -v uvx >/dev/null 2>&1 && [ ! -x "$HOME/.local/bin/uvx" ]; then
  echo "==> Installing uv (used to run the Hugging Face CLI)..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi

echo "==> Prerequisites OK."

if [ -d "$DIR/.git" ]; then
  echo "==> $DIR already exists here, pulling latest..."
  git -C "$DIR" pull --ff-only
else
  echo "==> Cloning $REPO_URL..."
  git clone "$REPO_URL" "$DIR"
fi

cd "$DIR"

echo "==> Downloading the BF16 checkpoint (~71GB, resumes if interrupted)..."
./download.sh

if [ -f models/Qwen3.8-35B-A3B-Distill-NVFP4/model.safetensors.index.json ]; then
  echo "==> Quantized checkpoint already present, skipping quantization."
else
  echo "==> Quantizing to NVFP4 experts + FP8 attention (~20 minutes)..."
  ./quantize.sh
  # The quantizer runs as root inside its container; hand the output back.
  docker run --rm -v "$PWD/models":/m --entrypoint chown qwen38-quant:local -R "$(id -u):$(id -g)" /m
fi

echo "==> Building image and starting the server..."
./run.sh

echo
echo "==> Done. Tail logs with: docker logs -f vllm-qwen38-a3b"
echo "==> Server will be reachable at http://localhost:8002/v1 once startup completes (4-5 minutes)."
