#!/usr/bin/env bash
# Fetch the BF16 source checkpoint (plain HTTP; Xet transfers stalled on this box).
cd "$(dirname "$0")"
export HF_HUB_DISABLE_XET=1
exec ~/.local/bin/uvx --from huggingface_hub hf download empero-ai/Qwen3.8-35B-A3B-Distill \
  --local-dir models/Qwen3.8-35B-A3B-Distill --max-workers 8
