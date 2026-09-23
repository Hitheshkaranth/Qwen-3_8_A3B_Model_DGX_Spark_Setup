"""Quantize empero-ai/Qwen3.8-35B-A3B-Distill for vLLM on the GB10 box.

Recipe mirrors nvidia/Qwen3.6-35B-A3B-NVFP4 (what ../vllm-qwen3.6 serves), but
data-free so it streams shard-by-shard and never needs the full BF16 model in
memory (the prod vLLM container holds most of the unified memory):

  routed experts + shared expert   NVFP4 weight-only (W4A16, group 16)
  full + linear attention projs    FP8 W8A8, dynamic per-token activations
  lm_head, embeddings, routers,
  conv1d/in_proj_a/b, norms, MTP,
  vision tower                     left in BF16

Run inside the vLLM image (see quantize.sh).
"""

import re
import sys

from compressed_tensors.quantization import QuantizationConfig, QuantizationScheme
from compressed_tensors.quantization.quant_scheme import FP8_DYNAMIC, NVFP4A16

from llmcompressor import model_free_ptq
from llmcompressor.entrypoints.model_free import microscale

SRC, DST = sys.argv[1], sys.argv[2]

# llm-compressor only pairs `mlp.gate_proj`/`mlp.up_proj` for a shared NVFP4
# global scale. Qwen3.6 names them `mlp.experts.N.*` and `mlp.shared_expert.*`,
# and vLLM keeps only gate's global scale for the fused w13, so unpaired
# gate/up would silently mis-scale every up_proj. Register both layouts.
microscale._DEFAULT_FUSED_MAPPINGS_LIST += [
    [r"re:.*mlp\.experts\.\d+\.gate_proj\.weight$",
     r"re:.*mlp\.experts\.\d+\.up_proj\.weight$"],
    [r"re:.*mlp\.shared_expert\.gate_proj\.weight$",
     r"re:.*mlp\.shared_expert\.up_proj\.weight$"],
]
# Shared-expert gate/up can land in different shards; the owning shard must
# fetch its partner. (Routed experts are one fused 3D tensor, so no partner.)
microscale.DEFAULT_FUSED_MAPPINGS[
    r"^(?P<prefix>.+?)\.(?P<mlp>mlp\.shared_expert)\.gate_proj\.weight$"
] = [r"{prefix}.{mlp}.up_proj.weight"]

# Unanchored: vLLM matches its own module names (model.layers.N...), not the
# HF ones, and passes re: targets through unmapped. Lookahead skips mtp.*.
LM = r"(?!mtp\.).*layers\.\d+"
config = QuantizationConfig(
    config_groups={
        "group_0": QuantizationScheme(
            targets=[
                rf"re:^{LM}\.mlp\.experts\.\d+\.(gate|up|down)_proj$",
                rf"re:^{LM}\.mlp\.shared_expert\.(gate|up|down)_proj$",
            ],
            **NVFP4A16,
        ),
        "group_1": QuantizationScheme(
            targets=[
                rf"re:^{LM}\.self_attn\.(q|k|v|o)_proj$",
                rf"re:^{LM}\.linear_attn\.(in_proj_qkv|in_proj_z|out_proj)$",
            ],
            **FP8_DYNAMIC,
        ),
    },
    ignore=[
        "lm_head",
        "re:.*embed_tokens.*",
        "re:.*mlp\\.gate$",
        "re:.*shared_expert_gate.*",
        "re:.*in_proj_(a|b)$",
        "re:.*conv1d.*",
        "re:.*norm.*",
        "re:^mtp\\..*",
        "re:.*visual.*",
    ],
)

model_free_ptq(
    model_stub=SRC,
    save_directory=DST,
    config=config,
    max_workers=2,
    device="cuda:0",
)

# Sanity check: every fused gate/up pair must share one global scale.
from safetensors import safe_open  # noqa: E402
import glob  # noqa: E402
import torch  # noqa: E402

scales = {}
for f in glob.glob(f"{DST}/*.safetensors"):
    with safe_open(f, "pt") as st:
        for k in st.keys():
            if k.endswith("weight_global_scale") and re.search(r"\.(gate|up)_proj\.", k):
                scales[k] = st.get_tensor(k)
bad = 0
for k, v in scales.items():
    if ".gate_proj." in k:
        u = scales.get(k.replace(".gate_proj.", ".up_proj."))
        if u is None or not torch.equal(u, v):
            bad += 1
print(f"gate/up global-scale pairs checked: {len(scales) // 2}, mismatched: {bad}")
sys.exit(1 if bad else 0)
