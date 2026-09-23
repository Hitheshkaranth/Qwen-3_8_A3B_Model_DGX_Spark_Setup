# Engineering Notes

This is the technical log behind this repository: what was built, what broke, how each problem
was diagnosed, and the raw data behind every number in the README. Everything here was captured
on one **NVIDIA DGX Spark (GB10)**, the same machine that runs the production
[Qwen3.6 deployment](https://github.com/Hitheshkaranth/Qwen-3_6_Model_DGX_Spark_Setup).

## 1. Hardware profile

```
$ nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
NVIDIA GB10, 580.173.02

$ free -g | head -2
               total        used        free      shared  buff/cache   available
Mem:             121          95          12           0          14          25
```

GB10 is a **single-GPU, unified-memory** system: the CPU and GPU share one 128 GB pool, of which
the OS sees about 121 GiB. Two consequences shaped this work:

- **Only one 0.70-utilization vLLM server fits at a time.** The production Qwen3.6 server holds
  about 85 GB, so testing this model meant stopping it (see §4).
- **Quantization had to stream.** With production running, only about 22–25 GB was free, so
  loading the 71 GB BF16 model for a classic `oneshot` calibration pass was not possible
  (see §3).

## 2. Which "Qwen3.8 A3B"?

Qwen has not released an official Qwen3.8 A3B model. On 2026-09-23 the Qwen organisation on
Hugging Face lists `Qwen3.8-27B` (dense), `Qwen3.8-Flash-Next` (~180B, 360 GB in BF16) and
`Qwen3.8-2.4T-A95B`. The only 35B-A3B variant is Empero's community distillation:

| Field | Value |
|---|---|
| Repo | [`empero-ai/Qwen3.8-35B-A3B-Distill`](https://huggingface.co/empero-ai/Qwen3.8-35B-A3B-Distill) (Apache-2.0) |
| Base | `Qwen/Qwen3.6-35B-A3B` (SFT / off-policy distillation) |
| Teachers | Qwen3.8-2.4T-A95B, Qwen3.8-Flash-Next |
| `architectures` | `Qwen3_5MoeForConditionalGeneration` (same as the Qwen3.6 production model) |
| Layers | 40 (30 Gated DeltaNet + 10 full attention) |
| Experts | 256 routed, 8 per token, plus 1 shared expert |
| `max_position_embeddings` | 262,144 |
| `mtp_num_hidden_layers` | 1 (MTP head present, kept in BF16, not used) |
| Checkpoint | 21 BF16 shards + `model-mtp.safetensors`, ~67 GiB |

Because the architecture matches the Qwen3.6 model, every serving flag from the production
deployment carries over unchanged.

## 3. Quantization

### 3.1 Recipe choice

NVIDIA's `nvidia/Qwen3.6-35B-A3B-NVFP4` (`hf_quant_config.json`) uses:

```
quant_algo: MIXED_PRECISION, kv_cache_quant_algo: FP8, exclude_modules: [mtp*]
  mlp.experts / mlp.shared_expert.*     W4A16_NVFP4 (group 16)
  self_attn.{q,k,v,o}_proj              FP8 (static input_scale)
  linear_attn.{in_proj_qkv,in_proj_z,out_proj}  FP8 (static input_scale)
  lm_head                               W4A16_NVFP4
```

Static FP8 activation scales need calibration, and calibration needs the whole model in memory.
Instead, this repo uses llm-compressor's `model_free_ptq`, which processes one shard at a time.
It keeps the same layout but changes two things:

| | NVIDIA | This repo | Consequence |
|---|---|---|---|
| Attention FP8 activations | static (calibrated) | **dynamic per-token** | No calibration data needed; usually as accurate or more |
| `lm_head` | NVFP4 | **BF16** | More accurate output layer, but slower decode (§5.2) |

Run stats (`quantize.log`):

```
Distributing 22 shard(s), estimated memory: 1.69-6.74 GB per shard, 127.54 GB total
Quantizing: 100%|██████████| 22/22 [19:15<00:00, 52.52s/it]
gate/up global-scale pairs checked: 10280, mismatched: 0
```

Output: 23 GB, `format: mixed-precision` (compressed-tensors), `group_0` NVFP4 (`nvfp4-pack-quantized`)
and `group_1` FP8 channel-wise with dynamic activations (`float-quantized`).

### 3.2 Pitfall 1: unpaired gate/up global scales

NVFP4 stores one FP32 *global* scale per tensor. vLLM fuses each expert's `gate_proj` and
`up_proj` into a single `w13` weight, and it can only use one global scale for it
(`compressed_tensors_moe_w4a4_nvfp4.py`):

```python
if self.moe.is_act_and_mul and not torch.allclose(
    layer.w13_weight_global_scale[:, 0], layer.w13_weight_global_scale[:, 1]):
    logger.warning_once("w1_weight_global_scale must match w3_weight_global_scale. "
                        "Accuracy may be affected.")
w13_weight_global_scale = layer.w13_weight_global_scale[:, 0].contiguous()
```

llm-compressor 0.14 computes a shared scale only for names that match
`re:.*mlp\.gate_proj\.weight$` / `re:.*mlp\.up_proj\.weight$`. This model's names are
`mlp.experts.N.gate_proj` and `mlp.shared_expert.gate_proj`, so none of them would be paired, and
vLLM would dequantize every `up_proj` with gate's scale. vLLM only logs a single warning about
it, which is easy to miss.

**Fix** (`quantize.py`): register both layouts in llm-compressor's fused-name tables before
running, and add a partner-fetch rule so a shared expert whose gate and up land in different
shards still gets paired. After writing the checkpoint, the script re-reads every
`weight_global_scale` and exits non-zero if any pair differs. Result: 10,280 pairs
(40 layers × 256 experts + 40 shared experts), 0 mismatched.

### 3.3 Pitfall 2: target regexes vLLM can't see

The first launch crash-looped:

```
ValueError: moe_backend='marlin' is not supported for unquantized MoE.
Expected one of ['triton', 'flashinfer_trtllm', 'flashinfer_cutlass', 'aiter'].
```

vLLM decided the MoE layers were *unquantized*. It resolves a MoE layer's scheme by building
`<vllm module name>.0.gate_proj` and matching it against the config's `targets`:

```python
unfused_names = [layer_name + proj_name
                 for proj_name in [".0.gate_proj", ".0.up_proj", ".0.down_proj"]]
```

Two details combine here:

1. vLLM's module names are `model.layers.N.mlp.experts`, not the HF
   `model.language_model.layers.N...` (`hf_to_vllm_mapper`: `"model.language_model." -> "model."`).
2. `CompressedTensorsConfig.apply_vllm_mapper` rewrites plain layer paths, but explicitly passes
   `re:` targets through unchanged.

So targets anchored as `re:^model\.language_model\.layers\.\d+\...` never matched. **Fix:**
targets are written as `re:^(?!mtp\.).*layers\.\d+\.<module>$`, which matches both naming styles
and still excludes the BF16 MTP head. Checked against sample names:

```
model.layers.3.mlp.experts.0.gate_proj                    -> group_0 (NVFP4)
language_model.model.layers.3.mlp.experts.0.up_proj       -> group_0
model.language_model.layers.3.mlp.shared_expert.down_proj -> group_0
model.layers.3.self_attn.q_proj                           -> group_1 (FP8)
model.layers.0.linear_attn.in_proj_qkv                    -> group_1
mtp.layers.0.mlp.experts.0.gate_proj                      -> (none)
model.layers.3.mlp.gate                                   -> (none)  router stays BF16
model.layers.0.linear_attn.in_proj_a                      -> (none)
```

After the fix, vLLM selects the intended kernels:

```
Selected CutlassFP8ScaledMMLinearKernel for CompressedTensorsW8A8Fp8
Using MarlinNvFp4LinearKernel for NVFP4 GEMM
Using 'MARLIN' NvFp4 MoE backend
Model loading took 20.23 GiB memory and 130.952721 seconds
```

The served checkpoint had its `config.json` patched in place. The committed `quantize.py` writes
the same targets directly, so a fresh `./quantize.sh` produces the fixed config.

### 3.4 Smaller issues

- **Hugging Face Xet transfers stalled** at 0 bytes (the process stayed alive with no I/O).
  `download.sh` sets `HF_HUB_DISABLE_XET=1`. Plain HTTP ran at 10–25 MB/s with occasional
  `read operation timed out` retries, which resume automatically; the download took about
  50 minutes.
- **The quantizer's output is owned by root** because it runs as root inside the container.
  `install.sh` chowns it back.
- **Build context:** `run.sh` builds from the repo root, which contains `models/` (~90 GB).
  `.dockerignore` excludes it.
- **Tool calling on 26.07:** the same `xgrammar==0.2.4` patch as the Qwen3.6 repo (see
  `Dockerfile`).

## 4. Test procedure

The production Qwen3.6 container was stopped with `docker stop vllm` (not removed), so
`docker start vllm` restores it exactly. The 16-user throughput baseline for Qwen3.6 was measured
immediately before it was stopped.

| Step | Duration |
|---|---|
| Qwen3.8 cold start (`docker start` → `/v1/models` ready) | 258 s |
| Qwen3.8 weight load | 131 s |
| Qwen3.6 restore (`docker start vllm` → ready) | about 6 min |

## 5. Benchmark methodology

### 5.1 Throughput (`benchmarks/throughput_bench.py`)

- Every request generates exactly 512 tokens (`ignore_eos: true, min_tokens: 512`), so both models
  do identical work regardless of when they would naturally stop.
- One warm-up request, then 1 user, then 16 users at once (the `--max-num-seqs` limit).
- Aggregate = total tokens ÷ wall time. Per-user = mean of each request's tokens ÷ its latency.
- Only one model was loaded when each was measured.

### 5.2 Why this build decodes more slowly

At 1 user, decoding is bound by memory bandwidth: each token has to read the active weights.
This build keeps `lm_head` in BF16: 248,320 vocab × 2,048 hidden × 2 bytes ≈ **1.0 GB read per
token**, compared with about 0.26 GB for NVIDIA's NVFP4 `lm_head`. That's the likely main cause of
the 25% single-user gap, but it hasn't been isolated. The gap shrinks to 11% at 16 users,
because one `lm_head` read is shared by the whole batch.

### 5.3 Reasoning and output length (`benchmarks/reasoning_bench.py`)

- **GSM8K**: the first 100 test problems. **MATH-500**: the first 100 problems whose answer is
  an integer, so they can be scored exactly.
- The prompt asks the model to finish with `ANSWER: <number>`. The extractor falls back to
  `\boxed{}` and then to the last number.
- Sampling as recommended on the model card: `temperature=0.6, top_p=0.95, top_k=20`,
  `max_tokens=16384`, 16 in flight.
- **Long-form**: 4 open-ended "as long as possible" prompts, capped at 32,768 tokens, measuring
  how many tokens the model writes before stopping on its own. The Empero card warns the distill
  was trained on examples of at most 8,192 tokens.

## 6. Monitoring

The existing Prometheus/Grafana stack scrapes `host.docker.internal:8002` in the `vllm` job, and
the dashboard separates servers by the `model_name` label (`qwen3.8-35b-a3b`).

`prometheus.yml` is bind-mounted as a single file. Editing it with `sed -i` writes a new inode,
so the container keeps seeing the old file, and neither SIGHUP nor `/-/reload` picks up the
change. `docker restart vllm-prometheus` does.

## 7. Known limitations / open items

- **Quantize `lm_head` to NVFP4** to recover most of the single-user speed gap. It's a one-line
  change in `quantize.py` (move `lm_head` from `ignore` to a group_0 target); it needs a
  before/after accuracy check.
- **MTP speculative decoding** is not enabled. The distill's MTP head was probably not updated
  during SFT, so draft acceptance may be low. This hasn't been measured.
- **Static-activation FP8** (NVIDIA's approach) was not tried, because it needs a calibration
  pass with the full model in memory.
- The Empero card notes that a v2 with longer training examples is in progress.
