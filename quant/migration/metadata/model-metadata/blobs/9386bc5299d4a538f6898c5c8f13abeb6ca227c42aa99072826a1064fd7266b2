---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
library_name: vllm
pipeline_tag: image-text-to-text
tags:
  - prismaquant
  - prismaaqua
  - aqua-aura
  - compressed-tensors
  - nvfp4
  - fp8
  - mixed-precision
  - quantized
  - blackwell
  - dgx-spark
  - mtp
  - speculative-decoding
  - qwen3.8
  - vllm
---

# Qwen3.8-27B — PrismaQuant AQUA-AURA, 5.5 bpp

A mixed-precision quantization of [`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B)
in which **every Linear was priced individually and given the format it earned**,
rather than one format being applied to the whole network. It ships as a stock
`compressed-tensors` checkpoint: **vanilla vLLM serves it with no forked runtime
and no custom kernels.**

**21.98 GiB on disk** (23,603,517,768 bytes), **5.5002 bits per parameter** over
the 24.35 B quantizable body parameters.

Method, sensitivity data and the rest of the artifact family:
**[prismaquant.org](https://prismaquant.org)**

## Measured

Every number below is in `shipcard.json` with its tool, git commit, serving-stack
fingerprint and corpus hash. All five gates PASS and the card verifies.

| measurement | value |
|---|---|
| KL vs BF16, all positions (n=8 × 512 → 4088 positions) | **0.0338** |
| KL vs BF16, confident positions (teacher top-1 > 0.5; 2067 of them) | **0.0187** |
| KL p99 / max | 0.2917 / 1.3292 |
| WikiText-2 **test** PPL (8176 tokens @ 512) | **9.580** (mean NLL 2.2597) |
| — same measurement on the **BF16 source** | 9.361 (mean NLL 2.2366) |
| — **quantization cost** | **+2.34% PPL, +0.0231 nats/token** |
| vLLM eager load + generate | PASS |
| vLLM CUDA-graph load + generate | PASS |
| Ship gate (PPL / p99 per-prompt NLL / coherence) | PASS (4.030 / 1.947) |

**How to read the KL, precisely.** The teacher is the BF16 source scored on the
same corpus bytes. Each position keeps the teacher's top-1024 logprobs plus a
single lumped bucket for the residual mass (coverage: mean 0.988, min 0.573).
Lumping outcomes can only *reduce* KL, so by the data-processing inequality this
is a **lower bound** on true full-vocab KL — it understates damage, never
overstates it. Calibration windows come from WikiText **train**; PPL scores
WikiText **test**, so the two are disjoint.

**The PPL pair is a fair pair.** Both arms scored the identical corpus bytes
(same `corpus_sha256`), the same 8176-token prefix, with the same tool in the
same container. Their `serve_fingerprint` values differ, and 15 of the 16
differing manifest keys are incidental (timestamps, PIDs, container hostname,
ephemeral ports, the model path — which must differ). The one substantive
difference is that the quantized arm has `fp4_gemm_cutlass_sm120.so` and
`trtllm_utils.so` resident and the BF16 arm does not. That is not a confound:
loading the CUTLASS FP4 kernel *is* what serving this artifact means, so the
delta measures what you actually get.

**What is not established here.** p99 KL is ~9× the mean and max is ~39× it. A
low mean KL can hide a heavy tail, which is why the tail statistics are
published rather than reduced away, and why no claim is made about behaviour at
the worst positions. No downstream task suite was run by the pipeline; a
community ToolEvalBench result is quoted under Limitations.

## What the allocator chose

![Per-Linear format allocation across all 64 layers](allocation-map.png)

Each column is a layer, each row a projection, each cell one Linear:
<span title="NVFP4">**violet = NVFP4**</span> (4-bit, W4A4, group 16),
**teal = FP8 E4M3** (8-bit, per-channel), **orange = BF16** (16-bit).
Blank cells are projections that layer does not have — only 16 of the 64 layers
use full attention; the rest use linear attention.

Read the two adjacent rows in the middle: `linear_attn.in_proj_qkv` is nearly
solid violet while `linear_attn.out_proj`, directly below it in the *same
layers*, is nearly solid teal. The input projection earns 4 bits and the output
projection earns 8. Nothing about the layers' shapes predicts that split — it
came out of measuring each Linear's contribution to end-to-end KL.

This map is generated from the **served bytes** (the checkpoint's own
`config.json` and safetensors headers), not from the pipeline's recipe file, so
it shows what vLLM actually loads.

Formats are not spread uniformly, and that is the entire point. Of 496 body
Linears:

| format | count | where it landed |
|---|---|---|
| NVFP4 (W4A4, group 16) | 247 | mostly the MLPs — where the parameters are |
| FP8 E4M3 (channel) | 223 | mostly attention output and the full-attention layers |
| BF16 (passthrough) | 26 | only the tiny `linear_attn` gate projections `in_proj_a`/`in_proj_b` |

Per role, from the shipped recipe:

| role | n | NVFP4 | FP8 | BF16 |
|---|---|---|---|---|
| `mlp.gate_proj` / `mlp.up_proj` | 64 each | 54 | 10 | — |
| `mlp.down_proj` | 64 | 48 | 16 | — |
| `linear_attn.in_proj_qkv` / `in_proj_z` | 48 each | 34 | 14 | — |
| `linear_attn.out_proj` | 48 | 11 | 37 | — |
| `linear_attn.in_proj_a` / `in_proj_b` | 48 each | — | 35 | 13 |
| `self_attn.{q,k,v,o}_proj` | 16 each | 3 | 13 | — |

Read the `out_proj` row against the `in_proj_qkv` row: the same layer's input
projection takes 4 bits while its output projection takes 8. No uniform-format
method can express that, and nothing about the layer's *shape* predicts it — it
came out of measurement.

**Held at full precision (not quantized):** `lm_head`, `embed_tokens`, the entire
visual tower, the multi-token-prediction heads, and all norms and routers. This
is a deliberate, disclosed choice, not an oversight — and it is large enough to
be worth drawing:

![Disk footprint split into allocated and immutable bytes](byte-budget.png)

**29.1% of the file was never the allocator's to decide.** The hatched bytes are
pinned at source precision and never enter the knapsack, so the 5.5 bpp label —
which by convention covers quantizable body parameters only — describes the
solid 70.9%. This is exactly why cross-artifact comparisons here must use **disk
bytes** and not the bpp label: two checkpoints can carry the same bpp number and
differ by gigabytes in what that number excludes.

## How it was built

**AURA** prices each `(Linear, format)` pair by a KL-adjoint probe multiplied
against the **production-rendered** weight delta — the same bytes the exporter
will actually ship, so the surrogate, the validation and the artifact are one
rendering. A multi-choice knapsack DP then allocates the byte budget across the
menu `{NVFP4, FP8_E4M3, BF16}`.

**AQUA** adds the activation side. NVFP4 is W4A4: it quantizes activations too,
and a weight-only cost model is structurally blind to that — NVFP4 and NVFP4A16
render weights *bit-identically* (max |ΔW| difference 0.0), so a weight-only DP
is provably indifferent between a format that quantizes activations and one that
does not, while their activations differ by 9.42% RMS. AQUA prices that term
from the same probe, with an exact folded-normal tail integral for block-scaled
formats.

The render recipe is GPTQ (fixed damping 1.0) + static activation ordering +
joint scale optimization (JSO), which chooses NVFP4 block scales *inside* the
GPTQ loop under an activation-weighted MSE.

## Serving

Vanilla vLLM, no plugin, no custom kernels. This is the full command, including
the shipped multi-token-prediction (MTP) head for speculative decoding:

```bash
vllm serve rdtand/Qwen3.8-27B-PrismaAQUA-5.5bit-vllm \
  --host 0.0.0.0 --port 8000 \
  --quantization compressed-tensors \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.85 \
  --kv-cache-dtype fp8 \
  --enable-prefix-caching \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_xml \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}'
```

- `--speculative-config` turns on the MTP head (`mtp.*` tensors, BF16) with
  three draft tokens per step. Drop the line to serve without speculation.
  The pipeline's ship gate scores perplexity on a no-speculation serve and
  recorded no draft statistics for this artifact, so **no acceptance rate is
  claimed here**; users report the head working under vLLM 0.27 with this
  command.
- `--reasoning-parser qwen3` keeps `<think>` blocks out of `content`;
  `--tool-call-parser qwen3_xml` matches the Qwen3.8 tool-call format
  (`qwen3_coder` also parses it).
- For image inputs add `--limit-mm-per-prompt '{"image":4}'` (or your own
  limit); the visual tower ships in BF16.

**Memory.** At 21.98 GiB of weights this wants a **32 GB card** (5090-class)
or larger. On a 32 GB RTX PRO 4500 a user reports ~140k tokens of context with
fp8 KV at `--gpu-memory-utilization 0.95` under vLLM 0.27.1; raise
`--max-model-len` as your card allows. Verified on GB10 / DGX Spark (Blackwell
sm_121) under vLLM 0.26 with the CUTLASS NVFP4 and FP8 paths (eager and
CUDA-graph modes both pass; see `shipcard.json`).

## Limitations — read these

- **`lm_head` and `embed_tokens` are BF16 and account for 4.74 GiB, 21% of the
  artifact.** They are excluded from the bit-per-parameter figure, which follows
  this project's convention of reporting bpp over *quantizable* parameters only.
  Compare against other checkpoints on **disk bytes**, not on the bpp label.
  Measurement on this model shows `lm_head`'s sensitivity is extraordinarily
  row-concentrated (Gini 0.996; 99.3% of it in the top 1% of vocabulary rows),
  which means a uniform format is the wrong instrument for it in either
  direction. A per-row treatment is future work, not a claim made here.
- **KL is a screening metric.** A lower mean KL can hide a heavier tail. The
  per-sequence tail statistics are recorded in the shipcard rather than reduced
  to a single number.
- **No downstream task suite was run by the pipeline** (GSM8K, IFEval, MMLU,
  ToolEvalBench), so nothing here is a pipeline claim about task accuracy.
  Community measurement: on a single DGX Spark (downclocked to 2 GHz), vLLM
  0.27.2rc1 with MTP at 5 draft tokens and a tuned thinking rate, this artifact
  scored **93/100 on tool-eval-bench v2.5.1** (129/138 points; 61 passed,
  7 partial, 1 failed), with 2,311 prefill tok/s and 20.0 → 66.4 decode tok/s
  from 1 to 4 concurrent requests —
  [report by @m0l0, 2026-08-15](https://forums.developer.nvidia.com/t/qwen3-8-27b-upcoming-release-countdown/380012/44),
  sparkrun recipe included. Tool-call fidelity is sensitive to small
  probability shifts at decision points.
- **MTP draft acceptance is unmeasured** by this pipeline (see Serving).
- **Calibration is WikiText-derived**, and the held-out split used for selection
  is disjoint from the split the cost surrogates saw. Behaviour on distant
  domains is unmeasured.
- **The visual tower is unquantized**, so multimodal inputs run at full
  precision and get no size benefit.

## Provenance

Every gate this artifact passed — and the exact tool, threshold, git commit and
model hash behind each — is in `shipcard.json` in this repository. It is a
refusal contract: `tools/publish_artifact.py` will not upload an artifact whose
card does not verify. If a number in this card is not also in that file, treat
this card as the stale one.

- Project: <https://prismaquant.org>
- Code: <https://github.com/RobTand/prismaquant>
- Source: `Qwen/Qwen3.8-27B` @ `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`
- Method: PrismaQuant AURA + AQUA (`COST_MODE=aura`)
- Target profile: `vllm_packed_moe`
- Container: `compressed-tensors` mixed-precision (`nvfp4-pack-quantized` +
  `float-quantized`)

## Citation

```bibtex
@software{tand_prismaquant,
  author  = {Tand, Robert},
  title   = {PrismaQuant: production-faithful KL--Fisher bit allocation},
  url     = {https://prismaquant.org}
}
```

Contact: robert.tand@icloud.com
