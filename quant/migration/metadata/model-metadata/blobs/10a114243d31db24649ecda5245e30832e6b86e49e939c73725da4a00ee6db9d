---
pipeline_tag: image-text-to-text
base_model:
- Qwen/Qwen3.8-27B
license: apache-2.0
library_name: Model Optimizer
tags:
- RadixArk
- ModelOpt
- Qwen3.8
- quantized
- FP4
- fp4
- NVFP4
---

# Model Overview

## Description:

The RadixArk Qwen3.8-27B-NVFP4-BF16-LMHead model is a variant of [RadixArk/Qwen3.8-27B-NVFP4](https://huggingface.co/RadixArk/Qwen3.8-27B-NVFP4), the quantized version of [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) produced at RadixArk using [NVIDIA Model Optimizer](https://github.com/NVIDIA/Model-Optimizer) with a mixed NVFP4 W4A4 recipe. The only difference from the source checkpoint is that `lm_head` is **not** quantized: it uses the original BF16 `lm_head` weights from Qwen/Qwen3.8-27B instead of the NVFP4 W4A4 quantized ones. All other tensors are identical to the source checkpoint.

**Run on [SGLang](https://github.com/sgl-project/sglang)**: launch command and per-platform recipes in the [Qwen3.8-27B cookbook](https://cookbook.sglang.ai/autoregressive/Qwen/Qwen3.8-27B).

## Third-Party Community Consideration

This model is not owned or developed by RadixArk. It is a quantized derivative of Qwen's model; see the upstream [Qwen3.8-27B model card](https://huggingface.co/Qwen/Qwen3.8-27B) for the source model's capabilities, training information, limitations, and license.

### License/Terms of Use:

[Apache License 2.0](./LICENSE)

### Deployment Geography:

Global <br>

### Use Case: <br>

Developers looking to deploy an off-the-shelf, pre-quantized model in AI agent systems, chatbots, RAG systems, and other AI-powered applications. <br>

### Release Date: <br>

Hugging Face 08/14/2026 via https://huggingface.co/RadixArk/Qwen3.8-27B-NVFP4 <br>

## Model Architecture:

**Architecture Type:** Transformers (Dense Multimodal) <br>
**Network Architecture:** Qwen3.8-27B <br>
**Number of Model Parameters:** 27B <br>

## Input:

**Input Type(s):** Text, image, and video <br>
**Input Format(s):** String and visual media <br>
**Other Properties Related to Input:** Native context length up to 262,144 tokens. <br>

## Output:

**Output Type(s):** Text <br>
**Output Format:** String <br>

## Software Integration:

**Supported Runtime Engine(s):** <br>
* SGLang <br>

**Supported Hardware Microarchitecture Compatibility:** <br>
* NVIDIA Blackwell (this checkpoint was produced and validated on GB300) <br>

**Preferred Operating System(s):** <br>
* Linux <br>

## Model Version(s):

Quantized with [NVIDIA Model Optimizer](https://github.com/NVIDIA/Model-Optimizer), commit `87c9f8cf83021957d1a1a575c90c9a4eaaf7ef0c`. <br>

## Training, Testing, and Evaluation Datasets:

### Calibration Data:

Calibration used 1,024 samples from the `abisee/cnn_dailymail` training split with sequence length 512. <br>

### Training Dataset:

RadixArk did not train or fine-tune this checkpoint. Training information is inherited from the upstream [Qwen3.8-27B model card](https://huggingface.co/Qwen/Qwen3.8-27B). <br>

### Evaluation Dataset:

The model was evaluated on GSM8K and Terminal-Bench 2.1. <br>

## Post Training Quantization

The MLP `gate_proj`, `up_proj`, and `down_proj` layers use dynamic NVFP4 W4A4 quantization with group size 16. Attention weights use FP8, while `lm_head`, MTP and vision tensors retain the source BF16 precision.

## Usage

The following SGLang configuration uses four NVIDIA Blackwell GPUs:

```sh
sglang serve \
  --trust-remote-code \
  --model-path RadixArk/Qwen3.8-27B-NVFP4-BF16-LMHead \
  --tp-size 4 \
  --mem-fraction-static 0.75 \
  --speculative-algorithm NEXTN \
  --speculative-num-steps 3 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 4 \
  --reasoning-parser qwen3 \
  --tool-call-parser qwen3_coder \
  --host 0.0.0.0 \
  --port 30000
```

For other deployment topologies and hardware-specific configurations, see the [SGLang Qwen3.8-27B cookbook](https://docs.sglang.io/cookbook/autoregressive/Qwen/Qwen3.8-27B).

### Evaluation

The results below were measured on this checkpoint (not inherited from the source NVFP4 checkpoint) on 4x NVIDIA GB300 GPUs with a TP4 SGLang deployment (SGLang main @ `d287880a`, NEXTN 3/1/4, seed 0).

| Benchmark | Evaluation protocol | Score |
|---|---|---:|
| GSM8K | Full 1,319-example split, thinking mode, model-default sampling, sgl-eval | **96.13% (1,268/1,319)** |
| Terminal-Bench 2.1 | Claude Code 2.1.228, DFlash2 drafter, pass@1 | **69.4% (247/356, rep 4)** / 68.18% (60/88, rep 1) |

GSM8K stop rate was 100% with zero truncations and zero errors. For reference, the source NVFP4 checkpoint (quantized `lm_head`) scored 96.36% (1,271/1,319) on the same protocol; the delta is within single-run sampling noise at temperature 1.0. The reported evaluations were text-only.

## Model Limitations:

The base model may generate inaccurate, incomplete, irrelevant, biased, or otherwise undesirable responses. Developers should evaluate the model for their intended use case and apply appropriate safeguards.

## Ethical Considerations

RadixArk believes trustworthy AI is a shared responsibility. Developers should ensure that use of this model complies with the upstream license and meets the safety, privacy, and reliability requirements of their application.
