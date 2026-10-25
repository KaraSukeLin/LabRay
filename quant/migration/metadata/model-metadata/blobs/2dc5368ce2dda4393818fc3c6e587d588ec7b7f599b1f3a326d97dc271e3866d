---
pipeline_tag: text-generation
base_model:
- Qwen/Qwen3.8-27B
license: apache-2.0
library_name: Model Optimizer
tags:
- nvidia
- ModelOpt
- Qwen3.8
- quantized
- FP4
- fp4
- FP8
- fp8
---

# Model Overview

## Description:
The NVIDIA Qwen3.8-27B NVFP4 model is a quantized version of Alibaba's Qwen3.8-27B model, which is an auto-regressive language model that uses an optimized transformer architecture. For more information on the model, please check [here](https://huggingface.co/Qwen/Qwen3.8-27B). The model is quantized with [Model Optimizer](https://github.com/NVIDIA/Model-Optimizer).

This model is ready for commercial or non-commercial use.  <br>

## Third-Party Community Consideration
This model is not owned or developed by NVIDIA. This model has been developed and built to a third-party's requirements for this application and use case; see link to Non-NVIDIA [(Qwen3.8-27B) Model Card](https://huggingface.co/Qwen/Qwen3.8-27B) from Qwen.

### License/Terms of Use:
**GOVERNING DOWNLOAD TERMS:** Use of the model is governed by the [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0)

### Deployment Geography:
Global <br>

### Use Case: <br>
Developers looking to take off-the-shelf, pre-quantized models for deployment in AI Agent systems, chatbots, RAG systems, and other AI-powered applications. <br>

### Release Date:  <br>
Hugging Face 09/08/2026 via https://huggingface.co/nvidia/Qwen3.8-27B-NVFP4 <br>

## References
NVIDIA Model Optimizer: https://github.com/NVIDIA/Model-Optimizer

## Model Architecture:
**Architecture Type:** Transformer  <br>
**Network Architecture:** Qwen3.8-27B (`Qwen3_5ForConditionalGeneration`) <br>
**Number of Model Parameters:** 27B <br>

## Input:
**Input Type(s):** Text, Image, Video <br>
**Input Format(s):** String, Red, Green, Blue (RGB), Video (MP4/WebM) <br>
**Input Parameters:** One-Dimensional (1D), Two-Dimensional (2D), Three-Dimensional (3D) <br>
**Other Properties Related to Input:** Context length up to 262K <br>

## Output:
**Output Type(s):** Text <br>
**Output Format:** String <br>
**Output Parameters:** One-Dimensional (1D): Sequences <br>
**Other Properties Related to Output:** None <br>

Our AI models are designed and/or optimized to run on NVIDIA GPU-accelerated systems. By leveraging NVIDIA's hardware (e.g. GPU cores) and software frameworks (e.g., CUDA libraries), the model achieves faster training and inference times compared to CPU-only solutions. <br>

## Software Integration:
**Supported Runtime Engine(s):** <br>
* **vLLM** <br>
* **SGLang** <br>

**Supported Hardware Microarchitecture Compatibility:** <br>
* NVIDIA Blackwell <br>

**Preferred Operating System(s):** <br>
* Linux <br>

The integration of foundation and fine-tuned models into AI systems requires additional testing using use-case-specific data to ensure safe and effective deployment. Following the V-model methodology, iterative testing and validation at both unit and system levels are essential to mitigate risks, meet technical and functional requirements, and ensure compliance with safety and ethical standards before deployment.

## Model Version(s):
This checkpoint uses mixed NVFP4/FP8 quantization and was produced with nvidia-modelopt **v0.48.0**. <br>

## Training and Evaluation Datasets:

## Calibration Dataset:
**Link:** [Nemotron-Post-Training-Dataset-v3](https://huggingface.co/collections/nvidia/nemotron-post-training-v3) <br>
**Data Modality:** Text <br>
**Data Collection Method by dataset:** Varies by dataset. <br>
**Labeling Method by dataset:** Varies by dataset. <br>
**Properties:** The Nemotron-Post-Training-Dataset-v3 is a multi-million-sample corpus developed by NVIDIA for Supervised Fine-Tuning (SFT) and Reinforcement Learning (RL) to power alignment, reasoning, and agentic capabilities in the Nemotron-3 model family. <br>

## Training Dataset:
**Data Modality:** Undisclosed <br>
**Data Collection Method by dataset:** Undisclosed <br>
**Labeling Method by dataset:** Undisclosed <br>
**Properties:** Undisclosed<br>
**Image Training Data Size:** Undisclosed<br>
**Text Training Data Size:** Undisclosed<br>
**Video Training Data Size:** Undisclosed<br>


## Evaluation Dataset:
**Datasets:** GPQA Diamond, Terminal-Bench, AA-LCR, MMMU-Pro, SciCode, IFBench <br>
**Data Collection Method by dataset:** Hybrid: Automated, Manually-Collected <br>
**Labeling Method by dataset:** Hybrid: Manually-Labeled, Automated <br>
**Properties:** We evaluated the model on text-based reasoning, coding, agentic tasks, long-context recall, instruction following, and multimodal reasoning. GPQA Diamond contains graduate-level multiple-choice questions written by domain experts in biology, physics, and chemistry. Terminal-Bench evaluates agents on terminal-based tasks. AA-LCR (Artificial Analysis Long Context Recall) evaluates a model's ability to accurately retrieve and recall information from long input contexts. MMMU-Pro is a challenging multimodal understanding benchmark that measures college-level reasoning across diverse disciplines. SciCode evaluates scientific coding capabilities. IFBench evaluates instruction-following capabilities across diverse and structured task constraints. <br>

## Inference:
**Acceleration Engine:** **vLLM** <br>
**Test Hardware:** **NVIDIA Grace Blackwell GB300** <br>

## Post-Training Quantization

Qwen3.8-27B was quantized using a mixed-precision recipe. NVFP4 quantization was applied to the MLP layers and language model head (`lm_head`), while FP8 quantization was applied to the self-attention and linear-attention layers. The NVFP4 layers were calibrated on 2,048 samples using the [Model Optimizer Local-Hessian calibration algorithm](https://nvidia.github.io/Model-Optimizer/reference/generated/modelopt.torch.quantization.config.html#modelopt.torch.quantization.config.LocalHessianCalibConfig).

To learn more about the algorithm, see [Local Hessian for NVFP4 Quantization](https://nvidia.github.io/Model-Optimizer/announcements/local-hessian.html). To reproduce this checkpoint, follow the command in the [Using Local Hessian](https://nvidia.github.io/Model-Optimizer/announcements/local-hessian.html#using-local-hessian) section.

## Usage

To serve this checkpoint with [vLLM](https://github.com/vllm-project/vllm), you can start the docker `vllm/vllm-openai:nightly` and run the sample command below:

```sh
vllm serve nvidia/Qwen3.8-27B-NVFP4 \
    --port 8000 \
    --kv-cache-dtype fp8_e4m3 \
    --tensor-parallel-size 4 \
    --max-model-len 262144 \
    --reasoning-parser qwen3 \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder \
    --mm-encoder-tp-mode data \
    --seed 0 \
    --gpu-memory-utilization 0.85 \
    --max-num-seqs 32 \
    --max-num-batched-tokens 32768 \
    --enable-chunked-prefill
```

To serve this checkpoint with [SGLang](https://github.com/sgl-project/sglang), you can start the docker `lmsysorg/sglang:dev` and run the sample command below:

```sh
sglang serve \
  --trust-remote-code \
  --model-path nvidia/Qwen3.8-27B-NVFP4 \
  --kv-cache-dtype fp8_e4m3 \
  --mem-fraction-static 0.85 \
  --chunked-prefill-size 2048 \
  --reasoning-parser qwen3 \
  --tool-call-parser qwen3_coder \
  --mamba-full-memory-ratio 4.59 \
  --host 0.0.0.0 \
  --port 30000 \
  --mamba-radix-cache-strategy extra_buffer \
  --mamba-ssm-dtype float32
```

For more details please refer to [SGLang cookbook](https://docs.sglang.io/cookbook/autoregressive/Qwen/Qwen3.8-27B).

## Evaluation
The accuracy benchmark results are presented in the table below:
<table>
  <tr>
    <th>Benchmark</th>
    <th>Qwen3.8-27B BF16</th>
    <th>Qwen3.8-27B NVFP4</th>
  </tr>
  <tr><td>GPQA Diamond</td><td>88.92</td><td>88.01</td></tr>
  <tr><td>Terminal-Bench</td><td>75.56</td><td>74.02</td></tr>
  <tr><td>AA-LCR</td><td>72.63</td><td>73.38</td></tr>
  <tr><td>MMMU-Pro</td><td>75.14</td><td>74.86</td></tr>
  <tr><td>SciCode</td><td>47.93</td><td>48.41</td></tr>
  <tr><td>IFBench</td><td>80.07</td><td>78.93</td></tr>
</table>

> **Evaluation settings:** All evaluations used `temperature=1.0`, `top_p=0.95`, and a vLLM context limit of 262,144 tokens. All benchmarks except Terminal-Bench 2.1 used `max_new_tokens=65536`.
  Terminal-Bench 2.1 did not set an explicit generation-token limit;

## Model Limitations:
The base model was trained on data that contains toxic language and societal biases originally crawled from the internet. Therefore, the model may amplify those biases and return toxic responses especially when prompted with toxic prompts. The model may generate answers that may be inaccurate, omit key information, or include irrelevant or redundant text producing socially unacceptable or undesirable text, even if the prompt itself does not include anything explicitly offensive.

## Ethical Considerations

NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. Developers should work with their internal model team to ensure this model meets requirements for the relevant industry and use case and addresses unforeseen product misuse.

Please make sure you have proper rights and permissions for all input image and video content; if image or video includes people, personal health information, or intellectual property, the image or video generated will not blur or maintain proportions of image subjects included.

Please report model quality, risk, security vulnerabilities or NVIDIA AI Concerns [link](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail).
