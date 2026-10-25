# B5 Equal-VRAM Smoke: StartLux Instant vs Original Elephant BF16

**Decision: CONDITIONAL GO.** StartLux native decision inference worked on one H100. Peak GPU memory was close to Elephant: 22,344 MiB versus 23,644 MiB, a 5.5% difference. The memory-matched comparison is defensible. StartLux was slower: warm median request latency was 0.224 s versus 0.102 s. Its runtime also differed from the version listed as tested in the model card.

Use StartLux for a formal B5 extension only if this latency is acceptable. For that extension, use the existing evaluation and calibration path, preserve StartLux probabilities and temperatures, and record the llama.cpp version. Do not treat this 24-case smoke as an evaluation result.

## Model and runtime

| Item | StartLux | Elephant control |
|---|---|---|
| Model | StartLux-Decision v1.1 27B Instant Q4_K_M | Original Elephant-1 9B BF16 |
| Source revision | GGUF `67834e9`; base configs `32e7d35` | `b3f8a93` |
| Model storage | 16,547,400,128 bytes (GGUF) | 18,841,854,787 bytes across regular checkpoint files |
| Inference | `startlux-decide==1.1.0`, official llama.cpp CUDA image b11804 | Native `QevBackend`, identity calibration |
| Attention and device | Bidirectional; 65/65 layers on GPU0 | Noncausal; CUDA0 |

The GGUF repository revision is `67834e921e8a6f977088d9b560ee9e02dffc87bc`; the base-config revision is `32e7d35d74db615c4a059181efbf21c5421f3561`. The GGUF SHA-256 is `cf5d358624862e30cc9ee71182498c4795d31e9fb23315f29ac98ec440ba8b11`. The weights use CC BY-NC 4.0 for non-commercial use. The inference code uses Apache-2.0. The official model page and serving source are linked here: [StartLux GGUF model](https://huggingface.co/startlux-models/StartLux-Decision-v1.1-27B-Instant-GGUF), [base model and config](https://huggingface.co/startlux-models/StartLux-Decision-v1.1-27B-Instant), [official runtime at the pinned revision](https://github.com/StartLuxLabs/StartLux-Decision/tree/83fe6c7d61a687a6e17f118b4f7d8f429f5d3a01), and [license](https://github.com/StartLuxLabs/StartLux-Decision/blob/83fe6c7d61a687a6e17f118b4f7d8f429f5d3a01/README.md#license).

The StartLux model card lists llama.cpp b11549 as tested. The smoke used the official upstream CUDA 12.8.1 image at b11804, pinned by digest. A local CUDA 12.8 build was not available because the installed compiler tools lacked `nvcc` and `fatbinary`. The b11804 run worked, but this version difference remains a formal-run caveat. See [runtime provenance](./run_metadata.json).

## Native decision compatibility

The StartLux server reported `kind=instant`, bidirectional prompt attention, and `bidir_mode=server`. It used one slot, context 20,480, and batch size 1. The llama.cpp flags included `-ngl 99`, `-c 20480`, `--parallel 1`, `--ctx-checkpoints 0`, `--split-mode none`, and `-b/-ub 16384`. The log reported `offloaded 65/65 layers to GPU`. Only GPU0 was visible to the process; the other four H100s stayed idle. No model layers were offloaded to CPU.

The native Choice, Noul, and Score probes all returned finite, normalized probabilities. The Choice probe returned Yes 0.00938, No 0.92463, and Inquire 0.06599. The Noul probe returned P(true)=0.79052. The five-level Score probe returned an expected score of 3.416. Each of the 24 paired Choice calls succeeded, returned all three probabilities, and had no NaN values or truncation.

StartLux kept its configured temperatures: Choice 1.0867, Noul 1.7238, and Score 1.1841. Elephant used identity calibration at temperature 1.0. Both native Choice confidence fields use normalized margin above a uniform split. The field is not the same as the probability of the selected option. The paired output records both confidence and the full probability map.

## Paired smoke results

The fixed sample has 24 development-selection cases. It has four cases for each Yes/No/Inquire × Seen/Unseen cell. Rule text length ranges from 48 to 171 words. The fixed selection covers the 10th, 40th, 70th, and 95th length quantiles with deterministic ID tie breaks. The prompts use the same rule, initial question, scenario, history, and candidate meanings. Each engine uses its own native prompt format. A separate sample-0 warmup was excluded for each model. No gold labels were sent to either model. StartLux used 178–330 input tokens; Elephant used 204–356. StartLux returns typed probabilities without answer-generation tokens; Elephant's native readout returns one output token.

| Measure | StartLux | Elephant |
|---|---:|---:|
| Correct | 18/24 (75.0%) | 17/24 (70.8%) |
| Mean native confidence | 0.775 | 0.868 |
| Mean maximum option probability | 0.850 | 0.912 |
| Warm request p50 / p95 | 0.224 / 0.264 s | 0.102 / 0.112 s |
| Sequential request rate | 4.58/s | 9.71/s |

The models agreed on the top prediction in 21/24 cases (87.5%). Mean total variation between their option distributions was 0.135; mean Jensen–Shannon divergence was 0.052 bits. Accuracy is only a diagnostic on this small sample. It does not support a formal accuracy or calibration claim. The observed probability differences also reflect each model's native calibration: StartLux kept its shipped type temperatures, and Elephant kept identity calibration.

## GPU memory and load

Both peak readings use external `nvidia-smi` sampling at about 100 ms intervals. Each run used GPU0 alone, with no other GPU application. H100 reported 81,559 MiB total.

| Measure | StartLux | Elephant |
|---|---:|---:|
| Peak device memory | 22,344 MiB (21.82 GiB) | 23,644 MiB (23.09 GiB) |
| Peak as share of reported HBM | 27.4% | 29.0% |
| Load timing | First successful load 46.73 s; later cache-hot model load 4.69 s; final server health ready in 7.84 s | QEV model load 8.43 s |
| Framework memory metric | llama.cpp; use external device reading | PyTorch peak allocated 24,010,347,520 B; reserved 24,071,110,656 B |

The StartLux GGUF is smaller on disk than its peak VRAM. Do not use file size as a VRAM estimate. The 22.39 GiB Phase 3 Elephant allocator figure is not used here; this table uses the same external `nvidia-smi` method for both models. The model was already in the OS file cache on later starts, so the cache-hot load and first successful load are both recorded. Loader timers come from different engines and are not a controlled cold-storage comparison.

The GPU sampler recorded about 0.164 H100-hours with GPU0 memory above 1,000 MiB, including StartLux retries and the Elephant control. The smoke stayed within the one-H100-hour limit. At the final check, all five GPUs had 1 MiB used and zero utilization; no compute application remained.

## Limits and warnings

- The sample is a small development-only diagnostic. No formal OR-ShARC evaluation, training, B5 matrix, or official final-test data was used.
- Elephant emitted a warning that `causal_conv1d_fn` used a reference PyTorch fallback because `causal_conv1d` was not installed. This can reduce Elephant speed.
- A high-verbosity StartLux warmup produced heavy logging. It was excluded from latency results. The final measured run used normal logging.
- Image inference was not tested. The smoke used text only and did not download a separate vision encoder.
- Existing Experiment B and Elephant files were checked against the pre-run snapshot: 94,134 files checked, none missing or changed. Generated package build metadata was removed.

## Artifacts

- [Machine-readable run metadata](./run_metadata.json)
- [Artifact checksum manifest](./artifact_manifest.json)
- [Resource and GPU memory summary](./resource_summary.json)
- [Paired aggregate metrics](./paired_comparison.json)
- [Paired sample predictions, probabilities, confidence, and latency](./paired-results.jsonl)
- [Frozen sample and selection method](./samples.jsonl), [selection record](./sample_selection.json)
- Raw responses: [StartLux](./runtime/startlux-results.jsonl), [Elephant](./runtime/elephant-results.jsonl)
- Native type probes and confidence semantics: [type probe](./runtime/startlux-type-probe.json), [confidence check](./runtime/confidence-semantics.json)
- Runtime logs and 100 ms GPU samples are in [`runtime/`](./runtime/).
