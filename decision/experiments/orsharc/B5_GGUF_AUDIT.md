# Experiment B5: Elephant-1 GGUF audit

2026-10-23 UTC. Read-only source and metadata audit; no model execution.

**Assessment: PARTIAL. Recommendation: NO-GO for B5 smoke with the existing
standard GGUF files.** Q4_K_M and Q8_0 both exist from a second publisher, but
their headers omit Elephant's decision settings. Stock llama.cpp does not expose
them as decision models. An independent fork implements Elephant decisions and
publishes other quantization formats, but its complete H100 path remains
unverified. This is a current integration gap, not a fundamental limit of GGUF.

## Verified repositories and files

Sizes below are exact bytes from the Hub API, not rounded card estimates.
Pinned recursive trees returned complete listings with no next-page link.

**Requested publisher:**
[prithivMLmods/elephant-1-GGUF](https://huggingface.co/prithivMLmods/elephant-1-GGUF/tree/77d938a13309d7b3da168d87abf5589f9329b122),
revision `77d938a13309d7b3da168d87abf5589f9329b122`.
The complete tree contains `.gitattributes`, `README.md`, and these six files:

| Filename | Type verified in header | Bytes |
|---|---|---:|
| `elephant-1.BF16.gguf` | BF16 (`file_type=32`) | 17,920,697,472 |
| `elephant-1.Q3_K_L.gguf` | Q3_K_L (`13`) | 4,925,514,880 |
| `elephant-1.Q4_K_M.gguf` | Q4_K_M (`15`) | 5,629,109,376 |
| `elephant-1.Q5_K_M.gguf` | Q5_K_M (`17`) | 6,467,970,176 |
| `elephant-1.Q6_K.gguf` | Q6_K (`18`) | 7,359,259,776 |
| `elephant-1.mmproj-bf16.gguf` | BF16 vision/projector supplement | 921,704,960 |

**Q8_0 is absent from this inspected revision's complete file list.** This is
not a claim about unpublished files, other revisions, or other publishers.
The projector is not an 8-bit language checkpoint.
[API](https://huggingface.co/api/models/prithivMLmods/elephant-1-GGUF/revision/77d938a13309d7b3da168d87abf5589f9329b122?blobs=true),
[tree API](https://huggingface.co/api/models/prithivMLmods/elephant-1-GGUF/tree/77d938a13309d7b3da168d87abf5589f9329b122?recursive=true).

**Alternative with both required types:**
[mradermacher/elephant-1-GGUF](https://huggingface.co/mradermacher/elephant-1-GGUF/tree/247371a14d51cb5af533fcaf93485048a4db3ac4),
revision `247371a14d51cb5af533fcaf93485048a4db3ac4`.
The complete tree contains `.gitattributes`, `README.md`, and these files:

| Filename | Published type | Bytes |
|---|---|---:|
| `elephant-1.IQ4_XS.gguf` | IQ4_XS | 5,227,898,432 |
| `elephant-1.Q2_K.gguf` | Q2_K | 3,827,263,040 |
| `elephant-1.Q3_K_L.gguf` | Q3_K_L | 4,925,515,328 |
| `elephant-1.Q3_K_M.gguf` | Q3_K_M | 4,623,525,440 |
| `elephant-1.Q3_K_S.gguf` | Q3_K_S | 4,259,407,424 |
| `elephant-1.Q4_K_M.gguf` | **Q4_K_M; header verified** | **5,629,109,824** |
| `elephant-1.Q4_K_S.gguf` | Q4_K_S | 5,351,630,400 |
| `elephant-1.Q5_K_M.gguf` | Q5_K_M | 6,467,970,624 |
| `elephant-1.Q5_K_S.gguf` | Q5_K_S | 6,305,310,272 |
| `elephant-1.Q6_K.gguf` | Q6_K | 7,359,260,224 |
| `elephant-1.Q8_0.gguf` | **Q8_0; header verified** | **9,527,502,400** |
| `elephant-1.f16.gguf` | F16 | 17,920,697,920 |
| `elephant-1.mmproj-Q8_0.gguf` | Q8_0 projector; header verified | 624,229,760 |
| `elephant-1.mmproj-f16.gguf` | F16 projector | 918,166,016 |

Unmarked alternative types are verified as listed files; their binary headers
were not read. [API](https://huggingface.co/api/models/mradermacher/elephant-1-GGUF/revision/247371a14d51cb5af533fcaf93485048a4db3ac4?blobs=true),
[tree API](https://huggingface.co/api/models/mradermacher/elephant-1-GGUF/tree/247371a14d51cb5af533fcaf93485048a4db3ac4?recursive=true).

Hub-reported LFS SHA-256 identifiers for the main candidates:

```text
prithivMLmods Q4_K_M: 764ef1834ffe7721f11ff7ed2ba42c3616140493b7505e2928eee057496c95a4
mradermacher Q4_K_M: 10c77ea2339ffe7e012db453e77fa30762ba27377b117985db5141743ad17939
mradermacher Q8_0:   01d599611d5156af6c9ca48becd4141b8e18b9db7eff02f97a6033d0215f8b84
```

These identify the published objects; full-file hashes were not recomputed.

## Provenance and actual GGUF metadata

The original developer is Base Labs at Baseten. Its
[Elephant release](https://huggingface.co/baselabs/elephant-1/tree/b3f8a93462fb1051a9de031e94a7ff380e254072)
is revision `b3f8a93462fb1051a9de031e94a7ff380e254072`, the same source pinned in
Experiment B. The inspected developer account lists Elephant and Mammoth;
neither is an official GGUF release. The searches below found two standard
Elephant GGUF publishers, one ROCm fork publisher, and MLX alternatives. This
search does not prove that no other publisher exists.

Both standard GGUF cards name `baselabs/elephant-1` as their source. Neither
provides an exact source revision, converter commit, complete conversion command,
or Elephant decision-parity test. Their generic `general.base_model.0.repo_url`
points to **Qwen/Qwen3.5-9B**, the ancestor; this field alone cannot prove the
Elephant fine-tune was preserved. mradermacher provides stronger conversion
traceability: static quantization, `convert_type=hf`, quantizer version `2`,
publisher, timestamps, and `general.source.url=baselabs/elephant-1` in the binary
metadata. It remains a third-party conversion with incomplete reproduction
details. [Requested card](https://huggingface.co/prithivMLmods/elephant-1-GGUF/blob/77d938a13309d7b3da168d87abf5589f9329b122/README.md),
[alternative card](https://huggingface.co/mradermacher/elephant-1-GGUF/blob/247371a14d51cb5af533fcaf93485048a4db3ac4/README.md).

Bounded HTTP `Range` reads parsed complete GGUF v3 metadata and tensor
directories. Each standard language file inspected has 427 tensors and
8,953,803,264 parameters. Both Q4_K_M candidates have 217 Q4_K, 33 Q6_K and 177
F32 tensors; mradermacher Q8_0 has 250 Q8_0 and 177 F32 tensors. Thus “4-bit” is a
mixed precision recipe. The output LM head is Q6_K in Q4_K_M and Q8_0 in Q8_0.

All seven inspected standard language headers lack `qwen35.decision.*`, a named
`systemone` template, and Elephant's block-attention/readout settings. They carry
the ordinary chat template and `context_length=262144`; Elephant's released
decision limit is **16384**, so the GGUF context field does not establish a
larger valid decision context. mradermacher also declares 33 blocks and one MTP
layer while its tensor directory contains only blocks 0–31. The requested
publisher declares 32 blocks. Whether the alternative's MTP metadata is handled
correctly at load time is untested; it is not proof of a loading failure.

## Decision runtime and H100 feasibility

| Requirement | Evidence and result |
|---|---|
| Native attention | The released [`config.json`](https://huggingface.co/baselabs/elephant-1/blob/b3f8a93462fb1051a9de031e94a7ff380e254072/config.json) requires block-bidirectional full attention and causal linear attention. The validated [native readout](artifacts/upstream/onejev/qev/readout.py) uses separate prefix/suffix blocks. Generic causal inference changes the model. |
| Native decisions and option probabilities | Elephant uses `slot_lm`, verified slot tokens, its own request rendering and option softmax. Stock llama.cpp at `f5027fb21307eb264aed483242603d68c87b0fe3` requires `<arch>.decision.type`; absent metadata yields “not a decision model”. The existing Qwen35 converter has OpenJev support with a different prompt/calibration, not Elephant support. Adding an OpenJev label would not establish compatibility. |
| Calibration | The standard files have no decision-temperature metadata. No evidence establishes native probability, Noul, Score, option-order or calibration parity. A generated label or successful LM load is insufficient. |
| CUDA hardware | Stock llama.cpp supports CUDA and Q4_K/Q6_K/Q8_0 kernels. Local read-only `nvidia-smi` reports H100 80 GB, compute capability 9.0, driver 570.86.10; Phase 3 records CUDA 12.8. No Apple/MLX dependency is required for this general backend. This verifies hardware suitability, not Elephant decision execution. |

Pinned runtime sources:
[decision detection](https://github.com/ggml-org/llama.cpp/blob/f5027fb21307eb264aed483242603d68c87b0fe3/common/common.cpp#L1165),
[endpoint rejection](https://github.com/ggml-org/llama.cpp/blob/f5027fb21307eb264aed483242603d68c87b0fe3/tools/server/server-context.cpp#L6481),
[Qwen/OpenJev conversion](https://github.com/ggml-org/llama.cpp/blob/f5027fb21307eb264aed483242603d68c87b0fe3/conversion/qwen.py#L674),
[System One behavior](https://github.com/ggml-org/llama.cpp/blob/f5027fb21307eb264aed483242603d68c87b0fe3/tools/server/README.md#post-v1systemone-typesafe-compatible-system-one-api),
[CUDA build](https://github.com/ggml-org/llama.cpp/blob/f5027fb21307eb264aed483242603d68c87b0fe3/docs/build.md#cuda),
[CUDA quantized operations](https://github.com/ggml-org/llama.cpp/blob/f5027fb21307eb264aed483242603d68c87b0fe3/ggml/src/ggml-cuda/mmvq.cu).

**Relevant independent fork, not an established H100 solution:**
[Harsha001/elephant-1-ROCmFPX](https://huggingface.co/Harsha001/elephant-1-ROCmFPX/tree/846a8fd80721ae92b466b5b29cc1da38a3075407),
revision `846a8fd80721ae92b466b5b29cc1da38a3075407`:

| Filename | Type | Bytes |
|---|---|---:|
| `elephant-1-Q4_0_ROCMFP4_FAST.gguf` | ROCmFP4 FAST | 4,771,360,768 |
| `elephant-1-Q4_0_ROCMFP4_STRIX.gguf` | ROCmFP4 STRIX | 5,092,634,624 |
| `elephant-1-Q6_0_ROCMFPX.gguf` | ROCmFPX 6-bit | 7,370,059,776 |
| `elephant-1-Q8_0_ROCMFPX.gguf` | ROCmFPX 8-bit | 9,247,731,712 |

The STRIX and 8-bit headers were checked: both contain
`decision.type=baselabs-qev`, `decision.attention=block_bidirectional` and 256
labels. Their main quantized tensor IDs are 100/101 and 103, respectively;
**these are not Q4_K_M and standard Q8_0**. The card documents conversion and
importance-matrix steps, and reports CPU decision probabilities within 0.004 of
the developer runtime. This is a publisher claim, not a reproduced H100 result.
[Card](https://huggingface.co/Harsha001/elephant-1-ROCmFPX/blob/846a8fd80721ae92b466b5b29cc1da38a3075407/README.md),
[API](https://huggingface.co/api/models/Harsha001/elephant-1-ROCmFPX/revision/846a8fd80721ae92b466b5b29cc1da38a3075407?blobs=true).

The fork at `b130d49950435b6634a351c3060c620a53b08a28` has a real
[`llama-decide` implementation](https://github.com/HarshaVardhan3002/ROCmFPX/blob/b130d49950435b6634a351c3060c620a53b08a28/tools/decide/decide.cpp)
and [Elephant converter](https://github.com/HarshaVardhan3002/ROCmFPX/blob/b130d49950435b6634a351c3060c620a53b08a28/conversion/qwen.py#L693).
It implements block execution, slot probabilities, typed answers and temperature
lookup. However, a standard file without decision metadata defaults to **causal**
execution there too. The fork contains CUDA build support and
[ROCmFPX CUDA operation cases](https://github.com/HarshaVardhan3002/ROCmFPX/blob/b130d49950435b6634a351c3060c620a53b08a28/ggml/src/ggml-cuda/convert.cu);
it must not be dismissed as inherently AMD-only. Successful H100 compilation,
complete quantized operation coverage, native decision parity and calibration
remain unverified. No build or smoke was attempted. Metadata repair/reconversion
of the standard pair would also be additional work.

Search also found MLX checkpoints, outside the requested non-MLX path:

| Repository | Revision | Weight files and bytes |
|---|---|---|
| `midium-ai/elephant-1-8bit` | `68917ae4280a2218d3e0dbcd360e38eddeb3dd95` | two safetensors shards: 5,346,606,478 + 4,167,925,545 |
| `0x36/elephant-1-9B-MLX-4bit` | `16de0e500eea4e44fc7db79928c03c53e324bcb0` | `model.safetensors`: 5,038,160,953 |
| `0x36/elephant-1-9B-MLX-8bit` | `5a79d9097ccfaeb913f88f65607f02199ceb24ab` | two safetensors shards: 5,346,606,478 + 4,167,925,545 |

Their Hub APIs identify MLX and list no GGUF files. Midium's
[card](https://huggingface.co/midium-ai/elephant-1-8bit/blob/68917ae4280a2218d3e0dbcd360e38eddeb3dd95/README.md)
explicitly requires its decision runtime; this does not establish CUDA GGUF support.

## Fair B5 comparison and minimal alternative

**A fair 4-bit/8-bit decision comparison on H100 is not yet established.** The
mradermacher pair removes the missing-weight problem, but not the attention,
runtime, conversion-provenance or calibration gaps. Changing publisher for each
precision would add another conversion difference. Quantized heads also differ
from the validated native FP32 output head and must be reported.

All public candidates claim **Original Elephant**, not our **OR-ShARC FT4096**
checkpoint. After suitable runtime verification, they could support only an
Original-Elephant quantization result. They cannot show quantization robustness
of FT4096 or be compared with FT4096 BF16 as a precision-only change. The current
[B5 matrix](configs/matrices.json) uses `original`; an FT4096 extension needs its
own matched BF16/4-bit/8-bit checkpoint and condition identity. The existing
comparison gate also checks engine and checkpoint identity; no bypass is proposed.

**Simplest next implementation candidate: bitsandbytes with the existing
Transformers/OneJev runtime**, in an isolated environment:

- 8-bit: `BitsAndBytesConfig(load_in_8bit=True)` (LLM.int8).
- 4-bit: `load_in_4bit=True`, `bnb_4bit_quant_type="nf4"`, BF16 compute.
- Quantize eligible linear layers; keep the native output head in FP32, and
  preserve attention, renderer, slot mapping and non-linear/recurrent operations.
  Use the same exclusions in both conditions and record them.

These are established CUDA framework paths with prebuilt kernels; no custom
kernel is proposed. Official bitsandbytes 0.50.2 wheels cover CUDA 12.8 / sm90.
Elephant integration is **plausible, not verified**: the current `DecisionEngine`
loader does not accept a quantization config, and Experiment B deliberately
rejects quantized Elephant. A later change must verify native decision behavior,
head handling, probabilities, save/reload and per-precision dev calibration.
Use the same local Original checkpoint for current B5; quantize the same merged
FT4096 checkpoint in all conditions only if that extension is selected later.
[Transformers integration](https://huggingface.co/docs/transformers/quantization/bitsandbytes),
[CUDA wheel support](https://huggingface.co/docs/bitsandbytes/v0.50.2/en/installation#nvidia-cuda).

## Read-only verification commands

These commands retrieve metadata, cards and source only. They do not launch a model.

```bash
# Exact file inventories, byte sizes, LFS identifiers and cards at audited revisions:
curl -fsSL 'https://huggingface.co/api/models/prithivMLmods/elephant-1-GGUF/revision/77d938a13309d7b3da168d87abf5589f9329b122?blobs=true'
curl -fsSL 'https://huggingface.co/api/models/mradermacher/elephant-1-GGUF/revision/247371a14d51cb5af533fcaf93485048a4db3ac4?blobs=true'
# Alternative discovery; inspect pagination before making an absence claim:
curl -fsSL -D - 'https://huggingface.co/api/models?search=elephant&filter=gguf&limit=1000&full=true'
curl -fsSL -D - 'https://huggingface.co/api/models?filter=base_model:quantized:baselabs/elephant-1&limit=1000&full=true'
curl -fsSL 'https://huggingface.co/api/models?author=baselabs&limit=1000&full=true'
# Hardware metadata only:
nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv,noheader
```

For binary verification, each immutable `resolve/<revision>/<filename>` URL was
read with bounded `Range` requests, rejecting any response other than HTTP 206
with the requested `Content-Range`. GGUF v3 header fields, all key/value entries,
and all tensor descriptors were parsed according to the
[GGUF specification](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md).
The type IDs above follow pinned
[GGUF constants](https://github.com/ggml-org/llama.cpp/blob/f5027fb21307eb264aed483242603d68c87b0fe3/gguf-py/gguf/constants.py).
The metadata/tensor-directory end offsets were:

| Files | Metadata end | Tensor-directory end |
|---|---:|---:|
| requested publisher's five language files | 10,942,996 | 10,968,174 |
| alternative Q4_K_M / Q8_0 | 10,943,435 | 10,968,613 |
| ROCm STRIX / 8-bit | 10,945,939 | 10,971,117 |

At most 11,538,432 bytes per language file were transferred; no complete weights
were downloaded. Temporary API responses, header summaries and the reader are
in `/tmp/b5-audit/` (`read_headers.py`, `*gguf.json`). They are audit scratch
files, not Experiment B inputs or admission evidence.
The exact standard-file inspection command was
`python3 /tmp/b5-audit/read_headers.py > /tmp/b5-audit/header-summaries.jsonl`;
the same reader's `inspect()` function checked the two ROCm files separately.

**Stopped after the audit.** No GPU smoke, quantization, training, formal
evaluation or dependency installation ran. Phase 3 admission and all existing
Experiment B files were preserved; this document is the only added project file.
