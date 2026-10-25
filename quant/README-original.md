> 2026-10-21 Terminal-Bench Low／MTP3 Smoke 已完成並暫停：30／30，24 PASS、4 MODEL_FAIL、2 TIMEOUT。正式 267 Trials 尚未啟動，等待使用者確認。使用者已確認沿用既有 NaviQuant／Aqua checkpoints。結果與限制見 [Smoke 報告](build/terminal_bench/low_mtp3/REPORT.md)。

# 模型 fidelity 實驗

完整交付統一以 [REPORT.md](REPORT.md) 為主：包含九模型 Fidelity、五模型全量雙模式／三 seeds IFEval、100K 補測、配對統計、GPU 成本、MMMU-Pro pilot 與後續限制。本文保留環境與操作指令；各階段 Markdown、CSV、JSONL 及舊快照作為可追查證據，不取代整合報告的最終結果。

## Terminal-Bench 2.1／Agentic Trajectory Fidelity（2026-10-20）

**目前 PAUSED，尚不具正式啟動資格。** Script／Config、五卡 BF16 同時 serving、多輪 Thinking 與 12-token full-vocabulary Fidelity 已驗證；Docker Engine／Compose 不可用，因此指定五題 × False／True 官方 Smoke 為 **0／10**。合成測試不列為 Terminal-Bench 分數。89 題 Gate、801 trials 與正式 Agentic KL 均未啟動。

[完整驗證、限制與成本](build/terminal_bench/VALIDATION.md)／[固定配置](assets/terminal_bench_config.json)／[最終計畫與狀態](build/terminal_bench/prepared_20261020/STATE.json)。Terminal-Bench 準備未中止既有 IFEval；IFEval 目前已完成全量結果，詳見 [REPORT.md](REPORT.md)。

2026-10-21 [Docker 環境診斷與處理方式](build/terminal_bench/DOCKER_DIAGNOSIS.md)：已確認缺少程式與 daemon 入口，且掛載／cgroup 受到外層容器限制；rootless 所需條件也未齊備。這是目前選定 Docker 後端的限制；Harbor 的雲端 sandbox 是另一條官方支援路徑，尚未在本專案驗證。

另完成 [PRoot 可行性 Smoke](build/terminal_bench/diagnostics/proot_smoke/REPORT.md)：獨立 Ubuntu rootfs 的 20 項基本檢查通過，保留缺 `/dev/null` 的失敗與修正對照；未實作 Harbor backend、未跑官方任務，不能取代 Docker／遠端 sandbox 的正式相容性驗證。

安裝獨立 Harbor 環境、核對固定來源與 image digests；不修改已驗證 vLLM 環境：

```sh
bash scripts/setup_terminal_bench.sh
.modelopt-venv/bin/python src/terminal_bench.py plan \
  --output build/terminal_bench/prepared_20261020
.modelopt-venv/bin/python src/terminal_bench.py preflight \
  --output build/terminal_bench/prepared_20261020
```

`preflight` 現在會明確失敗，不會啟動 GPU。取得可用 Docker／Compose、相同可掛載工作路徑及 CPU／RAM enforcement 後，才補做已授權 Smoke；五個 endpoint 18100–18104，各自固定一 GPU／一個活動 Agent。所有 task.toml／Verifier／timeout／資源原樣保留：

```sh
# 只 pull 指定五題的固定官方 image digest；不是執行 Agent。
.modelopt-venv/bin/python src/terminal_bench.py lock-images \
  --output build/terminal_bench/prepared_20261020 --stage smoke
.modelopt-venv/bin/python src/terminal_bench.py preflight --container-probe \
  --output build/terminal_bench/prepared_20261020

# 最多十個官方 Smoke：兩波、每波五題、全部 BF16；本次尚未能執行。
.modelopt-venv/bin/python -u src/terminal_bench.py run --stage smoke \
  --output build/terminal_bench/prepared_20261020 --gpus 0 1 2 3 4
.modelopt-venv/bin/python src/terminal_audit.py --stage smoke \
  --runs build/terminal_bench/prepared_20261020 \
  --output build/terminal_bench/prepared_20261020/smoke_audit.json
.modelopt-venv/bin/python src/terminal_analysis.py --stage smoke \
  --runs build/terminal_bench/prepared_20261020 \
  --output build/terminal_bench/official_smoke_analysis
```

同一 `run` 指令及相同 output 即為續跑。已完成整個 job 不載入 GPU；部分完成沿用 Harbor resume，未完成 trial 原始紀錄先存至 `interrupted/`。不要改 seed、來源、code identity 或 GPU 分片後混入同一計畫。Ctrl-C／SIGTERM 會清理本次 workers／services，`STATE.json` 回到 PAUSED；不會自動續跑。完成但失敗的 trial 不為提高分數自動重試。

**以下兩階段未獲本次執行授權，僅供下一次明確授權後使用。** Gate 必須已有通過的 SMOKE_READY；matrix 還必須通過 GATE_READY。每階段結束都退出，不會由 Gate 自動啟動候選：

```sh
# 未執行：完整 BF16 repeat 0，五卡分配全部 89 題一次。
.modelopt-venv/bin/python src/terminal_bench.py lock-images \
  --output build/terminal_bench/prepared_20261020 --stage gate
.modelopt-venv/bin/python -u src/terminal_bench.py run --stage gate --authorize-full \
  --output build/terminal_bench/prepared_20261020 --gpus 0 1 2 3 4
.modelopt-venv/bin/python src/terminal_audit.py --stage gate \
  --runs build/terminal_bench/prepared_20261020 \
  --output build/terminal_bench/prepared_20261020/gate_audit.json

# 未執行：其餘 712；重用既有 BF16 repeat 0，最終合計 801。
.modelopt-venv/bin/python -u src/terminal_bench.py run --stage matrix --authorize-full \
  --output build/terminal_bench/prepared_20261020 --gpus 0 1 2 3 4
.modelopt-venv/bin/python src/terminal_analysis.py --stage formal \
  --runs build/terminal_bench/prepared_20261020 \
  --output build/terminal_bench/formal_analysis --resamples 199999
```

主設定維持 Terminus-2 `interleaved_thinking=False`；True 為分開的敏感度協定。兩者均用 Qwen Thinking／xhigh、固定官方 sampling、preserve_thinking=True、MTP Off／BF16 KV。原生服務 context 262,144，共用 output 131,072，Harbor input 預算 131,072；與 Qwen 1M context 下分開 reasoning／final budgets 的建議差異已明列。原生 summarization 保持開啟，若發生 history reset，audit 先停止 READY，需核對 ATIF／API 後再處理。

正式 BF16 trajectories 完成並取得 Agentic KL 執行授權後，保留整個 run 目錄（ATIF 加同存的 API audit），建立固定 windows，再重用原 Fidelity 核心：

```sh
# 準備資料是 CPU 作業；只使用 BF16 的實際 prompt／生成 IDs，不重新 tokenize prefix。
.modelopt-venv/bin/python src/agentic_fidelity.py prepare \
  --trajectories build/terminal_bench/prepared_20261020 \
  --output build/terminal_bench/agentic_windows

# 未執行：完整 Agentic KL 仍需新授權；先獨立 BF16 sanity，再 Fast／Prisma。
.modelopt-venv/bin/python src/agentic_fidelity.py run --authorize-full \
  --prepared build/terminal_bench/agentic_windows \
  --output build/terminal_bench/agentic_results --gpus 0 1
```

本次已執行的極小資料處理與 GPU 路徑見 `build/terminal_bench/validation/fidelity_windows_v1/` 與 `fidelity_smoke_v1/`。如需重新驗證，使用新 output，prepare 加 `--smoke --windows-per-task 3 --tokens-per-window 4` 並指向 `validation/protocol_v1/`；`--smoke` 硬限最多 32 positions。合成來源不能進正式 Agentic KL。

針對性 CPU 測試：

```sh
.modelopt-venv/bin/python -m pytest -q tests/test_terminal_bench.py \
  tests/test_terminal_observer.py tests/test_agentic_fidelity.py tests/test_fidelity.py
```


## IFEval 正式矩陣完成（2026-10-20）

五個模型 × Thinking／Non-thinking × 三個固定 seeds（20261016、20261017、20261018）× Google 官方 541 題，共 **16,230／16,230 筆**；每個模型、模式和 seed 均使用完整相同題目集合。正式輸出與進度在 [run 目錄](build/benchmark/ifeval_modes_formal_20261017/)，完整 199,999-resample 題目層級 paired analysis 在 [analysis_full_20261020](build/benchmark/ifeval_modes_formal_20261017/analysis_full_20261020/)。研究摘要、完整方法、統計比較、補測與限制均已整合至 [REPORT.md](REPORT.md)。

正式模型為 Qwen BF16、NaviQuant Fast、PrismaAQUA 5.5-bit、ModelOpt AutoQuantize 5.5-bit、Unsloth NVFP4。On 使用 xhigh 和 Qwen 建議 sampling；Off 使用 Qwen 建議 sampling。MTP Off、BF16 KV cache。評分器沿用 Google IFEval 官方工具，固定 revision `3ef52111c1fabc301880818f5a547570e926a76b`；逐題固定亂數的 scorer／parser audit 為 30／30 組、16,230 筆通過。

主矩陣原分析記錄 335.087 worker GPU-hours；整合報告補計一筆缺 wall 的中斷起訖後約 335.182 GPU-hours，明細見 [REPORT.md](REPORT.md)。包含中斷續跑的首次到末次 worker 時間跨度 73.95 小時。執行期間依要求調整過輸出上限：Off 實際用過 4,096／8,192／16,384，On 用過 20,480／32,768／49,152／65,000；每題的 cap 有記錄。五模型在同題同 seed 的 cap 有 239 個 Off cells、170 個 On cells 不完全相同；只有三筆輸出實際觸及上限，且都以 65K 截斷。逐模型分布與配對統計見[輸出上限稽核](build/benchmark/ifeval_modes_formal_20261017/output_cap_audit_20261020.md)。主矩陣三筆 65K 截斷的獨立 100K replay 都自然停止、取出 final 並答對；補測沒有覆寫主矩陣。七筆自然停止但 missing-final 的 replay 完全重現原始輸出和 token IDs；NaviQuant Fast `ifeval/3048` seed 20261018 的 replay 依指示停止，沒有保存答案。補測總耗用 6.359 worker GPU-hours。逐題差異見 [補測 CSV](build/benchmark/ifeval_modes_formal_20261017/failure_retry_comparison_20261020.csv)／[Markdown](build/benchmark/ifeval_modes_formal_20261017/failure_retry_comparison_20261020.md)；100K cap 敏感度分析在 [cap_recovery_sensitivity_20261020](build/benchmark/ifeval_modes_formal_20261017/cap_recovery_sensitivity_20261020/)。

正式 IFEval 工具支援逐題保存、檢查和續跑；以下是查看或重現用命令，正式全量推論屬高成本操作：

```sh
# 查看已完成 run 的矩陣狀態
.modelopt-venv/bin/python src/ifeval_modes.py status \
  --output build/benchmark/ifeval_modes_formal_20261017

# 只有在明確安排新的正式 run 時才續跑；已完成題目會略過
.modelopt-venv/bin/python -u src/ifeval_modes.py run --full --resume \
  --output build/benchmark/ifeval_modes_formal_20261017 --gpus 0 1 2 3 4

# 重新產生正式 paired analysis，保留題目內三個 seeds，使用預先指定的 300 項比較家族
.modelopt-venv/bin/python src/analyze_ifeval_modes.py \
  --runs build/benchmark/ifeval_modes_formal_20261017 \
  --output build/benchmark/ifeval_modes_reanalysis \
  --resamples 199999 --family-size 300
```

IFEval prep 階段已完成 60/60 筆必要 smoke，並通過 Google 官方 CLI 重評及 20 個針對性 CPU checks；模型設定、使用方式和 smoke 細節見 [準備紀錄](build/benchmark/ifeval_modes_validation/USAGE.md)、[結果](build/benchmark/ifeval_modes_validation/RESULTS.md)及[官方流程複核](build/benchmark/OFFICIAL_REVIEW.md)。完整 MMMU-Pro 能力 benchmark 尚未執行。

較早的九模型 pilot、Fidelity 與 MTP／DSpark 啟動驗證，見[完整研究報告](REPORT.md)、[九模型 Fidelity CSV](build/autoquant/comparison.csv)、[pilot 一頁摘要](build/benchmark/pilot_v1/decision_report/MONDAY_SUMMARY.md)和[模型選擇分析](build/benchmark/pilot_v1/decision_report/DECISION_REPORT.md)。以下第一輪與擴大實驗章節保留原八模型的實驗紀錄。

2026-10-17 官方複核發現 IFEval 的 HF 與 GitHub 資料有一道 prompt 差異，
正式輸入改用官方 README 指定的 GitHub 版本；正式 sampling 改依 Qwen 的 non-thinking 建議。
MTP 開關已加入相同 Benchmark 流程。後續正式設定及證據以
[官方流程複核](build/benchmark/OFFICIAL_REVIEW.md)為準；下文原 smoke 設定保留供追溯。

程式位於 `src/fidelity.py`。權重讀取自 `/root/model/`；不修改原始權重或模型設定。

安裝環境（Linux x86_64、Python 3.12）：

```sh
uv venv --python 3.12 .vllm-venv
uv pip install --python .vllm-venv/bin/python -r requirements.txt
.vllm-venv/bin/python scripts/setup_cuda_compat.py
export LD_LIBRARY_PATH="$PWD/build/cuda-compat/cuda-compat${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
```

本機核心驅動為 570.86.10，因此把 CUDA 12.9 forward-compatibility 使用者空間函式庫 575.57.08 解壓到 `build/`，並透過上面的 `LD_LIBRARY_PATH` 指定。沒有安裝或修改系統驅動；新 shell 執行測試前也需要設定這個路徑。此作法適用本機的資料中心 H100，參見 [NVIDIA forward compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/forward-compatibility.html)。

本機採用 CUDA 12.9 的官方 wheels。`torchcodec` 也固定為 CUDA 12.9，避免安裝到要求 CUDA 13 的套件。

執行所有模型：

```sh
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8 VLLM_NO_USAGE_STATS=1 HF_HUB_OFFLINE=1 \
  .vllm-venv/bin/python src/fidelity.py run --all --output build/comparison
```

指定一個或數個模型：

```sh
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8 VLLM_NO_USAGE_STATS=1 HF_HUB_OFFLINE=1 \
  .vllm-venv/bin/python src/fidelity.py run \
  --models Qwen-Qwen3.8-27B-FP8 RedHatAI-Qwen3.8-27B-NVFP4 \
  --output build/selected
```

`--models` 可接受 `/root/model/` 下的名稱，或其他本機模型的完整路徑。`--baseline` 可指定 BF16 目錄，`--model-root` 可指定掃描目錄。輸出目錄必須不存在，以保留已有結果。`--max-tokens` 預設為 64，支援 2–511；`--texts` 可指定同格式的 JSON 字串陣列。

每次 `run` 都在兩個獨立的子程序中重新載入 BF16，先通過 sanity check，再執行候選模型。每次只執行一個模型；程序結束後釋放 GPU 記憶體。

輸出包含：

- `comparison.csv`、`comparison.md`：比較表；空白指標表示該次失敗。
- `sanity.json`：第二次獨立 BF16 與第一次比較的結果。
- `tokens.json`：共用 token IDs、tokenizer 路徑、文字檔 SHA-256。
- `inventory.json`：模型清單、設定及權重分片檢查。
- `build/checkpoints.json`、`build/tokenizer_audit.json`、`build/environment.json`：本次模型版本、所有 tokenizer ID 對應的檢查，以及驅動／套件環境。
- `results.json`：所有執行結果。
- 各模型的 `result.json` 與 `run.log`：環境版本、完整分布的雜湊、基本推論 token、錯誤及引擎紀錄。

計算方式：

固定資料為 `assets/fidelity_texts.json` 的 8 段自製中英文文字，包含敘述、程式碼、算術及實驗說明。使用 BF16 tokenizer，`add_special_tokens=False`，不套用 chat template，不加入影像或影片。每段各自建立前文，最多取 64 tokens；預設共 432 個計分位置，不跨段連接。

對 token 序列 `x[0:L]`，計算位置 `t=1..L-1` 的 `p(x[t] | x[:t])`。vLLM 的 `prompt_logprobs[t]` 已完成這個位移，程式將其存為矩陣第 `t-1` 列，target 為 `x[1:]`。第一個 token 無前文，因此不計分。為確認基本推論，每段另外產生一個 token；該 token 不納入指標。

透過 `prompt_logprobs=-1`、`max_logprobs=-1` 取得完整 248,320 維模型輸出 vocabulary（tokenizer 有 248,077 個 token） 的原始 log-softmax；不使用 top-k 近似。每個位置都檢查完整 vocabulary 覆蓋與機率總和。log probabilities 以 FP32 擷取，指標以 FP64 加總，使用自然對數：

- `PPL = exp(-sum(log q(target)) / N)`。
- `PPL delta = PPL(candidate) - PPL(BF16)`。
- `KL = sum_t sum_v p_BF16(v|prefix) * (log p_BF16 - log q_candidate) / N`。
- `Top-1 flip rate = count(argmax p_BF16 != argmax q_candidate) / N`，表格值為 0–1 的比例。

所有樣本按 token 數加權；遮罩或 padding vocabulary 中機率為零的項目，對 KL 貢獻為零。Sanity 門檻為 `abs(KL) <= 1e-6`、flip rate 恰為 0、`abs(PPL delta) <= 1e-6`；若未通過，程式停止候選比較。

推論使用單張 H100、TP=1、BF16 基礎 dtype、eager mode、固定 seed=0，MTP/speculative decoding 關閉，prefix cache 關閉。KV cache 使用 BF16，沒有 KV 量化。block FP8 的 GEMM 固定使用 Triton，以避開本機缺少 CUDA Toolkit 的 DeepGEMM JIT 路徑；其他核心由 vLLM 選擇。完整引擎設定記錄於各次 `run.log`。

限制：

這是短樣本 fidelity 檢查，不是一般能力、長上下文或多模態評測。PPL 較低不表示品質一定較好；KL 與 flip rate 測量對 BF16 的偏離，也不是任務正確率。此實驗不測速度或記憶體效益。

H100 沒有原生 FP4 計算。NVFP4 的實際支援取決於引擎及 checkpoint 格式；Marlin 使用 W4A16 權重推論，不能把結果當成 Blackwell 原生 W4A4 的結果。FP8 層可能仍含 activation quantization，因此這是 MTP 關閉後，各 checkpoint 在已記錄後端上的 fidelity，不能聲稱只測到權重捨入誤差。更換 kernel、GPU 或套件版本後應重新執行 sanity check。

參考官方說明：[vLLM full-vocabulary prompt logprobs](https://docs.vllm.ai/en/v0.17.1/api/vllm/sampling_params/)、[NVFP4 Marlin W4A16](https://docs.vllm.ai/en/latest/api/vllm/model_executor/kernels/linear/nvfp4/marlin/)。

針對性 CPU 測試：

```sh
.vllm-venv/bin/python -m pytest -q tests/test_fidelity.py
```


第一輪（432 tokens）完成結果：8 個模型均成功，包含獨立 BF16 sanity 與簡化後全模型重測。比較使用完整 vocabulary，432 個計分位置。詳見 [第一輪報告](build/REPORT_round1.md)、[CSV](build/comparison.csv)、[Markdown 表格](build/comparison.md)、[模型載入與環境紀錄](build/model_checks.csv)、[消融驗證](build/ablation.json)。最終原始結果與日誌位於 `build/final/`；簡化前結果保留於 `build/validated/`。

## 擴大實驗 v2

資料固定於 [assets/expanded_v2](assets/expanded_v2/README.md)：288 份獨立公開來源文件、55,296 個計分 tokens，涵蓋中英文知識、Lean 形式化數學、Python 程式碼，以及 128 / 512 / 2048 個初始 context tokens。來源 revision、授權與 SHA-256 見 [設定](assets/expanded_config.json) 與 [manifest](assets/expanded_v2/manifest.json)。每個文件只用一次，每組 24 份文件。完整原文、來源網址與共用 token IDs 都已保存。

安裝更新的 requirements 後，可直接重跑全部 8 個模型、獨立 BF16 sanity、統計分析與圖表：

```sh
scripts/run_expanded.sh build/reproduction 0 1 2 3 4
```

若只有一張 GPU，使用 `scripts/run_expanded.sh build/reproduction 0`。輸出目錄必須不存在。每次先在不同程序重新載入 BF16 兩次，整份資料 sanity 通過後，才允許候選模型執行。不同 GPU 只同時執行不同模型，TP 仍是 1。模型之間使用完全相同的 token IDs、前文與計分位置。

`src/fidelity.py run` 新增 `--tokens`、`--gpus`、`--logprob-format dense`。dense 直接使用 vLLM 0.31.0 官方 `prompt_logprob_token_ids=range(vocab_size)`、`prompt_logprob_start=context-1`，只擷取最後 192 個位置的完整 vocabulary logprobs；模型仍看見整段前文。需要設定 `VLLM_USE_V2_MODEL_RUNNER=1`，上面的 shell script 已設定。原本 flat 路徑與 432-token 實驗仍可直接執行。

若只重新建立資料或重做分析：

```sh
.vllm-venv/bin/python src/prepare_dataset.py --output build/dataset_rebuild
.vllm-venv/bin/python src/analyze.py --run build/expanded/full --output build/reanalysis
.vllm-venv/bin/python src/plot_intervals.py build/reanalysis
```

統計使用 SciPy 1.18.0 的 `bootstrap`：先將每個文件的 192 個計分位置聚合，在 domain × context 各層重抽完整文件，模型之間保持配對，固定 seed 20261016，19,999 次 percentile bootstrap。這會保留文件內的 token 相關性。輸出包含各模型整體及分組結果、95% 區間、Fast 對四個指定模型的配對差異、主要 12 項比較的 Bonferroni 校正區間、移除單一領域的敏感度，以及第一輪比較。分組區間屬探索性結果，沒有對全部分組做多重比較校正。

新增針對性 CPU 檢查：

```sh
.vllm-venv/bin/python -m pytest -q tests/test_fidelity.py tests/test_analysis.py
```

開源工具選擇及定義核對見 [TOOL_REVIEW](build/expanded/TOOL_REVIEW.md)。完整 vocabulary KL 沿用已驗證的簡單 NumPy 計算，官方 vLLM 負責模型推論和 log-softmax，SciPy 負責 bootstrap，PyArrow 負責 parquet。沒有引入新的評估 framework。


v2.1 最終結果：全部 8 模型完成每模型 55,296 個計分 tokens；BF16 獨立 sanity 的 288 份分布完全相同。最終簡化後再跑全部模型的 2,736-token 消融，180 份分布雜湊全部相同，7 個針對性測試通過。

- [更新實驗報告](build/REPORT.md)／[第一輪報告](build/REPORT_round1.md)
- [完整比較 CSV](build/expanded/analysis/comparison.csv)／[Markdown](build/expanded/analysis/comparison.md)
- [分組分析](build/expanded/analysis/grouped.csv)／[95% 區間](build/expanded/analysis/intervals.csv)
- [Fast 配對比較與校正區間](build/expanded/analysis/fast_pairwise.md)／[圖表](build/expanded/analysis/fast_intervals.png)
- [移除單一領域的敏感度](build/expanded/analysis/leave_one_domain_out.csv)／[逐文件差異](build/expanded/analysis/document_stability.csv)
- [正確性與消融證據](build/expanded/ablation.json)／[獨立公式稽核](build/expanded/independent_metrics_audit.json)
- [固定公開來源資料集包](build/fidelity-public-v2.1.tar.gz)（2.2 MiB，含原文、IDs、版本與授權，未另上傳外部平台）

核心結果須分指標解讀：Fast 對 RadixArk BF16-LMHead 的 KL 較低、對 NVIDIA 的 Flip 較低；對 RedHatAI／Unsloth 的 KL 較高。其餘 PPL／Flip 差異與各領域限制，請依配對區間與報告判斷，不能將單一點估計解讀為全面品質排名。

可重做 calibration 重疊稽核；程式會下載並驗證固定的公開來源：

```sh
.vllm-venv/bin/python src/audit_calibration.py --output build/expanded/calibration_overlap_recheck.json
```

精確套件版本與官方 wheel 來源見 `build/expanded/environment.freeze.txt`、`package_provenance.json`。完整分布 reference 保留在 `build/expanded/full/bf16_reference/`；候選保留逐文件指標與分布雜湊，獨立 Fast 稽核另保留 20 份原始矩陣。

## 自動量化擴充與下一階段 Benchmark

本次成果、容量範圍、校準資料、搜尋成本、相容性及限制，均見 [REPORT.md](REPORT.md)。
原數值核心 `src/fidelity.py` 未改；新 wrapper 重用既有 full reference，先跑完整 BF16 gate，
只評估兩個新模型，再合併相同版本與條件的舊結果。舊的 `build/final/`／`build/expanded/` 保留。

建立隔離的 ModelOpt／Benchmark overlay（不修改 `.vllm-venv`）：

```sh
bash scripts/setup_research.sh
HF_HUB_DISABLE_XET=1 .vllm-venv/bin/python src/download_prisma.py
```

上游工具、資料與 Prisma revision 均固定。環境使用 `site.addsitedir`，讓 spawned workers
也載入 base environment 的 CUDA／CUTLASS `.pth`。首次建立 Benchmark inputs：

```sh
.modelopt-venv/bin/python src/benchmark.py prepare
```

已備妥的權重位於 `/root/model/rdtand-Qwen3.8-27B-PrismaAQUA-5.5bit-vllm/` 與
`/root/model/ModelOpt-Qwen3.8-27B-AutoQuantize-5.5bit/`。直接重跑新模型 Fidelity：

```sh
bash scripts/run_autoquant_fidelity.sh build/new_autoquant_run 0
```

若要重新搜尋，使用新的 output 與 checkpoint 目錄，避免覆蓋已完成模型：

```sh
export LD_LIBRARY_PATH="$PWD/build/cuda-compat/cuda-compat${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8 HF_HUB_OFFLINE=1 \
  .modelopt-venv/bin/python src/modelopt_autoquant.py \
  --output build/repeated_search \
  --checkpoint /root/model/ModelOpt-Qwen3.8-27B-AutoQuantize-repeat \
  --score-steps 8
```

校準／validation 與 token IDs 固定於 `assets/autoquant/`，分別為 128／32 份 CNN/DailyMail 文件。
這是原生 `mtq.auto_quantize(method='kl_div')`，沒有 Navi layer map；5.5 bits 為 body cost 上限，
不含 Vision／MTP／lm_head／embedding。實際搜尋為 5.478834 bits、23.536 GB、0.213 GPU-hours。
128 calibration／8 sensitivity 步小於官方建議預設值，不能把本次結果當作 ModelOpt 最佳可能品質。

快速啟動 smoke 每模型每 task 只取兩題，最多輸出 128 tokens：

```sh
bash scripts/smoke_benchmarks.sh 0 build/new_benchmark_smoke
.modelopt-venv/bin/python src/benchmark_analysis.py \
  --runs build/new_benchmark_smoke --output build/new_benchmark_smoke_analysis
```

此啟動 smoke 只驗證載入、輸入配對及評分器，不適合推估完整執行時間。代表性 pilot
核心依分層抽樣使用160題／模型，涵蓋IFEval、MMMU-Pro Standard與Vision；對1,024-token
截斷回答及固定控制另測4,096，再追加獨立40題IFEval樣本，合計200題／模型。
本輪沒有完成8,192探測；舊控制器誤觸的部分輸出已隔離，下列wrapper不再含8K階段。長答案使本輪實際執行約9小時。
只有在確定需要重新推論時，才重跑pilot（新輸出目錄；本次沒有重跑）：

```sh
bash scripts/run_representative_pilot.sh build/benchmark/new_pilot 0 1 2 3 4
```

九模型與主要 inference 設定位於 `assets/benchmark_config.json`，三個 tasks 分別是
`ifeval`、`mmmu_standard`、`mmmu_vision`。每題結果與成本表由 `src/benchmark.py`、
`src/benchmark_budget.py` 保存；完整能力 Benchmark 需明確指定 `--full`，目前尚未執行。
配對分析可指定 `--resamples 199999`。

僅重新分析現有資料，不載入模型：

```sh
.modelopt-venv/bin/python src/benchmark_budget.py \
  --runs build/benchmark/pilot_v1/combined_runs --data assets/benchmark/pilot_v2 \
  --output build/benchmark/reanalysis --gpus 0 1 2 3 4 --resamples 19999
.modelopt-venv/bin/python src/benchmark_analysis.py \
  --runs build/benchmark/pilot_v1/combined_runs --manifest assets/benchmark/pilot_v2/manifest.json \
  --mmmu-cap 4096 --family-size 160 --resamples 199999 --output build/benchmark/repaired_4096
.modelopt-venv/bin/python src/summarize_benchmark_pilot.py \
  --pilot build/benchmark/pilot_v1 --output build/benchmark/new_decision \
  --start 2026-10-18T00:00:00+08:00 --deadline 2026-10-19T09:00:00+08:00
```

`summarize_benchmark_pilot.py`以既有`analysis/`、`combined_runs/`及`official_replay/`產生
各模型／task子集GPU-hours、五GPU排程、截止時間與不確定性情境，並連結Fidelity結果。
IFEval已知38題固定、剩餘503題由獨立40題估算；MMMU按學科加權，不把壓力題當機率樣本。
本輪兩cap共160項探索性family；正式分析依最後名單與假設明列family。
`benchmark_analysis.py --config SUBSET_CONFIG`可用同一模型子集清單作配對分析。
未執行的正式4K設定已存於`assets/benchmark_4096_config.json`（九模型）與
`assets/benchmark_research4_4096_config.json`（BF16/Fast/Prisma/Auto）。完整指令見整合報告。

10-17 更新：預設輸入是 `assets/benchmark/prepared_v2/` 的 Google GitHub canonical IFEval
與官方 MMMU-Pro；sampling 採 Qwen non-thinking 建議，影像上限回復官方 processor 預設。
`src/benchmark.py run --mtp` 啟用原生 MTP（兩個 speculative tokens），省略則 Off；
`--config assets/benchmark_greedy_config.json` 可做 greedy 檢查。
`benchmark_analysis.py --reference-runs OFF_RUNS --runs MTP_RUNS` 比較同 checkpoint 的 On−Off。
完整兩模式與全部比較可明列 `--family-size 300 --resamples 199999`。
重做官方來源／逐題資料／既有 smoke 評分核對（CPU only）：

```sh
.modelopt-venv/bin/python src/verify_benchmark_official.py --output build/new_official_check
```

本輪計分、工期與子集分析的targeted CPU檢查（17項通過）：

```sh
.modelopt-venv/bin/python -m pytest -q \
  tests/test_benchmark.py tests/test_benchmark_budget.py tests/test_benchmark_decision.py \
  tests/test_benchmark_official.py tests/test_benchmark_pilot_merge.py
```

[本輪完成證據](build/benchmark/pilot_v1/decision_report/decision_audit.json)／
[等價消融](build/benchmark/pilot_v1/decision_report/analysis_validation.json)。
先前Fidelity擴充的13項檢查與啟動smoke另見[完成證據](build/autoquant/completion_audit.json)、
[真實 sensitivity scores](build/autoquant/search_v2/sensitivities.csv)、
[容量與 allocation](build/autoquant/checkpoints/inventory.json)、
[MTP／DSpark 公開證據及實測矩陣](build/benchmark/compatibility.md)、
[逐模型 smoke](build/benchmark/smoke/)。
