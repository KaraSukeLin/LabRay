# 最終移轉版：還原與重現

本目錄已包含最終指定檔案；不需要前版 migration-review 目錄。quant/ 與 decision/ 保留原專案相對路徑。原來源路徑、SHA-256、轉換方式見 source-manifest.csv；全部實際交付檔案另見 FILE-MANIFEST.csv 與 SHA256SUMS。

## 1. 驗證內容與還原小型模型資產

先從壓縮包旁的 `.sha256` 驗證整個 archive。解壓縮後，在本目錄執行：

```sh
sha256sum -c SHA256SUMS
python3 tools/verify-evidence.py
```

第二個命令只使用標準函式庫，重新檢查 hash、九模型 Fidelity 聚合值、IFEval 正式準確率、Terminal-Bench 主分析分母與三筆 cap sensitivity 替換列，不啟動模型。

`metadata/model-metadata/index.json` 列出原 `/root/model` 各小型檔案和 blob。可先還原到一個全新目錄：

```sh
python3 tools/restore-model-metadata.py --output /tmp/restored-model-assets
```

依 index 對應到模型目錄；包含實際 config、tokenizer、template、readout、generation config、Elephant patch 等，共 154 個邏輯檔案。相同內容只保存一次。模型權重不在包內：公開權重依模型指紋中指定 repo/revision 下載；本地 ModelOpt checkpoint 需重新搜尋，Elephant FT checkpoint 需重新訓練。下載後核對權重指紋，再覆回本次本地設定；不要以遠端預設 template 取代本次實際 template。

歷史 manifest 含 `/root/projects/{quant,decision}` 及 `/root/model` 絕對路徑。完整重跑最穩定的方式是沿用這些位置。改路徑時須同步核對 manifest、source signature 和 runtime config。只做包內數值重算可在其他路徑進行。

## 2. quant：環境、固定輸入與重跑

從 quant/ 讀 REPORT.md；Terminal-Bench 以 build/terminal_bench/FINAL_REPORT.md 指向的 formal_three_seed_20261022/FINAL_REPORT_ZH_TW.md 為最新。舊 README、pre-run plan 或單 seed 中間表不能代替最終狀態。

原環境是 Linux x86_64、Python 3.12、H100 80GB，vLLM 0.31.0+cu129、Torch 2.13.0+cu129、Transformers 5.17.0、compressed-tensors 0.17.0。沿用 requirements、freeze、package provenance 及 scripts/setup_cuda_compat.py、setup_research.sh、setup_terminal_bench.sh，分開重建 vLLM、research overlay、Harbor 環境。原驅動與 CUDA compatibility 設定保存在 environment 與 provenance；新環境仍需先跑 BF16 sanity。

保留 assets/expanded_v2 的固定原文與 token IDs、autoquant calibration/validation、benchmark prepared/pilot inputs、所有正式 configs。MMMU 圖片/parquet、NLTK 可下載資源和 Terminal-Bench task fixtures 不保存；以固定 revision 重建，不重新抽樣。NLTK resource 可依原 setup/prepare 程式重新取得；版本由 freeze 固定。

Terminal-Bench 題目來源在 metadata/public-inputs.json：repo `https://github.com/harbor-framework/terminal-bench-2-1.git`，revision `7131e4375048a0e408a8fb404b5f499d726b695b`。清單記錄 1,035 個正式 task 檔案，加 tasks/README.md、dataset.toml 兩個上層檔案的 SHA-256。完整重跑前，先在全新暫存目錄 checkout 此版本，核對清單，再還原到 quant/build/terminal_bench/sources/terminal-bench-2-1；不要使用省略了 fixtures 的部分 task tree。這避免移轉約 56 MiB 可從公開版本取回的素材。

ModelOpt、MMMU、Harbor 與 Terminal-Bench 的實際 HEAD/origin/乾淨工作目錄檢查保存在 metadata/upstream-verified.json。Google IFEval 是下載的來源目錄，沒有 .git；本包已直接保存實際 scorer 原始碼、LICENSE、revision.json 與原 provenance。MMMU-Pro 的官方 Python 評分器也已補存。需要安裝上游工具時，先在空目錄 checkout 固定 commit，再覆回包內需保存的檔案；本包省略 Git object history。

Fidelity：288 文件、每文件 192 scoring positions，共 55,296 tokens。統計使用文件為 cluster、domain×context 分層、配對 bootstrap，固定 seed/比較家族見 assets/fidelity_extension_analysis.json 與原 statistics.json。逐文件 result、sanity、分布 hashes、配對 CI、搜尋 allocation/sensitivities 與校準稽核皆保存，可直接重算聚合統計。完整 vocabulary `.npy` 與搜尋 `.pt` state 不保存；重查原始分布或重做新模型比較時，需重新推論 BF16 reference。

完整重跑指令仍以原 README 的 `scripts/run_expanded.sh`、`scripts/run_autoquant_fidelity.sh` 為準，使用新 output。ModelOpt 使用保存的 assets/autoquant、search_v2/parameters.json、allocation.json、sensitivities.csv 重建本地 checkpoint；不能假設 `ModelOpt-Qwen3.8-27B-AutoQuantize-5.5bit` 是可下載的遠端 repo。

## 3. quant：精簡 benchmark 數據如何重新分析

本版將精簡的 questions.jsonl/extensions.jsonl 放回原相對位置，並由 source-manifest 標記為轉換資料。移除 prompt、token IDs、reasoning、raw_output、response，保留這些欄位原 canonical JSON 的 SHA-256；另存 token 陣列長度與 response 是否為空。final_answer、answer、原 strict/loose score、sampling、cap、timing、cost 等數據保留。截斷與 missing-final 的完整原始列保存在同目錄 exception-original-records.json。

因此原 analyzer 對 raw 欄位的 assertion 需要換成 hash/count 檢查。本包的 tools/analysis-adapters/ 是對原程式的窄範圍相容調整，各附 `.diff`；統計公式、resampling、家族大小與配對規則未變，原 src/ 也未改。不能把精簡資料宣稱為完整原始輸出。

從 quant/ 執行（先依原環境建立 Python/NumPy/SciPy）：

```sh
export PYTHONPATH="$PWD/src"
.vllm-venv/bin/python ../tools/analysis-adapters/analyze_ifeval_modes.py \
  --runs build/benchmark/ifeval_modes_formal_20261017 \
  --output build/reanalysis-ifeval --resamples 199999 --family-size 300

.vllm-venv/bin/python ../tools/analysis-adapters/benchmark_analysis.py \
  --runs build/benchmark/pilot_v1/combined_runs \
  --manifest assets/benchmark/pilot_v2/manifest.json \
  --mmmu-cap 4096 --family-size 160 --resamples 199999 \
  --output build/reanalysis-pilot

.vllm-venv/bin/python ../tools/analysis-adapters/benchmark_budget.py \
  --runs build/benchmark/pilot_v1/combined_runs --data assets/benchmark/pilot_v2 \
  --gpus 0 1 2 3 4 --resamples 19999 --output build/reanalysis-pilot-cost
```

最後一個命令的 GPU 編號只用於排程估算，不啟動 GPU。載入/warmup log 僅保留原 analyzer 實際讀取的原始行，原 log hash 在 metadata/pilot-log-extraction.json。

IFEval 主矩陣是五模型×兩模式×三 seeds×541=16,230 筆，MTP Off、KV BF16。保留 cap mismatch、scorer/parser audits、missing-final 和 100K replay；不把補測覆寫到主矩陣。敏感度原本另複製的兩份完整模型資料，改成三筆 replacements.jsonl，可從本目錄重建：

```sh
python3 tools/rebuild-cap-sensitivity.py --output /tmp/ifeval-sensitivity-runs
```

再於 quant/ 使用上述 IFEval adapter，將 --runs 改為該新目錄。此分析只比較品質；重用的 timing/attempts 不代表新實驗成本。

本次只用 19 次 resamples 做定向相容性檢查，已確認正式 accuracy、266 列 paired estimates、pilot 原始/加權 paired estimates 與九模型載入成本符合原值。沒有用小次數 CI 替換原 199,999-resample 正式 CI；原正式表保留不動。

## 4. Terminal-Bench：保留的證據與範圍

正式 Low/MTP3 三模型×89 題×三 seeds=801 格，533 格有分數；主分析只用 53 題×三 seeds／模型。Oracle/NOP controls、reward/CTRF/stdout、cleanup、801 matrix、554 attempts、排除/缺分/重試、資源紀錄、report_review 敏感度與原正式 CI 均保留。

554 個 native result 移除 agent_result.rollout_details 與 metadata.all_messages，reward、exception、config、timing 保留。Agentic Fidelity 已保存 BF16 實際 prefix token IDs、sampling、59 tasks/234 windows/7,488 positions 的逐窗口數據、sanity 與結果。完整 ATIF/API、rootfs、工具工作目錄、錄影、pane 不保留；因此涉及工具行為、一般 parser 及完整 wire audit 的檢查需重跑。原要求完整 trajectory 的分析／稽核命令不能直接假設在精簡包上全部通過。

正式來源依 PROTOCOL、AMENDMENT、CONFIG_THREE_SEEDS、各 seed PLAN、runtime audit，不使用歷史 Xhigh/MTP-Off 基礎配置。PRoot 5.5.0 binary、current_exe.c 與已用 shim `.so` 保存；不要任意換成另一個 PRoot 而忽略程式中的 SHA 檢查。PRoot 共享資源且不等於官方 Docker。97.04% 是服務存活率，不是 GPU 計算利用率；CI、29 題 Gate 排除與七格缺分需和主結果一起解讀。

## 5. decision：不需權重即可重建全部報表

**使用 Python 3.12**，與原實驗一致。Python 3.11 的 math 浮點尾差會改變部分同 confidence 值的分組；原完整 metrics/risk–coverage curves 已保存。以下工具會用原 report 程式重建 14 條件的 dev_select/dev_calibrate/test 報表，再重建總摘要及 HTML/SVG，並以 1e-12 容差逐項核對原數據：

```sh
python3.12 tools/rebuild-decision-reports.py --output /tmp/decision-rebuilt
```

輸出目錄必須是新的。這只需標準函式庫，不安裝套件、不載入模型、不重訓。保留的 predictions.jsonl、calibration.json、prepared/sealed、manifest、complete 與 metrics 足以完成此流程；不用移轉重複的 scored_raw/scored_calibrated 或圖檔。

完整實驗則依 decision/experiments/orsharc/EXPERIMENT_B_REPORT.md、PHASE4.md、configs/、實際 training command/args 及 phase4 plans 重建。Elephant runtime 使用 elephant-runtime.lock：Torch 2.8.0/CUDA 12.8、Triton 3.7.1、Transformers 5.19.0；保留原已驗證 override，不讓 resolver 自動替換版本。三種訓練預算均為完整一輪，保留 native loss、complete、checkpoint hashes、readout、ft_exposure；不保存 adapter/merged weights。B4 重用 B1，不能當成獨立重複。

OneJev 的實際本地修改在 metadata/upstream/onejev.json，未追蹤的 readout.py 改放 metadata/onejev-readout.py，避免還原後先產生不完整 checkout 目錄。流程是：先 checkout 固定 OneJev HEAD，再套用相對該 HEAD 的完整保存 diff，最後複製 readout.py 到 qev/。該 diff 已包含 Elephant patch 的修改，不要再次重複套用 patch。原 patch 及 upstream receipts 另保存於 artifacts/upstream/，供驗證來源。其餘 Kev/StartLux/Unsloth 原始 checkout 為乾淨版本，固定 SHA 已保存。

每條件先 dev_select=456 選 threshold，再 dev_calibrate=649 fit temperature，最後 test=2,373。B5 是 StartLux 27B Instant Q4_K_M，不是 Elephant 量化；實際 repo/revision、llama.cpp b11804 commit、image digest、兩段 temperature 與 native API adapter 在 phase4-preflight 及 B5 smoke。Qwen LoRA 與 Elephant 原生量化仍未驗證，不加入正式結果。

## 6. 可重現性的界限

本版保留數據與原始身份，不重寫觀察值。原報告與歷史稽核保留的舊路徑可能指向已省略的過程檔案；應依本文件與 final manifest 判定可直接檢查的內容。完整原始分布、推論軌跡與本地訓練權重需重新執行。固定來源及 seeds 不保證跨 kernel、硬體、網路日期或 Python 版本 bitwise 相同；本次驗證的是包內證據足以重算已聲明的數值，並提供重新執行的方法。
