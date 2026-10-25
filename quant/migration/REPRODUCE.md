# quant：校驗與重現

所有命令從本專案目錄執行。本目錄已展開，不需合併資料包或另一個實驗專案。研究檔案與原移轉快照的對照在 `original-files.csv`；目前檔案完整性由 `FILE-MANIFEST.csv` 和 `SHA256SUMS` 記錄，這三份清單都位於本 `migration/` 目錄。

## 校驗與模型小型資產

```sh
sha256sum -c migration/SHA256SUMS
python3 migration/tools/verify-evidence.py
python3 migration/tools/restore-model-metadata.py --output /tmp/quant-model-assets
```

模型設定目錄須是新的或空目錄。此專案的 index 包含 128 個小型檔案，位於 `migration/metadata/model-metadata/index.json`；blob 僅包含此專案使用的資產，相同內容以 SHA-256 去重。模型權重不在 repo 內，依原 repo/revision 取得或依訓練 recipe 重建，再核對權重指紋與本次實際 tokenizer/template/readout。

歷史 manifest 的絕對來源路徑保留不變；新檔案位置以本文件為準。只重算保存的數據可直接在目前 checkout 進行；完整推論若改變模型、資料或程式位置，仍需核對固定 identity 與執行環境。

## 環境、固定輸入與重跑

從 quant/ 讀 REPORT.md；Terminal-Bench 以 build/terminal_bench/FINAL_REPORT.md 指向的 formal_three_seed_20261022/FINAL_REPORT_ZH_TW.md 為最新。舊 README、pre-run plan 或單 seed 中間表不能代替最終狀態。

原環境是 Linux x86_64、Python 3.12、H100 80GB，vLLM 0.31.0+cu129、Torch 2.13.0+cu129、Transformers 5.17.0、compressed-tensors 0.17.0。沿用 requirements、freeze、package provenance 及 scripts/setup_cuda_compat.py、setup_research.sh、setup_terminal_bench.sh，分開重建 vLLM、research overlay、Harbor 環境。原驅動與 CUDA compatibility 設定保存在 environment 與 provenance；新環境仍需先跑 BF16 sanity。

保留 assets/expanded_v2 的固定原文與 token IDs、autoquant calibration/validation、benchmark prepared/pilot inputs、所有正式 configs。MMMU 圖片/parquet、NLTK 可下載資源和 Terminal-Bench task fixtures 不保存；以固定 revision 重建，不重新抽樣。NLTK resource 可依原 setup/prepare 程式重新取得；版本由 freeze 固定。

Terminal-Bench 題目來源在 migration/metadata/public-inputs.json：repo `https://github.com/harbor-framework/terminal-bench-2-1.git`，revision `7131e4375048a0e408a8fb404b5f499d726b695b`。清單記錄 1,035 個正式 task 檔案，加 tasks/README.md、dataset.toml 兩個上層檔案的 SHA-256。完整重跑前，先在全新暫存目錄 checkout 此版本，核對清單，再還原到 quant/build/terminal_bench/sources/terminal-bench-2-1；不要使用省略了 fixtures 的部分 task tree。這避免移轉約 56 MiB 可從公開版本取回的素材。

ModelOpt、MMMU、Harbor 與 Terminal-Bench 的實際 HEAD/origin/乾淨工作目錄檢查保存在 migration/metadata/upstream-verified.json。Google IFEval 是下載的來源目錄，沒有 .git；本包已直接保存實際 scorer 原始碼、LICENSE、revision.json 與原 provenance。MMMU-Pro 的官方 Python 評分器也已補存。需要安裝上游工具時，先在空目錄 checkout 固定 commit，再覆回包內需保存的檔案；本包省略 Git object history。

Fidelity：288 文件、每文件 192 scoring positions，共 55,296 tokens。統計使用文件為 cluster、domain×context 分層、配對 bootstrap，固定 seed/比較家族見 assets/fidelity_extension_analysis.json 與原 statistics.json。逐文件 result、sanity、分布 hashes、配對 CI、搜尋 allocation/sensitivities 與校準稽核皆保存，可直接重算聚合統計。完整 vocabulary `.npy` 與搜尋 `.pt` state 不保存；重查原始分布或重做新模型比較時，需重新推論 BF16 reference。

完整重跑指令仍以原 README-original.md 的 `scripts/run_expanded.sh`、`scripts/run_autoquant_fidelity.sh` 為準，使用新 output。ModelOpt 使用保存的 assets/autoquant、search_v2/parameters.json、allocation.json、sensitivities.csv 重建本地 checkpoint；不能假設 `ModelOpt-Qwen3.8-27B-AutoQuantize-5.5bit` 是可下載的遠端 repo。


## 精簡 benchmark 數據重新分析

本版將精簡的 questions.jsonl/extensions.jsonl 放回原相對位置，並由 source-manifest 標記為轉換資料。移除 prompt、token IDs、reasoning、raw_output、response，保留這些欄位原 canonical JSON 的 SHA-256；另存 token 陣列長度與 response 是否為空。final_answer、answer、原 strict/loose score、sampling、cap、timing、cost 等數據保留。截斷與 missing-final 的完整原始列保存在同目錄 exception-original-records.json。

因此原 analyzer 對 raw 欄位的 assertion 需要換成 hash/count 檢查。本包的 migration/tools/analysis-adapters/ 是對原程式的窄範圍相容調整，各附 `.diff`；統計公式、resampling、家族大小與配對規則未變，原 src/ 也未改。不能把精簡資料宣稱為完整原始輸出。

從 quant/ 執行（先依原環境建立 Python/NumPy/SciPy）：

```sh
export PYTHONPATH="$PWD/src"
.vllm-venv/bin/python migration/tools/analysis-adapters/analyze_ifeval_modes.py \
  --runs build/benchmark/ifeval_modes_formal_20261017 \
  --output build/reanalysis-ifeval --resamples 199999 --family-size 300

.vllm-venv/bin/python migration/tools/analysis-adapters/benchmark_analysis.py \
  --runs build/benchmark/pilot_v1/combined_runs \
  --manifest assets/benchmark/pilot_v2/manifest.json \
  --mmmu-cap 4096 --family-size 160 --resamples 199999 \
  --output build/reanalysis-pilot

.vllm-venv/bin/python migration/tools/analysis-adapters/benchmark_budget.py \
  --runs build/benchmark/pilot_v1/combined_runs --data assets/benchmark/pilot_v2 \
  --gpus 0 1 2 3 4 --resamples 19999 --output build/reanalysis-pilot-cost
```

最後一個命令的 GPU 編號只用於排程估算，不啟動 GPU。載入/warmup log 僅保留原 analyzer 實際讀取的原始行，原 log hash 在 migration/metadata/pilot-log-extraction.json。

IFEval 主矩陣是五模型×兩模式×三 seeds×541=16,230 筆，MTP Off、KV BF16。保留 cap mismatch、scorer/parser audits、missing-final 和 100K replay；不把補測覆寫到主矩陣。敏感度原本另複製的兩份完整模型資料，改成三筆 replacements.jsonl，可從本目錄重建：

```sh
python3 migration/tools/rebuild-cap-sensitivity.py --output /tmp/ifeval-sensitivity-runs
```

再於 quant/ 使用上述 IFEval adapter，將 --runs 改為該新目錄。此分析只比較品質；重用的 timing/attempts 不代表新實驗成本。

本次只用 19 次 resamples 做定向相容性檢查，已確認正式 accuracy、266 列 paired estimates、pilot 原始/加權 paired estimates 與九模型載入成本符合原值。沒有用小次數 CI 替換原 199,999-resample 正式 CI；原正式表保留不動。


## Terminal-Bench 證據與範圍

正式 Low/MTP3 三模型×89 題×三 seeds=801 格，533 格有分數；主分析只用 53 題×三 seeds／模型。Oracle/NOP controls、reward/CTRF/stdout、cleanup、801 matrix、554 attempts、排除/缺分/重試、資源紀錄、report_review 敏感度與原正式 CI 均保留。

554 個 native result 移除 agent_result.rollout_details 與 metadata.all_messages，reward、exception、config、timing 保留。Agentic Fidelity 已保存 BF16 實際 prefix token IDs、sampling、59 tasks/234 windows/7,488 positions 的逐窗口數據、sanity 與結果。完整 ATIF/API、rootfs、工具工作目錄、錄影、pane 不保留；因此涉及工具行為、一般 parser 及完整 wire audit 的檢查需重跑。原要求完整 trajectory 的分析／稽核命令不能直接假設在精簡包上全部通過。

正式來源依 PROTOCOL、AMENDMENT、CONFIG_THREE_SEEDS、各 seed PLAN、runtime audit，不使用歷史 Xhigh/MTP-Off 基礎配置。PRoot 5.5.0 binary、current_exe.c 與已用 shim `.so` 保存；不要任意換成另一個 PRoot 而忽略程式中的 SHA 檢查。PRoot 共享資源且不等於官方 Docker。97.04% 是服務存活率，不是 GPU 計算利用率；CI、29 題 Gate 排除與七格缺分需和主結果一起解讀。


## 重現界限

完整 vocabulary 分布、模型權重及已省略的工具軌跡需重新執行取得。保留的是可重算的數據與原始身份；原報告的部分歷史路徑可能指向未移轉的過程檔案。固定 seeds 不保證跨硬體、kernel、網路日期的 bitwise 結果相同。此版本只重排路徑，研究數據未改寫。
