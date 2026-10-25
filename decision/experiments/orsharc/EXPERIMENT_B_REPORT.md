# Experiment B：完整執行報告

日期：2026-10-23 UTC

## 摘要

本輪完成 **14 個唯一條件**。每個條件都跑完 `dev_select` 456 筆、`dev_calibrate` 649 筆和官方 test 2,373 筆。合計為 **48,692 個條件－樣本列、55,648 次後端請求、69,560 個問題**。B4 與 B1 完全相同，直接重用 B1 輸出；B3 的單選基線也重用 B1。

主要結果：

- Elephant 單選 BF16 的 test accuracy 從原始模型的 **73.6%**，隨訓練量增加到 FT-4,096 的 **81.0%**。FT-4,096 的 Seen/Unseen accuracy 為 80.9%/81.1%。
- Qwen3.5-9B 從 0-shot 的 **69.5%** 提升到 4-shot 的 **74.6%**；平均單筆延遲由 47 ms 增至 89 ms。
- StartLux 27B Instant Q4_K_M 達 **77.2%**，平均單筆延遲 213 ms。預檢量到約 22.3 GiB GPU 記憶體；同次 Elephant BF16 量到約 23.6 GiB。
- GPU 資源估算約 **2.37 H100 GPU 小時**，低於 8 小時上限。只使用 GPU 0；GPU 1–4 在正式 sampler 期間保持閒置。

完整指標、分組、延遲和 token 數可在[條件總表 CSV](artifacts/reports/phase4/condition_metrics.csv)及[機器可讀摘要](artifacts/reports/phase4/summary.json)中查看。

## 唯一條件矩陣

| 區塊 | 模型／checkpoint | 任務與策略 | Shot | 條件數 | 執行方式 |
|---|---|---|---:|---:|---|
| B1/B4 | Elephant Original、FT-256、FT-1,024、FT-4,096 | Benchmark／Single Choice／BF16 | 0 | 4 | B4 完全重用 B1，不重跑 |
| B2 | Elephant Original | Sufficiency／Noul、Choice2、Score5／BF16 | 0 | 3 | threshold 僅由 `dev_select` 選擇 |
| B3 | Elephant Original、FT-4,096 | Benchmark／Single Choice、Hierarchical、Parallel Noul／BF16 | 0 | 6 cells；4 個新條件 | Original 與 FT-4,096 的兩個 Single Choice cell 重用 B1 |
| Qwen | Qwen3.5-9B BF16 | Benchmark／Single Choice | 0、4 | 2 | 只做推論；未訓練 LoRA |
| B5 | StartLux-Decision-v1.1-27B-Instant Q4_K_M | Benchmark／Single Choice | 0 | 1 | native StartLux API，llama.cpp GPU 0 |
| **總計** |  |  |  | **14** | 三個完整分割各自評估 |

每個條件的分割大小固定為 456/649/2,373。Hierarchical 每筆使用兩次請求；Parallel Noul 將三個問題放在一次請求中。因此總請求數和問題數不同。B2 是二分類 sufficiency 任務；它的 accuracy 不與三分類 benchmark 直接比較。

B3 的完整設計是 **2 個 checkpoint × 3 種策略，共 6 個比較 cell**。其中 Original/FT-4,096 的 Single Choice 已由 B1 完成，所以 B3 只新增 Original/FT-4,096 的 Hierarchical 與 Parallel Noul 四個條件。

## 執行與校準方法

1. Elephant 使用同一個已驗證的 native 訓練 recipe，跑完 256、1,024、4,096 三種完整一輪預算。訓練完成收據記錄有限 loss、完整預算、checkpoint hash 和 rule exposure。B4 重用 B1；沒有重複訓練或重複推論。
2. 每個條件先完成 `dev_select`，再完成 `dev_calibrate`，最後才推論 test。Benchmark threshold 固定為 0.5。B2 的 sufficiency threshold 用 `dev_select` 上 99 個候選值選擇；所有條件的 bounded NLL temperature 都只在 `dev_calibrate` 擬合，範圍為 0.05–20。
3. 原始和校準後的報告使用同一組凍結決策。Temperature scaling 改變機率與信心值，不重新挑選預測標籤。ECE 使用 10 個等寬區間，並以所選 policy label 的機率作為 confidence；Brier 是每筆樣本跨類別機率平方誤差總和的平均。
4. Seen/Unseen 使用官方完整 train split 的 rule membership。訓練預算實際曝光的 rule 另存於 `ft_exposed`，不改寫官方 Seen 標記。
5. 所有正式預測檔都不含 gold label。報告階段才將預測與 gold 對齊計分。既有 verifier 檢查完整 row count、example ID 順序、prediction SHA-256、機率有限且總和為 1、calibration 來源和 Seen/Unseen 筆數。Aggregate builder 也核對 14 個條件在各 partition 使用相同 example ID 順序，並把三個 ID 清單的 SHA-256 寫入 `summary.json`。

StartLux 使用外部 compatibility adapter 呼叫原生 `/v1/systemone` API；adapter 位於 artifacts，沒有更動凍結的 `src`。正式設定將 Choice native temperature 1.0867 套用在 StartLux 輸出上，再於 `dev_calibrate` 擬合 OR-ShARC temperature 1.1847。Adapter SHA-256 為 `7f75599267ed07d5c5cfe607745ef5b4e3fe068954a042e344e207ac0299c260`。llama.cpp 版本為 b11804，runtime image digest 為 `sha256:490d8cd8000203a3ab8fdfce14ef931e2aa85affd4dd4447bcc75f299f00fd9a`。2 筆 train smoke 只檢查無標籤路徑，不納入正式矩陣。

## Test 結果：Benchmark 任務

Accuracy、macro F1、Seen/Unseen 和延遲欄位來自原始決策。Temperature scaling 保留凍結的預測標籤。ECE 欄位列出 raw → calibrated；較低較好。延遲是每筆樣本的平均／p95 秒數。

| 條件 | Accuracy | Macro F1 | Seen / Unseen accuracy | ECE raw → cal | 平均 / p95 秒 | Test requests / questions |
|---|---:|---:|---:|---:|---:|---:|
| Elephant Original · Choice | 73.6% | 0.734 | 72.0% / 74.8% | 0.119 → 0.038 | 0.094 / 0.099 | 2,373 / 2,373 |
| Elephant FT-256 · Choice | 74.0% | 0.738 | 72.0% / 75.5% | 0.104 → 0.041 | 0.095 / 0.101 | 2,373 / 2,373 |
| Elephant FT-1,024 · Choice | 76.0% | 0.757 | 73.6% / 77.8% | 0.032 → 0.033 | 0.095 / 0.100 | 2,373 / 2,373 |
| Elephant FT-4,096 · Choice | 81.0% | 0.810 | 80.9% / 81.1% | 0.032 → 0.028 | 0.096 / 0.101 | 2,373 / 2,373 |
| Elephant Original · Hierarchical | 73.1% | 0.730 | 70.4% / 75.1% | 0.102 → 0.059 | 0.185 / 0.191 | 4,746 / 4,746 |
| Elephant Original · Parallel Noul | 60.5% | 0.594 | 58.6% / 61.9% | 0.046 → 0.085 | 0.099 / 0.105 | 2,373 / 7,119 |
| Elephant FT-4,096 · Hierarchical | 79.6% | 0.796 | 79.4% / 79.7% | 0.040 → 0.031 | 0.189 / 0.194 | 4,746 / 4,746 |
| Elephant FT-4,096 · Parallel Noul | 75.0% | 0.752 | 75.7% / 74.5% | 0.100 → 0.046 | 0.100 / 0.106 | 2,373 / 7,119 |
| Qwen3.5-9B · Choice · 0-shot | 69.5% | 0.688 | 65.4% / 72.5% | 0.089 → 0.053 | 0.047 / 0.053 | 2,373 / 2,373 |
| Qwen3.5-9B · Choice · 4-shot | 74.6% | 0.739 | 69.8% / 78.1% | 0.116 → 0.013 | 0.089 / 0.095 | 2,373 / 2,373 |
| StartLux 27B Instant · Q4_K_M · Choice | 77.2% | 0.772 | 77.2% / 77.3% | 0.042 → 0.024 | 0.213 / 0.260 | 2,373 / 2,373 |

![Benchmark test accuracy by condition](artifacts/reports/phase4/test_accuracy_by_condition.svg)

![Benchmark test ECE before and after dev calibration](artifacts/reports/phase4/test_ece_before_after.svg)

### B5：跨模型等 VRAM 比較

Elephant Original 9B BF16 與 StartLux 27B Instant Q4_K_M 的正式 test 使用同一組 **2,373 個 example IDs**。StartLux accuracy 為 77.2%，比 Elephant Original 的 73.6% 高 **3.6 個百分點**；Seen/Unseen 分別高 5.2/2.5 個百分點。StartLux 平均延遲為 213 ms，約為 Elephant 的 2.27 倍。正式 run 的 GPU0 peak 為 22,350 MiB 與 23,650 MiB，差約 5.5%。

這是不同模型的跨模型比較。它不估算同一模型從 BF16 轉為量化格式的精度損失；本輪沒有執行 StartLux BF16 或 Elephant 量化條件。24 筆 paired smoke 的資源讀值（22,344/23,644 MiB）和原生 runtime 檢查只作為 B5 admission 證據，沒有拿來宣稱正式準確度。

相對 Elephant Original，單選 FT 的 All/Seen/Unseen accuracy gain 為：FT-256 **+0.4/0.0/+0.7 個百分點**、FT-1,024 **+2.4/+1.6/+3.0**、FT-4,096 **+7.4/+8.9/+6.3**。B3 的 FT-4,096 Hierarchical 相對 Original 增加 **+6.4/+9.0/+4.6 個百分點**；Parallel Noul 增加 **+14.5/+17.1/+12.6 個百分點**。這些是同一 test set 上的描述性差值，沒有多 seed 信賴區間。

## Test 結果：Sufficiency 任務

| 條件 | Accuracy | Macro F1 | Seen / Unseen accuracy | ECE raw → cal | Frozen threshold | Temperature | 平均延遲 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Elephant Original · Noul | 80.2% | 0.758 | 78.6% / 81.4% | 0.095 → 0.061 | 0.330 | 1.9651 | 0.093 秒 |
| Elephant Original · Choice2 | 80.5% | 0.753 | 78.0% / 82.2% | 0.106 → 0.076 | 0.250 | 1.9971 | 0.096 秒 |
| Elephant Original · Score5 | 80.2% | 0.760 | 78.5% / 81.5% | 0.021 → 0.020 | 0.430 | 1.0204 | 0.093 秒 |

![Sufficiency test accuracy by condition](artifacts/reports/phase4/sufficiency_accuracy.svg)

## Brier、NLL、risk–coverage、throughput 與 VRAM

以下都是 test 的 All 切片。Brier、NLL 和 AURC 列為 raw → calibrated；AURC 越低，整體 risk–coverage 越好。Throughput 是 `examples / reported request wall seconds`；Parallel Noul 的一個 example 內含三個問題。GPU 記憶體是依 run manifest 的建立時間到 partition completion 時間，從 H100 GPU0 的約 100 ms sampler 取最大 device memory 使用量。

| Benchmark 條件 | Brier raw → cal | NLL raw → cal | AURC raw → cal | Examples/s | GPU0 peak MiB |
|---|---:|---:|---:|---:|---:|
| Elephant Original · Choice | 0.3934 → 0.3714 | 0.7298 → 0.6510 | 0.1327 → 0.1334 | 10.67 | 23,650 |
| Elephant FT-256 · Choice | 0.3848 → 0.3683 | 0.7019 → 0.6464 | 0.1298 → 0.1305 | 10.49 | 23,650 |
| Elephant FT-1,024 · Choice | 0.3341 → 0.3349 | 0.5880 → 0.5929 | 0.1059 → 0.1059 | 10.57 | 23,650 |
| Elephant FT-4,096 · Choice | 0.2774 → 0.2755 | 0.4833 → 0.4818 | 0.0695 → 0.0694 | 10.46 | 23,650 |
| Elephant Original · Hierarchical | 0.3957 → 0.3872 | 0.7349 → 0.6773 | 0.1421 → 0.1426 | 5.41 | 23,644 |
| Elephant Original · Parallel Noul | 0.4159 → 0.4345 | 0.7355 → 0.7535 | 0.2014 → 0.2027 | 10.08 | 23,636 |
| Elephant FT-4,096 · Hierarchical | 0.3090 → 0.3067 | 0.5430 → 0.5403 | 0.0856 → 0.0856 | 5.30 | 23,644 |
| Elephant FT-4,096 · Parallel Noul | 0.3275 → 0.3131 | 0.5773 → 0.5498 | 0.0898 → 0.0898 | 10.00 | 23,636 |
| Qwen3.5-9B · Choice · 0-shot | 0.4397 → 0.4325 | 0.8032 → 0.7548 | 0.1833 → 0.1829 | 21.40 | 17,960 |
| Qwen3.5-9B · Choice · 4-shot | 0.3955 → 0.3707 | 0.7716 → 0.6650 | 0.1292 → 0.1291 | 11.27 | 18,610 |
| StartLux 27B Instant · Q4_K_M · Choice | 0.3174 → 0.3145 | 0.5462 → 0.5410 | 0.0940 → 0.0945 | 4.69 | 22,350 |

| Sufficiency 條件 | Brier raw → cal | NLL raw → cal | AURC raw → cal | Examples/s | GPU0 peak MiB |
|---|---:|---:|---:|---:|---:|
| Elephant Original · Noul | 0.3236 → 0.3141 | 0.5243 → 0.4804 | 0.1115 → 0.1115 | 10.73 | 23,644 |
| Elephant Original · Choice2 | 0.3249 → 0.3151 | 0.5191 → 0.4815 | 0.1086 → 0.1086 | 10.45 | 23,644 |
| Elephant Original · Score5 | 0.2928 → 0.2928 | 0.4512 → 0.4513 | 0.0965 → 0.0965 | 10.71 | 23,650 |

![Benchmark risk–coverage area before and after dev calibration](artifacts/reports/phase4/test_aurc_before_after.svg)

![Sufficiency risk–coverage area before and after dev calibration](artifacts/reports/phase4/sufficiency_aurc_before_after.svg)

Per-run `reports/test/metrics.json` preserves the full risk–coverage curves for All/Seen/Unseen. The [threshold dashboard](artifacts/reports/phase4/test_threshold_dashboard.html) also recomputes accepted count, coverage and selective risk from sample-level raw/calibrated confidences.

## 校準與選擇性預測

大多數條件在 dev temperature fit 後有較低的 test ECE。兩個例外是 FT-1,024 Choice（0.032 → 0.033）和 Elephant Original Parallel Noul（0.046 → 0.085）。這說明 temperature scaling 不保證每個切片都改善校準。決策標籤維持不變，accuracy 也相同。

Benchmark 條件的 calibrated Brier/NLL 也有保存。全條件的 AURC、Brier、NLL、reliability bins、token usage 和成本在[總表](artifacts/reports/phase4/condition_metrics.csv)和各 run 的 `reports/test/metrics.json` 中。

可用[互動 threshold dashboard](artifacts/reports/phase4/test_threshold_dashboard.html)切換條件、Seen/Unseen 分組、raw/calibrated confidence 和最低信心門檻。Dashboard 的 slider 只供檢查 test selective-risk 曲線；正式 threshold 仍是上表中只由 `dev_select` 選出的值。

## 成本、資源與時間

- 矩陣共 **55,648 backend requests、69,560 questions、17,710,008 input tokens**。各條件的延遲、token 和 request 數已寫入 run report 與總表。
- 各 run report 的後端 request wall time 合計 **5,522.7 秒（1.53 小時）**。這是 API request wall time，不等於整個 GPU process reservation。
- Elephant 完整訓練收據的 elapsed time：FT-256 **186.7 秒**、FT-1,024 **436.7 秒**、FT-4,096 **1,468.4 秒**；合計 **34.9 分鐘**。這是整個訓練命令的 elapsed time，包含載入、存檔和 hash 等步驟。
- GPU 0 sampler 從 14:14:49 到 16:33:35 UTC：裝置記憶體使用高於 100 MiB 的時間為 **1.943 小時**，utilization 大於 0 的時間為 1.893 小時，按 sampled utilization 加權為 0.830 小時。GPU 1–4 均未超過 4 MiB，也沒有非零 utilization。
- 12:35 左右的 B5 預檢 receipt 記錄 GPU 0 高於 1 GiB 共 589.51 秒，即 **0.164 H100 小時**。三個 Elephant training receipt 合計 2,091.7 秒；其中約 962.9 秒早於正式 sampler 啟動，約 **0.267 小時**。合併各來源後，Experiment B GPU 0 使用量估算約 **2.37 H100 小時**；這低於 8 小時上限約 5.63 小時。
- StartLux Q4_K_M 預檢峰值為 22,344 MiB；同次 Elephant BF16 為 23,644 MiB，Q4 測值低 5.5%。這是兩個不同模型的峰值記憶體對照，不是精度或速度對照。
- B5 正式 test 平均 213 ms、p95 260 ms；官方 test 2,373 筆用約 506 秒 request wall time。正式 run 結束後 llama.cpp、StartLux API 和 telemetry sampler 都已停止，所有 GPU 回到約 1 MiB、0% utilization。

## 驗證收據與限制

- [Phase 4 總驗證收據](artifacts/smoke/phase4-preflight-20261023T1355Z/verified-phase4-matrix.json)列出 14 個條件與三份完整 Elephant 訓練收據。B5 的[個別 condition verifier 收據](artifacts/smoke/phase4-preflight-20261023T1355Z/verified-b5-startlux.json)確認三個 partition 的完整筆數、prediction hashes、無 gold/label 洩漏、normalized finite probabilities、calibration links 和 raw/calibrated Seen/Unseen counts。
- B1/B2/B3/Qwen 收據位於[Phase 4 preflight 目錄](artifacts/smoke/phase4-preflight-20261023T1355Z/)。三份 Elephant 訓練收據記錄 full budget、finite losses 和 checkpoint hashes。B5 原生 API、模型檔與成對資源量測在[StartLux smoke 證據](artifacts/smoke/b5-startlux-instant-20261023T120642Z/B5_EQUAL_VRAM_SMOKE.md)。
- 每個條件只跑一個固定設定；本報告沒有多 seed 信賴區間或顯著性檢定。結果描述此 benchmark 的觀測值，不外推為其他資料分布的保證。
- Qwen LoRA 仍受 Phase 3 的依賴 gate 阻擋；本輪只跑 Qwen BF16 0/4-shot。Elephant 原生量化路徑也仍未驗證；B5 指的是 StartLux Q4_K_M，而非 Elephant 量化。

## 重現摘要與互動圖生成

原始輸出在 `experiments/orsharc/artifacts/runs/phase4/`。B5 run 位於 `startlux27b-original-single_choice-q4_k_m-0shot-llamacpp/`。完整摘要和圖可從 repo root 重新生成：

```bash
python3 experiments/orsharc/artifacts/reports/phase4/build_summary.py
```

這個命令只讀已完成的 run reports，並重建 artifacts 下的 JSON、CSV、SVG 和離線 HTML dashboard。
