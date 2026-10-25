# Experiment B：簡報洞見

## 一句話主張

在這組 OR-ShARC 條件中，Elephant FT-4,096 單選 BF16 得到最高三分類 benchmark accuracy：**81.0%**。Seen/Unseen accuracy 幾乎相同（80.9%/81.1%）。這是單次固定設定的 benchmark 結果，沒有多 seed 顯著性檢定。

## 可直接使用的重點

### 1. 增加 Elephant 訓練資料後，單選表現逐步上升

| Elephant checkpoint | Test accuracy | Δ All | Δ Seen | Δ Unseen |
|---|---:|---:|---:|---:|
| Original | 73.6% | — | — | — |
| FT-256 | 74.0% | +0.4 pp | +0.0 pp | +0.7 pp |
| FT-1,024 | 76.0% | +2.4 pp | +1.6 pp | +3.0 pp |
| FT-4,096 | 81.0% | +7.4 pp | +8.9 pp | +6.3 pp |

FT-4,096 在 Seen 和 Unseen 都有提升。配圖：[Benchmark test accuracy](artifacts/reports/phase4/test_accuracy_by_condition.svg)。

### 2. Qwen 4-shot 有增益，推論也變慢

Qwen3.5-9B accuracy 從 0-shot 的 69.5% 升至 4-shot 的 74.6%，提高 5.1 個百分點。平均延遲由 47 ms 增至 89 ms，約為 1.9 倍。這兩個條件使用相同 BF16 checkpoint；本輪沒有跑 Qwen LoRA。

### 3. 推論策略會改變成本和表現

- Hierarchical 每筆需兩次請求。Original accuracy 為 73.1%，平均延遲 185 ms；FT-4,096 為 79.6%，平均延遲 189 ms。
- Parallel Noul 每筆一次請求、三個問題。Original accuracy 為 60.5%；FT-4,096 為 75.0%。每個 test 條件仍只有 2,373 次請求，但有 7,119 個問題。
- FT-4,096 的 Parallel Noul 延遲為 100 ms，接近單選的 96 ms；Hierarchical 延遲約為單選的 2 倍。
- Parallel Noul 的 AURC 從 Original 的 0.2014 降到 FT-4,096 的 0.0898，選擇性預測也隨 checkpoint 改善。

### 4. StartLux Q4_K_M 位於準確度與延遲之間

StartLux 27B Instant 的 accuracy 為 77.2%、macro F1 為 0.772，Seen/Unseen 為 77.2%/77.3%。平均延遲 213 ms、p95 260 ms，約 **4.69 examples/s**。它高於 Elephant Original 和 Qwen 4-shot，低於 Elephant FT-4,096；這些數字是描述性比較，模型架構和 runtime 不同。

正式 run 的 GPU0 peak 為 22,350 MiB；同次資源預檢中 StartLux Q4_K_M 峰值為 22,344 MiB，Elephant BF16 為 23,644 MiB，差 5.5%。這是跨模型的等 VRAM 比較；沒有把它當作單一模型的量化精度損失。這些數字只說明這次執行的記憶體使用。

### 5. Temperature scaling 多數情況降低 ECE，但有例外

14 個條件中有 12 個在校準後的 test ECE 較低。Elephant FT-1,024 Choice 從 0.032 升至 0.033；Elephant Original Parallel Noul 從 0.046 升至 0.085。StartLux 從 0.042 降至 0.024；Qwen 4-shot 從 0.116 降至 0.013。Elephant FT-4,096 Choice 的 raw AURC 為 0.0695，低於 Original 的 0.1327。Temperature scaling 保留每筆決策標籤；跨樣本的 confidence 排序可改變，所以報告 raw 和 calibrated AURC。

校準用獨立的 649 筆 `dev_calibrate`。Temperature scaling 保留已凍結的預測標籤，所以 accuracy 不變。配圖：[ECE before/after](artifacts/reports/phase4/test_ece_before_after.svg)與[risk–coverage AURC](artifacts/reports/phase4/test_aurc_before_after.svg)。

### 6. B2 是獨立的 sufficiency 任務

Noul、Choice2、Score5 的 test accuracy 約 80.2–80.5%。Score5 的 calibrated ECE 為 0.020，低於 Noul 的 0.061 和 Choice2 的 0.076。三者的 threshold 由 456 筆 `dev_select` 選擇；Score5 的 threshold 為 0.430。這些是二分類 sufficiency 指標，不要與三分類 benchmark accuracy 合併排名。配圖：[Sufficiency accuracy](artifacts/reports/phase4/sufficiency_accuracy.svg)。

## 建議的簡報順序

1. **範圍與可信度：** 14 個唯一條件；每個條件完整跑 456 筆 threshold selection、649 筆 temperature calibration 和 2,373 筆 test。
2. **主要結果：** 用 FT budget 圖呈現 73.6% → 81.0% 的單選改善，並顯示 Seen/Unseen 接近。
3. **模型與策略：** 對照 Qwen 0/4-shot、StartLux Q4_K_M、Hierarchical 和 Parallel Noul；同時報 accuracy 和 latency。
4. **信心校準：** 用 ECE 圖說明多數條件改善，也列出兩個 ECE 上升的條件。
5. **資源與限制：** 約 2.37 H100 GPU 小時、單 seed、無顯著性檢定、Qwen LoRA 與 Elephant 量化未執行。

## 建議避免的說法

- 不要將 B2 的 binary sufficiency accuracy 和 benchmark 的三分類 accuracy 放在同一排名。
- 不要把不同架構的結果描述成已證明的模型優劣；未做多 seed 或顯著性檢定。
- 不要稱 temperature scaling 提升 accuracy；它保留預測標籤，本輪改善的是多數條件的 confidence calibration。
- 不要將 StartLux Q4_K_M 說成 Elephant 的量化結果。Elephant 原生量化仍未驗證。
- 不要將 GPU sampler 的記憶體常駐小時、API request wall time 和模型訓練 elapsed time 當作同一個計時量。

## 互動檢查

[離線 threshold dashboard](artifacts/reports/phase4/test_threshold_dashboard.html)可選條件、Seen/Unseen、raw/calibrated confidence 和最低信心值。Slider 只供 test-set selective-risk 探索；正式門檻仍來自 `dev_select`。

完整資料與方法見[Experiment B 報告](EXPERIMENT_B_REPORT.md)。
