# Qwen3.8-27B：Fidelity、代表性能力 pilot 與正式實驗預算

2026-10-16。PrismaAQUA 與真正的 NVIDIA ModelOpt Weight AutoQuantize 均已完成部署、載入、推論及每模型 **55,296 個計分 tokens** 的 Fidelity。正式清單共九個模型，排除 RadixArk 原版；舊八模型實驗保留於 [build/expanded](build/expanded/REPORT.md)。

**主要結果：NaviQuant Fast 的 KL 與 Flip 比 PrismaAQUA 高；KL 比本次 ModelOpt AutoQuantize 低。Fast 與兩個新方案的 PPL 方向差異，在 18 項校正後均仍不確定。** 這是固定 checkpoint 在 H100 後端上的 Fidelity 比較，不是能力排名，也不是三個量化演算法的普遍排名。

## 正式 IFEval：541 題矩陣已完成（2026-10-20）

五模型 × Thinking On／Off × 三個固定 seeds × 官方 541 題共 **16,230／16,230 筆**；五模型在全部 541 題皆可配對比較。正式答案、進度與完整 CSV／Markdown 分析位於 [正式執行目錄](build/benchmark/ifeval_modes_formal_20261017/) 和 [全量分析目錄](build/benchmark/ifeval_modes_formal_20261017/analysis_full_20261020/)。

| 模型 | Non-thinking：prompt／instruction strict | Thinking：prompt／instruction strict |
|---|---:|---:|
| Qwen BF16 | 80.84%／86.45% | 92.24%／94.72% |
| NaviQuant Fast | 80.78%／86.73% | 91.99%／94.60% |
| PrismaAQUA 5.5-bit | 82.07%／87.41% | 92.48%／94.96% |
| ModelOpt AutoQuantize 5.5-bit | 82.07%／87.65% | 92.54%／94.96% |
| Unsloth NVFP4 | 82.32%／87.65% | 91.81%／94.52% |

以完整題目為 bootstrap cluster，三個 seeds 留在題目內，做 199,999 次 paired bootstrap；300 項預先指定比較家族採 Bonferroni 校正，家族區間為 99.9833%。相對 BF16 的 strict accuracy 家族區間全跨 0。NaviQuant Fast 對 PrismaAQUA 的 prompt strict 差值在 Off 為 −1.29 pp（95% CI −3.02 至 +0.43；校正 CI −4.62 至 +1.87），On 為 −0.49 pp（95% CI −1.60 至 +0.62；校正 CI −2.67 至 +1.54）。Fast 對 ModelOpt 與 Unsloth 的六個模式比較，其校正區間也都跨 0。因此本次 IFEval 沒有確認 NaviQuant Fast 或其他量化模型的 strict accuracy 排名；前一階段 KL／Top-1 Flip 差異也沒有在 IFEval 上顯示可確認的能力差距。

本輪固定官方 IFEval 輸入、BF16 tokenizer／chat template、prompt token IDs、Qwen 官方 On／Off sampling 配方、MTP Off 與 BF16 KV cache。Thinking On 使用 xhigh；Non-thinking 使用其官方 sampling 配方。這是兩套完整推論配方比較，不能把 On−Off 差值解讀成只由 thinking 開關造成。Google 官方評分器固定在 revision `3ef52111c1fabc301880818f5a547570e926a76b`。主矩陣共耗用 **335.087 worker GPU-hours**；首次與最後一次 worker 記錄相隔 **73.95 小時**，包含暫停、SIGINT 後續跑與間隔時間，不代表連續占卡時長。Thinking 每模型輸出長度 p95 約 7.6K–8.4K tokens；主矩陣記有 3 筆 65K 輸出截斷（NaviQuant 1、PrismaAQUA 2）及 8 筆自然停止但沒有 final answer 的結果。

執行期間依使用者要求調整過 `max_tokens`，各逐題列都保留實際值：Non-thinking 用過 4,096／8,192／16,384，Thinking 用過 20,480／32,768／49,152／65,000。每個 mode 有 1,623 個 question×seed cells；五模型 cap 全相同的有 Off 1,384 個、On 1,453 個，另有 239 個 Off 與 170 個 On cells 的 cap 不完全相同。全主矩陣只有上述三列實際以 `finish_reason=length` 撞到 65,000；其他列均在各自上限前自然停止，且 cap 不同的配對 cells 都沒有截斷列。這是設定完全同一性的限制；逐模型分布、配對一致性與截斷統計見[輸出上限稽核](build/benchmark/ifeval_modes_formal_20261017/output_cap_audit_20261020.md)。

補測清單有 11 個指定 case，使用 100K 輸出上限、131,072 context；sampling、MTP Off、BF16 KV cache 均沿用主矩陣。**10 筆保存了補測結果；NaviQuant Fast `ifeval/3048` seed 20261018 依使用者指示中止，沒有保存答案。** 七筆自然停止但 missing-final 的已保存重跑，raw output、generated token IDs、prompt token IDs、finish reason、token 數及評分全部與主矩陣相同，仍 missing-final 且 strict incorrect；沒有「原本有 final、重跑後消失」的案例。

原本三筆 65K 截斷均在 100K 內自然停止，成功抽取 final answer 且 prompt strict 正確：NaviQuant Fast `ifeval/3439` seed 20261016 產生 80,396 tokens（reasoning 58,987、final 21,407），主矩陣 incorrect、補測 correct；PrismaAQUA 同題 seeds 20261017／20261018 分別產生 67,680／65,221 tokens（reasoning／final 為 46,738／20,938 與 50,964／14,250），兩筆均維持 correct。三筆 prompt token IDs 都與原矩陣一致。**100K 已足夠，沒有再提高上限；補測結果未覆寫正式主矩陣。** 詳見[逐題 CSV](build/benchmark/ifeval_modes_formal_20261017/failure_retry_comparison_20261020.csv)及[補測分析](build/benchmark/ifeval_modes_formal_20261017/failure_retry_comparison_20261020.md)。

為量化 cap 對結論的影響，另以三筆 100K replay 替換獨立敏感度資料中的三筆截斷答案，沿用同一分析器、199,999 次題目層級 paired bootstrap 與 300 項比較家族；正式主矩陣和正式分數檔保持不變。NaviQuant Fast Thinking prompt strict 從 91.990% 升至 92.052%（+0.062 pp），instruction strict 從 94.604% 升至 94.644%（+0.040 pp）。Fast 對 PrismaAQUA Thinking prompt strict 差值由 −0.493 pp 變為 −0.431 pp（95% CI −1.540 至 +0.678；99.9833% familywise CI −2.649 至 +1.561）；Fast 對 BF16 為 −0.185 pp（95% CI −1.109 至 +0.801；familywise CI −2.033 至 +1.725）。所有候選模型對 BF16 的 strict accuracy 校正區間仍跨 0，沒有改變「IFEval 不支持確認量化模型排名」的結論。敏感度資料、全量 paired 結果與重現方式見[cap-recovery sensitivity](build/benchmark/ifeval_modes_formal_20261017/cap_recovery_sensitivity_20261020/)。這是品質敏感度分析，該目錄中的 timing 輸出不是實際執行成本。

所有補測共 11 份 worker attempt 記錄，8 份 status `ok`、3 份 `interrupted`；兩個中斷 worker 已先保存前一題的完成結果。補測 worker wall 合計 **6.359 GPU-hours**（含中止與重複執行成本），首次啟動至最後結束跨 **1.71 小時**。主矩陣的 335.087 GPU-hours 另行記錄，不和補測口徑混合。補測完成後五張 H100 均已釋放。

另以逐題固定亂數的 wrapper 重播 Google 官方 CLI 評分器，**30／30 個模型／模式／seed 組合、共 16,230 筆已通過**；official CLI responses、parser replays、engine sampling matches 均為 16,230，跨模型／seed prompt token IDs 相同。未固定逐題亂數的 CLI 重播會在官方字母計數器對 `#`／`!` 使用隨機字母時出現分數差異；既有 scorer 本身未修改。此重播在 CPU 執行，不占用 H100；詳見 [audit.json](build/benchmark/ifeval_modes_formal_20261017/verification_seeded_20261020/audit.json)。

完整結果表、seed 分層、耗時分位數、配對覆蓋、Harmful／Recovery／Answer Flip、每題效應與執行 attempts，見 [`analysis_full_20261020/`](build/benchmark/ifeval_modes_formal_20261017/analysis_full_20261020/)。

**下方日期較早的 IFEval 章節均為當時停止點或暫停點的歷史快照；其部分題數、預估與分數不代表上述完成後的全量結果。**

**原 30 小時預算停止點（2026-10-18）**保存 7,189／16,230 筆、223 題五模型完整配對；當時未完成全部 541 題。使用者後續取消時間限制並授權續跑，因此此數字只代表該次停止快照，不是目前終點。舊 200 題暫停報告和前期估算保留作歷史快照。

**IFEval 研究摘要**：[2026-10-20 一頁摘要](build/benchmark/ifeval_modes_formal_20261017/IFEVAL_SUMMARY_20261020.md)。MMMU-Pro 完整能力 benchmark 尚未開始。先前九模型混合任務 pilot 的[歷史摘要](build/benchmark/pilot_v1/decision_report/MONDAY_SUMMARY.md)／[模型選擇分析](build/benchmark/pilot_v1/decision_report/DECISION_REPORT.md)仍保留。

## IFEval 30 小時預算終局快照（2026-10-18 22:41 UTC）

正式五模型 × 雙模式 × 三 seeds × 541 題矩陣到達 30 小時 wall-time 預算後停止。實際從首次啟動至最後 worker 結束為 **29.961 小時**；五卡 worker ledger 共 **141.813 GPU-hours**，launcher 的 GPU reservation 合計 **143.526 GPU-hours**，都在 150 GPU-hours 上限內。五個最後 worker 因計時器的 SIGINT 中止當時正在執行的題目；已寫入的逐題紀錄已刷新，未完成的生成沒有納入評分。

### 結果與可比較範圍

凍結資料共有 **7,189／16,230 筆已保存回答（44.3%）**。只有 **223／541 題（41.2%）**在全部 5 模型 × 2 模式 × 3 seeds 都有結果；主能力比較每個模型／模式使用相同的 669 筆（223 題 × 3 seeds），共 **6,690 筆配對回答**。另有 499 筆已保存回答來自不完整題目，沒有混進主比較。此交集來自固定題目執行順序和續跑，不是隨機／難度分層樣本；結果只代表這 223 題，不能外推為完整 IFEval 排名。

| 模型 | Off 完成題數：seed 16／17／18 | On 完成題數：seed 16／17／18 | 已保存列 | worker GPU-h | 剩餘 541 題 ETA GPU-h |
|---|---:|---:|---:|---:|---:|
| Qwen BF16 | 280／280／280 | 280／280／279 | 1,679 | 27.697 | 25.6 |
| NaviQuant Fast | 224／224／224 | 224／223／223 | 1,342 | 28.754 | 39.9 |
| PrismaAQUA 5.5-bit | 239／239／239 | 238／238／238 | 1,431 | 28.645 | 35.0 |
| ModelOpt AutoQuantize 5.5-bit | 229／229／229 | 228／228／228 | 1,371 | 28.081 | 38.2 |
| Unsloth NVFP4 | 228／228／228 | 228／227／227 | 1,366 | 28.636 | 37.1 |

停止時還有 9,041 個未保存的模型／模式／seed 評估格（含 6 個待重試失敗格）。按 `progress.csv` 各格已完成工作的 preprocessing＋inference＋scoring 平均秒數直線外推，並為 6 個失敗格各加一題重試，補完 541 題約需 **175.7 GPU-hours**，Non-thinking 約 25.6 GPU-h、Thinking 約 150.1 GPU-h；五卡各固定一模型並行約 **39.9 小時 wall time**。這是單點規劃值，不是信賴區間，未計重新載入成本，Thinking 長尾也可能使實際時間更久；現有 150 GPU-hour 預算只剩約 6.5 GPU-h reservation，無法在原預算內補完。

下表是相同 223 題、每題三 seeds 的 prompt-level 與 instruction-level strict accuracy。百分比以所有 669 個題目-seed 答案計算；完整 seed 分層值另見 `per_seed_accuracy.csv`。

| 模型 | Off prompt／instruction | On prompt／instruction | On−Off prompt／instruction Δ |
|---|---:|---:|---:|
| Qwen BF16 | 79.82%／85.28% | 92.23%／94.78% | +12.41／+9.50 pp |
| NaviQuant Fast | 79.97%／86.23% | 92.08%／94.59% | +12.11／+8.36 pp |
| PrismaAQUA 5.5-bit | 81.17%／86.42% | 93.72%／95.63% | +12.56／+9.21 pp |
| ModelOpt AutoQuantize 5.5-bit | 79.97%／86.04% | 92.53%／94.87% | +12.56／+8.83 pp |
| Unsloth NVFP4 | 81.46%／86.89% | 91.63%／94.30% | +10.16／+7.41 pp |

這裡的 On−Off 是整套官方模式配方差異：On 用 temperature 1.0、top-p 0.95、reasoning effort xhigh；Off 用 temperature 0.7、top-p 0.8 和 presence penalty 1.5。兩模式 sampling 不同，因此不能把差異稱為只由 thinking 開關造成。配對區間支持兩模式配方在這批題目上的 prompt strict accuracy 點估計均提高；仍應保留上述因果限制。

### 相對 BF16 與 NaviQuant 直接比較

分析以題目為 bootstrap 單位，將三 seeds 留在題目內成組重抽；共 199,999 次 resample。主比較家族預先包含 13 個對照 × 7 個指標（accuracy delta、harmful/recovery/answer flip 及 instruction 對應指標），Bonferroni 家族區間為 **99.945%**。下表是 prompt strict accuracy delta；所有 95% 與家族校正區間均跨 0，不能據此排名或宣稱等效。

| 模式／候選（相對 BF16） | Δ pp | 95% paired CI | 99.945% 家族 CI |
|---|---:|---:|---:|
| Off／NaviQuant Fast | +0.15 | [−2.54, +2.84] | [−4.63, +4.93] |
| Off／PrismaAQUA | +1.35 | [−1.05, +3.74] | [−2.84, +5.68] |
| Off／ModelOpt AutoQuantize | +0.15 | [−2.54, +2.84] | [−4.48, +4.93] |
| Off／Unsloth NVFP4 | +1.64 | [−1.35, +4.63] | [−3.59, +7.03] |
| On／NaviQuant Fast | −0.15 | [−1.79, +1.49] | [−2.84, +2.99] |
| On／PrismaAQUA | +1.49 | [−0.15, +3.29] | [−1.35, +4.63] |
| On／ModelOpt AutoQuantize | +0.30 | [−1.35, +1.94] | [−2.54, +3.29] |
| On／Unsloth NVFP4 | −0.60 | [−2.24, +0.90] | [−3.59, +2.09] |

對 BF16 的 harmful／recovery flip 點估計：Off 候選分別落在 **4.93–5.68%／5.53–6.58%**，On 候選落在 **1.20–2.39%／1.64–2.69%**。Off／On 的 discordant answers 都不少，故接近零的 accuracy delta 不表示每一題答案相同。所有 flip 的完整區間與 answer flip 都在 CSV。

使用同一題目及 bootstrap 方法，另做 NaviQuant Fast 對 PrismaAQUA、ModelOpt AutoQuantize、Unsloth 的 6 個直接對照 × 7 指標（42 項）家族分析。Fast−peer 的 prompt strict 點估計在 Off 為 −1.49 至 +0.00 pp，On 為 −1.64 至 +0.45 pp；六個 Bonferroni 校正區間都涵蓋 0。On 的 Fast−Prisma 95% 區間接近負向邊界，但家族校正後跨 0，不能說 Prisma 已確認勝出。結果見 [`direct_fast_contrasts.md`](build/benchmark/ifeval_modes_final_20261018_2241/analysis/direct_fast_contrasts.md)。

### 成本、生成長度與 token cap

以下計時只取 223 題共同面板的實際推論，不含模型載入與 scoring；每格 669 個回答，五卡各自 TP=1。worker GPU-hours 表則包含整個實驗的啟動、推論、評分、重試和中斷前已耗用的工作，兩種口徑不可相加。

| 模型 | Off 推論 GPU-h | On 推論 GPU-h | On／Off | On output tok/s | On p50／p95 單題秒 | On output 均值（reasoning／final） |
|---|---:|---:|---:|---:|---:|---:|
| Qwen BF16 | 2.975 | 17.380 | 5.84× | 21.0 | 41／371 | 1,964（1,656／307） |
| NaviQuant Fast | 4.034 | 23.709 | 5.88× | 16.5 | 64／468 | 2,106（1,770／334） |
| PrismaAQUA 5.5-bit | 3.480 | 19.487 | 5.60× | 17.4 | 51／382 | 1,826（1,499／326） |
| ModelOpt AutoQuantize 5.5-bit | 3.893 | 22.531 | 5.79× | 16.6 | 58／457 | 2,013（1,710／302） |
| Unsloth NVFP4 | 3.972 | 21.342 | 5.37× | 16.3 | 57／477 | 1,871（1,562／308） |

On 的 p95 回答長度約 6.7K–7.7K output tokens，reasoning 平均約 82–85% output；每個模型都有少數長題，最慢一題約 16–36 分鐘。共同面板 On 中 407／3,345 個回答超過 4K tokens；Off 為 1／3,345。五模型、兩模式、三 seeds、所有 7,189 筆已保存回答的 truncation count 均為 **0**。

每列 `max_tokens` 實際值曾隨續跑逐步提高：Off 為 4,096／8,192／16,384，On 為 32,768／49,152／65,000；sampling、seed、reasoning effort、MTP Off、BF16 KV cache 不變。沒有已保存答案觸及 cap，因此本批沒有觀察到 cap 截斷；此歷史仍是解讀成本與跨題生成長度時的限制。

全矩陣有 3 列 `missing_final`，都未截斷且 `finish_reason=stop`：BF16 On `ifeval/2246`（在 223 題共同面板外）、ModelOpt On seed 20261018 `ifeval/1259`、Unsloth On seed 20261017 `ifeval/1906`。按 Google scorer 儲存的 strict 結果，後兩筆在主比較中算錯；為避免將答案結構／抽取問題歸因量化，再共同剔除 `1259`、`1906` 做 221 題敏感度分析。點估計略變，候選相對 BF16 區間結論不變；資料見 `missing_final_sensitivity_*.csv`。

### 驗證與可重用產物

停止後已核對 30 個 progress cell、`progress.json`／`progress.csv` 與逐題 JSONL 的列數相符；7,189 筆 row key 唯一，sampling 與已記錄設定相符，answer/raw-output/strict-instruction 欄位一致，無截斷。既有分析器成功產出 199,999-resample 的結果；其 partial analysis 只對 223 題 complete-case 交集做 paired contrasts。完整 `verify` 命令要求 541 題全矩陣，因此沒有對這份 partial run 假稱完成全量 verifier replay。IFEval 使用 Google canonical revision `3ef52111c1fabc301880818f5a547570e926a76b`。

- [週一摘要](build/benchmark/ifeval_modes_final_20261018_2241/MONDAY_SUMMARY.md)
- [主分析目錄：accuracy、per-seed、timing、paired comparison、question effects、coverage 的 CSV／Markdown](build/benchmark/ifeval_modes_final_20261018_2241/analysis/)
- [NaviQuant Fast 直接比較](build/benchmark/ifeval_modes_final_20261018_2241/analysis/direct_fast_contrasts.md)／[完整 CSV](build/benchmark/ifeval_modes_final_20261018_2241/analysis/direct_fast_contrasts.csv)
- [missing-final 敏感度](build/benchmark/ifeval_modes_final_20261018_2241/analysis/missing_final_sensitivity_accuracy.csv)／[paired 區間](build/benchmark/ifeval_modes_final_20261018_2241/analysis/missing_final_sensitivity_paired.csv)
- [凍結逐題結果、progress.json／CSV、attempts 與 logs](build/benchmark/ifeval_modes_formal_20261017/)

本輪續跑指令（當時由外層 timer 限制在剩餘 wall-time 內）：

```sh
timeout --signal=INT --kill-after=90s 11200s env \
  OMP_NUM_THREADS=8 VLLM_NO_USAGE_STATS=1 HF_HUB_OFFLINE=1 VLLM_USE_V2_MODEL_RUNNER=1 \
  .modelopt-venv/bin/python -u src/ifeval_modes.py run --full --resume \
  --output build/benchmark/ifeval_modes_formal_20261017 --gpus 0 1 2 3 4
```

**目前的研究結論**：這 223 題未顯示 NaviQuant Fast、PrismaAQUA、ModelOpt AutoQuantize 或 Unsloth NVFP4 相對 BF16 有可確認的 IFEval strict accuracy 差異；NaviQuant Fast 相對三個候選也沒有經校正後可確認的勝負。先前 Fidelity 的 KL／Top-1 Flip 差異沒有在這個 IFEval 子集中表現為可確認的能力差距；223 題區間仍寬，不能據此斷言兩者無關。PrismaAQUA On 點估計最高，但不能把它當已確認排名。若需根據 IFEval 做模型選擇，下一步應在一致設定下補完剩餘題目；目前不足以用量化品質指標替代完整能力測試。

## 歷史快照：IFEval 雙模式／三 seeds 200 題暫停分析（2026-10-18 18:28 UTC；已由上方終局快照更新）

在當時暫停快照中，五模型都完成相同 200 題的 Thinking On／Off 與三個 seeds，共 **6,000 筆完整配對回答**。五模型 Non-thinking prompt strict accuracy 為 **79.67–81.33%**，Thinking 為 **91.17–93.50%**。量化候選相對 BF16、以及 Fast 相對三個候選的 accuracy 差異，95% 與擴大多重比較校正後的配對區間均跨零；當時不能確認模型排名，也不能宣稱已證明等效。舊快照保留供追溯，不再代表目前進度。

本節取代原 17:09「Live IFEval run」快照。舊段落完整保存在本次分析目錄的 `REPORT_before_update.md`。下方早期成本、九模型 pilot 與 Fidelity 區段保留各自歷史條件；不能把九模型的 200 題混合任務 pilot，與這次五模型的 200 題完整 IFEval 配對視為同一批實驗。

### 週一報告可直接使用的摘要

在 18:28 快照時可交付五模型 × 雙模式 × 三 seeds × 相同 200 題的比較。當時最明確的觀察是模式配方與成本差異，而非量化模型間的能力排名：Thinking 相對 Non-thinking 增加 **9.83–12.50 個百分點** prompt strict accuracy，配對推論成本約為 **5.30–5.81 倍**。Prisma 的 Thinking 點估計最高，Fast 與 BF16 的 Thinking 都是 92.00%，但候選之間的區間仍重疊且直接差值跨零。此段是歷史快照；目前更新結果見上方終局章節。

現有結果累計占用 **126.254 GPU-hours**；完整配對 6,000 筆本身的推論為 **112.641 GPU-hours**。若沿用實測速度補齊原定 541 題矩陣，一般情境仍需約 **188.35 GPU-hours**，五卡各固定一模型約 **42.8 小時**；這是條件式排程估算，不能在原先剩餘 23.75 GPU-hours 內完成。研究報告可以先交付此 200 題結果與限制，再由使用者決定追加模型、模式、seeds 或題數。

本次 CPU 重評另揭露固定 Google 官方版本的兩題評分限制：`ifeval/1122` 的 `#` 與 `ifeval/1129` 的 `!` 會被官方字母計數器改選為隨機英文字母。既有 runner 按題目固定亂數，保存分數可重現；一般未設種子的 CLI 分數可能不同。主表保留原始評分政策，並另外提供共同排除這兩題的 198 題敏感度分析。

### 1. 資料來源、完成度與比較範圍

基準為當時的 [progress.json](build/benchmark/ifeval_modes_paused_200_20261018/source_snapshot/progress.json) 與 [progress.csv](build/benchmark/ifeval_modes_paused_200_20261018/source_snapshot/progress.csv) 固定副本（原檔位於正式 run 目錄），時間戳 **2026-10-18T18:28:47.530017+00:00**，即台北 **2026-10-19 02:28:47**。兩個進度檔的 30 個 cell 欄位已逐一核對，並與逐題 `questions.jsonl` 重算一致。

- 原定矩陣：5 × 2 × 3 × 541 = **16,230** 筆；目前保存 **6,458 筆（39.79%）**，尚缺 **9,772 筆**。
- 完整共同集合：**200／541 題（36.97%）**，每題皆有 30 筆、合計 **6,000 筆**；每個模型／模式有 600 筆回答、951 次指令檢查，分別來自 200 題與 317 條指令各重複三 seeds。
- 其餘 **458 筆**已保存並可續跑，但沒有混入任何能力或公平成本比較。所有 30 組模型／模式／seed 都至少完成 200 題，尚無一組完成 541 題。
- `progress.csv` 的 `correct/completed` 涵蓋各模型不同題目範圍，不能直接作公平排名；本節的能力分數與配對推論成本全部限制於相同 200 題。

| 模型 | Off 完成題數：seed 16／17／18 | On 完成題數：seed 16／17／18 | 保存列／額外列 | 累計占用 GPU-h |
|---|---|---|---|---|
| Qwen BF16 | 242／241／241 | 241／241／241 | 1447／247 | 24.585 |
| NaviQuant Fast | 200／200／200 | 200／200／200 | 1200／0 | 25.642 |
| PrismaAQUA 5.5-bit | 228／228／228 | 228／227／227 | 1366／166 | 25.533 |
| ModelOpt AutoQuantize 5.5-bit | 201／201／201 | 201／201／201 | 1206／6 | 24.970 |
| Unsloth NVFP4 | 207／207／207 | 207／206／205 | 1239／39 | 25.524 |

三個 seeds 完整值為 **20261016、20261017、20261018**。共同集合依既有固定題目順序累積，另重用已核對的 smoke 題 `ifeval/1508`、`ifeval/3753`；它不是隨機／分層抽取的 200 題，也不是完整 IFEval 分數。為達配對目標，本輪最後補齊 `ifeval/2071`、`ifeval/2078`；長尾 `ifeval/2245` 尚未完整配對，未納入主分析。按完成進度取共同集合仍有題目組成與完成時間選擇偏差，bootstrap 不能消除此偏差。

### 2. 實際推論與評分條件

| 設定 | Non-thinking Off | Thinking On |
|---|---|---|
| temperature／top_p／top_k／min_p | 0.7／0.8／20／0 | 1.0／0.95／20／0 |
| presence_penalty／repetition_penalty | 1.5／1 | 0／1 |
| enable_thinking | false | true |
| reasoning_effort | 不適用 | xhigh，以固定官方 chat template 實際渲染到 prompt |
| 目前後續請求 max_tokens | 16,384 | 65,000 |
| 已記錄的歷史上限 | 4,096／8,192／16,384 | 32,768／49,152／65,000 |

所有模型使用同一 BF16 tokenizer／chat template、相同資料、相同模式內相同 prompt token IDs；Google 官方 IFEval revision 固定為 `3ef52111c1fabc301880818f5a547570e926a76b`。MTP Off、BF16 KV cache、TP=1、每模型一張 H100 80GB HBM3、`max_num_seqs=1`、`enforce_eager=true`、prefix caching Off、context 65,536。硬體紀錄 driver 570.86.10；五張 H100 各自執行一個模型。此設定測的是既有可重現推論路徑，沒有測 MTP／DSpark 加速。

上述是 sampling，並非 greedy。每個 request 都保存實際 `sampling` 與 engine 收到的 `engine_sampling`；6,458 筆逐列匹配，只有上表已記錄的輸出上限歷史不同。所有保存的回答均正常 `stop`；舊上限已自然結束的回答保留，僅截斷回答安排補跑。主表 200 題目前沒有保留 `length` 回答，但不能推論後續 341 題不會截斷。

vLLM `qwen3` parser 分離 reasoning 與 final，Google scorer 僅評 final。`reasoning_tokens` 由 parser 計算，`final_answer_tokens` 另以 tokenizer 編碼；受分隔 token 與重新編碼邊界影響，兩者相加不要求完全等於 output token 數。

### 3. 共同 200 題的 strict accuracy

Prompt strict 要求該回答所有指令均通過；instruction strict 的分母是全部指令。表內為三 seeds 平均／合併比例，沒有把三 seeds 當成 600 個獨立題目。方括號為描述性 **95% 題目 cluster bootstrap CI**。

| 模型 | 模式 | Prompt 正確／600 | Prompt strict % [95% CI] | Instruction 通過／951 | Instruction strict % [95% CI] |
|---|---|---|---|---|---|
| Qwen BF16 | off | 478 | 79.67 [74.67, 84.33] | 810 | 85.17 [81.44, 88.69] |
| Qwen BF16 | on | 552 | 92.00 [88.50, 95.17] | 900 | 94.64 [92.13, 96.83] |
| NaviQuant Fast | off | 480 | 80.00 [75.00, 84.83] | 820 | 86.23 [82.65, 89.59] |
| NaviQuant Fast | on | 552 | 92.00 [88.50, 95.17] | 899 | 94.53 [92.00, 96.76] |
| PrismaAQUA 5.5-bit | off | 486 | 81.00 [76.17, 85.50] | 822 | 86.44 [82.74, 89.87] |
| PrismaAQUA 5.5-bit | on | 561 | 93.50 [90.33, 96.33] | 908 | 95.48 [93.11, 97.52] |
| ModelOpt AutoQuantize 5.5-bit | off | 478 | 79.67 [74.67, 84.50] | 817 | 85.91 [82.25, 89.35] |
| ModelOpt AutoQuantize 5.5-bit | on | 553 | 92.17 [88.67, 95.17] | 900 | 94.64 [92.24, 96.77] |
| Unsloth NVFP4 | off | 488 | 81.33 [76.50, 85.83] | 827 | 86.96 [83.54, 90.17] |
| Unsloth NVFP4 | on | 547 | 91.17 [87.33, 94.50] | 894 | 94.01 [91.41, 96.33] |

Prisma Thinking 為 561／600，BF16／Fast 各為 552／600；這個差別是合計 9 個 seed-question 回答，獨立題目單位仍為 200。Unsloth Off 點估計最高、Thinking 最低；模式之間的表面排序改變，提醒不能只用單一模式評價量化。各表格固定按研究模型順序排列，未按分數選擇排名。

### 4. 相對 BF16 的配對差值、多重比較與 Flip

本次沿用 SciPy percentile paired question cluster bootstrap，**199,999 次**、固定 bootstrap seed 20261016，保留同一題內三個 seeds 與全部指令。prompt 差值是每題三 seeds 的平均差再聚合；instruction 差值按指令總數加權。Harmful 與 Recovery 的分母都是全部配對回答／指令，並核對 **Δaccuracy = Recovery − Harmful**。

既有主分析定義為 8 個候選對 BF16 contrasts＋5 個 On−Off contrasts，乘 7 項指標，共 **91 項**；原始 91 項校正輸出保留。為本報告新增 Fast 對 Prisma／Auto／Unsloth 在兩模式的 6 個直接比較後，整份擴大分析採 **19 × 7 = 133 項 Bonferroni family**，每項名目區間 **99.9624%**。下表與後續直接比較均採較保守的 133 項區間；一般 95% 區間另外列出，不依哪個較有利而切換。7 指標是 prompt Δ、Harmful、Recovery、Answer Flip，及 instruction Δ、Harmful、Recovery。seed 個別分數與敏感度點估計屬描述性結果。校正區間為有限 bootstrap 的近似值；133 項時每端約落在 38 個極端重抽樣的尾部，端點仍有 Monte Carlo 誤差，故不解讀微小端點差異。

差值單位皆為**百分點（pp）**，正值代表候選較高；instruction 欄同時列出 133 項校正 CI。

| 候選 | 模式 | ΔPrompt pp | 95% CI | 133 項校正 CI | ΔInstruction pp [校正 CI] |
|---|---|---|---|---|---|
| NaviQuant Fast | off | +0.33 | [-2.50, +3.17] | [-4.83, +5.67] | +1.05 [-2.62, +5.05] |
| PrismaAQUA 5.5-bit | off | +1.33 | [-1.17, +3.83] | [-3.33, +6.00] | +1.26 [-2.37, +4.98] |
| ModelOpt AutoQuantize 5.5-bit | off | +0.00 | [-2.83, +2.83] | [-5.00, +5.33] | +0.74 [-2.96, +4.60] |
| Unsloth NVFP4 | off | +1.67 | [-1.50, +4.83] | [-4.00, +7.67] | +1.79 [-2.01, +6.12] |
| NaviQuant Fast | on | +0.00 | [-1.67, +1.83] | [-3.00, +3.67] | -0.11 [-2.21, +2.25] |
| PrismaAQUA 5.5-bit | on | +1.50 | [-0.33, +3.50] | [-1.67, +5.17] | +0.84 [-1.32, +3.25] |
| ModelOpt AutoQuantize 5.5-bit | on | +0.17 | [-1.50, +2.00] | [-3.00, +3.50] | +0.00 [-2.77, +2.42] |
| Unsloth NVFP4 | on | -0.83 | [-2.67, +0.83] | [-4.33, +2.17] | -0.63 [-3.50, +1.58] |

![IFEval paired accuracy intervals](build/benchmark/ifeval_modes_paused_200_20261018/paired_accuracy_intervals.png)

[可匯出 PDF](build/benchmark/ifeval_modes_paused_200_20261018/paired_accuracy_intervals.pdf)。藍色為 95% CI，灰色為 133 項校正 CI；此圖只描述目前共同子集。

所有模型差異的區間均跨零，包含一般 95% 區間；因此既不能確認某個候選優於 BF16，也不能把「未顯著」當作等效／非劣性證明。本輪沒有預先設定非劣性 margin，無法補作這類宣稱。

| 候選 | 模式 | Harmful % | Recovery % | Answer Flip % |
|---|---|---|---|---|
| NaviQuant Fast | off | 5.33 | 5.67 | 94.50 |
| PrismaAQUA 5.5-bit | off | 4.67 | 6.00 | 94.67 |
| ModelOpt AutoQuantize 5.5-bit | off | 5.33 | 5.33 | 95.67 |
| Unsloth NVFP4 | off | 4.83 | 6.50 | 94.17 |
| NaviQuant Fast | on | 2.50 | 2.50 | 98.33 |
| PrismaAQUA 5.5-bit | on | 1.33 | 2.83 | 97.83 |
| ModelOpt AutoQuantize 5.5-bit | on | 2.33 | 2.50 | 98.50 |
| Unsloth NVFP4 | on | 2.50 | 1.67 | 97.50 |

Fast Off 相對 BF16 是 32／600 個 Harmful、34／600 個 Recovery，淨增 2／600；Thinking 則各 15／600，淨差為零。相同總 accuracy 不表示逐題完全相同。Answer Flip 是去除首尾空白後的**最終回答字串不同率**；在抽樣、開放文字任務中約 94–99% 很正常，不能視為錯誤率，也不能與 Fidelity 的 next-token Top-1 Flip 混用。完整各項 95%／校正 CI 皆在 CSV，避免只呈現有利指標。

### 5. Thinking／Non-thinking 配方差異與 seed 穩定性

以下 On−Off 以同一模型、同一題與相同 seed 配對。五模型 prompt 與 instruction 差值的 133 項校正區間皆為正；這支持目前固定子集上的**整套 Thinking 配方**有較高 strict accuracy。兩模式 temperature、top_p、presence penalty 與輸出上限不同，不能將全部增益歸因於 reasoning 本身。

| 模型 | ΔPrompt pp | 95% CI | 133 項校正 CI | ΔInstruction pp [校正 CI] | Harmful % | Recovery % | On／Off 推論成本 |
|---|---|---|---|---|---|---|---|
| Qwen BF16 | +12.33 | [+7.50, +17.33] | [+3.67, +21.67] | +9.46 [+3.23, +16.07] | 3.50 | 15.83 | 5.80 |
| NaviQuant Fast | +12.00 | [+7.33, +16.83] | [+3.77, +21.17] | +8.31 [+2.82, +14.08] | 3.00 | 15.00 | 5.69 |
| PrismaAQUA 5.5-bit | +12.50 | [+8.00, +17.17] | [+4.50, +21.07] | +9.04 [+3.66, +14.82] | 2.50 | 15.00 | 5.60 |
| ModelOpt AutoQuantize 5.5-bit | +12.50 | [+8.00, +17.17] | [+4.67, +21.00] | +8.73 [+3.47, +14.48] | 2.50 | 15.00 | 5.81 |
| Unsloth NVFP4 | +9.83 | [+5.00, +14.67] | [+1.33, +18.83] | +7.05 [+1.35, +12.86] | 4.17 | 14.00 | 5.30 |

| 模型 | Off seeds 16／17／18：Prompt % | On seeds 16／17／18：Prompt % | Off seed SD pp | On seed SD pp |
|---|---|---|---|---|
| Qwen BF16 | 80.00／81.00／78.00 | 93.50／92.00／90.50 | 1.53 | 1.50 |
| NaviQuant Fast | 82.00／78.00／80.00 | 91.50／93.50／91.00 | 2.00 | 1.32 |
| PrismaAQUA 5.5-bit | 81.50／82.00／79.50 | 93.50／93.50／93.50 | 1.32 | 0.00 |
| ModelOpt AutoQuantize 5.5-bit | 79.00／78.50／81.50 | 92.50／92.00／92.00 | 1.61 | 0.29 |
| Unsloth NVFP4 | 81.50／80.00／82.50 | 91.00／92.50／90.00 | 1.26 | 1.26 |

seed SD 是三個固定 seeds 的樣本標準差，只描述這三次抽樣；Prisma Thinking 三次合計 accuracy 都是 93.50%，不代表每題回答相同，也不保證其他 seeds 的變異為零。bootstrap 的主要單位仍為題目，沒有用 seed SD／√3 代替題目配對 CI。

### 6. NaviQuant Fast 與先前 Fidelity 結果的整合解讀

以下方向固定為 **Fast 減對象**，Harmful 指對象答對、Fast 答錯；不反轉方向挑選較好敘述。

| 對象 | 模式 | ΔPrompt pp | 95% CI | 133 項校正 CI | Harmful % | Recovery % |
|---|---|---|---|---|---|---|
| PrismaAQUA 5.5-bit | off | -1.00 | [-4.00, +1.83] | [-6.33, +4.17] | 5.67 | 4.67 |
| ModelOpt AutoQuantize 5.5-bit | off | +0.33 | [-2.50, +3.00] | [-5.00, +5.33] | 4.83 | 5.17 |
| Unsloth NVFP4 | off | -1.33 | [-4.67, +1.83] | [-7.50, +4.33] | 6.50 | 5.17 |
| PrismaAQUA 5.5-bit | on | -1.50 | [-3.33, +0.17] | [-5.00, +1.40] | 2.67 | 1.17 |
| ModelOpt AutoQuantize 5.5-bit | on | -0.17 | [-2.33, +1.83] | [-4.17, +3.67] | 2.67 | 2.50 |
| Unsloth NVFP4 | on | +0.83 | [-0.67, +2.50] | [-1.83, +3.83] | 1.17 | 2.00 |

先前 55,296-token Fidelity 中，Prisma 的 KL／Top-1 Flip 為 0.014083／4.275%，Fast 為 0.017938／4.798%，Prisma 較接近 BF16；Fast 的 KL 則優於本次 AutoQuantize 的 0.019988，但兩者 Top-1 Flip 差異在原 18 項校正後不確定。

本次 Prisma 相對 Fast 的 Thinking 點估計高 1.50 pp，方向與較佳 Fidelity 一致，但直接比較 CI 跨零，尚不能認定已轉化為 IFEval 優勢。Fast 與 Auto 的 Thinking 為 92.00%／92.17%，沒有觀察到 Fast 的較低 KL 形成可確認的 accuracy 優勢；Fast 相對 Unsloth 的兩模式差值還會改變正負方向。teacher-forced token 分布距離與長篇抽樣後能否符合指令是不同測量，少數 checkpoint 與固定 200 題不足以建立 KL→能力的普遍關係。

容量方面 Fast／Prisma／Auto／Unsloth 為 23.752／23.604／23.536／23.418 GB，約在相近權重容量下比較，但不同 allocation、activation 路徑與 H100 kernel 仍有差異。本次共同 200 題 Fast 的 On 推論成本為 21.26 GPU-h，Prisma 為 18.02、Auto 為 20.83、Unsloth 為 19.61；Fast 沒有在這個 batch=1 路徑顯示速度優勢，名稱中的 Fast 不能當作此工作負載的速度證據。

依 18:28 快照，當時建議保留五模型 200 題配對作報告主結果；若需縮減矩陣，BF16 為基準，Fast 與 Prisma／Auto 為研究核心對照，Unsloth 為 NVFP4 容量對照。這是歷史研究範圍建議，沒有由未顯著差異刪除模型或宣稱勝負。

### 7. 實測推論成本與生成長度（同一 6,000 筆）

下表時間僅計 `llm.generate` 的實際推論，GPU-h 為 TP=1 的逐題秒數總和／3,600；不含載入、評分及中止重跑成本。每個 model/mode 為 200 題 × 三 seeds、600 回答。吞吐為總 output tokens／總 inference seconds，而非逐題吞吐的平均。

| 模型 | 模式 | 均時 s | p50 s | p95 s | output tokens/s | 推論 GPU-h |
|---|---|---|---|---|---|---|
| Qwen BF16 | off | 16.5 | 11.0 | 54.2 | 20.99 | 2.748 |
| Qwen BF16 | on | 95.6 | 41.1 | 381.0 | 21.02 | 15.931 |
| NaviQuant Fast | off | 22.4 | 13.9 | 69.4 | 16.60 | 3.738 |
| NaviQuant Fast | on | 127.5 | 63.5 | 468.6 | 16.52 | 21.256 |
| PrismaAQUA 5.5-bit | off | 19.3 | 13.2 | 61.4 | 17.41 | 3.216 |
| PrismaAQUA 5.5-bit | on | 108.1 | 50.0 | 392.0 | 17.41 | 18.020 |
| ModelOpt AutoQuantize 5.5-bit | off | 21.5 | 13.9 | 68.8 | 16.53 | 3.588 |
| ModelOpt AutoQuantize 5.5-bit | on | 125.0 | 58.1 | 489.1 | 16.63 | 20.834 |
| Unsloth NVFP4 | off | 22.2 | 14.9 | 69.2 | 16.16 | 3.700 |
| Unsloth NVFP4 | on | 117.7 | 56.1 | 493.2 | 16.27 | 19.610 |

| 模型 | 模式 | Output mean／p95／max tokens | Reasoning mean tokens | Final mean tokens | >4K／>8K／>16K 回答數 |
|---|---|---|---|---|---|
| Qwen BF16 | off | 346／1151／2603 | 0.0 | 345.7 | 0／0／0 |
| Qwen BF16 | on | 2009／7932／25273 | 1690.4 | 317.2 | 77／28／4 |
| NaviQuant Fast | off | 372／1138／10900 | 0.0 | 372.1 | 1／1／0 |
| NaviQuant Fast | on | 2107／7790／30174 | 1762.2 | 343.2 | 81／27／3 |
| PrismaAQUA 5.5-bit | off | 336／1067／1977 | 0.0 | 335.5 | 0／0／0 |
| PrismaAQUA 5.5-bit | on | 1882／6835／18090 | 1545.7 | 334.9 | 76／25／3 |
| ModelOpt AutoQuantize 5.5-bit | off | 356／1127／2582 | 0.0 | 355.6 | 0／0／0 |
| ModelOpt AutoQuantize 5.5-bit | on | 2079／8012／31657 | 1766.1 | 311.5 | 78／29／4 |
| Unsloth NVFP4 | off | 359／1146／3013 | 0.0 | 358.5 | 0／0／0 |
| Unsloth NVFP4 | on | 1915／8011／15843 | 1595.2 | 317.8 | 74／27／0 |

五模型合計 Non-thinking 推論 **16.990 GPU-h**、Thinking **95.650 GPU-h**；Thinking output 共 5,995,117 tokens，其中 reasoning 5,015,756，約 **83.66%**。Final answer 平均約 311–343 tokens，並沒有隨 Thinking 膨脹到同等程度。Thinking 的 3,000 筆中 386 筆 >4K、136 筆 >8K、14 筆 >16K；最慢 5% 回答占 Thinking 推論時間 **29.36%**。因此只用短 smoke 或 p50 推估會明顯低估成本。

BF16 在本路徑約 21 tokens/s，量化候選約 16–17 tokens/s。這是目前 eager、batch=1、固定 kernel 的實測結果，不能外推為高並行 serving 或其他硬體下的量化速度結論。Google 評分在共同 6,000 筆的累計時間為 **1,519.57 秒（0.422 CPU-hours）**，遠低於生成；但它在 worker 內序列執行時也占少量 GPU 配置時間。原始 input/output/final tokens、每 seed 推論成本與最慢回答均另有 CSV。

### 8. 實際已花成本、wall time 與預算剩餘

| 模型 | 保存回答數 | 全部已保存推論 GPU-h | 共同配對推論 GPU-h | worker 累計占用 GPU-h | 累計初始化 s |
|---|---|---|---|---|---|
| Qwen BF16 | 1447 | 24.223 | 18.678 | 24.585 | 309.6 |
| NaviQuant Fast | 1200 | 24.994 | 24.994 | 25.642 | 316.0 |
| PrismaAQUA 5.5-bit | 1366 | 24.319 | 21.235 | 25.533 | 316.4 |
| ModelOpt AutoQuantize 5.5-bit | 1206 | 24.733 | 24.422 | 24.970 | 313.9 |
| Unsloth NVFP4 | 1239 | 24.339 | 23.311 | 25.524 | 285.8 |

已保存 6,458 筆的推論合計 **122.608 GPU-h**，其中共同集合 **112.641 GPU-h**、額外 458 筆 **9.967 GPU-h**。worker 占用合計 **126.254 GPU-h**，比已保存推論多約 **3.646 GPU-h**；差額包含初始化、評分、其他 setup／shutdown、已隔離截斷回答及被中止尚未保存的生成，不能全稱作載入開銷，也沒有假定這些成本為零。

既有六次 launch 的 wall time 相加為 **25.593 小時**，五卡全程預留口徑為 **127.963 GPU-h**；launcher child process wall 總和為 **125.964 GPU-h**。worker attempt 口徑含重用 smoke 的歷史成本，與 launcher 口徑不同，不能相加。各模式成本也不能再各自加一遍相同載入時間。這些是程序時間／資源配置口徑，不是 GPU kernel 持續滿載時數。

依 `progress` 的 worker 占用口徑，以原上限 150 GPU-hours 計，18:28 快照尚餘 **23.746 GPU-hours**；若用五卡全程預留口徑，則剩 **22.037 GPU-hours**，兩種口徑不能混用。當時這份 CPU 分析沒有啟動模型推論；當時也按使用者指示暫停。後續實驗已於 19:34 UTC 恢復，並在 30 小時 wall-time 預算到達前停止，終局數據見上方。

### 9. 補齊 541 題的條件式時間預算（未執行）

一般值按 `progress.csv` 各 model/mode/seed 的 **(541 − completed) × 實測 preprocessing＋inference＋scoring 均時**加總；四個主動中止且尚無回答的 task 仍算待完成，不因 `failed` 欄而漏算。時間只估尚未完成的工作，不重算已有回答。各模型目前觀測題數不同，這裡用於排程預估；公平能力與成本比較仍用共同 200 題。

樂觀／一般／保守採未來均時為目前的 **0.75×／1×／1.5×**，是明示的排程情境，**不是預測信賴區間**。固定順序子集與未完成 q2245 的長尾，可能使實際耗時超出範圍。65K 新上限若實際遇到更長生成，也會增加成本；估值未假設 MTP 加速、改 greedy、改 reasoning effort、增加並行或縮短輸出。

| 模型 | 剩餘 Off 三 seeds GPU-h | 剩餘 On 三 seeds GPU-h | 一般合計 GPU-h／單卡 h | 該模型樂觀／一般／保守 h |
|---|---|---|---|---|
| Qwen BF16 | 4.34 | 25.92 | 30.27 | 22.7／30.3／45.4 |
| NaviQuant Fast | 6.44 | 36.31 | 42.76 | 32.1／42.8／64.1 |
| PrismaAQUA 5.5-bit | 4.91 | 28.75 | 33.66 | 25.2／33.7／50.5 |
| ModelOpt AutoQuantize 5.5-bit | 6.14 | 35.84 | 41.99 | 31.5／42.0／63.0 |
| Unsloth NVFP4 | 6.19 | 33.48 | 39.68 | 29.8／39.7／59.5 |

| 排程口徑 | 樂觀 | 一般 | 保守 |
|---|---|---|---|
| 剩餘總 GPU-hours／單 GPU 順序執行 h | 141.3 | 188.3 | 282.5 |
| 五 H100，各卡固定一模型 wall h | 32.1 | 42.8 | 64.1 |
| 五卡完美平均分工之理論下界 wall h | 28.3 | 37.7 | 56.5 |

瓶頸一般值仍為 NaviQuant Fast 約 **42.8 小時**；即使其他卡先做完，共同配對前緣仍受最慢模型限制。完美平均下界未包含重新載入、切換、同模型 lock 與題目長尾，現有單 worker/model 工具不能直接保證達到。正常部署續跑至少還需一次模型初始化，沒有包括在表內。

剩餘 Off 合計約 **28.03 GPU-h**、On 約 **160.32 GPU-h**；連 Off 三 seeds 全部補齊也超過原剩餘 23.75 GPU-h。若從零依共同 200 題均時外推完整矩陣，單純推論約 **304.69 GPU-h**；加上目前進度估得全程累計約 **314.60 GPU-h**，差異來自觀測集合、重跑與其他開銷，兩者都不是完整實驗的實測值。

當時建議週一報告採用已完成的 200 題配對資料，避免為趕時間靜默降低 sampling／reasoning 或讓不同模型使用不同題數。後續正式執行已擴大到 223 題共同配對；若再追加預算，仍應固定五模型相同題目集合。

### 10. 截斷、缺少 final、主動中止與敏感度分析

目前主動 JSONL 中 **6,458 筆全部 `stop`、截斷 0**，共同 6,000 筆也為 0。這是補測與隔離後的狀態，不代表這輪沒有發生過截斷。

| 回答 | 歷史結果 | 本輪處理與現況 |
|---|---|---|
| Fast Off `ifeval/1265` seed 20261016 | 4,096、8,192 tokens 均 `length`，原 strict 已通過 | 上限 16,384 補跑後在 **10,900 tokens** 正常停止、仍通過；與兩次舊結果的 token prefix 完全相同；此列納入共同 200 題 |
| Prisma On `ifeval/2245` seed 20261016 | 32,768 tokens `length`、無 final | 49,152 上限補跑在 **38,472 tokens** 正常停止且通過；舊 token prefix 相同；該題尚非完整配對，未納主表 |
| Unsloth On `ifeval/2245` seed 20261017 | 49,152 tokens 仍 `length`、無 final | 截斷列備份並自 active JSONL 排除；後續上限已設 65,000，但此列尚未補完，不宣稱已解決 |

q2245 是帶「不可使用逗號」約束的謎題，模型可在 reasoning 中耗用大量 tokens；不能把它的長尾成本當成一般長答案題。為達共同 200 題停止目標，最後補齊 q2071／q2078，q2245 保留為未完成工作。截斷備份與 prefix 核對見 `cap_rerun_audit.json`；同一 Fast 8K 備份出現兩份只算同一歷史回答，沒有重複納入品質／成本表。

三筆正常 `stop` 卻沒有 final answer：

| 模型／題目／seed | Output／reasoning tokens | 是否在共同 200 題 | 處理 |
|---|---:|---|---|
| BF16 On q2246／20261017 | 28,165／28,164 | 否 | 留存額外進度，未納主分析 |
| AutoQuantize On q1259／20261018 | 14,319／14,318 | 是 | final 為空，官方 strict 計錯 |
| Unsloth On q1906／20261017 | 424／423 | 是 | final 為空，官方 strict 計錯 |

共同集合缺 final 為 **2／6,000**，Thinking 內 **2／3,000（0.067%）**。保存資料與 parser 重播均確認 final 為空；不能把 reasoning 填作答案，也不能把這兩筆全部歸因於權重量化。主分析照既定規則保留失敗，另以所有模型共同排除相應整題的方式作敏感度檢查，沒有只替單一模型刪除失敗 seed。

`progress` 的 `failed=4` 對應 BF16 Off q2303 seed17、Prisma On q2245 seed17、Unsloth On q2118 seed18、Unsloth On q2245 seed18，都是排程切換時主動中止、尚未產出保存回答的 task。attempt 紀錄為 23 個 `interrupted`、8 個 `ok`、1 個 `partial`，沒有 `failed` attempt；不能把進度的 4 當成四筆模型推論錯誤，也不能當成已完成錯答。

### 11. 官方評分器限制與正確性驗證

本次直接呼叫未修改 Google CLI 的重評，在 BF16 Off seed16 的 q1122／q1129 重現不一致。最小案例顯示：固定 revision 的 `LetterFrequencyChecker.build_description` 只接受 a–z，對 `#`／`!` 會 `random.choice(string.ascii_letters)`；因此 canonical 輸入中的兩條符號次數限制，實際會改成隨機字母次數。將同一 q1122 回答的 Python random seed 由 0 改成 1，官方 strict 列表由 `[true,false]` 變成 `[true,true]`，足以改變分數。這是已執行的診斷，不是推測。

官方固定來源可直接核對 [LetterFrequencyChecker](build/autoquant/tools/google-research/instruction_following_eval/instructions.py)。既有 `benchmark.score` 在每題先設 `random.seed(int(sha256(id)[:16],16))`，並固定 `langdetect` seed=0；對所有模型／模式／generation seeds 使用相同評分亂數，結果可重現，但**無法修正符號指令的語意**。本次沒有更換官方 revision、沒有修改 checker、沒有把原始分數靜默換成自製符號計數分數。一般未設種子的 CLI 不能當成這兩題的唯一穩定標準值。

為明確區分影響，主表保留 200 題原政策；198 題只共同排除兩個 grader 問題題目，196 題再共同排除 q1259／q1906 缺 final 的整題。以下皆是描述性點估計，不重新以敏感度結果挑選顯著排名。

| 模型 | 200 題 Off／On % | 排除 grader 2 題後：198 題 Off／On % | 再排除 missing-final 整題：196 題 Off／On % |
|---|---|---|---|
| Qwen BF16 | 79.67／92.00 | 79.97／92.42 | 79.76／92.35 |
| NaviQuant Fast | 80.00／92.00 | 80.30／92.42 | 80.10／92.35 |
| PrismaAQUA 5.5-bit | 81.00／93.50 | 81.31／93.77 | 81.12／93.71 |
| ModelOpt AutoQuantize 5.5-bit | 79.67／92.17 | 79.97／92.59 | 79.76／92.69 |
| Unsloth NVFP4 | 81.33／91.17 | 81.65／91.58 | 81.46／91.67 |

移除四題後 Fast−Prisma 為 Off **−1.02 pp**、On **−1.36 pp**；Fast−Auto 為 Off **+0.34 pp**、On **−0.34 pp**；Fast−Unsloth 為 Off **−1.36 pp**、On **+0.68 pp**。相較主表沒有形成大幅新優勢，但這些僅為敏感度點估計；沒有用它們替代主分析 CI 或宣稱排名。兩個 grader 問題即使總分變動不大，仍應在正式完整 IFEval 前明確決定如何報告並固定政策。

驗證紀錄：

1. 原始 `progress.json`／CSV 與 6,458 筆逐題資料、200 題交集、30 cells、GPU-h、三 seeds、保存上限均核對一致；沒有重複 record keys、未完成 JSONL 尾列或非預期題目。
2. 原始實驗記錄的資料、tokenizer／chat template、runner、官方 scorer 檔案 SHA-256 均相符；32 次 attempt 的實際 engine 設定均為 MTP Off、BF16 KV、TP=1。五 checkpoint 的 config SHA 及 31 個權重檔案大小／mtime 符合固定版本 metadata，本輪未重新讀取全部權重做內容 hash。重新分析前後的來源 manifest 保留，未改動原始回答、進度快照、模型或 sampling 配方。
3. 全部 **6,458 筆**以原有 deterministic 官方 library 路徑重播，prompt strict／loose 與每條 instruction 結果全部一致。
4. 對共同 6,000 筆使用既有 `ifeval_modes.py verify`；為與 runner 一致，僅在 CPU CLI 啟動層固定相同每題 Python RNG／langdetect seed，strict／loose 呼叫仍使用原始官方函式。解析器、實際 engine sampling、token decode、重新渲染 prompt 與跨模型／seed token IDs 一併核對。**6,000 筆官方 CLI strict／loose、parser 與 token 核對全部通過。** 最終結果見 `verification_seeded/audit.json`；未設種子 CLI 的差異資料保留供診斷，不能當作通過證據。
5. 針對性 CPU 測試：`tests/test_ifeval_modes.py -k 'question_cluster or seed_repeats or partial_analysis'`，**3 passed、12 deselected**。檢查 SciPy paired resampling 公式、seeds 不能膨脹獨立樣本數，以及所有比較必須使用相同完整題目集合。
6. 既有主分析與補充分析的 accuracy、instruction 分母與 Δ＝Recovery−Harmful 均相同；舊截斷與提高上限後的可比較 token prefix 全部相同。本次沒有改寫推論／評分／bootstrap 系統，沒有新增 GPU 實驗，也沒有跑不相關的完整測試套件。

### 12. 可重現產物、執行方式與尚未完成項目

本次分析獨立保存於 [build/benchmark/ifeval_modes_paused_200_20261018](build/benchmark/ifeval_modes_paused_200_20261018/statistics.json)，不覆寫舊 interim 或正式 Fidelity：

- [執行與驗證紀錄](build/benchmark/ifeval_modes_paused_200_20261018/EXECUTION.md)／[最終完成稽核](build/benchmark/ifeval_modes_paused_200_20261018/completion_audit.json)／[engine 與 checkpoint 核對](build/benchmark/ifeval_modes_paused_200_20261018/engine_checkpoint_audit.json)。
- [主 accuracy CSV](build/benchmark/ifeval_modes_paused_200_20261018/accuracy.csv)／[95% absolute CI](build/benchmark/ifeval_modes_paused_200_20261018/accuracy_intervals.csv)、[逐 seed](build/benchmark/ifeval_modes_paused_200_20261018/paired_seed_details.csv)。
- [原 91 項主比較](build/benchmark/ifeval_modes_paused_200_20261018/paired_comparison.csv)／[擴大 133 項比較與所有 CI](build/benchmark/ifeval_modes_paused_200_20261018/expanded_paired_comparison.csv)、[逐題 effects](build/benchmark/ifeval_modes_paused_200_20261018/question_effects.csv)、[配對 coverage](build/benchmark/ifeval_modes_paused_200_20261018/paired_coverage.csv)。
- [共同題目時間／tokens CSV](build/benchmark/ifeval_modes_paused_200_20261018/paired_timing.csv)／[全部模型成本](build/benchmark/ifeval_modes_paused_200_20261018/model_cost.csv)、[補齊矩陣預算](build/benchmark/ifeval_modes_paused_200_20261018/remaining_budget.csv)、[最慢回答](build/benchmark/ifeval_modes_paused_200_20261018/longest_responses.csv)。原 `timing.csv` 是各模型全部已觀測回答的成本，題目範圍不同，勿混作公平比較。
- [進度與來源稽核](build/benchmark/ifeval_modes_paused_200_20261018/snapshot_audit.json)、[全部保存分數重播](build/benchmark/ifeval_modes_paused_200_20261018/saved_grades_replay.json)、[CLI／parser 稽核](build/benchmark/ifeval_modes_paused_200_20261018/verification_seeded/audit.json)、[截斷補跑 prefix](build/benchmark/ifeval_modes_paused_200_20261018/cap_rerun_audit.json)、[grader 最小重現](build/benchmark/ifeval_modes_paused_200_20261018/grader_randomness_reproduction.json)。
- [敏感度 accuracy](build/benchmark/ifeval_modes_paused_200_20261018/grader_sensitivity_accuracy.csv)／[配對差值](build/benchmark/ifeval_modes_paused_200_20261018/grader_sensitivity_comparison.csv)、[原先僅排除 missing-final 的分析](build/benchmark/ifeval_modes_paused_200_20261018/missing_final_sensitivity.csv)。CSV 同名 Markdown 皆可直接閱覽。
- 本輪另保存包含全部 6,458 筆回答、32 個 attempts、progress 與 experiment 的 [source_snapshot](build/benchmark/ifeval_modes_paused_200_20261018/source_snapshot/experiment.json)，58 個檔案逐檔 SHA 與原始來源一致，見 [manifest](build/benchmark/ifeval_modes_paused_200_20261018/frozen_snapshot_manifest.json)；後續續跑不會改變這份快照。原始逐題回答、token IDs、reasoning／final、時間與 attempts 仍在 [正式 run 目錄](build/benchmark/ifeval_modes_formal_20261017/experiment.json) 的各模型子目錄；`paired_input/` 只是供現有 verifier 使用的固定 200 題衍生視圖，不取代正式 541 題 experiment。

本次主分析已實際執行的 CPU 命令：

```sh
.modelopt-venv/bin/python src/analyze_ifeval_modes.py \
  --runs build/benchmark/ifeval_modes_formal_20261017 \
  --output build/benchmark/ifeval_modes_paused_200_20261018 \
  --allow-partial --resamples 199999
```

分析輸出目錄採 `exist_ok=False` 防止覆寫；若重新執行，請改用新目錄，並將 `--runs` 指到本次保存的 `build/benchmark/ifeval_modes_paused_200_20261018/source_snapshot`，即可固定在這次快照。`audit_snapshot.py`、`supplement_analysis.py`、`grader_sensitivity.py`、`reproduce_grader_randomness.py`、`google_cli_seeded.py`、`verify_seeded_cli.py` 與本報告 renderer 均保存在本次產物目錄，完整記錄這次 CPU audit 的操作；它們不會啟動模型生成。評分來源與邏輯仍為固定 Google repository，官方檔案 SHA 與環境版本在 `experiment.json`，原始上游來源見本報告既有來源區段與 benchmark 使用文件。

**尚未完成：**其餘 341 題完整配對、剩餘 9,772 筆評估、q2245 的 65K 補測、正式 MMMU-Pro 全量及 MTP On 能力比較。本次 200 題分析不能取代這些工作。後續推論保持暫停，等待使用者指定追加範圍與預算。

## Early IFEval cost snapshot（2026-10-17 20:38 UTC，歷史）

正式全量矩陣已啟動。依照調整後順序，當前先跑五模型 Thinking On、seed 20261016；20:38:08 UTC 原始 JSONL 完成數分別為 BF16 129/541、NaviQuant Fast 93/541、PrismaAQUA 108/541、ModelOpt AutoQuantize 93/541、Unsloth NVFP4 97/541（合計520/2,705），其中93題已由五模型共同完成。各模型本 seed 的 ETA 約10.6／16.0／13.6／15.6／14.0小時；瓶頸約16小時，長推理使估計仍有較大不確定性。progress.json於20:37:43記錄615/16,230及19.68 GPU-hours；原始 JSONL 更新至619筆，其中99筆為先前 smoke／pilot。當前正式階段已完成結果無錯誤、無截斷、無 missing final。五個 worker 均仍在生成；20:38 GPU 利用率約22–41%，顯存約68.8 GB／卡。progress-history 最近5／10／15分鐘完成速率為2.19／1.79／2.12筆／分鐘；短窗速率受題目長度影響，不應線性外推整輪耗時。

下表是 20:38:08 UTC 同一批 Thinking On seed 20261016 原始逐題結果的生成成本分布；包含仍在執行的模型累計樣本，不是完整能力結果。

| 模型 | n | 推論時間 p95 / max (s) | 輸出 tokens p95 / max | reasoning tokens p95 / max | final tokens p95 / max |
|---|---:|---:|---:|---:|---:|
| BF16 | 129 | 377 / 759 | 7,860 / 15,848 | 7,015 / 13,948 | 1,254 / 3,684 |
| NaviQuant Fast | 93 | 468 / 1,520 | 7,763 / 24,532 | 6,988 / 21,830 | 1,231 / 2,744 |
| PrismaAQUA | 108 | 358 / 1,024 | 6,121 / 17,569 | 5,441 / 14,286 | 1,461 / 4,242 |
| ModelOpt AutoQuantize | 93 | 397 / 1,102 | 6,661 / 18,439 | 6,090 / 17,195 | 1,095 / 2,513 |
| Unsloth NVFP4 | 97 | 495 / 957 | 8,150 / 15,843 | 7,512 / 13,526 | 1,480 / 3,490 |

慢速可由已完成的長題直接解釋：Unsloth 的至少900字 essay `ifeval/152` 花702.9秒，生成11,669 tokens（其中 reasoning 10,056、final 1,611），未截斷；PrismaAQUA 同題花491.4秒，8,400 tokens，亦未截斷。NaviQuant 的 `ifeval/1300` 另曾花1,520秒、生成24,532 tokens，未截斷。這些長尾題說明短 Smoke 不足以預測 Thinking 正式執行成本。

本段記錄早期成本樣本及當時的排程推估；其 20:38 進度、22:00 軟截止與 seed 分階段續跑方案都已由後續正式矩陣取代。當前狀態以本報告上方「IFEval 雙模式／三 seeds」暫停分析為準。

## 完整模型比較

| 模型 | 權重 GB | PPL | ΔPPL 對 BF16 | 完整 vocabulary KL | Top-1 Flip % |
|---|---:|---:|---:|---:|---:|
| Qwen BF16 | 55.5630 | 3.67269657 | 0 | 0 | 0 |
| Qwen FP8 | 30.8669 | 3.68110552 | +0.00840896 | 0.00427723 | 2.41247 |
| NaviQuant Fast | 23.7520 | 3.71469028 | +0.04199371 | 0.01793785 | 4.79782 |
| NVIDIA ModelOpt NVFP4 | 21.9217 | 3.71269762 | +0.04000105 | 0.01864520 | 5.26440 |
| RedHatAI NVFP4 | 24.6885 | 3.70135964 | +0.02866307 | 0.01541646 | 4.43974 |
| Unsloth NVFP4 | 23.4176 | 3.69764107 | +0.02494450 | 0.01447626 | 4.73633 |
| RadixArk BF16-LMHead | 23.7493 | 3.71602372 | +0.04332716 | 0.01895551 | 4.91175 |
| PrismaAQUA 5.5-bit | 23.6035 | 3.70565699 | +0.03296042 | 0.01408319 | 4.27517 |
| ModelOpt AutoQuantize，5.5-bit target | 23.5360 | 3.72163010 | +0.04893353 | 0.01998751 | 4.84122 |

GB 為十進位、所有 safetensors 權重分片的實際檔案大小，不含 tokenizer。KL 使用自然對數，方向為 BF16→候選；CSV 的 Flip 是 0–1 比例，本表改為百分比。全部九個模型均為 `ok`，每模型 288 份文件、55,296 個計分位置。

- [完整精度、容量、allocation、版本與 95% 區間 CSV](build/autoquant/comparison.csv)／[Markdown](build/autoquant/comparison.md)；[精簡 Fidelity 表格](build/autoquant/analysis/comparison.md)
- [所有模型與分組的 95% 區間](build/autoquant/analysis/intervals.csv)／[分組結果](build/autoquant/analysis/grouped.csv)
- [全部 Fast 配對區間](build/autoquant/analysis/fast_pairwise.csv)／[Markdown](build/autoquant/analysis/fast_pairwise.md)
- [移除單一領域](build/autoquant/analysis/leave_one_domain_out.csv)／[逐文件穩定性](build/autoquant/analysis/document_stability.csv)

## NaviQuant 相對兩個自動量化方案

差值均為 **Fast 減對象**。下表為 Bonferroni 99.7222% 區間，涵蓋六個對象 × 三項主要指標的一個 18 項 family；Flip 差值單位為百分點。

| 對象 | ΔNLL，nats/token | ΔKL，nats/token | ΔFlip，百分點 |
|---|---:|---:|---:|
| PrismaAQUA | +0.002435 [-0.001006, +0.006087] | +0.003855 [+0.003027, +0.004741] | +0.522642 [+0.218009, +0.826461] |
| ModelOpt AutoQuantize | -0.001866 [-0.005044, +0.001367] | -0.002050 [-0.002810, -0.001273] | -0.043403 [-0.295184, +0.213397] |

PrismaAQUA 在略小的檔案容量下，比 Fast 更接近 BF16：KL 與 Flip 的校正區間都支持這個方向。PPL 點估計也較低，但 ΔNLL 區間跨零，不能宣稱其 PPL 優勢已由本次樣本確立。

Fast 的 KL 比這次 ModelOpt 搜尋結果低；PPL 點估計與 Flip 也較低，但後兩者的校正區間跨零。這次 ModelOpt checkpoint 的容量比 Fast 小約 0.216 GB，且搜尋只用八份 sensitivity 文件；結果不能代表 ModelOpt 在更多校準資料或其他 scoring 方法下的最佳品質。沒有依據 Fidelity 分數重新挑選配置。

Fast 的 checkpoint 將 192 個 MLP Linear 放在 NVFP4，其餘 attention Linear 多為 FP8；兩個自動方案重新分配了不同 projection 的容量。相近的平均 bits 不保證相近的誤差。本實驗也未分離 calibration、權重 renderer、activation quantization 與後端 kernel 的各別影響。

![Fast 配對區間](build/autoquant/analysis/fast_intervals.png)

[可匯出 PDF](build/autoquant/analysis/fast_intervals.pdf)。藍色為一般 95% 區間，灰色為 18 項 family 校正區間。

### 多重比較重新核對

沿用 SciPy 1.18.0 的 whole-document paired percentile bootstrap，按 domain × context 固定各層文件數，seed=20261016、19,999 次重抽；文件內 192 個位置不拆開重抽。主要指標是 ΔNLL、ΔKL、ΔFlip；PPL ratio／difference 是衍生描述，不另挑選它們作顯著性宣稱。分組與 leave-one-domain-out 區間仍屬探索性結果。

加入兩個對象後，**全部六個對象一併使用 18 項校正**，沒有讓新增比較沿用較窄的舊 12 項區間。這也改變一項舊結論：Fast−RedHatAI 的 ΔNLL 新區間是 `[-0.00001660, +0.00726280]`，已跨零；舊報告的該項 PPL 方向顯著性不能沿用。RedHatAI 的 KL／Flip 優勢仍獲校正區間支持。

[固定分析範圍](assets/fidelity_extension_analysis.json)／[實際統計參數與程式 SHA](build/autoquant/analysis/statistics.json)。

## 模型版本、容量與實際配置

PrismaAQUA 為 [rdtand/Qwen3.8-27B-PrismaAQUA-5.5bit-vllm](https://huggingface.co/rdtand/Qwen3.8-27B-PrismaAQUA-5.5bit-vllm)，固定 revision `6dc090346f4f32acc320e48d9c413ea96a98d4c6`。五個 safetensors 共 **23,603,517,768 bytes**，已逐檔核對 Hub LFS SHA-256。格式為 `compressed-tensors` mixed precision；實際為 NVFP4 group 16、FP8 E4M3 及 BF16 passthrough。

ModelOpt 使用現有 BF16 `Qwen/Qwen3.8-27B` revision `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`。新權重位於 `/root/model/ModelOpt-Qwen3.8-27B-AutoQuantize-5.5bit/`，三個分片共 **23,536,028,696 bytes**，匯出格式為原生 ModelOpt mixed precision。Prisma 的來源 revision 與此 BF16 相同。

| 模型 | Body Linear 的實際 bits/parameter | NVFP4 Linear 數 | FP8 Linear 數 | BF16 Linear 數 |
|---|---:|---:|---:|---:|
| NaviQuant Fast | 5.548951 | 192 | 208 | 96 |
| PrismaAQUA | 5.500161 | 247 | 223 | 26 |
| ModelOpt AutoQuantize | 5.477986 | 301 | 99 | 96 |

此表按相同 **24,350,556,160 個 language-body Linear 參數**，由實際 safetensors 計算其 weight／scale／相關 buffer bytes；不是把 Linear 數目直接平均。完整 allocation 包含每個 module、logical parameter 數與實際 bytes。

兩個新模型的 Vision 333 個 tensor、MTP 15 個 tensor、embedding 與 lm_head 各一個 tensor，合計 **350 個，均與 BF16 原檔逐 tensor SHA-256 相同**。Fast 的這些元件也相同。各模型沒有缺少原模型所需權重；稽核已正確辨認 `.weight_packed` 的標準改名。

- [容量 CSV](build/autoquant/checkpoints/capacity.csv)／[完整 inventory](build/autoquant/checkpoints/inventory.json)
- [Prisma allocation](build/autoquant/checkpoints/rdtand-Qwen3.8-27B-PrismaAQUA-5.5bit-vllm.json)
- [ModelOpt allocation](build/autoquant/checkpoints/ModelOpt-Qwen3.8-27B-AutoQuantize-5.5bit.json)
- [Prisma 下載與公開 SHA 驗證](build/autoquant/prisma_download.json)／[ModelOpt 分片指紋](build/autoquant/modelopt_fingerprints.json)
- [重驗既有模型的完整檔案指紋](build/autoquant/checkpoints/historical_fingerprints_verified.json)

### 5.5 effective bits 的範圍

ModelOpt 本次使用 `cost_model='weight'`，分母為 **24,352,522,240 個參數**：language-body Linear 加上 1,966,080 個 GDN Conv1d 參數。被停用的 GDN gates／Conv1d 仍按 BF16 計入成本；Vision、MTP、lm_head 明確排除，embedding 與 norm 不屬於此 quantizable-module 分母。

成本是 `sum(n_i × b_i) / sum(n_i)`。NVFP4 的官方成本為 **4.5 bits**（4-bit value 加每 16 個值的 8-bit block scale），FP8 為 8、BF16 為 16。5.5 是上限；實際 solver 結果是 **5.478834**。由實際匯出檔算出的 Linear body bits 略不同，因為分母不含 Conv1d，且實際檔案還有各格式的 scale／buffer 成本。

這個 bits 數字不含完整模型的約 6.856 GB 固定精度元件，不能用 `27B × 5.5 / 8` 推算整個 checkpoint。比較容量時以所有分片的實際 bytes 為準；Prisma、ModelOpt 與 Fast 的這些固定元件已對齊。其他既有模型的元件精度及總容量各異，見 inventory。

## ModelOpt 真正搜尋的執行證據

使用官方 [Model-Optimizer](https://github.com/NVIDIA/Model-Optimizer/tree/ed66a9e0a33db681f0d550f65e7f47f817f6e3c2)，commit `ed66a9e0a33db681f0d550f65e7f47f817f6e3c2`，套件版本 `0.0.1.dev1+ged66a9e0a`，屬固定 commit 的開發版本。入口為 [modelopt_autoquant.py](src/modelopt_autoquant.py)，直接呼叫 `mtq.auto_quantize`。

先以相同 hybrid GDN／full-attention 多模態結構的縮小模型完成 calibration、真正的搜尋與匯出，再在完整 BF16 上完成 forward、validation、calibration、搜尋與匯出。官方 AutoQuantize 的此路徑尚不支援 image-text calibration；本次僅搜尋語言主體，Vision 保持 BF16。原生 exporter 另從來源帶回 Transformers 沒有載入的 15 個 MTP tensor；本機 MTP 與影像推論亦已通過。

| 搜尋設定 | 實際值 |
|---|---|
| 方法 | 官方 `kl_div`，逐組量測，再以 threshold binary search 選配置 |
| 候選 | `NVFP4_DEFAULT_CFG`、`FP8_DEFAULT_CFG`、框架自動加入的 BF16 |
| 目標／實際 | 5.5／5.478834 effective bits |
| Calibration | CNN/DailyMail train，128 份 × 512 tokens，batch=1 |
| Sensitivity | Calibration 中固定前八份 × 512 logits positions |
| 獨立 validation | 另 32 份文件 × 512 tokens，共 16,352 個 next-token 計分位置 |
| 搜尋空間 | 256 個 runtime groups，512 個量化候選；融合相關 Linear 由框架分組 |
| 固定外部 recipe／Navi layer map | 無；`fixed_quantization_config=null`，module search spaces 為空 |
| 保留 BF16 | Vision／MTP／lm_head；GDN Conv1d 與 a/b gates 依官方 base exclusions |
| 搜尋期間 forward | 4,360 次；含兩種格式 calibration 與八批 sensitivity |
| 搜尋時間，含 calibration | 765.792 秒＝12.763 分鐘＝**0.212720 GPU-hours** |
| 載入、validation、搜尋、匯出總時間 | 817.257 秒＝**0.227016 GPU-hours**，一張 H100 |
| 獨立 validation PPL | BF16 8.8229 → fake-quant searched model 9.0297 |

實際選樣及 token IDs 固定於 [calibration](assets/autoquant/calibration.json)、[validation](assets/autoquant/validation.json) 與 [manifest](assets/autoquant/manifest.json)。兩組沒有相同文件 ID、token prefix 或 13-word spans；相對全部 Fidelity input windows 的 13-word overlap 也為零。validation 不參與搜尋，Fidelity 資料不參與校準、選模或 bit 配置。

本次 calibration／sensitivity 分別為 128／8 步，**小於官方 API 建議的 512／128**。搜尋涵蓋全部可搜尋 groups，但沒有宣稱涵蓋官方預設資料量或已達最佳配置。成本是單 GPU 配置的 wall time，不是以 GPU utilization 積分的時間；未計入既有 BF16 訓練成本。

公開 Prisma card 表示 calibration 來自 WikiText train。本次另用其 PPL contract 的 WikiText-2 raw revision，檢查完整 train／validation／test，對固定 Fidelity windows 的 13-word／中文 32-character overlap 均為零；但它沒有完整公開實際 calibration window IDs 與 train revision，因此仍只是來源篩查。[篩查結果](build/autoquant/prisma_calibration_overlap.json)。

[完整參數](build/autoquant/search_v2/parameters.json)／[真實 sensitivities CSV](build/autoquant/search_v2/sensitivities.csv)／[搜尋稽核](build/autoquant/search_v2/search_audit.json)／[搜尋日誌](build/autoquant/logs/search_v2.log)／[validation](build/autoquant/search_v2/validation.json)。

## Fidelity 控制條件與驗證

沿用 [src/fidelity.py](src/fidelity.py)，程式 SHA-256 仍為 `90fcfa567abee03b3af1c90f6f1afcab5e0bccfa5ed3d09f14ca61198a2eb0dd`。固定公開 v2.1 的 tokens.json SHA-256 仍為 `09b41e1a10044975acc579331d3b22ac4ba54f26f89882bb704f6a319d4c2e0d`。

新模型與原實驗使用相同 288 份文件、token IDs、初始 context 128／512／2048、每文件 192 個 scoring positions、teacher forcing、完整 248,320 維 vocabulary、自然對數及 FP64 指標加總。推論維持 H100、TP=1、BF16 dtype／KV cache、MTP Off、seed=0、eager、prefix cache Off、max model length／batched tokens=2304、每次一個序列及相同 FP8 block Triton 設定。

新一輪完整 BF16 gate 與原 288 份分布的 SHA 全相同，KL／Flip／PPL delta 全為零，才執行新模型。既有六個正式候選模型的完整 Fidelity 結果直接重用；BF16 另跑一次 sanity。沒有重存一份大型 BF16 reference。

| 本次 Fidelity 工作 | GPU wall time |
|---|---:|
| 新 BF16 gate | 555.693 秒 |
| PrismaAQUA 全資料 | 583.346 秒 |
| AutoQuantize 全資料 | 573.980 秒 |

H100 的 NVFP4 由 Marlin W4A16 執行，不能解讀為 Blackwell 原生 W4A4。ModelOpt 搜尋時的 NVFP4 fake quant 包含 activation 誤差，而本機該 kernel 使用 BF16 activation；FP8 路徑也可包含 activation quantization。此差異限制對搜尋目標與部署誤差的直接推論。KV cache 始終未量化。

正確性與簡化證據：

- 原 Fidelity 數值核心未改。`extend_fidelity.py` 僅呼叫原 capture，加入完整 gate 與結果重用條件核對。
- `analyze.py`／plot 只讓對象數量可變並明列主要 family。原四對象設定的 **14 個分析 CSV／Markdown 逐位元組相同**：[等價稽核](build/autoquant/analysis_equivalence.json)。
- 官方 saved-score solver 在 5.5 bits 完全重現原配置；只改為 5.0／6.0 bits 時，FP8 Linear 數分別變為 53／153，5.5 為 99。未新增 GPU 搜尋或匯出替代模型：[預算重播消融](build/autoquant/search_v2/replay_ablation.json)。
- 兩個新模型各重跑一份文件／stratum，共 24 份、每模型 2,304 tokens；全部分布 hash 與完整 run 相同。另從原始矩陣獨立重算 NLL、完整 KL、Flip，誤差小於 `1e-7`：[raw-array audit](build/autoquant/distribution_audit/audit.json)。
- safetensors 官方 reader 交叉核對 17,571 個 tensor headers；既有 checkpoint 完整 SHA 重驗通過。
- **13 項 targeted CPU checks 通過**，涵蓋因果位移、完整 KL、文件 bootstrap、paired flips、官方評分、隨機答案 fallback、影像順序及子程序 CUTLASS 匯入。[測試紀錄](build/autoquant/logs/final_targeted_tests.log)。未執行 full test suite。

## IFEval／MMMU-Pro 最終代表性pilot

正式清單九模型排除RadixArk原版。資料與評分器均沿用官方固定來源：Google IFEval GitHub commit `3ef52111c1fabc301880818f5a547570e926a76b`，MMMU-Pro工具commit `268471d0d488258990025331c7528359c324aa25/mmmu-pro`、資料revision `0d7426df4ccb3d8704a992fe6850abee7c262127`。正式母體IFEval541、Standard1730、Vision1730，每模型4001、九模型36009；[資料manifest](assets/benchmark/prepared_v2/manifest.json)。使用Google strict／loose及instruction graders、MMMU作者direct prompt／image mapping／parser／grader；不宣稱逐項重現其示範模型或leaderboard推論設定。

本輪實測條件：[config](assets/benchmark_config.json)，thinking=false、temperature0.7、top_p0.8、top_k20、min_p0、presence_penalty1.5、repetition_penalty1、seed20261016，MTP Off、BF16 KV，TP=1、每GPU單模型逐題、eager、prefix cache Off。context65536，max_images35，影像min65536/max16777216 pixels。主上限IFEval4096、MMMU1024，較長探測4096；原始greedy／1M-pixel啟動smoke另存，未混入。

| 模型 | IFEval strict /78 | Standard 1K→4K /61 | Vision 1K→4K /61 | MMMU 截斷 1K→4K /122 |
| --- | --- | --- | --- | --- |
| Qwen BF16 | 58 | 26→33 | 25→41 | 49→22 |
| Qwen FP8 | 58 | 33→33 | 27→37 | 43→22 |
| NaviQuant Fast | 55 | 26→32 | 25→37 | 51→18 |
| NVIDIA NVFP4 | 61 | 31→35 | 21→36 | 50→19 |
| RedHatAI NVFP4 | 56 | 32→36 | 26→37 | 49→23 |
| Unsloth NVFP4 | 62 | 29→37 | 26→34 | 47→19 |
| RadixArk BF16-LMHead | 57 | 26→36 | 29→33 | 53→22 |
| PrismaAQUA | 61 | 29→34 | 23→36 | 51→19 |
| ModelOpt AutoQuantize | 60 | 28→38 | 29→32 | 58→23 |

每模型200題：78 IFEval、61 Standard、61 Vision，含壓力檢查；分數為原始正確數。IFEval追加為條件抽樣，MMMU每科2個機率樣本，不能把原始比例當完整能力分數。完整流程與統計見[整合分析](build/benchmark/pilot_v1/decision_report/DECISION_REPORT.md)，[CSV](build/benchmark/pilot_v1/decision_report/quality_by_cap.csv)。

MMMU 1K截斷451/1098（41.1%），4K仍187/1098（17.0%），涉及40個題目ID。501探測含451個截斷與50個控制，token prefix全相同，50控制完整tokens與答案相同。IFEval702回答無截斷；8K未完成；背景誤觸的17個回答已隔離，未納入表格。BF16 Vision25→41而Auto29→32，證明輸出上限足以改變表面排序。

![Fidelity與截斷敏感度](build/benchmark/pilot_v1/decision_report/fidelity_and_cap_sensitivity.png)

配對分析保存Accuracy Delta、Harmful／Recovery／Answer Flip及95%／校正區間。IFEval Answer Flip為字串變化，非語意錯誤率。品質bootstrap199999次，成本19999次；按實際抽樣設計加權、層內保持模型配對。IFEval已知38題固定，追加40題估算剩餘503題；MMMU各學科加權，壓力題權重0。

兩cap統一探索性family為8候選×4指標×5不同task/cap單元＝160。每科兩題可造成bootstrap零經驗變異而區間過窄；額外未加權observed-case exact McNemar檢查的Bonferroni p全部為1。不能以無顯著宣稱等價，也不能單靠小樣本bootstrap排名。[1K配對區間](build/benchmark/pilot_v1/paired_1024/paired_comparison.csv)／[4K母體加權區間](build/benchmark/pilot_v1/paired_4096/weighted_paired_comparison.csv)／[exact檢查](build/benchmark/pilot_v1/paired_4096/accuracy_tests.csv)。Fidelity仍使用原先獨立18項family。

## 已花時間與正式實驗預算

| 階段 | 主要回答 | 整體 wall | 實際占用 GPU-h | 五 GPU整組保留 GPU-h |
| --- | --- | --- | --- | --- |
| 核心＋4K探測 | 1,440＋501探測 | 8小時37分42秒 | 36.885 | 43.141 |
| IFEval獨立補樣 | 360 | 24分46秒 | 1.726 | 2.064 |
| 兩階段合計 | 1,800＋501探測 | 9小時02分28秒（執行時間相加） | 38.610 | 45.206 |

初始化平均45.8秒／次，inline官方評分全部合計5.72秒；主瓶頸是501個長探測，Standard平均147秒／回答、Vision168秒。逐題tokens／延遲／throughput與mean/p50/p95/max見[CSV](build/benchmark/pilot_v1/analysis/question_timings.csv)、[主要結果統計](build/benchmark/pilot_v1/analysis/smoke_results.csv)、[載入/推論/評分/wall](build/benchmark/pilot_v1/analysis/model_load_and_wall.csv)。硬體為五張H10080GB、TP1、每次一題；此吞吐量不是最大服務效能。

以下每格是一般GPU-hours，單GPU串行wall同值；分攤每模型一次固定成本。完整MMMU上限1K：

| 模型 | IFEval 541 | Standard 1,730 | Vision 1,730 | 合計 GPU-h | 合計樂觀–保守 |
| --- | --- | --- | --- | --- | --- |
| Qwen BF16 | 1.98 | 14.41 | 16.81 | 33.20 | 31.39–43.76 |
| Qwen FP8 | 2.72 | 18.40 | 24.60 | 45.73 | 43.19–60.34 |
| NaviQuant Fast | 2.48 | 18.65 | 22.04 | 43.17 | 41.22–56.43 |
| NVIDIA NVFP4 | 2.39 | 14.33 | 22.33 | 39.05 | 36.83–51.53 |
| RedHatAI NVFP4 | 2.64 | 19.71 | 21.24 | 43.58 | 41.52–57.04 |
| Unsloth NVFP4 | 2.61 | 16.96 | 20.14 | 39.71 | 37.56–52.30 |
| RadixArk BF16-LMHead | 2.46 | 17.13 | 21.27 | 40.86 | 38.98–53.43 |
| PrismaAQUA | 2.50 | 15.32 | 20.05 | 37.87 | 35.75–49.95 |
| ModelOpt AutoQuantize | 2.76 | 16.06 | 22.12 | 40.93 | 38.92–53.68 |
| 九模型合計 | 22.52 | 150.97 | 190.61 | 364.10 | 345.36–478.46 |

完整MMMU上限4K：

| 模型 | IFEval 541 | Standard 1,730 | Vision 1,730 | 合計 GPU-h | 合計樂觀–保守 |
| --- | --- | --- | --- | --- | --- |
| Qwen BF16 | 1.98 | 28.75 | 38.05 | 68.77 | 61.52–94.90 |
| Qwen FP8 | 2.72 | 36.38 | 57.40 | 96.50 | 86.29–133.14 |
| NaviQuant Fast | 2.48 | 39.44 | 44.55 | 86.47 | 79.36–116.97 |
| NVIDIA NVFP4 | 2.39 | 32.01 | 49.82 | 84.22 | 75.86–115.51 |
| RedHatAI NVFP4 | 2.64 | 39.93 | 50.18 | 92.75 | 83.53–127.43 |
| Unsloth NVFP4 | 2.61 | 34.41 | 43.09 | 80.11 | 71.89–110.24 |
| RadixArk BF16-LMHead | 2.46 | 32.51 | 53.28 | 88.24 | 80.55–119.88 |
| PrismaAQUA | 2.50 | 31.96 | 46.71 | 81.17 | 73.60–110.90 |
| ModelOpt AutoQuantize | 2.76 | 34.03 | 48.96 | 85.75 | 77.59–117.30 |
| 九模型合計 | 22.52 | 309.41 | 432.04 | 763.97 | 690.19–1046.26 |

| MMMU cap | 情境 | 單 GPU wall h＝占用 GPU-h | 五 GPU wall h | 整組保留 GPU-h |
| --- | --- | --- | --- | --- |
| 1024 | optimistic | 345.4 | 77.0 | 384.8 |
| 1024 | typical | 364.1 | 81.0 | 405.2 |
| 1024 | conservative | 478.5 | 106.4 | 531.9 |
| 4096 | optimistic | 690.2 | 153.5 | 767.3 |
| 4096 | typical | 764.0 | 170.0 | 849.8 |
| 4096 | conservative | 1046.3 | 232.5 | 1162.4 |

樂觀取每模型總成本bootstrap2.5%邊界，一般取分層點估計，保守取97.5%邊界加25%規劃餘量；不是工期保證或整體同時95%區間。4K仍截斷、單seed與每科2題皆可能造成未見成本偏差。合計區間先對task抽樣總和取分位數；不加總各task邊際區間。每task／模型三情境詳見[CSV](build/benchmark/pilot_v1/analysis/full_estimates.csv)、[逐GPU排程](build/benchmark/pilot_v1/decision_report/subset_schedules.csv)。

## 建議保留模型與週一排程

保留BF16、Fast、Prisma、Auto四個研究核心；第五個Unsloth增加實用部署比較，或換FP8提供較高精度正控制。NVIDIA如以最小權重容量為目標仍值得保留；RedHatAI與RadixArk此次精簡的優先度較低，沒有證據宣稱其普遍較差。

| 名單 | MMMU cap | 占用 GPU-h／單 GPU h | 五 GPU一般 h | 五 GPU樂觀–保守 h | 假設週日00:00啟動，一般完成時間 |
| --- | --- | --- | --- | --- | --- |
| 研究核心四模型 | 1024 | 155.2 | 43.2 | 41.2–56.4 | 2026-10-19 19:10+08:00 |
| 研究核心四模型 | 4096 | 322.2 | 86.5 | 79.4–117.3 | 2026-10-21 14:28+08:00 |
| 核心＋Unsloth | 1024 | 194.9 | 43.2 | 41.2–56.4 | 2026-10-19 19:10+08:00 |
| 核心＋Unsloth | 4096 | 402.3 | 86.5 | 79.4–117.3 | 2026-10-21 14:28+08:00 |
| 核心＋FP8 | 1024 | 200.9 | 45.7 | 43.2–60.3 | 2026-10-19 21:43+08:00 |
| 核心＋FP8 | 4096 | 418.7 | 96.5 | 86.3–133.1 | 2026-10-22 00:29+08:00 |

週一09:00前可先報告已完成數據，之後補完整能力。若從週日00:00啟動、保留最後3h整理，九模型完整IFEval一般5.1h、保守8h可放入窗口；核心＋Unsloth的IFEval＋Standard1K約21.1h／保守28.3h，雖可排入，但MMMU截斷嚴重，僅能標為受限探索結果。完整4K Standard加IFEval約41.9h，已超過報告時間。建議先補IFEval，再小量確認正式MMMU解碼／cap後安排長實驗。本次沒有啟動這些選項。

五模型內增加模型不一定延長critical path，但增加GPU-hours；六模型會出現第二波。不能把GPU-h/5當成實際可執行工期。未實測MTP加速、較大batch或跨GPU分題，不把假設加速算進預算。按時價R，成本＝占用GPU-h×R，整組保留則用reserved GPU-h×R；未知租價不填金額。

## MTP／DSpark 的證據與限制

九模型MTP=2／DSpark=7共18個startup worker成功；恢復官方解析度後另有33個greedy smoke與非零draft/accepted counters。[相容性矩陣](build/benchmark/compatibility.md)／[官方流程複核](build/benchmark/OFFICIAL_REVIEW.md)。公開model card的特定target支援與本機實測分開記錄。

目前無代表性MTP On工期估計。舊108個Off／On回答使用1M-pixel，不能將其107/197 GPU-h解讀為加速。六個greedy配對有兩個tokens不同，尚未定位；128-token且有fallback的分數相同不證明品質等價。MTP On／Off完整九模型共72018次；先選定模型與Off基準，再單獨作代表性On配對。DSpark暫不增加正式矩陣。

## 正確性、簡化、執行紀錄與使用

官方重評已通過：702個IFEval strict／loose及instruction向量、1599個MMMU回答與保存結果一致。541個canonical IFEval prompts、3460個MMMU prompts/options/gold/subject及3778張影像像素／順序／SHA再核對通過。[完整audit](build/benchmark/pilot_v1/official_replay/audit.json)。重評13.18秒，資料來源／影像稽核190.51秒，未重新推論。

本輪新增分析重用既有估算器與SciPy；wrapper移除未授權的自動8K階段，沒有新增評估framework。17個相關CPU tests通過。等價消融：三task預設公式與原顯式公式抽樣陣列完全相同、96列既有95%paired統計不變、18個原JSONL逐byte保留、九模型joint情境與原估算器一致。[驗證](build/benchmark/pilot_v1/decision_report/analysis_validation.json)／[EOS完整稽核](build/benchmark/pilot_v1/decision_report/termination_audit.json)／[整體完成證據](build/benchmark/pilot_v1/decision_report/decision_audit.json)。未執行full test suite；意外8K背景事件另列於下方。

可重用CPU分析、模型子集／task預算、未執行的正式命令見[README](README.md)及[完整整合分析](build/benchmark/pilot_v1/decision_report/DECISION_REPORT.md)。[原始core](build/benchmark/pilot_v1/runs/launch.json)／[補樣](build/benchmark/pilot_v1/ifeval_supplement_runs/launch.json)／[工作紀錄](build/benchmark/pilot_v1/WORK_LOG.md)。Google IFEval、MMMU作者工具、vLLM均沿用既有Apache-2.0來源及LICENSE；無付費judge，無完整Benchmark能力分數。

### 背景續跑事件

最後檢查發現，先前殘留的舊控制器在本輪合併資料後誤觸8K補測，並非本次CPU分析命令直接啟動。已停止控制器及17個子程序，確認五張GPU已釋放。17個部分輸出及日誌完整隔離，未合併、未重評，不能作8K結果或預算依據；正式核心／IFEval／4K資料未改。額外約26.9分鐘／2.24占用GPU-h，依程序時間粗估，另列不含於正式pilot38.61 GPU-h。[事件證據](build/benchmark/pilot_v1/decision_report/unexpected_8192_stop.json)／[隔離目錄](build/benchmark/pilot_v1/interrupted_unexpected_8192/README.md)。當時完整能力Benchmark尚未開始。


## Thinking／Non-thinking IFEval 工具準備與 pre-run smoke（歷史紀錄）

本節記錄正式全量 launch 前的工具準備與必要 smoke；其 **0/16,230** 狀態及下方「未啟動」文字是啟動前快照，現況以本報告上方「IFEval 雙模式／三 seeds」暫停分析為準。五模型BF16／NaviQuant Fast／PrismaAQUA／ModelOpt AutoQuantize／Unsloth，Thinking On/Off、三相同seeds20261016/17/18的完整541題工具已備妥。啟動前執行兩題×五模型×雙模式×三seeds的60次必要Smoke；不是新增代表性能力結果。

五模型各模式prompt strict6/6、instruction strict9/9；60筆皆無截斷、無missing_final，30筆On均有reasoning及final。Google原版CLI的strict/loose與instruction向量全部一致；實際engine SamplingParams、prompt token IDs與官方qwen3解析重播均通過。On採官方xhigh（模板system instruction），Off及On sampling依Qwen建議；上限Off4096/On32768為本研究配置，不是官方IFEval專用規範。

| 模型 | 載入初始化合計 s | Off 推論 s | On 推論 s | 解析＋評分 s | worker wall s | GPU-h |
|---|---:|---:|---:|---:|---:|---:|
| Qwen BF16 | 100.94 | 4.15 | 128.31 | 2.91 | 254.75 | 0.07076 |
| NaviQuant Fast | 44.76 | 4.99 | 135.02 | 3.08 | 197.53 | 0.05487 |
| PrismaAQUA 5.5-bit | 53.09 | 11.46 | 160.53 | 2.90 | 237.61 | 0.06600 |
| ModelOpt AutoQuantize 5.5-bit | 43.70 | 4.95 | 151.67 | 2.83 | 212.71 | 0.05909 |
| Unsloth NVFP4 | 46.92 | 5.02 | 141.20 | 3.05 | 205.85 | 0.05718 |

主批59筆五H100約239.5秒，另有BF16單筆續跑驗證約59秒；含兩次BF16載入共0.3079占用GPU-h。每題長度、p50/p95、吞吐與完整wall紀錄見[時間CSV](build/benchmark/ifeval_modes_validation/analysis_final/timing.csv)。兩題短樣本不能精確估算正式Thinking成本。

工具逐題append/flush/fsync，續跑跳過已完成的mode/seed/ID；BF16先1後11、完成後再resume皆證實無重算。Timing快照用atomic replace，CPU中斷注入測試通過。CI以題目cluster保留三seeds，13對比×7指標＝91比較family，保存199999次bootstrap的95%及校正區間。兩題全對不代表模型等價。Raw reasoning計分消融顯示30筆On中15筆會與只評final的prompt strict不同，因此正式使用final answer計分。

20個針對性CPU檢查通過；共用引擎／541題prompt等價、原始回覆SHA、官方來源與CLI、final metadata修正的AST消融均有紀錄。當時正式plan已固定最終程式與環境，且在正式 launch 前尚未執行541題矩陣。

[完整使用與啟動指令](build/benchmark/ifeval_modes_validation/USAGE.md)／[Smoke與正確性紀錄](build/benchmark/ifeval_modes_validation/RESULTS.md)／[配對統計](build/benchmark/ifeval_modes_validation/analysis_final/paired_comparison.csv)／[config](assets/ifeval_modes_config.json)。
