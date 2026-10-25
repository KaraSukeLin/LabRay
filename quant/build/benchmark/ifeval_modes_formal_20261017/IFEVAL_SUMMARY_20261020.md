# IFEval 研究摘要（2026-10-20）

五個 Qwen3.8-27B 模型已完成官方 IFEval 全量比較：541 題 × 2 種模式 × 3 個固定 seeds，共 **16,230／16,230 筆**；所有模型、模式與 seed 都涵蓋相同 541 題。Google 官方評分器固定在 commit `3ef52111c1fabc301880818f5a547570e926a76b`。完整逐題結果及 paired analysis 位於 [`formal run`](.) 與 [`analysis_full_20261020`](analysis_full_20261020/)；逐題 scorer 重播 audit 30／30 組通過。

| 模型 | Non-thinking prompt／instruction strict | Thinking prompt／instruction strict |
|---|---:|---:|
| Qwen BF16 | 80.84%／86.45% | 92.24%／94.72% |
| NaviQuant Fast | 80.78%／86.73% | 91.99%／94.60% |
| PrismaAQUA 5.5-bit | 82.07%／87.41% | 92.48%／94.96% |
| ModelOpt AutoQuantize 5.5-bit | 82.07%／87.65% | 92.54%／94.96% |
| Unsloth NVFP4 | 82.32%／87.65% | 91.81%／94.52% |

分析以題目為 bootstrap cluster，三個 seeds 留在同一題內；199,999 次 paired resample，300 項預先指定比較使用 Bonferroni 校正。量化模型相對 BF16 的 prompt strict accuracy 校正區間均跨 0；NaviQuant Fast 對 PrismaAQUA／ModelOpt／Unsloth 的 Thinking strict accuracy 區間也跨 0。結果不支持以 IFEval 排出可信的量化模型名次。Thinking 與 Non-thinking 同時改變 sampling／reasoning 配方，兩者差值不能單獨解讀為 thinking 開關的因果效果。

正式矩陣使用 **335.087 worker GPU-hours**；首次至最後一筆 worker 跨 73.95 小時，包含中斷和續跑。100K cap 補測另使用 **6.359 worker GPU-hours**。補測三筆原 65K 截斷均在 100K 內自然完成並正確回答；NaviQuant Fast 的一筆主矩陣 strict incorrect 在補測後轉為 correct，兩筆 PrismaAQUA 分數不變。依既有統計器重算的 cap-recovery sensitivity 顯示 Fast Thinking prompt strict 從 91.990% 升至 92.052%（+0.062 pp），Fast 對 PrismaAQUA 的差值由 −0.493 pp 變為 −0.431 pp，95% CI −1.540 至 +0.678 pp，校正 CI −2.649 至 +1.561 pp；仍不足以確認排名。敏感度資料與 199,999-resample 輸出見 [`cap recovery sensitivity`](cap_recovery_sensitivity_20261020/)。主矩陣正式分數不覆寫。

執行中曾依要求調整 `max_tokens`：Off 4,096／8,192／16,384，On 20,480／32,768／49,152／65,000。相同題目與 seed 的五模型 cap 在 239 個 Off cells、170 個 On cells 不完全相同；除三筆 65K 截斷外，所有輸出都在各自 cap 前自然停止。這是比較設定完全一致性的限制，不能將矩陣描述為每筆都使用相同輸出上限。

逐模型 cap 次數及配對統計見[輸出上限稽核](output_cap_audit_20261020.md)。

七筆自然停止但缺 final 的重跑結果逐字及 token IDs 重現原輸出，仍判錯；NaviQuant Fast `ifeval/3048` seed 20261018 的補測依使用者指示停止，沒有保存答案。這些結果說明 missing-final 是可重現的輸出／解析問題，不能歸因於量化。完整狀態與逐題對照見 [`failure retry CSV`](failure_retry_comparison_20261020.csv) 和 [Markdown](failure_retry_comparison_20261020.md)。

**建議：** IFEval 沒有統計證據支持淘汰 NaviQuant Fast、PrismaAQUA、ModelOpt AutoQuantize 或 Unsloth 中任何一個。若下一階段資源允許，保留 BF16 正控制及四個量化 checkpoint；用 MMMU-Pro 等不同能力面向決定研究名單。正式比較保留 MTP Off、BF16 KV cache；MTP／DSpark 尚未納入這次能力排名。
