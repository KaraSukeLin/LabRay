# IFEval 100K 補測逐題比較（2026-10-20）

此檔對照正式 541 題主矩陣與獨立補測；不覆寫 `ifeval_modes_formal_20261017/` 的正式結果。補測固定相同題目、mode、seed、prompt token IDs、sampling、MTP Off、BF16 KV cache；只將 `max_tokens` 設為 100,000、`max_model_len` 設為 131,072。

補測清單共 11 個指定 case，**10 筆保存重跑結果，1 筆自然停止 missing-final case 依使用者指示中止且沒有保存答案**。11 份 worker attempt 記錄中，8 筆 status `ok`、3 筆 status `interrupted`；兩筆 interrupted worker 在停止前已保存前一筆 case 的結果。全部補測 worker wall 合計 **6.359 GPU-hours**（含中止／重複的成本），首次至最後一次結束跨 **1.71 小時**。

## 原先 65K 截斷的三筆

| 模型 | 題目／seed | 主矩陣 tokens／結束原因 | 100K 補測 tokens／結束原因 | reasoning／final tokens | prompt strict |
|---|---|---:|---:|---:|---:|
| NaviQuant Fast | `ifeval/3439` / 20261016 | 65000 / `length` | 80396 / `stop` | 58987 / 21407 | false → true |
| PrismaAQUA | `ifeval/3439` / 20261017 | 65000 / `length` | 67680 / `stop` | 46738 / 20938 | true → true |
| PrismaAQUA | `ifeval/3439` / 20261018 | 65000 / `length` | 65221 / `stop` | 50964 / 14250 | true → true |

三筆均在 100K 之前自然停止，成功抽出 final answer，沒有截斷且 prompt strict 正確。NaviQuant Fast 的 seed 20261016 由主矩陣的 incorrect 變為補測 correct；兩筆 PrismaAQUA 均維持 correct。三筆的 prompt token IDs 都與主矩陣相同。**沒有再提高上限；正式主矩陣仍保留原始 65K 結果。**

## 原先自然停止但沒有 final answer

| 模型 | 題目／seed | 主矩陣／補測 tokens | 結果 |
|---|---|---:|---|
| BF16 | `ifeval/2246` / 20261017 | 28165 / 28165 | 原始輸出、generated token IDs、prompt token IDs 完全相同；仍 missing-final、prompt strict incorrect。 |
| ModelOpt AutoQuantize | `ifeval/1259` / 20261018 | 14319 / 14319 | 原始輸出、generated token IDs、prompt token IDs 完全相同；仍 missing-final、prompt strict incorrect。 |
| ModelOpt AutoQuantize | `ifeval/3439` / 20261018 | 17823 / 17823 | 原始輸出、generated token IDs、prompt token IDs 完全相同；仍 missing-final、prompt strict incorrect。 |
| NaviQuant Fast | `ifeval/3048` / 20261017 | 37635 / 37635 | 原始輸出、generated token IDs、prompt token IDs 完全相同；仍 missing-final、prompt strict incorrect。 |
| PrismaAQUA | `ifeval/3048` / 20261017 | 19716 / 19716 | 原始輸出、generated token IDs、prompt token IDs 完全相同；仍 missing-final、prompt strict incorrect。 |
| Unsloth NVFP4 | `ifeval/1906` / 20261017 | 424 / 424 | 原始輸出、generated token IDs、prompt token IDs 完全相同；仍 missing-final、prompt strict incorrect。 |
| Unsloth NVFP4 | `ifeval/3223` / 20261016 | 31958 / 31958 | 原始輸出、generated token IDs、prompt token IDs 完全相同；仍 missing-final、prompt strict incorrect。 |
| NaviQuant Fast | `ifeval/3048` / 20261018 | 61475 / — | 原始為自然停止但缺 final；補測依使用者指示中止，沒有重跑答案可比較。 |

七筆已完成的自然停止重跑均逐字重現原始輸出及 token IDs，沒有一筆出現「原本有 final、重跑後消失」。唯一未保存的自然停止重跑為 NaviQuant Fast `ifeval/3048` seed 20261018；同題 seed 20261017 重跑也逐字重現原結果。

CSV 含每筆原始／補測 finish reason、輸出長度、截斷、final、prompt strict 及 token-ID 比對；原始全文輸出留在各自的 retry run 資料夾。
