# 正式 IFEval 輸出上限稽核（2026-10-20）

逐題 JSONL 內的 `sampling.max_tokens` 是實際使用上限。此稽核只統計正式 16,230 筆主矩陣，不含 100K retries。上限在執行中依要求調整，沒有把所有列都當成同一 cap。

## 各模型使用數量

每格是該 model/mode 下以該 cap 執行的題目×seed 數，總計每格 mode 1,623 筆。

| Model | Mode | 4096 | 8192 | 16384 | 20480 | 32768 | 49152 | 65000 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Qwen-Qwen3.8-27B | off | 279 | 445 | 899 | 0 | 0 | 0 | 0 |
| Qwen-Qwen3.8-27B | on | 0 | 0 | 0 | 0 | 723 | 0 | 900 |
| eNeural-Qwen3.8-27B-fast | off | 218 | 354 | 1051 | 0 | 0 | 0 | 0 |
| eNeural-Qwen3.8-27B-fast | on | 0 | 0 | 0 | 1 | 570 | 24 | 1028 |
| rdtand-Qwen3.8-27B-PrismaAQUA-5.5bit-vllm | off | 231 | 453 | 939 | 0 | 0 | 0 | 0 |
| rdtand-Qwen3.8-27B-PrismaAQUA-5.5bit-vllm | on | 0 | 0 | 0 | 0 | 681 | 1 | 941 |
| ModelOpt-Qwen3.8-27B-AutoQuantize-5.5bit | off | 192 | 408 | 1023 | 0 | 0 | 0 | 0 |
| ModelOpt-Qwen3.8-27B-AutoQuantize-5.5bit | on | 0 | 0 | 0 | 7 | 597 | 3 | 1016 |
| unsloth-Qwen3.8-27B-NVFP4 | off | 207 | 411 | 1005 | 0 | 0 | 0 | 0 |
| unsloth-Qwen3.8-27B-NVFP4 | on | 0 | 0 | 0 | 9 | 617 | 1 | 996 |

## 五模型同題同 seed 的 cap 一致性

| Mode | Question×seed cells | 五模型 cap 相同 | cap 不同 | cap 不同且有截斷 | 實際截斷列 |
|---|---:|---:|---:|---:|---:|
| off | 1623 | 1384 | 239 | 0 | 0 |
| on | 1623 | 1453 | 170 | 0 | 3 |

全矩陣只有三筆 `finish_reason=length`，皆在 Thinking 的 65,000 cap；另外所有實際 cap 不同的 question×seed cells 均沒有截斷列。這不等於所有反事實 cap 都可證明無影響，但表示觀測到的截斷只有這三筆；其 100K replay 與敏感度分析另列。

CSV：[`output_cap_histogram_20261020.csv`](output_cap_histogram_20261020.csv)／[`output_cap_pairing_20261020.csv`](output_cap_pairing_20261020.csv)。
