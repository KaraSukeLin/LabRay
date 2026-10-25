# Thinking／Non-thinking IFEval：完成與驗證紀錄

已完成可啟動的五模型雙模式三-seed 工具與正式設定清單。正式推論 **0／16,230**；目前僅有下列 60 筆小型 Smoke 結果。五張 H100 已釋放。完整操作方式見 [USAGE.md](USAGE.md)。

## 實測範圍

沿用 canonical Google IFEval 的 `ifeval/1508`（haiku＋必含 keywords＋不得有逗號）及 `ifeval/3753`（算術敘述＋限定回答），每個 model/mode/seed 都使用同兩題：5×2×3×2＝60 筆。這是載入、參數、答案解析、評分與續跑檢查，並非代表性能力 pilot，也不支援精確全量時間預測。

MTP Off、BF16 KV；五張 H100 80GB HBM3，driver 570.86.10，vLLM 0.31.0，TP1／單 request／eager／prefix cache Off。On 用官方 `xhigh`、temperature1/top_p.95；Off temperature.7/top_p.8/presence_penalty1.5；其餘設定與三 seeds 詳見 [config](../../../assets/ifeval_modes_config.json)。On/Off 上限分別32,768/4,096，兩模式皆保留 EOS，沒有自訂 stop string 或強制提早結束 reasoning。

## 逐模型結果

| 模型 | Off／On prompt strict | Off／On instruction strict | 截斷／缺 final |
|---|---:|---:|---:|
| Qwen BF16 | 6/6／6/6 | 9/9／9/9 | 0／0 |
| NaviQuant Fast | 6/6／6/6 | 9/9／9/9 | 0／0 |
| PrismaAQUA 5.5-bit | 6/6／6/6 | 9/9／9/9 | 0／0 |
| ModelOpt AutoQuantize 5.5-bit | 6/6／6/6 | 9/9／9/9 | 0／0 |
| Unsloth NVFP4 | 6/6／6/6 | 9/9／9/9 | 0／0 |

每個模式的6次回答來自**兩題×三seeds**，不是六道獨立題目；instruction分母9亦含三seeds。所有30筆Thinking回答均有非空reasoning及final，Off無reasoning。最高生成1215 tokens，60筆皆自然stop，沒有length截斷或missing_final。這兩題全通過不表示模型等價，也不表示完整541題沒有截斷風險。

## 實際耗時

| 模型 | 載入初始化合計 s | Off 推論 s | On 推論 s | 解析＋評分 s | worker wall s | GPU-h |
|---|---:|---:|---:|---:|---:|---:|
| Qwen BF16 | 100.94 | 4.15 | 128.31 | 2.91 | 254.75 | 0.07076 |
| NaviQuant Fast | 44.76 | 4.99 | 135.02 | 3.08 | 197.53 | 0.05487 |
| PrismaAQUA 5.5-bit | 53.09 | 11.46 | 160.53 | 2.90 | 237.61 | 0.06600 |
| ModelOpt AutoQuantize 5.5-bit | 43.70 | 4.95 | 151.67 | 2.83 | 212.71 | 0.05909 |
| Unsloth NVFP4 | 46.92 | 5.02 | 141.20 | 3.05 | 205.85 | 0.05718 |

主批續跑59筆、五GPU並行 wall **239.54秒（約4分鐘）**。此前另花BF16單筆約59秒驗證保存後停止；因此BF16載入兩次，其餘模型各一次。六次worker wall合計 **0.307906 GPU-h**（包含載入、CPU解析/評分及shutdown的分配時段），與保留整批五GPU的reservation成本不同。主批reservation 0.332690 GPU-h；獨立BF16驗證另列於attempts，不可漏計。最後一次全完成resume只有CPU略過，整批約0.26秒，沒有新增模型載入或回答。

Scoring欄包含官方reasoning parser與Google grader；不是純grader耗時。模型共用初始化成本在按模式成本表平均攤分；原始attempt整體成本另存，兩者不可相加。首次套件初始化、檔案快取及BF16續跑的第二次載入都會影響小樣本成本。

| 模型 | Off output mean／p95 | On output mean／p95 | On 推論 mean／p95 s | On tokens/s |
|---|---:|---:|---:|---:|
| Qwen BF16 | 11.3／17.5 | 426.7／793.0 | 21.38／39.65 | 19.95 |
| NaviQuant Fast | 12.3／19.0 | 385.8／681.5 | 22.50／39.68 | 17.15 |
| PrismaAQUA 5.5-bit | 31.5／97.2 | 466.8／967.5 | 26.75／55.35 | 17.45 |
| ModelOpt AutoQuantize 5.5-bit | 12.2／18.8 | 427.2／1033.8 | 25.28／61.14 | 16.90 |
| Unsloth NVFP4 | 12.3／19.0 | 395.8／726.5 | 23.53／43.10 | 16.82 |

完整mean/p50/p95/max、input/output/reasoning tokens、逐題總處理時間與純推論／整個流程throughput見 [timing.csv](analysis_final/timing.csv)；[逐次載入與wall](analysis_final/attempts.csv)、[整體統計](analysis_final/statistics.json)。所有逐題原始值保留於 [Smoke目錄](../ifeval_modes_smoke_v1/)。官方CLI重評與parser/token/參數複核為額外CPU作業，wall 47.73秒，不計入GPU推論成本。

正式Thinking成本不能用這兩道短題外推為精確預測。先前代表性pilot只有non-thinking/單seed，亦不能直接當本次Thinking三seed的實測預算。正式跑時可持續使用status與allow-partial成本表更新進度；本階段沒有為了預測成本擴大GPU樣本。

## 正確性與簡化證據

- [官方來源稽核](upstream_audit.json)：Google五份原碼及canonical資料與固定官方commit逐byte一致。
- [541題template稽核](template_audit.json)：全部Off prompts與舊runner一致；全部On顯式xhigh與本機官方預設一致；核對已安裝vLLM request確實將effort送到template。xhigh是soft instruction，不是token budget。
- [官方CLI重評](official_replay/audit.json)：60筆strict/loose與各instruction向量完全一致；60筆官方parser重播、實際prompt token IDs與engine SamplingParams核對通過。參數在vLLM process_inputs驗證及generation-config處理後再記錄，沒有只依config聲稱生效。
- [reasoning計分消融](reasoning_scoring_ablation.csv)：30筆On中，15筆若直接把raw reasoning＋final送評，prompt strict會不同；只評final的版本通過官方CLI。解析缺final的情形另外以CPU測試驗證，不會改把reasoning當答案。
- [續跑驗證](validation.json)：BF16先存1筆、停止後只補其餘11筆；原始prefix逐byte相同。全矩陣再resume時，五個JSONL SHA完全不變、attempt數仍6、沒有新增模型載入。使用不同seed要求resume會[明確拒絕](resume_mismatch_check.json)。JSONL torn尾行另外以CPU中斷案例測試，殘片保留、已完成行不動。
- [targeted_tests.log](targeted_tests.log)：20 passed, 14 warnings in 9.57s。包含題目cluster、instruction ratio、三seeds不得變成獨立題目、官方parser、sampling、共用引擎與中斷寫入。未執行full test suite或額外GPU regression。
- [共用引擎等價](engine_reuse_audit.json)：抽出的create_llm設定與前輪已通過pilot相同；無新套件／新framework。沿用Google與SciPy，未自行重寫grader或bootstrap。
- 最後稽核補上mutable attempt JSON的atomic replace，避免程序在timing更新中被殺死時留下破損JSON；CPU測試注入中斷後舊快照仍可讀。[AST消融](simplification_audit.json)證明只有三個metadata寫入呼叫改變，推論、prompt、sampling、解析、評分完全相同；品質與CI CSV也逐byte一致。保留[原Smoke source](ifeval_modes_smoke_source.py)符合原manifest hash；最終正式plan重新固定目前程式hash。此檔案寫入修正不需要重跑同樣60次推論。

## 配對統計

[paired_comparison.csv](analysis_final/paired_comparison.csv)含13個對比×7指標×兩種信賴水準＝182列，bootstrap199,999次；[per-seed敏感度](analysis_final/per_seed_comparison.csv)、[逐題效果](analysis_final/question_effects.csv)、[accuracy](analysis_final/accuracy.csv)。題目是主要單位，同題三個固定seeds一起重抽樣；instruction指標重算比值。主要family為91，保存95%與Bonferroni99.945%區間。

兩題皆全對使accuracy差異與Harmful/Recovery的經驗變異為零，這種退化區間不能拿來證明模型等價或完整能力沒有差異。CPU合成例另含錯對互換與不同instruction數，並證明錯把三seeds當獨立題目會把CI縮窄。

正式清單在 [prepared/experiment.json](../ifeval_modes_prepared/experiment.json)。Full plan已通過最終程式的CPU resume identity check，沒有任何questions.jsonl／GPU回答。完整啟動命令見USAGE；本階段停在可啟動狀態。
