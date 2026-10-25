# 九模型量化實驗整合分析與週一排程決策

更新：2026-10-17。週一報告時間已確認為 **2026-10-19 09:00 台北時間**。本輪使用已完成的 Fidelity 與代表性 pilot；正式分析只使用已完成的核心／IFEval補樣與4K資料，完整 36,009 次 Benchmark 尚未開始。

本文1K＝1,024、4K＝4,096 output tokens；正式Benchmark時間均為推估，pilot時間及正確數均為實測。

## 建議決策

優先保留 **Qwen BF16、NaviQuant Fast、PrismaAQUA、ModelOpt AutoQuantize** 四個研究核心模型，直接回答「Fast 相對兩種自動量化方案是否有能力差異」。若增加一個部署對照，建議 **Unsloth NVFP4**；若報告更重視較高精度對照，則以 **Qwen FP8** 作第五個模型。

週一先用目前完整 Fidelity 與 pilot 報告即可。若後續決定在報告前補完整數據，優先考慮九模型完整 IFEval：五張 H100 一般 **5.1 小時、保守 8.0 小時**。完整三項 Benchmark 即使只保留四模型，也需要約 **43.2 小時（MMMU 1K）／86.5 小時（4K）**，超出週一上午的窗口。1K 有明顯截斷，不能把這個較快預算當作建議的最終能力設定。

## 已完成的證據與實驗範圍

Fidelity 已完成九模型各55,296計分tokens、288文件、固定公開資料v2.1；沿用 teacher forcing、相同token IDs、完整248,320 vocabulary、BF16 KV及MTP Off。Prisma／AutoQuantize加入前的完整BF16 gate分布hash與既有reference全相同；保留的既有結果直接重用。[完整報告](../../../../REPORT.md)保留量化配置、搜尋及公平性分析。

代表性 pilot 每模型200題：IFEval 78、MMMU-Pro Standard 61、Vision 61，九模型共1,800次主要推論。核心160題包含35題IFEval分層機率樣本、Standard／Vision各60題（30學科各2題），另5題壓力檢查；再追加40題獨立IFEval樣本。圖片數包含1–35張，覆蓋官方Easy／Medium／Hard與長輸入；IFEval沒有官方難度標籤，指令數／字數要求只是成本代理。

另完成501次4K探測：451次1K截斷回答及50次未截斷控制。主要與探測prompt/token IDs/圖片完全一致，501次都保留相同原始token prefix；50個未截斷控制的完整token IDs及答案全部相同。8K補測未完成；另有17個背景誤觸輸出已隔離，見下方事件紀錄。原始核心、補樣與probe分開保存，合併只是derived union，18個原始JSONL檔案皆逐byte保留。

實際使用五張NVIDIA H100 80GB HBM3（GPU0–4，driver570.86.10），每GPU一個TP=1模型，每次只執行一題。vLLM0.31.0+cu129、eager、prefix cache Off、max_num_seqs=1、max_num_batched_tokens=4096，context65,536、max_images35，processor min65,536/max16,777,216 pixels。

共同解碼為 **非greedy、thinking Off、MTP Off、BF16 KV**：temperature0.7、top_p0.8、top_k20、min_p0、presence_penalty1.5、repetition_penalty1、seed20261016。IFEval上限4096；MMMU主回答1024，較長探測4096。這是明列的Qwen non-thinking研究設定；舊greedy／1M-pixel啟動smoke不混入本輪數值。

## Fidelity 與模型容量

| 模型 | 權重 GB | PPL | KL (BF16→候選) | Top-1 Flip % |
| --- | --- | --- | --- | --- |
| Qwen BF16 | 55.56 | 3.672697 | 0.000000 | 0.000 |
| Qwen FP8 | 30.87 | 3.681106 | 0.004277 | 2.412 |
| NaviQuant Fast | 23.75 | 3.714690 | 0.017938 | 4.798 |
| NVIDIA NVFP4 | 21.92 | 3.712698 | 0.018645 | 5.264 |
| RedHatAI NVFP4 | 24.69 | 3.701360 | 0.015416 | 4.440 |
| Unsloth NVFP4 | 23.42 | 3.697641 | 0.014476 | 4.736 |
| RadixArk BF16-LMHead | 23.75 | 3.716024 | 0.018956 | 4.912 |
| PrismaAQUA | 23.60 | 3.705657 | 0.014083 | 4.275 |
| ModelOpt AutoQuantize | 23.54 | 3.721630 | 0.019988 | 4.841 |

GB為所有safetensors實際檔案的十進位容量，並非載入後VRAM。Prisma23.6035GB、Fast23.7520GB、Auto23.5360GB相近；三者350個保留Vision/MTP/lm_head/embedding tensors與BF16逐byte相同，共6.856GB。可比language-body Linear有效bits分別為5.500161／5.548951／5.477986，不能把5.5解讀為整個27B checkpoint都用5.5bits。

Fast−Prisma在18項Bonferroni family的校正區間：ΔKL **+0.003855 [0.003027,0.004741]**、ΔFlip **+0.522642pp [0.218009,0.826461]**，支持Prisma較接近BF16。ΔNLL+0.002435 [-0.001006,0.006087]，PPL方向仍不確定。Fast−Auto的ΔKL **−0.002050 [−0.002810,−0.001273]**支持Fast較接近BF16；ΔNLL與ΔFlip校正區間均跨零。這些是固定checkpoint／本機後端的Fidelity證據，不能外推為普遍量化演算法排名或能力排名。

Auto確實使用官方KL-guided Weight AutoQuantize搜尋，不是固定layer map；301 NVFP4／99 FP8／96 BF16 Linears。搜尋使用128 calibration、8 sensitivity文件，約0.213GPU-h；sensitivity樣本偏少，這一個結果不代表ModelOpt最佳可能品質。Prisma為公開checkpoint，完整上游搜尋成本沒有本機可驗證數值。

## 代表性能力結果與截斷敏感度

下表是相同題目的原始正確數，IFEval主指標為prompt-level strict。包含壓力題，IFEval又有條件追加抽樣，**不可把這些比率當作完整Benchmark的準確分數**。4K欄使用實際保存的較長回答替換原本截斷的MMMU回答，未截斷回答沿用；有50個控制驗證此重用方式。

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

主要MMMU共1,098回答，**451/1,098＝41.1%**在1K截斷；提高4K後仍有**187/1,098＝17.0%**截斷，涉及40個不同題目ID。451個截斷例延長後仍有187個未完成，不能用187/501探測比例當作母體截斷率。IFEval702回答全部自然結束。全2,301筆EOS稽核：638個length回答都未含EOS，1,663個自然結束回答末尾皆為248046，沒有漏設EOS的證據。

截斷會改變表面排名：BF16 Vision由25/61提高至41/61；Auto由29/61到32/61。Auto相對BF16由1K的+4題變成4K的−9題。這支持先處理生成上限與解碼設定，再作最終能力比較。4K仍有截斷；不能假設4K等於無截斷或以此預算保證8K成本。主回答MMMU官方parser fallback為56/1,098；4K替換後剩1/1,098。官方fallback是固定question seed的猜選項，不能悄悄移除。

![Fidelity與輸出上限敏感度](fidelity_and_cap_sensitivity.png)

[圖表PDF](fidelity_and_cap_sensitivity.pdf)；[完整逐模型cap結果CSV](quality_by_cap.csv)／[Markdown](quality_by_cap.md)；[cap變動的Harmful／Recovery／Answer Flip](cap_sensitivity.csv)。

## 配對分析與不確定性

逐題定義：Accuracy Delta=candidate−BF16；Harmful Flip=BF16對而candidate錯，Recovery相反，兩者分母都是所有題目，故Delta=Recovery−Harmful。MMMU Answer Flip比較解析後選項；IFEval比較去除首尾空白的完整回答字串，因此不能作語意錯誤率。

成本與母體加權品質採同一實際抽樣設計。IFEval已觀測38題視為固定已知總和，剩餘503題用獨立40題估算，權重503/40；不可把78題當一次SRS。MMMU按各學科母體題數加權，每學科2個機率樣本，壓力題權重0。SciPy於層內重抽題目、模型共用索引；不同task使用獨立亂數流，已知census層不重抽。品質199,999次bootstrap，成本19,999次。

以下為4K的**探索性母體加權**配對點估計與一般95%區間。它們可指出值得正式驗證的方向，無法確立完整能力優劣。

| 候選−BF16 | task | 母體加權 ΔAccuracy pp | 一般95%區間 pp |
| --- | --- | --- | --- |
| NaviQuant Fast | ifeval | -0.55 | [-9.85, 8.74] |
| NaviQuant Fast | mmmu_standard | -1.73 | [-10.35, 6.91] |
| NaviQuant Fast | mmmu_vision | -7.05 | [-15.38, 1.27] |
| PrismaAQUA | ifeval | 4.83 | [-4.46, 14.13] |
| PrismaAQUA | mmmu_standard | 1.68 | [-6.33, 9.68] |
| PrismaAQUA | mmmu_vision | -8.41 | [-16.71, -0.12] |
| ModelOpt AutoQuantize | ifeval | 4.65 | [-4.65, 13.95] |
| ModelOpt AutoQuantize | mmmu_standard | 8.61 | [1.85, 15.35] |
| ModelOpt AutoQuantize | mmmu_vision | -15.14 | [-23.79, -6.50] |

原始觀測題的Flip如下；全部八候選、兩種cap、四指標及加權／未加權區間保留於CSV。

| 候選−BF16 | task | Harmful % | Recovery % | Answer Flip % |
| --- | --- | --- | --- | --- |
| NaviQuant Fast | ifeval | 8.97 | 5.13 | 93.59 |
| NaviQuant Fast | mmmu_standard | 13.11 | 11.48 | 34.43 |
| NaviQuant Fast | mmmu_vision | 11.48 | 4.92 | 22.95 |
| PrismaAQUA | ifeval | 3.85 | 7.69 | 91.03 |
| PrismaAQUA | mmmu_standard | 9.84 | 11.48 | 31.15 |
| PrismaAQUA | mmmu_vision | 11.48 | 3.28 | 21.31 |
| ModelOpt AutoQuantize | ifeval | 5.13 | 7.69 | 94.87 |
| ModelOpt AutoQuantize | mmmu_standard | 4.92 | 13.11 | 36.07 |
| ModelOpt AutoQuantize | mmmu_vision | 18.03 | 3.28 | 36.07 |

本輪對8候選×4指標×5個不同task/cap單元（IFEval算一次、兩MMMU各兩上限）統一使用**160項Bonferroni family**，同時保存95%與99.96875%區間。這是看到截斷後明列的探索性family，並非事先註冊的確認性研究；舊96項結果保留追溯，不沿用為本輪全部比較的校正。

**每學科只有2題，bootstrap無法產生未見的錯誤型態。** 某些層的paired差值完全相同，區間可過窄；部分bootstrap區間即使排除零，也不宜直接宣稱能力退化或優勢。本次額外保存原始觀測題exact McNemar檢查：1K最小未校正p約0.0654，4K約0.0225，乘160後所有p均為1。這是未加權觀測樣本的保守交叉檢查，並不等於分層母體的精確檢定，也不能把無顯著差異當作品質等價。正式資料與新seed仍有必要。

[1K配對結果](../paired_1024/paired_comparison.csv)／[1K加權區間](../paired_1024/weighted_paired_comparison.csv)；[4K配對結果](../paired_4096/paired_comparison.csv)／[4K加權區間](../paired_4096/weighted_paired_comparison.csv)／[exact配對檢查](../paired_4096/accuracy_tests.csv)。

## 實測成本：為什麼這次 smoke 要九小時

| 階段 | 主要回答 | 整體 wall | 實際占用 GPU-h | 五 GPU整組保留 GPU-h |
| --- | --- | --- | --- | --- |
| 核心＋4K探測 | 1,440＋501探測 | 8小時37分42秒 | 36.885 | 43.141 |
| IFEval獨立補樣 | 360 | 24分46秒 | 1.726 | 2.064 |
| 兩階段合計 | 1,800＋501探測 | 9小時02分28秒（執行時間相加） | 38.610 | 45.206 |

核心台北時間10-17 12:12:45開始、20:50:27結束；補樣20:50:34開始、21:15:20結束。含中間約7秒空檔的calendar span為9小時02分35秒。九模型共18個程序退出碼皆0；模型初始化共824.44秒，平均每次45.80秒；內嵌官方評分合計5.72秒。另行CPU官方重評13.18秒、來源／全圖片稽核190.51秒。初始化與評分都不是主要瓶頸。

| 模型 | 初始化秒（兩次合計） | 主推論 IFEval／Standard／Vision 分 | 4K探測分 | 程序 wall h |
| --- | --- | --- | --- | --- |
| Qwen BF16 | 101.1 | 22.0 / 30.5 / 35.3 | 118.2 | 3.47 |
| Qwen FP8 | 89.9 | 31.7 / 38.9 / 51.6 | 165.0 | 4.82 |
| NaviQuant Fast | 92.1 | 27.7 / 39.2 / 46.3 | 146.1 | 4.36 |
| NVIDIA NVFP4 | 91.5 | 27.0 / 30.4 / 47.0 | 150.6 | 4.28 |
| RedHatAI NVFP4 | 94.9 | 28.6 / 41.7 / 44.9 | 157.7 | 4.58 |
| Unsloth NVFP4 | 88.4 | 29.3 / 35.8 / 42.4 | 136.7 | 4.10 |
| RadixArk BF16-LMHead | 85.3 | 27.4 / 36.2 / 44.8 | 154.6 | 4.41 |
| PrismaAQUA | 99.4 | 28.6 / 32.5 / 42.4 | 141.7 | 4.12 |
| ModelOpt AutoQuantize | 81.9 | 29.6 / 33.9 / 46.2 | 155.8 | 4.46 |

| task | 階段 | 回答數 | input tokens 均值/p50/p95 | output tokens 均值/p50/p95 | 推論秒 均值/p50/p95 | output tok/s 均值/p50/p95 |
| --- | --- | --- | --- | --- | --- | --- |
| ifeval | main | 702 | 63.0 / 57.0 / 100.7 | 361.2 / 256.0 / 1084.7 | 21.5 / 14.9 / 65.2 | 16.7 / 16.4 / 20.9 |
| mmmu_standard | main | 549 | 793.8 / 375.0 / 3018.0 | 583.1 / 614.0 / 1024.0 | 34.9 / 36.2 / 63.7 | 14.7 / 16.4 / 20.5 |
| mmmu_standard | long_probe | 226 | 802.5 / 359.0 / 1250.0 | 2471.5 / 2207.0 / 4096.0 | 147.3 / 131.7 / 253.8 | 16.6 / 16.7 / 20.6 |
| mmmu_vision | main | 549 | 3101.6 / 2334.0 / 7154.0 | 722.7 / 925.0 / 1024.0 | 43.8 / 49.6 / 67.2 | 16.5 / 16.4 / 20.5 |
| mmmu_vision | long_probe | 275 | 3260.7 / 2678.0 / 7154.0 | 2807.6 / 3101.0 / 4096.0 | 168.4 / 186.5 / 264.1 | 16.8 / 16.7 / 20.7 |

本表為這批題目的未加權分布。長探測是選中的截斷／控制題，與main不是同一抽樣分布，不能直接比較均值。每題input/output tokens、preprocess／inference／score秒數、平均throughput及p50/p95/max另見[逐題CSV](../analysis/question_timings.csv)、[27組主要結果統計](../analysis/smoke_results.csv)、[載入與完整wall ledger](../analysis/model_load_and_wall.csv)。

4K探測沿用已開啟的圖片／模板，MM processor cache可能已暖身，因此對fresh完整4K推論可能略低估前處理成本；長decode在此占主要時間，但沒有用這點假設偏差必為零。非census層未使用有限母體修正，抽樣方差的這一項處理偏保守，仍無法涵蓋未觀測長尾。

主要時間花在501次長回答：Standard探測平均約147秒、Vision約168秒；部分接近5分鐘。H100的NVFP4使用Marlin W4A16，沒有Blackwell原生FP4收益；此處每次一題、eager的觀測throughput，並非各模型最大服務throughput。BF16較快的整體結果可能同時含kernel、回答長度與題目差異；不能說低bit一定加速。`LLM.generate` wall含prefill/decode/引擎工作，output tokens/該時間不是純decode kernel tok/s。

## 完整36,009次 Benchmark：每模型、每task成本

單GPU與多GPU都維持TP=1、逐題、相同硬體／設定。以下每格是GPU allocation hours，包含分攤固定成本；單GPU串行執行的wall hours數值相同。每模型正式全資料僅計一次平均初始化與固定開銷；不把兩階段pilot的兩次初始化照搬。

### MMMU上限1024；IFEval4096

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

### MMMU上限4096；IFEval4096

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

各task樂觀／保守邊界及加權預測截斷率見[完整估算CSV](../analysis/full_estimates.csv)／[模型合計CSV](../analysis/full_model_estimates.csv)。模型合計的區間先對三task的bootstrap抽樣總和取分位數，**不是相加各task的邊際分位數**。

| MMMU cap | 情境 | 單 GPU wall h＝占用 GPU-h | 五 GPU wall h | 整組保留 GPU-h |
| --- | --- | --- | --- | --- |
| 1024 | optimistic | 345.4 | 77.0 | 384.8 |
| 1024 | typical | 364.1 | 81.0 | 405.2 |
| 1024 | conservative | 478.5 | 106.4 | 531.9 |
| 4096 | optimistic | 690.2 | 153.5 | 767.3 |
| 4096 | typical | 764.0 | 170.0 | 849.8 |
| 4096 | conservative | 1046.3 | 232.5 | 1162.4 |

一般情境=分層母體加權均值；樂觀=各模型總成本bootstrap2.5%邊界；保守=97.5%邊界再加25%未見長尾、資源競爭及失敗重試餘量。25%是規劃判斷，並非量測出的信賴上界。模型邊界相加與排程形成的是情境範圍，不是整體同時95%工期區間；單一sampling seed、每科2題及仍截斷輸出均限制精度。

4K五GPU典型排程（長工作優先，一模型不跨GPU分題）如下。起點是假設週日10-18 00:00台北時間，**不是已啟動工作**。

| GPU | 模型 | 開始 h | 結束 h | 台北完成時間 |
| --- | --- | --- | --- | --- |
| 0 | Qwen FP8 | 0.00 | 96.50 | 2026-10-22 00:29+08:00 |
| 1 | RedHatAI NVFP4 | 0.00 | 92.75 | 2026-10-21 20:44+08:00 |
| 2 | RadixArk BF16-LMHead | 0.00 | 88.24 | 2026-10-21 16:14+08:00 |
| 3 | NaviQuant Fast | 0.00 | 86.47 | 2026-10-21 14:28+08:00 |
| 4 | ModelOpt AutoQuantize | 0.00 | 85.75 | 2026-10-21 13:44+08:00 |
| 4 | NVIDIA NVFP4 | 85.75 | 169.96 | 2026-10-25 01:57+08:00 |
| 3 | PrismaAQUA | 86.47 | 167.64 | 2026-10-24 23:38+08:00 |
| 2 | Unsloth NVFP4 | 88.24 | 168.36 | 2026-10-25 00:21+08:00 |
| 1 | Qwen BF16 | 92.75 | 161.52 | 2026-10-24 17:31+08:00 |

![正式成本範圍](../analysis/full_cost_ranges.png)

每GPU時價若為R，按實際模型占用計費的金額為「occupied GPU-h×R」；若五卡整組保留，為「reserved GPU-h×R」。未取得實際租價，不編造金額。4K一般占用約764GPU-h；整組保留約850GPU-h。

## 名單縮減與週一時間窗口

研究核心四模型：BF16是品質基準；Fast是主研究對象；Prisma與Auto分別代表公開自動量化checkpoint與本機真正自動搜尋。Auto Fidelity較差但Standard有探索性正向結果、Vision相反，保留它才能回答「較接近BF16是否代表較好能力」，而不是依結果挑選有利對象。

Unsloth可作第五個實用部署對照：23.42GB，Fidelity PPL／KL有競爭力，IFEval原始62/78；Vision4K34/61仍需完整驗證。FP8則提供30.87GB的較高精度正控制。NVIDIA以21.92GB最小檔案容量為保留理由；若記憶體／儲存是主題可優先於Unsloth。RedHatAI與RadixArk BF16-LMHead在本次精簡研究問題中優先度較低；這是資訊增益與預算取捨，沒有證據宣稱它們普遍較差。

| 名單 | MMMU cap | 占用 GPU-h／單 GPU h | 五 GPU一般 h | 五 GPU樂觀–保守 h | 假設週日00:00啟動，一般完成時間 |
| --- | --- | --- | --- | --- | --- |
| 研究核心四模型 | 1024 | 155.2 | 43.2 | 41.2–56.4 | 2026-10-19 19:10+08:00 |
| 研究核心四模型 | 4096 | 322.2 | 86.5 | 79.4–117.3 | 2026-10-21 14:28+08:00 |
| 核心＋Unsloth | 1024 | 194.9 | 43.2 | 41.2–56.4 | 2026-10-19 19:10+08:00 |
| 核心＋Unsloth | 4096 | 402.3 | 86.5 | 79.4–117.3 | 2026-10-21 14:28+08:00 |
| 核心＋FP8 | 1024 | 200.9 | 45.7 | 43.2–60.3 | 2026-10-19 21:43+08:00 |
| 核心＋FP8 | 4096 | 418.7 | 96.5 | 86.3–133.1 | 2026-10-22 00:29+08:00 |

四模型減成五模型內，只要一模型占一GPU，牆鐘時間由最慢模型決定。核心＋Unsloth不增加目前一般critical path，但增加GPU-hours；FP8在本後端較慢，會把4K五模型工期拉到96.5小時。六模型超過五卡會出現第二波，不能直接用總GPU-h/5當可執行排程。跨GPU分題是可能的後續優化，但目前未驗證，不把理想完全可切分下界當工期承諾。

已確認週一10-19 09:00報告，可先交付現有證據、數據之後補。排程基準假設週日00:00啟動，至報告共33小時，保留最後3小時整理與QA；若實際啟動較晚，全部完成時間需等量後移。

| 可選工作 | 五GPU一般／保守 | 假設週日00:00開始的結束時間 | 判讀 |
|---|---:|---|---|
| 只用已完成Fidelity＋pilot寫報告 | 無新增GPU成本 | 報告已可交付 | 本次實際交付 |
| 九模型完整IFEval，541×9 | 5.1／8.0小時 | 週日05:05／07:57 | 優先補數據選項，0個觀測截斷 |
| 核心＋Unsloth，IFEval＋Standard1K | 21.1／28.3小時 | 週日21:08／週一04:16 | 可放入窗口，但MMMU僅可標為1K受限結果 |
| 核心＋Unsloth，IFEval＋Standard4K | 41.9／58.9小時 | 週一17:55／週二10:55 | 超過報告時間 |

以上是替代排程；不能先跑全部IFEval，再把IFEval＋Standard的整組時間當作純新增Standard時間相加。建議週一主報告保留Fidelity結論與pilot限制，若需要完整分數先補IFEval；MMMU待小量確認greedy或其他允許的direct輸出策略、較長上限與停止行為後，再決定正式上限。改用greedy屬新實驗條件，不能混入目前sampling結果，也沒有證據保證它一定更快。

保存pilot題目若在同一模型／資料／完整設定下可節省重複推論；[可重用的已算案例](pilot_reuse_savings.csv)列出每task案例與已觀測秒數。此處預算保守按全資料重算，尚未承諾可直接續跑或省下固定比例，避免把尚未驗證的full merge算進預算。

## MTP／DSpark：啟動證據與成本未知要分開

九模型MTP=2與DSpark=7共18個啟動worker均成功，另有恢復官方解析度後的33個greedy smoke回答與非零draft／accepted counters；這些證明有參與解碼。[相容性矩陣](../../compatibility.md)／[完整官方複核](../../OFFICIAL_REVIEW.md)。DSpark公開card的特定SGLang target不能替其他模型／後端推定相容，實際測試另外標示。

目前沒有代表性MTP On pilot可用於工期估算。先前108個Off／On小smoke使用舊1M-pixel限制，生成長度也不同，不能把舊107/197GPU-h外推為加速倍數。六個greedy配對有兩個token不一致，原因尚未單獨定位；128-token且有fallback的分數相同不代表品質等價。vLLM將speculative decoding描述為分布無損，但浮點／batch及sampling軌跡仍需實測。[官方說明](https://docs.vllm.ai/en/latest/features/speculative_decoding/)。

建議先做選定模型的MTP Off完整能力基準，再用相同題目分開作On／Off。九模型兩模式共72,018次；若僅作「On成本=Off」假設，4K算量約1,528占用GPU-h，這是預算敏感度情境，**不是On的實測估計或加速承諾**。下一輪小量On測試需含短／長文字及圖片，記錄acceptance、token長度、wall與配對分數後才能更新成本；暫不把DSpark再加入正式矩陣。

## 官方正確性、簡化與重現

官方來源逐檔核對八個所用upstream檔案；Google官方CLI重評702個IFEval strict／loose及instruction向量；MMMU作者`mmmu_process_results`重評1,599個回答，全部與保存結果一致。完整資料再核對541個Google canonical prompts、3,460個MMMU prompts/options/gold/subject及3,778張影像像素／順序／SHA，全部通過。[官方重評audit](../official_replay/audit.json)。

IFEval使用Google官方GitHub commit `3ef52111c1fabc301880818f5a547570e926a76b`；MMMU-Pro工具commit `268471d0d488258990025331c7528359c324aa25`、資料revision `0d7426df4ccb3d8704a992fe6850abee7c262127`。MMMU採官方direct prompt／parser／grader，影像照官方mapping。作者允許direct或cot；IFEval README沒有指定唯一sampling。本研究的BF16、Qwen sampling與較長output cap為明列選擇，不能宣稱逐項複製官方Transformers示例或leaderboard設定。

新增分析繼續重用既有runner／計時／SciPy估算器，沒有新的評估framework。移除wrapper自動8192階段，僅重現已授權核心＋IFEval補樣。17項相關CPU tests通過，未執行full suite；等價消融確認三task的預設成本公式與顯式原公式抽樣陣列逐元素相同、96列原始95%paired統計不變、18個原JSONL逐byte保留、所有九模型joint情境與原估算器一致。[驗證](analysis_validation.json)／[EOS](termination_audit.json)／[整體audit](decision_audit.json)。

### CPU重新分析既有資料（不啟動模型）

在專案根目錄執行，output使用新的資料夾；只分析已合併資料時不需再次merge。

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
.modelopt-venv/bin/python src/verify_benchmark_official.py \
  --runs build/benchmark/pilot_v1/combined_runs --include-extensions --output build/benchmark/new_replay
```

`summarize_benchmark_pilot.py`讀取pilot根目錄的`analysis/`、`combined_runs/`、`official_replay/`與既有Fidelity CSV，輸出模型／task子集的joint bootstrap預算及排程。[所有子集CSV](subset_budgets.csv)／[Markdown](subset_budgets.md)／[逐GPU排程](subset_schedules.csv)／[排程假設](planning_assumptions.json)。

### 未執行的正式命令範例

以下只供選定範圍後使用，**本次沒有執行**。單模型完整IFEval：

```sh
export LD_LIBRARY_PATH="$PWD/build/cuda-compat/cuda-compat${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8 VLLM_NO_USAGE_STATS=1 HF_HUB_OFFLINE=1 \
  .modelopt-venv/bin/python -u src/benchmark.py run \
  --model Qwen-Qwen3.8-27B --data assets/benchmark/prepared_v2 \
  --tasks ifeval --output build/capability/ifeval/Qwen-Qwen3.8-27B
```

選定四模型完整三task，4096上限的命令形式如下；三task均使用4096設定。launcher完整資料需明確`--full`，原default config仍保留本輪1K以維持重現。

```sh
.modelopt-venv/bin/python -u src/run_benchmarks.py \
  --data assets/benchmark/prepared_v2 --config assets/benchmark_research4_4096_config.json \
  --output build/capability/research4_4096 --gpus 0 1 2 3 4 --full \
  --models eNeural-Qwen3.8-27B-fast ModelOpt-Qwen3.8-27B-AutoQuantize-5.5bit \
    rdtand-Qwen3.8-27B-PrismaAQUA-5.5bit-vllm Qwen-Qwen3.8-27B
```

四模型分析使用同一config選取名單；`--mmmu-cap 4096`在pilot讀取既有較長回答，直接4K完整run則保留原主回答，無須補probe。

```sh
.modelopt-venv/bin/python src/benchmark_analysis.py \
  --runs build/capability/research4_4096 --config assets/benchmark_research4_4096_config.json \
  --mmmu-cap 4096 --family-size 96 --resamples 199999 --output build/capability/research4_analysis
```

這個96項family對原九模型單模式三task四指標保守預留；若加入MTP／額外cap，需在啟動前依最後名單與假設另定。不要將pilot篩選後的名單假裝成事先註冊。

來源：[Google IFEval README](https://github.com/google-research/google-research/blob/3ef52111c1fabc301880818f5a547570e926a76b/instruction_following_eval/README.md)、[MMMU-Pro README](https://github.com/MMMU-Benchmark/MMMU/blob/268471d0d488258990025331c7528359c324aa25/mmmu-pro/README.md)、[Qwen non-thinking建議](https://huggingface.co/Qwen/Qwen3.8-27B#best-practices)、[SciPy bootstrap](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html)、[SciPy binomtest](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.binomtest.html)。官方評分器與vLLM均沿用Apache-2.0工具，原碼／LICENSE保留於既有tools目錄；未新增付費judge或外部服務。

## 未預期背景8K續跑：已停止並排除

最後GPU檢查發現，先前暫停時遺留的控制器PID9718在本次合併資料之後啟動8K階段（worker於14:28:03 UTC啟動）。本次CPU分析程式沒有呼叫推論，但未先清除等待中的舊控制器，讓本輪範圍受到影響。發現後已停止該控制器與全部17個子程序，五張GPU均已釋放，沒有啟動完整Benchmark。

這段約26.9分鐘、約2.24占用GPU-hours是**額外意外成本**，依程序啟動與停止時間粗估，沒有完整launcher ledger；不包含在前述完成pilot的38.61 GPU-h。17個已寫出的8K回答及日誌全部移至`interrupted_unexpected_8192/`，未合併、未官方重評、未用來計算品質或8K預算，也不能稱為完整補測成果。原核心／IFEval／4K原始檔未變動。

[停止與成本證據](unexpected_8192_stop.json)／[隔離目錄說明](../interrupted_unexpected_8192/README.md)。可重用wrapper已移除8K自動階段；未再建立新的背景續跑工作。
