# Fidelity 擴大實驗：固定公開資料 v2.1

2026-10-16 完成。全部 8 個模型均實際完成每模型 **55,296 個計分 tokens、288 份不同文件**的評估；含 BF16 獨立兩次的 9 個工作程序全部退出碼 0。計分位置是第一輪的 128 倍。Fast 並未全面優於四個比較對象：對 RadixArk BF16-LMHead 有穩定 KL 優勢；對 NVIDIA 有穩定 Flip 優勢；對 RedHatAI 與 Unsloth 的 KL 較差。PPL／Flip 的部分差距仍不確定，詳見下方配對區間。

Fast 指本機 `eNeural-Qwen3.8-27B-fast`，依本次指定對象稱為 NaviQuant Fast。其本機 revision 為 `284b37dd7b7a5e0071268c0f17fbbb47e4f09b17`；目前公開端點無法取得模型卡，因此以實際本機權重、設定及 recipe 的 SHA-256 為版本依據。完整模型名稱與本機檔案見 [模型紀錄](expanded/model_checks.csv)、[全部權重指紋](expanded/model_fingerprints.json)。

## 完整模型比較

| 模型 | 狀態 | PPL | ΔPPL 對 BF16 | KL(BF16→候選) | Flip % |
|---|---|---:|---:|---:|---:|
| Qwen BF16 | ok | 3.67269657 | +0.00000000 | 0.00000000 | 0.00000 |
| Qwen FP8 | ok | 3.68110552 | +0.00840896 | 0.00427723 | 2.41247 |
| RadixArk NVFP4 | ok | 3.73610922 | +0.06341266 | 0.02506744 | 6.31872 |
| RadixArk BF16-LMHead | ok | 3.71602372 | +0.04332716 | 0.01895551 | 4.91175 |
| RedHatAI | ok | 3.70135964 | +0.02866307 | 0.01541646 | 4.43974 |
| NaviQuant Fast | ok | 3.71469028 | +0.04199371 | 0.01793785 | 4.79782 |
| NVIDIA | ok | 3.71269762 | +0.04000105 | 0.01864520 | 5.26440 |
| Unsloth | ok | 3.69764107 | +0.02494450 | 0.01447626 | 4.73633 |

PPL 是 token 加權的 `exp(總 NLL / N)`；ΔPPL 為候選減 BF16。KL 是完整 vocabulary 的 BF16→候選方向，使用自然對數；Flip 為 argmax token 與 BF16 不同的比例，這裡以百分比顯示。CSV 保留完整精度。PPL 較低與更接近 BF16 是不同的觀察量；這些指標不是任務正確率。

完整 [CSV](expanded/analysis/comparison.csv)／[Markdown](expanded/analysis/comparison.md)；[各領域、context 與交叉分組](expanded/analysis/grouped.csv)；[各模型與分組的 95% 區間](expanded/analysis/intervals.csv)。

## 資料與控制條件

| 領域 | 公開來源 | 文件數 | 計分 tokens |
|---|---|---:|---:|
| 英文一般知識 | Wikimedia Wikipedia 20231101.en 固定第一個 shard | 72 | 13,824 |
| 中文一般知識 | Wikimedia Wikipedia 20231101.zh 固定第一個 shard | 72 | 13,824 |
| 形式化數學 | Mathlib4 的 Lean 定理、證明及說明 | 72 | 13,824 |
| 程式碼 | CPython Python 測試程式 | 72 | 13,824 |

每個領域分成初始前文 128、512、2,048 tokens，各 24 份文件；每份只計分接下來 192 個 tokens，總 prompt 最長 2,240 tokens。各 context 組使用不同文件，沒有重複計分 target。選樣依固定 seed 20261016 的 SHA-256 排序與文件內 offset，不依模型分數選樣。288 份原文與 target 均不同；target 之間沒有共用的連續 32-token 片段。

[資料說明與授權](../assets/expanded_v2/README.md)、[固定設定](../assets/expanded_config.json)、[來源 revision／SHA manifest](../assets/expanded_v2/manifest.json)、[資料稽核](expanded/dataset_audit.json)。tokens.json SHA-256：`09b41e1a10044975acc579331d3b22ac4ba54f26f89882bb704f6a319d4c2e0d`。

直接來源避開 CNN/DailyMail、PerfectBlend 與 Nemotron calibration corpus。初版稽核仍發現兩份 Wikipedia 文件有 13-word 片段重疊，其中一份涉及 target；已固定排除來源 ID，補入兩份文件並重新執行完整實驗。v2.1 對 **CNN/DailyMail 全 train 287,113 筆及 PerfectBlend train[:512]** 的 NFKC／casefold 13-word grams、中文去空白 32-character grams 稽核均為 0 matches。[最終稽核](expanded/calibration_overlap.json)、[初版重疊紀錄](expanded/calibration_overlap_initial.json)。CNN revision 與 RadixArk manifest 相同；PerfectBlend 是官方 adapter 對應的公開版本，不能保證與當時轉換完全同版。Fast、Unsloth、Qwen 與 NVIDIA 選用的完整 calibration 文本未全部公開，不能保證完全無重疊，也沒有宣稱對基座預訓練資料去重。

使用 BF16 tokenizer、同一份 token IDs 與計分位置，`add_special_tokens=False`，不套 chat template、不跨文件串接。對初始 context C，計分 input indices C..C+191；官方 causal row 起點為 C−1，沒有再多做一次 shift。輸出完整 **248,320 維 vocabulary**，包含 padding 維度；tokenizer 實際有 248,077 個 token。logprobs 以 FP32 擷取，指標以 FP64 加總，不用 top-k 近似，也不另外 renormalize。

每模型 TP=1、H100 80GB、BF16 基礎 dtype、**MTP Off、BF16 KV cache**、prefix cache Off、seed=0、eager、max model length／batched tokens=2304、每次一個序列。不同 GPU 僅平行執行不同模型。環境沿用第一輪 vLLM 0.31.0+cu129、Torch 2.13.0+cu129、Transformers 5.17.0、compressed-tensors 0.17.0；新增 SciPy 1.18.0、PyArrow 23.0.1、Matplotlib 3.11.2。[環境](expanded/environment.json)、[套件來源及 wheel 雜湊](expanded/package_provenance.json)、[完整 freeze](expanded/environment.freeze.txt)、[原始結果與日誌](expanded/full/results.json)。

## 文件層級統計與 Fast 比較

直接使用 SciPy `stats.bootstrap`，percentile、19,999 次、固定 seed 20261016。每個文件的全部 192 個 tokens 先聚合為一個 cluster，在 domain × context 各層獨立重抽完整文件，模型之間保持配對。沒有把文件內 tokens 當成獨立樣本。每個交叉 cell 只有 24 個獨立文件，分組區間可能較寬。

各模型與分組提供一般 95% 描述區間；Fast 對四個指定模型的 ΔNLL、ΔKL、ΔFlip 共 12 項主要比較，另外提供 Bonferroni **99.5833%** 區間。下表均為這個校正區間，差值定義 **Fast 減對象**，負值表示 Fast 較低。ΔNLL 的方向等同 PPL ratio 與 PPL difference 的方向。

| Fast 相對對象 | ΔNLL，nats/token | ΔKL，nats/token | ΔFlip，百分點 |
|---|---:|---:|---:|
| RadixArk BF16-LMHead | -0.000359 [-0.002236, +0.001457] | -0.001018 [-0.001698, -0.000495] | -0.113932 [-0.323712, +0.100075] |
| NVIDIA | +0.000537 [-0.002897, +0.004085] | -0.000707 [-0.001583, +0.000076] | -0.466580 [-0.767393, -0.168186] |
| RedHatAI | +0.003595 [+0.000196, +0.007083] | +0.002521 [+0.001673, +0.003356] | +0.358073 [+0.084997, +0.637184] |
| Unsloth | +0.004600 [+0.001553, +0.007663] | +0.003462 [+0.002848, +0.004077] | +0.061487 [-0.227865, +0.356264] |

[所有配對區間 CSV](expanded/analysis/fast_pairwise.csv)／[Markdown](expanded/analysis/fast_pairwise.md)；[區間圖 PNG](expanded/analysis/fast_intervals.png)／[PDF](expanded/analysis/fast_intervals.pdf)。

![Fast 配對區間](expanded/analysis/fast_intervals.png)

- **RadixArk BF16-LMHead：** Fast 的整體 KL 較低；三種 context 的探索性 95% 區間都在 0 以下，移除任一領域後仍成立。288 文件中 194 份的 KL 較低。不過中文單領域 KL 區間跨 0；整體 NLL／PPL 和 Flip 也跨 0。可支持較接近 BF16 的 KL，不能支持全面品質優勢。
- **NVIDIA：** Fast 整體 Flip 低 0.46658 個百分點，校正區間排除 0，移除任一領域後仍維持。差距主要在中英文知識；程式碼與數學的 Flip 區間跨 0。KL 的一般 95% 區間雖較低，校正區間跨 0，且移除英文或中文領域後也會變得不確定。NLL／PPL 無明確差異，不能把 NVIDIA 排成全面較差。
- **RedHatAI：** 在固定的整體資料比例下，Fast 的 NLL／PPL、KL、Flip 均較高，校正區間都排除 0。KL 在四個領域、三種長度的點估計及探索性區間一致較高，移除任一領域仍成立；221/288 文件的 KL 較高。NLL 結論較敏感：移除英文領域後的 95% 區間跨 0；程式碼與數學也沒有明確的 NLL／Flip 差異。
- **Unsloth：** Fast 整體 NLL／PPL 與 KL 較高，校正區間排除 0，移除任一領域後仍成立。KL 在各領域與三種長度都較高，238/288 文件較高；NLL 的程式碼與數學分組仍不確定。整體及所有單一領域／context 的 Flip 區間均跨 0，不應按 0.06149 個百分點的點估計差距判定勝負。

分組 95% 區間屬探索性結果，沒有對所有分組再做多重比較校正；不能把個別 cell 的符號當成已確認的差異。[移除單一領域分析](expanded/analysis/leave_one_domain_out.csv)、[逐文件勝負與分位數](expanded/analysis/document_stability.csv)。另外用 seeds 20261017、20261018 重做主要校正區間，12 項判讀均一致；這是 Monte Carlo 穩定性檢查，並非額外獨立資料驗證。[seed 敏感度](expanded/bootstrap_seed_sensitivity.json)。

## 與第一輪比較

第一輪只有 8 份自製文字、432 個計分位置，BF16 PPL=12.94307238；本輪 BF16 PPL=3.67269657。語料與 context 改變，不能把絕對 PPL 的下降視為模型品質改善。第一輪 NVIDIA、Unsloth 的 PPL 點估計低於 BF16；本輪所有量化候選的 PPL 點估計都高於 BF16。

Fast 對 RedHatAI 的 Flip 點估計方向翻轉：第一輪 Fast 7.87037% < RedHatAI 8.79630%；本輪 Fast 4.79782% > RedHatAI 4.43974%，且本輪校正區間排除 0。Fast 對 Unsloth 的 Flip 也由第一輪較低改成略高，但本輪區間跨 0。這顯示短 fixture 的排名不足以作為穩定品質結論。

簡化後同時重跑第一輪全部 8 段文字，所有模型的舊分布雜湊與指標重現，因此這些跨輪變化來自資料／context，而非已檢查的擷取介面或程式簡化。[逐指標跨輪比較](expanded/analysis/round1_comparison.csv)、[第一輪報告](REPORT_round1.md)。

## 正確性、開源工具與簡化證據

優先沿用原本已驗證的 NumPy 指標與 worker 流程。直接使用官方 vLLM V2 的 `prompt_logprob_token_ids=range(vocab_size)`、`prompt_logprob_start=C-1`，跳過不計分的前文分布；vLLM 負責 forward 與 log-softmax。已檢查 MIT 授權的 lm-evaluation-harness commit `d6de81643928d653435c431bae19945d41d32520`：其 vLLM adapter 回傳 loglikelihood／greedy match，沒有此處所需的配對完整 vocabulary KL；引入仍需自訂 adapter，故保留較簡單的現有指標。SciPy 負責 bootstrap，PyArrow 負責 parquet。來源、版本與授權見 [開源工具檢查](expanded/TOOL_REVIEW.md)；[SciPy 官方文件](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html)、[vLLM 官方實作](https://github.com/vllm-project/vllm/tree/v0.31.0/vllm/v1/worker/gpu/sample)。

- 完整資料 BF16 在獨立程序、兩張 H100 載入兩次：PPL 完全相同、KL=0、Flip=0，288 份分布雜湊全部相同。[sanity](expanded/full/sanity.json)。
- 官方 dense 與原 flat 路徑：短樣本起點 0／20、實際 2048 context 的 192×248320 分布都逐值相同；長樣本也匹配完整 run 的 reference。原 flat 長樣本曾因全排序 OOM，僅對照 probe 將 GPU KV 預留比例降至 0.75 後完成；正式 dense 一律 0.85。[短對照](expanded/dense_probe.json)、[長對照](expanded/long_equivalence.json)。
- 獨立重新載入 Fast 擷取原始矩陣，對每個 domain/context 各一份文件，直接用 SciPy `special.rel_entr` 及獨立 target／argmax 計算；對主實驗的最大加總誤差 1.78e-15。[獨立公式稽核](expanded/independent_metrics_audit.json)。
- 抽出資料與 calibration 共用的下載／SHA 函式；重建的 tokens、原文及 manifest 三檔逐位元組相同，完整 calibration 稽核輸出也相同。[資料簡化](expanded/dataset_rebuild_audit.json)、[稽核簡化](expanded/calibration_simplification_audit.json)。
- 評分程式移除重複 NaN 檢查、預先 tokenized 路徑中不用的 tokenizer 載入，並共用 vocabulary ID 清單。隨後獨立載入 BF16 兩次，重跑全部 8 模型，每模型 2736 tokens（12 個擴大分組文件 + 第一輪 8 段）。9 程序全部退出 0，**108 份擴大分布 +72 份第一輪分布，共 180 份雜湊全部相同**；擴大文件指標逐值相同，舊指標差小於 1e-12。[GPU 消融證據](expanded/ablation.json)、[簡化 diff](expanded/fidelity_simplification.diff)。
- 分析移除一一對應領域的重複 language 分組；保留的 8 模型整體、160 個分組列、960 個區間列及配對／敏感度結果全部相同。[分析簡化證據](expanded/analysis_simplification_audit.json)。
- 7 個針對性 CPU 測試通過，涵蓋 shift／target、完整 vocabulary、KL 方向、遮罩維度、token 加權、配對文件相關性與 context 排除。[測試輸出](expanded/targeted_tests.txt)；未執行無關的完整測試套件。

完整 55,296-token 評估及最終簡化後的 2,736-token 消融是兩個清楚分開的實際 run；來源雜湊與日誌都保留。初版重疊資料及被中止的 BF16 run 保留於 `pre_decontamination/`，沒有混入最終表格。

## 重跑與範圍限制

```sh
scripts/run_expanded.sh build/reproduction 0 1 2 3 4
```

單張 GPU 可只指定 `0`。安裝與個別資料／分析命令見 [README](../README.md)。[資料集包 2.2 MiB](fidelity-public-v2.1.tar.gz) 包含公開原文、固定 IDs、版本與授權，可直接保留與分享；來源均公開，本次沒有上傳外部平台。

這是這些固定 checkpoint 在本機後端上的文字 fidelity，不能直接推論真實問答、程式正確率或數學推理正確率。數學來源主要是 Lean 形式化證明，程式碼主要是 Python 測試；知識只抽固定 Wikipedia shard。不同長度使用不同文件，無法將長度差當成相同 target 的因果效應。最長初始前文 2048，不涵蓋模型的極長 context 或多模態。

H100 的 NVFP4 使用 Marlin W4A16，混合 FP8 層仍可能使用 W8A8 activation quantization；不是 Blackwell 原生 W4A4，也不能把差異全歸因於權重捨入。[vLLM Marlin 說明](https://docs.vllm.ai/en/latest/api/vllm/model_executor/kernels/linear/nvfp4/marlin/)。BF16 KV 是公平比較的固定條件，可能不同於個別模型預設配置。本次沒有速度／記憶體效益 benchmark；執行時間包含載入與完整分布處理，不可當成吞吐量排名。

文件 bootstrap 區間反映這些固定來源與領域比例下的取樣變異，不涵蓋其他來源、未公開 calibration、重新量化的隨機性或所有部署情境。可支持的結論是上述有條件、分指標的差異，沒有證據支持 Fast 或任何 NVFP4 模型全面勝出。
