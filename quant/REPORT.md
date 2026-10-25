# Qwen3.8-27B 量化研究完整報告

整合日期：2026-10-21。結果範圍截至 2026-10-20 的已保存紀錄；原始時間戳為 UTC，換算台北時間加 8 小時。本文件為完整交付入口，整合 Fidelity、量化配置、全量 IFEval、補測、執行成本、MMMU-Pro pilot 與後續限制。CSV、原始 JSONL 和 audit 保留作可追查證據。

## 1. 研究摘要與完成範圍

**已完成九模型 Fidelity，以及五模型 × 541 題 × Thinking On／Off × 三 seeds 的完整 IFEval，共 16,230 筆。** 兩個新增自動量化 checkpoint 均已載入、推論與評估；ModelOpt 使用真正的官方 Weight AutoQuantize 搜尋。

Fidelity 中，PrismaAQUA 相對 NaviQuant Fast 有較低 KL 與 Top-1 Flip；Fast 相對本次 ModelOpt AutoQuantize 有較低 KL。這些校正後的分布差異，**尚未在全量 IFEval 轉為可確認的 strict accuracy 排名**。量化模型對 BF16、Fast 對三個量化對照的 strict accuracy 家族校正區間都涵蓋 0；這也不能當作等效性證據。

| 模型 | Off prompt strict | Off instruction strict | On prompt strict | On instruction strict |
| --- | --- | --- | --- | --- |
| Qwen BF16 | 80.838% | 86.451% | 92.237% | 94.724% |
| NaviQuant Fast | 80.776% | 86.731% | 91.990% | 94.604% |
| PrismaAQUA 5.5-bit | 82.070% | 87.410% | 92.483% | 94.964% |
| ModelOpt AutoQuantize | 82.070% | 87.650% | 92.545% | 94.964% |
| Unsloth NVFP4 | 82.317% | 87.650% | 91.805% | 94.524% |

Thinking 配方提升 prompt strict 約 9.49–11.40 pp，但推論時間約為 Non-thinking 的 6.44–7.06 倍。On／Off 同時改變 sampling 與 reasoning 設定，因此此差值屬於整套配方的比較，不能解讀為單獨 thinking 開關的因果效果。

主矩陣原分析記錄 **335.087 worker GPU-hours**；本次核對發現一筆中斷缺 `wall_seconds`，依保存起訖補計後約 **335.182 GPU-hours**。獨立 100K 補測另耗 **6.359 GPU-hours**，補計後合計約 **341.541 GPU-hours**。主矩陣首次至最後一次 worker 橫跨 73.95 小時，包含中斷、暫停和續跑，不能視為連續保留五張卡的時間。三筆原 65K 截斷在 100K 內都自然完成；主結果不覆寫，另列 cap-recovery 敏感度分析，結論未變。

| 工作 | 完成情況 | 可支持的結論 |
|---|---|---|
| 九模型 Fidelity | 每模型 288 文件、55,296 scoring tokens | 固定 checkpoint／H100 路徑的分布保真比較 |
| 五模型正式 IFEval | 30／30 組，各 541 題；16,230／16,230 筆 | 全量、同題、同模式／seed 的配對能力比較 |
| 100K 獨立補測 | 11 個指定 case；10 筆保存、1 筆依指示停止 | 截斷恢復與可重現 missing-final 診斷 |
| 九模型能力 pilot | 1,800 主回答＋501 個 4K 探測 | 流程、截斷與成本估算；不是完整能力排名 |
| 完整 MMMU-Pro | 尚未啟動 | 只有 pilot，沒有全量能力分數 |
| MTP／DSpark | 啟動相容性／小量推論已測 | 尚無代表性能力／加速結論 |
| Terminal-Bench／Agentic Fidelity | 工具與合成路徑部分驗證；PAUSED | Docker 阻礙未解，官方 smoke 0／10，沒有能力成績 |

建議後續保留 BF16 正控制與 Fast、PrismaAQUA、AutoQuantize、Unsloth 四個量化 checkpoint。IFEval 沒有足夠證據淘汰其中任一個；下一階段應補不同能力面向，先解決 MMMU-Pro 的生成長尾與截斷設定，再安排全量。這次文件整理沒有啟動新推論。

## 2. 正式 IFEval 實驗設計與官方來源

五模型使用 Google canonical IFEval 全部 **541 題**，兩模式，各用 **20261016、20261017、20261018**。每個 model／mode 有 1,623 個回答、2,502 次 instruction checks；每個 model 共 3,246 筆。每一個 model／mode／seed 都涵蓋同一組 541 題，沒有將不完整交集或早期 pilot 混入主表。

官方資料／評分來自 [Google Research 固定 commit](https://github.com/google-research/google-research/tree/3ef52111c1fabc301880818f5a547570e926a76b/instruction_following_eval)，revision `3ef52111c1fabc301880818f5a547570e926a76b`。使用官方 README 指定的 GitHub 輸入；Hub 版本的 key 2785 prompt 有差異，未拿來混用。來源複核見[官方流程稽核](build/benchmark/OFFICIAL_REVIEW.md)與[資料 manifest](assets/benchmark/prepared_v2/manifest.json)。

| 參數 | Non-thinking Off | Thinking On |
|---|---|---|
| temperature | 0.7 | 1.0 |
| top_p／top_k／min_p | 0.8／20／0 | 0.95／20／0 |
| presence_penalty／repetition_penalty | 1.5／1 | 0／1 |
| enable_thinking | false | true |
| reasoning effort | 不適用 | xhigh，實際渲染入官方 template 的 system instruction |
| 最後採用的輸出上限 | 16,384 | 65,000 |
| 主矩陣實際上限歷史 | 4,096／8,192／16,384 | 20,480／32,768／49,152／65,000 |
| context | 65,536 | 65,536 |

Sampling 遵循固定版本的 [Qwen 官方配方](https://huggingface.co/Qwen/Qwen3.8-27B/tree/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0)；**並非 greedy**。xhigh 是模型模板中的 soft instruction，不是強制 token budget。已核對本機 vLLM 將 effort 傳入 template，並核對生成引擎實際收到的 SamplingParams，沒有只根據設定檔宣稱生效。保留 EOS，沒有自訂 stop string 來提早結束 reasoning。

共用條件：同一 BF16 tokenizer／chat template，同模式下相同 prompt token IDs；五張 H100 80GB HBM3，TP=1、每個 worker 單 request、`max_num_seqs=1`、eager、prefix cache Off、MTP Off、BF16 KV cache。後段以獨立分片把剩餘工作分派給空卡，各 worker 仍各用單 GPU，不要求所有模型逐題同步。

保存環境：vLLM `0.31.0+cu129`、PyTorch `2.13.0+cu129`、Transformers `5.17.0`、SciPy `1.18.0`、NumPy `2.3.5`、langdetect `1.0.9`；driver `570.86.10`，沿用專案 CUDA compat。完整程式、資料、template 與 checkpoint identity 見[experiment.json](build/benchmark/ifeval_modes_formal_20261017/experiment.json)。

vLLM 官方 `qwen3` parser 分離 reasoning／final，Google IFEval **只評 final answer**；missing-final 不以 reasoning 代填。`reasoning_tokens` 與重新編碼的 `final_answer_tokens` 受分隔符及編碼邊界影響，相加不要求等於原始 output tokens。逐題保留 raw output、生成 IDs、prompt IDs、實際 sampling、final、finish reason、截斷、耗時與 attempt。

## 3. Checkpoint、容量與真正的 AutoQuantize

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

### ModelOpt 真正搜尋的執行證據

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

### 全部九模型的版本

| Checkpoint | 固定 revision／產生方式 |
| --- | --- |
| Qwen-Qwen3.8-27B | `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` |
| Qwen-Qwen3.8-27B-FP8 | `017b9c7af6b5689d5dd426a76e0bc077eb5ca20a` |
| eNeural-Qwen3.8-27B-fast | `284b37dd7b7a5e0071268c0f17fbbb47e4f09b17` |
| nvidia-Qwen3.8-27B-NVFP4 | `482ca0f3832238542f8f5295dde86b5f22711d80` |
| RedHatAI-Qwen3.8-27B-NVFP4 | `2889ddeeda282f17fc9e159a42125e535159a5e5` |
| unsloth-Qwen3.8-27B-NVFP4 | `f0b7c9e722f5565102fff8481c99e4d86ae099c7` |
| RadixArk-Qwen3.8-27B-NVFP4-BF16-LMHead | `009632fef96dd349150baa780c984e62e70e91fe` |
| rdtand-Qwen3.8-27B-PrismaAQUA-5.5bit-vllm | `6dc090346f4f32acc320e48d9c413ea96a98d4c6` |
| ModelOpt-Qwen3.8-27B-AutoQuantize-5.5bit | `locally generated` |

本專案自行產生的量化 checkpoint 只有上述 ModelOpt AutoQuantize；其他為固定公開 checkpoint。NaviQuant／PrismaAQUA／Unsloth 等並未在本機用同一 calibration data 重新量化。因此目前公平性在於評估資料及部署條件對齊，**不是 calibration、搜尋預算、演算法與輸出格式均相同的受控演算法比較**。若要比較 PrismaQuant／其他框架本身，仍須另外預先固定共同 calibration／validation 與資料量，保留每個框架所需處理及計時，再重新產生 checkpoint；本次未執行這項新實驗。

## 4. Fidelity：九模型完整比較

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

### NaviQuant 相對兩個自動量化方案

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

### Fidelity 的多重比較

沿用 SciPy 1.18.0 的 whole-document paired percentile bootstrap，按 domain × context 固定各層文件數，seed=20261016、19,999 次重抽；文件內 192 個位置不拆開重抽。主要指標是 ΔNLL、ΔKL、ΔFlip；PPL ratio／difference 是衍生描述，不另挑選它們作顯著性宣稱。分組與 leave-one-domain-out 區間仍屬探索性結果。

加入兩個對象後，**全部六個對象一併使用 18 項校正**，沒有讓新增比較沿用較窄的舊 12 項區間。這也改變一項舊結論：Fast−RedHatAI 的 ΔNLL 新區間是 `[-0.00001660, +0.00726280]`，已跨零；舊報告的該項 PPL 方向顯著性不能沿用。RedHatAI 的 KL／Flip 優勢仍獲校正區間支持。

[固定分析範圍](assets/fidelity_extension_analysis.json)／[實際統計參數與程式 SHA](build/autoquant/analysis/statistics.json)。

### Fidelity 控制條件與驗證

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

## 5. 完整 IFEval：能力、配對差異與 seed 穩定性

主表採第 1 節的正式 541 題結果。Prompt strict 要求回答的所有指令都通過；instruction strict 是總通過指令數／總指令數。三 seeds 等重；統計以題目為主要單位，不把 1,623 個 seed-question 答案當作 1,623 道獨立題目。

沿用 SciPy percentile **paired question-cluster bootstrap**，共 199,999 次，同題三 seeds 一起重抽，instruction 指標每次重算分子／分母。結果條件於這三個固定 seeds，不能宣稱代表所有 sampling seeds。

最終全量分析統一使用預先指定 **300 項 Bonferroni family**：19 個對比 × 7 指標＝133 個已分析比較，描述性逐-seed 值不計入。一般區間為 95%，家族校正區間為 **99.9833%**；舊 200 題／停止點使用的 91 或 133 family，以及其數值，均不作本次主結論依據。設定與 SHA 見[statistics.json](build/benchmark/ifeval_modes_formal_20261017/analysis_full_20261020/statistics.json)。

### 候選模型相對 BF16

差值單位均為百分點；正值代表候選 accuracy 較高。

| 模式 | 候選−BF16 | Prompt Δ pp | 95% CI | 99.9833% CI | Instruction Δ pp | 99.9833% CI |
| --- | --- | --- | --- | --- | --- | --- |
| off | NaviQuant Fast | -0.06 | [-1.73, +1.60] | [-3.33, +3.04] | +0.28 | [-2.03, +2.64] |
| off | PrismaAQUA 5.5-bit | +1.23 | [-0.43, +2.96] | [-1.91, +4.56] | +0.96 | [-1.36, +3.40] |
| off | ModelOpt AutoQuantize | +1.23 | [-0.43, +2.90] | [-2.03, +4.58] | +1.20 | [-1.08, +3.55] |
| off | Unsloth NVFP4 | +1.48 | [-0.25, +3.27] | [-1.87, +4.95] | +1.20 | [-1.01, +3.60] |
| on | NaviQuant Fast | -0.25 | [-1.23, +0.74] | [-2.09, +1.73] | -0.12 | [-1.44, +1.22] |
| on | PrismaAQUA 5.5-bit | +0.25 | [-0.80, +1.36] | [-1.79, +2.40] | +0.24 | [-1.20, +1.76] |
| on | ModelOpt AutoQuantize | +0.31 | [-0.62, +1.23] | [-1.48, +2.09] | +0.24 | [-1.20, +1.67] |
| on | Unsloth NVFP4 | -0.43 | [-1.36, +0.43] | [-2.22, +1.23] | -0.20 | [-1.63, +1.14] |

所有 prompt／instruction strict 校正區間都跨 0，因此點估計較高也不能宣稱已確認優於 BF16；跨 0 同樣不證明等效。

### NaviQuant Fast 相對三個量化對照

差值方向固定為 Fast 減對象。

| 模式 | Fast−對象 | Prompt Δ pp | 95% CI | 99.9833% CI | Instruction Δ pp | 99.9833% CI |
| --- | --- | --- | --- | --- | --- | --- |
| off | PrismaAQUA 5.5-bit | -1.29 | [-3.02, +0.43] | [-4.62, +1.87] | -0.68 | [-3.00, +1.68] |
| off | ModelOpt AutoQuantize | -1.29 | [-2.96, +0.37] | [-4.56, +1.79] | -0.92 | [-3.08, +1.21] |
| off | Unsloth NVFP4 | -1.54 | [-3.39, +0.31] | [-5.11, +1.85] | -0.92 | [-3.34, +1.46] |
| on | PrismaAQUA 5.5-bit | -0.49 | [-1.60, +0.62] | [-2.67, +1.54] | -0.36 | [-1.79, +0.97] |
| on | ModelOpt AutoQuantize | -0.55 | [-1.66, +0.55] | [-2.90, +1.54] | -0.36 | [-1.93, +1.07] |
| on | Unsloth NVFP4 | +0.18 | [-0.86, +1.17] | [-1.81, +2.16] | +0.08 | [-1.29, +1.42] |

Fast 在 Off 的點估計比三個對照低約 1.29–1.54 pp；On 與對照相差 −0.55 至 +0.18 pp。六組 prompt strict 的一般及校正區間均涵蓋 0。Fidelity 中 Prisma 對 Fast 的 KL／Flip 優勢、Fast 對 AutoQuantize 的 KL 優勢，在這次 IFEval 上都未形成可確認的 strict accuracy 差異；不能由單一分布指標推定能力排名。

### Harmful／Recovery／Answer Flip

Harmful 是 BF16 正確而候選錯誤；Recovery 是 BF16 錯誤而候選正確，分母為全部配對回答。`Accuracy Delta = Recovery − Harmful`。Instruction 版本按全部配對 instructions 計算。Answer Flip 是 final 回答字串不同，**不是語意錯誤率**。

| 模式 | 候選（參考 BF16） | Harmful % [95% CI] | Recovery % [95% CI] | Answer Flip % [95% CI] |
| --- | --- | --- | --- | --- |
| off | NaviQuant Fast | 5.237 [4.067, 6.531] | 5.176 [4.005, 6.470] | 94.208 [92.483, 95.810] |
| off | PrismaAQUA 5.5-bit | 4.868 [3.758, 6.038] | 6.100 [4.744, 7.579] | 93.839 [92.052, 95.502] |
| off | ModelOpt AutoQuantize | 4.375 [3.327, 5.484] | 5.607 [4.313, 7.024] | 95.256 [93.654, 96.734] |
| off | Unsloth NVFP4 | 4.929 [3.820, 6.100] | 6.408 [4.991, 7.887] | 93.222 [91.312, 95.009] |
| on | NaviQuant Fast | 1.972 [1.294, 2.773] | 1.725 [1.047, 2.526] | 97.843 [96.673, 98.891] |
| on | PrismaAQUA 5.5-bit | 1.910 [1.232, 2.711] | 2.157 [1.356, 3.019] | 97.535 [96.303, 98.583] |
| on | ModelOpt AutoQuantize | 1.602 [0.986, 2.280] | 1.910 [1.232, 2.649] | 97.905 [96.796, 98.891] |
| on | Unsloth NVFP4 | 1.910 [1.171, 2.711] | 1.479 [0.924, 2.095] | 97.412 [96.180, 98.521] |

接近零的 accuracy delta 仍可同時存在數個百分點的 harmful／recovery，不能說逐題行為完全保留。全部七指標的一般／校正 CI、19 組對比與逐題 effects 見[完整 paired CSV](build/benchmark/ifeval_modes_formal_20261017/analysis_full_20261020/paired_comparison.csv)及[question effects](build/benchmark/ifeval_modes_formal_20261017/analysis_full_20261020/question_effects.csv)。

### Thinking 與 Non-thinking 的整套配方差異

| 模型 | On−Off prompt Δ pp | 99.9833% CI | Instruction Δ pp | 99.9833% CI | On／Off 純推論時間 |
| --- | --- | --- | --- | --- | --- |
| Qwen BF16 | +11.40 | [+5.85, +16.96] | +8.27 | [+4.46, +12.21] | 6.77× |
| NaviQuant Fast | +11.21 | [+5.79, +17.01] | +7.87 | [+4.38, +11.46] | 7.06× |
| PrismaAQUA 5.5-bit | +10.41 | [+5.18, +15.90] | +7.55 | [+3.96, +11.16] | 6.75× |
| ModelOpt AutoQuantize | +10.47 | [+5.18, +15.98] | +7.31 | [+3.86, +10.83] | 6.68× |
| Unsloth NVFP4 | +9.49 | [+4.31, +15.16] | +6.87 | [+3.34, +10.59] | 6.44× |

兩模式的能力差值區間都支持本次 Thinking 配方較高；代價主要來自 reasoning 長度。這不是相同 sampling 下只切換 thinking 的消融。

### 三 seeds 的描述性穩定性

下表是 prompt strict；SD 為三個 seed 百分比的樣本標準差，僅描述這三個 seeds，不當作獨立題目 CI。

| 模型 | 模式 | Seed 20261016 | Seed 20261017 | Seed 20261018 | Seed 間 SD（pp） |
| --- | --- | --- | --- | --- | --- |
| Qwen BF16 | off | 82.070% | 80.591% | 79.852% | 1.129 |
| Qwen BF16 | on | 93.346% | 91.682% | 91.682% | 0.960 |
| NaviQuant Fast | off | 80.961% | 81.331% | 80.037% | 0.666 |
| NaviQuant Fast | on | 91.867% | 92.791% | 91.312% | 0.747 |
| PrismaAQUA 5.5-bit | off | 83.549% | 81.516% | 81.146% | 1.294 |
| PrismaAQUA 5.5-bit | on | 92.606% | 92.606% | 92.237% | 0.213 |
| ModelOpt AutoQuantize | off | 82.255% | 82.070% | 81.885% | 0.185 |
| ModelOpt AutoQuantize | on | 92.421% | 92.421% | 92.791% | 0.213 |
| Unsloth NVFP4 | off | 82.070% | 81.885% | 82.994% | 0.594 |
| Unsloth NVFP4 | on | 91.312% | 92.052% | 92.052% | 0.427 |

完整 instruction 逐-seed 值與逐-seed 對比保留於[per_seed_accuracy.csv](build/benchmark/ifeval_modes_formal_20261017/analysis_full_20261020/per_seed_accuracy.csv)及[per_seed_comparison.csv](build/benchmark/ifeval_modes_formal_20261017/analysis_full_20261020/per_seed_comparison.csv)。

## 6. 全量 IFEval 的實際時間、tokens 與 GPU 成本

下表主矩陣每個 model／mode 都是同一 541 題 × 三 seeds。純推論時間只加總已保存回答；原分析 worker GPU-hours 加總 attempts 已保存的 wall × GPU 數，另含載入、解析／評分、shutdown、被中斷或重試的工作。末欄另外補入一筆缺 wall 的起訖估值，詳見下段。兩種口徑不可再相加。各模型後期可能分配多張卡的獨立分片，故 model GPU-h 不等於該模型實際日曆工期。

| 模型 | 載入初始化累計秒 | Off 推論 h | On 推論 h | 解析／評分累計秒 | 原分析 GPU-h | 補計後約 GPU-h |
| --- | --- | --- | --- | --- | --- | --- |
| Qwen BF16 | 416.2 | 6.839 | 46.315 | 817.9 | 53.710 | 53.710 |
| NaviQuant Fast | 1328.7 | 9.088 | 64.117 | 825.6 | 74.737 | 74.831 |
| PrismaAQUA 5.5-bit | 927.7 | 8.195 | 55.312 | 795.4 | 66.095 | 66.095 |
| ModelOpt AutoQuantize | 1038.4 | 8.928 | 59.612 | 836.4 | 69.276 | 69.276 |
| Unsloth NVFP4 | 930.4 | 9.279 | 59.763 | 808.5 | 71.268 | 71.268 |

全部載入初始化合計約 4641.5 秒；已保存回答的純推論共 327.449 GPU-h，解析／評分共 1.134 小時。後者是 worker 中的 CPU wall，沒有另算成新增 GPU-hours。100 個 attempt 記錄為 58 ok、41 interrupted、1 partial；中斷次數不能當成模型失敗題數，最終 30 cells 全部齊全。

原 30 小時／150 GPU-h 預算後續已由使用者取消並授權續跑；舊停止點保留作歷史，不代表最終完成度。原 `statistics.json` 的 335.086915 GPU-h 可由 99 筆有 `wall_seconds` 的 attempts 重現；第 100 筆為 Fast 在分片重平衡時中斷，沒有保存新回答且缺 wall。其保存起訖為 2026-10-20 15:17:43.744938 至 15:23:25.103312 UTC，差 **341.358 秒（約 0.094822 GPU-h）**。本報告單獨以起訖補計，得到主矩陣約 **335.181737 GPU-h**；原始 attempt／分析 CSV 未改寫。該筆的 phase timers 未完整結算，不能用來補推論／評分分項。[原始中斷紀錄](build/benchmark/ifeval_modes_formal_20261017/eNeural-Qwen3.8-27B-fast/attempts/1792509463744928816.json)／[成本核對](build/report_consolidation_20261021/audit.json)。

Launcher 衍生欄位只涵蓋部分啟動器範圍，沒有完整包括後期獨立 helper／分片，不能用其較小的 reservation 值取代 worker 總成本。首次 worker 為 2026-10-17 15:50:33 UTC，末次為 2026-10-20 17:47:16 UTC，跨度 73.95 小時。

### 每題純推論時間與吞吐

| 模型 | 模式 | Mean 秒 | p50 秒 | p95 秒 | Max 秒 | Output tok/s |
| --- | --- | --- | --- | --- | --- | --- |
| Qwen BF16 | off | 15.2 | 8.9 | 47.4 | 357.3 | 21.21 |
| Qwen BF16 | on | 102.7 | 41.5 | 402.2 | 2599.9 | 21.32 |
| NaviQuant Fast | off | 20.2 | 11.4 | 60.4 | 647.6 | 16.43 |
| NaviQuant Fast | on | 142.2 | 64.2 | 494.1 | 3816.6 | 16.44 |
| PrismaAQUA 5.5-bit | off | 18.2 | 11.0 | 57.6 | 413.9 | 17.08 |
| PrismaAQUA 5.5-bit | on | 122.7 | 52.6 | 438.2 | 3918.2 | 17.07 |
| ModelOpt AutoQuantize | off | 19.8 | 12.0 | 60.4 | 415.7 | 16.41 |
| ModelOpt AutoQuantize | on | 132.2 | 60.9 | 477.4 | 3725.4 | 16.50 |
| Unsloth NVFP4 | off | 20.6 | 12.2 | 62.4 | 551.5 | 16.16 |
| Unsloth NVFP4 | on | 132.6 | 58.1 | 483.0 | 3519.2 | 16.25 |

BF16 在此 H100／單序列後端比各量化 checkpoint 快；這是實測部署結果，不能推論 FP4 理論運算較慢。H100 的 NVFP4 路徑使用 Marlin W4A16，混合精度的解碼／scale／kernel 與 BF16 路徑不同，並未做 kernel 級消融來分離各原因。Unsloth 也沒有原生 Blackwell W4A4 的硬體條件；其實際時間還受輸出長度影響。此次不是最大服務吞吐量 benchmark。

### 生成長度與異常

以下均為主矩陣原始 output tokens；p99 直接從同一批已保存逐題值按既有 linear percentile 方式計算。Final mean 是 parser 提取後重新編碼的長度。

| 模型 | 模式 | Output mean | p50 | p95 | p99 | Max | Reasoning mean | Final mean | 截斷／缺 final |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen BF16 | off | 322 | 190 | 1014 | 1914 | 7759 | 0 | 324 | 0／0 |
| Qwen BF16 | on | 2191 | 884 | 8417 | 18145 | 56407 | 1892 | 300 | 0／1 |
| NaviQuant Fast | off | 331 | 188 | 1016 | 1994 | 10900 | 0 | 334 | 0／0 |
| NaviQuant Fast | on | 2338 | 1051 | 8074 | 17418 | 65000 | 2025 | 314 | 1／2 |
| PrismaAQUA 5.5-bit | off | 311 | 186 | 989 | 1668 | 7365 | 0 | 313 | 0／0 |
| PrismaAQUA 5.5-bit | on | 2095 | 906 | 7580 | 17159 | 65000 | 1790 | 305 | 2／1 |
| ModelOpt AutoQuantize | off | 325 | 196 | 990 | 1824 | 6793 | 0 | 328 | 0／0 |
| ModelOpt AutoQuantize | on | 2182 | 1002 | 7924 | 15535 | 60635 | 1896 | 286 | 0／2 |
| Unsloth NVFP4 | off | 333 | 197 | 995 | 1973 | 8859 | 0 | 335 | 0／0 |
| Unsloth NVFP4 | on | 2154 | 946 | 7764 | 15327 | 57578 | 1853 | 302 | 0／2 |

Off input mean 約 58.7 tokens、p95 95；On mean 約 98.7、p95 135。On output p95 約 7.6K–8.4K，遠低於個別最高 65K；少數長尾才是補跑和排程的主要不確定性。主矩陣 On 有 3／8,115（0.037%）截斷、8／8,115（0.099%）自然停止但缺 final；Off 兩者皆 0。其餘錯答不能全部歸因於量化，也不因自然停止就保證答案解析完整。

原始計時、逐次嘗試與模式成本分攤見[timing.csv](build/benchmark/ifeval_modes_formal_20261017/analysis_full_20261020/timing.csv)和[attempts.csv](build/benchmark/ifeval_modes_formal_20261017/analysis_full_20261020/attempts.csv)。

## 7. 輸出上限變更、100K 補測與敏感度分析

### 主矩陣的上限一致性限制

執行中依要求提高上限、短暫降至約 20K 後恢復，所有正式列保留實際 `sampling.max_tokens`。早期已自然停止的回答沿用；早期截斷另補跑並保留歷史。主矩陣最後保留的 cap 次數如下，每個 model／mode 共 1,623 筆。

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

| 模式 | Question×seed cells | 五模型 cap 相同 | Cap 不同 | Cap 不同且發生截斷 | 全部實際截斷列 |
|---|---:|---:|---:|---:|---:|
| Off | 1,623 | 1,384 | 239 | 0 | 0 |
| On | 1,623 | 1,453 | 170 | 0 | 3 |

全部 cap 不同的配對 cells 都在各自 cap 前自然停止，但這不能證明所有反事實上限均無影響。**不能把本矩陣描述為每筆輸出上限完全相同**；已對齊的是題目、prompt IDs、sampling 配方、seed、MTP／KV 及評分政策。詳見[逐 cap CSV](build/benchmark/ifeval_modes_formal_20261017/output_cap_histogram_20261020.csv)與[配對一致性 CSV](build/benchmark/ifeval_modes_formal_20261017/output_cap_pairing_20261020.csv)。

早期恢復案例包括 Fast Off q1265 seed16：4K／8K 截斷後在 10,900 tokens 自然停止；Prisma On q2245 seed16：32K 截斷後在 38,472 tokens 自然停止，已核對共同 token prefix。這些是歷史重試，不再列成最終未完成工作。原備份和 prefix 稽核保留於[200 題截斷紀錄](build/benchmark/ifeval_modes_paused_200_20261018/cap_rerun_audit.json)。

### 100K 獨立補測

最終補測固定同題／mode／seed、相同 prompt IDs、sampling、MTP Off、BF16 KV；輸出 cap 為 100,000，context 同時由 65,536 提高為 131,072，兩項變更均明列。補測不覆寫主矩陣。清單 11 個 case 中保存 10 筆；Fast q3048 seed18 依指示停止，沒有重跑答案。11 份 worker attempts 中 8 ok、3 interrupted；兩個被中斷的 worker 已先保存前一 seed 的完整結果，故 attempt status 不等於每題結果狀態。

### 原 65K 截斷的三筆

| 模型 | 題目／seed | 主矩陣 tokens／結束原因 | 100K 補測 tokens／結束原因 | reasoning／final tokens | prompt strict |
|---|---|---:|---:|---:|---:|
| NaviQuant Fast | `ifeval/3439` / 20261016 | 65000 / `length` | 80396 / `stop` | 58987 / 21407 | false → true |
| PrismaAQUA | `ifeval/3439` / 20261017 | 65000 / `length` | 67680 / `stop` | 46738 / 20938 | true → true |
| PrismaAQUA | `ifeval/3439` / 20261018 | 65000 / `length` | 65221 / `stop` | 50964 / 14250 | true → true |

三筆均在 100K 之前自然停止，成功抽出 final answer，沒有截斷且 prompt strict 正確。NaviQuant Fast 的 seed 20261016 由主矩陣的 incorrect 變為補測 correct；兩筆 PrismaAQUA 均維持 correct。三筆的 prompt token IDs 都與主矩陣相同。**沒有再提高上限；正式主矩陣仍保留原始 65K 結果。**

### 原先自然停止但沒有 final answer

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

補測 worker wall 共 **6.359 GPU-hours**，含中止／重複工作；首次至末次跨約 1.71 小時。沒有超過 100K 再補跑。CSV 包含各題原始／補測 metadata 及逐 token 比對：[failure_retry_comparison_20261020.csv](build/benchmark/ifeval_modes_formal_20261017/failure_retry_comparison_20261020.csv)。

### 只替換三筆截斷的 cap-recovery 敏感度

獨立衍生完整 16,230 筆資料，只將上述三筆原 65K 截斷替換為已保存的 100K 回答；其餘列保留，主矩陣不變。重用原分析器、199,999 次 paired question bootstrap 與同一 300 family，沒有額外推論。

| 指標 | 正式主結果 | Cap-recovery sensitivity |
|---|---:|---:|
| Fast On prompt strict | 91.990% | 92.052%（+0.062 pp） |
| Fast On instruction strict | 94.604% | 94.644%（+0.040 pp） |
| Fast−Prisma On prompt Δ | −0.493 pp | −0.431 pp |
| Fast−Prisma On prompt 95% CI | [−1.602, +0.616] | [−1.540, +0.678] |
| Fast−Prisma On prompt 99.9833% CI | [−2.670, +1.540] | [−2.649, +1.561] |
| Fast−BF16 On prompt Δ | −0.246 pp | −0.185 pp |
| Fast−BF16 On prompt 95% CI | [−1.232, +0.739] | [−1.109, +0.801] |
| Fast−BF16 On prompt 99.9833% CI | [−2.095, +1.725] | [−2.033, +1.725] |

所有候選對 BF16 的 strict accuracy 校正 CI 仍涵蓋 0，沒有改變「尚無可確認量化模型排名」的結論。此衍生資料只供品質敏感度，其 timing／GPU-hour 輸出不作成本估算。完整[敏感度分析](build/benchmark/ifeval_modes_formal_20261017/cap_recovery_sensitivity_20261020/analysis/paired_comparison.csv)與[重現說明](build/benchmark/ifeval_modes_formal_20261017/cap_recovery_sensitivity_20261020/README.md)另存作證據。

## 8. 官方評分器限制、驗證與簡化證據

固定 Google revision 的 `LetterFrequencyChecker` 對 canonical q1122／q1129 的 `#`／`!` 參數會隨機改選字母，無法直接評估符號次數原意。既有 runner 對每題使用 `random.seed(int(sha256(id)[:16],16))`，並固定 langdetect seed=0；所有模型／模式／生成 seeds 用同一評分亂數。這能重現官方工具的結果，**無法修正符號指令語意**；本輪未更改官方 checker、未換 revision 或靜默改自製 grader。未固定亂數的 CLI 在這兩題可能不同，不能當成唯一穩定值。

對最終全量資料使用相同逐題亂數 wrapper 重播 Google 官方 CLI：**30／30 cells、16,230／16,230 回答通過**；official CLI responses、parser replays、engine sampling matches 均為 16,230，跨模型／seed prompt token IDs 相同。[最終 audit](build/benchmark/ifeval_modes_formal_20261017/verification_seeded_20261020/audit.json)。主結果保留全部 541 題；舊 200 題排除 grader／missing-final 的分析僅保留作歷史診斷，沒有用來挑選本次排名。

| 已完成驗證 | 證據與範圍 |
|---|---|
| 正式 IFEval 來源、parser、實際 sampling | 全 16,230 回答官方 CLI／engine／prompt audit 通過 |
| Pre-run 五模型雙模式 | 60／60 smoke；兩題×五模型×兩模式×三 seeds，均能載入、解析、評分；不能當代表性能力樣本 |
| Resume／中斷 | BF16 先 1 再補 11；全完成後 resume 不增回答、不重載，原 JSONL SHA 不變；尾行與 timing 原子更新有 CPU 注入測試 |
| Reasoning 計分消融 | 30 個 On smoke 中 15 個若把 raw reasoning 一起評分會變分；正式只評 final |
| 共用引擎與簡化 | engine 配置與原 pilot 等價；AST 消融只改 metadata 原子寫入，不改推論／parser／grader；品質及 CI CSV 相同 |
| IFEval targeted CPU tests | 準備階段 20 passed；200 題階段另 3 passed，驗證 paired question cluster／seed 非獨立與同題交集 |
| Fidelity 正確性 | 獨立 BF16 gate 全 hash 相同；raw-array 指標獨立重算誤差小於 1e−7；原四對象 14 個分析檔 byte 等價 |
| Pilot 官方重評 | 702 個 IFEval、1,599 個 MMMU 回答與官方 grader 一致；全部 541／3,460 prompts、3,778 images 稽核 |
| Pilot 簡化／統計 | 17 個 targeted CPU tests；96 列原 paired 統計不變、18 個 JSONL byte 不變、估算公式陣列等價 |
| 本次報告整理 | 只讀既有結果重算表格與完整性、連結／數字核對；未修改評估原碼，未啟動新 GPU 工作 |

詳細[IFEval 準備驗證](build/benchmark/ifeval_modes_validation/RESULTS.md)、[原理／續跑檢查](build/benchmark/ifeval_modes_validation/validation.json)、[pilot audit](build/benchmark/pilot_v1/official_replay/audit.json)、[pilot 分析驗證](build/benchmark/pilot_v1/decision_report/analysis_validation.json)。各階段僅執行相關 CPU checks，沒有以本次文件整理重跑全測試或 GPU regression。

## 9. MMMU-Pro 與九模型代表性 pilot

下列是先前完成的**單 seed、Non-thinking pilot**，與正式雙模式三-seed IFEval 分開解讀。其目的為測試流程、生成上限及成本；不能以少量樣本準確率替代完整 benchmark。

正式清單九模型排除RadixArk原版。資料與評分器均沿用官方固定來源：Google IFEval GitHub commit `3ef52111c1fabc301880818f5a547570e926a76b`，MMMU-Pro工具commit `268471d0d488258990025331c7528359c324aa25/mmmu-pro`、資料revision `0d7426df4ccb3d8704a992fe6850abee7c262127`。正式母體IFEval541、Standard1730、Vision1730，每模型4001、九模型36009；[資料manifest](assets/benchmark/prepared_v2/manifest.json)。使用Google strict／loose及instruction graders、MMMU作者direct prompt／image mapping／parser／grader；不宣稱逐項重現其示範模型或leaderboard推論設定。

歷史 pilot 實測條件：[config](assets/benchmark_config.json)，thinking=false、temperature0.7、top_p0.8、top_k20、min_p0、presence_penalty1.5、repetition_penalty1、seed20261016，MTP Off、BF16 KV，TP=1、每GPU單模型逐題、eager、prefix cache Off。context65536，max_images35，影像min65536/max16777216 pixels。主上限IFEval4096、MMMU1024，較長探測4096；原始greedy／1M-pixel啟動smoke另存，未混入。

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

MMMU 1K截斷451/1098（41.1%），4K仍187/1098（17.0%），涉及40個題目ID。501探測含451個截斷與50個控制，token prefix全相同，50控制完整tokens與答案相同。IFEval702回答無截斷；8K未完成，也未納入分析；背景誤觸的17個回答已隔離，未納入表格。BF16 Vision25→41而Auto29→32，證明輸出上限足以改變表面排序。

![Fidelity與截斷敏感度](build/benchmark/pilot_v1/decision_report/fidelity_and_cap_sensitivity.png)

配對分析保存Accuracy Delta、Harmful／Recovery／Answer Flip及95%／校正區間。IFEval Answer Flip為字串變化，非語意錯誤率。品質bootstrap199999次，成本19999次；按實際抽樣設計加權、層內保持模型配對。IFEval已知38題固定，追加40題估算剩餘503題；MMMU各學科加權，壓力題權重0。

兩cap統一探索性family為8候選×4指標×5不同task/cap單元＝160。每科兩題可造成bootstrap零經驗變異而區間過窄；額外未加權observed-case exact McNemar檢查的Bonferroni p全部為1。不能以無顯著宣稱等價，也不能單靠小樣本bootstrap排名。[1K配對區間](build/benchmark/pilot_v1/paired_1024/paired_comparison.csv)／[4K母體加權區間](build/benchmark/pilot_v1/paired_4096/weighted_paired_comparison.csv)／[exact檢查](build/benchmark/pilot_v1/paired_4096/accuracy_tests.csv)。Fidelity仍使用原先獨立18項family。

### 實際 pilot 成本與完整矩陣的歷史估算

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

上述估算是完整九模型、**單 seed、Non-thinking** 的 4,001 cases／模型，合計 36,009 回答；不是本次 16,230 筆雙模式 IFEval 的預算，也不是把現有已完成 IFEval 又當待跑工作。新增 Thinking、三 seeds、較高 cap、MTP、較大 batch 或跨 GPU 分題都需另估，不能直接乘減後宣稱已實測。

若僅規劃下一階段 MMMU-Pro，1K 的兩項 task 合計點估計為 341.58 GPU-h，4K 為 741.45 GPU-h（直接相加表內已四捨五入欄位）；排程須依所選模型與實際分片重算。1K／4K 都仍有嚴重截斷，這些僅是有截斷設定下的規劃情境，**不建議直接用作無截斷正式能力實驗的工期承諾**。當前沒有經代表性樣本驗證的更高上限全量 ETA。

舊版週一／週日絕對日期排程已過期，未列為新排程。依目前已完成的 IFEval，下一步以 BF16、Fast、Prisma、AutoQuantize 為研究核心，加 Unsloth 作部署對照；若需要高精度正控制可再加 FP8。NVIDIA 較小容量仍有研究用途，RedHatAI／RadixArk 不因本次精簡就被判定普遍較差。完整模型子集／情境排程仍可查[subset_budgets.csv](build/benchmark/pilot_v1/decision_report/subset_budgets.csv)與[subset_schedules.csv](build/benchmark/pilot_v1/decision_report/subset_schedules.csv)。

Pilot 後曾有殘留控制器誤觸 8K 補測；17 個部分輸出已隔離，未納入評分或成本估算。其額外約 26.9 分鐘、粗估 2.24 占用 GPU-h，與 pilot 的 38.610 GPU-h 分列。[事件紀錄](build/benchmark/pilot_v1/decision_report/unexpected_8192_stop.json)／[隔離資料](build/benchmark/pilot_v1/interrupted_unexpected_8192/README.md)。沒有把未完成 8K 探測當作正式結果。

## 10. MTP／DSpark：已測相容性與尚缺證據

九模型 MTP=2／DSpark=7 共 18 個 startup worker 成功；恢復官方圖片解析度後另有 33 個 greedy smoke 及非零 draft／accepted counters，確認 MTP 實際參與解碼。[相容性矩陣](build/benchmark/compatibility.md)區分公開 model card 證據與本機實測通過情況，沒有以公開聲稱取代測試。

六個 greedy Off／On 配對中四個 token IDs 相同、兩個不同；差異原因尚未單獨定位。該批只有 128 output tokens，四個 MMMU 回答使用官方 parser fallback，分數相同不能證明能力或 token 等價。舊 108 回答 pilot 的圖片上限曾為 1M pixels，亦不能用兩邊不同生成長度來宣稱 MTP 加速。

本次全量 IFEval 一律 MTP Off。沒有代表性 MTP On 的能力或工期估算；DSpark 不加進正式矩陣。後續若測 On／Off，須同 checkpoint、同題／seed、同圖與 sampling 分開配對，保存 acceptance counters，將推論差異與量化差異分開解讀。九模型、三 tasks 各跑 MTP On／Off 的單 seed 工作量為 72,018 回答，**未執行**。[官方流程複核](build/benchmark/OFFICIAL_REVIEW.md)。

## 11. Terminal-Bench／Agentic Trajectory Fidelity 的準備狀態

**目前 PAUSED，正式啟動資格：否。** 兩套工具、配置、原生 Harbor job plan、針對性 CPU 驗證、五張 H100 同時 BF16 serving，以及極小量 full-vocabulary Fidelity 已完成實測。現有工作環境缺 Docker Engine／Compose／socket，無法執行指定官方任務容器與 Verifier；五題 × 兩種 interleaved 協定的官方 Smoke 為 **0／10**。BF16 89 題 Gate、801-trial 正式矩陣與正式 Agentic KL 均未啟動，沒有將合成測試當 Benchmark 成績。

[完整證據與限制](build/terminal_bench/VALIDATION.md)／[可重現指令](README.md)／[最終配置](assets/terminal_bench_config.json)／[準備計畫](build/terminal_bench/prepared_20261020/plan.json)。

2026-10-21 另完成[Docker 環境診斷](build/terminal_bench/DOCKER_DIAGNOSIS.md)。目前容器沒有 Docker CLI／Engine／Compose 或 daemon 入口；缺 SYS_ADMIN／NET_ADMIN，cgroup v2 唯讀，rootless 工具與目前 root 使用者的 subordinate IDs 未備齊。user＋mount＋net namespaces 可建立，但在其內具備完整 capabilities 時，tmpfs mount 仍回 EACCES；enforced AppArmor profile 與 containerd 預設 `deny mount` 規則高度吻合，實際宿主機規則與 audit 尚不可見。此次只診斷，未安裝、啟動服務或執行 GPU 實驗。Harbor 雲端 sandbox 是官方支援的替代方向，尚未適配／實測；本機 Docker 並非 Benchmark 的唯一選項。官方 Smoke 仍 0／10，實驗維持 PAUSED。

同日完成獨立的 [PRoot Feasibility Smoke](build/terminal_bench/diagnostics/proot_smoke/REPORT.md)：本機解包的 Debian PRoot 5.1.0-1.3 與具有效簽章／SHA256 的 Ubuntu Base 24.04.5 通過 20 項 shell、檔案、子程序、timeout、寫入路徑與重建檢查。首輪背景程序因 rootfs 缺 `/dev/null` 失敗，僅明確 bind 該裝置即通過，保留兩輪紀錄與對照。Harbor 外部 BaseEnvironment 介面靜態可接入，但沒有實作 backend 或執行官方 task／Verifier；tmux、正式資源限制、安全隔離與五 trial 併發未獲驗證。建議正式評測優先遠端 sandbox。此次無 GPU 推論，原正式實驗狀態維持 PAUSED。

已固定官方 dataset revision `7131e4375048a0e408a8fb404b5f499d726b695b`、Harbor revision `96ff9c6db90cc015f0fc0987d896670c9c5f1fad`（0.24.0）、全部 89 個官方 image digests 與逐檔 task resources SHA。指定五題的 Agent timeout 600／750／900／900／900 秒一致，未修改任何題目、Oracle、Verifier、資源或 timeout。獨立 `.harbor-venv` 沿用官方 Terminus-2／LiteLLM；既有推論環境與原 `src/fidelity.py` 數值核心未改。

五卡同時 BF16：原生 262,144 context、TP=1、MTP Off、BF16 KV、Qwen Thinking／xhigh、官方 sampling；所有五個 endpoint 正常推論。五卡整波 wall 82.68 秒、0.112785 GPU-h；每卡最高顯存 77,134 MiB，之後正常釋放。全部 14 次非 Benchmark 的 GPU API 請求均通過實際 SamplingParams、seed、prompt IDs 與 reasoning／final parser 重播。False／True 各三輪的終端回饋能連續傳遞：False 有意省略歷史 reasoning，True 則真正渲染入 Qwen prompt；preserve_thinking=True 不能恢復 Agent 根本沒傳入的 reasoning。主協定暫定維持官方 False，True 分開作敏感度測試，不按成功率選擇。

最終共用 output budget 為 131,072，Harbor input 預算 131,072；與 Qwen 在 1M context 下建議分開 reasoning 262,144／final 131,072 的配置不同，沒有加 YaRN。Summarization 原設定開啟、餘量 8,000；短 GPU 測試觸發 0 次。修正 input reservation 的 12 次 CPU HTTP 重播保持 request body 相同，並核對原生閾值。長歷史壓縮及其 ATIF／API 對應尚未實際 GPU 驗證；正式 Gate 若遇 history reset，會先停止 READY。

Agentic Fidelity 已重用原完整 248,320 維 vocabulary 路徑，在同一組實際 BF16 prompt＋generated IDs 上作 teacher forcing。此次只有一個合成 task、三個 windows、12 個 scoring positions；BF16 獨立重載的分布 hash 相同，KL／Flip／PPL delta 全零；Fast／Prisma 均完成。這些樣本只驗證資料及計算，不足以排名，沒有拿 benchmark 資料重新 calibration 或改量化配置。

最終工具準備了 89 題 BF16 repeat-0 Gate＋其餘 712 trials，依官方 Agent timeout 做五個固定分片；已完成整個 job 不重載 GPU，部分結果交給 native resume，未完成 logs 先保留至 interrupted/。正式配對統計以 task 為 cluster 保留三 seeds，六項 primary family；環境／timeout／parse 診斷另列，不剔除失敗改善分數。16 個 targeted CPU tests 通過，沒有執行 full test suite。

本次 GPU 配置成本共約 0.285436 GPU-h（serving 0.241985＋Fidelity 0.043451）。目前不能用合成 printf 預測真實任務耗時。若完整使用所有宣告 Agent timeout，89 題合計 42.2083 GPU-h／五分片最長 8.5 小時；801 trials 為 379.875 Agent GPU-h／五卡 76.5 小時，另加載入、容器與 Verifier，這是規劃口徑而非實測 ETA。25%／50% timeout 使用情境約 95.0／189.9 Agent GPU-h，仍需官方 Smoke 更新預估。

**未解決事項：**可用 Docker／Compose、相同 bind-mount 路徑與資源 enforcement；指定十個官方 Smoke、Verifier、真實 task 多輪與 Docker 併發干擾；極小 real-trajectory ingestion；候選完整長 context API 路徑與實際 summary。原 IFEval 起初占卡時未被中止，五卡測試等它自行釋放後才執行。這項準備工作結束時，相關 workers／inference services 均已停止。狀態保留 PAUSED；補齊官方 Smoke 後，仍須下一次明確授權才可啟動 89 題 Gate。

此節成本與合成測試不加入 IFEval／MMMU 能力分數，亦不計入補計後約 341.541 GPU-h 的 IFEval 主矩陣＋補測口徑。

## 12. 可重現使用、交付索引與後續決策

### 查看完成狀態與 CPU 重分析

沿用現有環境，不重建實驗系統。以下 status／analysis 不進行模型生成；新分析使用新 output，避免覆寫固定結果。

```sh
cd /root/projects/quant
.modelopt-venv/bin/python src/ifeval_modes.py status \
  --output build/benchmark/ifeval_modes_formal_20261017

.vllm-venv/bin/python src/analyze_ifeval_modes.py \
  --runs build/benchmark/ifeval_modes_formal_20261017 \
  --output build/benchmark/ifeval_reanalysis_NEW \
  --resamples 199999 --family-size 300

.vllm-venv/bin/python src/analyze_ifeval_modes.py \
  --runs build/benchmark/ifeval_modes_formal_20261017/cap_recovery_sensitivity_20261020/runs \
  --output build/benchmark/ifeval_cap_reanalysis_NEW \
  --resamples 199999 --family-size 300
```

五 GPU 正式啟動／resume、模型／模式／seed／題目範圍指令見[既有使用說明](build/benchmark/ifeval_modes_validation/USAGE.md)與[README](README.md)。已完成的 run 不需再啟動；如另建正式 run，應先固定共同上限與分析 family，明確保留來源與 config。MMMU、MTP On、Terminal-Bench 全量仍待下一次實驗指示。

| 交付內容 | 主要證據／檔案 |
|---|---|
| 本次完整整合報告 | [REPORT.md](REPORT.md)，含研究摘要、完整比較、成本及限制 |
| 九模型 Fidelity／容量／95% CI | [comparison.csv](build/autoquant/comparison.csv)、[完整統計](build/autoquant/analysis/)、[inventory](build/autoquant/checkpoints/inventory.json) |
| AutoQuantize 搜尋／配置／成本 | [parameters](build/autoquant/search_v2/parameters.json)、[sensitivities](build/autoquant/search_v2/sensitivities.csv)、[audit](build/autoquant/search_v2/search_audit.json)、[log](build/autoquant/logs/search_v2.log) |
| IFEval 原始逐題資料與進度 | [正式 run](build/benchmark/ifeval_modes_formal_20261017/)，每模型 `questions.jsonl`，含 reasoning／final／token IDs／時間；[progress.json](build/benchmark/ifeval_modes_formal_20261017/progress.json)／[CSV](build/benchmark/ifeval_modes_formal_20261017/progress.csv) |
| 全量 IFEval 表格／CI | [accuracy](build/benchmark/ifeval_modes_formal_20261017/analysis_full_20261020/accuracy.csv)、[paired](build/benchmark/ifeval_modes_formal_20261017/analysis_full_20261020/paired_comparison.csv)、[coverage](build/benchmark/ifeval_modes_formal_20261017/analysis_full_20261020/paired_coverage.csv)；目錄內均附 Markdown |
| 每題成本／生成長度與 worker 成本 | 各 JSONL；[timing.csv](build/benchmark/ifeval_modes_formal_20261017/analysis_full_20261020/timing.csv)、[attempts.csv](build/benchmark/ifeval_modes_formal_20261017/analysis_full_20261020/attempts.csv) |
| 100K 補測與 cap 稽核 | [逐題對照](build/benchmark/ifeval_modes_formal_20261017/failure_retry_comparison_20261020.csv)、[cap histogram](build/benchmark/ifeval_modes_formal_20261017/output_cap_histogram_20261020.csv)、[cap pairing](build/benchmark/ifeval_modes_formal_20261017/output_cap_pairing_20261020.csv) |
| Cap-recovery 全量敏感度 | [derived data／analysis](build/benchmark/ifeval_modes_formal_20261017/cap_recovery_sensitivity_20261020/) |
| 官方評分／來源／parser audit | [全量 IFEval audit](build/benchmark/ifeval_modes_formal_20261017/verification_seeded_20261020/audit.json)、[官方來源複核](build/benchmark/OFFICIAL_REVIEW.md) |
| MMMU pilot／工期估算 | [pilot decision report](build/benchmark/pilot_v1/decision_report/DECISION_REPORT.md)、[逐題時間](build/benchmark/pilot_v1/analysis/question_timings.csv)、[全情境預算](build/benchmark/pilot_v1/analysis/full_estimates.csv) |
| Terminal-Bench 阻礙與驗證 | [VALIDATION.md](build/terminal_bench/VALIDATION.md)、[STATE](build/terminal_bench/prepared_20261020/STATE.json) |
| 原始歷史快照 | [200 題快照](build/benchmark/ifeval_modes_paused_200_20261018/)、[30 小時停止點](build/benchmark/ifeval_modes_final_20261018_2241/)、[本次整合前 REPORT](build/report_consolidation_20261021/REPORT_before_consolidation.md) |
| 本次整合核對 | [audit.json](build/report_consolidation_20261021/audit.json)，核對主矩陣、表格、成本、敏感度與本地連結 |

開源來源沿用 Google Research IFEval、MMMU 作者工具、vLLM、NVIDIA Model-Optimizer 與 SciPy；保留既有 LICENSE／固定版本，沒有付費 judge，也未自製替代官方 grader。Harbor／Terminus-2 的版本與來源另列於 Terminal-Bench 驗證資料。實際 license 與來源檔以各固定 checkout 內附檔為準。

後續決策所需的關鍵事實是：IFEval 已完成且不足以確認量化模型排名；Fidelity 仍能指出分布保真差異；MMMU-Pro 現有 4K pilot 仍有 17% 截斷，尚需決定可接受的正式輸出設定；MTP On 和 Agentic benchmark 尚無可直接引用的全量能力結果。所有舊停止點保留作執行歷史，不再當成當前完成度或剩餘 ETA。
