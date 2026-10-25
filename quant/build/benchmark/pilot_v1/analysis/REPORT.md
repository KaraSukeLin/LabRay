# IFEval／MMMU-Pro 代表性 pilot 與正式成本預估

本報告只包含 MTP Off、thinking Off 的 pilot；完整 36,009 次能力評估尚未執行。

每模型 200 題，九模型 1800 次主要推論。
IFEval 78 題、Standard 61 題、Vision 61 題。
MMMU-Pro 兩種格式各以每學科兩題的隨機樣本覆蓋 30 科；另列極端題。IFEval 依指令數與字數要求分層。
IFEval 初步區間過寬，因此將已觀測 38 題逐題計入，再以剩餘 503 題的獨立隨機 40 題估算。
條件估算為已知總秒數＋503×追加樣本平均；bootstrap 將已知 38 題固定，重抽新樣本。
MMMU 使用各學科母體權重；兩題極端檢查不加入成本加權平均。

硬體：NVIDIA H100 80GB HBM3、driver 570.86.10；本次使用 5 張。TP=1、每 GPU 一個模型、每模型逐題執行。
沿用官方評分器、Qwen non-thinking sampling、BF16 KV、官方影像解析度與 eager kernels。

## 實際 pilot

分數是這批含極端題的原始正確數，不是完整 Benchmark 的能力排名。時間為主要推論，不含長上限探測。

| 模型 | IFEval 正確數；推論分 | Standard 正確數；推論分 | Vision 正確數；推論分 | 載入與初始化合計秒 | 程序總分 |
|---|---:|---:|---:|---:|---:|
| Qwen BF16 | 58/78；22.02 | 26/61；30.49 | 25/61；35.31 | 101.1 | 208.13 |
| Qwen FP8 | 58/78；31.65 | 33/61；38.93 | 27/61；51.63 | 89.9 | 289.20 |
| NaviQuant Fast | 55/78；27.70 | 26/61；39.25 | 25/61；46.31 | 92.1 | 261.40 |
| NVIDIA NVFP4 | 61/78；27.01 | 31/61；30.42 | 21/61；47.00 | 91.5 | 257.03 |
| RedHatAI NVFP4 | 56/78；28.65 | 32/61；41.69 | 26/61；44.92 | 94.9 | 275.05 |
| Unsloth NVFP4 | 62/78；29.33 | 29/61；35.80 | 26/61；42.43 | 88.4 | 246.20 |
| RadixArk BF16-LMHead | 57/78；27.40 | 26/61；36.22 | 29/61；44.75 | 85.3 | 264.84 |
| PrismaAQUA 5.5-bit | 61/78；28.57 | 29/61；32.54 | 23/61；42.39 | 99.4 | 247.37 |
| ModelOpt AutoQuantize 5.5-bit | 60/78；29.65 | 28/61；33.90 | 29/61；46.23 | 81.9 | 267.40 |

九模型程序合計 38.610 GPU-hours；5 GPU 實際整體 wall time 9.043 小時。
程序總時間包含載入、前處理、主要推論、官方評分、長上限探測及結束程序。

## 完整九模型預估

表格為一般情境的 GPU-hours；單 GPU 按同設定逐題執行時，其數值也等於 wall hours。載入等固定成本每模型只計一次。

### mmmu_1024（IFEval 維持 4096）

| 模型 | IFEval 541 題 | Standard 1730 題 | Vision 1730 題 | 合計 GPU-hours |
|---|---:|---:|---:|---:|
| Qwen BF16 | 1.98 | 14.41 | 16.81 | 33.20 |
| Qwen FP8 | 2.72 | 18.40 | 24.60 | 45.73 |
| NaviQuant Fast | 2.48 | 18.65 | 22.04 | 43.17 |
| NVIDIA NVFP4 | 2.39 | 14.33 | 22.33 | 39.05 |
| RedHatAI NVFP4 | 2.64 | 19.71 | 21.24 | 43.58 |
| Unsloth NVFP4 | 2.61 | 16.96 | 20.14 | 39.71 |
| RadixArk BF16-LMHead | 2.46 | 17.13 | 21.27 | 40.86 |
| PrismaAQUA 5.5-bit | 2.50 | 15.32 | 20.05 | 37.87 |
| ModelOpt AutoQuantize 5.5-bit | 2.76 | 16.06 | 22.12 | 40.93 |
| 九模型合計 | 22.52 | 150.97 | 190.61 | 364.10 |

### mmmu_4096（IFEval 維持 4096）

| 模型 | IFEval 541 題 | Standard 1730 題 | Vision 1730 題 | 合計 GPU-hours |
|---|---:|---:|---:|---:|
| Qwen BF16 | 1.98 | 28.75 | 38.05 | 68.77 |
| Qwen FP8 | 2.72 | 36.38 | 57.40 | 96.50 |
| NaviQuant Fast | 2.48 | 39.44 | 44.55 | 86.47 |
| NVIDIA NVFP4 | 2.39 | 32.01 | 49.82 | 84.22 |
| RedHatAI NVFP4 | 2.64 | 39.93 | 50.18 | 92.75 |
| Unsloth NVFP4 | 2.61 | 34.41 | 43.09 | 80.11 |
| RadixArk BF16-LMHead | 2.46 | 32.51 | 53.28 | 88.24 |
| PrismaAQUA 5.5-bit | 2.50 | 31.96 | 46.71 | 81.17 |
| ModelOpt AutoQuantize 5.5-bit | 2.76 | 34.03 | 48.96 | 85.75 |
| 九模型合計 | 22.52 | 309.41 | 432.04 | 763.97 |

## 排程與不確定性

長工作優先，每張 GPU 同時只放一個模型；各模型都連續跑三項 tasks，不增加模型載入次數。
樂觀／一般／保守為規劃情境，不是工期保證。樂觀取每模型 bootstrap 2.5% 邊界，
一般取加權均值，保守取各模型 97.5% 邊界再加 25% 未見長尾、競爭及重試餘量。
共同樣本的 bootstrap 保留九模型配對；模型別邊界的總和不是共同 95% 信賴區間。

| MMMU 上限 | 情境 | 單 GPU 小時／占用 GPU-hours | 多 GPU wall 小時 | 整組 GPU 保留成本 GPU-hours |
|---|---|---:|---:|---:|
| mmmu_1024 | optimistic | 345.4 | 77.0 | 384.8 |
| mmmu_1024 | typical | 364.1 | 81.0 | 405.2 |
| mmmu_1024 | conservative | 478.5 | 106.4 | 531.9 |
| mmmu_4096 | optimistic | 690.2 | 153.5 | 767.3 |
| mmmu_4096 | typical | 764.0 | 170.0 | 849.8 |
| mmmu_4096 | conservative | 1046.3 | 232.5 | 1162.4 |

## 截斷與長上限探測

| task | 主要回答數 | length 截斷數 | 官方 parser fallback 數 | 平均輸出 tokens |
|---|---:|---:|---:|---:|
| ifeval | 702 | 0 | 0 | 361.2 |
| mmmu_standard | 549 | 193 | 18 | 583.1 |
| mmmu_vision | 549 | 258 | 38 | 722.7 |

4096-token 探測 501 次，包含先前截斷 451 次及控制 50 次。
仍截斷 187 次；原主要回答 prefix 相同 501/501。
對前一級上限的正確性改變 200 次。
高上限成本逐級替換仍截斷的 MMMU 回答；未截斷回答保留，固定控制核對此假設。
延長探測包含特選截斷例及控制，其比例不能直接當成整體截斷率；母體加權率保存在 full_estimates.csv。
4096 探測共用已開啟圖片及模板，MM processor cache 可能已暖身。本分析輸入未包含 8192 探測；不能據此估算 8192 或無截斷的成本。

每題延遲與輸出 tokens 的 mean／p50／p95／max、fallback、empty response、題目難度及圖片數均保留於 CSV。
吞吐量為 output tokens / LLM.generate wall time，包含前處理、prefill 與 decode，不能視為純 decode kernel 速度。
每學科只有兩個機率樣本；bootstrap 無法補出未見長尾、不同 seeds 或硬體差異。仍截斷的較長回答會造成成本下界。

## 可追溯輸出

- `smoke_results.csv`：27 組實際分數、時間與長度分布。
- `question_timings.csv`：所有主要與長上限回答的逐題計時。
- `model_load_and_wall.csv`：權重載入、初始化、推論、評分與程序 wall time。
- `full_estimates.csv`／`full_model_estimates.csv`：各 task／模型的三情境成本。
- `scenarios.json`：多 GPU 逐模型配置、開始／結束時間及整組保留成本。
- `long_cap_pairs.csv`：截斷、答案抽取與正確性變化、token prefix 對照。
- `audit.json`：資料、共同條件、input token IDs 及圖片 SHA 一致性。

另有舊背景控制器誤觸的不完整8K資料，已停止並隔離，未加入本分析；見[事件紀錄](../decision_report/unexpected_8192_stop.json)。
