# Terminal-Bench 2.1 Low／MTP3 Smoke 報告

**30／30 Smoke 已完成，正式 0／267；依使用者要求暫停等待確認。** 本次為 PRoot 非標準協定下的內部比較，不是官方 Docker Leaderboard 成績。

固定 BF16、NaviQuant=eNeural Fast、Aqua=PrismaAQUA 5.5-bit；三模型五題各跑 False／True 一次。Low、MTP3、seed=20261020、131072 共用 output budget、262144 context、BF16 KV、原官方 Agent／Verifier timeout。False 為主要模式，True 為獨立診斷。Oracle／NOP／預檢不混入 30 個 Smoke。

| 模型 | interleaved | reward=1／5 | 分類 | Agent 秒中位數 | 工具呼叫 | MTP 接受／draft |
|---|---|---:|---|---:|---:|---:|
| Qwen-Qwen3.8-27B | False | 4/5 | {'MODEL_FAIL': 1, 'PASS': 4} | 210.0 | 109 | 43153/58029 |
| Qwen-Qwen3.8-27B | True | 5/5 | {'PASS': 5} | 268.7 | 105 | 39527/50277 |
| eNeural-Qwen3.8-27B-fast | False | 4/5 | {'MODEL_FAIL': 1, 'PASS': 4} | 292.5 | 98 | 43810/63117 |
| eNeural-Qwen3.8-27B-fast | True | 3/5 | {'TIMEOUT': 2, 'PASS': 3} | 270.9 | 158 | 51997/70164 |
| rdtand-Qwen3.8-27B-PrismaAQUA-5.5bit-vllm | False | 4/5 | {'MODEL_FAIL': 1, 'PASS': 4} | 291.2 | 89 | 40499/54489 |
| rdtand-Qwen3.8-27B-PrismaAQUA-5.5bit-vllm | True | 4/5 | {'PASS': 4, 'MODEL_FAIL': 1} | 241.9 | 91 | 34200/45126 |

| 模型 | Smoke service GPU-hours | 完整 API output tokens | reasoning tokens | 完整 API tokens/s | 模型載入次數 |
|---|---:|---:|---:|---:|---:|
| Qwen-Qwen3.8-27B | 0.9489 | 118731 | 81875 | 51.99 | 4 |
| eNeural-Qwen3.8-27B-fast | 1.3643 | 136757 | 99937 | 40.39 | 7 |
| rdtand-Qwen3.8-27B-PrismaAQUA-5.5bit-vllm | 1.0790 | 107858 | 78432 | 42.71 | 5 |

Smoke 模型服務合計 3.3922 GPU-hours；實際跨模型切換 6 次。服務啟動到 health-ready 中位數 47.0 秒，範圍 41.0–52.0 秒；包含初始化與權重載入，並非純磁碟讀取。

| GPU | 取樣平均 utilization | 取樣峰值 VRAM MiB | 模型服務 GPU-hours | 載入次數 | VRAM<500MiB 取樣秒數 |
|---|---:|---:|---:|---:|---:|
| 0 | 23.6% | 78172 | 0.9768 | 5 | 789.6 |
| 1 | 20.8% | 78002 | 0.7046 | 4 | 1759.4 |
| 2 | 17.9% | 77972 | 0.5354 | 2 | 2320.8 |
| 3 | 17.5% | 78240 | 0.7022 | 3 | 1744.5 |
| 4 | 10.9% | 78242 | 0.4732 | 2 | 2546.3 |

Token 成本以全部完整 wire API usage 累計，包含模型格式錯誤後的原生修正回合；另保留 native trajectory totals。完整 API tokens/s 為完整回覆 output tokens 除以這些 API 的總耗時，含 prefill，不是純 decode 速度。中斷 API 無法回收的 token 不捏造，另列 incomplete 計數。

逐題原始結果、token／工具／timeout／MTP 與載入成本在 SMOKE_SUMMARY.json；完整 prompt IDs、parser、history、trajectory、Verifier、cleanup audit 在 SMOKE_AUDIT.json。單次 timeout 的未完成 API 會另外列出，沒有捏造其 output token IDs。

1→2→5 Gate 的實際 Agent 重疊證據在 concurrency-1.json、concurrency-2.json、concurrency-5.json。前段為必要 Gate 而保留 GPU；之後以五卡工作佇列執行，優先接續同模型，佇列清空才換模型。取樣 utilization 包含載入、Agent 工具、Verifier 等待及尾段工作量不足，不能以低於 100% 判定無效排程，也未用無意義重跑填滿 GPU。CPU／RAM／I/O 壓力保留在 resources.jsonl，這些是共享環境量測，不是每題隔離的 cgroup 成本。

4 個 MODEL_FAIL 的官方 assertion 與 2 個 TIMEOUT 詳見 [FAILURES.md](FAILURES.md)。三模型 False 的 extract-elf 覆蓋率皆為 0%（門檻 75%）；Aqua True 的 overfull-hbox 使用了不允許的 younger→newer 替換。BF16 也有相同抽取失敗，不能直接歸因於量化。

原始 Terminus parser 的逐回覆 error／warning 與各模型統計另存 TERMINUS_PARSE_AUDIT.json；這些是模型格式診斷，不覆寫原 Trial reward 或任意修補回覆。

五卡 Gate 已由原始時間證據驗證至少 37.86 秒五 Agent 重疊。原先整批等待曾讓已完成卡閒置；初批四個 Trial 完成並清理後，續跑監督暫停原協調器，保留較慢 Trial／服務繼續執行，讓空閒卡以原 worker 接續工作；所有工作完成後恢復原協調器。詳見 concurrency-5-observed.json、continuation-controller.json 與 continuation.log。沒有重跑原 Trial 或改推論／評分設定。

PRoot 原始環境控制覆蓋為 5／89；其餘 84 題尚未驗證，不以它們不相容作結論。原 Oracle=1、NOP=0、原 Verifier 不改。H100 上 FP4 層採 Marlin weight-only 路徑，非原生 FP4 計算；原量化 checkpoint 與 native MTP3 已由三模型 runtime audit 核對。

本輪每個 model/task/mode 只有一次，不能估計 seed 穩定性，亦不足以隔離多模型並行對模型成績的因果影響。五卡 overlap 與原始 infrastructure 分類是工程證據，不能當作「並行完全不影響結果」的證明。Smoke 五題分數不是 89 題正式能力保留率。Full-vocabulary target-model Fidelity 與正式比較尚待後續階段；沒有以 draft logits 代替 target logits。
