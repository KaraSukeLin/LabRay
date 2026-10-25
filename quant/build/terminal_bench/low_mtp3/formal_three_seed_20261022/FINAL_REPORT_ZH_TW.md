# LabRay Terminal-Bench 2.1 三模型量化能力保留實驗：最終方法與結果報告

**報告版本：2026-10-23，完成後重新核對版。** 正式模型服務區間為 2026-10-21 18:03:06 至 2026-10-22 23:50:59 UTC；矩陣、Fidelity 與最終彙整於 2026-10-23 00:29:26 UTC 完成。本報告依保存的配置、逐格結果、原 Verifier、attempt 歷史及資源紀錄整理，另增加明確標示的事後敏感度分析。沒有重新執行正式 Trial 或 Smoke。

> **結果適用範圍：固定三個 checkpoints、Thinking Low、MTP3、三個固定 seeds，在共同 PRoot 執行環境中的內部比較。** 主要能力比較使用三模型、三 Seed 全部有有效分數的共同 53 題；並非官方 Docker 環境下完整 89 題的 leaderboard 成績。

## 閱讀摘要

正式設計為 **3 Models × 89 Tasks × 3 Seeds = 801 個矩陣槽位**。801 格均有可稽核終態；89 題均有環境控制結論。60 題通過 Oracle／NOP／Verifier 控制，產生 540 個應執行槽位；其餘 29 題形成三模型、三 Seed 共用的 261 個不可評測槽位。540 格中有 533 格有效分數、7 格個別基礎設施或 Verifier 缺分；包含一次性基礎設施重試，共保存 554 個 attempts。

主分析要求每題的九格都有效，因此使用共同 53 題、每模型 159 個 task–seed 觀察值。BF16 成功 85/159（53.46%）；eNeural Fast 成功 78/159（49.06%）；PrismaAQUA 成功 82/159（51.57%）。以候選平均成功率除以 BF16 平均成功率，點估計保留率分別為 **91.76% 與 96.47%**。

| 模型 | 共同 53 題 × 3 Seed 成功數 | 三 Seed 平均 Pass Rate | 95% task-cluster CI | 相對 BF16 保留率 | 保留率 95% CI | 相對 BF16 差值 |
|---|---:|---:|---:|---:|---:|---:|
| BF16 | 85/159 | 53.46% | 41.51–65.41% | 100% | 參照值 | 0 |
| eNeural Fast（NaviQuant） | 78/159 | 49.06% | 37.11–61.01% | 91.76% | 81.11–102.74% | −4.40 百分點 |
| PrismaAQUA（Aqua） | 82/159 | 51.57% | 39.62–63.52% | 96.47% | 84.27–110.59% | −1.89 百分點 |

兩候選相對 BF16 的配對差值 CI 均涵蓋 0。這組資料支持上述點估計與不確定性範圍；**不足以證明量化沒有能力損失、兩模型等效、非劣於 BF16，或 Aqua 顯著優於 Fast**。本實驗沒有事先設定等效或非劣性界值。

有三項限制必須與數字一起引用。第一，29/89 題未通過環境控制，且 system-administration 類別 9 題全數未進入正式能力比較。第二，7 個缺分格使主分析再排除 7 題，主結果條件化於共同可評分集合。第三，PRoot 共享核心、網路及主機資源，沒有逐題 Docker cgroup 限制；記錄中出現資源壓力、外部下載失敗及清理例外，不能宣稱並行完全不影響成績。

另以 Low BF16 真實 trajectories 做 target-model Fidelity：59 題、234 個窗口、7,488 個評分位置，完整 vocabulary 248,320 維，BF16 獨立重載 sanity 通過。這是固定 BF16 上下文的分布偏移量測，不替代任務成功率。

圖一顯示主要估計與配對不確定性；完整逐題表見 [89 題附錄](report_review_20261023/APPENDIX_TASKS.md)。

![共同 53 題的成功率與配對差值](report_review_20261023/primary_results.png)

## 1. 研究問題、分析單位與預定範圍

研究問題是：在相同 Agent、推論參數、題目、seed 與評分流程下，兩個固定量化 checkpoint 的多輪終端任務成功率，與固定 BF16 checkpoint 相差多少；相同 BF16 trajectory 上的 target-model 分布又相差多少。

正式實驗單位是 `(task, model, seed)`，一格只計入一個依既定規則選出的最終 attempt。同一格因合格基礎設施原因重試，不產生新的矩陣槽位，也不以「任一 attempt 成功」挑選分數。三 Seed 是三次固定隨機條件下的 Pass@1 觀察；不把三次中最好一次當作主成績。

先前已完成的 30 個 Smoke（3 模型 × 5 題 × 2 個 interleaved 模式）、Oracle／NOP 控制、推論診斷、Fidelity 均不計入 801 格。正式模式只用 `interleaved_thinking=False`。Smoke 的 True 模式不與正式結果合併。

原本單 Seed 的 267 格設計，依使用者明確指示恢復三 Seed。固定 seeds 為 **20261020、20261021、20261022**，所有模型一致。補正過程與原配置快照都保留，沒有依觀察到的正式成績更換 seed；這也不是整個研究從未修訂的外部預註冊。依優先順序完成第一 Seed，之後執行剩餘 Seed；尾端允許不同 Seed 的既有基礎設施重試填入空卡。

協定以 [PROTOCOL.json](PROTOCOL.json)、[PROTOCOL_AMENDMENT.json](PROTOCOL_AMENDMENT.json) 與 [三 Seed 配置](CONFIG_THREE_SEEDS.json) 為依據。實際每次啟動再以各 Seed 的 `PLAN.json`、Trial `config.json` 及 wire API audit 核對。

## 2. 固定來源、模型與推論設定

### 2.1 資料、Agent 與執行軟體

| 項目 | 本次固定值／證據 |
|---|---|
| Dataset | Terminal-Bench 2.1，89 tasks |
| Dataset revision | `7131e4375048a0e408a8fb404b5f499d726b695b` |
| Harbor | 0.24.0；revision `96ff9c6db90cc015f0fc0987d896670c9c5f1fad` |
| Agent | 原 Harbor Terminus-2 2.0.0，多輪終端工具流程及原 JSON parser |
| 任務映像 | 原 89 個 image tags 逐一解析並固定 registry digest；見 [image manifest](../../../../assets/terminal_bench_images.json) |
| PRoot | 5.5.0；binary SHA-256 `db185300a9daabbce86aca7632d8c7fdc4249f6f153171954013b70c5f491ec0` |
| GPU | 五張 NVIDIA H100 80GB HBM3，每個模型服務 TP=1 |
| vLLM／Torch | 0.31.0+cu129／2.13.0+cu129 |
| Transformers／compressed-tensors | 5.17.0／0.17.0 |
| Harbor 依賴 | 獨立 `.harbor-venv`；保存 [freeze](../../harbor.freeze.txt)；例如 LiteLLM 1.93.0、OpenAI client 2.24.0 |
| CPU／記憶體總額 | cgroup CPU quota 50 cores、RAM 上限 1,000 GiB；各 Trial 共享，並非逐題配額 |

官方題目的 `task.toml`、instruction、solution、environment 與 tests 按固定 manifest 保存。此次報告重新核對 **1,035 個 manifest 所列檔案，共 58,337,992 bytes，SHA-256 無差異**；核對結果見 [REVIEW_AUDIT.json](report_review_20261023/REVIEW_AUDIT.json)。這項檢查證明當前保存的題目檔案與固定 manifest 一致；不等於對整個執行期間每一個程序行為做了獨立第三方認證。

映像 digest 固定了初始檔案系統，沒有固定網路上的 apt repository、套件下載站或服務回應。正式紀錄確實發生外部下載失敗，因此固定 image digest 不足以保證不同日期重跑的環境完全相同。

### 2.2 三個 checkpoints 的身分

| 報告簡稱 | 固定本機 checkpoint | 保存 revision | safetensors 合計 bytes |
|---|---|---|---:|
| BF16 | `/root/model/Qwen-Qwen3.8-27B` | `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` | 55,563,006,776 |
| eNeural Fast／NaviQuant | `/root/model/eNeural-Qwen3.8-27B-fast` | `284b37dd7b7a5e0071268c0f17fbbb47e4f09b17` | 23,752,003,280 |
| PrismaAQUA／Aqua | `/root/model/rdtand-Qwen3.8-27B-PrismaAQUA-5.5bit-vllm` | `6dc090346f4f32acc320e48d9c413ea96a98d4c6` | 23,603,517,768 |

名稱是本地既定 checkpoint 識別，不以名稱推定所有層的實際位元數。候選使用既有 compressed-tensors 混合精度格式；config 中 Fast 有 208 個 FP8 group targets、192 個 FP4 group targets，Aqua 有 295 個 FP8 targets、338 個 FP4 targets。這些是 config target 項目數，**不是參數加權的平均位元數**。FP8 group 使用 channel weights、dynamic token activations；FP4 group 記載 tensor-group、group size 16。完整 ignore lists、MTP tensors、權重逐檔 SHA 及 config SHA 見 [model-audit.json](../model-audit.json)。

H100 上 FP4 層的已驗證執行路徑是 **Marlin weight-only**；不能把 checkpoint 的 FP4 activation metadata 當成本次硬體原生 FP4 計算證據。三模型 Low／MTP3 runtime 分別有保存的 `preflight/<model>/RUNTIME.json`，均通過；正式服務也做 model／TP／MTP／KV 設定核對。本輪沒有重校準、搜尋 bit allocation 或依任務成績修改權重。

全部模型服務明確共用 BF16 checkpoint 的 tokenizer 與已驗證 chat template。需揭露的是：model audit 已記錄 BF16／Fast 本機 `chat_template.jinja`、`generation_config.json`、`tokenizer_config.json` 的 source CRC 與來源 inventory 不同；共用 tokenizer 則符合本實驗固定 manifest。故本報告不聲稱 tokenizer/template 與最初下載包逐 byte 完全相同。這些是沿用的本地固定資產；模型權重指紋與模板指紋分別保存，完整 prompt token IDs 亦可核對。

這是一個 **固定 checkpoint 與其實際執行路徑的整體比較**。候選可能同時有量化配置、數值 kernel 與 MTP 接受路徑差異；本設計無法把結果唯一歸因於「位元數」或某個量化演算法。

### 2.3 共同正式推論參數

| 參數 | 固定值 |
|---|---|
| Thinking | `enable_thinking=True`, `reasoning_effort="low"`, `preserve_thinking=True` |
| interleaved_thinking | `False` |
| Native speculative decoding | `method="mtp"`, `num_speculative_tokens=3` |
| temperature／top_p／top_k | `1.0 / 0.95 / 20` |
| min_p／presence_penalty／repetition_penalty | `0 / 0 / 1.0` |
| 每次 API `max_tokens` | 131,072；reasoning 與後續輸出共用 |
| Model context | 262,144；無 YaRN context extension |
| Harbor max_input_tokens | 131,072，為共用 context 保留上述 output budget |
| Context summarization | 原 Terminus summarization 開啟；proactive margin 8,000；沿用其近似 token counter |
| KV cache | BF16 |
| Serving dtype | BF16；候選量化層仍依其 checkpoint 與 backend 執行 |
| 服務並行度 | TP=1、`max-num-seqs=1`；每 endpoint 一個活動 Trial |
| 服務其他值 | `max-num-batched-tokens=4096`, `gpu-memory-utilization=0.95`, eager，prefix caching 關閉，`generation-config=vllm`，language-model-only，reasoning parser `qwen3` |
| Seed | 每次 API request 使用該槽位的固定 seed；三模型一致 |
| Agent／Verifier timeout | 各題官方 `task.toml` 原值；`timeout_multiplier=1.0` |

`max_tokens` 是每次 API 呼叫的上限，不是整個多輪 Trial 的總 token 額度。因此累計 input/output usage 可以超過單次 context 或 output 上限；不能據此判斷配置違規。實際長對話也可能觸發原 Terminus summarization。

服務命令中的 `--seed 0` 是服務初始化 seed；正式抽樣 seed 由 API request 明確傳入。Runner 在保存的 Trial config 與 wire audit 中逐次核對 seed、model、Low template，以及 engine 實際 sampling 參數，不能只看服務命令的 seed 就認定正式三 Seed 未生效。

三模型有相同 sampling seed，不代表生成 token 或每個後續隨機數位置相同。模型輸出、工具回饋、重試回合及 MTP 接受路徑會使 trajectories 分歧；固定 seed 的作用是固定實驗條件，而不是保證跨模型逐 token 配對或跨平台 bitwise 可重現。

### 2.4 PRoot 與標準 Docker 的差異

每個 Oracle、NOP 及模型 Trial 都從 pinned image 建立自己的 rootfs 與短 `/tmp`；不把已跑完 Oracle 的 rootfs 當作模型起點。PRoot transport 使用獨立 exec receipt、程序 PID 與 start ticks 記錄；原 task、solution 與 Verifier 腳本沒有改寫。

仍需明確區分檔案隔離與標準容器隔離。PRoot 共享主機 kernel、網路與 PID 環境，沒有逐題 CPU、RAM 或 storage cgroup enforcement；image CMD 也不以 Docker PID 1 方式啟動。各 Trial 有自己的 `/dev/shm` 目錄，位於 workspace 檔案系統，不是 Docker tmpfs。DNS 複製進 rootfs，使用選定 terminal devices；不把 host home/model/proc/sys 直接綁入。

另有既有 `/proc/self/exe` 相容處理：只在受信任 Verifier 命令執行時注入 `libpilot-current-exe.so`，以 AT_EXECFN／LD_PRELOAD 提供 own-executable readlink 相容性。**Verifier 腳本檔案未變，執行 transport 有此非標準處理**；見 [terminal_proot.py](../../../../src/terminal_proot.py)。沒有做一組完整 Docker 對照，所以無法量化此環境差異對成績的影響。

## 3. 89 題環境驗證、排除規則與覆蓋偏差

### 3.1 Gate 如何判定

CPU workers 準備 pinned images，執行原 Oracle solution 與 NOP，再交給原 Verifier。通過條件同時要求：Oracle 狀態 PASS 且 reward=1；NOP 狀態 MODEL_FAIL 且 reward=0；有效 CTRF、tests 數大於 0，對照 assertion 合理；官方 timeout 值、task/agent identity 及 cleanup receipt 一致。這項控制驗證的是本環境中的正負對照與評分流程。

已通過且保存完整證據的既有控制沿用；其餘任務以 process workers 續做。每題通過就立即釋放三模型第一 Seed 的工作，不等待 89 題全做完。第二、第三 Seed 共用第一輪最終 Gate ledger，沒有依模型分數各自選題，也沒有重新跑完整 Smoke。

Gate 排除發生於該題正式模型工作釋放之前；排除規則以控制證據決定。主分析從 60 題縮為 53 題則是**正式執行後的完整配對條件**，兩種篩選不可混為一談。

### 3.2 最終環境結論

| Gate 結論 | Tasks | 占 89 題 | 三模型 × 三 Seed 槽位 | 意義 |
|---|---:|---:|---:|---|
| CONTROLS_PASS | 60 | 67.42% | 540 | 有可比的正負控制，應執行正式 Trials |
| ORACLE_CONTROL_FAIL | 28 | 31.46% | 252 | 原控制未能建立有效可比基準，共同排除 |
| SCORING_GATE_FAIL | 1 | 1.12% | 9 | 評分結構／證據檢查未通過，共同排除 |
| 合計 | 89 | 100% | 801 | 每題都有結論 |

28 個 `ORACLE_CONTROL_FAIL` 是既有 ledger 的共同分類，並非 28 個都已證明 PRoot 不相容。具體觀察是 24 題 Oracle/NOP 均得到 MODEL_FAIL、2 題 qemu 任務缺 reward 檔、`query-optimize` Oracle/NOP Verifier timeout，以及 `torch-pipeline-parallelism` Oracle Verifier timeout。`reshard-c4-data` 為 SCORING_GATE_FAIL：控制 reward 雖有 0，但沒有所需 CTRF。原審核檔的 explicit PRoot signals 清單為空，但此欄未完整涵蓋 stdout 線索；本次直接讀取 `reshard-c4-data` stdout，發現 `error: no /proc/self/exe available. Is /proc mounted?`。這是 transport 相容性的直接線索，不能用原清單為空宣稱 PRoot 無關；仍保留原 SCORING_GATE_FAIL 分類。

這些失敗可能涉及映像、執行環境、依賴、網路、控制解法或評分流程。單憑 stack trace 出現 `terminal_proot.py` 不足以定因；也不能把所有排除題都寫成模型不會做。29 題逐題 Oracle/NOP 狀態、reward 與例外保存在 [ENVIRONMENT_VALIDATION_AUDIT.json](ENVIRONMENT_VALIDATION_AUDIT.json)。本版以原控制 ledger 的 native result 路徑建立 [排除附錄](report_review_20261023/APPENDIX_ENVIRONMENT.md)，另列每題 CTRF 成功／失敗數、失敗 test 名稱與原 stdout 連結；[原生控制路徑索引](report_review_20261023/environment_control_evidence.csv) 可直接追到檔案。原 audit 的 evidence_path 有些為推導路徑，實際檔案位置以此 native result 索引為準。

### 3.3 類別覆蓋

| 原題目 category | 原 89 題 | Gate 通過 | 主分析完整配對 |
|---|---:|---:|---:|
| data-processing | 4 | 3 | 3 |
| data-querying | 1 | 1 | 1 |
| data-science | 8 | 3 | 3 |
| debugging | 5 | 3 | 3 |
| file-operations | 5 | 5 | 5 |
| games | 1 | 1 | 1 |
| machine-learning | 3 | 2 | 2 |
| mathematics | 4 | 4 | 4 |
| model-training | 4 | 4 | 3 |
| optimization | 1 | 1 | 1 |
| personal-assistant | 1 | 1 | 1 |
| scientific-computing | 8 | 6 | 5 |
| security | 8 | 7 | 4 |
| software-engineering | 26 | 18 | 16 |
| system-administration | 9 | 0 | 0 |
| video-processing | 1 | 1 | 1 |
| 合計 | 89 | 60 | 53 |

覆蓋缺口不是均勻隨機的：system-administration 完全缺席，security 只剩 4/8 題。共同選題提高三模型內部的配對可比性，不能消除對完整 89 題的外推偏差，也沒有做類別重新加權來假造缺席類別的成績。

## 4. 五 GPU 動態排程、checkpoint 與必要修復

### 4.1 實際派工方式

每張 GPU 由自己的 worker 管理一個 TP=1 服務及一個活動 Trial。SQLite queue 以交易方式 claim，protocol hash 與 `(model, task, mode, seed, attempt)` 唯一限制去重，終態不可由 `finish()` 覆寫。程序身分包含 PID 與 start time，避免中斷恢復時誤認重用 PID。

空閒 worker 依各模型待辦量及現有活動量選模型：優先值為 `remaining/(active+1)`，同模型服務內接續最多三個 Trials，再重新平衡。題目按固定 identity 次序 claim，並非完全隨機排列或嚴格 GPU crossover。因此模型不固定綁一張卡，也不保證所有模型每題經歷相同的主機負載或時間順序。

Trial 完成並清理後，該 GPU 可立即接新工作，不等待其他卡的較慢 Trial。服務 health-ready 後才進入 Agent 執行；rootfs 準備、Verifier 與清理仍占用服務存活時間。MTP metrics 抓取失敗記錄為觀察缺失，不拿來阻擋正式 Trial 啟動。

這是「最多三題共用一次模型載入」的必要小改動；仍有每三題重啟服務的成本，沒有宣稱已完全消除重複載入。原始服務紀錄共有 192 次載入。

### 4.2 修復的證據與範圍

| 修復／調整 | 證據 | 能支持的結論與界限 |
|---|---|---|
| Gate 通過即釋放工作，移除整批等待 | 固定模擬 89 tasks、3 models、5 workers：live queue 570 秒，batch barrier 600 秒 | 此模擬 makespan 減少 5%；不是實測完整 GPU 實驗加速 5% |
| 同模型一次服務最多三題 | 模擬 15 題需 5 次 load；每題重載基線為 15 次 | 證明排程重用邏輯；未執行額外完整矩陣 A/B |
| OCI layer download lock | 8 個並行 cache requests 只做一次已驗證下載 | 避免重複下載及 cache 競爭 |
| queue／snapshot／resume | 初期 21 個相關 targeted regressions 通過 | 支持已測的去重、恢復、釋放與 source snapshot 行為 |
| 窄範圍 pre-agent tmux 基礎設施重試 | 修復前 targeted repro 失敗，修復後 scheduler module 13 tests 通過 | 只允許合格 attempt-0 一次重試，正常 MODEL_FAIL／TIMEOUT 仍不重跑 |
| 跨 Seed 重試填入空卡 | seed3 無 pending、只剩兩卡活動時，把 seed2 既有 retry rows 派到空卡 | 不建立額外模型試次；配置直接讀 frozen seed file，不更改另一 Seed 的共享配置 |
| USD 與 runner-audit 報告修正 | 分別有 3 個 targeted analysis regressions 通過及 source amendment | 缺價格顯示 NA；最終複製正確 parent audit，不改 Trial |
| Fidelity ingestion | 對 `request/response=null` 與重複 ATIF session ID 的定向修復；最後 4 個相關 CPU tests 通過 | 只影響已完成 Low trajectory 的讀取與 task mapping |

以上 tests 是不同時間點、部分重疊的 targeted 測試，不相加宣稱成一個更大的獨立 test suite。Smoke 沒有重跑。初次 source snapshot 缺目錄／證據路徑的 prelaunch 失敗發生於正式 GPU workers 啟動前，保存於 RUNNER_AUDIT；不是模型失敗。

### 4.3 重試規則與 tmux 的正確解讀

正式計分採同格最後的合格 attempt，所有先前 attempts 均保留。正常 `MODEL_FAIL`、`TIMEOUT` 不為改善分數而重跑。可恢復的乾淨中斷與基礎設施錯誤只允許一次，且要求程序清理證據；attempt 1 不再重試。

實際 14 次額外 attempts 全部是 **Agent/API 尚未執行前的 `tmux: command not found` 啟動失敗**：原 native result 中 `agent_result=null`、`agent_execution=null`，且 cleanup `after=[]`。這種事件發生在 BF16、Fast、Aqua：分別 2、5、7 格；Seed 20261021 有 11 格，Seed 20261022 有 3 格，Seed 20261020 沒有同類合格重試。

因此，這 14 格的 tmux 啟動錯誤**不能當作某模型傾向安裝 tmux 的行為證據**。若要評估模型自主下了哪些安裝指令，必須另外讀該模型的 trajectory；本報告不以啟動錯誤推論行為傾向。

原 `PLAN.json` 曾將 `api_or_environment_trial=0` 寫在 policy，而 scheduler 已存在乾淨 infrastructure retry 的行為；此不一致於執行中以 [RETRY_POLICY_AMENDMENT.json](RETRY_POLICY_AMENDMENT.json) 明確記錄，並修復過度排除「有 native finished result 但尚未開始 Agent」的判定。這是一項需揭露的執行中政策釐清，不能描述為從開始到結束都完全沒有修訂。三模型使用相同資格條件，不按成績決定誰重跑。

14 次重試最終為 **8 PASS、4 TIMEOUT、2 MODEL_FAIL**；失敗的重試同樣保留，沒有繼續跑到成功。逐格 attempt0/1 IDs 與最終狀態見 [retry_outcomes.csv](report_review_20261023/retry_outcomes.csv)。

| Seed | BF16 attempts | Fast attempts | Aqua attempts | 合計 attempts | 原槽位數 |
|---|---:|---:|---:|---:|---:|
| 20261020 | 60 | 60 | 60 | 180 | 180 |
| 20261021 | 62 | 64 | 65 | 191 | 180 |
| 20261022 | 60 | 61 | 62 | 183 | 180 |
| 合計 | 182 | 185 | 187 | 554 | 540 |

## 5. 終態、reward 與缺分的定義

### 5.1 801 槽位怎麼結算

| 最終狀態 | 格數 | 是否正式 Trial | 主分析如何處理 |
|---|---:|---|---|
| PASS | 277 | 是 | 有效原 reward=1 |
| MODEL_FAIL | 74 | 是 | 有效原 reward=0 |
| TIMEOUT | 182 | 是 | 保留 timeout；以原有效 reward 計分 |
| VERIFIER_FAIL | 5 | 是 | 缺乏有效評分證據，score 留空 |
| ENVIRONMENT_FAIL | 2 | 是 | 缺乏有效模型成績，score 留空 |
| ORACLE_CONTROL_FAIL | 252 | 否，共同 Gate 排除 | score 留空，不當作模型答錯 |
| SCORING_GATE_FAIL | 9 | 否，共同 Gate 排除 | score 留空 |
| 合計 | 801 | 540 格曾執行正式 Trial | 533 有效分數，268 格無模型分數 |

`801/801 terminal` 表示帳本完整，**不代表 801 次成功執行，也不代表有 801 個有效 reward**。540 個可評測槽位都已執行到最終處置；7 格是明確保存的無分數失敗。未補出不存在的成績。

### 5.2 為什麼 TIMEOUT 也可能得到 1 分

Runner 狀態反映執行過程；score 優先讀原 Verifier 二元 reward。Agent 可能已完成檔案修改但持續操作或等待，達到 Agent timeout 後，原 Verifier 仍判定產物正確。此時保留 `TIMEOUT` 同時記 `reward=1`，不人工改成 PASS，也不因狀態字串把原 1 分抹去。

本次 182 個 TIMEOUT 中，**173 個 reward=0、9 個 reward=1**。因此全 60 題正式有效成功數是 `277 + 9 = 286`，不是 277；分模型為 BF16 `94+4=98`、Fast `88+3=91`、Aqua `95+2=97`。9 個原 reward=1 的 TIMEOUT 恰好都位於主分析 53 題。

現有分析程式對「TIMEOUT 沒有 reward」準備了 score=0 fallback，但本次 182 格都有原始二元 reward，沒有用到該 fallback。VERIFIER_FAIL 即使有 reward.txt=0，只要原評分結構未完成，仍不將該 0 充作有效模型失敗。

### 5.3 七個缺分格及主分析排除題

| Task | 模型／Seed | 最終狀態 | 保存的直接證據 |
|---|---|---|---|
| crack-7z-hash | Fast／20261020 | VERIFIER_FAIL | Verifier 下載 `astral.sh` TLS 錯誤；後續 uvx 不存在，沒有有效 CTRF |
| gpt2-codegolf | Fast／20261020 | ENVIRONMENT_FAIL | send-keys／subprocess 路徑出現 `ValueError: embedded null byte` |
| openssl-selfsigned-cert | Aqua／20261020 | VERIFIER_FAIL | GitHub uv 下載 TLS 錯誤；後續 uvx 不存在，沒有有效 CTRF |
| password-recovery | BF16／20261020 | VERIFIER_FAIL | Verifier `curl: command not found`，後續 uvx 不存在，沒有有效 CTRF |
| train-fasttext | Fast／20261021 | ENVIRONMENT_FAIL | 原 AgentTimeoutError 3600 秒，清理 receipt 尚有 python3 程序；不符合乾淨重試資格 |
| git-leak-recovery | BF16／20261022 | VERIFIER_FAIL | Verifier 下載 `astral.sh` TLS 錯誤；後續 uvx 不存在，沒有有效 CTRF |
| modernize-scientific-stack | Aqua／20261022 | VERIFIER_FAIL | Verifier 下載 `astral.sh` TLS 錯誤；後續 uvx 不存在，沒有有效 CTRF |

這七題每題只有一格缺分，但主分析為完整配對，將該題整組九格排除，確保分母一致。其餘已得分的 56 格沒有刪除，仍存在矩陣和完整 60 題敏感度分析中。缺分不等同缺分機制與模型完全無關；例如 NUL 指令或清理異常可能與模型執行行為有關。因此無法假定這七題是隨機缺失。

本輪沒有將官方 Verifier 的網路安裝步驟改成成功，也沒有擅自重跑這些已完成且不合格的 Trials。逐格 result 路徑、例外與 Verifier 摘錄見 [missing_scores.csv](report_review_20261023/missing_scores.csv)。

## 6. 統計方法與主要結果

### 6.1 分母、保留率與配對指標

令 `Y[t,m,s]` 為原 Verifier 的二元結果。主分析集合 `C` 要求 task 通過共同 Gate，且該題三模型 × 三 Seed 九格全部有有效 score；本次 `|C|=53`。

- 每 Seed Pass Rate：`p[m,s] = sum(t in C) Y[t,m,s] / 53`。
- 三 Seed 平均：`p[m] = mean(s) p[m,s] = sum(t,s) Y[t,m,s] / 159`。
- 相對 BF16 保留率：`R[m] = p[m] / p[BF16]`。
- 配對差值：`Delta[m] = mean(t,s) (Y[t,m,s] − Y[t,BF16,s])`。
- Harmful flip：同題同 Seed BF16=1、候選=0，占全部 159 組的比例。
- Recovery flip：BF16=0、候選=1，占全部 159 組的比例。

所有模型使用同一 53 題和同一三 Seed；沒有把各模型自己的有效題目分母直接相除。保留率的主定義是「平均率之比」。逐 Seed 比值的平均為另一描述量，Fast 為 91.54%、Aqua 為 98.05%，與主數字 91.76%、96.47% 並不相同；不混用這兩種口徑。

### 6.2 不確定性如何計算

正式配置固定 199,999 次 percentile bootstrap。每次抽樣以 **整個 task 為 cluster**：抽 53 題、可重複，同時保留該題全部三 Seed 及 BF16／候選配對，不把 159 格當作彼此獨立的題目。如此保留了同題不同 Seed 的依賴。

平均 Pass Rate、保留率及配對指標均保存 95% CI。主要比較 family 為 2 候選 × 3 指標（accuracy delta、harmful、recovery）共六項；另列 Bonferroni `1−0.05/6=99.1667%` 單項區間。保留率只提供一般 95% CI，不聲稱已納入六項 family。

固定的 bootstrap RNG：平均率分別為 20261030、20261031、20261032；Fast/BF16 配對為 20261020，Aqua/BF16 配對為 20261021。這些是重抽樣 RNG seeds，與正式模型 sampling seeds 的角色不同。詳細欄位見 [paired_intervals.csv](analysis/paired_intervals.csv) 與 [model_summary.csv](analysis/model_summary.csv)。

CI 描述對題目重抽樣的變異，條件化於本次選定的三 Seed、共同可評分題目及執行環境；沒有重抽新的模型 seeds，也無法涵蓋被排除的系統管理題、網路不可重現或未知平台偏差。只有三個 seeds，不宜用 seed SD 作為精確的未來 seed 誤差界限。

### 6.3 每 Seed 結果

| 模型 | Seed 20261020 | Seed 20261021 | Seed 20261022 | 三 Seed 平均 | Seed SD |
|---|---:|---:|---:|---:|---:|
| BF16 | 25/53＝47.17% | 32/53＝60.38% | 28/53＝52.83% | 53.46% | 6.63 百分點 |
| eNeural Fast | 22/53＝41.51% | 30/53＝56.60% | 26/53＝49.06% | 49.06% | 7.55 百分點 |
| PrismaAQUA | 28/53＝52.83% | 24/53＝45.28% | 30/53＝56.60% | 51.57% | 5.76 百分點 |

Seed SD 使用三個 rate 的 sample SD（ddof=1）。Aqua 在 Seed 1、3 高於 BF16，但 Seed 2 較低；這種方向變動是保留三 Seed 的理由，也使單 Seed 結論不穩定。Fast 在此三個 Seeds 的點估計均低於 BF16，但這仍不是超出本次 Seed/題目集合的必然規律。

### 6.4 配對結果與能力保留的意義

| 候選 | BF16/候選皆成功 | BF16 成功→候選失敗 | BF16 失敗→候選成功 | 皆失敗 | 淨成功差 |
|---|---:|---:|---:|---:|---:|
| Fast | 69 | 16 | 9 | 65 | −7/159 |
| Aqua | 70 | 15 | 12 | 62 | −3/159 |

| 候選／指標 | 點估計 | 95% task-cluster CI | 六項 family 調整區間 |
|---|---:|---:|---:|
| Fast accuracy delta | −4.40 百分點 | −10.06 至 +1.26 | −11.95 至 +3.77 |
| Fast harmful flip | 10.06% | 5.66–15.09% | 4.40–16.98% |
| Fast recovery flip | 5.66% | 2.52–9.43% | 1.26–11.32% |
| Aqua accuracy delta | −1.89 百分點 | −8.81 至 +5.03 | −10.69 至 +8.18 |
| Aqua harmful flip | 9.43% | 5.03–14.47% | 3.77–16.35% |
| Aqua recovery flip | 7.55% | 3.14–13.21% | 1.89–15.72% |

「Fast 保留 91.76%」是 `78/85` 的整體成功率比例，**不是 BF16 原本成功的案例有 91.76% 仍成功**。若問後者，Fast 是 `69/85=81.18%`；Aqua 是 `70/85=82.35%`。兩候選都有新增成功與原成功流失，淨分差可能抵銷行為改變。這也說明不能由接近的平均分推論 trajectories 完全保留。

三 Seed 的題目穩定性描述如下，不把「至少一次成功」替代主 Pass@1：

| 模型 | 三次皆失敗 | 僅一次成功 | 兩次成功 | 三次皆成功 | 至少一次成功 |
|---|---:|---:|---:|---:|---:|
| BF16 | 17 | 9 | 5 | 22 | 36/53 |
| Fast | 20 | 7 | 7 | 19 | 33/53 |
| Aqua | 19 | 8 | 4 | 22 | 34/53 |

完整逐題、逐 Seed 差值保存在 [paired_task_seed_differences.csv](analysis/paired_task_seed_differences.csv)。附錄同時顯示 TIMEOUT reward=1 與各缺分，避免只列挑選出的成功或失敗範例。60 題九格矩陣圖另見 [PNG](report_review_20261023/task_matrix.png)／[PDF](report_review_20261023/task_matrix.pdf)，其中未知分數用橘色標示，不混成失敗的 0。

## 7. 事後敏感度分析：分母與缺失處理會改變多少

本節是在本次報告重整時新增的描述性檢查，**不是事先指定的另一個主 endpoint**。不改原 reward，不用假設值填回正式矩陣。

### 7.1 全 60 題的已知分數與未知結果界限

每模型同樣有 180 個 Gate 通過槽位。若未知缺分結果可以是 0 或 1，完整 180 格成功率必在以下範圍：

| 模型 | 已評分格數 | 已知成功 | 未知格數 | 僅已知格成功率 | 全 180 格成功率界限 |
|---|---:|---:|---:|---:|---:|
| BF16 | 178 | 98 | 2 | 55.06% | 54.44–55.56% |
| Fast | 177 | 91 | 3 | 51.41% | 50.56–52.22% |
| Aqua | 178 | 97 | 2 | 54.49% | 53.89–55.00% |

「僅已知格成功率」的分母在三模型間不同，只作資料完整度描述，不取代主配對結果。界限保留共同 60 題分母：下界把該模型未知格視為 0，上界視為 1；這只是未知值可能範圍，不宣稱任何未知格實際答對或答錯。

| 候選 | 全 60 題相對 BF16 比例界限 | 全 60 題成功率差值界限 |
|---|---:|---:|
| Fast | `91/100` 至 `94/98`＝91.00–95.92% | −5.00 至 −2.22 百分點 |
| Aqua | `97/100` 至 `99/98`＝97.00–101.02% | −1.67 至 +0.56 百分點 |

這些是**固定已觀察 60 題與三 Seed 下的缺失結果界限，不是統計 CI**，也未涵蓋另外 29 題。主分析與此範圍不同，顯示完整配對選題確實會影響點估計；不能只挑較有利的一組保留率引用。Fast 的此固定集合上下界低於 100%，與主 task-bootstrap CI 可跨 100% 並不矛盾：兩者回答不同問題。

### 7.2 若把所有 TIMEOUT 都算失敗

原主要 endpoint 尊重 Verifier reward；另以同一 53 題，將所有 TIMEOUT 一律視為 0，得到以下替代定義：

| 模型 | 原 reward 主結果 | TIMEOUT 全算 0 | 替代成功率 | 替代相對 BF16 比例 |
|---|---:|---:|---:|---:|
| BF16 | 85/159 | 81/159 | 50.94% | 100% |
| Fast | 78/159 | 75/159 | 47.17% | 92.59% |
| Aqua | 82/159 | 80/159 | 50.31% | 98.77% |

此表回應對 timeout 結果的解讀疑問；它更換了成功定義，不覆寫正式評分，不拿較高的替代保留率當主結果。可機讀資料見 [sensitivity_60_tasks.csv](report_review_20261023/sensitivity_60_tasks.csv) 與 [timeout_strict_sensitivity.csv](report_review_20261023/timeout_strict_sensitivity.csv)。

## 8. Timeout、失敗與執行完整性

### 8.1 各模型最終正式狀態

此表每模型都是共同 Gate 通過的 180 格；重試只取同格最終狀態。

| 模型 | PASS | MODEL_FAIL | TIMEOUT | VERIFIER_FAIL | ENVIRONMENT_FAIL | 有效分數 | TIMEOUT 比例 |
|---|---:|---:|---:|---:|---:|---:|---:|
| BF16 | 94 | 25 | 59 | 2 | 0 | 178 | 32.78% |
| Fast | 88 | 29 | 60 | 1 | 2 | 177 | 33.33% |
| Aqua | 95 | 20 | 63 | 2 | 0 | 178 | 35.00% |
| 合計 | 277 | 74 | 182 | 5 | 2 | 533 | 33.70% |

TIMEOUT 不全是缺乏能力，也可能是工具等待、下載、編譯、長 reasoning 或共享資源造成；此設計將它們視為在固定 wall-clock 協定下的實際表現，不能把 timeout 差異直接換算為純模型推理能力差異。MODEL_FAIL 則是原有效 Verifier 未通過，沒有為得到成功而重跑。

原分析 `integrity_audit.json` 三 Seed 均通過，對 176、179、178 個 scored slots 檢查了清理與評分必要證據。這個 audit 的涵蓋範圍是 **可計分結果**；它不表示那 7 個已排除失敗格都清理無誤。尤其 train-fasttext 的未乾淨 receipt 明確保留，不以 audit overall passed 掩蓋。

所有服務均保存起訖時間並結束；最後 finalizer 確認 801 格終態與 Fidelity sanity。本次報告又以 native result 重核 540 格的 reward、Low／interleaved／seed／timeout_multiplier，與 CSV 一致；逐題源檔 hash 核對亦無差異。這些是檔案一致性檢查，不是新的 GPU 重跑。

### 8.2 資源競爭與執行事故

下列情況保留原結果並揭露，不以「沒有 OOM」推定沒有影響：

| 紀錄 | 觀察 | 已採處置與限制 |
|---|---|---|
| 2026-10-22 約 13:35–14:19 UTC 遞迴診斷掃描 | 約一個 CPU core，重疊最後一個 Seed2 Trial 及 14 個 Seed3 cells | 停止已核實的 scanner；沒有停止 Trial 或改分。該區間壓力較低，但不能排除 CPU-bound timeout 影響 |
| 2026-10-22 14:54:44–15:45:05 UTC 第二次掃描 | 約一個 core，估計 0.837 core-hours，與 16 個 Trial attempt 時段重疊 | 停止 scanner；僅停止時有部分 counters，不能當作整段負載證據 |
| 2026-10-22 約 16:32 UTC 短時掃描 | 兩命令合計約 10.9 秒，當時五卡有活動 Trial | 沒有測得完整 CPU/I/O counters；無法量化瞬時影響 |
| 孤立 apt-get update 程序 | 10-21 06:40 啟動，10-22 14:32 清理；觀察時約 0% CPU、8 MiB RSS，原 task 不明 | 僅停止確認的孤立 PID；不能把它歸因於某正式模型或更改其分數 |
| train-fasttext 清理例外 | 一個 Python 程序仍出現在 receipt | 該格維持 ENVIRONMENT_FAIL 且不合格重試；沒有把原 timeout 當有效 score |

詳細證據：[RESOURCE_INCIDENTS.json](RESOURCE_INCIDENTS.json)、[第二次掃描](RESOURCE_INCIDENT_20261022_recursive_scan_2.json)、[短時掃描](RESOURCE_INCIDENT_20261022_short_recursive_inspection.json)。受影響 task/attempt IDs 都在原紀錄，沒有因掃描重疊而選擇性刪分或重跑。

監控的 CPU／memory／I/O PSI 讀自主機 `/proc/pressure`；CPU quota、memory 使用量與 events 另讀本工作環境 cgroup。全部監控區間的主機 CPU pressure `some avg10` 峰值依 Seed 為 74.59%、61.41%、19.39%；其時間平均分別約 1.27%、0.92%、0.16%。記憶體 cgroup 使用量峰值為 961.84、999.99、999.99 GiB；其中可能含 page cache，不等同模型 RSS。`memory.events max` 最後值為 0、18、39；監控中 OOM／OOM-kill 均為 0。這表示有觸及記憶體上限的紀錄，不能寫成整段資源完全寬裕。

PSI 是主機層的共享壓力觀察，memory.events／memory.current 是本工作環境共享 cgroup 的 counters；兩者都不是逐 Trial 的資源用量或因果歸因。沒有順序／並行完整交叉消融，無法證明五 GPU 與同時 CPU 控制對三模型完全無差別，也無法精確修正受干擾題目的反事實分數。

## 9. 推論成本與五 GPU 排程效率

### 9.1 成本口徑

| 模型 | 服務 GPU-hours | 原生 input tokens | 原生 output tokens | trajectory 工具呼叫 | MTP accepted/draft tokens | MTP token acceptance |
|---|---:|---:|---:|---:|---:|---:|
| BF16 | 46.27 | 26,117,897 | 3,565,319 | 4,166 | 4,104,692 / 5,800,602 | 70.76% |
| Fast | 49.11 | 23,028,313 | 3,238,634 | 3,930 | 3,537,767 / 5,044,116 | 70.14% |
| Aqua | 49.20 | 25,809,971 | 3,366,349 | 4,186 | 3,695,345 / 5,192,592 | 71.17% |
| 合計 | 144.58 | 74,956,181 | 10,170,302 | 12,282 | — | — |

成本包含所有 554 個 attempts，與主表的 53 題計分範圍不同；GPU-hours 是模型服務從開始到停止的時間，包含載入、Agent 工具等待、Verifier 等待、清理等。沒有加上另屬 Smoke、獨立環境控制或後續 Fidelity 的 GPU 時間。Fidelity 四次 capture 的 worker elapsed 合計約 1,589.43 秒（0.442 個單 GPU worker-hours），只供該檢查自身成本參考，與服務 GPU-hours 不是完全相同的計時邊界。

Token 表來自原 native `agent_result` 累計；工具呼叫由 ATIF trajectory 計數。不是 tokenizer 意義上的「只計最後答案」，也不保證涵蓋被取消而尚未回傳的 API 生成。attempt ledger 保存的 API audit 摘要共 6,544 requests，其中 340 incomplete；這是已記錄值，不補算未知 token。wire API usage、原 prompt/output token IDs 與 native totals 分別保存，兩個口徑不混成同一數字。

MTP counters 取各 Trial 前後的服務 metrics 差值，是 observational statistics，不證明 MTP 與 Off 模式 token 相等或能力不變。accepted tokens 不等於 native output_tokens：兩者的捕捉邊界、原生修正回合與取消請求可能不同，不用前者代替正式輸出 token 數。

各模型總 output tokens 除以原 Agent seconds，BF16/Fast/Aqua 約為 24.15／20.56／21.48 tokens/s；包含多輪 prefill、工具與其他等待，**不是純 decode throughput**。任務早停、失敗與不同 trajectories 也會影響成本，不能據此宣稱量化模型 kernel 本身較慢。成本未有美元價格來源，正式記為 **NA，非 0 美元**；沒有估算電力或憑空套用雲端價格。

### 9.2 服務存活率重算與舊報表修正

舊 aggregate 報表以各 Seed 的 `resources.jsonl` 首尾作 wall denominator，卻把後來重試的 service time 也加進分子。Seed2 監控主檔於 13:44:59 結束，該 Seed 重試服務卻延續至 23:47:15，導致該 Seed 的 allocated fraction 可達 102.36%。原 aggregate 97.0954% 不應直接當完整排程效率引用。

本版在同一全域時鐘，以 **所有 Seed 與重試服務的最早 started_at 至最晚 stopped_at** 為觀察區間；每 GPU 對服務區間取聯集，避免重複計時，五卡合計除以五卡容量。原資料沒有發現同 GPU 服務區間重疊。

| 重算指標 | 值 | 定義 |
|---|---:|---|
| 全域正式服務 wall | 29.79797 小時 | 2026-10-21 18:03:06 至 2026-10-22 23:50:59 UTC |
| 五卡可用容量 | 148.98985 GPU-hours | wall × 5 |
| 模型服務區間聯集 | 144.57953 GPU-hours | 五 GPU 各自區間聯集後相加 |
| 服務存活率 | **97.0399%** | 144.57953 / 148.98985 |
| 未有模型服務的容量 | 4.41032 GPU-hours | 包含切換間隙、開始／尾端不足五個工作等 |
| 模型服務啟動數 | 192 | 已保存的所有正式服務 sessions |
| ready 前時間中位數 | 47.01 秒 | 範圍 39.01–54.02 秒；含初始化與載入 |
| ready 前時間總計 | 8,767.84 秒＝2.44 GPU-hours | 非純磁碟權重讀取時間 |
| 完成槽位吞吐 | 18.12 格／wall 小時 | 540 個有最終處置的正式槽位 / wall |
| 有效分數吞吐 | 17.89 格／wall 小時 | 533 個有效分數 / wall |

97.04% 是服務存在於 GPU 上的時間比例，**不是 GPU 計算 utilization 97.04%**。原監控共有 7,036 個時間點、五卡共 35,180 個 GPU utilization 讀值，簡單樣本平均約 **31.37%**。這種 sampling mean 不按不規則時距加權，而且 Seed2 重試沒有完整獨立監控，不能宣稱為整段精確 utilization 或有效 FLOPs。

Agentic 任務在工具、套件安裝、編譯、Verifier 時可能保持模型載入但 GPU 閒置。圖二顯示服務存活區間，可以檢查派工與模型切換；不能由這張圖單獨判定 GPU 一直在計算。服務模型重用與動態釋放減少可避免的等待，但沒有用重跑正常失敗來填滿 GPU。

![五張 GPU 正式服務時間線](report_review_20261023/gpu_schedule.png)

重算原始區間與公式保存在 [gpu_schedule_recomputed.json](report_review_20261023/gpu_schedule_recomputed.json)、[service_intervals.csv](report_review_20261023/service_intervals.csv)；監控覆蓋見 [resource_monitor_coverage.csv](report_review_20261023/resource_monitor_coverage.csv)。

## 10. Agentic target-model Fidelity

### 10.1 目的、來源與取樣

正式矩陣全部完成後才執行，沒有搶用正式 Trials 的 GPU。來源固定為第一 Seed 的 **BF16 Low 真實 API trajectories**；不把候選自己的自由生成上下文拿來與 BF16 分布對比，也不使用 MTP draft logits。

從 audit 的實際 prompt_token_ids 與 output token_ids 建立 teacher-forced prefix。重建 tokenizer template 後要求 prompt IDs 完全相同，再固定資料。ATIF 的 session ID 在不同 Trials 重複為 `trial__agent`，因此以 `x-terminal-identity` 中 task/trial identity 解出 task，避免 session dictionary 覆寫。`request` 或 `response` 為 null 的不完整事件跳過。

| 取樣項目 | 實際值 |
|---|---|
| 可取樣 tasks | 59 |
| 窗口上限 | 每題 4 個，每窗口 32 個 scoring positions |
| 最終 windows／positions | 234／7,488 |
| 固定取樣 seed | 20261020 |
| Context eligibility 上限 | 65,536；不截斷原 prefix |
| 實際評分起點 context 範圍 | 863–39,434 tokens；完整窗口最長 39,466 tokens |
| 標籤 | reasoning、terminal_command、code、after_tool |
| 精確來源去重 | 相同 event/start/length 的窗口不重複納入 |
| 輸入 token 檔 SHA-256 | `9d4ac6c4c3e588b922d01de81e87006935b107fddda8abaa45f36a0820b38ea4` |
| 原 Fidelity 核心 SHA-256 | `90fcfa567abee03b3af1c90f6f1afcab5e0bccfa5ed3d09f14ca61198a2eb0dd` |

60 個 Gate 通過 tasks 中，只有 `regex-chess` 沒有合格來源：BF16 三 Seed 都 TIMEOUT，且每 Seed 都沒有完整 API 回覆（0 complete、6 incomplete）；Fidelity 只用第一 Seed，也不能從取消回覆捏造 IDs。正式能力比較仍包含 regex-chess，其三 Seed BF16 score 都是 0。Fidelity 的 59 題與主能力比較的 53 題不是同一集合：前者只要求有合格 BF16 prefix，可包括模型任務失敗或其他模型缺分的題目；後者要求三模型九格全有分數。

取樣 manifest 記錄排除：1,406 個非 BF16 audit events、38 個不完整 BF16 events、0 synthetic events，以及 983 個不滿足長度／context 條件的候選 span。最後一項是候選 span 檢查次數，不能寫成另外排除了 983 個獨立 tasks 或 events。共有 699 個 BF16 source audit 檔案被 hash 記錄；不等於 699 個最終窗口。

候選窗口以每題相同最大額度、輪替可用標籤、固定 RNG 選 event/span/start。部分 task 沒有某些標籤；完全相同窗口去重會使最終少於 59×4=236。不同窗口仍可能有相鄰或重疊 token，因此以 task cluster 做不確定性，不把 7,488 個位置當作 7,488 個獨立實驗。

`after_tool` 的實作以 request 中已有 assistant 歷史作為後續回合 proxy，取新生成前 32 個位置；`code` 以 JSON keystrokes 的 lexical pattern 辨識。這些是程式標籤，不是經人工逐段確認的完整語意分類。context 最長實際僅約 39k，不能據此證明 262k 上下文的全部區段同樣保真。

### 10.2 計算與 sanity

使用未改的 `src/fidelity.py`，先 BF16 reference，再獨立重載 BF16 repeat，sanity 通過後才跑候選。四次均為 target-model、**MTP Off、KV BF16** 的 teacher-forced 計算。這與正式任務 MTP3 的自由生成目的不同，不是改變正式 Trials 協定。

每個 scoring position 使用完整 **248,320 維 vocabulary** 的 FP32 log probabilities，以既有 FP64 累計 KL／NLL；沒有 top-k 近似。這裡比較的是未經正式 top-k/top-p 抽樣截斷的 target-model 完整詞彙分布；不是宣稱正式抽樣策略下的有效 token 分布完全相同。KL 方向為 `D_KL(P_BF16 || P_candidate)`，採自然對數。Top-1 flip 比較同一固定 prefix 上機率最高 token 是否不同，不等於實際抽樣 token 是否不同。

Sanity 門檻是 `abs(KL)<=1e-6`、Top-1 flip=0、`abs(PPL_delta)<=1e-6`。本次 BF16 repeat 三項實測均為 0，`sanity.json passed=true`。這證明此次固定輸入的 BF16 reference/repeat 相符，不證明所有 GPU、seeds 或自由生成都 bitwise 相同。

主要 Fidelity 聚合讓每個 task 權重相同：先在 task 內聚合所有窗口、將 NLL/KL/flip 按該 task token 數正規化，再跨 task 平均；PPL 為平均 task NLL 的指數，並非先算每題 PPL 再做算術平均。另保存 token-weighted 結果。95% CI 以整題及其全部窗口為 cluster，SciPy percentile bootstrap 19,999 次、RNG seed 20261016；沒有對此描述性 strata 表宣稱六項 primary family 調整。

### 10.3 整體 Fidelity 結果

| 模型 | Task 等權 PPL | PPL 相對 BF16 增量 | KL BF16→模型 | KL 95% CI | Top-1 flip | Flip 95% CI |
|---|---:|---:|---:|---:|---:|---:|
| BF16 | 1.220972 | 0 | 0 | 0–0 | 0 | 0–0 |
| Fast | 1.232845 | +0.011872 | 0.010500 | 0.009237–0.011908 | 2.454% | 2.079–2.843% |
| Aqua | 1.236823 | +0.015850 | 0.011011 | 0.009623–0.012426 | 2.485% | 1.991–3.015% |

Token 加權 PPL 分別為 BF16 1.220696、Fast 1.232260、Aqua 1.236367；KL 為 Fast 0.010456、Aqua 0.010964；Top-1 flip 為 2.444%、2.471%。權重不同造成小差異，兩種統計都有保存，表頭明確區分。

| Stratum | 覆蓋 tasks | Positions | Fast KL／Flip | Aqua KL／Flip |
|---|---:|---:|---:|---:|
| reasoning | 59 | 3,360 | 0.009744／2.476% | 0.010509／2.767% |
| after_tool proxy | 57 | 1,920 | 0.016592／3.289% | 0.017441／3.454% |
| lexical code | 48 | 1,536 | 0.006291／1.237% | 0.004986／0.781% |
| terminal_command | 21 | 672 | 0.005810／2.381% | 0.007167／2.083% |

這些 PPL 是在 BF16 自己生成的 token 上做 teacher forcing，通常會較低；不是獨立真實語料的泛化 PPL，也不是任務解答的正確性分數。整體 Fast 的 KL/PPL 點估計略低，任務成功率則 Aqua 較高；兩者不矛盾。Fidelity 問的是 BF16 軌跡上局部分布改變，正式任務成功率還受多輪分歧、工具、錯誤恢復及 timeout 影響。沒有做候選間專門配對顯著性檢定，也不能由這個表推論某種 calibration 原因或所有場景品質排序。

結果與完整 CI：[agentic_summary.json](agentic_fidelity/agentic_summary.json)；來源取樣及 exclusions：[trajectory_sampling.json](agentic_fidelity/trajectory_sampling.json)；原始 target 結果與 BF16 repeat：[results.json](agentic_fidelity/results.json)、[sanity.json](agentic_fidelity/sanity.json)。

## 11. 可支持的結論與不支持的說法

| 問題 | 本次證據允許的說法 |
|---|---|
| 是否完成 801 格？ | 是，全部有終態；只有 540 格屬正式可執行 Trial，533 格有有效模型分數 |
| 是否測了完整 89 題能力？ | 89 題都有控制結論；沒有取得完整 89 題能力成績 |
| 量化保留率多少？ | 在共同 53 題，Fast 91.76%、Aqua 96.47%，必須同列 CI 與選題條件 |
| 能否保證 Fast≥90%、Aqua≥95%？ | 不能；相應 95% 下界只有約 81.11%、84.27%，亦未做非劣性設計 |
| Aqua 是否確定比 Fast 好？ | 只能說此主樣本成功率點估計較高；不能聲稱顯著較好或普遍較好 |
| 有沒有對某模型挑分或多試幾次？ | 同一規則只重試合格 pre-agent 基礎設施失敗；14 格都保存兩次證據，不重跑正常模型失敗或 timeout |
| tmux 錯誤代表模型行為偏差嗎？ | 已重試的 14 格發生於 Agent 前，不能作該推論；自主工具行為須另查 trajectory |
| PRoot 是否等同官方 Docker？ | 否；共享資源、程序與相容 shim 等差異已列出，本次未建立 Docker 等效性 |
| 97.04% 是否是 GPU utilization？ | 是服務存活比例；GPU 計算 utilization 的監控樣本平均約 31.37%，兩者不同 |
| Fidelity 是否證明 agentic 能力無損？ | 不能；證明固定 Low BF16 prefix 上可量得分布偏移，且 BF16 repeat sanity 通過 |
| 是否可歸因於量化演算法本身？ | 本次比較固定 checkpoints 及其執行路徑，無法隔離所有 kernel、MTP 與 checkpoint 設計差異 |

本報告的合理結論是：在這組可配對題目與固定執行協定下，兩量化 checkpoint 的平均成功率點估計接近 BF16，但都有原成功案例流失及新成功案例補回；配對差值不確定性仍大，且環境覆蓋有明顯限制。公開引用應使用「PRoot 內部比較」與完整分母，不將結果包裝成官方完整 benchmark 或能力無損證明。

## 12. 稽核入口、歷史修訂與重現方式

### 12.1 建議稽核順序

| 要核對的內容 | 檔案 |
|---|---|
| 正式設計與修訂 | [PROTOCOL](PROTOCOL.json)、[三 Seed amendment](PROTOCOL_AMENDMENT.json)、[三 Seed config](CONFIG_THREE_SEEDS.json) |
| 完成狀態 | [FINAL_STATE](FINAL_STATE.json)、[POST_RETRY_FINALIZATION](POST_RETRY_FINALIZATION.json) |
| 完整 801 格含狀態／reward／score_basis | [matrix_801.csv](analysis/matrix_801.csv) |
| 554 attempts，不隱藏原失敗 | [attempts.csv](analysis/attempts.csv)、[14 次重試對照](report_review_20261023/retry_outcomes.csv) |
| 89 題控制及 29 題排除證據 | [ENVIRONMENT_VALIDATION_AUDIT](ENVIRONMENT_VALIDATION_AUDIT.json)、[共同 Gate ledger](analysis/environment_validation.json) |
| 共同排除逐題附錄與控制索引 | [29 題排除表](report_review_20261023/APPENDIX_ENVIRONMENT.md)、[89 題 Oracle/NOP 路徑](report_review_20261023/environment_control_evidence.csv) |
| 重試逐格人讀表 | [14 格 attempt0/1 對照](report_review_20261023/APPENDIX_RETRIES.md) |
| 本版來源與歷史差異 | [REPORT_SOURCE_SNAPSHOT](report_review_20261023/REPORT_SOURCE_SNAPSHOT.json) |
| 逐題表與題目 timeout | [APPENDIX_TASKS](report_review_20261023/APPENDIX_TASKS.md)、[可機讀 89 題表](report_review_20261023/task_outcomes_89.csv) |
| 主結果與 CI | [model_summary](analysis/model_summary.csv)、[seed_model_summary](analysis/seed_model_summary.csv)、[paired_intervals](analysis/paired_intervals.csv) |
| 逐題配對差值 | [paired_task_seed_differences](analysis/paired_task_seed_differences.csv) |
| 缺分及敏感度 | [missing_scores](report_review_20261023/missing_scores.csv)、[60 題界限](report_review_20261023/sensitivity_60_tasks.csv)、[timeout 替代定義](report_review_20261023/timeout_strict_sensitivity.csv) |
| Runner 修復與 tests | [RUNNER_AUDIT](RUNNER_AUDIT.json)、[TARGETED_VALIDATION](TARGETED_VALIDATION.json)、[retry policy amendment](RETRY_POLICY_AMENDMENT.json) |
| 服務起訖與排程重算 | [service_intervals](report_review_20261023/service_intervals.csv)、[gpu_schedule_recomputed](report_review_20261023/gpu_schedule_recomputed.json) |
| 此報告重新核對 | [REVIEW_AUDIT](report_review_20261023/REVIEW_AUDIT.json)、[REPORTING_CORRECTIONS](report_review_20261023/REPORTING_CORRECTIONS.json)、[SHA256SUMS](report_review_20261023/SHA256SUMS.json) |
| Fidelity | [summary](agentic_fidelity/agentic_summary.json)、[sanity](agentic_fidelity/sanity.json)、[sampling](agentic_fidelity/trajectory_sampling.json)、[results](agentic_fidelity/results.json) |

原始每 Seed 目錄如下。每個 `trials/<trial_id>/trial/` 內保存 `config.json`、`result.json`、`agent/trajectory.json`、Verifier stdout／reward／CTRF（若成功產生），以及 `proot/` exec/cleanup 證據；服務 audit 位於該 Seed 的 `services/gpu*/.../audit/`。Trial 與服務的直接路徑由 CSV 提供，可避免遞迴掃描龐大 rootfs。

| Seed | Run directory |
|---|---|
| 20261020 | [formal_20261021](../formal_20261021) |
| 20261021 | [formal_20261021_seed_20261021](../formal_20261021_seed_20261021) |
| 20261022 | [formal_20261021_seed_20261022](../formal_20261021_seed_20261022) |

### 12.2 如何理解保存的舊欄位與失敗紀錄

為保留稽核鏈，沒有抹掉舊配置或失敗痕跡。部分檔案必須搭配後續紀錄解讀：

1. 通用 `assets/terminal_bench_config.json` 是歷史基礎配置，含 Xhigh／MTP Off；正式 Low 配置與各 Trial config 才是此輪實際設定。CONFIG_THREE_SEEDS/PROTOCOL 中保留的「267」、`stop_after_smoke`、需要確認等歷史文字／旗標，須依使用者後續授權與 PROTOCOL_AMENDMENT 解讀，不可單獨當最終設計。
2. 第一 Seed 原始 `environment_validation.json` 頂層 `completed` 仍是舊值 0；本版依其 **89 個 task 最終 statuses** 重算，與 final state／derived environment ledger 一致。沒有為美化檔案而修改原紀錄。
3. `SOURCE_MANIFEST.json` 是初始來源快照，不代表後續最小修復沒有改過程式。Runner retry、analysis USD、audit provenance 及 Fidelity ingestion 修復分別有 amendment、baseline 或 tests 證據；本版另 hash 當前來源，避免混稱為完全相同 source revision。
4. `agentic_summary.json` 尾端 note 尚保留程式共用的「Small smoke data」說明文字；實際此輪 `trajectory_sampling.json smoke=false`，59 tasks／7,488 positions。依取樣 manifest 與實際數量判定，不將此 note 當成只跑 Smoke；也不因樣本較大就宣稱能做普遍排名。
5. ORCHESTRATOR_FAILURE、POST_RETRY_FINALIZATION_FAILURE 與 `analysis_before_retry*` 都是已保存的歷史錯誤／中間結果，不混進最終表。完成判斷使用新的 FINAL_STATE 與 POST_RETRY_FINALIZATION，以及 801 格與 Fidelity 結果本身。
6. 原 97.0954% aggregate allocated fraction 保留在原 analysis；本版以同一時鐘重算為 97.0399% 服務存活率，明列原因，不直接改寫舊 CSV/JSON。

上述修訂不改任何題目、reward、Verifier 或正式 timeout。本次報告新增的敏感度與排程重算是事後呈現層分析，沒有為得到較好數字回頭改實驗輸入。

### 12.3 只重建分析，不重跑模型

從專案根目錄，可以用原保存的分析程式在新目錄重建原聚合輸出；這會讀既有 raw artifacts，**不啟動 GPU inference**。新 output 必須尚不存在，保留正式報告目錄。

```sh
PYTHONPATH="$PWD/src" .harbor-venv/bin/python \
  build/terminal_bench/low_mtp3/formal_three_seed_20261022/analysis/terminal_formal_multiseed_analysis.py \
  --protocol build/terminal_bench/low_mtp3/formal_three_seed_20261022/PROTOCOL.json \
  --runs \
    build/terminal_bench/low_mtp3/formal_20261021 \
    build/terminal_bench/low_mtp3/formal_20261021_seed_20261021 \
    build/terminal_bench/low_mtp3/formal_20261021_seed_20261022 \
  --output build/terminal_bench/reproduced_three_seed_analysis
```

命令中的 `PYTHONPATH="$PWD/src"` 讓保存的 script 副本匯入同一專案的依賴；也可直接以當前同 hash 的 `src/terminal_formal_multiseed_analysis.py` 執行。命令重建的是原 aggregate；本版特別說明的 GPU 區間重算與敏感度另保存在 review 附件，不聲稱舊 generator 會自動重建本文所有敘述。

排程指標可由 `service_intervals.csv` 重算：每 GPU 對 `[started_at, stopped_at]` 取聯集後相加作分子；全體 `max(stopped_at)−min(started_at)` 乘五作分母。未知分數界限可由每模型 `known_successes` 與 `unknown` 代入第 7 節公式直接重算。本文表格與附件由已完成資料導出，不依賴主觀挑選案例。

完整數據、原 logs、配置、checkpoint identities、修復證據和新報告附件均在原實驗樹內保存。SHA manifest 是本次整理時點的本地完整性索引，不是外部可信時間戳，也不聲稱對沒有來源紀錄的資料提供額外保證。
