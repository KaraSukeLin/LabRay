> 2026-10-21 Low／MTP3 Smoke 已完成並暫停：30／30，24 PASS、4 MODEL_FAIL、2 TIMEOUT；正式比較為 0。新協定證據見 [low_mtp3/REPORT.md](low_mtp3/REPORT.md)。下文保留歷史 Xhigh 驗證，不作新協定 Gate。

# Terminal-Bench 2.1 工具與驗證紀錄（2026-10-20）

狀態：**PAUSED／BLOCKED_FOR_OFFICIAL_SMOKE**。工具、配置、CPU 檢查、五卡 BF16 serving 與極小量完整 vocabulary Fidelity 路徑已驗證；**本階段仍未完成**，因為現有環境沒有可用的 Docker Engine／Compose。指定五題 × 兩種協定的官方 Smoke 為 **0／10**；89 題 BF16 Gate、801-trial 矩陣、正式 Agentic KL 均為 **0**。合成測試不是 Terminal-Bench 分數，也沒有 Verifier 結果。

## 已沿用與固定的來源

完整閱讀現有 README、REPORT、`src/benchmark.py`、`src/ifeval_modes.py`、`src/fidelity.py`、`src/extend_fidelity.py`、`src/analyze.py`，並核對 `build/benchmark/OFFICIAL_REVIEW.md`、IFEval validation、expanded ablation 與 checkpoint inventory。原 Fidelity 數值核心未改，SHA-256 仍為 `90fcfa567abee03b3af1c90f6f1afcab5e0bccfa5ed3d09f14ca61198a2eb0dd`。

- [官方 Terminal-Bench repository](https://github.com/harbor-framework/terminal-bench-2-1/tree/7131e4375048a0e408a8fb404b5f499d726b695b)：revision `7131e4375048a0e408a8fb404b5f499d726b695b`，89 題。原 task.toml、instruction、environment、solution、tests 均未修改；逐檔 SHA 在 `assets/terminal_bench_manifest.json`。
- [官方 Harbor／Terminus-2](https://github.com/harbor-framework/harbor/tree/96ff9c6db90cc015f0fc0987d896670c9c5f1fad)：revision `96ff9c6db90cc015f0fc0987d896670c9c5f1fad`，Harbor 0.24.0／Terminus-2 2.0.0。使用其原生 job、trial、Docker、Tmux、JSON parser、Verifier、timeout、resume 與 ATIF；未改官方原碼。
- Harbor 的 uv.lock 安裝至獨立 `.harbor-venv`。LiteLLM 1.93.0、OpenAI client 2.24.0、Pydantic 2.13.5；完整 freeze 見 `harbor.freeze.txt`。原 `.vllm-venv`、`.modelopt-venv` 套件未變更。
- BF16 revision `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`；Fast `284b37dd7b7a5e0071268c0f17fbbb47e4f09b17`；Prisma `6dc090346f4f32acc320e48d9c413ea96a98d4c6`。沿用既有 checkpoint inventory 與完整指紋驗證；本次 resume identity 記錄 revision、config SHA、權重大小／mtime。
- 所有 89 個官方 Docker image tag 已透過公開 registry HEAD 固定 digest，見 `assets/terminal_bench_images.json`。未下載 layers、未啟動任務容器。之後按 digest pull，再還原官方 tag 的本機別名；task.toml 保持原樣。執行前核對 RepoDigests。
- Harbor／dataset 為 Apache-2.0；保留來源 LICENSE。研究三次重複依使用者要求，與[目前官方 leaderboard 至少五次且由維護者執行的提交政策](https://github.com/harbor-framework/terminal-bench-2-1/blob/7131e4375048a0e408a8fb404b5f499d726b695b/README.md)不同。本專案不宣稱取得 leaderboard 提交資格，也未上傳任何紀錄。

## 現有環境是否足夠

推論環境可用：五張 H100 80GB、driver 570.86.10、CUDA compatibility 使用者空間 12.9，vLLM 0.31.0+cu129、Torch 2.13.0+cu129、Transformers 5.17.0、compressed-tensors 0.17.0。cgroup CPU 上限為 50 cores、RAM 上限 1,000 GiB；主機 `/proc/meminfo` 的約 2 TiB 不是本工作環境的可用上限。

**任務容器環境不可用**：Docker CLI、daemon socket、DOCKER_HOST 均不存在；cgroup2 唯讀，缺 CAP_SYS_ADMIN／CAP_NET_ADMIN。單獨 user namespace 可建立；將 user namespace 與 mount／net namespace 一起建立也可成功。因此不把初次 `unshare` 失敗誤報成「所有 namespace 都被禁用」。但此環境沒有 newuidmap／newgidmap，也沒有可用的 rootless cgroup delegation；未證實能落實官方 CPU／RAM 限制。僅安裝 Python venv 或 Docker CLI 不足以完成正式 Smoke。[Docker rootless 必要條件](https://docs.docker.com/engine/security/rootless/)、[資源限制條件](https://docs.docker.com/engine/security/rootless/troubleshoot/)。

2026-10-21 [詳細 Docker 診斷](DOCKER_DIAGNOSIS.md)補充：user＋mount＋net namespaces 在 `--propagation unchanged` 時可建立，但變更 propagation 及在其中掛載 tmpfs 仍被拒絕。程序處於 enforced `cri-containerd.apparmor.d`，與官方預設 `deny mount` 高度吻合；尚無宿主機 audit 可確證實際拒絕規則。舊 namespace 檢查的成功只證明 namespace 建立，不代表完整 rootless 容器可用。

若沿用目前 Docker runner，需由環境提供者供應可用 Docker daemon／Compose，且 daemon 能掛載相同工作目錄；或改用具備這些條件的 VM。`preflight --container-probe` 準備以官方 fix-git image 做獨立、非計分的 bind mount／cgroup CPU／RAM 檢查。目前該檢查尚不能執行。Harbor 也支援雲端 sandbox，並非只有本機 Docker；替代後端尚需適配及驗證官方映像、資源、timeout、Verifier 與資料擷取。

原 IFEval 起初占 GPU 0、2；本次先用 1、3、4，之後用已釋放的 2，等 0 自行釋放後才做五卡同時測試。未中止任何既有推論。既有 IFEval 控制器 PID 8949 仍存在，非本次建立、未修改或終止；最後沒有活動 GPU 推論。不能把本次清理寫成整台機器所有其他控制器都已停止。

## 有效模型與 Agent 設定

| 項目 | 最終配置與實證 |
|---|---|
| Context | 原生 262,144；未加 YaRN，五個 BF16 實例均成功建立 |
| 每次輸出上限 | reasoning＋final 共用 131,072，所有實際短請求的 engine SamplingParams 相同 |
| Harbor 輸入預算 | 131,072，為輸出保留容量；服務的總 context 仍為 262,144 |
| Thinking／effort | enabled／xhigh；API 欄位與 chat_template_kwargs 均傳遞，實際 prompt 有 xhigh system instruction 與 `<think>` 開頭 |
| Sampling | temperature 1.0、top_p .95、top_k 20、min_p 0、presence_penalty 0、repetition_penalty 1 |
| Seed | 重複序號 0／1／2 對應 20261020／21／22；同一 trial 每一輪與原生 summary 呼叫使用同一 seed。14 次實際 GPU 請求均驗證第一個 seed；其他 seeds 的傳遞另有 CPU 轉換測試與正式 runtime audit |
| API | LiteLLM `hosted_vllm/` → OpenAI-compatible `/v1/chat/completions`，非 Responses API，非 streaming |
| Template／tokenizer | 全部使用同一 BF16 tokenizer 與發布模板；preserve_thinking=True |
| 推論 | TP=1、BF16 dtype／KV、MTP Off、eager、prefix cache Off、chunked prefill 4096、每 endpoint 一個序列 |
| GPU 配置 | utilization .95，較既有 IFEval 的 .85 明列提高，以保留原生長 context；沒有更換量化格式或權重 |
| Summarization | 官方 enable_summarize=True、threshold=8000，沿用 LiteLLM 近似 token counter；短 GPU 測試中觸發 0 次 |
| Timeout／資源 | task.toml 原值，所有 multiplier=1；未覆寫 Agent／Verifier timeout、CPU、RAM、storage、難度或題目集合 |

[Qwen 官方 card](https://huggingface.co/Qwen/Qwen3.8-27B/tree/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0) 建議在 1M context 下給 reasoning 262,144、final 131,072 的獨立預算；本後端使用單一共用 output budget，沒有重現該配置。1M 需 YaRN，並不是本次原生 context；BF16 TP=1／BF16 KV 的實測 cache 容量約 390,113 tokens，不能宣稱本機已支持 1M。保留原生配置及差異，沒有靜默降級。

## 多輪 Thinking 的實際流程

`Terminus2._query_llm` → 官方 `Chat` → Harbor `LiteLLM.call` → LiteLLM hosted_vllm → vLLM chat request → 官方 Qwen 模板 → engine。指令是 JSON 文字，由 Terminus parser 轉為 Tmux 操作；終端回饋作為下一個 user message，並非 OpenAI function-call tool role。[LiteLLM 官方路由說明](https://docs.litellm.ai/docs/providers/vllm)、[vLLM Qwen recipe](https://recipes.vllm.ai/Qwen/Qwen3.8-27B)。

本版 vLLM 回覆使用 `message.reasoning`；LiteLLM 將它交給 Harbor 的 `reasoning_content`。下一輪的 HTTP messages 中，True 保留 `reasoning_content`，False 省略它。vLLM/Pydantic 會正規化 reasoning 別名；驗證器使用 deep copy，避免解析重播改變原始 request 物件。

GPU 1 使用相同 instruction、seed、sampling，各跑三輪：執行 allowlist 中的 `printf ROUND1_OK`、接收輸出、執行 `printf ROUND2_OK`、接收輸出後完成。只執行兩條完全相符的 allowlisted host shell 指令；不是官方容器任務。

| 模式／輪次 | 實際 prompt IDs 數 | output IDs 數 | 歷史 assistant reasoning |
|---|---:|---:|---|
| False／1 | 821 | 587 | 無歷史 |
| False／2 | 958 | 139 | 未傳入 |
| False／3 | 1,092 | 123 | 未傳入 |
| True／1 | 821 | 587 | 無歷史 |
| True／2 | 1,429 | 146 | 第一輪 reasoning 傳入並渲染 |
| True／3 | 1,596 | 99 | 前兩輪 reasoning 均傳入並渲染 |

兩組第一輪的 prompt／生成 IDs 完全相同。True 第二輪比 False 多 471 個 prompt tokens，對應第一輪 reasoning；Qwen 模板實際接收到該歷史，不只存在 API JSON。

True 第二輪將 preserve_thinking 改為 False 的 CPU 渲染消融，會移除 475 tokens（reasoning 加框架分隔）；True 第三輪移除 513。False 對應僅移除 4／8 個空 thinking 分隔 tokens，**不能恢復從未傳入的 reasoning**。每輪 assistant final 與 user 終端回饋均保留，歷史連續性檢查未見遺失或重複。完整 raw requests／responses、prompt／生成 IDs、reasoning、final、commands、shell 回饋均已保存。

最終建議：主協定暫定 **interleaved_thinking=False**，維持官方預設；True 作明確分開的敏感度協定。現有證據顯示兩者短多輪協定均能工作，沒有證據證明預設會破壞必要推理，也沒有以成功率挑選配置。此建議仍須指定官方 Smoke 驗證後才可取得正式資格。

Summarization 在 GPU 短測試未觸發，不能宣稱已驗證長歷史壓縮。原生實作會以 summary／questions／answers 取代歷史，確實改變內容。輸入預算修正前後，12 次 CPU HTTP 重播的 request body 完全一致；實際呼叫原生 threshold 判斷也一致。正式 audit 遇到 history reset／summarization 會拒發 READY，要求先核對 ATIF 與 API 紀錄。這是待處理的條件式停止點，不是排除任務或改 timeout 的理由。

## 五卡實測與官方 Smoke 缺口

五卡同時 BF16 serving：初始化 48.03–48.45 秒，整波 wall 82.68 秒，共 0.112785 GPU-h；五個 ready service 共同存活 31.67 秒，每卡實際回覆同一個合成 Terminal 請求，均為 821 prompt／587 output tokens，固定 seed 下生成 IDs 相同。每卡最高顯存 77,134 MiB，engine log 顯示 BF16 KV、MTP None、原生 context 262,144；可用 KV 約 24.11 GiB。原始五卡 resource samples、server logs、PID／GPU UUID／endpoint ledger 已保存。這證明 serving 併發，不證明五個 Docker 任務沒有資源干擾。

| 指定官方 Task | 固定 Agent timeout | A：False | B：True |
|---|---:|---|---|
| modernize-scientific-stack | 600 s | 未執行 | 未執行 |
| overfull-hbox | 750 s | 未執行 | 未執行 |
| fix-git | 900 s | 未執行 | 未執行 |
| regex-log | 900 s | 未執行 | 未執行 |
| extract-elf | 900 s | 未執行 | 未執行 |

所有 timeout 已在固定 revision 逐項核對。沒有 success rate、沒有 Verifier 分數，也沒有將合成 printf 測試混入結果。

## Trajectory Fidelity 路徑

`src/agentic_fidelity.py` 以 native ATIF session ID 連結同存的 API audit，僅使用 BF16 來源。從實際 prompt IDs＋BF16 生成 IDs 建立 teacher-forced windows；候選不自由生成 trajectory 來代替 reference。不重新 tokenize 評分 prefix；另外重建模板，要求與服務實際 IDs 完全相同。

預設每 task 最多四個 32-token windows，固定 seed，輪流覆蓋可用的 reasoning、JSON keystrokes、詞法辨識的 code、工具回饋後前 32 個生成位置；缺少的層會明列。固定 prefix eligibility 上限 65,536，超長位置不入樣、計數保留，**不裁切原 prefix**。長 prefix 下的完整原 Fidelity backend 尚未以 GPU 做大型驗證；本階段只准極小正確性檢查。

核心計算直接呼叫原 `src/fidelity.py`：先完整 BF16 reference，再獨立重載 BF16 sanity，通過後才跑 Fast／Prisma。完整 248,320 維 FP32 log probabilities，原 FP64 KL／NLL 累計與 Top-1 Flip；沒有 top-k 近似。整體另列 task 等權與 token 加權結果，CI 將同 task 所有 windows 聚合後，以原 SciPy whole-document routine 重抽整個 task。單一 task 不產生虛假的 CI。Terminal-Bench 成功率與 KL 分開報告。

本次僅一個合成 task、三個 windows、共 **12 個 scoring positions**，含 reasoning／Terminal command／after-tool，沒有 code window。完整 BF16 reference／repeat 分布 hash 相同；KL=0、Flip=0、PPL delta=0。Fast、Prisma 均成功取得完整 vocabulary，Flip 均為 0；完整數值在 `validation/fidelity_smoke_v1/`。數量太少，**不能用數字宣稱 Prisma／Fast 的品質排序**，也不能推論 calibration data 的因果效果。

未使用任何 benchmark 資料 calibration、調整 bit allocation、搜尋配置或改模型權重。正式使用時必須一起保存 ATIF、API audits 與模型／來源 manifest；單有格式化後的 ATIF 文字不足以可靠還原所有特殊 token 邊界。

## 排程、失敗政策與統計

五個 TP=1 serving replicas，每個 endpoint 一個 Harbor job、concurrency=1。全部 serving ready 才啟動 Agent timeout；沒有讓多個活動 trials 共享 endpoint 的排隊。官方任務按宣告的 Agent timeout 做固定、簡單的 greedy 五分片，各模型／seed 的 task 分片相同。Smoke 則固定五題各一張卡。Docker／trial 隔離沿用 Harbor；CPU／RAM enforcement 設為原生 limit，無 resource override。

Gate 為 BF16 repeat 0 的 89 題；後續八個 model／repeat cells 為 712 題，兩者不重疊，總共 801。已完成整個 shard 時，GPU 載入前即略過；部分完成交給 native resume。Harbor 原生會刪除沒有 result.json 的未完成 trial folder，wrapper 先把它移到 `interrupted/` 保存。完成的模型失敗或 timeout 不為改善分數自動重跑。未知系統／Verifier／環境錯誤會停止後續階段。所有例外與舊 attempt 保留。

主分數重用 Harbor 官方 aggregate_reward_dicts，無 reward 的已結束 trial 為 0；已有 reward 則遵循官方值，包括 timeout 後 Verifier 判成功的情況。診斷另列 success、model task failure、Agent timeout、Verifier／環境／provider 錯誤、aborted、pending、parse errors。中止或未完成矩陣不輸出完整主配對結果。若官方 timeout 取消非 streaming 的進行中請求，audit 保留 request／參數／取消資訊；該未返回回覆的 token IDs 不可用，不能假稱完整生成或混進 Fidelity。

統計準備了平均 Pass@1、相對 BF16 delta、harmful／recovery、三次全部成功、empirical Pass@3、三 seeds accuracy／SD、task／token 成本與耗時分位數。主要配對以 89 tasks 為 cluster，三 seeds 保留在同一 task；2 候選 × 3 primary metrics 的六項 family，199,999 percentile bootstrap，95% 與 Bonferroni 99.1667% CI。Pass@3／seed summaries 為描述性指標，未擅加顯著性宣稱。

## 成本與啟動資格

本次 serving 檢查 0.241985 GPU-h，Fidelity 四次 capture 0.043451 GPU-h，合計約 **0.285436 GPU-h**。CPU 工程時間不計入 GPU-h；初次 CLI flag 檢查在模型載入前退出，未分配 GPU。沒有正式任務成本。

完整 89 題的官方 Agent timeout 合計 **42.2083 小時**，固定五分片最長為 8.5 小時；九個 model／repeat cells 合計 **379.875 GPU-h／76.5 小時五卡 wall**，這只是全部使用滿 Agent timeout 的配置上限口徑，未含載入、環境與 Verifier。若未來平均只使用宣告 timeout 的 25%／50%，對應約 95.0／189.9 Agent GPU-h、19.1／38.25 小時 wall；是規劃情境，**不是由 printf 外推的任務耗時預測**。Verifier 的 timeout 合計另為 41.4333 小時／89 trials，實際通常不等於其上限，目前沒有本機測值。不能把不同成本口徑混成精確 ETA。

**正式啟動資格：否。** 需補足 Docker／Compose／mount／資源 probe，完成指定 10 個官方 Smoke（不增加題目或 timeout），核對 Verifier、真實多輪 ATIF、取消行為及資源干擾；再做極小 real-trajectory Fidelity ingestion 檢查。Fast／Prisma 的長 context API serving 未在本階段另跑候選 Benchmark。之後仍須新的明確指示，才可啟動 89 題 Gate；Gate 通過且另獲授權後才續其餘 712 trials。

## 驗證與證據索引

- `validation/all_protocol_audit.json`：14 個真實 GPU API requests，prompt IDs／sampling／parser／歷史均通過。
- `validation/protocol_v1/`：False／True 各三輪、shell 回饋、另外兩卡啟動檢查。
- `validation/protocol_gpu2/`：native session header 的實際傳遞。
- `validation/five_gpu_protocol/`：五卡同時推論、resource samples、完整 server logs 與停止 ledger。
- `validation/configuration_replay.json`／`replay_configuration.py`：12 次 CPU wire-equivalence 與原生 summarization threshold 檢查。
- `validation/fidelity_windows_v1/`、`validation/fidelity_smoke_v1/`：固定 IDs、抽樣來源、BF16 full reference、sanity、候選 metrics、GPU 時間。
- `validation/targeted_tests.log`：16 個相關 CPU tests；未跑全套或不相關 GPU tests。
- `validation/mapping_regression_before.log`／`mapping_regression_after.log`：BatchEncoding 取值錯誤的失敗／修正重播。修正資料讀取，不改模型回覆。
- `prepared_20261020/`：最終 55 個 native job configs、801-trial 計畫、環境／model identity、PAUSED 狀態與未通過的 SMOKE_READY。
- `validation/container_preflight.json`、`prepared_20261020/preflight.json`：實際環境限制。
- `validation/costs.json`、`validation/final_state.json`、`validation/source_sha256.json`：成本、結束狀態與程式版本。

先前生成的 `prepared/`、`preparation_draft/`、`experiment/` 僅為未執行的配置草稿；不作正式結果，也不要用其舊 code identity 啟動。最終操作以 `prepared_20261020/` 為準。
