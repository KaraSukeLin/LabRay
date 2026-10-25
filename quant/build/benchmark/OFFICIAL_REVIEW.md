# IFEval／MMMU-Pro 官方流程複核（2026-10-17）

本次複核發現並修正 IFEval 資料來源差異；先前「已準備」不代表所有設定都與官方建議相同。完整能力 Benchmark 仍未執行。

## 官方來源與資料

- IFEval：使用 [Google Research 作者原始程式與 README](https://github.com/google-research/google-research/tree/3ef52111c1fabc301880818f5a547570e926a76b/instruction_following_eval)，沒有重寫評分器。這是研究團隊的官方實作，不是 Google 提供產品支援的服務。
- MMMU：實際準備的是 **MMMU-Pro**，不是原版 MMMU。使用 [作者 repository](https://github.com/MMMU-Benchmark/MMMU/tree/268471d0d488258990025331c7528359c324aa25/mmmu-pro) 的 `direct` prompts、影像順序 helper、答案 parser 及 grader；資料為作者發布的 `MMMU/MMMU_Pro`。
- 2026-10-17 再次下載並逐 byte 核對：上述使用的十份 README／評分／prompt 檔案與目前官方主分支相同。[主分支核對](official_recheck/current_official_sources.json)。執行仍固定 commit，以便重現。
- Google HF 資料與官方 GitHub 的 541 題僅一個欄位不同：key `2785` 的 HF prompt 要求「至少一個 placeholder」，GitHub 要求「至少三個」，kwargs grader 要求三個。正式輸入改用官方 README 指定的 GitHub `data/input_data.jsonl`。[差異原文](official_recheck/ifeval_source_differences.json)。
- 正式資料位於 `assets/benchmark/prepared_v2/`，只修正上述 prompt，原始影像重用 `prepared/images/`；舊 prepared 與舊 smoke 保留。IFEval 541 題、MMMU-Pro standard (10 options) 及 vision 各 1,730 題，合計 4,001 題／模型。

## 推論與評分的範圍

IFEval 官方 README 規定 prompt/response JSONL 交給 `evaluation_main`，沒有規定唯一 sampling。這次除直接呼叫官方 functions，也實際使用官方 CLI 重評舊 smoke。保存 strict／loose、prompt／instruction 四種結果；主分析仍以 strict prompt-level 為準。

MMMU-Pro 官方 README 允許 `direct` 或 `cot`。本次固定 **direct、thinking=false**，與官方 prompt 及 image mapping 完全相同。官方 Transformers 範例使用 FP16、50 output tokens；本專案保留統一 BF16、BF16 KV cache，MMMU output 上限 1,024、IFEval 4,096，以降低回答截斷。這些是明列的研究設定，不能宣稱逐項複製示範 script 或重現官方 leaderboard。

正式 sampling 依 [Qwen3.8-27B 官方 non-thinking 建議](https://huggingface.co/Qwen/Qwen3.8-27B#best-practices)：temperature=0.7、top-p=0.8、top-k=20、min-p=0、presence penalty=1.5、repetition penalty=1。greedy sampling 另存於 `assets/benchmark_greedy_config.json`，可用 `--config` 選取，供數值／MTP 檢查；其資料與解析度跟隨正式流程，兩種 sampling 的結果不得合併。

移除原先額外的 1,048,576-pixel 限制，正式改回模型發布的 processor 預設 min=65,536、max=16,777,216 pixels。使用官方 `smart_resize` 逐題計算，最大為 8,733 visual tokens；加上文字、保守 framing 餘量與 output 預算，全部題目均在 65,536 context 以內。[context 稽核](official_recheck/official_resolution_context.json)。這是 CPU 長度核對，實際圖片執行另有 smoke。

MMMU 無法抽取時保留官方隨機猜選項；每題 seed 固定且跨模型相同，額外記錄 `extraction_fallback`。IFEval Answer Flip 只是完整回答字串是否不同，並非語意錯誤率。正式報告應同時公布截斷與 fallback 比例。

## MTP Off／On

`src/benchmark.py run --mtp` 已將原生 MTP 整合至相同文字／圖片流程，使用兩個 speculative tokens；省略則 Off。數量是固定研究參數，官方 vLLM recipe 的三個 tokens 是另一個可調範例，不是正確性門檻。thinking 兩邊都維持關閉。

比較分兩個方向：同一模式下量化模型相對 BF16，以及同一 checkpoint 的 MTP On 相對 Off。`benchmark_analysis.py --reference-runs OFF_DIRECTORY` 支援後者，並檢查除 MTP 外的資料、prompt IDs、圖片與條件一致。一般模式拒絕混入不同 MTP 條件。

[vLLM 官方說明](https://docs.vllm.ai/en/latest/features/speculative_decoding/)將 speculative decoding 定義為分布無損，但提醒浮點數／batch 差異。使用 stochastic sampling 時，相同 seed 也不能保證 On／Off 的回答逐 token 相同；觀察到 Flip 不可全歸因於量化或 MTP 精度損失。greedy 對照與實際 acceptance counters 用來檢查解碼實作。

若九模型三項 tasks 都跑兩種模式，總數為 **72,018 次回答**。若所有量化差異及 On／Off 差異一起作確認性推論，預先宣告共同 family 為 `2 × 8 × 3 × 4 + 9 × 3 × 4 = 300`。各次分析可使用 `--family-size 300 --resamples 199999`；這個 family 尚未產生完整能力結果。

授權正式執行後，每模型的命令形式為：

```sh
.modelopt-venv/bin/python src/benchmark.py run --model MODEL --output build/capability/off/MODEL
.modelopt-venv/bin/python src/benchmark.py run --model MODEL --output build/capability/mtp/MODEL --mtp
.modelopt-venv/bin/python src/benchmark_analysis.py --runs build/capability/off --output build/capability/off_analysis --family-size 300 --resamples 199999
.modelopt-venv/bin/python src/benchmark_analysis.py --runs build/capability/mtp --output build/capability/mtp_analysis --family-size 300 --resamples 199999
.modelopt-venv/bin/python src/benchmark_analysis.py --runs build/capability/mtp --reference-runs build/capability/off --output build/capability/on_off_analysis --family-size 300 --resamples 199999
```

使用與 README 相同 CUDA compat／GPU／offline 環境。每種模式都須有 BF16；保留逐題原始輸出、token IDs、時間與 finish reason。

## 實際驗證與時間估算

完整資料 CPU 稽核通過：541 題 IFEval 與 Google GitHub 相同；3,460 題 MMMU-Pro 的 prompt、gold、options、subject 相同；3,778 張影像的像素、順序與檔案 SHA 相同。Google 官方 CLI 重評 18 個 IFEval 回答的 strict／loose 與 instruction 向量，MMMU 官方 `mmmu_process_results` 重評 36 個回答，均與已保存結果相同。[完整稽核](official_recheck/source_and_scoring_v2/audit.json)。

使用 Qwen 建議 sampling、每 task 兩題及完整 output 上限，已完成九模型 Off／On 共 18 個 runs、108 個回答，全部成功；其中使用的圖片上限仍是修正前的 1M pixels。這批是獨立小量 pilot，不是正式能力分數。Off／On 配對、同模式 candidate−BF16 的三個分析流程均實際通過，使用 300-comparison family 的 smoke 設定。[Off](official_recheck/off/)／[On](official_recheck/mtp/)／[On−Off 分析](official_recheck/on_off_analysis/)。

八項 targeted CPU tests 通過，包含新增的 canonical IFEval 資料 regression 與 MTP 條件不得混用的檢查；沒有執行 full suite。[測試](official_recheck/final_targeted_tests.log)。

恢復官方解析度後，另完成 **33 個 greedy smoke 回答**：九模型各以原生 MTP 執行修正的 IFEval 題、35 個圖片參照題及最高 visual-token 題；BF16／Fast 另各跑同三題的 Off。全部成功，實際圖片 prompt 最大 8,789 tokens。九模型的 draft／accepted counters 均大於零，證明 MTP 參與解碼。[逐題與原始 counters](official_recheck/official_resolution_greedy/)／[readiness audit](official_recheck/readiness_audit.json)。

六個 greedy Off／On 配對中，四個輸出 token IDs 完全相同，兩個不同：BF16 的 35-image 題在第 65 個輸出 token 首次不同；Fast 的 IFEval 題在第 43 個輸出 token 首次不同。尚未單獨定位原因，不把它斷言為浮點誤差或模型品質退化。這六對的已存分數相同，但 smoke 只有 128 output tokens，四個 MMMU 回答都走官方 parser fallback，因此**這不是品質無損的證據**。正式 Off／On 必須分開評估，不能直接沿用另一模式的能力分數。

以下為已被新pilot取代的歷史預估，**不得用於目前排程**。每task只有兩個非隨機樣本的latency外推：Off約197 GPU-hours、On約107 GPU-hours；五張H100兩模式合計約64小時。來源為舊1M-pixel pilot、生成長度不同，不能聲稱MTP加速倍數。當時尚未有代表性資料，曾建議：

| 範圍 | 五張 H100 的規劃預算 |
|---|---:|
| 九模型、三 tasks、MTP Off | 約 2–3 天 |
| 九模型、三 tasks、MTP Off／On | 約 3–5 天 |

這是小樣本外推，不是統計信賴區間或工期保證；thinking=true、額外 seeds、DSpark、完整成本／throughput benchmark 均未包含。完整能力工作量為 72,018 次回答，尚未執行。[逐模型估算與限制](official_recheck/runtime_estimate.json)。

2026-10-17代表性pilot已完成：1,800個主回答＋501個4K探測。九模型MTP Off、完整三task的最新五H100一般預算為MMMU1K約81h、4K約170h；4K仍17%截斷。MTP On沒有代表性預估，舊2–3天／3–5天區間不再採用。[最新整合分析](pilot_v1/decision_report/DECISION_REPORT.md)／[週一摘要](pilot_v1/decision_report/MONDAY_SUMMARY.md)／[全2,301回答官方重評](pilot_v1/official_replay/audit.json)。
