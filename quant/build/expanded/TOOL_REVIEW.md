# 開源工具採用檢查

已閱讀現有 src/fidelity.py、README.md、第一輪 REPORT 與原始紀錄。保留原本的目標對齊、自然對數、FP32 logprob、FP64 指標、token 加權、MTP Off、BF16 KV cache 與獨立程序 sanity gate。

- vLLM 0.31.0+cu129，Apache-2.0。直接採官方 V2 model runner 的 SamplingParams.prompt_logprob_token_ids 與 prompt_logprob_start，指定全部 range(248320)。透過官方 compute_token_logprobs 取得 log-softmax，沒有重寫模型前向、softmax、推論排程或取 logits 核心。能只擷取計分區段，省去前文分布、top-k 全排序與 Python flat 結構重建。小型實測 dense_probe.json 的起點 0 與 20，兩者與既有 full-vocabulary flat 路徑逐值相同。保留 flat 選項作為消融對照。
- EleutherAI lm-evaluation-harness，檢查 commit d6de81643928d653435c431bae19945d41d32520，MIT。原始碼保存在 tool_review/harness_vllm.py。其 vLLM adapter 使用 prompt_logprobs=1，回傳 continuation loglikelihood 與 greedy-match，未提供本實驗需要的配對完整 vocabulary KL。直接引入仍需自訂 adapter 且新增 task/tokenization/aggregation 層，故不加 harness 依賴。PPL 定義同為 exp(總 NLL / token 數)，但不採其不同的 rolling-window 取樣規則。
- SciPy 1.18.0，BSD-3-Clause。直接採 scipy.stats.bootstrap，percentile、vectorized、固定 RNG、19999 次。每份文件的全部 tokens 先聚合；配對模型的資料保留在同一矩陣欄，domain/context 各層獨立重抽文件。沒有自行重寫抽樣迴圈或 bootstrap CI。
- PyArrow 23.0.1，Apache-2.0。直接讀官方 Wikipedia parquet；標準函式庫 tarfile 讀固定 GitHub archives。沒有加 datasets task framework 或額外資料轉換框架。
- NumPy 2.3.5 的原本指標函式保留，避免引入會自行 renormalize 或改變 KL 定義的函式。完整有限維度包含 padded vocabulary，零機率項目貢獻為零。

官方來源：

- https://github.com/vllm-project/vllm/tree/v0.31.0/vllm/v1/worker/gpu/sample
- https://github.com/EleutherAI/lm-evaluation-harness/blob/d6de81643928d653435c431bae19945d41d32520/lm_eval/models/vllm_causallms.py
- https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html
- https://huggingface.co/datasets/wikimedia/wikipedia/tree/b04c8d1ceb2f5cd4588862100d08de323dccfbaa

保留的研究程式只負責固定資料、呼叫官方引擎、配對指標與報告。沒有新 framework。

圖表使用 Matplotlib 3.11.2，輸出獨立 PNG／PDF；套件完整版本與 wheel 雜湊見 package_provenance.json。公式另外透過 SciPy special.rel_entr 對獨立重擷取的 Fast 矩陣驗證，沒有改用會自行 renormalize 的 entropy。最終資料 v2.1 已完成已知 calibration 片段排除與重新實跑。
