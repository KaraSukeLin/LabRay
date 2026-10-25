# Thinking／Non-thinking IFEval 工具

本文件先記錄工具驗證方式。正式矩陣 541 題 × 5 模型 × 2 模式 × 3 seeds＝16,230 次回答已於 2026-10-17 啟動，目前續跑狀態以根目錄 README 的快照與 `status` 指令為準。不要因閱讀本頁而另開第二個 launcher。

沿用 `.modelopt-venv`、原有權重、BF16 tokenizer／processor／chat template、`src/benchmark.py` 的引擎與 Google 官方 scorer。沒有新增評估 framework，也沒有修改既有環境或權重。

## 模型與共同設定

| GPU | 模型目錄（`/root/model/` 下） |
|---|---|
| 0 | Qwen-Qwen3.8-27B |
| 1 | eNeural-Qwen3.8-27B-fast |
| 2 | rdtand-Qwen3.8-27B-PrismaAQUA-5.5bit-vllm |
| 3 | ModelOpt-Qwen3.8-27B-AutoQuantize-5.5bit |
| 4 | unsloth-Qwen3.8-27B-NVFP4 |

此對應由模型清單順序與 `--gpus 0 1 2 3 4` 決定；每張卡 TP=1、max_num_seqs=1，模型載入一次後跑六個 mode/seed 單元。GPU 較少時，模型依固定輪流分配依序執行。MTP Off、BF16 KV、eager、prefix caching Off，沿用既有 H100 核心設定。

| 參數 | Thinking Off | Thinking On |
|---|---:|---:|
| temperature | 0.7 | 1.0 |
| top_p | 0.8 | 0.95 |
| top_k | 20 | 20 |
| min_p | 0 | 0 |
| presence_penalty | 1.5 | 0 |
| repetition_penalty | 1 | 1 |
| enable_thinking | false | true |
| reasoning_effort | 不傳入 | xhigh |
| preserve_thinking | true | true |
| max_tokens（正式矩陣） | 16,384 | 65,000（reasoning＋final 合計） |

Seeds 固定 `20261016 20261017 20261018`，每個 request 明確指定相同 seed。完整設定見 [config](../../../assets/ifeval_modes_config.json)。max_model_len=65,536；541 題 Thinking prompt 最長 415 tokens，不會因這個總 context 上限切掉 prompt。

Sampling 依 [Qwen 官方模型卡](https://huggingface.co/Qwen/Qwen3.8-27B)。`xhigh` 是官方預設，透過本機原版 chat template 注入指定 system instruction，再將實際 prompt token IDs 傳給 vLLM；不是一個會限制 thinking tokens 的硬性計算預算。已核對 vLLM 0.31.0 的 `ChatCompletionRequest.build_chat_params` 也把 reasoning_effort 傳入相同 template kwargs。

上限是本研究可修改的設定，不是 Qwen 的官方 IFEval 專用上限。官方針對 1M context 的長程 agentic 任務建議更大的 reasoning／final 容量；本工具不宣稱重現那些設定。若更改上限或 sampling，請另建 config 與輸出目錄。On/Off 差異包含 sampling 與上限差異，不能解讀為只切換 thinking 的純因果效果。

## 執行方式

所有命令在 `/root/projects/quant` 執行。以下前兩個為 CPU 操作，不會載入 GPU 模型：

```sh
# 檢視已準備的正式矩陣（目前完成數應全為 0）
.modelopt-venv/bin/python src/ifeval_modes.py status \
  --output build/benchmark/ifeval_modes_prepared

# 另建完整設定清單，只有計畫而不推論
.modelopt-venv/bin/python src/ifeval_modes.py plan --full \
  --output build/benchmark/ifeval_modes_new_plan
```

正式矩陣已使用下列相同命令與輸出目錄啟動。只有在 `status` 與 `ps` 均確認沒有該 run 的 worker 時，才可用同一目錄和 `--resume` 恢復：

```sh
.modelopt-venv/bin/python -u src/ifeval_modes.py run \
  --full --resume --output build/benchmark/ifeval_modes_prepared \
  --gpus 0 1 2 3 4
```

中斷後重複**同一條命令**即可續跑。`--resume` 核對 config、資料、程式碼、套件版本、tokenizer/template、模型 config 與權重檔大小/mtime；若條件已改變就停止，避免混合不同實驗。權重指紋沿用既有 revision，沒有為了續跑重讀數百 GB 做全檔 hash。

每筆完整回答評分後立即 JSONL append、flush、fsync。重啟只處理未完成的 `(mode, seed, question_id)`；若最後一行寫到一半，先另存該殘片，再移除尾端並重算該未完成題。已完成的回答不重算；中斷當下尚未保存的生成須重算。每個模型的 flock 避免重複 worker 同時寫入。Ctrl-C launcher 會將 SIGINT 轉送給各 worker 的 process group。

也可只對單張 GPU 續跑清單中的指定模型：

```sh
export LD_LIBRARY_PATH="$PWD/build/cuda-compat/cuda-compat${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=8 VLLM_NO_USAGE_STATS=1 \
HF_HUB_OFFLINE=1 VLLM_USE_V2_MODEL_RUNNER=1 \
  .modelopt-venv/bin/python -u src/ifeval_modes.py worker \
  --experiment build/benchmark/ifeval_modes_prepared/experiment.json \
  --model eNeural-Qwen3.8-27B-fast
```

建立自選的**小型**實驗（不會合併進完整矩陣）：

```sh
.modelopt-venv/bin/python src/ifeval_modes.py run \
  --models Qwen-Qwen3.8-27B --modes on --seeds 20261017 \
  --start 90 --stop 92 --gpus 0 --output build/benchmark/ifeval_modes_custom_smoke
```

`--start`／`--stop` 使用 canonical IFEval 順序的 Python 半開區間；也可用 `--question-ids ifeval/1508 ifeval/3753`。未指定範圍與 `--full` 時只取前兩題。完整 541 題選取須明確給 `--full`。

## 輸出、監控與分析

`experiment.json` 保存矩陣、固定題目、設定、來源 hash、模型版本與執行環境。每個模型的 `questions.jsonl` 保存 mode／seed／ID、原始輸出與 token IDs、reasoning、final_answer、實際 prompt token IDs、requested/engine SamplingParams、官方 strict/loose 與逐 instruction 成績、token 數、截斷、缺失 final、逐題推論／評分／前處理時間。可更新的attempt快照以temporary file＋atomic replace保存，避免中斷時破壞舊快照。

Thinking 分離使用 [vLLM recipe](https://recipes.vllm.ai/Qwen/Qwen3.8-27B) 指定的官方 `qwen3` parser，同時傳入模式與 prompt 初始狀態。Google 只評 final_answer；reasoning 未結束則 final_answer 為空，保留失敗與截斷紀錄，不把 reasoning 當答案救回。Raw output 與 token IDs 始終保留。

`attempts/*.json` 保存每次載入、初始化、推論、解析與官方評分、前處理、shutdown 與 worker wall time；`scoring_seconds` 包含 reasoning parser 與 Google grader，並非只有 grader 的純耗時。`launch-*.json` 保存模型/GPU 對應與程序 wall time、整批 wall time、保留 GPU-hours；硬體 CSV 保存 H100／driver。硬中斷留下 running attempt 時，最後記錄的成本是下界，分析會標示，不虛構完整耗時。模型層共用載入成本在模式成本表平均攤分，並保留原始整體成本，不重複計費。

```sh
# 可在執行中使用：僅讀結果，不推論
.modelopt-venv/bin/python src/ifeval_modes.py status --output RUN_DIRECTORY
.modelopt-venv/bin/python src/analyze_ifeval_modes.py \
  --runs RUN_DIRECTORY --output NEW_PROGRESS_DIRECTORY --allow-partial --family-size 300

# 全部完成後的正式分析與官方 CLI 重評（均為 CPU）
.modelopt-venv/bin/python src/analyze_ifeval_modes.py \
  --runs RUN_DIRECTORY --output NEW_ANALYSIS_DIRECTORY --resamples 199999 --family-size 300
.modelopt-venv/bin/python src/ifeval_modes.py verify \
  --runs RUN_DIRECTORY --output NEW_VERIFY_DIRECTORY
```

分析輸出 CSV 與 Markdown：`accuracy`、`per_seed_accuracy`、`paired_comparison`、`per_seed_comparison`、`question_effects`、`timing`、`attempts`。`statistics.json` 記錄完整度、GPU-hours、wall time、方法與多重比較 family。Timing 有 output/reasoning 長度、每題推論時間的 mean/p50/p95/max，以及 aggregate tokens/s；準確率表含截斷率與 missing_final。

prompt strict 為所有 instructions 都通過的題目比例；instruction strict 是通過 instructions 總數／instructions 總數，先對固定 seeds 平均。配對分析包含四個量化模型對同模式 BF16（八個對比），以及五模型各自 On−Off（五個對比）。七個指標為 prompt Accuracy Delta、Harmful／Recovery／Answer Flip，及 instruction Accuracy Delta、Harmful／Recovery Flip。Harmful 與 Recovery 的分母是所有配對題目／instructions，且 Delta＝Recovery−Harmful。Answer Flip 是 final answer 去除外圍空白後的字串差異，不代表語意錯誤。

每題先保留同 seed 的兩模型／模式配對，再平均三個 seeds；SciPy 對**題目**重抽樣，整題三 seeds 與 instructions 一起保留。Instruction 指標每次重抽樣重算分子與分母之比。不能把 1,623 個 question×seed 當 1,623 個獨立題目。CI 條件於這三個固定 seeds，不代表未知 seeds 的完整不確定性。正式分析包含 13 個 BF16／On-Off 主對比，以及 6 個 Fast 對 Prisma、ModelOpt、Unsloth 的直接對比，共 19 對比 × 7 指標。依專案預先指定的 300 項 joint family 輸出 95% 與 Bonferroni `1−0.05/300` 區間；目前列出的 IFEval contrasts 為 133 個指標。逐 seed 數值供敏感度檢查，不另做獨立顯著性宣稱；不足的配對單元不使用動態交集計算 CI。

## 來源與驗證

Google [官方使用方式](https://raw.githubusercontent.com/google-research/google-research/3ef52111c1fabc301880818f5a547570e926a76b/instruction_following_eval/README.md) 指定 prompt/response JSONL 與 evaluation_main。本工具直接呼叫其 evaluation_lib；verify 再用原版 CLI 重評 strict/loose 與逐 instruction 向量。Google commit 固定 `3ef52111c1fabc301880818f5a547570e926a76b`，沿用 canonical GitHub 541 題，不使用有一道差異的 HF 副本。工具 Apache-2.0；SciPy BSD-3-Clause；沿用已安裝套件及授權，未複製或重寫評分器。

[upstream_audit.json](upstream_audit.json) 記錄五份 Google 原碼及資料與 pinned upstream 逐 byte 相同；[template_audit.json](template_audit.json) 記錄 541 題模板等價。針對性 CPU 測試包含 official parser 的 implicit/explicit `<think>`、只有 reasoning 與只有閉合標記、sampling 實際值、題目分群 CI／instruction ratio、torn JSONL 修復、重複紀錄拒絕與共用引擎設定。沒有執行全套測試。

小型 Smoke 的統計區間只驗證程式能運作，不能據此排名模型或精確預測完整 Thinking 成本。實測表、官方重評、續跑與簡化證據見同目錄的 [RESULTS.md](RESULTS.md)。
