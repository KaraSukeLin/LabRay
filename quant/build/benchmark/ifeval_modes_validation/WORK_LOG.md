# 2026-10-17：Thinking／Non-thinking IFEval 工具執行紀錄

範圍：準備五模型、兩模式、三 seeds 的完整541題實驗；只執行必要 Smoke。正式矩陣16,230次未啟動。

1. 閱讀README、REPORT、AGENTS、benchmark.py、run_benchmarks.py、benchmark_analysis.py、官方重評工具及上一輪WORK_LOG。先確認無遺留控制器及GPU程序。
2. 核對本機Qwen模型卡／chat_template、官方公開模型卡及vLLM recipe；使用安裝的vLLM 0.31.0實作檢查reasoning_effort路由與qwen3 parser。541題Off模板與舊流程相同；On顯式xhigh與本機官方預設相同。Google來源及canonical資料六個檔案與pinned upstream逐byte相同。
3. 從benchmark.py抽出相同引擎設定create_llm，新增ifeval_modes.py、analyze_ifeval_modes.py與固定config。沒有安裝套件、修改權重或重新建立評分系統。
4. 建立兩個固定題目ID的60筆Smoke plan，先以BF16保存1筆後正常停止，再以五GPU執行剩餘59筆；各worker exit0。BF16共兩次載入，其餘模型各一次，沒有額外生成控制題。

```sh
.modelopt-venv/bin/python src/ifeval_modes.py plan \
  --question-ids ifeval/1508 ifeval/3753 --output build/benchmark/ifeval_modes_smoke_v1

export LD_LIBRARY_PATH="$PWD/build/cuda-compat/cuda-compat${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8 VLLM_NO_USAGE_STATS=1 HF_HUB_OFFLINE=1 \
VLLM_USE_V2_MODEL_RUNNER=1 .modelopt-venv/bin/python -u src/ifeval_modes.py worker \
  --experiment build/benchmark/ifeval_modes_smoke_v1/experiment.json \
  --model Qwen-Qwen3.8-27B --max-new-records 1

.modelopt-venv/bin/python -u src/ifeval_modes.py run \
  --question-ids ifeval/1508 ifeval/3753 --output build/benchmark/ifeval_modes_smoke_v1 \
  --resume --gpus 0 1 2 3 4
```

5. 實際執行中的CPU progress分析通過，當時57/60，正確標記complete=false；不完整的配對單元不計CI。完畢後以199999次question cluster bootstrap分析，含13對比、7指標、91 family；完整結果在analysis_final。初次analysis為no-op續跑前版本，analysis_final為最終交付，品質表與CI完全相同。
6. 原版Google CLI重評60筆，strict/loose與instruction向量全部一致；所有prompt IDs、raw token decode、官方parser重播與vLLM post-process engine SamplingParams核對通過。重評及稽核CPU wall47.73秒。計時使用Python標準庫time.perf_counter與subprocess；系統未提供/usr/bin/time，初次計時命令未執行到分析或評分，隨即改用標準庫。
7. 全完成後重複相同resume命令：五worker皆輸出「All records already complete; no model load」，五JSONL hash與六個attempt數不變，沒有新GPU工作。不同seed要求resume被拒絕，既有manifest不變。保存前綴、torn尾行及duplicate拒絕另有CPU檢查。
8. CPU消融重評30筆Thinking raw output與final：15筆prompt strict不同，證實不能直接評包含reasoning的raw文字。原始60筆不變。
9. 最後修正僅將三個mutable timing快照寫入改為atomic replace，CPU注入中斷驗證舊快照可讀。AST比對確認推論、prompt、sampling、parser、grader邏輯不變；原Smoke source另存符合原manifest hash，無重跑推論。正式plan重新建立以固定最終程式hash。

```sh
.modelopt-venv/bin/python -m pytest -q \
  tests/test_ifeval_modes.py tests/test_benchmark.py \
  tests/test_benchmark_official.py::test_official_ifeval_strict_and_loose_are_used \
  tests/test_benchmark_official.py::test_prepared_ifeval_matches_google_repository_input
```

最後20個針對性CPU檢查通過。未執行全套測試、額外GPU regression或正式全量實驗。

10. 更新README、REPORT、USAGE與RESULTS，保存逐模式時間CSV、182列配對CI、逐seed／逐題效果與各項audit。最終檢查133個本地Markdown links、目前程式與正式manifest source identity、原Smoke檔案hash及GPU狀態全通過。最終五卡空閒；正式0/16230。正式命令僅記錄於USAGE/README，未執行。

最終正式manifest：`build/benchmark/ifeval_modes_prepared/experiment.json`。
`prepared_before_atomic/`是沒有推論的早期CPU plan，僅供來源追溯；不應用於正式啟動。舊Smoke因程式hash保留於原manifest，後續修改程式後不應直接續寫；新的實驗請使用新output。正式prepared已對齊最終程式。
