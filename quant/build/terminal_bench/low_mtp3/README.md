# Terminal-Bench 2.1 Low / MTP3

30／30 Smoke 已完成並依使用者指示暫停：24 PASS、4 MODEL_FAIL、2 TIMEOUT。正式 267 Trials 未啟動；需使用者新的確認。完整結果、成本與限制見 [REPORT.md](REPORT.md)。

矩陣是三模型 × 五題 × interleaved_thinking False/True，各一次；seed=20261020、Thinking Low、native MTP3、BF16 KV、原 sampling/context/tokenizer/template 及原 Agent/Verifier timeout。主要比較為 False；True 是獨立診斷。模型對應已由使用者確認：BF16=Qwen-Qwen3.8-27B，NaviQuant=eNeural-Qwen3.8-27B-fast，Aqua=rdtand-Qwen3.8-27B-PrismaAQUA-5.5bit-vllm。固定配置在 `assets/terminal_bench_low_mtp3.json`。

MODEL_FAIL 的原官方 assertion：三模型 False 的 extract-elf 參考位址覆蓋率均為 0%，低於 75%；Aqua True 的 overfull-hbox 將 younger 改成不在允許同義詞表中的 newer。兩個 TIMEOUT 都是 NaviQuant True（overfull-hbox、extract-elf）。完成的失敗不重試或改寫答案。

本次使用 PRoot 非標準協定，是內部比較，不能作官方 Docker Leaderboard 成績。全部 89 題保留在 `coverage.json`；其中五題原 Oracle=1、NOP=0、Verifier 與 cleanup 通過，其餘 84 題尚未驗證，不代表不相容。官方任務、測試與固定模型權重保留。H100 的 FP4 使用 Marlin weight-only 路徑。舊 Xhigh、pilot 與前期失敗紀錄保留，不混入本次計分。

實際 1→2→5 並行驗證已完成。為減少 GPU 閒置，續行控制器保留原 worker、相同 SQLite queue 與 frozen inference identity，讓空閒 GPU 接續工作；原 coordinator 在 queue 排空後恢復並正常退出。末端 queue 已空時不另造工作。所有 serving 已結束，未啟動正式批次。

證據：

- `smoke.sqlite`、`smoke/<id>/`：30 個唯一 trial、原 result、trajectory、recording、Verifier 與 cleanup。
- `SMOKE_AUDIT.json`：逐 API 的 token IDs、sampling、history、session identity 與原測試數核對。
- `TERMINUS_PARSE_AUDIT.json`：原官方 parser 重播；格式修正輪次保留，未改模型答案。
- `SMOKE_SUMMARY.json`、`REPORT.md`：結果、時長、MTP counters、GPU 成本與限制。
- `COMPLETION_AUDIT.json`：完整矩陣、固定來源、服務停止與暫停邊界。
- `source_snapshot/`、`smoke-readiness.json`：固定推論來源與 hashes。
- `resources.jsonl`、`services/`、`concurrency-*.json`：資源、服務生命週期與真實並行證據。
- `STATE.json`、`GOAL.json`：本次暫停與後續未完成範圍。

只重建 CPU 分析，不重新推論：

```sh
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 .modelopt-venv/bin/python src/terminal_smoke_audit.py
.harbor-venv/bin/python src/terminal_parse_audit.py
python3 src/terminal_smoke_report.py
.harbor-venv/bin/python src/terminal_smoke_completion.py
```

完整 89 題環境控制、Low full-vocabulary trajectory Fidelity 與正式 267 Trials 仍待後續指示。不得把 Smoke 的五題分數當成完整 benchmark 結果。前期準備文字保留在 `pre-smoke-REPORT.md`。
