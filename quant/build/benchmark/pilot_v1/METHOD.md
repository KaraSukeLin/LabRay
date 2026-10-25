# 代表性 pilot 的方法與重跑方式

本次只執行代表性 pilot，不啟動完整 36,009 次能力評估。模型、固定資料、官方評分工具與隔離環境均沿用既有專案。MTP Off、thinking Off、BF16 KV；九模型使用同一題目順序、相同 Qwen non-thinking sampling 及官方 processor 圖片上限。

抽樣 seed 為 20261017。IFEval 按指令數（1、2、3+）和明示字數參數（無／小於 300／至少 300）形成九層，每層隨機取四題，少於四題則取全部，共 35 題；字數參數是成本代理，包含 at-most 與 at-least，並不是保證生成長度。MMMU-Pro Standard／Vision 各自按官方 30 學科，每科獨立不放回隨機取兩題，各 60 題。再加入 5 題極端／資料回歸檢查；極端題不進入成本的機率加權平均。總共每模型 160 題。

完整官方輸入的每列不變，只以 IDs 選題；`assets/benchmark/pilot_v1/manifest.json` 保存母體特徵、官方 topic_difficulty、樣本、層大小、權重與 SHA。每層的權重為母體題數／樣本題數。IFEval 的權重總和是 541，兩種 MMMU 各為 1730。

主要輸出上限保持 IFEval 4096、MMMU-Pro 1024。另以 4096 重新產生所有 MMMU 截斷回答，以及預先固定的 12 個未按結果選擇的控制題（每 task、每官方難度兩題）。重測從相同 prompt、seed 重新生成，分開保存於 `extensions.jsonl`，並檢查 token prefix、抽取、正確性及截斷。固定控制若本來已截斷，會歸為截斷探測。長上限探測共用已開啟的圖片與模板；實際時間另列。

計時採 Python 標準函式庫 `time.perf_counter`；外部 launcher 計整個子程序與所有工作的 wall time，vLLM 原生日誌提供權重載入及 engine 初始化內部時間。內部時間有包含關係，不可重複加總。逐題 `LLM.generate` 的時間包含引擎前處理、prefill、decode，不等於純 decode kernel 時間。已啟用每十秒一次的 `nvidia-smi` 利用率、記憶體、功率和溫度紀錄。每張 GPU 同時只有一個 TP=1 worker、每次一題；不推估較大 batch 的加速。

評分直接使用 [Google IFEval](https://github.com/google-research/google-research/tree/3ef52111c1fabc301880818f5a547570e926a76b/instruction_following_eval) 及 [MMMU-Pro](https://github.com/MMMU-Benchmark/MMMU/tree/268471d0d488258990025331c7528359c324aa25/mmmu-pro) 作者程式，兩個 repository 均為 Apache-2.0。MMMU 的抽取失敗保留官方 random fallback，每個 question ID 固定相同 seed，另列 fallback 指標。計時與 bootstrap 使用既有 Python／NumPy／SciPy，未新增評估 framework。

重跑第一階段的方法如下；輸出資料夾須為新路徑：

```sh
.modelopt-venv/bin/python src/prepare_benchmark_pilot.py --output assets/benchmark/new_pilot
.modelopt-venv/bin/python -u src/run_benchmarks.py --data assets/benchmark/new_pilot --output build/benchmark/new_pilot/runs --gpus 0 1 2 3 4 --long-max-tokens 4096
.modelopt-venv/bin/python src/benchmark_budget.py --data assets/benchmark/new_pilot --runs build/benchmark/new_pilot/runs --output build/benchmark/new_pilot/analysis
.modelopt-venv/bin/python src/benchmark_analysis.py --runs build/benchmark/new_pilot/runs --output build/benchmark/new_pilot/paired_analysis --family-size 96 --resamples 19999
```

Launcher 自行設定 CUDA compatibility、離線模式及 OMP=8，保存完整 argv、GPU 硬體、CPU、逐程序 exit code 與 wall time。不會自動重試、拼接失敗的結果或啟用 MTP。完整資料必須另給 `--full` 才會啟動；本次沒有使用此旗標。

成本估算以各層平均 pipeline 秒數乘母體題數，加上每模型一次初始化／程序固定開銷。SciPy percentile bootstrap 在各層獨立重抽題目、九模型共用相同重抽索引；完整涵蓋的 census 層不重抽。其餘層忽略有限母體修正，略偏保守。另用獨立的直接權重內積核對估算點，並輸出未加權／加入極端題的敏感度表。小樣本及單一 sampling seed 無法涵蓋所有未見長尾。

所有品質分析是 pilot 的探索性結果；正式能力排名仍需完整評估。這次的時間估計只適用目前 MTP Off 設定。MTP On 的速度及能力效果需要另行配對實測。

追加40題 IFEval 的決策與條件估算保留於 `extension_decision.json`。已觀測38題逐題計入，剩餘503題以獨立 SRS40 推估；不能把資料驅動追加的樣本當成一次簡單隨機抽樣直接混池。最終資料為 `assets/benchmark/pilot_v2/`，每模型200題。重跑完整兩階段請使用：

```sh
bash scripts/run_representative_pilot.sh build/benchmark/new_representative_pilot 0 1 2 3 4
```

合併輸出明列為 derived union，所有原始回答逐 byte 複製，保存 component_runs；實測初始化有兩次，正式工期只計每模型一次平均初始化。此腳本的各階段命令用於本次實際流程；未額外重跑整批以測腳本。

## 2026-10-17 完成後分析

使用者在核心及IFEval補樣完成後，已授權整合分析與週一報告。最終1,800主要回答、501個4K探測；未完成8K或完整Benchmark；舊控制器誤觸的17個8K部分輸出另行隔離。wrapper已移除自動8K階段，只重現這兩個已完成階段。

最終品質分析採199,999次bootstrap，兩cap合併為8候選×4指標×5不同task/cap單元＝160項探索性family，IFEval只算一次；保留一般95%與99.96875%區間。母體加權品質重用同一條件抽樣設計，另保存原始觀測題exact McNemar檢查，提示每學科兩題的bootstrap限制。不是預註冊確認性檢定，也不宣稱無差異等於等價。

成本19,999次bootstrap先合計各task抽樣成本再取模型分位數，不能相加各task邊際分位數；task使用獨立亂數流，模型在task內保持配對。子集只計一次完整模型初始化。排程基準週日10-18 00:00台北時間、週一10-19 09:00報告、留最後3h整理；只是規劃假設，沒有啟動此正式排程。

[整合報告](decision_report/DECISION_REPORT.md)／[一頁摘要](decision_report/MONDAY_SUMMARY.md)／[等價驗證](decision_report/analysis_validation.json)。

補充：最後發現舊背景控制器於本次CPU合併後誤觸8K，已停止並隔離17個部分回答，未加入正式資料。額外約2.24 GPU-h另列；[事件](decision_report/unexpected_8192_stop.json)。
