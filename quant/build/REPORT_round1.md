# Fidelity 實驗結果

2026-10-16，8 個模型均成功完成 text-only 載入、8 段固定文字的基本推論，以及完整 vocabulary 的 teacher-forcing 比較。最終 9 個工作程序（BF16 兩次、候選七次）退出碼均為 0，沒有退出清理錯誤。

環境：單張 NVIDIA H100 80GB HBM3，TP=1；vLLM 0.31.0+cu129、PyTorch 2.13.0+cu129、Transformers 5.17.0、compressed-tensors 0.17.0。核心驅動 570.86.10，專案內 CUDA 12.9 相容函式庫 575.57.08。完整版本見 `environment.json`、`environment.freeze.txt`，各模型核心與狀態見 `model_checks.csv`。

固定資料：8 段自製中英文文字，每段最多 64 tokens，共 432 個計分位置；每個位置保留完整 248,320 維模型輸出 vocabulary。使用 BF16 tokenizer 的同一份 token IDs；各模型 tokenizer 的 token→ID 對應已確認完全一致。沒有 chat template、影像或影片。MTP/speculative decoding 關閉，KV cache 明確使用 BF16，prefix cache 關閉，eager mode，seed=0。

| Model | Status | PPL | PPL delta | KL (BF16 → candidate) | Top-1 flip rate | Tokens | Reason |
|---|---|---:|---:|---:|---:|---:|---|
| Qwen-Qwen3.8-27B | ok | 12.94307238 | 0 | 0 | 0 | 432 |  |
| Qwen-Qwen3.8-27B-FP8 | ok | 12.96038273 | 0.01731035334 | 0.003925763023 | 0.06481481481 | 432 |  |
| RadixArk-Qwen3.8-27B-NVFP4 | ok | 12.97835833 | 0.0352859498 | 0.02837077559 | 0.1018518519 | 432 |  |
| RadixArk-Qwen3.8-27B-NVFP4-BF16-LMHead | ok | 12.94757018 | 0.004497799461 | 0.01719896963 | 0.08101851852 | 432 |  |
| RedHatAI-Qwen3.8-27B-NVFP4 | ok | 13.00634535 | 0.06327297154 | 0.01388166067 | 0.08796296296 | 432 |  |
| eNeural-Qwen3.8-27B-fast | ok | 13.02917552 | 0.08610313951 | 0.01556708823 | 0.0787037037 | 432 |  |
| nvidia-Qwen3.8-27B-NVFP4 | ok | 12.87672579 | -0.06634659229 | 0.01838929671 | 0.09722222222 | 432 |  |
| unsloth-Qwen3.8-27B-NVFP4 | ok | 12.92673526 | -0.01633712056 | 0.0138618388 | 0.08564814815 | 432 |  |

Flip Rate 是 0–1 的比例；PPL Delta 為候選 PPL 減 BF16 PPL。CSV 保留完整浮點數，Markdown 顯示 10 位有效數字。

BF16 sanity：兩個獨立程序重新載入並執行。兩次 PPL 都是 12.943072381460716，KL=0、Flip=0，全部 8 份完整分布 SHA-256 相同。詳見 `final/sanity.json`。

簡化與消融：移除每個位置的 vocabulary 排序，改以 NaN 初值檢查是否覆蓋全部維度；仍檢查機率總和。候選與 BF16 repeat 不再保存不使用的完整分布檔案，只保存指標與雜湊；比較仍使用完整分布。另明確關閉引擎並記錄程序退出碼。簡化後重跑 BF16 sanity 及全部 7 個候選，所有指標完全相同，72 份分布的 SHA-256 全部相同。詳見 `ablation.json`、`simplification.diff` 及 `source_versions.json`。

另從第一輪保存的原始矩陣獨立重新計算所有指標，最大絕對差為 3.47e-18；詳見 `raw_distribution_audit.json`。4 個針對性 CPU 測試通過，涵蓋位置對齊、KL 方向、拒絕不完整 vocabulary、遮罩維度與 token 加權。沒有執行全測試套件。

已修正的環境問題：原先 torchcodec 錯裝 CUDA 13 版本，改用匹配的 CUDA 12.9 wheel；DeepGEMM 缺少 Toolkit 路徑，block FP8 固定使用 Triton；舊驅動無法處理部分 PTX，以專案內 forward-compatibility 函式庫解決；KV auto 會讀取 checkpoint 而改用 FP8，已改為明確指定 BF16。原先部分程序在數據寫出後出現 Python interpreter finalization 錯誤，改為在離開前明確 shutdown，最終所有程序均正常退出。原始診斷保留於 `initial/`、`before/`、`validated/` 與 `diagnose-radix.log`，未把這些失敗當成測量值。

限制：這是短樣本 fidelity 檢查，不能代表一般能力、長上下文或多模態品質；PPL 較低不能直接解讀為任務品質較好。H100 上 NVFP4 實際使用 Marlin W4A16，FP8 層使用 Triton 或 CUTLASS W8A8；因此結果不等同 Blackwell 原生 W4A4，也不能宣稱只測到權重捨入誤差，還包括量化 activation 與不同核心的數值影響。沒有測量推論速度或記憶體效益。

方法與執行命令見專案 `README.md`；模型來源版本及雜湊見 `checkpoints.json`。全部數值都來自實際執行，最終沒有無法啟動的模型。
