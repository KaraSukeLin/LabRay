# LabRay

兩個實驗各自存放，程式、報告、數據和重現工具可直接在 GitHub 瀏覽。

| 專案 | 內容 | 入口 |
|---|---|---|
| [quant](quant/) | 量化 Fidelity、IFEval、MMMU-Pro pilot、Terminal-Bench | [專案說明](quant/README.md) · [重現方式](quant/migration/REPRODUCE.md) |
| [decision](decision/) | OR-ShARC 原生 decision model 訓練、校準與評測 | [專案說明](decision/README.md) · [重現方式](decision/migration/REPRODUCE.md) |

每個專案的 `migration/` 包含自己的資料清單、SHA-256、模型設定、來源紀錄與工具；兩個專案可以分別複製使用。

本版由先前已驗證的移轉資料包展開並分組，研究檔案逐一核對原 SHA-256。舊的合併壓縮包與重複文件已從目前版本移除，仍可在先前 commit 查看。

保留範圍為重現資訊與關鍵證據數據。不含大型權重、虛擬環境、完整機率矩陣、rootfs 或完整推論軌跡；完整重新推論需依各專案說明重建環境與取得公開素材。
