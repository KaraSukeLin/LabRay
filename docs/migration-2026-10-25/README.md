# quant / decision：最終確認移轉版

版本：2026-10-25，第二次複核後。**此版本取代 migration-review-20261025；只需移轉此資料包及壓縮包校驗檔。** 已將指定原始檔案、精簡數據與必要依賴收齊，無須再從舊盤點清單逐一補抓。尚未推送到外部，原始兩專案未修改。

## 最終保留範圍

| 交付位置 | 保留內容 | 原因 |
|---|---|---|
| quant/src、scripts、tests、requirements、assets、REPORT | 實際算法、環境版本、固定文件/token IDs、正式設定及授權 | 可重建實驗條件、資料與統計方法 |
| quant/build/final、expanded、autoquant 的精選檔案 | 逐文件數據、BF16 sanity、模型/權重指紋、配對 CI、量化 allocation/sensitivity、校準稽核 | 支撐九模型 Fidelity 與量化配置推論；不保存完整分布矩陣 |
| quant/build/benchmark 的精選檔案 | 正式 IFEval 逐題分數/final answer、sampling/cap、配對表、pilot、官方重核、100K retry 與成本 | 可重算表格與配對差異，保留截斷/解析/輸出上限對結論的影響 |
| quant/build/terminal_bench 的精選檔案 | Low/MTP3 最終 protocol、801 matrix、554 attempts、控制/排除/缺分/重試、原 verifier 證據、資源數據、Agentic Fidelity prefix/逐窗口結果、PRoot shim | 保存 53 題主分析的實際分母及限制，不把環境失敗當模型失敗 |
| decision/experiments/orsharc 的程式、configs、lock、文件 | 資料分割、原生 decision head、訓練與校準/評分方法 | 可重建 native runtime 與實驗流程 |
| decision/.../artifacts/prepared、exports、plans、training 的精選檔案 | 固定 inputs/gold/partitions、訓練資料與命令、完整預算/loss/checkpoint 身份收據 | 維持 Seen/Unseen、巢狀訓練預算及資料隔離 |
| decision/.../artifacts/runs/phase4、reports、resource、smoke 的精選檔案 | 14 條件 48,692 筆預測/機率、calibration、原 metrics、資源與 admission/B5 adapter | 不需模型即可重建全部指標、曲線、圖表與成本摘要 |
| metadata/ | 154 份小型模型資產、固定上游版本/本地 diff、OneJev readout、來源及萃取記錄 | 補齊主專案外的實際模板、runtime 修改與來源鏈 |
| tools/ | 完整性/數據驗證、模型小型資產還原、報表重建、敏感度還原與分析相容工具 | 精簡後仍有明確且已測的重算入口，改動附可閱讀 diff |

## 第二次複核的修改

- **補齊：** 本地 Google IFEval scorer 原始碼及 LICENSE、MMMU-Pro Python 評分器、upstream receipts/patch，以及 pilot 成本分析實際需要的 loading/warmup log 數據。
- **精簡：** 敏感度分析的兩份完整模型複本改成三筆替換列；省略舊 analysis-before-retry 表、前版 22 MiB 全樹掃描索引與舊匯出腳本。
- **省略可下載素材：** Terminal-Bench 官方 task fixtures 約 56 MiB，改保留固定 repo/revision、1,035 個正式檔案及兩個上層檔案的 SHA-256。這些素材可從固定公開版本取回，不影響現有分數重算。
- **去重：** 重複 tokens、輸入、結果與模型資產按內容 SHA-256 保存一份，保留原邏輯路徑；壓縮包使用內部 hardlinks，不連到外部檔案。
- **改寫重現說明：** 明確區分已保存的數值與需重跑的軌跡。補上精簡欄位的 hash/count 分析介面；另明定 decision 報表使用 Python 3.12。

## 已完成驗證

- IFEval 16,230 筆，30 cells 各 541 題；十組正式 accuracy 與 266 列 paired estimates 重算符合原值。
- 九模型 Fidelity，各 288 文件／55,296 scoring tokens；由逐文件數據重算 PPL、KL 與 flip rate 符合原值。
- Pilot 的 192 列配對與 192 列加權配對點估計、九模型載入成本符合原值；token 數、空答案與配對 identity 檢查通過。
- decision 14 條件 × 三分割＝42 份完整報表與總摘要，已從本包預測/校準/gold 重建並以 1e-12 容差比對通過；包括 risk–coverage，而非只比 headline accuracy。
- Terminal-Bench 801 格、533 有分數、53 題主分析、各模型 159 observations，重算成功率符合原值；三筆 cap sensitivity 替換的相同 prompt 身份已核對。
- 全部交付檔案另有 SHA-256；封裝後逐一驗證 archive 成員與數據。

本次只執行這些資料與 CPU 定向檢查，無模型下載/載入、訓練、GPU 推論或完整測試套件。分析相容性檢查用 19 次 resamples；原 199,999 次正式 CI 保留，沒有被小次數結果覆寫。詳見 metadata/ 中的驗證收據與 REPRODUCE.md。

## 明確不移轉的內容

不含模型/LoRA/merged 權重、虛擬環境本體、完整 vocabulary `.npy`、搜尋 `.pt` state、可再下載的大型資料/OCI layers、rootfs、工具工作目錄、錄影、pane、完整 ATIF/API/rollout，以及可重建的 scored 複本與圖檔。相同條件重新推論仍需取得權重及公開素材；重查已省略的工具或 reasoning 過程則需重跑。

## 報告解讀不可省略

Terminal-Bench 是 PRoot 內部比較，主分母為 53 題，不是完整官方 Docker leaderboard；環境排除、缺分及 CI 要和成功率一併引用，97.04% 指服務存活率。IFEval 的 cap 差異與補測不應被主表掩蓋。decision B4 重用 B1、B2 是另一個任務，B5 是跨模型約等 VRAM 比較；未做多 seed 顯著性檢定或 Elephant 量化損失實驗。

## 檔案索引

- `FILE-MANIFEST.csv`：最終實際交付的逐檔路徑、分類、大小及 SHA-256。
- `source-manifest.csv`：來源路徑、轉換方式與來源選取群組。
- `SHA256SUMS`：解壓縮後的完整性檢查入口。
- `REPRODUCE.md`：固定版本、還原順序、離線重算命令與明確限制。
- `metadata/package-size.json`：實際檔案數、邏輯與去重後大小；壓縮包大小另列於包旁交付收據。

此清單即為建議最終移轉範圍。先前約 656 MiB 的「原檔＋盤點目錄」方案不再使用。
