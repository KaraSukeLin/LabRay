# decision / OR-ShARC

OR-ShARC 原生 decision model 的訓練、信心校準與評測專案。程式、資料、報告及重現工具均在此目錄，不依賴 `quant/`。

## 研究報告

- [完整 Experiment B 報告](experiments/orsharc/EXPERIMENT_B_REPORT.md)
- [簡報洞見與比較限制](experiments/orsharc/PRESENTATION_INSIGHTS.md)
- [實驗命令與架構](experiments/orsharc/README.md)

已完成 14 個唯一條件，每個條件各有 dev_select 456 筆、dev_calibrate 649 筆、test 2,373 筆，合計 48,692 筆條件－樣本數據。B4 重用 B1；B5 為 Elephant BF16 與 StartLux Q4_K_M 的跨模型比較。

## 目錄

| 位置 | 內容 |
|---|---|
| `experiments/orsharc/src/`、`configs/` | 原生 runtime、資料處理、訓練、校準與評分程式 |
| `experiments/orsharc/artifacts/prepared/`、`exports/`、`plans/` | 固定分割、隔離 gold、訓練資料與正式命令 |
| `experiments/orsharc/artifacts/runs/phase4/` | 14 條件的逐筆預測、機率、校準、完成收據與指標 |
| `experiments/orsharc/artifacts/training/`、`smoke/`、`resource/` | 訓練、相容性、B5 與資源證據 |
| `migration/` | 此專案的檔案清單、模型小型資產、上游修補與重建工具 |

## 校驗與重建報表

從本目錄執行：

```sh
python3 migration/tools/verify-files.py
python3.12 migration/tools/rebuild-decision-reports.py --output /tmp/decision-rebuilt
```

第二個命令使用原 Python 3.12，只讀保存數據即可重建 42 份報表與總摘要，不需要模型權重或 GPU。輸出目錄必須是新的。

- [重現命令與環境要求](migration/REPRODUCE.md)
- [移轉內容與路徑變更](migration/README.md)
- [目前檔案清單](migration/FILE-MANIFEST.csv)
- [原研究檔案逐檔對照](migration/original-files.csv)

大型權重及訓練 adapters 未移轉；完整重跑需依固定版本取得或重訓。`README-original.md` 是原 Phase 1 交接文字，最新成果以上述完整報告為準。
