# quant

量化模型的 Fidelity 與能力評測專案。此目錄已包含篩選後的程式、固定輸入、實驗數據及重現資訊，不需解開合併資料包，也不依賴 `decision/`。

## 研究報告

- [九模型 Fidelity、IFEval、pilot 整合報告](REPORT.md)
- [Terminal-Bench 最終報告入口](build/terminal_bench/FINAL_REPORT.md)
- [Terminal-Bench 三模型三 seed 完整報告](build/terminal_bench/low_mtp3/formal_three_seed_20261022/FINAL_REPORT_ZH_TW.md)

IFEval 正式矩陣為 16,230 筆。Terminal-Bench 有 801 格終態、533 格有效分數；主分析使用共同可評分的 53 題 × 3 seeds／模型，屬 PRoot 內部比較。引用結果時需一併保留原報告的排除、缺分與 CI。

## 目錄

| 位置 | 內容 |
|---|---|
| `src/`、`scripts/`、`tests/` | 原實驗程式、執行腳本與定向檢查 |
| `assets/` | 固定文本/token IDs、正式設定、來源 manifest 與授權 |
| `build/` | 已完成實驗的精選數據、結果、統計與原稽核證據 |
| `migration/metadata/` | 此專案的模型設定、固定上游來源、萃取與驗證紀錄 |
| `migration/tools/` | 校驗、精簡數據分析、cap sensitivity 重建與模型設定還原工具 |

## 校驗與重現

從本目錄執行：

```sh
sha256sum -c migration/SHA256SUMS
python3 migration/tools/verify-evidence.py
```

- [重現命令與環境要求](migration/REPRODUCE.md)
- [移轉內容與路徑變更](migration/README.md)
- [目前檔案清單](migration/FILE-MANIFEST.csv)
- [原研究檔案逐檔對照](migration/original-files.csv)

大型模型權重與完整 vocabulary 分布不在 repo 內；完整重新推論需取得固定權重並先做 BF16 sanity。`README-original.md` 保留原交接紀錄，部分舊狀態文字已過時；完成狀態以此頁連結的最終報告為準。
