# LabRay

保存 `quant` 與 `decision / OR-ShARC` 已完成實驗的重現資訊及關鍵證據數據。

此版本為 2026-10-25 第二次複核後的最終移轉版。資料包已收齊指定內容，不需要原環境或先前盤點目錄即可取得保留的檔案；完整重新推論仍需依固定版本下載權重、公開素材並重建環境。

## 下載與驗證

建議移轉以下兩個檔案：

- [最終資料包：85.1 MiB](migration/2026-10-25/migration-final-20261025.tar.gz)
- [SHA-256 校驗檔](migration/2026-10-25/migration-final-20261025.tar.gz.sha256)

Clone 此 repo 後，在 repo 根目錄執行：

```sh
cd migration/2026-10-25
sha256sum -c migration-final-20261025.tar.gz.sha256
tar -xzf migration-final-20261025.tar.gz
cd migration-final-20261025
sha256sum -c SHA256SUMS
python3 tools/verify-evidence.py
```

資料包的 SHA-256：

```text
c651dc879ffffa888b4c66070c30a01f8bdd60137e73a863abd0d41c70b98155
```

## 保留範圍

- 實際程式、固定設定、套件版本、tokenizer/template、模型指紋與必要 runtime 修補。
- 九模型 Fidelity 的逐文件數據、配對統計、sanity、校準與量化搜尋證據。
- IFEval 正式逐題數據、pilot、cap 差異、100K 補測、官方評分器與成本分析依據。
- Terminal-Bench 的 801 格終態、554 attempts、控制組、排除/缺分/重試，以及 Agentic Fidelity 數據。
- decision 的 14 條件、48,692 筆預測/機率、校準、完整訓練收據與資源數據。

不包含大型模型或 LoRA 權重、虛擬環境、完整機率矩陣、rootfs、工具工作目錄、錄影或完整推論軌跡。

## 說明與驗證

- [最終移轉範圍與保留理由](docs/migration-2026-10-25/README.md)
- [還原順序、版本與重算命令](docs/migration-2026-10-25/REPRODUCE.md)
- [最終逐檔清單](docs/migration-2026-10-25/FILE-MANIFEST.csv)
- [來源與精簡方式](docs/migration-2026-10-25/source-manifest.csv)
- [資料包交付與完整性收據](docs/migration-2026-10-25/delivery.json)

這些文件描述資料包解開後的目錄。包內已保存完整指引與工具；repo 的 `docs/` 是便於直接閱讀的副本。

已從保留數據重建 decision 的 42 份報表與總摘要，並核對 Fidelity、IFEval、pilot 配對點估計及成本。壓縮包內全部 6,840 個檔案已逐一核對 SHA-256。只執行資料與 CPU 定向驗證，沒有重新訓練或 GPU 推論。

decision 報表重建須使用原實驗的 Python 3.12。Terminal-Bench 主分析為共同可評分的 53 題 × 3 seeds／模型，屬 PRoot 內部比較；環境排除、缺分及 CI 須與成功率一起引用。其他限制詳見移轉說明及包內原研究報告。
