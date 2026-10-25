# decision：校驗與重現

所有命令從本專案目錄執行。本目錄已展開，不需合併資料包或另一個實驗專案。研究檔案與原移轉快照的對照在 `original-files.csv`；目前檔案完整性由 `FILE-MANIFEST.csv` 和 `SHA256SUMS` 記錄，這三份清單都位於本 `migration/` 目錄。

## 校驗與模型小型資產

```sh
sha256sum -c migration/SHA256SUMS
python3 migration/tools/verify-files.py
python3 migration/tools/restore-model-metadata.py --output /tmp/decision-model-assets
```

模型設定目錄須是新的或空目錄。此專案的 index 包含 26 個小型檔案，位於 `migration/metadata/model-metadata/index.json`；blob 僅包含此專案使用的資產，相同內容以 SHA-256 去重。模型權重不在 repo 內，依原 repo/revision 取得或依訓練 recipe 重建，再核對權重指紋與本次實際 tokenizer/template/readout。

歷史 manifest 的絕對來源路徑保留不變；新檔案位置以本文件為準。只重算保存的數據可直接在目前 checkout 進行；完整推論若改變模型、資料或程式位置，仍需核對固定 identity 與執行環境。

## 報表與完整實驗

**使用 Python 3.12**，與原實驗一致。Python 3.11 的 math 浮點尾差會改變部分同 confidence 值的分組；原完整 metrics/risk–coverage curves 已保存。以下工具會用原 report 程式重建 14 條件的 dev_select/dev_calibrate/test 報表，再重建總摘要及 HTML/SVG，並以 1e-12 容差逐項核對原數據：

```sh
python3.12 migration/tools/rebuild-decision-reports.py --output /tmp/decision-rebuilt
```

輸出目錄必須是新的。這只需標準函式庫，不安裝套件、不載入模型、不重訓。保留的 predictions.jsonl、calibration.json、prepared/sealed、manifest、complete 與 metrics 足以完成此流程；不用移轉重複的 scored_raw/scored_calibrated 或圖檔。

完整實驗則依 experiments/orsharc/EXPERIMENT_B_REPORT.md、PHASE4.md、configs/、實際 training command/args 及 phase4 plans 重建。Elephant runtime 使用 elephant-runtime.lock：Torch 2.8.0/CUDA 12.8、Triton 3.7.1、Transformers 5.19.0；保留原已驗證 override，不讓 resolver 自動替換版本。三種訓練預算均為完整一輪，保留 native loss、complete、checkpoint hashes、readout、ft_exposure；不保存 adapter/merged weights。B4 重用 B1，不能當成獨立重複。

OneJev 的實際本地修改在 migration/metadata/upstream/onejev.json，未追蹤的 readout.py 改放 migration/metadata/onejev-readout.py，避免還原後先產生不完整 checkout 目錄。流程是：先 checkout 固定 OneJev HEAD，再套用相對該 HEAD 的完整保存 diff，最後複製 readout.py 到 qev/。該 diff 已包含 Elephant patch 的修改，不要再次重複套用 patch。原 patch 及 upstream receipts 另保存於 artifacts/upstream/，供驗證來源。其餘 Kev/StartLux/Unsloth 原始 checkout 為乾淨版本，固定 SHA 已保存。

每條件先 dev_select=456 選 threshold，再 dev_calibrate=649 fit temperature，最後 test=2,373。B5 是 StartLux 27B Instant Q4_K_M，不是 Elephant 量化；實際 repo/revision、llama.cpp b11804 commit、image digest、兩段 temperature 與 native API adapter 在 phase4-preflight 及 B5 smoke。Qwen LoRA 與 Elephant 原生量化仍未驗證，不加入正式結果。


## 重現界限

原 predictions、calibration、metrics 與訓練收據逐檔保留。完整重新訓練需要重建原環境及權重，不能假設重訓 checkpoint 與原 checkpoint 逐 byte 相同。Python 3.12 的報表重建已驗證，原資料與歷史 manifest 沒有為配合新路徑而改寫。
