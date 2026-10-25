# Fidelity Public v2.1

固定版本的純文字 fidelity 測試資料。共 288 份獨立來源文件、55,296 個不同的計分 token 位置。每份文件只使用一次，不串接不同文件，不重複計分同一段 target。

| 領域 | 公開來源 | 文件數 | 初始 context tokens | 每份計分 tokens |
|---|---|---:|---|---:|
| 英文一般知識 | Wikimedia Wikipedia 20231101.en 第一個固定 parquet shard | 72 | 128 / 512 / 2048，各 24 份 | 192 |
| 中文一般知識 | Wikimedia Wikipedia 20231101.zh 第一個固定 parquet shard | 72 | 128 / 512 / 2048，各 24 份 | 192 |
| 形式化數學 | Mathlib4 的 Mathlib/*.lean，含定理、證明與說明 | 72 | 128 / 512 / 2048，各 24 份 | 192 |
| 程式碼 | CPython 的 Lib/test/**/*.py | 72 | 128 / 512 / 2048，各 24 份 | 192 |

所有來源的 commit / dataset revision、下載網址、SHA-256 與授權固定於 `../expanded_config.json`、`manifest.json`。資料由公開來源建立；沒有把檔案上傳到外部服務。`documents.jsonl` 保留完整原文、標題、逐文件來源網址、授權與雜湊；Wikipedia 作者名單可由各文章的編輯歷史取得。來源檔案含少量無效 UTF-8 的程式碼測試 fixture 時，以 Unicode replacement character 解碼；其餘不清理內容。授權文件在 `../expanded_licenses/`；Wikipedia 內容沿用 CC BY-SA 3.0 / GFDL，Mathlib Apache 2.0，CPython PSF 2.0。不可把混合資料宣稱為單一 Apache 授權。

選樣不使用模型分數。各領域以 seed 20261016 和來源 document ID 的 SHA-256 排序，先取足夠長的文件給 2048 context，再取 512、128；各 cell 24 份。只選沒有用過且完整原文雜湊不同的文件。用另一個固定 SHA-256 選文件內的 token offset。使用 BF16 tokenizer、`add_special_tokens=False`，不套 chat template、不加 BOS/EOS。tokenizer 路徑可改成本機相同版本；實際 token IDs 已保留於 `tokens.json`。

每個 prompt 長度為 context + 192，最後 192 tokens 計分。若 context=C，計分 input token indices C..C+191；vLLM 的 causal row 起點是 C-1。第一個 target 有 C 個前文 tokens，最後一個 target 有 C+191 個前文 tokens。三個 context 組使用不同文件，因此長度比較包含文件與位置差異，不能當成相同 target 上的因果實驗。

避免直接使用模型卡揭露的 CNN/DailyMail、PerfectBlend、Nemotron calibration corpus，也不使用常見 WikiText/C4/Pile benchmark 作為直接來源。已排除與完整 CNN/DailyMail train 有 13-word 片段重疊的兩份來源文件，排除 ID 固定於設定。公開來源仍可能被其他 calibration 集合引用；部分模型沒有公開完整 calibration 文本，不能保證沒有重疊。這也不是對基座預訓練資料去重的測試集。詳細檢查與限制見實驗報告。

數學組是 Lean 形式化數學，不代表自然語言數學問答能力；程式碼組主要是 Python 測試碼，不代表所有語言。抽樣範圍是已固定的來源 shard / repository，而不是整個網路。每個文件的 192 個 tokens 視為同一統計 cluster；信賴區間不把它們當成 192 份獨立樣本。

重新建立：

```sh
.vllm-venv/bin/python src/prepare_dataset.py --output build/dataset_rebuild
```

可使用 `--cache` 指定來源下載目錄、`--config` 指定同格式設定。輸出目錄必須不存在。重新建立的 documents.jsonl 與 tokens.json 應與本版逐位元組相同（相同 tokenizer 路徑）。
