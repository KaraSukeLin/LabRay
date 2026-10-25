# Smoke 非 PASS 原因

MODEL_FAIL 是原官方 Verifier 的 assertion 未通過；四次皆無 native infrastructure exception。

| 模型 | 題目 | interleaved | 原官方失敗原因 |
|---|---|---|---|
| BF16 | extract-elf | False | test_output_matches_reference：參考位址覆蓋率 0.00%，要求至少 75% |
| NaviQuant | extract-elf | False | 同上 |
| Aqua | extract-elf | False | 同上 |
| Aqua | overfull-hbox | True | test_input_file_matches：younger → newer 不符合指定同義詞表 |

extract-elf 各有一項測試通過、一項失敗；Aqua overfull-hbox 三項通過、一項失敗。上述是原評分器觀察，未推斷三模型抽取失敗具有相同程式根因，也不能直接歸因於量化。

另兩次 TIMEOUT 為 NaviQuant True 的 overfull-hbox（原 Agent timeout 750 秒）與 extract-elf（900 秒），分類與 MODEL_FAIL 分開。原始 CTRF、result、trajectory、recording 在各 trial 目錄；所有失敗皆保留，沒有自動重跑。
