# 一次性 pre-agent tmux 重試逐格對照

14 個 attempt0 均為 Agent/API 執行前的 ENVIRONMENT_FAIL；固定同 task/model/seed，只允許 attempt1。此表使用最終資料，與原 amendment 建立時的 PENDING 快照不同。

| Seed | 模型 | Task | Attempt 0 ID | Attempt 1 ID | 最終狀態 | 原 reward |
|---|---|---|---|---|---|---:|
| 20261021 | Aqua | db-wal-recovery | `fd3240f93e39e12ec060893a` | `6e333973459eb121905a7849` | TIMEOUT | 0.0 |
| 20261021 | BF16 | dna-insert | `e8941c5e834c16c7b9a02984` | `64e64142d17c6e1cd8057a7d` | PASS | 1.0 |
| 20261021 | Aqua | dna-insert | `4529d0e69ec02cf3f60765e8` | `7503850060478eb7e7e7818e` | MODEL_FAIL | 0.0 |
| 20261021 | Aqua | extract-elf | `c413bb229f645d68bb6fcb31` | `3eb61ccb98f75802332110ca` | MODEL_FAIL | 0.0 |
| 20261021 | Aqua | financial-document-processor | `435edc17f9db6ea59dc110e1` | `f82da04311236a102d0ac647` | TIMEOUT | 0.0 |
| 20261021 | Fast | git-leak-recovery | `b657097b4cc3eb58ecb9e1b9` | `fe3c0e5473f04258d9a31c19` | PASS | 1.0 |
| 20261021 | Fast | llm-inference-batching-scheduler | `6aa4b7f43a2af0289f455b86` | `901f0b8175e8876f39d76cb1` | PASS | 1.0 |
| 20261021 | Aqua | overfull-hbox | `dedc8d580feb03dc4f551a5e` | `7da7663364cb7976471cefa8` | PASS | 1.0 |
| 20261021 | BF16 | password-recovery | `bef913943f15d67d7d0c42ad` | `c00fa110b8143c8978c83f59` | PASS | 1.0 |
| 20261021 | Fast | polyglot-rust-c | `ec81d9060e259b1d770c1559` | `0fc646cd6def806b0e2e885b` | PASS | 1.0 |
| 20261021 | Fast | regex-log | `ac409194e7619fbb430fe142` | `e8b4faf5c09843de617238c9` | PASS | 1.0 |
| 20261022 | Aqua | dna-assembly | `c9160e6e7bf71ccb698773fa` | `06d88551231091a5f63aafee` | TIMEOUT | 0.0 |
| 20261022 | Aqua | extract-moves-from-video | `1e5f247e6cb0cb0d1b5d4134` | `50225483dc3ab4c8f4fd183f` | TIMEOUT | 0.0 |
| 20261022 | Fast | git-leak-recovery | `eac56e21d69654c94ccd3418` | `c367a1db0800b21c6a72a42c` | PASS | 1.0 |

結果為 8 PASS、4 TIMEOUT、2 MODEL_FAIL。未重跑正常模型失敗或 timeout；完整 result 路徑見 [retry_outcomes.csv](retry_outcomes.csv)。
