# 共同排除的 29 題：控制證據

這些是共同環境／評分 Gate 的排除，三模型、三 Seed 均相同；不是正式模型答錯。表格只陳述保存的控制結果，不將全部原因歸為 PRoot 不相容。完整 89 題、每題 Oracle/NOP 的結果路徑見 [environment_control_evidence.csv](environment_control_evidence.csv)。

| Task | Gate | Oracle 狀態／reward | NOP 狀態／reward | 原始 Oracle 證據 |
|---|---|---|---|---|
| bn-fit-modify | ORACLE_CONTROL_FAIL | MODEL_FAIL／0.0 | MODEL_FAIL／0.0 | [result](../../controls/bn-fit-modify/oracle/attempt-0/trial/result.json) |
| break-filter-js-from-html | ORACLE_CONTROL_FAIL | MODEL_FAIL／0.0 | MODEL_FAIL／0.0 | [result](../../controls/break-filter-js-from-html/oracle/attempt-0/trial/result.json) |
| build-cython-ext | ORACLE_CONTROL_FAIL | MODEL_FAIL／0.0 | MODEL_FAIL／0.0 | [result](../../controls/build-cython-ext/oracle/attempt-0/trial/result.json) |
| build-pmars | ORACLE_CONTROL_FAIL | MODEL_FAIL／0.0 | MODEL_FAIL／0.0 | [result](../../controls/build-pmars/oracle/attempt-0/trial/result.json) |
| build-pov-ray | ORACLE_CONTROL_FAIL | MODEL_FAIL／0.0 | MODEL_FAIL／0.0 | [result](../../controls/build-pov-ray/oracle/attempt-0/trial/result.json) |
| caffe-cifar-10 | ORACLE_CONTROL_FAIL | MODEL_FAIL／0.0 | MODEL_FAIL／0.0 | [result](../../controls/caffe-cifar-10/oracle/attempt-0/trial/result.json) |
| compile-compcert | ORACLE_CONTROL_FAIL | MODEL_FAIL／0.0 | MODEL_FAIL／0.0 | [result](../../controls/compile-compcert/oracle/attempt-1/trial/result.json) |
| configure-git-webserver | ORACLE_CONTROL_FAIL | MODEL_FAIL／0.0 | MODEL_FAIL／0.0 | [result](../../controls/configure-git-webserver/oracle/attempt-1/trial/result.json) |
| custom-memory-heap-crash | ORACLE_CONTROL_FAIL | MODEL_FAIL／0.0 | MODEL_FAIL／0.0 | [result](../../controls/custom-memory-heap-crash/oracle/attempt-1/trial/result.json) |
| git-multibranch | ORACLE_CONTROL_FAIL | MODEL_FAIL／0.0 | MODEL_FAIL／0.0 | [result](../../controls/git-multibranch/oracle/attempt-1/trial/result.json) |
| headless-terminal | ORACLE_CONTROL_FAIL | MODEL_FAIL／0.0 | MODEL_FAIL／0.0 | [result](../../controls/headless-terminal/oracle/attempt-1/trial/result.json) |
| hf-model-inference | ORACLE_CONTROL_FAIL | MODEL_FAIL／0.0 | MODEL_FAIL／0.0 | [result](../../controls/hf-model-inference/oracle/attempt-1/trial/result.json) |
| install-windows-3.11 | ORACLE_CONTROL_FAIL | MODEL_FAIL／0.0 | MODEL_FAIL／0.0 | [result](../../controls/install-windows-3.11/oracle/attempt-1/trial/result.json) |
| log-summary-date-ranges | ORACLE_CONTROL_FAIL | MODEL_FAIL／0.0 | MODEL_FAIL／0.0 | [result](../../controls/log-summary-date-ranges/oracle/attempt-0/trial/result.json) |
| mailman | ORACLE_CONTROL_FAIL | MODEL_FAIL／0.0 | MODEL_FAIL／0.0 | [result](../../controls/mailman/oracle/attempt-0/trial/result.json) |
| mcmc-sampling-stan | ORACLE_CONTROL_FAIL | MODEL_FAIL／0.0 | MODEL_FAIL／0.0 | [result](../../controls/mcmc-sampling-stan/oracle/attempt-0/trial/result.json) |
| nginx-request-logging | ORACLE_CONTROL_FAIL | MODEL_FAIL／0.0 | MODEL_FAIL／0.0 | [result](../../controls/nginx-request-logging/oracle/attempt-0/trial/result.json) |
| path-tracing | ORACLE_CONTROL_FAIL | MODEL_FAIL／0.0 | MODEL_FAIL／0.0 | [result](../../controls/path-tracing/oracle/attempt-0/trial/result.json) |
| path-tracing-reverse | ORACLE_CONTROL_FAIL | MODEL_FAIL／0.0 | MODEL_FAIL／0.0 | [result](../../controls/path-tracing-reverse/oracle/attempt-0/trial/result.json) |
| protein-assembly | ORACLE_CONTROL_FAIL | MODEL_FAIL／0.0 | MODEL_FAIL／0.0 | [result](../../controls/protein-assembly/oracle/attempt-0/trial/result.json) |
| pypi-server | ORACLE_CONTROL_FAIL | MODEL_FAIL／0.0 | MODEL_FAIL／0.0 | [result](../../controls/pypi-server/oracle/attempt-0/trial/result.json) |
| qemu-alpine-ssh | ORACLE_CONTROL_FAIL | VERIFIER_FAIL／None | VERIFIER_FAIL／None | [result](../../controls/qemu-alpine-ssh/oracle/attempt-0/trial/result.json) |
| qemu-startup | ORACLE_CONTROL_FAIL | VERIFIER_FAIL／None | VERIFIER_FAIL／None | [result](../../controls/qemu-startup/oracle/attempt-0/trial/result.json) |
| query-optimize | ORACLE_CONTROL_FAIL | VERIFIER_FAIL／None | VERIFIER_FAIL／None | [result](../../controls/query-optimize/oracle/attempt-0/trial/result.json) |
| reshard-c4-data | SCORING_GATE_FAIL | VERIFIER_FAIL／0.0 | VERIFIER_FAIL／0.0 | [result](../../controls/reshard-c4-data/oracle/attempt-0/trial/result.json) |
| rstan-to-pystan | ORACLE_CONTROL_FAIL | MODEL_FAIL／0.0 | MODEL_FAIL／0.0 | [result](../../controls/rstan-to-pystan/oracle/attempt-0/trial/result.json) |
| sqlite-with-gcov | ORACLE_CONTROL_FAIL | MODEL_FAIL／0.0 | MODEL_FAIL／0.0 | [result](../../controls/sqlite-with-gcov/oracle/attempt-0/trial/result.json) |
| torch-pipeline-parallelism | ORACLE_CONTROL_FAIL | VERIFIER_FAIL／None | MODEL_FAIL／0.0 | [result](../../controls/torch-pipeline-parallelism/oracle/attempt-0/trial/result.json) |
| torch-tensor-parallelism | ORACLE_CONTROL_FAIL | MODEL_FAIL／0.0 | MODEL_FAIL／0.0 | [result](../../controls/torch-tensor-parallelism/oracle/attempt-0/trial/result.json) |

`reward=None` 是沒有原 reward；`reward=0` 也不保證 CTRF 完整。各 Gate 的具體判定以保存的 attempts/checked/error 欄位為準。

## 原 Oracle Verifier 的具體失敗

下表摘取原 CTRF summary 與失敗 test 名稱，不把部分 tests 通過視為 Oracle 成功。這是直接失敗位置，不宣稱已完成每題根因診斷。

| Task | Tests passed / total | Failed tests（最多前三項）／直接例外 | 原 stdout／CTRF |
|---|---:|---|---|
| bn-fit-modify | 0/9 | test_outputs.py::test_bn_sample_exists; test_outputs.py::test_learned_dag_structure_exists; test_outputs.py::test_learned_dag_structure_csv_col_names | [stdout](../../controls/bn-fit-modify/oracle/attempt-0/trial/verifier/test-stdout.txt) / [CTRF](../../controls/bn-fit-modify/oracle/attempt-0/trial/verifier/ctrf.json) |
| break-filter-js-from-html | 0/1 | test_outputs.py::test_out_html_bypasses_filter | [stdout](../../controls/break-filter-js-from-html/oracle/attempt-0/trial/verifier/test-stdout.txt) / [CTRF](../../controls/break-filter-js-from-html/oracle/attempt-0/trial/verifier/ctrf.json) |
| build-cython-ext | 10/11 | test_outputs.py::test_pyknotid_repository_tests | [stdout](../../controls/build-cython-ext/oracle/attempt-0/trial/verifier/test-stdout.txt) / [CTRF](../../controls/build-cython-ext/oracle/attempt-0/trial/verifier/ctrf.json) |
| build-pmars | 0/4 | test_outputs.py::test_pmars_works; test_outputs.py::test_headless_no_x11; test_outputs.py::test_debian_source_used | [stdout](../../controls/build-pmars/oracle/attempt-0/trial/verifier/test-stdout.txt) / [CTRF](../../controls/build-pmars/oracle/attempt-0/trial/verifier/ctrf.json) |
| build-pov-ray | 0/3 | test_outputs.py::test_illum1_render_and_verify; test_outputs.py::test_povray_version; test_outputs.py::test_povray_built_from_correct_source | [stdout](../../controls/build-pov-ray/oracle/attempt-0/trial/verifier/test-stdout.txt) / [CTRF](../../controls/build-pov-ray/oracle/attempt-0/trial/verifier/ctrf.json) |
| caffe-cifar-10 | 2/6 | test_outputs.py::test_caffe_version_and_source; test_outputs.py::test_cifar10_model_exists; test_outputs.py::test_training_completed_500_iterations | [stdout](../../controls/caffe-cifar-10/oracle/attempt-0/trial/verifier/test-stdout.txt) / [CTRF](../../controls/caffe-cifar-10/oracle/attempt-0/trial/verifier/ctrf.json) |
| compile-compcert | 0/3 | test_outputs.py::test_compcert_exists_and_executable; test_outputs.py::test_compcert_valid_and_functional; test_outputs.py::test_compcert_rejects_unsupported_feature | [stdout](../../controls/compile-compcert/oracle/attempt-1/trial/verifier/test-stdout.txt) / [CTRF](../../controls/compile-compcert/oracle/attempt-1/trial/verifier/ctrf.json) |
| configure-git-webserver | 0/1 | test_outputs.py::test_hello_html_exists | [stdout](../../controls/configure-git-webserver/oracle/attempt-1/trial/verifier/test-stdout.txt) / [CTRF](../../controls/configure-git-webserver/oracle/attempt-1/trial/verifier/ctrf.json) |
| custom-memory-heap-crash | 5/6 | test_outputs.py::test_no_memory_leaks_with_valgrind | [stdout](../../controls/custom-memory-heap-crash/oracle/attempt-1/trial/verifier/test-stdout.txt) / [CTRF](../../controls/custom-memory-heap-crash/oracle/attempt-1/trial/verifier/ctrf.json) |
| git-multibranch | 0/1 | test_outputs.py::test_multi_branch_https_deploy | [stdout](../../controls/git-multibranch/oracle/attempt-1/trial/verifier/test-stdout.txt) / [CTRF](../../controls/git-multibranch/oracle/attempt-1/trial/verifier/ctrf.json) |
| headless-terminal | 6/7 | test_outputs.py::test_background_commands | [stdout](../../controls/headless-terminal/oracle/attempt-1/trial/verifier/test-stdout.txt) / [CTRF](../../controls/headless-terminal/oracle/attempt-1/trial/verifier/ctrf.json) |
| hf-model-inference | 2/4 | test_outputs.py::test_sentiment_endpoint; test_outputs.py::test_api_error_handling | [stdout](../../controls/hf-model-inference/oracle/attempt-1/trial/verifier/test-stdout.txt) / [CTRF](../../controls/hf-model-inference/oracle/attempt-1/trial/verifier/ctrf.json) |
| install-windows-3.11 | 1/4 | test_outputs.py::test_network_status; test_outputs.py::test_qemu_running_with_correct_params; test_outputs.py::test_windows_keys_with_visual_feedback | [stdout](../../controls/install-windows-3.11/oracle/attempt-1/trial/verifier/test-stdout.txt) / [CTRF](../../controls/install-windows-3.11/oracle/attempt-1/trial/verifier/ctrf.json) |
| log-summary-date-ranges | 1/2 | test_outputs.py::test_summary_structure_and_counts | [stdout](../../controls/log-summary-date-ranges/oracle/attempt-0/trial/verifier/test-stdout.txt) / [CTRF](../../controls/log-summary-date-ranges/oracle/attempt-0/trial/verifier/ctrf.json) |
| mailman | 1/3 | test_outputs.py::test_simple_local_delivery; test_outputs.py::test_join_announce_leave_flow | [stdout](../../controls/mailman/oracle/attempt-0/trial/verifier/test-stdout.txt) / [CTRF](../../controls/mailman/oracle/attempt-0/trial/verifier/ctrf.json) |
| mcmc-sampling-stan | 2/6 | test_outputs.py::test_rstan_package_installed; test_outputs.py::test_hierarchical_model_implemented; test_outputs.py::test_r_script_created_and_used_rstan | [stdout](../../controls/mcmc-sampling-stan/oracle/attempt-0/trial/verifier/test-stdout.txt) / [CTRF](../../controls/mcmc-sampling-stan/oracle/attempt-0/trial/verifier/ctrf.json) |
| nginx-request-logging | 3/8 | test_outputs.py::test_nginx_running; test_outputs.py::test_index_page_content; test_outputs.py::test_custom_404_page | [stdout](../../controls/nginx-request-logging/oracle/attempt-0/trial/verifier/test-stdout.txt) / [CTRF](../../controls/nginx-request-logging/oracle/attempt-0/trial/verifier/ctrf.json) |
| path-tracing | 3/5 | test_outputs.py::test_runs_and_produces_output; test_outputs.py::test_image_similarity | [stdout](../../controls/path-tracing/oracle/attempt-0/trial/verifier/test-stdout.txt) / [CTRF](../../controls/path-tracing/oracle/attempt-0/trial/verifier/ctrf.json) |
| path-tracing-reverse | 2/3 | test_outputs.py::test_image_similarity | [stdout](../../controls/path-tracing-reverse/oracle/attempt-0/trial/verifier/test-stdout.txt) / [CTRF](../../controls/path-tracing-reverse/oracle/attempt-0/trial/verifier/ctrf.json) |
| protein-assembly | 0/1 | test_outputs.py::test_gblock | [stdout](../../controls/protein-assembly/oracle/attempt-0/trial/verifier/test-stdout.txt) / [CTRF](../../controls/protein-assembly/oracle/attempt-0/trial/verifier/ctrf.json) |
| pypi-server | 0/1 | test_outputs.py::test_api | [stdout](../../controls/pypi-server/oracle/attempt-0/trial/verifier/test-stdout.txt) / [CTRF](../../controls/pypi-server/oracle/attempt-0/trial/verifier/ctrf.json) |
| qemu-alpine-ssh | — | RewardFileNotFoundError: No reward file found at /root/projects/quant/build/terminal_bench/low_mtp3/controls/qemu-alpine-ssh/oracle/attempt-0/trial/verifier/reward.txt or /root/projects/quant/build/terminal_bench/low_mtp3/controls/qemu-alpine-ssh/oracle/attempt-0/trial/verifier/r | [result](../../controls/qemu-alpine-ssh/oracle/attempt-0/trial/result.json) |
| qemu-startup | — | RewardFileNotFoundError: No reward file found at /root/projects/quant/build/terminal_bench/low_mtp3/controls/qemu-startup/oracle/attempt-0/trial/verifier/reward.txt or /root/projects/quant/build/terminal_bench/low_mtp3/controls/qemu-startup/oracle/attempt-0/trial/verifier/reward. | [result](../../controls/qemu-startup/oracle/attempt-0/trial/result.json) |
| query-optimize | — | VerifierTimeoutError: Verifier execution timed out after 1800.0 seconds | [result](../../controls/query-optimize/oracle/attempt-0/trial/result.json) |
| reshard-c4-data | — | error: no /proc/self/exe available. Is /proc mounted? | [stdout](../../controls/reshard-c4-data/oracle/attempt-0/trial/verifier/test-stdout.txt) |
| rstan-to-pystan | 1/6 | test_outputs.py::test_output_files_exist; test_outputs.py::test_alpha_estimation_accuracy; test_outputs.py::test_sigma_estimation_accuracy | [stdout](../../controls/rstan-to-pystan/oracle/attempt-0/trial/verifier/test-stdout.txt) / [CTRF](../../controls/rstan-to-pystan/oracle/attempt-0/trial/verifier/ctrf.json) |
| sqlite-with-gcov | 0/3 | test_outputs.py::test_sqlite_compiled; test_outputs.py::test_sqlite_in_path; test_outputs.py::test_gcov_enabled | [stdout](../../controls/sqlite-with-gcov/oracle/attempt-0/trial/verifier/test-stdout.txt) / [CTRF](../../controls/sqlite-with-gcov/oracle/attempt-0/trial/verifier/ctrf.json) |
| torch-pipeline-parallelism | — | VerifierTimeoutError: Verifier execution timed out after 900.0 seconds | [result](../../controls/torch-pipeline-parallelism/oracle/attempt-0/trial/result.json) |
| torch-tensor-parallelism | 1/3 | test_outputs.py::test_column_parallel_linear; test_outputs.py::test_row_parallel_linear | [stdout](../../controls/torch-tensor-parallelism/oracle/attempt-0/trial/verifier/test-stdout.txt) / [CTRF](../../controls/torch-tensor-parallelism/oracle/attempt-0/trial/verifier/ctrf.json) |

`reshard-c4-data` 的 stdout 明確留下 `error: no /proc/self/exe available. Is /proc mounted?`。原 ENVIRONMENT_VALIDATION_AUDIT 的 explicit_proot_incompatibility_signals 清單沒有列入這個 stdout 線索；其空清單不代表排除了 PRoot 相容問題。正式 Gate 分類仍保留原 SCORING_GATE_FAIL，不事後改名或捏造可用分數。
