| model | label | mmmu_cap | task | questions | correct | accuracy | population_weighted_accuracy | truncated | fallback | correct_with_fallback | mean_output_tokens | inference_seconds | loose_accuracy | strict_instruction_accuracy | loose_instruction_accuracy |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Qwen-Qwen3.8-27B | Qwen BF16 | 1024 | ifeval | 78 | 58 | 0.74358974 | 0.74907579 | 0 | 0 | 0 | 357.33333 | 1321.0753 | 0.82051282 | 0.84782609 | 0.89130435 |
| Qwen-Qwen3.8-27B | Qwen BF16 | 1024 | mmmu_standard | 61 | 26 | 0.42622951 | 0.41763006 | 22 | 2 | 1 | 615.09836 | 1829.2694 |  |  |  |
| Qwen-Qwen3.8-27B | Qwen BF16 | 1024 | mmmu_vision | 61 | 25 | 0.40983607 | 0.39855491 | 27 | 2 | 0 | 709.42623 | 2118.6474 |  |  |  |
| Qwen-Qwen3.8-27B | Qwen BF16 | 4096 | ifeval | 78 | 58 | 0.74358974 | 0.74907579 | 0 | 0 | 0 | 357.33333 | 1321.0753 | 0.82051282 | 0.84782609 | 0.89130435 |
| Qwen-Qwen3.8-27B | Qwen BF16 | 4096 | mmmu_standard | 61 | 33 | 0.54098361 | 0.55202312 | 10 | 0 | 0 | 1256.0984 | 3723.1797 |  |  |  |
| Qwen-Qwen3.8-27B | Qwen BF16 | 4096 | mmmu_vision | 61 | 41 | 0.67213115 | 0.66820809 | 12 | 0 | 0 | 1596.541 | 4730.982 |  |  |  |
| Qwen-Qwen3.8-27B-FP8 | Qwen FP8 | 1024 | ifeval | 78 | 58 | 0.74358974 | 0.77047135 | 0 | 0 | 0 | 345.9359 | 1899.1902 | 0.80769231 | 0.84782609 | 0.88405797 |
| Qwen-Qwen3.8-27B-FP8 | Qwen FP8 | 1024 | mmmu_standard | 61 | 33 | 0.54098361 | 0.54942197 | 17 | 1 | 0 | 534.21311 | 2336.0651 |  |  |  |
| Qwen-Qwen3.8-27B-FP8 | Qwen FP8 | 1024 | mmmu_vision | 61 | 27 | 0.44262295 | 0.43381503 | 26 | 3 | 1 | 700.34426 | 3097.6413 |  |  |  |
| Qwen-Qwen3.8-27B-FP8 | Qwen FP8 | 4096 | ifeval | 78 | 58 | 0.74358974 | 0.77047135 | 0 | 0 | 0 | 345.9359 | 1899.1902 | 0.80769231 | 0.84782609 | 0.88405797 |
| Qwen-Qwen3.8-27B-FP8 | Qwen FP8 | 4096 | mmmu_standard | 61 | 33 | 0.54098361 | 0.55144509 | 9 | 0 | 0 | 1087.8525 | 4743.9934 |  |  |  |
| Qwen-Qwen3.8-27B-FP8 | Qwen FP8 | 4096 | mmmu_vision | 61 | 37 | 0.60655738 | 0.60433526 | 13 | 0 | 0 | 1623.9344 | 7135.3297 |  |  |  |
| eNeural-Qwen3.8-27B-fast | NaviQuant Fast | 1024 | ifeval | 78 | 55 | 0.70512821 | 0.7435305 | 0 | 0 | 0 | 357.01282 | 1661.744 | 0.80769231 | 0.8115942 | 0.87681159 |
| eNeural-Qwen3.8-27B-fast | NaviQuant Fast | 1024 | mmmu_standard | 61 | 26 | 0.42622951 | 0.43150289 | 23 | 0 | 0 | 632.7377 | 2354.9851 |  |  |  |
| eNeural-Qwen3.8-27B-fast | NaviQuant Fast | 1024 | mmmu_vision | 61 | 25 | 0.40983607 | 0.39682081 | 28 | 4 | 0 | 737.16393 | 2778.8021 |  |  |  |
| eNeural-Qwen3.8-27B-fast | NaviQuant Fast | 4096 | ifeval | 78 | 55 | 0.70512821 | 0.7435305 | 0 | 0 | 0 | 357.01282 | 1661.744 | 0.80769231 | 0.8115942 | 0.87681159 |
| eNeural-Qwen3.8-27B-fast | NaviQuant Fast | 4096 | mmmu_standard | 61 | 32 | 0.52459016 | 0.53468208 | 8 | 0 | 0 | 1339.7869 | 4976.3475 |  |  |  |
| eNeural-Qwen3.8-27B-fast | NaviQuant Fast | 4096 | mmmu_vision | 61 | 37 | 0.60655738 | 0.59768786 | 10 | 0 | 0 | 1487.5082 | 5562.4052 |  |  |  |
| nvidia-Qwen3.8-27B-NVFP4 | NVIDIA NVFP4 | 1024 | ifeval | 78 | 61 | 0.78205128 | 0.7974122 | 0 | 0 | 0 | 335.66667 | 1620.3385 | 0.83333333 | 0.86956522 | 0.89855072 |
| nvidia-Qwen3.8-27B-NVFP4 | NVIDIA NVFP4 | 1024 | mmmu_standard | 61 | 31 | 0.50819672 | 0.5 | 20 | 1 | 0 | 482.19672 | 1825.0343 |  |  |  |
| nvidia-Qwen3.8-27B-NVFP4 | NVIDIA NVFP4 | 1024 | mmmu_vision | 61 | 21 | 0.3442623 | 0.33092486 | 30 | 4 | 0 | 738.65574 | 2819.9761 |  |  |  |
| nvidia-Qwen3.8-27B-NVFP4 | NVIDIA NVFP4 | 4096 | ifeval | 78 | 61 | 0.78205128 | 0.7974122 | 0 | 0 | 0 | 335.66667 | 1620.3385 | 0.83333333 | 0.86956522 | 0.89855072 |
| nvidia-Qwen3.8-27B-NVFP4 | NVIDIA NVFP4 | 4096 | mmmu_standard | 61 | 35 | 0.57377049 | 0.58699422 | 7 | 0 | 0 | 1099.2131 | 4143.5767 |  |  |  |
| nvidia-Qwen3.8-27B-NVFP4 | NVIDIA NVFP4 | 4096 | mmmu_vision | 61 | 36 | 0.59016393 | 0.58468208 | 12 | 0 | 0 | 1639.8197 | 6210.0911 |  |  |  |
| RedHatAI-Qwen3.8-27B-NVFP4 | RedHatAI NVFP4 | 1024 | ifeval | 78 | 56 | 0.71794872 | 0.83096118 | 0 | 0 | 0 | 363.83333 | 1718.89 | 0.78205128 | 0.83333333 | 0.86956522 |
| RedHatAI-Qwen3.8-27B-NVFP4 | RedHatAI NVFP4 | 1024 | mmmu_standard | 61 | 32 | 0.52459016 | 0.53410405 | 21 | 3 | 0 | 660.44262 | 2501.3026 |  |  |  |
| RedHatAI-Qwen3.8-27B-NVFP4 | RedHatAI NVFP4 | 1024 | mmmu_vision | 61 | 26 | 0.42622951 | 0.4300578 | 28 | 5 | 0 | 702.78689 | 2695.1657 |  |  |  |
| RedHatAI-Qwen3.8-27B-NVFP4 | RedHatAI NVFP4 | 4096 | ifeval | 78 | 56 | 0.71794872 | 0.83096118 | 0 | 0 | 0 | 363.83333 | 1718.89 | 0.78205128 | 0.83333333 | 0.86956522 |
| RedHatAI-Qwen3.8-27B-NVFP4 | RedHatAI NVFP4 | 4096 | mmmu_standard | 61 | 36 | 0.59016393 | 0.59942197 | 8 | 1 | 0 | 1361.0328 | 5142.4414 |  |  |  |
| RedHatAI-Qwen3.8-27B-NVFP4 | RedHatAI NVFP4 | 4096 | mmmu_vision | 61 | 37 | 0.60655738 | 0.6 | 15 | 0 | 0 | 1646.0984 | 6266.3807 |  |  |  |
| unsloth-Qwen3.8-27B-NVFP4 | Unsloth NVFP4 | 1024 | ifeval | 78 | 62 | 0.79487179 | 0.84205176 | 0 | 0 | 0 | 372.85897 | 1759.9651 | 0.87179487 | 0.87681159 | 0.92028986 |
| unsloth-Qwen3.8-27B-NVFP4 | Unsloth NVFP4 | 1024 | mmmu_standard | 61 | 29 | 0.47540984 | 0.48265896 | 22 | 3 | 0 | 586.37705 | 2147.8981 |  |  |  |
| unsloth-Qwen3.8-27B-NVFP4 | Unsloth NVFP4 | 1024 | mmmu_vision | 61 | 26 | 0.42622951 | 0.41531792 | 25 | 4 | 0 | 688.06557 | 2545.8276 |  |  |  |
| unsloth-Qwen3.8-27B-NVFP4 | Unsloth NVFP4 | 4096 | ifeval | 78 | 62 | 0.79487179 | 0.84205176 | 0 | 0 | 0 | 372.85897 | 1759.9651 | 0.87179487 | 0.87681159 | 0.92028986 |
| unsloth-Qwen3.8-27B-NVFP4 | Unsloth NVFP4 | 4096 | mmmu_standard | 61 | 37 | 0.60655738 | 0.61791908 | 9 | 0 | 0 | 1219.1311 | 4458.1832 |  |  |  |
| unsloth-Qwen3.8-27B-NVFP4 | Unsloth NVFP4 | 4096 | mmmu_vision | 61 | 34 | 0.55737705 | 0.55086705 | 10 | 0 | 0 | 1464.3443 | 5372.6625 |  |  |  |
| RadixArk-Qwen3.8-27B-NVFP4-BF16-LMHead | RadixArk BF16-LMHead | 1024 | ifeval | 78 | 57 | 0.73076923 | 0.79001848 | 0 | 0 | 0 | 355.14103 | 1644.0594 | 0.80769231 | 0.82608696 | 0.87681159 |
| RadixArk-Qwen3.8-27B-NVFP4-BF16-LMHead | RadixArk BF16-LMHead | 1024 | mmmu_standard | 61 | 26 | 0.42622951 | 0.42947977 | 22 | 3 | 1 | 609.96721 | 2173.334 |  |  |  |
| RadixArk-Qwen3.8-27B-NVFP4-BF16-LMHead | RadixArk BF16-LMHead | 1024 | mmmu_vision | 61 | 29 | 0.47540984 | 0.46445087 | 31 | 3 | 1 | 743.36066 | 2685.1433 |  |  |  |
| RadixArk-Qwen3.8-27B-NVFP4-BF16-LMHead | RadixArk BF16-LMHead | 4096 | ifeval | 78 | 57 | 0.73076923 | 0.79001848 | 0 | 0 | 0 | 355.14103 | 1644.0594 | 0.80769231 | 0.82608696 | 0.87681159 |
| RadixArk-Qwen3.8-27B-NVFP4-BF16-LMHead | RadixArk BF16-LMHead | 4096 | mmmu_standard | 61 | 36 | 0.59016393 | 0.60202312 | 6 | 0 | 0 | 1166.377 | 4147.3751 |  |  |  |
| RadixArk-Qwen3.8-27B-NVFP4-BF16-LMHead | RadixArk BF16-LMHead | 4096 | mmmu_vision | 61 | 33 | 0.54098361 | 0.53236994 | 16 | 0 | 0 | 1848.9836 | 6619.9278 |  |  |  |
| rdtand-Qwen3.8-27B-PrismaAQUA-5.5bit-vllm | PrismaAQUA | 1024 | ifeval | 78 | 61 | 0.78205128 | 0.7974122 | 0 | 0 | 0 | 384.12821 | 1714.068 | 0.85897436 | 0.86231884 | 0.91304348 |
| rdtand-Qwen3.8-27B-PrismaAQUA-5.5bit-vllm | PrismaAQUA | 1024 | mmmu_standard | 61 | 29 | 0.47540984 | 0.48236994 | 21 | 3 | 1 | 561.91803 | 1952.157 |  |  |  |
| rdtand-Qwen3.8-27B-PrismaAQUA-5.5bit-vllm | PrismaAQUA | 1024 | mmmu_vision | 61 | 23 | 0.37704918 | 0.36763006 | 30 | 6 | 0 | 723.03279 | 2543.3881 |  |  |  |
| rdtand-Qwen3.8-27B-PrismaAQUA-5.5bit-vllm | PrismaAQUA | 4096 | ifeval | 78 | 61 | 0.78205128 | 0.7974122 | 0 | 0 | 0 | 384.12821 | 1714.068 | 0.85897436 | 0.86231884 | 0.91304348 |
| rdtand-Qwen3.8-27B-PrismaAQUA-5.5bit-vllm | PrismaAQUA | 4096 | mmmu_standard | 61 | 34 | 0.55737705 | 0.56878613 | 7 | 0 | 0 | 1177.082 | 4076.2329 |  |  |  |
| rdtand-Qwen3.8-27B-PrismaAQUA-5.5bit-vllm | PrismaAQUA | 4096 | mmmu_vision | 61 | 36 | 0.59016393 | 0.58410405 | 12 | 0 | 0 | 1669.5738 | 5812.7084 |  |  |  |
| ModelOpt-Qwen3.8-27B-AutoQuantize-5.5bit | ModelOpt AutoQuantize | 1024 | ifeval | 78 | 60 | 0.76923077 | 0.79556377 | 0 | 0 | 0 | 379.07692 | 1778.9243 | 0.83333333 | 0.85507246 | 0.89855072 |
| ModelOpt-Qwen3.8-27B-AutoQuantize-5.5bit | ModelOpt AutoQuantize | 1024 | mmmu_standard | 61 | 28 | 0.45901639 | 0.46473988 | 25 | 2 | 1 | 564.91803 | 2033.8838 |  |  |  |
| ModelOpt-Qwen3.8-27B-AutoQuantize-5.5bit | ModelOpt AutoQuantize | 1024 | mmmu_vision | 61 | 29 | 0.47540984 | 0.46763006 | 33 | 7 | 1 | 761.7541 | 2773.815 |  |  |  |
| ModelOpt-Qwen3.8-27B-AutoQuantize-5.5bit | ModelOpt AutoQuantize | 4096 | ifeval | 78 | 60 | 0.76923077 | 0.79556377 | 0 | 0 | 0 | 379.07692 | 1778.9243 | 0.83333333 | 0.85507246 | 0.89855072 |
| ModelOpt-Qwen3.8-27B-AutoQuantize-5.5bit | ModelOpt AutoQuantize | 4096 | mmmu_standard | 61 | 38 | 0.62295082 | 0.63815029 | 9 | 0 | 0 | 1225.4262 | 4404.4733 |  |  |  |
| ModelOpt-Qwen3.8-27B-AutoQuantize-5.5bit | ModelOpt AutoQuantize | 4096 | mmmu_vision | 61 | 32 | 0.52459016 | 0.51676301 | 14 | 0 | 0 | 1688.3279 | 6089.57 |  |  |  |
