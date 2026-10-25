# Qwen3.8-27B Fidelity：九模型整合結果

2026-10-16 已完成 PrismaAQUA 與真正的 ModelOpt AutoQuantize，兩者均為每模型 55,296-token Fidelity。

完整分析、容量、校準與搜尋成本、18 項多重比較、下一階段 Benchmark 準備與正確性證據見 [最新版 REPORT](../REPORT.md)。

[九模型 CSV](autoquant/analysis/comparison.csv)／[配對校正區間](autoquant/analysis/fast_pairwise.md)／[完成稽核](autoquant/completion_audit.json)。

| Model | Status | PPL | PPL delta | KL (BF16 → candidate) | Top-1 flip rate | Tokens | Reason |
|---|---|---:|---:|---:|---:|---:|---|
| Qwen-Qwen3.8-27B | ok | 3.672696566 | 0 | 0 | 0 | 55296 |  |
| Qwen-Qwen3.8-27B-FP8 | ok | 3.681105523 | 0.008408956846 | 0.004277229462 | 0.02412471065 | 55296 |  |
| eNeural-Qwen3.8-27B-fast | ok | 3.71469028 | 0.04199371396 | 0.0179378454 | 0.04797815394 | 55296 |  |
| nvidia-Qwen3.8-27B-NVFP4 | ok | 3.712697618 | 0.04000105268 | 0.01864519998 | 0.05264395255 | 55296 |  |
| RedHatAI-Qwen3.8-27B-NVFP4 | ok | 3.701359639 | 0.02866307372 | 0.01541646316 | 0.04439742477 | 55296 |  |
| unsloth-Qwen3.8-27B-NVFP4 | ok | 3.697641067 | 0.02494450128 | 0.01447625778 | 0.04736328125 | 55296 |  |
| RadixArk-Qwen3.8-27B-NVFP4-BF16-LMHead | ok | 3.716023725 | 0.04332715888 | 0.01895551107 | 0.04911747685 | 55296 |  |
| rdtand-Qwen3.8-27B-PrismaAQUA-5.5bit-vllm | ok | 3.705656988 | 0.03296042213 | 0.01408318748 | 0.04275173611 | 55296 |  |
| ModelOpt-Qwen3.8-27B-AutoQuantize-5.5bit | ok | 3.7216301 | 0.04893353383 | 0.01998750956 | 0.04841218171 | 55296 |  |

原八模型報告保留於 [歷史報告](REPORT_v2.1_eight_models.md)。第一輪及 expanded 原始 artifacts 均未覆寫。
