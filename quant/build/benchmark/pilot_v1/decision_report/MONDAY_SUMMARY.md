# 週一09:00研究報告摘要

報告日期：2026-10-19，台北時間。數據截至2026-10-17；完整能力Benchmark尚未開始。

**建議選四個研究核心：Qwen BF16、NaviQuant Fast、PrismaAQUA、ModelOpt AutoQuantize。** 第五個以Unsloth作實用部署對照；如需較高精度正控制，可換成Qwen FP8。

已完成九模型各55,296-token Fidelity；本輪九模型各200題代表性pilot（78 IFEval＋61 Standard＋61 Vision），共1,800主要回答＋501個4K探測。五張H100實際執行約9小時，合計38.61占用GPU-h。所有原始資料／條件配對通過，2,301個回答經官方評分程式重評一致；17項針對性CPU測試通過。

Fidelity支持Prisma比Fast更接近BF16（KL及Flip，18項校正）；Fast的KL優於本次Auto搜尋結果。兩組PPL方向仍不確定。三者權重約23.5–23.8GB、保留部件匹配；能力排名尚未確立，Auto校準搜尋較小，不能代表所有可能配置。

**目前最影響能力與成本判讀的是截斷。** MMMU 1K有451/1,098＝41.1%截斷；4K仍187/1,098＝17.0%。BF16 Vision從25/61變41/61，Auto從29變32，表面相對排序反轉。IFEval702回答沒有截斷。暫不以小樣本或1K能力分數決定淘汰模型。

| 工作選項（MTP Off） | 五H100一般 wall | 樂觀–保守 wall | 一般占用GPU-h |
|---|---:|---:|---:|
| 九模型完整IFEval | 5.1h | 4.0–8.0h | 22.65 |
| 九模型完整三項，MMMU1K | 81.0h | 77.0–106.4h | 364.10 |
| 九模型完整三項，MMMU4K | 170.0h | 153.5–232.5h | 763.97 |
| 核心四模型三項，MMMU4K | 86.5h | 79.4–117.3h | 322.15 |
| 核心＋Unsloth三項，MMMU4K | 86.5h | 79.4–117.3h | 402.26 |
| 核心＋FP8三項，MMMU4K | 96.5h | 86.3–133.1h | 418.65 |

單GPU串行wall hours等於占用GPU-h。多GPU採一模型一GPU逐題，工作不可任意切分；四模型加第五個Unsloth不延長目前critical path，但增加算量。區間為分層bootstrap加保守25%餘量的規劃情境，不是工期保證；4K也仍截斷，不能直接外推8K。

**週一可以先報告已完成的Fidelity、截斷敏感度與上述預算。** 若後續授權補數據，建議先跑九模型完整IFEval。以週日00:00開始為例，一般週日05:05、保守07:57完成；週一上午前保留充分整理時間。完整MMMU可在報告後補，先確認正式cap／解碼策略。核心＋Unsloth的IFEval＋Standard1K雖約21.1h（保守28.3h）可塞進窗口，卻仍受截斷限制，不優先建議用於最終排名。

本輪全部MTP Off。九模型MTP／DSpark啟動通過，但沒有代表性MTP On時間／品質估計；舊少量smoke不能作加速倍數。先定模型與Off基準，再另作On配對，避免一開始把36,009次擴成72,018次。

[完整整合分析](DECISION_REPORT.md)／[模型容量、Fidelity與pilot CSV](integrated_comparison.csv)／[子集預算與完成時程CSV](subset_budgets.csv)／[官方重評](../official_replay/audit.json)。完整Benchmark未啟動。另發現舊背景控制器誤觸8K，已停止並隔離17個不完整回答；額外約2.24 GPU-h未納入正式pilot成本與結論，詳見整合報告事件紀錄。
