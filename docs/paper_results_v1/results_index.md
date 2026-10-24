# 保存结果与研究索引

本轮只精选完整保存 JSON；没有重跑模型、VAE、FFT、票归约、路径选择或 codec。公共方法沿用[分段载荷说明](../segmented_payload_v1.md)。[机器可读索引](../evidence/paper_results_v1/index.json)记录原路径、计划、summary 和主审定位；来源信息可缺失或变化，不形成准入门禁。

| 保存账 | 固定行 / 段 / 位 | 状态与结构 | 入口 |
|---|---|---|---|
| 原四源 | 36 / 144 / 1152 | 36 READ；896存在、256结构缺失 | [计划](../evidence/paper_results_v1/original36/protocol.json)、[完整行](../evidence/paper_results_v1/original36/row_results.json)、[原summary](../evidence/paper_results_v1/original36/summary.json)、[原主审](../evidence/paper_results_v1/original36/root_acceptance.json) |
| P2 四新源 | 196 / 784 / 6272 | 188 READ、4 UNRESOLVED、4 UNSUPPORTED；3968存在、2304缺失 | [计划](../evidence/paper_results_v1/p2_196/protocol.json)、[完整行](../evidence/paper_results_v1/p2_196/row_results.json)、[原summary](../evidence/paper_results_v1/p2_196/summary.json)、[原主审](../evidence/paper_results_v1/p2_196/root_acceptance.json) |

P2 每源八个相关攻击 view；98 次物理读取、188 次逻辑读取、90 个 logical alias，四来源共用固定消息/key。不能以 196 行、784 段或 alias 数冒充独立来源/消息。候选路径并非全部正确，原主审保留两条非精确路径和相对 CENTERED 的一条退步。固定计划来自原审定 roster 的显式来源×view×arm；报告运行不从成功行构造计划。

后验研究单独索引，不纳入原196账：

- [64行组成分析原主审](../evidence/paper_results_v1/separate_research/posthoc64_acceptance.json)：2048计划位、1280存在、768结构缺失；57逻辑alias、4 UNRESOLVED、3 NO_SAVED_READ_MATCH；1160已读、120未读，完整存在BER为null。
- [单行16位补读原主审](../evidence/paper_results_v1/separate_research/single16_overlay_acceptance.json)：16新增存在位读对，16结构缺失仍保留；RAW后验组合达到640/640，但不是新独立确认，不回写原196行。单行summary/row/overlay的原本地定位在机器索引中。

原 P2 是其原预定协议确认；对后来在这些来源上开发/选择的方案，它是已见开发证据。原主审里的逐票审计、实际调用统计和执行隔离结论仍是历史证据，本轮只读复制不新增物理实验。

当前 `NATIVE_PRE` 本身含分段payload。candidate PRE 相对它是新增同步残差；POST 相对它还含 codec。总writer相对同源无水印OFF质量未测，缺项不填0。旧重复payload来源的质量不能替代P2；codec变化只能用同内容匹配PRE/POST记录，不能以dB相减造因果分解。LPIPS/flow、匹配峰值显存/规范wall time以及当前P2公平baseline均保留未知。

历史 baseline 同次证据定位：`/home/richar/projects/Video-WM/diagnostics/a-line-temporal-attack-20261010T031617441784Z-2536172d-audit/` 的 `audit_summary.txt`、`handoff_summary.json`、`baseline_setup_receipts.json`。本轮范围审计为 `diagnostics/paper-outer-main-audit-20261024/REPORT.md` 与 `BASELINES_AUDIT.md`；[baseline解释](baseline_status.md)区分已有embed/codec和缺失有效评测。

A/B算法继续保留各自研究分支，见[分支用途](../branch_guide.md)。本层不新增B数值成绩表，不整包引入旧 workflow、runner、攻击、恢复、Notebook 或论文稿。
