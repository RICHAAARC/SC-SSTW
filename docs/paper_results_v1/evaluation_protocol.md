# 保存结果只读外层

本层只读取审定的保存行与 summary，输出 JSON / CSV / Markdown。方法定义直接沿用[公共分段载荷](../segmented_payload_v1.md)和[机制证据索引](../mechanism_evidence_index.md)，不引入 A/B 同步算法。净容量为四段各八位；不存在用论文外层调用 writer、selector、receiver、模型、FFT 或 codec 的路径。

从普通源码目录执行，无需 `.git`、Torch、NumPy、FFmpeg 或模型：

```bash
python -m experiments.paper_results_v1.cli \
  --protocol docs/evidence/paper_results_v1/original36/protocol.json \
  --output /tmp/sc-sstw-original36-report
python -m experiments.paper_results_v1.cli \
  --protocol docs/evidence/paper_results_v1/p2_196/protocol.json \
  --output /tmp/sc-sstw-p2-report
```

输出目录中已有同名报告时选新目录，以免覆盖保存结果。空计划可运行：`experiments/paper_results_v1/manifest.template.json`。该文件只是普通分母配置，不是完整性证明。来源、摘要、版本和 `expected_top_level` 全部仅记录；缺失或差异不决定 `CONFLICT`。

## 计划与保存格式

协议 `schema=segmented-saved-report-v1`，`payload_layout={segments:4,bits_per_segment:8,net_bits:32}`。非空 `slots` 必须是已经采用的 `plan_status=ADOPTED`；新队列不能从成功结果、历史 baseline 配置或未采用提案自动导入。每个 slot 显式指定 `slot_id / saved_slot_id / binding_id / source_id / arm / view / information_condition / evidence_role / message_id`。`result_bindings` 指定相对协议文件的 `rows_path`，可附 `summary_path`。

保存行使用既有 `slot_id/case_id/receiver/view/oracle/planned_bits/planned_segments/status/reason`，以及 `eligible_planned_bits/structural_missing_bits/scored_present_bits/missing_present_bits/wrong_present_bits/false_attribution_bits`。后六个字段可以为 null 或缺项，表示未知；只在有数值时检查其合法性，不从状态猜 0。完整保存原字段置于 `raw_saved_rows`，原 summary 原样置于 `inputs.saved_summary`。

固定计划左连接：缺行保留 MISSING；文件不可读保留具体错误；重复保存身份保留全部候选并标 CONFLICT，不按答案择优。无计划行保留在 `unplanned_rows`，不增加分母。FAILED、UNRESOLVED、UNSUPPORTED 等原状态与原因保留。格式/身份/数值错误另列 `report_state=INVALID_SAVED_FORMAT` 与具体 `report_reason`，不当科学失败。非法协议不能建立可靠分母，直接报告具体格式错误。

分母分开记录：planned = structurally present + structural missing；present = read + present unread；error 只属于已读存在位。observed BER = error/read（read 为零或未知则 null）；complete present BER 仅在全部存在位已读时给出。任何行未知时总计为 null，另列已知小计和未知行数。结构缺失不是恢复，未读不是正确，零 observed BER 不能掩盖覆盖损失。路径误差保存原值，resolved 与固定计划路径分母分开；缺路径不补零。

部分字段为 null 时仍检查已知必要关系：例如 read 或 unread 不能超过已知 present；错误位需要已知可成立的支持范围。检查只判断是否仍可能满足固定等式，不把推导值写回未知字段。`physical_read_unit/read_alias_of` 只能为字符串或 null，非法类型保留原行并列格式问题，不中断总账或制造 alias 数。

## 信息与证据分栏

`BLIND` 输出只放在 `blind_output`；oracle 放 `oracle_output`；调用方映射放 `caller_map_output`，明确 `conditional_on_given_mapping=true / blind_sync_performed=false`。真值、缺段、路径误差与后评分放 `posthoc`。报告保留这些栏位只证明整理阶段隔离；实际执行时序引用原主审，不由报告重新证明。`evidence_role` 使原确认和后验开发分开；已见 P2 数据对新方案不是新独立确认。

旧 `false_attribution_bits` 仅是**结构缺失段仍有输出的位数**，在汇总中命名 `missing_segment_output_bits`。一般来源/cell 错归属固定为 UNKNOWN。`matches_other_expected_byte` 只保留特定字节巧合，不升级为一般归因证据。载荷正确不能消除路径错误。

`compare_saved_support(left,right)` 是显式的纯表 API，仅接受保存的完整 32 个 `posthoc.bit_support` 记录，每条包括 `bit`（0..31）、`present`、`read`、`correct`（已读存在位为 bool，其余 null）。来源、view、消息、信息条件、证据角色与结构 mask 必须可比。它报告共同已读位 BER、正确覆盖保留/损失/新增；没有逐位记录则 UNKNOWN，不从总量猜交集、不从答案重新评分。原36/P2196保存行没有这份新格式，不为其生成配对成绩；已有主审配对结论只以原审计记录索引。

逐位支持在接纳和配对前还要与同一行已知计数、decoded 非 null 位置及 true_missing_segments 一致；缺项本身允许，已知矛盾保留原证据并返回 INVALID_SAVED_FORMAT，不输出有效配对成绩。不用消息答案裁定或覆盖冲突的保存记录。

逻辑 READ、显式物理 read unit、alias、来源标签和消息标签分别计数。物理字段缺项为 unknown；来源标签数不证明独立来源数，alias 不新增物理试验。原 summary 的 physical attempts/calls 和主审物理统计保留原身份，不由行数替代。

## 分层与可拆卸性

依赖为 `cli -> report -> Python 标准库`，`native_adapters -> Python 标准库 + 调用方主动注入的 backend`。两者不互相导入。`main/tube_state`、`runtime` 和既有公共 CLI 不导入本层。删除整个 `experiments/paper_results_v1` 不影响公共核心/runtime/CLI；单独删除 `native_adapters.py` 不影响保存报告。无自动插件发现、baseline 默认选择、算法 fallback 或绘图补算。本轮没有增加绘图，因为保存表已足以呈现授权信息。

工程核验与旧证据整理不等于新实验、科学 PASS、完整公平 baseline 或论文质量 PASS。
