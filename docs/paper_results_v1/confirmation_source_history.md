# 八源 confirmation 使用历史更新

核对日期：2026-10-11（Asia/Shanghai）。本轮继续采用原定八源，不更换提示词、seed 或失败源。结论是：**在本次读到的 A 线记录中，八源只出现在预留计划内，没有已执行的生成或读取结果；其全局未见性仍未证实。** 报告继续使用 `USER_ADOPTED_LIMITED_LOCAL_NO_MATCH_UNPROVEN`，不称“独立未见集”，也不据此阻断确认运行。

## 固定名单

完整提示词和统一 negative prompt 取自 [real_eval.adopted.json](../../experiments/paper_results_v1/real_eval.adopted.json)；下表只缩写场景，不能用缩写替换配置。

| case | seed | 场景缩写 |
|---|---:|---|
| confirm_01 | 2026110201 | 白色瓷狐狸、蕨叶、深色橡木架 |
| confirm_02 | 2026110202 | 浅透明亚克力托盘中的紫色玻璃球 |
| confirm_03 | 2026110203 | 软木板上缓慢行走的绿色铁皮机器人 |
| confirm_04 | 2026110204 | 黑色毡面上缓慢转动的橙色纸鹤 |
| confirm_05 | 2026110205 | 透明长方形水槽中的银色玩具潜艇 |
| confirm_06 | 2026110206 | 浅色混凝土槽中滚动的紫色毛线球 |
| confirm_07 | 2026110207 | 哑光蓝桌面上的黄色木陀螺 |
| confirm_08 | 2026110208 | 白色转台上的黑色陶瓷猫 |

本轮不重新建立身份、摘要或不可篡改审计。[原来源核对](source_identity_audit.md)已记录当时有界文本/压缩包检索无字面匹配，同时明确未见性未证实；本次只更新之后的使用记录。

## 近期执行记录

| 已读取记录 | 八源实际状态 | 可支持的判断 |
|---|---|---|
| 20261009T123350716100Z-f5e1f880 原两 pilot `run_state.json` | 八个 generate 均 `PLANNED`；1,280 receiver slots、144 baseline slots、56 quality、552 artifacts 均 `PLANNED`；case records 只有 pilot_01/02 | 该运行没有执行八源；这些是旧九视图协议的预留数，不能当成本轮十五条件分母 |
| 20261010T031617441784Z-2536172d 时间攻击 `attack_run_state.json` | 1,920 receiver、240 baseline、56 quality 全 `NOT_EXECUTED_BY_NOTEBOOK` | 该时间攻击运行没有执行八源 |
| 20261010T063205774076Z-74da28d6 恢复 `attack_run_state.json` | 同上，所有八源仍 `NOT_EXECUTED_BY_NOTEBOOK` | 恢复只处理两 pilot |
| 20261010T190110828271Z-d3a0a045 payload follow-up `followup_state.json` | rows 仅 pilot_01/02，各45行；固定声明 confirmation_cases=8、confirmation_status=`NOT_EXECUTED` | 居中接收器选择和载荷改善来自两 pilot 开发批 |

证据目录分别为：

- [原两 pilot](../../../../diagnostics/a-line-20261009T123350716100Z-f5e1f880-audit/)
- [时间攻击失败运行](../../../../diagnostics/a-line-temporal-attack-20261010T031617441784Z-2536172d-audit/)
- [恢复运行](../../../../diagnostics/a-line-temporal-recovery-20261010T063205774076Z-74da28d6-audit/)
- [payload follow-up](../../../../diagnostics/a-line-receiver-payload-20261010T190110828271Z-audit/)

最后一个目录名省略了实际 run ID 的 `-d3a0a045` 后缀；运行身份以内部记录和审计正文为准。

## 本次有界检索及限制

以八个 seed 和八个有区分度的英文场景短语，对当前本地 `diagnostics/session/memory/other/alive/archive/release-candidates/worktrees` 的常用文本文件进行 `rg -l -i` 检索。返回命中来自当前 Paper-Results-V1 的定义文档/配置，以及上述 A 线保存的 effective configs 和原两 pilot state；没有返回另一条已执行八源的记录。当前工作树自身命中按定义记录处理，不作为使用证据。

本次检索有一个不可读路径：`diagnostics/step7-fixed-pixels-preparation-20261002/source_inputs/result.json`；没有把这个检索错误转成“全部扫描成功”。本轮未重新解压旧包、未遍历 Git 历史对象、未查询新的 Drive/Colab 数据，也未检查图片/视频/权重等二进制内容或同义改写。上述限制不影响使用既定名单，但不允许宣称全局未见、来源独立或无开发接触。

## 确认阶段使用规则

`CENTERED_U` 在查看本轮八源结果之前固定为首选；`RAW_U` 和 `CENTERED_D4` 保留为对照，D4 不因某个确认条件更好就替换主方法。结果按源及固定条件完整报告，失败、缺失、不支持和歧义均保留。八源数据不得反过来选接收器、调参数/阈值或替换失败源；若未来提出新方法，应另立开发问题并明确这八源已经被观察。
