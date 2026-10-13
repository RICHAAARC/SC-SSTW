# 固定 FINAL 六位导数诊断：本地实施交付

2026-10-13。按已采用六位提案实施；A1 本地 CPU/fake/static 验证，真实 GPU/Colab 由用户运行。主会话负责同版本 A2/A3 审查与发布。此交付没有运行真实模型、Drive/Colab、推送或合并 main。

## 直接入口

- Notebook：`notebooks/local_joint_six_bit_final_v1_colab.ipynb`。
- Builder：`scripts/build_local_joint_six_bit_final_v1_notebook.py`。
- 普通源码包：`notebooks/local_joint_six_bit_final_v1_portable_source.zip`。
- 配置：`experiments/wan_state_clock/configs/local_joint_six_bit_final_v1.json`。
- 独立 CLI：`python -m experiments.wan_state_clock.local_joint_six_bit_final_v1_run --config <config.json> --output <independent-output>`；不依赖 notebook、builder 或 `.git`。

Notebook 第一格仅两行 Drive mount，没有 force_remount。Run all 固定执行本项诊断或恢复其未完成单元，无模式选择。SOURCE_REF/输入路径为可编辑定位信息，无 B64、摘要、manifest、源码身份/精确版本/Git 状态门禁或 GPU 型号白名单。

## 固定方法与预算

实际输入为 Continue `resumable-task-v1-2x010/run/step_2_unit.pt["terminal"]`，即 CONT_FINAL；不读取旧 gradient checkpoint 的 MID `input_terminal`。原 BASE 为同任务 `run/inputs/terminal_latent.pt`，已保存 FINAL 的55量来自 `run/result.json` 中 `views.CONT_FINAL`。

固定 flat index `[4,12,18,20,27,30]`。保持原 mask/key/message/rho/半径1/读出。每位独立新图，直接对正 margin `read["margins"].reshape(-1)[i]` 求导、mask一次，保存原始未归一化梯度。复用已成功的 Wan FP32 causal checkpoint/spool；没有共享图多次 backward。

完整无中断名义预算 **1 load + 6 gradient decode + 6 VJP，276 forward /276 nominal recompute chunks**。普通 decode、encode、DiT、native、scheduler、codec 为0。每次保留32 margins、22 correlations、G，合计55量、88窗口/1408 chips，payload分母 `[24,24,20,20]`；六行固定总分母528/8448/330，失败和缺失行保留。这是同一输入上的重复诊断，不是六个独立样本。

不生成终点更新、候选实验、扫描、rollback、第三步、cap增加、COMMON refresh 或 M1/M2。`diagnostic_directions.pt` 只保存 CPU 数学诊断方向，不应用到 terminal。

## 数学报告

`diagnostic.json` 保留原始 Gram/cosine/norm、聚合方向逐位斜率、径向/切向分解，以及原始和纯切向的未归一化 simplex 最小范数解。旧五维求解器只作维数推广至最多六维；五维算法保持。每个解报告 primal、dual、complementarity、KKT residual、duality gap 和 FP64 误差范围。

只有六个斜率扣除报告的数值误差后下界均严格正，才标记 `NUMERICALLY_SUPPORTED_STRICT_COMMON_DIRECTION`；否则 `NUMERICALLY_UNRESOLVED`。非零 norm、负 cosine 或近零解都不构成不可行结论。误差覆盖存储梯度上的 FP64 算术、simplex 数值与切向投影，**不声称估计了真实 FP32 VAE/autodiff 导数误差**。

`-0.025 Σh_i` 仅在六次重算的 scalar-loss payload_active 恰为六位、六位仍严格负、state_active=false、系数适用时标为完整当前 loss 梯度。fragment-min/wrong-offset active 分支不是这个条件。55量与保存 FINAL 的差值始终作为诊断，不要求相等。

切向采用 `u=(FINAL-BASE)/r`、`t_i=h_i-(h_i·u)u`，报告投影误差与径向残差。到原 BASE 单位球边界的直线距离只表示几何，不预测过零步数或收益。26正位、状态和质量的导数未测量，不能据本报告声称保护它们、有限预算全位可达或整条路线不可行。

## 恢复与失败记录

新固定 Drive 目录：`MyDrive/Video-WM/Local-Joint-State-Payload-V1-Six-Bit-Final/resumable-six-bit-final-v1/`。每次运行有独立 notebook_attempts 日志；实际 runner 在其 `run/` 下。

- 模型加载前，以单个原子备份保存实际 FINAL、原 BASE、保存 FINAL 的55量及输入定位信息。
- 每位完成立即原子保存 `units/bit_XX.pt`，包含实际求导输入、bit index、raw gradient、q、55量、梯度范数和调用定位。保存 q/readout 后再完成 VJP 单元提交。
- 每60秒保存**最后已提交的可恢复状态与进度**，保留前一个可读快照。它不保存进行中的临时计算图。
- VJP中途到期/被打断时，只重新计算该未完成位及其他尚未完成位。临时 activation spool 在 finally 关闭，不能作为跨会话 autograd checkpoint。
- 六位全部提交后，即使报告未完成或源输入路径不可用，也只执行 CPU 报告；VAE load/decode/VJP 均为0。
- 实际 input/bit 配对、张量格式/有限值、固定支撑/半径是方法检查。错目录、无可恢复状态的非空目录或损坏的已提交单元会报告具体数据故障，不静默重算全部实验。
- 独立 call receipts、failure receipts、attempt_history 保留累计物理调用、失败、未知完成与本次计数；旧快照恢复不会抹去最后的提交失败。中断后的物理预算可能大于名义预算。

## 修改范围与验证

新增方法核、runtime、配置、CLI、builder、实际 notebook/ZIP 和两组针对性测试。共享 `simplex_minimum` 允许1至6维；共享 Journal 增加可选固定 step 名单，旧默认 step_1/step_2 不变。四个已有普通 ZIP 同步更新其包含的共享源码；没有改变旧 notebook 的实验输入或控制流。

必要 CPU/fake/static 检查覆盖真实55量索引/符号、条件聚合公式、原始 Gram/余弦/共同方向/纯切向/零/反向/近零、CONT_FINAL而非MID、串行图和spool释放、单元保存失败/旧快照恢复/VJP中途打断/完成后零模型调用、实际 notebook 顺序及重入、普通ZIP无.git独立CLI。共享组件仅补相应五维 simplex 和 Journal 快照回归，不运行旧全套实验测试。

20个不同测试目标已分批通过。故障注入曾定位并修复两项恢复问题：较早快照可能遗漏最后失败记录；新图初始化回调可能误覆盖上一轮同位的回放计数。最终对这两条实际恢复路径和新ZIP一致性重验3/3通过；累计物理调用与各次回放账本均保留，没有为得到绿灯重跑真实实验。

历史单VJP spool约64.53GiB、六位每方向累计I/O约387.2GiB、历史GPU allocated19.01GiB/reserved21.72GiB、六位估时约44分钟仅供参考；真实本项资源与方法信号尚未测量。静态或TinyWan成功不是真实GPU保证或科学PASS。

20个不同测试目标已分批通过。故障注入曾定位并修复两项恢复问题：较早快照可能遗漏最后失败记录；新图初始化回调可能误覆盖上一轮同位的回放计数。最终对这两条实际恢复路径和新ZIP一致性重验3/3通过；累计物理调用与各次回放账本保留。完整验证记录在工作树外 diagnostics/b-line-six-bit-final-implementation-20261013/validation.md。
