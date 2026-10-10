# M05 固定接收端四臂：已存分数 CPU 对照结果

**同批保存观测中，接收端列中位数去偏显著减少了时间路径错误；D4 单独也改善，但联合相对去偏单独只补回两条混合条件中的部分错误。** 这是事先固定构造的开发对照，不是八例 confirmation、独立泛化、盲检测成功或载荷结论。

定义见 [固定方法](m05_receiver_controls_v1.md)。Root 和独立 A5 先审阅通过后才实施和执行；没有根据新结果调整列中位数、最大驻留 4、四臂或 30 条件。使用两 pilot 各 15 条件的原 K0 signed_projection/rho；没有读取 RECON 或真值参与选择，没有生成、视频读取/重解码、VAE、GPU、codec、载荷或新模型调用。

## 完整分母与路径结果

120/120 case-arm 行均为 ESTIMATED 且事后 EVALUATED；每臂 30/30 条件、5,402/5,402 收到帧。此次真实矩阵没有平局、拒绝、缺失或失败；相关状态与固定槽位仍由实现保留。原 RAW_U 的 30 条 best path 和 best score 与原保存记录逐项一致，分数差均为 0。全部 30 条件的真值支持集合事后均存在 D4 可行路径；这不构成选择器输入。

| 臂 | 全路径均在 support 内 /30 | support 内帧 /5402 | 支持域误差：等条件均值（帧） | 支持域误差：按收到帧加权均值（帧） | weighted error：等条件均值（帧） | 最大驻留 |
|---|---:|---:|---:|---:|---:|---:|
| RAW_U | 0 | 2031 | 22.475748320 | 22.915771936 | 22.682890538 | 220 |
| CENTERED_U | 26 | 5388 | 0.002946593 | 0.002961866 | 0.168198081 | 7 |
| RAW_D4 | 0 | 4814 | 0.194357580 | 0.184931507 | 0.370904341 | 4 |
| CENTERED_D4 | 27 | 5396 | 0.001104972 | 0.001110700 | 0.164072850 | 4 |

支持域误差是每个收到帧到任一正权重真实贡献源的最小距离。混合帧的多个源都合法，因此“全路径在 support 内”不等于唯一真实路径恢复；weighted error 同时保留，不能用前者替代后者。等条件平均与按收到帧加权平均口径分开；原审计的 22.47575 对应前者。

RAW_U 起点 support error 为 0 的条件有 28/30，另外三臂均为 30/30。这再次说明起点定位不能替代完整路径恢复。不同臂的原始和去偏 objective 都保存在表中，但不同 score 的目标总分没有共同零点，不能把其数值差解释成提升。

## 已知坍缩条件与 FULL

下表仅展示已在根因审计中预先列出的 focus；[全部 120 行](m05_receiver_controls_case_arm_rows.csv) 完整保留，无按结果选择实验条件。

| 条件 | RAW_U support 内 | CENTERED_U | RAW_D4 | CENTERED_D4 |
|---|---:|---:|---:|---:|
| p1 FULL，181 帧 | 152 | 181 | 169 | 181 |
| p2 FULL，181 帧 | 169 | 181 | 172 | 181 |
| p1 speed075，241 帧 | 21 | 241 | 219 | 241 |
| p2 mean3，181 帧 | 3 | 179 | 140 | 179 |
| p2 mean5，181 帧 | 5 | 176 | 158 | 181 |

去偏单独、D4 单独相对 RAW_U，在所有 30 条件上均减少 support error 和 weighted error；它们不是两个可按条件择优合成的接收器。联合相对去偏单独在 2 条件改善、28 条件相同，没有条件变坏。这批保存观测更支持“先移除持续列偏置解决主要选路竞争”，同时支持有限驻留可减少余下部分长驻留误选；不能据此证明任意视频均可如此估计背景。

联合臂仍有 6 个收到帧在 support 外，全部最小误差 1 帧：p1 mean3 的 t=59；p1 mean5 的 t=33/117/155；p2 mean3 的 t=1/6。它们未被忽略或调参修掉。去偏单独还有 p2 mean5 的 5 帧及 p1 mean5 的额外 3 帧在 support 外。

## 支持域与工程上限

D4 是本次候选的固定建模假设，不是从 writer tubelet=4 推出的物理攻击界限。它允许每源最多 4 个连续收到帧、任意正向删帧跳跃和自由起止；五次真重复、超过四倍的局部慢速展开可能被排除。中位数去偏也可能在很短或长期重复的片段中减去真实信号；单帧去偏必然平局。此次 30 条件没有覆盖所有这些困难场景。

定向 CPU 检查 13/13 通过：独立小宽度完整路径穷举 top2、exact ties、4/5 次真重复边界、短片 1/2/3 帧、自由起止/大前向跳跃、N>724 无合法 D4 路径、中位数平移不变性、混合 support 与 weighted error 分离、缺失/格式错误/事后失败保留、途中中断、无 .git 的实际 CLI 运行。测试用于核对固定构造，没有选择参数。一次完整保存矩阵执行退出码 0，用时约 37.51 秒，模型/媒体调用均为 0。

运行入口：

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES= python -B \
  -m experiments.paper_results_v1.receiver_controls_cli \
  --diagnostics-root /home/richar/projects/Video-WM/diagnostics \
  --output /path/to/a/fresh/output
```

这会使用固定 30 个输入和四臂；不提供去偏/驻留/条件扫描参数。已有 partial/complete 输出可用 `--output <existing> --report-only` 刷新 CSV/summary，不重跑选择器。运行需要 NumPy，不加载模型依赖。缺文件和真实数值/格式问题保留为对应失败状态；不使用 hash、manifest、B64、Git 状态或精确版本门禁。

[精简汇总 JSON](m05_receiver_controls_summary.json) 与 120 行 CSV 随开发分支代码交付。完整 q/b/centered 矩阵、top2 路径、逐帧 support 和 weighted error、状态与日志留在 `/home/richar/projects/Video-WM/diagnostics/a-line-m05-receiver-controls-20261010/`，不提交原始大矩阵或私有下载索引。实现者仅创建本地审阅 commit、不 push，由根会话组织同版本独立实现审阅及已获授权的 dev 发布；main 和原 notebook/方法保持原样。
