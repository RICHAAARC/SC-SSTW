# 有限研究路线与混合架构边界

以 workspace 根《研究内容二_时空一致水印机制_开题设计与进度核对_2026-09-30.md》为历史目标参照，保留原文。本阶段采用的准确架构是：**生成轨迹频率payload＋末端逐帧VAE管状块时间码＋盲全局offset估计/phase校正**；动态状态路径仍待里程碑③。

Payload在Wan 50个真实采样步中的25…49控制。它对整张40×64 latent做FFT，在h8…19、w12…31的240个频率坐标、ch0…3承载32bits，每bit30个频率；同32bits重复于46个写入时间位置。频带局部不等于空间局部，不能称空间patch payload或绝对时间编码。

Sync在Wan native decode之后经2D VAE写入，使用4帧（源末块1帧）×4×4空间块、4通道的keyed时间证据。它是末端时间码，不是轨迹sync；latent局部也不保证严格独立的RGB支持。当前global offset不是状态路径；重复payload读对不证明时间相关片段编码完成。

| 已有对照入口 | 可支持的分离与边界 |
|---|---|
| P0→P1 | 额外2D VAE往返；P0已带payload，非无水印对照 |
| P1→M1/M05；M1→M05 | terminal sync增量；projection target变化，不是delta简单减半 |
| M05→T05 | 该平滑构造的幅度/质量/定位权衡，不能拼入G Stage2联合优胜 |
| origin / PREPEND1 / TAIL | 已知起点的phase相关兼容性与末尾内容对照，不证明唯一phase原因 |
| Stage2 | 同源M05新blind offset1/37→p1→payload的固定链路正证据，不覆盖全部phase或独立泛化 |

证据入口均在 workspace 根 diagnostics：trajectory-framewise-gt-real-run-audit-20261005/report.md、trajectory-framewise-m05-real-run-audit-20261005/report.md、trajectory-receiver-origin-real-run-audit-20261006/report.md、trajectory-receiver-prepend1-stage1-real-run-audit-20261006/report.md、trajectory-receiver-estimated-align-stage2-real-run-audit-20261006/report.md。本次不重跑这些控制，不为要求MULTI胜过LAST而新增消融。

| 有限里程碑 | 范围与次序 | 完成标准 / 当前状态 |
|---|---|---|
| ① 当前1A；随后必要1B | 四窗正确key p2/p3开发兼容性；规则稳定后预固定新源小复核。1B exact prompt/seed未采纳，不阻1A | 1A交付完整四窗结果；仅声明窗offset/phase与payload均成立才称该范围兼容，否则保留具体负结果，不调参求通过。当前工程待发布实测。1B先冻规则，再新源一次完整报告 |
| ② 并行架构/贡献整理 | 按上述混合架构限缩论文；不新增local payload或重做trajectory sync | 载体、时序、贡献与证据准确对应即完成，不强迫新消融；本文已整理 |
| ③ 最小动态路径 | 先内部删帧的路径与payload链；重复/变速仅在保留相应论文主张时增加预固定有限验证 | 预先冻结最小删帧与路径评价，盲路径解释跳变并联合核payload；整段offset对照、真值路径诊断与正式结果分列，结论依实测；未实施 |
| ④ 盲存在拒绝/擦除 | 开发规则与独立留出；不提前扩为极低FPR试验 | 冻结存在、拒绝、擦除规则后，独立留出正负完整判决；全拒绝不能称成功，无极低FPR前置；未实施 |
| ⑤ 同版联合验证 | 冻结有限声明范围与机制后，再做独立论文主实验 | 同一最终版在有限预声明名单联合交定位/路径、payload、质量、成本完整包后转主实验；未测范围仍未完成；未实施 |

精确offset、phase、最终bit与receiver时间位置的聚合余量须分别评价。FULL singleton和alias不增加独立证据，K1是描述性对照，sync_accepted=False。任何阶段均不把原说明的全部时域编辑、局部payload和泛化目标悄然记为完成；也不无限新增方法模块。当前1A代码没有实现③④或1B。
