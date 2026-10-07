# 有限研究路线与混合架构边界

以 workspace 根《研究内容二_时空一致水印机制_开题设计与进度核对_2026-09-30.md》为历史目标参照，保留原文。本阶段采用的准确架构是：**生成轨迹频率payload＋末端逐帧VAE管状块时间码＋盲全局offset估计/phase校正**；固定内部单跳已有有限开发源实测；当前推进同一源码内真实 trajectory 写入至盲路径/相位与 payload 的条件性联合验证，尚待本版用户 Run-all。

Payload在Wan 50个真实采样步中的25…49控制。它对整张40×64 latent做FFT，在h8…19、w12…31的240个频率坐标、ch0…3承载32bits，每bit30个频率；同32bits重复于46个写入时间位置。频带局部不等于空间局部，不能称空间patch payload或绝对时间编码。

Sync在Wan native decode之后经2D VAE写入，使用4帧（源末块1帧）×4×4空间块、4通道的keyed时间证据。它是末端时间码，不是轨迹sync；latent局部也不保证严格独立的RGB支持。当前global offset不是状态路径；重复payload读对不证明时间相关片段编码完成。

| 已有对照入口 | 可支持的分离与边界 |
|---|---|
| P0→P1 | 额外2D VAE往返；P0已带payload，非无水印对照 |
| P1→M1/M05；M1→M05 | terminal sync增量；projection target变化，不是delta简单减半 |
| M05→T05 | 该平滑构造的幅度/质量/定位权衡，不能拼入G Stage2联合优胜 |
| origin / PREPEND1 / TAIL | 已知起点的phase相关兼容性与末尾内容对照，不证明唯一phase原因 |
| Stage2 | 同源 M05 新 blind offset1/37→p1→payload：CROP/SHORT final errors 13/14→0；固定链路正证据，不覆盖全部 phase 或独立泛化 |
| 1A / 1B | 1A 同开发源固定四窗 errors 11/1/11/1→0；1B 预固定单新源四窗已接受，final errors 0→0、最弱余量提高，并非最终纠错或对齐必要性证明 |

证据入口均在 workspace 根 diagnostics：trajectory-framewise-gt-real-run-audit-20261005/report.md、trajectory-framewise-m05-real-run-audit-20261005/report.md、trajectory-receiver-origin-real-run-audit-20261006/report.md、trajectory-receiver-prepend1-stage1-real-run-audit-20261006/report.md、trajectory-receiver-estimated-align-stage2-real-run-audit-20261006/report.md。本次不重跑这些控制，不为要求MULTI胜过LAST而新增消融。

| 有限里程碑 | 范围与次序 | 完成标准 / 当前状态 |
|---|---|---|
| ① 固定裁剪兼容性 | 1A 开发源四窗与 1B 预固定蓝色玩具车单新源四窗均已审计接受 | 仅固定声明窗的 offset/phase 与最终 payload 兼容成立；1B baseline 已全对，不证明对齐必要性或广泛独立泛化 |
| ② 当前混合架构的机制作用与边界核对 | 核对混合架构中各机制的作用、边界与现有证据；不新增local payload或重做trajectory sync | 载体、时序、贡献与证据准确对应即完成，不强迫新消融；本文已整理 |
| ③ 最小动态路径 | 已采纳固定 C/D 内部单帧删除、709 条 H0/H1 路径及 RAW/GLOBAL_ALIGN/PATH_ALIGN/TRUTH_PATH；不扩成未知攻击全包 | 固定 C 的 H0 b2 与 D 的 H1 b2 k88 已实测对应 177/177；D 全局 H0 仅 88/177，RAW/GLOBAL/PATH final 均 0 错，因此不是 BER 收益证明。窗口对应、精确缺口和 payload 分别评价；整段 offset、blind 路径与封后 oracle 分列。重复/变速未实施，结论依实测 |
| ④ 盲存在拒绝/擦除 | 任意视频 presence、no-watermark、wrong-key rejection、FPR 与 uncertainty 未闭合，暂停 Stage1 判据修订；不作为条件恢复前置 | Stage1 已保存反例：177 正例2/2、null误授权1/6；89 正例2/2、null0/6但6个全UNCERTAIN。不能用唯一 argmax 或联合恢复成功抹去这些限制 |
| ⑤ 同一最终版本联合机制验证 | 含画质、时间编辑代价与适用范围；不自动进入论文实验 | 同一最终版在有限预声明名单联合交定位/路径、payload、画质、时间编辑代价与成本完整包；机制全部闭合后先报告用户，再讨论是否进入论文实验；当前同版条件联合候选工程实现，未真实执行；未测范围仍未完成 |

精确offset、phase、最终bit与receiver时间位置的聚合余量须分别评价。FULL singleton和alias不增加独立证据，K1是描述性对照，sync_accepted=False。任何阶段均不把原说明的全部时域编辑、局部payload和泛化目标悄然记为完成；也不无限新增方法模块。空间局部 sync 管状块已有实测；payload 仍为全 latent 频带、同 32 bits 时间重复，空间局部 payload 与状态相关片段编码未完成。当前采用混合架构，不把 sync 迁至生成轨迹；固定单跳实测不代表一般动态路线完成。

1A/1B 证据：workspace 根 diagnostics/trajectory-receiver-phase23-milestone1a-real-run-audit-20261006/report.md、trajectory-receiver-independent-source-milestone1b-real-run-audit-20261006/report.md。原单跳固定入口见 video_trajectory_internal_single_deletion_v1.md；本次同版条件联合范围/预算见 video_trajectory_conditional_joint_v1.md，黄帆船同参数重生成仍为已看开发源，不扩论文实验 roster。

论文脚本、测试集规划、图表模板、大规模比较及其准备暂不执行。空间局部 payload 与状态相关片段编码未完成仅记录原始目标与已采纳范围的差异，不据此自动新增 payload 路线。

后续同版条件联合验证只针对公开预声明协议、收到视频已有 trajectory payload + M05 的条件命题，接收器仍未知 offset、删除位置、message；不是自动识别任意攻击。已有单删证据为 workspace diagnostics/trajectory-internal-single-deletion-real-run-audit-20261006/{audit.json,numeric_audit.json,result.json}；Stage1 反例为 diagnostics/terminal-sync-path-decision-real-run-audit-20261007/audit_verification.json。历史原记录不改、不拼成新版本已实测。
