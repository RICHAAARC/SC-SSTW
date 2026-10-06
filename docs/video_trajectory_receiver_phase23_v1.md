# 固定开发源 phase2/3 长短裁剪（Milestone1A）

本入口实现已采纳 1A：验证保存 G M05 FULL181 RGB 的 SHA 后读一次，CPU clone [2:179]、[3:180]、[38:127]、[39:128]，接收长度177/177/89/89不变。不运行 FULL/phase1 对照，不新增 codec、生成、writer 或质量测量。

输入构造和接收选择分开：preparation JSON仅供构造四个fixture，记录source-start、四份CPU输入SHA和源帧图；接收公开config仅含opaque IDs、长度/形状及身份占位。构造后传入四份RGB，不把start/真phase/消息传给score、estimate或alignment。单独posthoc JSON只在最终payload seal后读取。CPU切片保留内存，不另存/编码四份媒体；可由固定FULL与准备receipt复现。

直接复用 Stage2 的 framewise/Wan适配器、完整唯一估计、phase_map、physical_plan与详细票。每观察一次fresh framewise encode，两key独立全候选评分。sync seal后冻结物理计划，完成payload后再seal与truth join。缺源/切片失败/中断/无唯一估计均保留16逻辑槽；不补0、不借K0估计给K1、不按真实phase或BER选择。

| 固定计数 | 数量 |
|---|---:|
| FULL读取 / CPU clone | 1 / 4 |
| Framewise encodes / 帧 / batch8底层批次 | 4 / 532 / 70 |
| Sync reads / candidates / locals | 8 / 392 / 9456 |
| Payload逻辑reads / votes / time-bit / final-bit | 16 / 506880 / 16896 / 512 |
| Wan物理encode / key read，完整输入范围 | 4…12 / 8…16 |

Wan load一次，framewise load一次；公共R44/R22、latent1…R、32bits/4channels/240coords、strict>0及Counter首遇平票保持。只在同观察同phase共享encode；key各读一次，p0 aligned引用同key baseline。物理实际attempted/completed与逻辑计划分开，别名不增独立证据。

四窗没有旧同窗history；代码不加载历史同步/载荷记录，也不以历史缺项阻断当前run。精确offset、phase、最终bit错误和全部time/channel余量分开报告；全time-bit正不是新PASS门槛。新短窗与旧start37高度重叠，既不是独立源，也不能唯一归因phase。

五代码单元Notebook沿成功安装/probe/subprocess清理/失败持久化路径，首cell精确mount；未发布模板的SOURCE_SHA=None在输出目录创建前守卫，已发布Notebook绑定不可变源码。

1A 已保存实测审计被接受：固定四窗 K0 的 offset/phase 均正确，最终错误数11/1/11/1→0；全部 time-bit 正是描述性结果而非新增门槛。证据：/home/richar/projects/Video-WM/diagnostics/trajectory-receiver-phase23-milestone1a-real-run-audit-20261006/report.md。结论限于同一开发源的高重叠四窗，不是 heldout、FPR、唯一phase因果或科学PASS。

1B 已另行采纳蓝色玩具车固定 prompt 与 seed 2026100601，见 video_trajectory_receiver_independent_source_v1.md；当前1B仅工程待实测，不改变本1A入口。新生成、动态删除、其他攻击和拒绝阈值不在本入口中。

入口：experiments.wan_state_clock.video_trajectory_receiver_phase23_v1_run
Builder：scripts/build_video_trajectory_receiver_phase23_notebook.py
配置：公开config、preparation config、posthoc config分别绑定SHA；source receipt包含新runner/config和复用源码。
