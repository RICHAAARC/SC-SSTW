# 冻结管状状态方法的跨内容裁剪复现 A

仅工程准备和 CPU/synthetic/static 验证；未执行模型、GPU、Colab或真实数据重新评分。

固定源：`FlowTubeStateHoldout/flow_tube_state_holdout_20260920T033109891109Z`，源码 `0797064cc5f1166170ed5c229e18defd681d8185`。已核此 Git 历史中的 holdout 配置：白天鹅 20261001、蓝色缆车 20261002，case 为 holdout_p0_s0、holdout_p1_s0。两内容/种子不同于本 tube/state 开发集；曾用于反演候选，不称全项目未见。实际运行才验证该源的结果、config/codebook及MP4哈希，不把历史源码存在当作媒体通过。

两个 case 各 OFF/LAST_A/LAST_B，共6源MP4。每源固定129帧、start0/4/5，共18相关裁剪、72接收VAEencode，12 marked裁剪；每start固定4 marked及2 OFF。0 Transformer、0 VAEdecode、0 MP4save，不生成新视频。只把既存MP4真实readback RGB恢复uint8裁剪保存npy，再按哈希实际重读；无第二次有损编码。内部4origin实际RGB长度129/125/125/125，latent长度33/32/32/32。

原成功开发crop入口和纯方法/runtime均不改；新编排独立，复用原flow_tube_crop_analysis与state_clock.read(obs,publicbook)。原五mode和三no-search mode、11窗/1760名义支持分母、missing惩罚、公开key/码本、margin、状态和时钟候选完全冻结。no-search从原固定g0/scale1/offset0/delta0/boundary11候选提取原分数；不按本次结果选算法或参数。

源start和消息truth只用于生成裁剪与盲read后的report，不传receiver。start0/4/5事后几何参考(g,offset)为(0,0)/(0,4)/(3,5)，结构完整窗8/7/7。实际可用完整窗才进入对应评价；缺失不能当匹配，VAE边界变化不保证精确latent等值。消息唯一不等于clock唯一；保留top ties/观察等价类，OFF只排名、不算FPR。

所有18片段72观测槽固定保留；逐start×8mode共24行紧凑摘要，每行分母4 marked/2 OFF、间隔、覆盖和窗口映射。12 marked裁剪不是12独立内容。科学结论需用户执行后审核，当前不声称泛化、FPR、容量或裁剪成功。

输出新目录 `MyDrive/Video-WM/FlowTubeHoldoutCrop/flow_tube_holdout_crop_<UTC>`；禁止输出位于源目录内。Run all固定入口，无菜单，首代码单元独立Drive两行；发布时绑定执行源码SHA（与历史源视频SHA分开）。保存运行源码、配置、公开码本、MP4来源hash、uint8裁剪、72观察、blind detection与全部失败。B多步机制对照独立实施，不在此入口执行。
