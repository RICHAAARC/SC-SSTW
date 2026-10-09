# 同版条件性联合恢复

> 2026-10-09 导航：下文保留条件联合恢复原版本的协议、发布阶段和结果范围。后续研究分支归属及不确定审计见[当前状态](research_status_2026-10-09.md)，不改变本实验的原结论或 main 实现范围。

固定入口：`python -B -m experiments.wan_state_clock.video_trajectory_conditional_joint_v1_run --config experiments/wan_state_clock/configs/video_trajectory_conditional_joint_v1.json --output <新的输出目录>`。用户 Run-all Notebook 为 `notebooks/video_trajectory_conditional_joint_v1_colab.ipynb`。源码 S 候选阶段全部正式 Notebook 保持 `SOURCE_SHA=None` 并在输出目录创建前拒绝运行；发布 S 后，交付 N 将全部正式 Notebook 绑定到该 S。main 发布已完成 CPU/fake/schema/portability 验证，其中包含 synthetic CPU FFmpeg fixture；未运行新的真实模型、GPU、Colab 或 Drive 实验，已有固定真实证据仍绑定原源码 `ac111d0fed253767651929d115c343fe1636c525`。

命题是预声明协议内的条件盲恢复：收到视频假定已含 trajectory payload 与 M05 terminal sync，接收器只用 RGB、当前 key、公开长度与协议族。它不知道真实 offset、删除位置、消息，也不通过真攻击类别选择算法。不是任意视频存在检测或自动攻击路由；不调用 Stage1 的阈值、m 或 ACCEPT_ACTION 门。

唯一来源沿已采用黄帆船 prompt / seed 2026100701 重新生成一次，属于已看开发源，不是 held-out；不要求重生成的 RGB 身份等于旧结果，不换 seed/重试。源 worker 使用 payload-only 的旧成功算术：50 次 native step、conditional/unconditional 各 50 次，25…49 共 25 次 local control/payload gradient、pilot 0；释放 transformer/初始状态/prompts 后才 load native VAE、decode/save，worker 退出。matched worker 共用一个未改 framewise latent，P1 直接解码，副本写 M05 projection target .5 再解码，各完整视频一次 CRF18/yuv420p/8fps MP4。P0 没有 codec，裁剪不再 codec。新 runtime 不导入历史 experiment、OLD8/state、G/T 或 shim；正式 main 的 reader/generation/VAE/native-step/media 通过隔离端口复用，main 与旧入口保持原字节。

| 协议 | 准备观察 | 原模式与两 key | 固定 payload 槽 |
|---|---|---|---:|
| phase1 | FULL181、177@1、89@37 | BASELINE / EST_ALIGN |12|
| phase2/3 |177@2、177@3、89@38、89@39 | BASELINE / EST_ALIGN |16|
| single-jump | C=X[2:179]；D=X[2:90]+X[91:180] | RAW / GLOBAL_ALIGN / PATH_ALIGN / TRUTH_PATH |16|

9 个逻辑观察使用 8 份高度重叠的同源物理输入；C 是 177@2 的输入别名，不是独立样本。全部 44 槽（40 blind + 4 oracle）：1605120 票、53504 time-bit、1408 final-bit。新 framewise receiver 共 8 encode / 1156 帧 / 151 batch8；18 score / 3426 candidates；14276 global tubelet rows，与 3540 single-jump frame-cache / 900 local-grid 分开记录。Wan 一般上界 blind 28 encode / 36 key-read，总计 30 / 40。实际完整 input identity、N、received-index-map 相同才共用 encode；key 相同才共用 read，失败也缓存，oracle 不重试。别名保留逻辑分母，不增加证据。

GLOBAL 保留原 tubelet→patch 累加；single-jump 保留原 frame 顺序与 H0/H1 709 候选、885 cache、source180 特殊 rho。C 的两套原评分分别计算，即使实数代数等价也不静默替换浮点顺序或 tie。原完整有限唯一估计与 tie_atol=1e-12 不变，未决无 fallback。按估计 b%4 prepend 首帧并 truncate；H1 先于估计 k 插入前一 received 帧占位。该操作补名义格，不恢复原缺帧，不保证 Wan 影响局限于 4 帧。完整 map 相同不等于 source path 唯一。

接收投票用稳定 R44（181/177）与 R22（89），latent1…R，4ch / 32bits / keyed240 coords / 每bit30频率；FFT2 ortho real、strict >0（exact0 为负）、time-major bit::8、Counter 首遇平票全部保持。保留 signed votes、zero mask、全部 final/time/channel 记录；跨长度只比较归一余量，名义 receiver time 不是匹配同源感受野。

顺序为 sync seal → blind plan → 40 blind payload seal → oracle 配置语义解析/4 oracle → oracle seal → message/真实 map posthoc。准备层合法提前知道 writer message 和裁剪构造；配置字节提前 hash 仅用于身份，不进入 selector。实际依赖闭包与四份实际 config 单独记录 SHA；无自身 `.git` 不读取外部父仓库身份。中断后保留全部失败槽；Notebook 的父进程 SIGTERM 会触发 runner 回收独立 worker 整组（含 codec 子进程），不留下后台模型占用。

六行质量为 PRE(P1/P0、M05/P1、M05/P0) 和 POST(P1_received/P0_native、M05_received/P1_received、M05_received/P0_native)。P0 已有 payload、未压缩，非无水印或 matched-codec 基准。报告原 RGB MSE/PSNR、clip 自身时差比；M05/P1 另报 residual L2、Dt、45 跨界/135 内部能量、归一粗糙度及 lag1。六行只加 CPU 统计/IO，无新增 codec 或感知模型；不把像素比值称感知闪烁，不设质量 PASS 阈值。每模式记录完整 map、合成/重复/丢弃/占位/裁尾成本；全 44 槽及 RAW/GLOBAL/PATH/oracle 的归一余量差分均封后展示。

有限 K0 主路径须原规则唯一、真实 offset/phase 或 single-jump family/b/k/完整 map 正确，并且对应 blind EST_ALIGN/PATH_ALIGN 32bits 正确。有效错误路径或正确路径但 payload 错是反例；完整但 tie 为证据不足，资源/输入失败为技术未决。FULL 仅 geometry/payload 对照，RAW/GLOBAL 不强求全对，oracle 不补 blind 成功，K1 不设 BER/拒绝阈值。全 time-bit 正仅描述。COMPLETE 是运行完整性，不是科学 PASS。

本轮仅将已采纳机制在一版源码组合；历史分版结果不自动变成本版结果。Stage1 反例保留：177 正例2/2、null误授权1/6；89 正例2/2、null0/6但全6 UNCERTAIN。任意视频存在/错钥拒绝/FPR/uncertainty 未闭合且暂停改判据；本次条件联合恢复不以前述门为前置。未新增独立源、未知攻击、阈值调参或论文实验。


## 已保留的原版本真实证据

原运行 `20261008T003135066923Z/fixed_reference` 绑定源码
`ac111d0fed253767651929d115c343fe1636c525`，result SHA256 为
`a3e3d64286514817d5ee3c6ce1fe9bd3486b0752a65477ec0682f48991941a44`。
审计保留 549 项全通过核对与独立数值复算无问题。8/8 K0 主判逻辑槽
取得正确路径并恢复 32 bits，但只对应同一已见开发源的 7 份物理输入；
44 逻辑槽实际为 21 encode / 32 key read。全部 22 个 K0 槽（含
RAW/BASELINE）都已经零错，因此不支持 BER 改善或校正必要性。D 的
GLOBAL/PATH 对应为 88/177 与 177/177，b=2、k=88；最小余量为
.504545/.587879/.659091，逐 bit 并非全面改善。正确校正含 3 个合成
输出、3 次重复及 3 个尾帧丢弃，只恢复名义索引，不恢复缺失画面。

同次 POST M05/P1 PSNR 为 37.26074 dB、clip 时差能量比为 2.07738；
P0 已含 payload，缺少 payload-off 对照。七个大媒体只按保存的元数据
核对，没有在审计中重新下载、解码或重算。证据上限仍是单个已见源、
预声明有限协议族中的条件恢复固定案例；不支持存在检测、FPR、错误钥
拒绝、独立源泛化、未知攻击路由、广义鲁棒性或科学 PASS。main 集成
只引用这份原版本证据，不把它迁移表述为新 main 实测。
