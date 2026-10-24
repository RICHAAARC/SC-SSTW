# 分支用途索引

核查日期：2026-10-24。本页回答“去哪个分支找什么”，不替代各线的方法定义与结果报告。

## 从这里进入

| 需求 | 分支与入口 |
| --- | --- |
| 使用已确认的公共正式机制 | **`main`**：[运行入口](../README.md)、[当前方法与范围](current_method.md)、[机制证据](mechanism_evidence_index.md)、[公共分段载荷](segmented_payload_v1.md)。本轮新增精选公共分段载荷增强；已有正式机制仍按各自声明范围解释。 |
| 查看 A 线成果、论文评价、同步对照 | **`dev/paper-results-v1`**：[A 线入口](https://github.com/RICHAAARC/SC-SSTW/blob/dev/paper-results-v1/docs/paper_results_v1/README.md)、[分段载荷用途验证](https://github.com/RICHAAARC/SC-SSTW/blob/dev/paper-results-v1/docs/paper_results_v1/segmented_payload_v1.md)。 |
| 继续 B 线后续开题与同步研究 | **`dev/local-joint-state-payload-v1`**：[盲时间对应与分段归位](https://github.com/RICHAAARC/SC-SSTW/blob/dev/local-joint-state-payload-v1/docs/blind_temporal_segmented_v1.md)、[近期 s01 整链观察](https://github.com/RICHAAARC/SC-SSTW/blob/dev/local-joint-state-payload-v1/docs/b_affine_margin_observation_v3.md)。 |
| 查找其他候选、旧诊断或发布准备记录 | 按下表定位对应分支，阅读该分支内协议、配置与结果入口。 |

A/B 两线继续保留。本轮将公共分段载荷机制精选进入 `main`，不表示 A 的论文评价/同步对照或 B 的后续同步研究已整包纳入；本轮没有新增同步整合结论。

## 逐分支简表

“在研”指本轮明确保留的 A/B 主线；“其他实验”指各自范围内的独立候选或诊断；“历史保存”指既有开发、早期路线或发布准备记录。分类只服务于导航，不判定分支可删，不表示分支内每个候选均已真实运行或取得科学结论。

| 层次 | 分支 | 实际用途 |
| --- | --- | --- |
| 正式 | `main` | 已确认的公共正式机制；完整 GROW 参考链、已采用的有限同步/接收机制与本轮公共分段载荷增强。 |
| 在研 A | `dev/paper-results-v1` | A 成果整理、论文外层评价、分段载荷用途验证与同步对照。 |
| 在研 B | `dev/local-joint-state-payload-v1` | B 后续开题同步研究：局部状态/载荷联合、盲时间对应、分段归位及局部观察。 |
| 其他实验 | `dev/bidirectional-cross-confirm-v1` | A→B / B→A 双向定位与留出确认的存在接收器候选。 |
| 其他实验 | `dev/content-background-existence-v1` | 用固定错误密钥背景归一化正确密钥分数的存在接收器候选。 |
| 其他实验 | `dev/flow-fixed-key-v1` | 固定单密钥标记的 Flow 写入与存在检测；不等同于多消息归属判定。 |
| 历史保存 | `dev/flow-tube-multistep` | 原 tube/state 载体的早步多步控制、末步补偿与自由尾程对照。 |
| 历史保存 | `dev/flow-tube-objective-alignment` | 固定 T46/终态条件下 hinge 与 tanh 目标、预算与媒体层对齐诊断。 |
| 历史保存 | `dev/flow-tube-partial-observation` | 恰好三组有效观测的局部接收器及既有数据重校准候选。 |
| 历史保存 | `dev/flow-tube-state` | tube-state 载体在 44/45/46 步的原生 Flow 控制历史。 |
| 历史保存 | `dev/flow-tube-state-guidance` | 原 tube/state 闭式 margin 写入与末步 LAST49 桥接。 |
| 历史保存 | `dev/flow-tube-sync-diagnostics` | 既有 LAST 检测结果的 CPU 同步诊断；读取旧 JSON，不新增独立样本。 |
| 历史保存 | `dev/flow-tube-terminal-feedback` | 固定 T46 的单次终端反馈、响应选择与相关诊断。 |
| 历史保存 | `dev/flow-tube-uniform-tanh` | 固定 clean proxy、同史原生响应预算下的 uniform-tanh 控制对照。 |
| 历史保存 | `dev/grow-single-step-jacobian` | 第 30 步共享状态下 local clean 梯度与完整 CFG 输入 Jacobian 对照。 |
| 历史保存 | `dev/grow-spatial-carrier` | GROW 原空间频域载体的 MULTI/LAST 完整媒体配对适配。 |
| 历史保存 | `dev/grow-video-frequency` | GROW 风格视频频域适配及其控制、媒体和读出诊断历史。 |
| 其他实验 | `dev/image-trajectory-reference-v1` | 作者公开 GROW 图像代码的单图参考入口，用于实现来源对照。 |
| 其他实验 | `dev/keyed-presence-t40-t49-v1` | 固定钥 T40–T49 存在标记源实验。 |
| 其他实验 | `dev/media-channel-retention-v1` | 对既存 Window-State-MSE 轨迹做终态到媒体各层存留诊断。 |
| 历史保存 | `dev/public-statistic-control` | 二维公共亮度统计、完整尾程梯度、终端反馈与显存/反向诊断。 |
| 历史保存 | `dev/real-video-public-observation` | Stage-1 帧差加权质心、固定镜头单主体公共观测与关系开发。 |
| 其他实验 | `dev/rgb-dct-b2-terminal-receiver-v1` | RGB-DCT B2 终端接收器梯度与完整/掩码下降诊断。 |
| 其他实验 | `dev/rgb-dct-group-consistency-h0-h1-v1` | RGB-DCT 时间组一致性统计的独立 H0/H1 功能检验。 |
| 其他实验 | `dev/rgb-dct-h0-h1-functional-v1` | RGB-DCT T49 的独立 OFF 校准与盲 H0/H1 功能检验。 |
| 其他实验 | `dev/rgb-dct-local-gradient-v1` | RGB-DCT 局部 VAE 梯度候选、checkpoint 与终端比较修复。 |
| 其他实验 | `dev/rgb-dct-multistep-comparison-v1` | 同源 OFF / SINGLE49 / SINGLE46 / MULTI44_46 写入时刻与多步比较。 |
| 其他实验 | `dev/rgb-dct-structured-terminal-feedback-v1` | 固定三维时间分箱方向、真实尾程响应的结构化终端反馈。 |
| 其他实验 | `dev/rgb-dct-temporal-balanced-v1` | 全帧 RGB-DCT 时间平衡载体、连续分数与 T49 开发候选。 |
| 其他实验 | `dev/rgb-dct-terminal-gradient-v1` | RGB-DCT 完整终端梯度与 T46 自由尾程写入候选。 |
| 历史保存 | `dev/sc-sstw-core-integration` | 早期四位载荷、44/46 控制、媒体攻击与盲同步的集成候选历史。 |
| 其他实验 | `dev/trajectory-attribution-v1` | 轨迹载荷归属判定、不确定补充与对应真实运行审计记录。 |
| 历史保存 | `dev/trajectory-payload-framewise-sync-v1` | trajectory payload、framewise M05、对齐/单删与 conditional-joint 的开发来源；仅已采用部分精选进入 main。 |
| 历史保存 | `dev/video-inversion` | Wan 初始噪声反演载体、盲同步及其开发诊断历史。 |
| 其他实验 | `dev/video-trajectory-blind-validation-v1` | 连续约束求解器及 OLD8、局部 Fourier/时间同步等盲接收开发记录。 |
| 其他实验 | `dev/videoseal-receiver-diagnostic-v1` | CPU VideoSeal 原始 logit 的诊断归约器。 |
| 其他实验 | `dev/window-state-mse-v1` | legacy 与 window-state MSE 的同源配对写入/媒体实验。 |
| 历史保存 | `c2a-2a-colab-preparation` | 早期 C2A、合成 AISB/校准/同步机制及 state-clock Colab 准备记录。 |
| 历史保存 | `release/conditional-joint-v1-main-20261008` | conditional-joint V1 的精选 main 发布候选快照。 |

## 核查范围

本次只读核对本地 `git branch -vv`、`git worktree list`、各分支末次提交，以及用途相关 README、协议、配置和入口：**39 个本地分支、38 个工作树**。发布候选 `release/conditional-joint-v1-main-20261008` 当前没有工作树；数量差异本身不表示缺失或错误。A/B 的远端分支入口另作只读确认；是否设置本地 upstream 不用于判断是否发布。

分支内旧说明中的“当前”“正式”按其当时任务理解。源码、配置、环境和来源信息仅供定位；分支用途表不设置摘要、版本或 Git 状态门禁，也不把实现/静态核验当作真实运行或科学结论。此整理不删除、重命名、reset 或归档任何分支与工作树。
