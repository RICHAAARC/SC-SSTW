# 分支用途与归档索引

更新日期：2026-10-24。本页用于定位当前入口与历史材料，不替代各线的方法定义和结果报告。两阶段整理已完成：**本地4条分支、远端4条分支、4个活动工作树**。用户已接受旧历史分支的GitHub／Colab链接失效；历史源码和证据保存在既有本地归档。

## 三个主入口

| 需求 | 分支与入口 |
| --- | --- |
| 使用已确认的公共正式机制 | **`main`**：[运行入口](../README.md)、[当前方法与范围](current_method.md)、[机制证据](mechanism_evidence_index.md)、[公共分段载荷](segmented_payload_v1.md)。 |
| 查看 A 线成果、论文评价、同步对照 | **`dev/paper-results-v1`**：[A 线入口](https://github.com/RICHAAARC/SC-SSTW/blob/dev/paper-results-v1/docs/paper_results_v1/README.md)、[分段载荷用途验证](https://github.com/RICHAAARC/SC-SSTW/blob/dev/paper-results-v1/docs/paper_results_v1/segmented_payload_v1.md)。 |
| 继续 B 线后续开题与同步研究 | **`dev/local-joint-state-payload-v1`**：[盲时间对应与分段归位](https://github.com/RICHAAARC/SC-SSTW/blob/dev/local-joint-state-payload-v1/docs/blind_temporal_segmented_v1.md)、[近期 s01 整链观察](https://github.com/RICHAAARC/SC-SSTW/blob/dev/local-joint-state-payload-v1/docs/b_affine_margin_observation_v3.md)。 |

另保护 **`c2a-2a-colab-preparation`**：它是 `alive/SC-SSTW` 的主 checkout，**不是 main**。本次保留其本地分支、工作区及远端。

main 当前包含精选的公共分段载荷增强；A 的论文评价/同步对照与 B 的后续同步研究仍各自保留。本次目录与引用整理不扩大方法或科学结论。

## 整理结果与材料位置

| 项目 | 最初 | 第一阶段完成 | 远端收尾完成 |
| --- | ---: | ---: | ---: |
| 本地分支 | 39 | 4 | 4 |
| live origin 分支 | 36 | 35 | 4 |
| 活跃工作树 | 38 | 4 | 4 |

4 个本地分支与活跃工作树为 main、A、B 和上述主 checkout；A/B/c2a 的 HEAD 与工作内容保持不变，main 仅提交本次整理文档。34 份非活动工作树已全量收起，包含未提交、未跟踪及 ignored 文件；研究文件仍保存在归档目录中。本次未创建 tag。

项目根目录下的 `archive/branch-cleanup-20261024/` 保存：

- `repository-history.bundle`：整理前全部 39 个本地 heads 及祖先提交的普通 Git bundle。
- `trees/{原工作树名}/`：34 份完整文件树；原 `.git` 指针仅改名为 `.git-worktree-pointer.saved.txt`，这些目录已不作为活跃工作树。
- `records/{原工作树名}/`：对应索引、patch 和定位记录。
- `inventory_before.json`、`inventory_after.json`：整理前后清单；恢复说明见同目录 `README.md`。

绝对归档位置：`/home/richar/projects/Video-WM/archive/branch-cleanup-20261024/`。完成报告：`/home/richar/projects/Video-WM/diagnostics/branch-cleanup-20261024/REPORT.md`。

第一阶段为兼容旧Notebook暂留31个历史origin。用户随后明确接受这些旧链接失效并授权删除，第二阶段已实际移除这31个远端分支；加上第一阶段的core，远端由最初36条收敛为4条。旧release原本仅本地。没有新建保留标签、搬入旧实验代码或替换现有方法。精确删除名单、入口检查与保存说明见[整理收尾记录](branch_cleanup_20261024.md)。

历史内容请从本地bundle／完整文件树查阅或按需恢复；不再以旧GitHub／Colab分支链接作为历史材料入口。归档里的未提交内容不据此视为已发布。

## 逐分支用途与保存状态

下表保留整理前全部 39 个分支的用途。“其他实验”“历史保存”是导航分类，不表示全部废弃、科学无效或全部合入 main。历史行的“本地位置”是上述 `trees/` 下的目录名；对应本地引用已清理，提交历史在 bundle 中。没有 live origin 的行仍可从本地归档查阅。

| 层次 | 分支 | 实际用途 | 本地位置 | origin 状态 |
| --- | --- | --- | --- | --- |
| 正式 | `main` | 已确认的公共正式机制；完整 GROW 参考链、已采用的有限同步/接收机制与公共分段载荷增强。 | `worktrees/SC-SSTW-main-publish`（活跃） | [保留](https://github.com/RICHAAARC/SC-SSTW/tree/main) |
| 在研 A | `dev/paper-results-v1` | A 成果整理、论文外层评价、分段载荷用途验证与同步对照。 | `worktrees/Paper-Results-V1`（活跃） | [保留](https://github.com/RICHAAARC/SC-SSTW/tree/dev/paper-results-v1) |
| 在研 B | `dev/local-joint-state-payload-v1` | B 后续开题同步研究：局部状态/载荷联合、盲时间对应、分段归位及局部观察。 | `worktrees/Local-Joint-State-Payload-V1`（活跃） | [保留](https://github.com/RICHAAARC/SC-SSTW/tree/dev/local-joint-state-payload-v1) |
| 其他实验 | `dev/bidirectional-cross-confirm-v1` | A→B / B→A 双向定位与留出确认的存在接收器候选。 | `Bidirectional-Cross-Confirm-V1` | 远端已移除；见本地归档 |
| 其他实验 | `dev/content-background-existence-v1` | 用固定错误密钥背景归一化正确密钥分数的存在接收器候选。 | `Content-Background-Existence-V1` | 远端已移除；见本地归档 |
| 其他实验 | `dev/flow-fixed-key-v1` | 固定单密钥标记的 Flow 写入与存在检测；不等同于多消息归属判定。 | `Flow-Fixed-Key-V1` | 远端已移除；见本地归档 |
| 历史保存 | `dev/flow-tube-multistep` | 原 tube/state 载体的早步多步控制、末步补偿与自由尾程对照。 | `Flow-Tube-Multistep` | 远端已移除；见本地归档 |
| 历史保存 | `dev/flow-tube-objective-alignment` | 固定 T46/终态条件下 hinge 与 tanh 目标、预算与媒体层对齐诊断。 | `Flow-Tube-Objective-Alignment` | 远端已移除；见本地归档 |
| 历史保存 | `dev/flow-tube-partial-observation` | 恰好三组有效观测的局部接收器及既有数据重校准候选。 | `Flow-Tube-Partial-Observation` | 无 live 分支 |
| 历史保存 | `dev/flow-tube-state` | tube-state 载体在 44/45/46 步的原生 Flow 控制历史。 | `Flow-Tube-State` | 远端已移除；见本地归档 |
| 历史保存 | `dev/flow-tube-state-guidance` | 原 tube/state 闭式 margin 写入与末步 LAST49 桥接。 | `Flow-Tube-State-Guidance` | 远端已移除；见本地归档 |
| 历史保存 | `dev/flow-tube-sync-diagnostics` | 既有 LAST 检测结果的 CPU 同步诊断；读取旧 JSON，不新增独立样本。 | `Flow-Tube-Sync-Diagnostics` | 远端已移除；见本地归档 |
| 历史保存 | `dev/flow-tube-terminal-feedback` | 固定 T46 的单次终端反馈、响应选择与相关诊断。 | `Flow-Tube-Terminal-Feedback` | 远端已移除；见本地归档 |
| 历史保存 | `dev/flow-tube-uniform-tanh` | 固定 clean proxy、同史原生响应预算下的 uniform-tanh 控制对照。 | `Flow-Tube-Uniform-Tanh` | 远端已移除；见本地归档 |
| 历史保存 | `dev/grow-single-step-jacobian` | 第 30 步共享状态下 local clean 梯度与完整 CFG 输入 Jacobian 对照。 | `GROW-Single-Step-Jacobian` | 远端已移除；见本地归档 |
| 历史保存 | `dev/grow-spatial-carrier` | GROW 原空间频域载体的 MULTI/LAST 完整媒体配对适配。 | `GROW-Spatial-Carrier` | 远端已移除；见本地归档 |
| 历史保存 | `dev/grow-video-frequency` | GROW 风格视频频域适配及其控制、媒体和读出诊断历史。 | `GROW-Video-Frequency` | 远端已移除；见本地归档 |
| 其他实验 | `dev/image-trajectory-reference-v1` | 作者公开 GROW 图像代码的单图参考入口，用于实现来源对照。 | `Image-Trajectory-Reference-V1` | 远端已移除；见本地归档 |
| 其他实验 | `dev/keyed-presence-t40-t49-v1` | 固定钥 T40–T49 存在标记源实验。 | `Keyed-Presence-T40-T49-V1` | 远端已移除；见本地归档 |
| 其他实验 | `dev/media-channel-retention-v1` | 对既存 Window-State-MSE 轨迹做终态到媒体各层存留诊断。 | `Media-Channel-Retention-V1` | 远端已移除；见本地归档 |
| 历史保存 | `dev/public-statistic-control` | 二维公共亮度统计、完整尾程梯度、终端反馈与显存/反向诊断。 | `Public-Statistic-Control` | 远端已移除；见本地归档 |
| 历史保存 | `dev/real-video-public-observation` | Stage-1 帧差加权质心、固定镜头单主体公共观测与关系开发。 | `Stage1-Public-Observation` | 远端已移除；见本地归档 |
| 其他实验 | `dev/rgb-dct-b2-terminal-receiver-v1` | RGB-DCT B2 终端接收器梯度与完整/掩码下降诊断。 | `RGB-DCT-B2-Terminal-Receiver-V1` | 远端已移除；见本地归档 |
| 其他实验 | `dev/rgb-dct-group-consistency-h0-h1-v1` | RGB-DCT 时间组一致性统计的独立 H0/H1 功能检验。 | `RGB-DCT-Group-Consistency-H0-H1-V1` | 远端已移除；见本地归档 |
| 其他实验 | `dev/rgb-dct-h0-h1-functional-v1` | RGB-DCT T49 的独立 OFF 校准与盲 H0/H1 功能检验。 | `RGB-DCT-H0-H1-Functional-V1` | 远端已移除；见本地归档 |
| 其他实验 | `dev/rgb-dct-local-gradient-v1` | RGB-DCT 局部 VAE 梯度候选、checkpoint 与终端比较修复。 | `RGB-DCT-Local-Gradient-V1` | 远端已移除；见本地归档 |
| 其他实验 | `dev/rgb-dct-multistep-comparison-v1` | 同源 OFF / SINGLE49 / SINGLE46 / MULTI44_46 写入时刻与多步比较。 | `RGB-DCT-Multistep-Comparison-V1` | 远端已移除；见本地归档 |
| 其他实验 | `dev/rgb-dct-structured-terminal-feedback-v1` | 固定三维时间分箱方向、真实尾程响应的结构化终端反馈。 | `RGB-DCT-Structured-Terminal-Feedback-V1` | 远端已移除；见本地归档 |
| 其他实验 | `dev/rgb-dct-temporal-balanced-v1` | 全帧 RGB-DCT 时间平衡载体、连续分数与 T49 开发候选。 | `RGB-DCT-Temporal-Balanced-V1` | 远端已移除；见本地归档 |
| 其他实验 | `dev/rgb-dct-terminal-gradient-v1` | RGB-DCT 完整终端梯度与 T46 自由尾程写入候选。 | `RGB-DCT-Terminal-Gradient-V1` | 远端已移除；见本地归档 |
| 历史保存 | `dev/sc-sstw-core-integration` | 早期四位载荷、44/46 控制、媒体攻击与盲同步的集成候选历史。 | `SC-SSTW-Core-Integration` | 第一阶段已移除 |
| 其他实验 | `dev/trajectory-attribution-v1` | 轨迹载荷归属判定、不确定补充与对应真实运行审计记录。 | `Trajectory-Attribution-V1` | 远端已移除；见本地归档 |
| 历史保存 | `dev/trajectory-payload-framewise-sync-v1` | trajectory payload、framewise M05、对齐/单删与 conditional-joint 的开发来源；仅已采用部分精选进入 main。 | `Trajectory-Payload-Framewise-Sync-V1` | 远端已移除；见本地归档 |
| 历史保存 | `dev/video-inversion` | Wan 初始噪声反演载体、盲同步及其开发诊断历史。 | `Video-Inversion` | 远端已移除；见本地归档 |
| 其他实验 | `dev/video-trajectory-blind-validation-v1` | 连续约束求解器及 OLD8、局部 Fourier/时间同步等盲接收开发记录。 | `Video-Trajectory-Blind-Validation-V1` | 远端已移除；见本地归档 |
| 其他实验 | `dev/videoseal-receiver-diagnostic-v1` | CPU VideoSeal 原始 logit 的诊断归约器。 | `VideoSeal-Receiver-Diagnostic-V1` | 无 live 分支 |
| 其他实验 | `dev/window-state-mse-v1` | legacy 与 window-state MSE 的同源配对写入/媒体实验。 | `Window-State-MSE-V1` | 远端已移除；见本地归档 |
| 主 checkout 保护 | `c2a-2a-colab-preparation` | 早期 C2A、合成 AISB/校准/同步机制及 state-clock Colab 准备记录。 | `alive/SC-SSTW`（活跃，非 main） | [保留](https://github.com/RICHAAARC/SC-SSTW/tree/c2a-2a-colab-preparation) |
| 历史保存 | `release/conditional-joint-v1-main-20261008` | conditional-joint V1 的精选 main 发布候选快照。 | 原无工作树；bundle 保存提交历史 | 无 live 分支 |

## 使用这些记录

活跃工作区继续承担 main、A、B 与主 checkout 的现有工作；历史文件可直接在归档树中查阅，当前四个保留入口继续可用；旧历史Notebook分支链接不再维持。本次整理不重新运行旧实验，不改写其固定分母、失败、未决或证据上限。

源码、配置、环境和来源信息只用于定位；恢复、运行、结果读取与交付不增加摘要、manifest、精确版本或 Git 状态相等门禁。整理过程与保存范围以完成报告及归档说明为定位依据。
