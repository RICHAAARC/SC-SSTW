# 研究状态核定（2026-10-09）

本页是项目当前状态入口，研究定位为**基于轨迹采样嵌入思想的视频水印方法**。GROW 是具体实现来源，不是整体方法名；保留[作者仓库](https://github.com/luopengchen/GROW)、实现基线 `6aa69a9c5d4a9e75df457fcca8dfc71b64a6b870` 与 [CVPR 2026 论文](https://openaccess.thecvf.com/content/CVPR2026/papers/Luo_GROW_Watermark_Generation_with_Progressive_Guidance_for_Diffusion_Models_CVPR_2026_paper.pdf)。当前可执行参考采用 FFT.real，论文使用 DCT，两者差异见 [GROW 固定参考](grow_video_reference_v1.md)。

## 分支与证据边界

| 分支 | 2026-10-09 核定实现基线 | 包含内容 | 不包含／不得推断 |
|---|---|---|---|
| `main` | `41120d7ac8261b24b950fff38886de0439fdaf88` | GROW 参考、M05、固定 offset/phase、single deletion、conditional joint V1 | 不含 Attribution V1 或 uncertainty follow-up 实现；引用研究分支证据不等于代码已合入 main |
| `dev/trajectory-attribution-v1` | `c8dda3ef18bd9df3af8b677d1befeb5e4489a1de` | 包含 main 同源机制，并发布 Attribution V1 与 uncertainty follow-up V1 | 两个确认源与后续复用输入不建立总体 FPR、总体泛化、未知攻击鲁棒性或科学 PASS |

这里的 SHA 是核定实现基线，不表示文档修订后永远等于分支 HEAD。历史 C2A checkout 与旧路线只作历史证据，不能替代上述两个发布分支的当前状态。

## 实现、运行、审计、发布与 main 分开记录

| 对象 | 已实现 | 已真实运行 | 已审计 | 已发布 | 已进入 main |
|---|---|---|---|---|---|
| Conditional joint V1 | 是，有限混合链 | 是，原 source `ac111d0…` 的单个已见源 | 是 | 是，main source `9054f67…` / Notebook `5145a3e…` | 是；集成不等于在 main 新跑模型 |
| Attribution V1 | 是 | 是，DEV 后两个确认源 | 是，2026-10-08 | 是，研究 source `cd5e212…` / Notebook `e1a2017…` | 否 |
| Uncertainty follow-up | 是，冻结规则不变 | 是，复用两个确认源的 22/22 查询 | 是，2026-10-09，**完整但未覆盖** | 是，研究 source `1fb1b129…` / Notebook `c8dda3e…` | 否 |

来源身份与收据见[机器证据索引](evidence/temporal_mechanisms.json)。已发布不等于三态真实验证完成；本轮文档更新也不启动后续实验。

## 要求—实现—结果—审计—分支—缺口

| 要求 | 当前实现 | 真实结果 | 审计 | 分支／版本 | 缺口 |
|---|---|---|---|---|---|
| R01 · §I 两条时间轴及同载体连接 | 已区分生成采样索引与视频帧索引；当前载荷链和 terminal sync 仍是分层组合 | 有限链路各自可读，尚非同载体统一时间编码 | conditional joint 审计 | main 已集成；原运行 `ac111d0…`；research 复用 | **部分满足**；共同载体与端到端对应未闭合 |
| R02 · §II/§VIII 多步生成轨迹载荷写入 | GROW 在 25–49 共 25 个真实生成步写 payload；46 个 latent 时刻重复同一 8 bit/channel | 固定真实链经 VAE/MP4 可读 | GROW fixed-run evidence | main/research GROW reference | **限定已满足**；不证明同步或多步必要性 |
| R03 · §II 局部 terminal 状态信号 | M05 使用 4 帧 × 4×4 patch、key/source-tubelet 序列 | 固定 M05 与后续接收链有真实正证据 | M05、conditional joint | main/research | **部分满足**；terminal 写入不等于生成轨迹状态编码 |
| R04 · §II/§III 空间局部身份载荷 | 当前 payload 在完整 40×64 latent FFT 空间读写，reader 读取整幅频率证据 | 没有空间局部身份载荷结果 | 源码/协议核对 | main/research | **机制待补、验证待补** |
| R05 · §I/§II/§VII 生成时 state+payload 共同嵌入 | conditional writer 为 payload-only、`pilot_gradient=0`；M05 在 terminal framewise latent 另写 | 混合链恢复已报告，但不是共同生成嵌入 | conditional joint report | `ac111d0…` | **机制待补、验证待补**；不得用串联结果替代 |
| R06 · §III 公开盲接收与逐窗软证据 | receiver 只用 RGB、公开 N、key/claim；有局部 q/support fraction 与逐 time-bit votes | 固定协议内搜索、读出与归因完成 | conditional/attribution 两次 root audit | main + research | **有限范围已满足**；平台额外转码与空间局部 payload 未验证 |
| R07 · §IV offset、crop、single deletion | H0/H1 支持 offset 与一次 single skip；校正时插入前帧只是名义索引占位 | C/D 路径对应恢复；D GLOBAL 88/177、PATH 177/177 | fixed deletion、conditional audit | main/research | **部分满足**；不恢复缺失画面，也不代表一般插入攻击 |
| R08 · §IV/§V repeat-stay 与声明范围内速度变化 | 当前 H0/H1 没有 repeat-stay 或 speed-transition 模型 | 无同链真实结果 | 设计/源码核对 | 当前 main / Attribution 无此路径模型；历史代理不自动继承 | **机制待补、验证待补** |
| R09 · §IV 任意插入／插值的独立观测模型 | 正式设计保留为独立观测模型要求 | 未实现、未验证、未启动 | 2026-09-30 正式设计 | 当前两分支无 | **机制待补、验证待补**；不自动扩大当前工作 |
| R10 · §V 时间／状态相关片段载荷 | 当前逐 time-bit 保存与 latent 索引不是独立 4 帧块或同源感受野编码 | 无状态相关片段 payload 真实结果 | 设计/源码核对 | main/research | **机制待补、验证待补** |
| R11 · §V 同步对恢复的实际作用 | blind sync 后才 correction/payload；保留 RAW/GLOBAL/PATH | conditional 中 22 个 K0（含 RAW/BASELINE）均已 0 bit error | report SHA `69f7c2bd…` | `ac111d0…` | **部分满足**；不能据此声称 BER 收益或同步必要性 |
| R12 · §V 片段序列聚合、重叠与冲突 | received-map hash/action 等价折叠、same-action paths 保留；cache 仅按完整 input/map/key 去重 | 固定路径与三态证据可审计 | attribution core + conditional core | research / main shared core | **部分满足**；没有任意重叠片段融合或消息冲突处理 |
| R13 · §IV/§V/§VIII 归因与 ACCEPT/REJECT/UNCERTAIN | 研究分支实现 action 折叠、局部 margin、identity evidence 与三态规则 | C1/C2 confirmation128 为 16 ACCEPT / 112 REJECT / 0 UNCERTAIN / 0 false-claim source | root audit SHA `cc3df988…` | research source `cd5e212…` | ACCEPT/REJECT 有固定有限实证；真实 uncertainty 分支未覆盖 |
| R14 · §IV/§V 不确定分支真实覆盖 | 冻结规则复用 C1/C2 saved RGB，10 个 primary probe 同时作为两类分母 | 22 行为 16 ACCEPT / 6 REJECT / 0 UNCERTAIN；sync 与 weak-identity 各 0/10，技术完整；0 新独立源；alias 无真实 tie | root audit SHA `8ef088e6…` | research source `1fb1b129…` | correct abstention **NOT_ESTABLISHED**；固定 roster 未观察到目标分支 |
| R15 · §VI 仿射不变同步 | 正式设计列为可选扩展，旧代理构造仅供参考 | 没有向当前真实链继承的证据 | 历史/设计核对 | 历史分支 | 可选，不是当前闭合前提 |
| R16 · §VIII 完整同版方法链 | 已有 generation payload、terminal sync、媒体、blind search、payload/归因的有限组合 | conditional 固定案例与 attribution 固定确认完成 | 两个 root audit + conditional audit | main/research 分层发布 | **部分满足**；不能宣称原研究或总体科学结论完成 |

章节编号对应外层正式设计《研究内容二_时空一致水印机制_开题设计与进度核对_2026-09-30.md》，第一至第八章的目标未改。第九章保留原完成条件和历史表，再追加当前核定。无需为匹配当前实现缩减原目标；将来若调整目标，须另行提出具体差异供用户决定。

源码定位（核定实现基线）：R02/R04/R10 对应 `main/tube_state/grow_video_reference.py:62–94` 与 `main/tube_state/payload_reader.py:8–16`；R03 对应 `main/tube_state/video_trajectory_payload_framewise_sync_v1.py:130–216`；R05 对应 `runtime/wan/video_trajectory_conditional_joint_v1.py:70–117`；R07 对应 `main/tube_state/video_trajectory_internal_single_deletion_v1.py:9–26`；该文件的有限 H0/H1 语法也是 R08 缺项判断的依据，R08/R09 在当前两分支无对应机制实现入口；R06 对应 `main/tube_state/video_trajectory_receiver_estimated_align_v1.py:92–113`；R12 的精确缓存对应 `main/tube_state/video_trajectory_conditional_joint_v1.py:63–73`。R13/R14 的完整 action 折叠及三态规则仅在研究分支 `main/tube_state/video_trajectory_attribution_v1.py:223–251,423–458`。这些位置用于区分机制有无，不重新审计原始实验。

## 已核定结果与不可覆盖的反例

- Conditional joint V1 只覆盖一个已见开发源。主路径 K0 8/8 正确且 32/32 bits，但对应 7 份物理输入；44 logical 为 40 blind + 4 oracle，实际 21 encode / 32 key read。全部 22 个 K0（含 RAW/BASELINE）本来就是零错，不能推出 BER 改善。writer 是 25 次 payload control、pilot 0，terminal M05 另写。
- Attribution V1 的 DEV64 只用于一次冻结规则，不是未见验证。C1/C2 是相对 DEV 的两个预注册确认 source cluster；confirmation128 为 16 ACCEPT、112 REJECT、0 UNCERTAIN，且技术缺失为 0。128 个查询不是 128 个独立来源。它只测试一个 wrong key、一个 alternative message 与固定时间编辑 roster。
- Uncertainty follow-up 复用上述 C1/C2 post-codec RGB，因此新增独立源数为 0。22 行中 16 ACCEPT、6 REJECT、0 UNCERTAIN、0 false claim；10 个共享 primary probe 对两类不确定覆盖均为 0/10 complete-but-not-covered。四个 alias control 都是 `top_paths=1`、`actions=1`，没有自然 tie。正负回归对照分别 2/2、6/6 保持；22/22 完整、技术缺失 0、未知归属 0。技术异常不能充当方法不确定覆盖。
- Stage1 反例保持有效：source `ab60d457…` 的 177 协议 positives 2/2，null 中 1/6 ACCEPT（4 REJECT、1 UNCERTAIN）；89 positives 2/2，null 0/6 ACCEPT，但六个 null 全部 UNCERTAIN。后续固定研究证据不能把它重标为新的未见反例修复。旧反例进入已知 DEV 校准：Q030 的 M=0.022594309085612687，小于冻结 tau_M=0.02283275070993452；m=0.003530171359717223 等于 tau_m，严格大于条件也不成立。这是已知 DEV 校准约束，不是新增独立反例验证。
- OFF 是无 payload、无 M05；A_P1 是含 OKOK payload、无 M05 的 sync-null 对照。A_P1 不是 no-watermark 输入。

因此，固定研究协议已经验证其特定 OFF、wrong-key、other-identity 与正例控制；但 main 本身不含 Attribution 实现，且任意视频存在检测、一般错误钥拒绝、总体 FPR、独立源泛化、真实不确定分支覆盖、payload-off 质量成本与不可感知性仍未建立。

## 审计入口

- Conditional joint: `/home/richar/projects/Video-WM/diagnostics/trajectory-conditional-joint-real-run-audit-20261008/report.md`，report SHA256 `69f7c2bd27366ee37b13e598567b1a318e0fda6a54d083518783d2e82b4a86c4`。
- Attribution V1: `/home/richar/projects/Video-WM/diagnostics/trajectory-attribution-v1-real-run-audit-20261008/root_raw_audit.json`，SHA256 `cc3df9882dbd6f5dad6db4d1419819523807ff363588cd11bf5639207fc73ff0`。
- Uncertainty follow-up: `/home/richar/projects/Video-WM/diagnostics/trajectory-attribution-uncertainty-v1-real-run-audit-20261009/root_raw_audit.json`，SHA256 `8ef088e62696b885482b4325299df1f59641d42668e6db859f0fd009ffe295ed`。

## 文档覆盖与历史边界

本轮检查外层入口、正式设计、两个当前 checkout 的 README、当前方法、开发/机制证据索引、机器状态、归属/不确定说明。对条件联合恢复、归属及不确定的原协议只添加日期化后续审计说明，原规则与发布阶段正文保留。

`archive/`、其他历史 `worktrees/`、`release-candidates/`、`diagnostics/*/frozen-*`、`session/` 和项目 `memory/` 按目录/入口核定为历史或不可变证据，未逐篇重审历史正文；`framework/` 是模板，不是当前方法实现。`alive/SC-SSTW` 是有既存未提交修改的旧 C2A checkout，本轮未改。Notebook、冻结配置、原始结果、既有审计与发布收据不改。旧“等待运行”发布收据表示当时状态，由本页与最新审计解释其后续完成情况。

此次只核定文档，没有新增实验、阈值、载荷或路径机制，没有合并分支，也未启动贡献对照或研究扩展。
