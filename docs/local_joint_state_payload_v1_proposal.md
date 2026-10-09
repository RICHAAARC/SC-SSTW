# Local Joint State + Payload V1 单一候选

状态：**PROPOSED_NOT_ADOPTED**

日期：2026-10-09

边界：本文只把一个可否决候选具体化，未实现候选数学，未冻结实验输入，也不授权模型、GPU、VAE、codec、Drive 或媒体执行。即使用户采纳方法语义，也不自动产生真实执行授权。

## 机制假设与贡献上限

OLD8、LOWBAND16、CONTRAST、DWELL4 直接约束 Wan 局部 latent 中有符号线性 Fourier 系数，接收又经过 VAE 重编码。本候选改用 received RGB 亮度上的成对局部 DCT 能量比，检验符号、相位和重编码敏感性是否降低。它不预先声称 codec 可靠、posterior-difference 回注有效或同步已成立。

RGB-DCT 的 RGB 直接读取、`E(X+A)-E(X)` VAE lift 与 RGB 引导已有历史先例。本候选只把“局部成对能量比、局部 state、不同 payload 分片、同一局部支持、生成时联合控制”作为新的待验证组合，不声称首次使用 RGB 或 RGB-DCT 引导。

## 待采纳的固定构造

### 视频、窗口与局部支持

- 视频固定为 181×320×512。帧 `[1,177)` 分成 22 个互不重叠的 8 帧 segment；帧 0 和 `[177,181)` 不承载，但仍作为显式边界观测状态保存。
- 四个 RGB ROI 坐标统一采用半开区间 `(y0,y1,x0,x1)`：`(64,128,96,160)`、`(64,128,352,416)`、`(224,288,96,160)`、`(224,288,352,416)`。每个 ROI 分成 64 个 8×8 tile。
- 对每个 tile 取 DCT-II `norm="ortho"`。亮度特征定义为 `Y=0.299R+0.587G+0.114B`；这只是公开特征，不声称是 codec 的物理 Y 通道。
- DCT 频率按 `(u+v,u,v)` 排序，去 DC 后取前 32 个；按 key/ROI 的固定域分离顺序组成 16 对。8 对给 state，8 对给 payload。没有频带、窗口、方向或幅度扫描。

### 软观测、state 与 payload 分片

对 segment `j`、局部块 `b`、系数对 `k`，保存

```text
q[j,b,k] = (E_plus - E_minus) / (E_plus + E_minus)
E_plus  = sum c_plus^2
E_minus = sum c_minus^2
```

求和跨该 segment 的 8 帧和 ROI 内全部 tile。若 `E_plus+E_minus=0`，记录缺支持；不得加新能量或用真值补值。每窗保留原始 `E_plus/E_minus`、完整 q 向量、几何支持及可用计数，不提前压成单个分数。

state 每个 ROI 有 8 chip，四 ROI 共 32 chip。22 个位置拟从 OLD8 已有 balanced RM(1,5) 的 62 个词中按 key 固定取 22 个，使载体变化可与既有码族区分；该选择仍属本候选整体的一部分，尚未采用。

32-bit message 明确拆成四个 8-bit fragment `F0..F3`。segment `j` 只承载 `F[j mod 4]`：该 segment 内每个 ROI 的 8 个 payload 对各对应一个 fragment bit，四 ROI 仅作冗余。22 个 segment 对四个 fragment 的重复数为 6/6/5/5；有效信息仍是 32 bits，不是 22 条独立消息，也不是 704-bit 容量。这里没有 time mask。

接收时依据 state/path 把观测放回四个 fragment 槽再重组。片段缺失保留为擦除；`path mod 4` 等价、message 自身重复或局部碰撞造成不可区分时必须保留多解，不能宣称任意错位都会失败。正式验证应预注册具体 message 并保留碰撞，不能按结果挑选有利 message。首先验证 state 与四分片 payload 在同一载体共同存留；之后才比较正确、错误与无对齐下的分片重组。

### 写端与 Flow 接线

提案在生成步 25..49 的 conditional clean estimate `x0=z-sigma*v_cond` 上执行以下单一控制：

1. VAE decode 得 `X`。对每个 tile/系数对保持原总能量 `E`，按 `E(1±rho*t)/2` 重分配，`t` 为 state 或 payload 的 ±1 目标；保留原系数符号，精确零的符号由公开 key 域确定。
2. 逆 DCT 差同量加到 RGB 三通道，clip 到 `[0,1]` 得 `X'`。同一冻结 Wan VAE 使用 posterior mode，取 `E(X')-E(X)`。
3. latent 硬支持为 all channels、`T[1,45)`、原 OLD8 四个 8×8 latent ROI。它只是 control support，不声称精确等于 VAE 感受野。
4. state/payload 经上述非线性局部构造形成一个 joint direction，再作一次 shrink-only 联合 L2 cap。`rho=0.5` 与每步 cap=1 都是新的科学选择，只有整体候选被采纳后才成为固定语义；state/payload 共享 cap，不假定 50/50 能量。cap=1 只限 conditional-clean delta L2，不能叫旧 `R*` 的等价预算，也不能叫 native response、终态质量或 CFG 后预算。CFG=5 的实际 velocity 影响另记。
5. Flow 映射为 `c'=c-d/sigma`，继续同一完整 scheduler；不做 terminal 像素添加，不对 denoiser 或 tail 回放反传，不强制扩幅、不回退、不扫描。

每步保留 raw/capped joint L2、conditional-clean 实际 delta、CFG velocity delta 与 native 实际更新；后评记录 RGB clip、ROI 外改变、终端层差异和质量。没有质量阈值。本轮公共 adapter 只接受外部 provider 给出的最终 `joint_delta`；可选 state/payload delta 只是诊断，adapter 不假设它们可加、各占比例或构成真实分解。

单个新 joint 臂的候选成本是 25 次完整 181 帧 clean decode 和 50 次 posterior encode，另加最终 terminal/float RGB 层解码；这不包括 baseline 臂。内存与端到端时延未经验证，因此资源或接线失败只能记为工程不足。

## 公开接收、路径和等价类边界

receiver 只接 received RGB、key 与 public protocol，直接从 RGB 窗口提取 state q 和 payload 软证据，不经 Wan VAE 重编码。state 用于候选路径；payload 不参与路径选择。逐窗独立保留 state/payload 的 `SCORED/MISSING/FAILED` 状态，使局部 partial observation 不被整窗抹掉。

FULL 诊断先枚举所有 8-frame 起点相位 0..7，并显式保存有限帧边界；公共支持应使所有相位保有固定分母。state-only 相关分数可作为后续盲定位输入，但本轮不要求每窗唯一峰，也未采用窗口边界、路径接受阈值或完整盲 decoder。任何 oracle 后评必须和 blind path 分开，不能以真码相关方向替代盲恢复。

未来有界 DP 所需的 crop/delete/repeat/speed 转移、接受阈值、candidate action 和 equivalence 规则须另行冻结。当前公共接口只预留 boundary/path/equivalence 容器，不实施 decoder。等价路径仍可恢复哪些 fragment、哪些槽必须擦除或拒绝，需由将来采用的协议明确。

## 与相关历史机制的差异

| 历史机制 | 已有事实 | 本候选的差异与不得继承的结论 |
|---|---|---|
| RGB-DCT T49 VAE lift | 已有 RGB 直接 score、显式载体和 `E(X+A)-E(X)` posterior-mode lift | 这些不是本候选创新；本候选用局部 paired-energy q、state 与四分片 payload 共支持、多步 conditional-x0 控制 |
| RGB-DCT MULTI/LOCAL/TERMINAL/B2 | signed DCT 差与固定 presence/time-code 目标；SINGLE46/MULTI、LOCAL、TERMINAL、B2 的 MP4 C 均低于同批 SINGLE49 C30/C30，且早期 float 也可失败 | 不得把旧失败全部归为 codec；paired energy、分片重组与 joint lift 仍需独立验证 |
| Structured Terminal Feedback | 无梯度三维固定 lift，但用完整真实 tail 有限响应、正组保护、候选 lattice 与 forward accept；后续真实运行中 18 个候选全丢正组，10 个代理合格候选均被真实 tail 拒绝 | 本候选没有 tail search、all-candidate margin 或 forward accept；conditional-x0 posterior difference 与一次既定 cap 尚未解决旧传播问题 |
| zero-mean terminal STE / composite gradient | 4×4 Walsh、8-state、1392 维支持；raw420 COMPOSITE 虽 aggregate loss 下降，真实 gap 从 `-5.4705554078e-7` 变差到 `-7.9898438557e-7` | 本候选是非梯度 RGB paired-energy/posterior difference 与局部 joint 多步；无 STE，也不能从旧 objective 下降推断媒体改善 |

历史边界来源：[方法机制审计 §2.1、§3.2 及 §6 后续实跑补充](../../../diagnostics/method-mechanism-audit-20260928/assessment.md)、[T49 VAE lift 协议](../../RGB-DCT-Structured-Terminal-Feedback-V1/docs/rgb_dct_t49_vae_lift_protocol_20260923.md)、[多步比较协议](../../RGB-DCT-Structured-Terminal-Feedback-V1/docs/rgb_dct_multistep_comparison_protocol_20260924.md)、[terminal gradient 协议](../../RGB-DCT-Structured-Terminal-Feedback-V1/docs/rgb_dct_terminal_gradient_v1_protocol.md)、[B2 协议](../../Video-Trajectory-Blind-Validation-V1/docs/rgb_dct_b2_terminal_receiver_protocol.md)、[Structured 协议](../../RGB-DCT-Structured-Terminal-Feedback-V1/docs/rgb_dct_structured_terminal_feedback_v1_protocol.md)及 [same-raster/zero-mean 设计复核](../../../diagnostics/local-fourier-rm-same-raster-real-run-audit-20261003/design_review.md)。这些历史只界定差异，不授权复跑或迁移阈值。

## 分阶段证据与停止条件

第一阶段只检验同一新构造的多步 state+payload 是否共同穿过媒体。层次分别记录 terminal latent（后评）、raw VAE float RGB、RGB8、同一 RGB8 经 codec 得到的 MP4；旧 `DIRECT_RGB8` 不能称为纯 VAE 通道。控制可按具体问题安排 OFF、local payload-only、current hybrid 与 new joint，但不跑大笛卡尔积，也不把不同载体臂称为 budget matched。实际 source/seed/key/codec/message/阈值须另行冻结。

早期媒体共同存留不等于正式四分片机制完成。后续阶段才比较正确、错误、无对齐的不同 fragment 重组，再进入预先声明的有界攻击与聚合。

候选的判别预测是：真实 25-step 记录生效；float→RGB8→MP4 后，state q 的正确路径或预定义等价集合仍可区分，同时局部 payload 四分片可重组。两者缺一都不闭合。仅排名改善、旧重复载荷成功或 CPU 理想信号不够。

- float 层即失败：是该构造或回注链的有效负证据。
- float 成立而 RGB8/MP4 失败：是媒体存留缺口，但不能把原因归给单个 codec 因子。
- 资源、OOM、接线或 instrumentation 失败：工程证据不足，不是构造负结果。
- 一个冻结候选得到有效负结果后停止该构造；不改 `rho`、cap、频率或支持后重跑。无实际数据前不预设科学 PASS 阈值。

## 用户待决定

是否采纳上述完整单一 **carrier + 四分片时间组织 + posterior-difference lift + `rho/cap` 口径**。这是方法语义决定，不是运行授权；若不采纳，需要给出要修改的具体构造点。

## 本地交付与验证收据

基线为 `527c4800292c296222c2c0533809eccb82a29a65`，分支 `dev/local-joint-state-payload-v1`。候选数学未进入 runtime；runtime 只提供 carrier-agnostic 外部注入和 received-only 逐窗记录接口。

变更文件：

- `main/tube_state/local_joint_state_payload_v1.py`
- `runtime/wan/local_joint_state_payload_v1.py`
- `runtime/wan/grow_video_reference.py`
- `tests/test_local_joint_state_payload_v1.py`
- `docs/local_joint_state_payload_v1_proposal.md`

验证环境为 `/home/richar/projects/CEG-WM/alive/CEG-WM/.venv/bin/python`，仅复用已安装的 pytest/torch；命令设置 `PYTHONDONTWRITEBYTECODE=1` 与 `CUDA_VISIBLE_DEVICES=''`，未安装依赖，未调用 CUDA、模型、VAE、codec 或媒体。

- `tests/test_local_joint_state_payload_v1.py`：`6 passed in 1.20s`。
- `tests/test_grow_video_reference.py`：`15 passed in 18.28s`。
- 四个变更 Python 文件 `py_compile`：通过。
- `git diff --check`：通过。

这些只证明公共接口、CPU tensor 接线、默认 GROW 旧路径回归和静态语法，不证明候选 carrier、媒体存留、盲路径、四分片恢复、资源可行性或任何科学 PASS。
