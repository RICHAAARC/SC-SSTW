# Local Joint State + Payload V1 单一候选

状态：**CONSTRUCTION_AUTHORIZED_PARAMETERS_PENDING**

日期：2026-10-09

边界：用户已授权在本地分支实现本文的单一 carrier、四分片时间组织和 posterior-difference 联合构造；`rho` 与联合 L2 cap 的具体数值仍待单独决定。当前授权不冻结实验输入，也不授权模型、GPU、真实 VAE、codec、Drive 或媒体执行。

## 机制假设与贡献上限

OLD8、LOWBAND16、CONTRAST、DWELL4 直接约束 Wan 局部 latent 中有符号线性 Fourier 系数，接收又经过 VAE 重编码。本候选改用 received RGB 亮度上的成对局部 DCT 能量比，检验符号、相位和重编码敏感性是否降低。它不预先声称 codec 可靠、posterior-difference 回注有效或同步已成立。

RGB-DCT 的 RGB 直接读取、`E(X+A)-E(X)` VAE lift 与 RGB 引导已有历史先例。本候选只把“局部成对能量比、局部 state、不同 payload 分片、同一局部支持、生成时联合控制”作为新的待验证组合，不声称首次使用 RGB 或 RGB-DCT 引导。

## 已授权实现的固定构造

### 视频、窗口与局部支持

- 视频固定为 181×320×512。帧 `[1,177)` 分成 22 个互不重叠的 8 帧 segment；帧 0 和 `[177,181)` 不承载，但仍作为显式边界观测状态保存。
- 四个 RGB ROI 坐标统一采用半开区间 `(y0,y1,x0,x1)`：`(64,128,96,160)`、`(64,128,352,416)`、`(224,288,96,160)`、`(224,288,352,416)`。每个 ROI 分成 64 个 8×8 tile。
- 对每个 tile 取 DCT-II `norm="ortho"`。亮度特征定义为 `Y=0.299R+0.587G+0.114B`；这只是公开特征，不声称是 codec 的物理 Y 通道。
- DCT 频率按 `(u+v,u,v)` 排序，去 DC 后取前 32 个。令该顺序中的 0 基编号为 `n`，对应坐标为 `coords[n]`。没有频带、窗口、方向或幅度扫描。

公共 key 是原样 Unicode 字符串，不作正规化。以下固定派生统一定义为：

```text
H(domain,key,*indices) = SHA256(
  json.dumps([domain,key,*indices], ensure_ascii=False,
             separators=(',',':')).encode('utf-8')
).digest()
```

所有 `indices` 必须是非负整数。排序时以 digest 原始 bytes 的字典序为主键，以原整数为 tie-break。该派生只用于固定候选码，不宣称密码学安全或认证强度。

对 ROI `b`，令 `O=sorted(n,key=(H('LJSP1/PAIR',key,b,n),n))`。pair `k` 的 plus/minus 分别是 `coords[O[2k]]` 与 `coords[O[2k+1]]`；`k=0..7` 给 state，`k=8..15` 给 payload。state `S[j,8b+k]` 对应 ROI `b` 的 state pair `k`，payload pair `8+i` 对应 `F[j mod 4][i]`。

### 软观测、state 与 payload 分片

对 segment `j`、局部块 `b`、系数对 `k`，保存

```text
q[j,b,k] = (E_plus - E_minus) / (E_plus + E_minus)
E_plus  = sum c_plus^2
E_minus = sum c_minus^2
```

求和跨该 segment 的 8 帧和 ROI 内全部 tile。若 `E_plus+E_minus=0`，记录缺支持；不得加新能量或用真值补值。每窗保留原始 `E_plus/E_minus`、完整 q 向量、几何支持及可用计数，不提前压成单个分数。

state 每个 ROI 有 8 chip，四 ROI 共 32 chip。balanced RM 词的候选整数为 `a=0..63` 且 `(a&31)!=0`。按 `(H('LJSP1/STATE_WORD',key,a),a)` 无放回取前 22 个作为 `a_j`。再把列 `x=0..31` 按 `(H('LJSP1/STATE_CHIP',key,x),x)` 排为 `p_i`，定义 `S[j,i]=(-1)^((a_j>>5)+popcount((a_j&31)&p_i))`。它保留 OLD8 balanced RM(1,5) 码族以便区分载体变化；本构造已有独立显式 API，但不替换既有 runtime 默认协议。

message 必须严格为 4 bytes，不接受字符串自动编码。按 byte 顺序、每 byte MSB→LSB 展开为 32 bits `m`，令 `F_r[i]=m[8r+i]`，写入目标符号为 `t=2*bit-1`。segment `j` 只承载 `F[j mod 4]`：该 segment 内每个 ROI 的 8 个 payload 对各对应一个 fragment bit，四 ROI 仅作冗余。22 个 segment 对四个 fragment 的重复数为 6/6/5/5；有效信息仍是 32 bits，不是 22 条独立消息，也不是 704-bit 容量。这里没有 time mask，也不得按结果挑选 message。

接收时依据 state/path 把观测放回四个 fragment 槽再重组。片段缺失保留为擦除；`path mod 4` 等价、message 自身重复或局部碰撞造成不可区分时必须保留多解，不能宣称任意错位都会失败。正式验证应预注册具体 message 并保留碰撞，不能按结果挑选有利 message。首先验证 state 与四分片 payload 在同一载体共同存留；之后才比较正确、错误与无对齐下的分片重组。

### 写端与 Flow 接线

提案在生成步 25..49 的 conditional clean estimate `x0=z-sigma*v_cond` 上执行以下单一控制：

1. VAE decode 得 `X`。对每个输出帧 `f`、ROI tile `(ty,tx)` 和 pair `k`，只取该处两个标量 `c+`,`c-`，独立定义 `E_loc=c+²+c-²`；这不同于 receiver 跨 frames/tiles 汇总的 pooled energy。令 `c'+=sign_plus*sqrt(E_loc*(1+rho*t)/2)`、`c'-=sign_minus*sqrt(E_loc*(1-rho*t)/2)`。原系数非零时保留其 sign；若某系数为零但 `E_loc>0`，令 `sign=2*(H('LJSP1/ZERO_SIGN',key,f,b,ty,tx,k,side)[0]&1)-1`，plus/minus 的 `side` 分别为 0/1。`E_loc=0` 时两者仍为 0，不作组内能量分配或随机回退方向。
2. writer 入口明确要求 Wan adapter 提供的 FP32 RGB，拒绝其他浮点 dtype，避免在 ROI 外发生隐式全域 cast。DCT、系数修改与逆 DCT 使用 FP64；逆 DCT 差同量加到 RGB 三通道后，先转 FP32，再按 FP32 preclip 值统计并 clip 到 `[0,1]` 得 `X'`。收据分别记录 FP64→FP32 rounding L2 与 FP32 clip adjustment L2。该数值口径不承诺跨硬件 bitwise 一致。
3. VAE 坐标严格复用 `runtime/wan/vae.py`：decode 前把 normalized `x0` 转为 raw `x0*latents_std+latents_mean`；RGB 输入使用 `[-1,1]`；posterior mode 得 raw `r` 后，以 `E_norm(X)=(r.float()-mean)/std` 回到 normalized 坐标。冻结 VAE 使用 FP32、`no_grad` 并在调用边界 clear cache。
4. joint raw 定义为 `M*(E_norm(X')-E_norm(X))`，以 FP32 形成后先施加 hard mask：all channels、`T[1,45)`、四个半开 latent 8×8 ROI `(8,16,12,20)`、`(8,16,44,52)`、`(28,36,12,20)`、`(28,36,44,52)`。该 mask 只是 control support，不声称精确等于 VAE 感受野。
5. 对 masked raw 以 FP64 跨全部 `B,C,T,H,W` reduce 得 L2 norm，令 `d=raw*min(1,cap/norm)`，再以 FP32 送入 Flow；`norm=0` 时 `d=0`。生产 API 要求调用者显式传入 `rho` 与 cap，不提供默认值或正式配置；只作 `0<=rho<=1`、`cap>=0` 的有限数学校验。历史提案中的 `rho=0.5`、每步 cap=1 尚未被用户采纳。state/payload 共享 cap，不假定 50/50 能量，不自动扩幅；该 cap 是 normalized conditional-clean 坐标的联合 L2 cap，不是旧 `R*`、native response、终态质量或 CFG 后预算。`rho=0` 会把有支持的 pair 改为等能量，一般不是 identity/OFF；`cap=0` 才使最终 latent operand 为零，但仍会执行已启用步的 decode/encode。
6. Flow 映射为 `c'=c-d/sigma`，继续同一完整 scheduler；不做 terminal 像素添加，不对 denoiser 或 tail 回放反传，不强制扩幅、不回退、不扫描。

每步分别保留 provider 给出的 nominal joint operand、FP32 实际 realized conditional-clean delta、FP32 realized CFG velocity delta，以及 scheduler 从本步输入到输出的 total native state update。最后一项不是控制的 counterfactual effect，也不是终态预算。posterior backend 分开累计 decode/encode 的 attempted 与 completed；backend callable、carrier apply、posterior 输出校验、raw difference 和 mask-cap 各 stage 的失败都保留 sampling step、具体 stage 和异常原因并原样抛出，不重复记录、不重试或回退。后评另记 RGB clip、ROI 外改变、终端层差异和质量；没有质量阈值。本轮公共 adapter 只应用外部 provider 给出的最终 `joint_delta`；可选 state/payload delta 只是诊断，adapter 不计算或声称二者构成 joint 分解。

单个新 joint 臂的候选成本是 25 次完整 181 帧 clean decode 和 50 次 posterior encode，另加最终 terminal/float RGB 层解码；这不包括 baseline 臂。当前 provider 只接收外部 backend，不加载模型、不选择 CUDA，也不管理模型驻留。既有 `generation.prepare_generation(load_vae=False)` 的成功方式只说明 transformer worker 不持有 VAE；`load_vae=True` 会让二者共存，旧离线 VAE 成功不能证明生成中交替 VAE 的资源可行性。

未来若另获真实执行授权，runner 应采用显式串行生命周期：每步完成 conditional/unconditional 后仅保留必要 state，按阶段管理 transformer/VAE 驻留，对完整 181 帧依次执行一次 decode 与两次 posterior encode，并及时释放临时 RGB、encoded tensor 和 cache，再回到下一生成步。25 个 VAE 阶段对应 25 次 transformer↔VAE 往返，按方向约 50 次驻留切换；初始化和最终解码另计。当前没有显存峰值或时延保证；资源失败保留为工程失败，不自动重跑、换 dtype 或切小时间 chunk。这是未来 runner 的资源计划，不是当前 provider 已实现的驻留能力。

## 公开接收、路径和等价类边界

receiver 只接 received RGB、key 与 public protocol，直接从 RGB 窗口提取 state q 和 payload 软证据，不经 Wan VAE 重编码。state 用于候选路径；payload 不参与路径选择。逐窗、逐 component 和逐 chip 独立保留 `SCORED/PARTIAL/MISSING/FAILED` 状态，使局部 partial observation 不被整窗抹掉。

FULL 原始目录机械枚举 `phase g=0..7`、`slot j=-1..22`，每行请求帧 `g+8*j+[0..7]`，每个 phase 固定 24×4 ROI，共 768 行。requested 坐标保留有符号值，实际 received 索引另存；实际索引必须是互异、非负且属于该行 requested 集合，否则该自定义行原样保留为 FAILED，绝不触发 Python 负索引或重复计数。短输入 `T<=181` 仍保留 768 行并逐窗标记 FULL/PARTIAL/MISSING，超过公开 T 的输入保留固定行并记录 FAILED。未攻击公共参考网格中的 writer segment 是 `phase=1,slot=0..21`，但 raw record 不写“正确 writer”标签，也不据此筛窗；额外边界行不是独立证据。state-only 相关分数可作为后续盲定位输入，但本轮不要求每窗唯一峰，也未采用窗口边界、路径接受阈值或完整盲 decoder。任何 oracle 后评必须和 blind path 分开，不能以真码相关方向替代盲恢复。

未来有界 DP 所需的 crop/delete/repeat/speed 转移、接受阈值、candidate action 和 equivalence 规则须另行冻结。当前公共接口只预留 boundary/path/equivalence 容器，不实施 decoder。等价路径仍可恢复哪些 fragment、哪些槽必须擦除或拒绝，需由将来采用的协议明确。

fragment helper 只按调用者显式 correspondence 路由原始 payload 条目。它逐 entry 保留 observation 自身的 `truth_used`，但 correspondence 是否使用真值标为 `unverified`，顶层 `truth_used=None`，不能把原始 observation 的盲性继承为 mapping 的盲性。输出分开保存完全无 route 的 fragment slot、每个 candidate/slot 的 `SCORED/PARTIAL/MISSING/FAILED` 原始条目计数，以及虽有 route 但全部 MISSING/FAILED 的 slot；PARTIAL 不投票、不聚合、不转硬 bit。

## 与相关历史机制的差异

| 历史机制 | 已有事实 | 本候选的差异与不得继承的结论 |
|---|---|---|
| RGB-DCT T49 VAE lift | 已有 RGB 直接 score、显式载体和 `E(X+A)-E(X)` posterior-mode lift | 这些不是本候选创新；本候选用局部 paired-energy q、state 与四分片 payload 共支持、多步 conditional-x0 控制 |
| RGB-DCT MULTI/LOCAL/TERMINAL/B2 | signed DCT 差与固定 presence/time-code 目标；SINGLE46/MULTI、LOCAL、TERMINAL、B2 的 MP4 C 均低于同批 SINGLE49 C30/C30，且早期 float 也可失败 | 不得把旧失败全部归为 codec；paired energy、分片重组与 joint lift 仍需独立验证 |
| Structured Terminal Feedback | 无梯度三维固定 lift，但用完整真实 tail 有限响应、正组保护、候选 lattice 与 forward accept；后续真实运行中 18 个候选全丢正组，10 个代理合格候选均被真实 tail 拒绝 | 本候选没有 tail search、all-candidate margin 或 forward accept；conditional-x0 posterior difference 与一次既定 cap 尚未解决旧传播问题 |
| zero-mean terminal STE / composite gradient | 4×4 Walsh、8-state、1392 维支持；raw420 COMPOSITE 虽 aggregate loss 下降，真实 gap 从 `-5.4705554078e-7` 变差到 `-7.9898438557e-7` | 本候选是非梯度 RGB paired-energy/posterior difference 与局部 joint 多步；无 STE，也不能从旧 objective 下降推断媒体改善 |

四个直接前序载体的已核定边界如下；rank 仅描述固定有限 catalog 中的位置，数值越小越靠前，不能跨 run 拼成一条成功链：

| 前序载体 | 原生终态 ABS/DIFF rank | 媒体层 ABS/DIFF rank | 本候选继承边界 |
|---|---:|---:|---|
| OLD8 | 1/1 | 同次旧臂 MP4 22/30 | 终态局部正证据保留；MP4 未闭合，[LOWBAND16 实跑复核](../../../diagnostics/local-fourier-rm-lowband-real-run-audit-20261003/design_review.md) |
| LOWBAND16 | 59/21 | MP4 60/82 | 失败已在终态出现，不能只归因 codec，[同一实跑复核](../../../diagnostics/local-fourier-rm-lowband-real-run-audit-20261003/design_review.md) |
| CONTRAST | 5/4 | MP4 118/106 | 终态改善未穿媒体，不授权选择性换 readout，[CONTRAST 实跑复核](../../../diagnostics/local-fourier-rm-contrast-real-run-audit-20261004/design_review.md) |
| DWELL4 | 5/4 | DIRECT 3/1；RAW420 7/1；MP4 49/45 | DIRECT/RAW420 的 DIFF rank1 是真实局部正证据；MP4 仍未闭合，不能把该路线写成仅 CPU 结果，[DWELL4 实跑审计 JSON](../../../diagnostics/dwell4-real-run-audit-20261004/audit_result.json) |

历史边界来源：[方法机制审计 §2.1、§3.2 及 §6 后续实跑补充](../../../diagnostics/method-mechanism-audit-20260928/assessment.md)、[T49 VAE lift 协议](../../RGB-DCT-Structured-Terminal-Feedback-V1/docs/rgb_dct_t49_vae_lift_protocol_20260923.md)、[多步比较协议](../../RGB-DCT-Structured-Terminal-Feedback-V1/docs/rgb_dct_multistep_comparison_protocol_20260924.md)、[terminal gradient 协议](../../RGB-DCT-Structured-Terminal-Feedback-V1/docs/rgb_dct_terminal_gradient_v1_protocol.md)、[B2 协议](../../Video-Trajectory-Blind-Validation-V1/docs/rgb_dct_b2_terminal_receiver_protocol.md)、[Structured 协议](../../RGB-DCT-Structured-Terminal-Feedback-V1/docs/rgb_dct_structured_terminal_feedback_v1_protocol.md)及 [same-raster/zero-mean 设计复核](../../../diagnostics/local-fourier-rm-same-raster-real-run-audit-20261003/design_review.md)。这些历史只界定差异，不授权复跑或迁移阈值。

## 分阶段证据与停止条件

第一阶段只检验同一新构造的多步 state+payload 是否共同穿过媒体。层次分别记录 terminal latent（后评）、raw VAE float RGB、RGB8、同一 RGB8 经 codec 得到的 MP4；旧 `DIRECT_RGB8` 不能称为纯 VAE 通道。控制可按具体问题安排 OFF、local payload-only、current hybrid 与 new joint，但不跑大笛卡尔积，也不把不同载体臂称为 budget matched。实际 source/seed/key/codec/message/阈值须另行冻结。

早期媒体共同存留不等于正式四分片机制完成。后续阶段才比较正确、错误、无对齐的不同 fragment 重组，再进入预先声明的有界攻击与聚合。

候选的判别预测是：真实 25-step 记录生效；float→RGB8→MP4 后，state q 的正确路径或预定义等价集合仍可区分，同时局部 payload 四分片可重组。两者缺一都不闭合。仅排名改善、旧重复载荷成功或 CPU 理想信号不够。

- float 层即失败：是该构造或回注链的有效负证据。
- float 成立而 RGB8/MP4 失败：是媒体存留缺口，但不能把原因归给单个 codec 因子。
- 资源、OOM、接线或 instrumentation 失败：工程证据不足，不是构造负结果。
- 一个冻结候选得到有效负结果后停止该构造；不改 `rho`、cap、频率或支持后重跑。无实际数据前不预设科学 PASS 阈值。

## 参数待决定与执行边界

carrier、四分片时间组织与 posterior-difference lift 已获本地实现授权。用户仍需单独决定具体 `rho` 与 cap 数值；API 中出现的 CPU fixture 数值不是实验冻结。任何 source/seed/key/message/codec、blind path、阈值、正式 arm 或真实执行范围仍需另行明确。

## 阶段 1 公共接口交付与验证收据（历史）

基线为 `527c4800292c296222c2c0533809eccb82a29a65`，分支 `dev/local-joint-state-payload-v1`。该阶段尚未实现 carrier 数学；runtime 只提供 carrier-agnostic 外部注入和 received-only 逐窗记录接口。

变更文件：

- `main/tube_state/local_joint_state_payload_v1.py`
- `runtime/wan/local_joint_state_payload_v1.py`
- `runtime/wan/grow_video_reference.py`
- `tests/test_local_joint_state_payload_v1.py`
- `docs/local_joint_state_payload_v1_proposal.md`

验证环境为 `/home/richar/projects/CEG-WM/alive/CEG-WM/.venv/bin/python`，仅复用已安装的 pytest/torch；命令设置 `PYTHONDONTWRITEBYTECODE=1` 与 `CUDA_VISIBLE_DEVICES=''`，未安装依赖，未调用 CUDA、模型、VAE、codec 或媒体。

- `tests/test_local_joint_state_payload_v1.py`：`7 passed in 0.87s`，包含 sub-ULP nominal/realized 分离和 native total-update 定向检查。
- `tests/test_grow_video_reference.py`：`15 passed in 14.34s`。
- 四个变更 Python 文件 `py_compile`：通过。
- `git diff --check`：通过。

这些只证明公共接口、CPU tensor 接线、默认 GROW 旧路径回归和静态语法，不证明候选 carrier、媒体存留、盲路径、四分片恢复、资源可行性或任何科学 PASS。

## 审查收据

受审代码版本为 `1e9219182d265533a3c657c744dcd14ea6e34c8c`。初版 `f3cacf7` 随后修正了 nominal provider operand、FP32 realized clean/CFG delta 与 native step total-update 的记录语义，删除了未经核验的 component decomposition 声称，并补清确定性构造和历史 delta 边界。A2 方法审查与 A3 实现审查均接受该同一版本且无必修 finding；A3 独立复跑两个受影响的定向 CPU 测试为 `2 passed in 0.97s`，它们是上述 7 项接口测试的子集，不另加到验证分母。A4 综合与 A5 阶段审计均接受，A5 未重复全量测试。

这些审查当时只支持把工程接口与 `PROPOSED_NOT_ADOPTED` 提案交给用户决定，不代表真实运行获授权或得到科学 PASS。DWELL4 的真实局部正证据与媒体失败边界继续同时保留。本节是后续授权前的历史收据；审查覆盖的代码版本仍为 `1e9219182d265533a3c657c744dcd14ea6e34c8c`。

## 当前构造实现与验证收据

本轮在同一分支新增 `main/tube_state/local_joint_state_payload_carrier_v1.py` 与 `runtime/wan/local_joint_state_payload_provider_v1.py`，并新增两个对应的 CPU fixture 文件。carrier 模块实现确定性 key/pair/RM/四分片、FP64 DCT 局部能量改写、固定 768 行 raw window 目录、逐 chip 原始能量/q/support 记录和无聚合 fragment evidence routing。provider 要求显式 `rho/cap`，在 0..24 返回 zero operand，在 25..49 经外部 posterior backend 生成 mask-then-cap joint operand；factory 返回可直接传给既有 `run_trajectory(..., injected_control=...)` 的 callable。既有控制 adapter 与 GROW runtime 未改。

验证环境仍为 `/home/richar/projects/CEG-WM/alive/CEG-WM/.venv/bin/python`，设置 `PYTHONDONTWRITEBYTECODE=1`、`CUDA_VISIBLE_DEVICES=''`，未安装依赖：

- carrier/provider 新增 10 项 fixture，定向复跑为 `10 passed in 4.06s`；包括真实小 8×8 patch DCT 算术、pre-clipping pair 能量与目标 ratio、原 sign/zero-sign/zero-support、RGB 局部性与 clip/FP32 误差记录、key/RM/bit 顺序、短 T fixed-row/partial-chip、raw fragment routing/collision/erasure、fake posterior 坐标/mask/cap、50-step 25 decode/50 encode 调用顺序、`rho=0`/`cap=0` 区分及 backend attempted/completed 失败记录。
- 既有 carrier-agnostic seam：`7 passed in 0.74s`。
- 冻结前曾有一次 `9 passed, 1 failed in 4.01s`：一般随机样本中的小能量 pair 会放大 post-clip/FP32 ratio error，原断言 `<2e-6` 缺少数值保证，实际原始最大误差为 `2.5844881314240897e-05`。修复保留受控无 clip writer→reader q 容差检查，并增加从最终 FP32 RGB 独立复算全部 writer window 的 `E+/E-/pooled q` 与收据逐项一致检查；一般样本的误差继续原样保存且检查有限、非负。该单元测试容差不是媒体成功阈值，修复未改 carrier 数学。
- 首次冻结时合并复跑以上不重复的 17 项：`17 passed in 4.12s`；四个相关 Python 模块 `py_compile` 与 `git diff --check` 通过。

这些 fixture 使用小型 CPU tensor 和 fake posterior；没有执行完整 181 帧 carrier、真实模型/VAE、codec 或媒体。它们不证明资源可行性、blind path、fragment 聚合规则、真实多步存留或科学 PASS。下一项真实问题仍是：同一次多步写入后，局部 state 证据与不同 payload fragment 是否共同穿过 MP4；该问题不能由本轮 CPU 结果代答。

### `1f9b2f0` 同版审查后的机械修复收据

A2/A3 对 `1f9b2f0f56d07f8e225d407f279ba21c91901b37` 的 finding 只引出记录与输入边界修复：correspondence truth 标为未核验并拆分 unrouted 与有 route 但全缺失/失败；非法 received frame 索引保留 FAILED 行；writer 拒绝非 FP32 输入并落实 FP32-before-clip；provider 统一记录非 backend stage 失败。carrier pair/state/payload 数学、显式 `rho/cap` 状态、25..49 调度与真实执行边界均未改变。

受影响 carrier/provider fixture 为 `13 passed in 4.15s`，其中实际 clip 顺序定向复跑 `5 passed in 1.05s`；最终连同既有 7 项 seam 合并为 `20 passed in 4.27s`。四个相关 Python 模块 `py_compile` 与 `git diff --check` 通过。验证仍仅为 CPU 小 tensor/fake backend，不增加任何科学结论。

### 第二阶段同版审查收据

受审代码版本为 `8f664a365b2e808a3db2db243ed18f11936ff5fe`。初版审查提出的五类 finding——correspondence 盲性不可推断、route 与有效证据混淆、非法 received frame 索引、非 FP32 输入导致的 ROI 外变化及 FP32/clip 顺序、非 backend stage 失败漏记——均已关闭。A2 与 A3 对该同一版本均为 ACCEPT 且无必修 finding；A3 独立复跑 `5 passed in 1.38s`，是上述 20 项 CPU 总分母的子集，不另加到分母。A4 综合与 A5 独立里程碑审计均为 ACCEPT；A5 未提出新 finding，并确认分支 clean、main 基线仍为 `527c4800292c296222c2c0533809eccb82a29a65`，架构没有反向导入 runtime、experiments 或 governance。carrier 内含按需导入的 torch 数学实现，不把它描述为仅标准库。

该版本提供可选 carrier 与显式注入 provider，不包含 model-residency loader 或独立实验 runner。20 项 CPU 只支持工程实现与记录语义，不是 scientific PASS；`rho/cap` 数值仍待决定，真实问题仍是多步写入后的局部 state 与不同 payload fragment 能否共同穿过 MP4。

上述“不包含 model-residency loader 或独立实验 runner”只描述受审版本 `8f664a3` 的历史状态。后续入口、显式配置 schema、资源成本和仍待用户裁定的最小真实机制名单见 [独立真实实验提案](local_joint_state_payload_v1_real_experiment_proposal.md)；新增 runner 不改变本文件中 carrier 数学、参数待决定状态或历史审查结论，也不表示真实执行获授权。

### 第三阶段 runner 版本绑定与审查收据

本阶段从 `6590c5cf5a92e9bbf92aaabf3c07152841f082bd` 开始。初版 runner `f9363074b74017a55ebfad3f15ecae0c553e6d44` 的 27 项 CPU 检查没有证明完整 runner 闭合；同版审查发现的 raw observation `as_dict`/`to_dict` 调用错误、terminal dtype/geometry/finite 验证顺序、persisted RGB8 lineage、相同像素积但错误宽高的媒体几何、cleanup 覆盖主异常、execution 标签和 CLI 配置 schema 问题，均在最终受审代码 `cb81e1f71c1bd1467dd1daf18f6ca79174487e69` 中修复。

A1 最终合并验证为 `34 passed in 14.72s`，由 14 项 runner/异常/media/no-git 工程检查与既有 20 项 carrier/provider/seam 检查组成；相关 Python `py_compile` 与 `git diff --check` 通过。A3 独立复跑 `7 passed in 7.75s`，是上述 34 项的子集，不另加到分母。A2 与 A3 对 `cb81e1f` 同版均为 ACCEPT 且无必修 finding，A4 综合为 ACCEPT，A5 独立里程碑审计亦为 ACCEPT 且未重复测试。

完整 non-preflight fake 成功集成只运行 OFF arm，使用 181×8×32 的小空间 RGB、fake residency/codec 与小 latent，真实经过同一 `run_trajectory` 的 50 steps，并为 float RGB、RGB8、MP4 各保存固定 768 行目录。JOINT binding 与 steps 25..49 的 25 次启用行为由组件和既有 CPU 测试覆盖；没有运行真实权重、模型、GPU、VAE、codec 或媒体。因此这些检查不构成完整真实 JOINT 验证，也不证明显存可行性、25 次驻留往返、27 次 VAE load 或媒体存留。

真实实验的参数、arm 名单、后评 reducer 与描述性进展条件仍为 PROPOSED；wrong-key 后评与 reducer 尚未实现。本线审查 ACCEPT 只接受该冻结版本的工程接线、记录和提案边界，不是主审计结论、真实执行授权或发布授权。
