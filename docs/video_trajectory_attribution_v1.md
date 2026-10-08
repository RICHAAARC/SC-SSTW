# Trajectory Attribution V1

本研究分支实现“基于轨迹采样嵌入思想的视频水印方法”的归属判定。它复用当前 main 已发布的 trajectory payload、M05、时间编辑搜索和稳定 payload reader，不导入退役 Stage1 runner，也不改变旧数学文件。GROW 仍是具体实现来源；总体方法名不再写成 GROW 扩展。

## 固定数据与资源层

固定来源为 DEV 黄帆船 seed 2026100701、C1 蓝色潜艇 seed 2026100802、C2 橙色纸风车 seed 2026100803。DEV 用于一次性冻结数值，C1/C2 是冻结后才允许生成的未见确认源，不换样、不重试、不调参。每个 source 在一个 generation 子进程内加载一次 transformer，共用同一 initial noise，但 OFF/A/B 各自使用新的 pristine scheduler 并完整执行 50 个 native steps；三条轨迹是一个配对 source cluster，不是三个独立源。generation 退出后，Wan VAE native decode 与 framewise/MP4 继续在串行独立 worker 中执行，避免多模型共存。

每个 source 产生四个 received conditions：

- OFF：真实无 payload 局部更新、无 M05，但经过相同 framewise encode/decode 和 MP4。
- A_P1：A=K0/OKOK 的独立 marked trajectory；复用 A 的原 framewise latent，不写 M05。它是 payload-bearing sync-null，不是无水印。
- A_M05：同一 A framewise latent 写入 target .5 M05 后保存。
- B_M05：B=K0/NOPE 的独立物理 trajectory，再写入 target .5 M05；它是 other identity，不是对 A 视频换读消息。

每个 condition 使用固定八个 received maps：FULL181、crop177 phase1/2/3、short89 start37/38/39、single deletion b2/k88。每个 observation 查询 K0/K1，公开 claim 永远是 query key 与 OKOK 32 bits。每 source 32 physical observations、64 logical queries；DEV64 + confirmation128 = 192。三源候选总数 74784，framewise 总帧 13872、batch8 批数 1812。alias path/action 不是独立 observation。

## 盲证据与 action 等价

receiver 只接收 received RGB、公开 N、query key 和公开 claim bits。181 执行完整 singleton；89 执行完整 G93；177 执行统一 J709，不由输入真值选择 global/deletion。所有候选必须完整且 finite。每条 candidate path 转换为完整 received_index_map，以该 map 的内容身份折叠 action 等价：181 为 1 path/1 action，177 为 709 paths/699 actions，89 为 93 paths/4 actions。

M 是完整搜索最高分。tie 使用既有 tie_atol。多个 top paths 若对应同一个完整 action，则 action resolved，但 absolute source path 保持歧义并单独报告；多个 top actions 才是同步歧义。唯一 top action 的时序代表只用于记录，不能由 payload 选择。m 不使用全局 score gap。177 从已封存 frame cache 逐格复算 q=numerator/rho；对每条 top path 与每条不同 action path，只在两者 source map 不同的 received 位置上平均 q 优势，再对全部组合取最小值。89 校验原 local geometry、q、numerator、rho 与原 candidate S 一致，计算 Q[b]=sum(observed_frames*q)/89，再对每条 top path 与每条不同 action path 取 Q 差的最小值。跨 action 的 top tie 保留为同步歧义；m 仍按上述局部对比公式计算，不强制设为 0。same-action tie 不把别名当竞争 action。181 没有 m。

固定顺序是：完整 sync 搜索并封存 → 唯一 action correction → stable payload reader 并封存 → 完整 64 行 decision seal → posthoc role/path。盲 seal 只使用不透明 observation/query ID、公开 N、RGB hash/shape 与 key identity；source/condition/view 名称、构造 map 和 expected action 只在 seal 后由 experiment protocol helper 接回。禁止先读 RAW payload 决定拒绝；历史 baseline 13/14 errors 经 alignment 变 0 的反例仍约束该顺序。posthoc 的 primary geometry 指标比较完整 action/map；same-action path tie 不因任意序列首代表与 absolute path 不同而误判归属。

## DEV 一次冻结

对每个 N：

1. tau_M 是所有预注册完整 sync-null 行的最大 M：OFF 任意 key、A_P1 任意 key、全部 K1。
2. tau_m=max(0,max(m))，取上述 sync-null 中可定义不同 action margin 的行；181 不定义。任一必要行缺失、非 finite，或非181没有可用 m，则规则不可冻结。
3. DEV 阶段只要 action 唯一便做 post-align payload，无阈值筛除。对公开 OKOK claim 的每 bit，计算 (2*claim_bit-1)*(ones-zeros)/count，I 是 32 bits 最小值。
4. tau_I=max(0,max(I))，identity-null 至少逐 N 覆盖 OFF/K0、A_M05/K1、B_M05/K0。缺类、技术失败或非 finite 均不可冻结。
5. A_M05/K0 的每个 DEV 正例必须严格满足 M>tau_M、非181时 m>tau_m、I>tau_I 且 exact32。它只检查可分性，不以 positive minima 调阈值。

冻结记录 formula、source roster、config 和 rule SHA。若任一 N 不可冻结，C1/C2 的 128 行保留为 NOT_RUN/UNCERTAIN_RULE_NOT_FREEZABLE，不启动确认模型，不自动改阈值或重试。

## 确认判定

顺序固定：

1. 规则不可用或技术不完整：UNCERTAIN。
2. 完整 M<=tau_M：REJECT_LOW_SYNC，先于 tie。
3. M 强但 top action 不唯一，或非181 m<=tau_m：UNCERTAIN_SYNC。
4. 只对唯一合格 action correction 后读 payload；失败为 UNCERTAIN。
5. I<0：REJECT_IDENTITY。
6. 0<=I<=tau_I：UNCERTAIN_IDENTITY_WEAK。
7. I>tau_I 且 exact32：ACCEPT；margin/bits 矛盾为 UNCERTAIN_INTERNAL_INCONSISTENCY。

K1 不能因为标签而直接拒绝。OFF、wrong key、other identity 与自然发生的 sync ambiguity 都保留为固定逻辑行；不新增时间攻击来制造结果。报告按 condition/query/N 给出三态和技术失败，并单列 OFF/K0、A_M05/K1、B_M05/K0、A_M05/K0、A_P1/K0。observation 级 false claim 向 query-key、任意 edit、source 逐级聚合：负例 ACCEPT 或正例 ACCEPT 但 action map 错误都记 false；任一 false 为 true；无 false 但有 required 缺失、未知技术状态或 posthoc true map 不可用为 UNRESOLVED；全部 required 科学判定完整且无 false 才为 false。单 observation 的 RGB read/construct、framewise encode 或 payload encode/read 失败保留该 observation 的两个 key 槽并继续其余 observation；中断保留已经完成的行，把仍未决确认槽落为 UNCERTAIN_TECHNICAL_INTERRUPTION/NOT_RUN，所有 source 都先形成 64 行 decision seal 再做 posthoc。

## 发布与证据边界

新 Notebook 是独立研究分支的固定 Run-all，自带配置，不读取 Drive 历史输入。首代码 cell 只挂载 Drive；源码候选先以 SOURCE_SHA=None 静态审查并发布 S，再单独重建此 Notebook 绑定 S 为 N。现有五份 main Notebook 保持各自已发布的 9054f67 绑定，不重写。本实现阶段只执行 CPU/fake/schema/portability 检查（portability 套件包含 synthetic CPU FFmpeg fixture），不自动运行新的真实视频/模型、GPU、Colab 或 Drive；因此独立确认是否完成仍是明确的真实执行 gap。
