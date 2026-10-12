# B 线 All-Bit 两小步续走：具体提案 v2（待采纳）

**当前状态（2026-10-12，v2）：恢复续走提案准备，整组新参数仍待用户集中采纳。** 本轮只更新本提案并执行必要 CPU 算术；没有修改 runtime/config/notebook，没有新梯度、模型或 GPU/Colab/Drive 执行，也没有 push/main 合并。

历史：v1 的 0.10 两小步提案曾按用户纠正搁置，先完成现有 All-Bit v1 的定时保存能力；恢复功能已在 `cd11bb7` 发布，最新 `resumable-task-v1` 任务也已完成一次真实完整运行。本次获授权恢复方法提案准备，不等于采纳 0.10 两步或授权执行。此前保存交付见 [恢复说明](local_joint_all_bit_resume_delivery_20261012.md)；其发布与新方法采纳是两个不同状态。

**首选仍只有一组：从最新保存 ALLBIT_TWO 的实际 terminal 开始，两次新 scalar VJP，各名义步长 0.10；原 BASE 中心、半径 1；四个评分视图，完整 1 load +6 decode +2 VJP。** 起草分支为 `dev/local-joint-state-payload-v1`，已发布恢复实现 `cd11bb7` 只作定位。本案新起点、步长、实际位移投影/FP32 数值处理及调用矩阵待整组采纳。

## 1. 依据与本轮要回答的问题

依据最新 `resumable-task-v1` 的 [主审检查点审计](../../../diagnostics/b-line-all-bit-resumable-task-v1-20261012-audit/root_checkpoint_audit.json)、[32 位/方法复核](../../../diagnostics/b-line-all-bit-resumable-task-v1-20261012-audit/reviewer_numeric_method.json) 及本地同次张量。主审确认完整 1 load +6D +2VJP，四视图 352/5632/220 全部完成，32 位读数、两份梯度、terminal 和保存的质量记录与上一 All-Bit 结果一致。只有一个 notebook/runtime attempt，已保存状态与梯度检查点可读；**并未实际经历 Colab 到期后的跨会话恢复**。本轮核对当前输入结构、范数和拟议算术，不重跑保存测试或既有实验。

保存 TWO 为26/32正位，原16个正位无丢失；六个最终负位在此前两步均改善，但不是舍入附近的缺口：

| 固定观察位（0起始） | BASE | 旧 HALF | 保存 TWO / 新起点 | 距0缺口 |
|---|---:|---:|---:|---:|
| f0b4 | −0.429410 | −0.346074 | −0.235336 | 0.235336 |
| f1b4 | −0.374644 | −0.293690 | −0.170786 | 0.170786 |
| f2b2 | −0.393566 | −0.333691 | −0.268020 | 0.268020 |
| f2b4 | −0.375441 | −0.275229 | −0.147293 | 0.147293 |
| f3b3 | −0.252368 | −0.194935 | −0.112479 | 0.112479 |
| f3b6 | −0.272495 | −0.225071 | −0.159602 | 0.159602 |

26个正位全部高于 μ。最弱正位 **f3b5=0.02017283573380433**，仅比 μ 高 `0.00017283573380432946`；其自身损失导数现在为0。G=`0.020974863037095552`，比 γ 高约0.01097486，状态项同样为0。因此同一目标在新起点恰有6个活动payload项，每位系数−0.025；**不把目标改成只求这6位，也不把分母32改成6**。其他位/状态跌回目标以下仍按原式重新激活。

机制问题是：在余下原球可行域内，两次更短、各自刷新方向的更新，能否继续改善六位/增加过零，同时保留26个正位、G和可接受的实际质量取舍。小步只是一组可检验设计，不能保证不丢位、有限步下降、补齐32位或维持G；不外推旧0.5步的逐位变化来承诺新步过零。旧损失与覆盖不等价的反例仍成立。

## 2. 一组固定参数与原球更新

沿用 `L=0.8 mean_32 H_0.02(m_i)+0.2 H_0.01(G)`；H在 x≤0 为a/2−x，在0<x<a为(a−x)²/(2a)，x≥a为0。沿用原key/message、public mask、readout、FP32 frozen Wan、实际归一化缩放和内外clamp梯度链。只改变续走起点及新两步的名义长度：**α1=α2=0.10**，即原0.5的五分之一；两步名义路程合计0.20。

理由是保留一次中点刷新，同时缩短单次未重新观察软保护的距离。按实际起点范数 `r=0.8436085787762565`，第一步三角上界 `r+0.10=0.9436085787762565<1`，故精确算术下第一步不触cap；第二步投影前上界 `r+0.20=1.0436085787762566`，仍可能触cap，必须按真实方向投影，最外向几何情况的累计缩放约0.958213664。`1−r=0.1563914212`只是径向差，**不是方向无关的剩余路程**。本选择没有由新响应、扫描或最优性证明选出；不能据上界预报新梯度朝向。 上述两步上界是忽略中间落盘舍入的几何量；FP32 实际第二步应使用已落盘 MID 的 `||z_MID−z_BASE||+0.10`，从初始 r 展开时须计入第一步舍入误差。每步都重新量取实际输入并投影/复核落盘半径，不以 `r+0.20` 代替实际预算检查。

输入采用最新任务内已经读到的两个实际 FP32 terminal，不依赖额外逐视图副本：

```text
ROOT=/content/drive/MyDrive/Video-WM/Local-Joint-State-Payload-V1-All-Bit/resumable-task-v1/run
z_BASE  = torch.load(ROOT/inputs/terminal_latent.pt, map_location="cpu", weights_only=True)
z_START = torch.load(ROOT/step_2_unit.pt, map_location="cpu", weights_only=True)["terminal"]
可选辅助记录 = 同一 step_2_unit.pt["delta"]
```

这里 `inputs/terminal_latent.pt` 就是同次 BASE；`step_2_unit.pt['terminal']` 是落盘的实际 ALLBIT_TWO。两者形状均为 `(1,16,46,40,64)`、FP32、数值有限。本轮直接从这对实际张量测得上述 r，未沿用旧日期目录或把计划 delta 当输入。

首个新 VJP 直接使用 `z_START` 原值，绝不以 `z_BASE+保存计划delta` 替换。辅助计划范数为 `0.8436085832765362`，实际范数为 `0.8436085787762565`；两个位移向量相差 L2 `9.605287979959581e−6`。辅助 delta 缺失不阻断可用的两个 terminal。**旧 `step_2_unit.pt['input_terminal']` 是旧 HALF**，距新 START 的 L2 为 `0.5000000182980807`；对应旧 `gradients/step_2_checkpoint.pt` 的梯度不能当新 START 梯度。新 g1 必须在 START、新 g2 必须在新 MID 计算。

对k=0、1，z0=z_START，实际累积向量每次由当前实际输入重算：

```text
d_k = FP64(z_k) - FP64(z_BASE)
g_k = M * grad_z L(actual_FP32_decode(z_k))  # 本逻辑步一次新标量VJP
u_k = -FP64(g_k) / ||FP64(g_k)||2
c_k = d_k + 0.10 * u_k
s_k = 1 if ||c_k||2==0 else min(1, 1/||c_k||2)
p_k = s_k * c_k                             # CPU FP64，原BASE球
z_(k+1) = 按下述数值规则将 z_BASE+p_k 保存为FP32
```

**FP32边界现在一并提出，不留到实施时再选：**先算FP64投影；若乘法/范数舍入使 `||p_k||>1`，仅将s向0取相邻FP64值（`nextafter`）至该算术范数≤1。先作常规最近FP32舍入q，计算实际 `a=FP64(q)−FP64(z_BASE)`；若其范数≤1，直接使用q，不作内缩。只有最近舍入造成实际越界时，逐坐标找出相对于p向外越过的分量：p_i>0且a_i>p_i，或p_i<0且a_i<p_i；这些q_i向原BASE取一个相邻FP32值，其余不动。于是修复后各位移分量不超过投影分量，保持原球且不补满。此规则不更换半径或名义步长，不读loss/位数挑选，不产生额外decode/VJP；仍异常越界则记录具体数值失败，不能静默放宽cap。

记录s、FP64 scale修正次数、内侧舍入分量数、计划投影范数、落盘实际范数/增量/mask外分量及舍入误差。上述仅越界时的内侧舍入是显式数值选择，与旧入口单纯最近舍入的差异需保留；不宣称旧程序逐位等同延长。两步之后仍以实际保存terminal−原BASE计费。预测只写 `g_k·(z_(k+1)−z_k)` 及投影前后的名义位移预测，实际变化写同一scalar loss差，不冒充32个位梯度。

退化/失败沿用已采纳范围：零masked梯度不生成伪零方向；仅G<γ且wrong-offset最大值并列时阻断依赖方向；payload-min并列及平坦state并列不阻断。合法非零方向相消到零累计向量仍保存/评分。真实shape/非有限/零能量/read/load/save/decode/VJP故障保留缺失与原异常。已形成方向与后续目标评分失败分开记录。无按表现回滚、缩步、择优或第三步；末步丢位/G跌破后不自动修复。

## 3. 四视图与完整不中断调用账

| 固定视图 | 实际输入 | 本次普通评分D | 用途 |
|---|---|---:|---|
| BASE | 同次保存原BASE | 1 | 同一新执行内的float质量参考 |
| START_SAVED | 保存ALLBIT_TWO terminal原值 | 1 | 新起点、主比较参考、新VJP1的普通读出 |
| CONT_MID | 新0.10步落盘terminal | 1 | 第二新VJP起点及中点 |
| CONT_FINAL | 第二新0.10步落盘terminal | 1 | 固定终点，主比较对START_SAVED |

保留 **4×88=352 windows、5632 chips、220原指标**。六位观察表不替代全32位分母；同次历史旧TWO_MEAN仅链接背景，不加入新矩阵。主比较 FINAL−START，另存 MID−START、FINAL−MID及各行对BASE。起终点重新评分只解码保存terminal，**不重跑原两个0.5更新、旧梯度、native链或生成过程**。这两次普通decode为同执行读出/质量参考，避免把历史质量与新RGB不加说明地拼在一起；本案固定这一种设计，不增加省略对照模式。

完整顺序：load → BASE普通D → START普通D → START梯度D/VJP1 → MID普通D → MID梯度D/VJP2 → FINAL普通D。完整预算 **1 load、6D、2 decoder VJP；0 encode/DiT/native/scheduler/codec**。普通chunks184，梯度forward92，名义replay92，实际attempted/returned和recompute逐份记录。两个梯度forward的55原值分别对照同输入START/MID普通读出；其差异只是诊断，不增加样本或成为身份相等准入。

BASE reader/评分失败仍保留其槽；只要 BASE float RGB 可读，其他视图相对 BASE 的质量仍可在 CPU 计算，只有 BASE 像素不可读、损坏或缺失时才将相关质量记为缺失。两个输入 terminal 可用且 START 可评分时，可继续新方向。START 评分/构造失败不进入新 VJP；MID 失败保留已形成第一方向且第二步缺失。FINAL 变差或未评分都保留终点槽。失败不凑齐预算、不把已有方向改成未定义，也不为补质量自动重新 decode 已完成观测。

## 4. 只恢复本次逻辑任务的已完成工作

已发布 `cd11bb7` 的 [Journal](../runtime/wan/local_joint_all_bit_resume_v1.py)、[runner](../runtime/wan/local_joint_all_bit_v1.py) 和 [builder](../scripts/build_local_joint_all_bit_v1_notebook.py) 已具备稳定逻辑目录、完整单元恢复、60 秒状态/进度保存、可读前态回退、跨次调用账和 lazy-load 接线。最新完整实跑确认这些文件成功落盘；到期跨会话恢复的证据仍是已通过的 CPU/fake 测试，不能把单 attempt 实跑称为真实断点续跑验证。[ReplayLedger/BoundarySpool](../runtime/wan/local_joint_readout_checkpoint_v1.py) 则仍只服务当前 autograd graph 的临时 storage/cache，与任务持久化分开。

采纳后复用上述实际保存器和恢复流程，窄接线本案四个 view/两个 step 标识、实际起点更新公式及相应完整单元的必要结构检查；不另造一套保存治理。本案新方法入口尚未实现，不能直接重跑旧任务或改其方法数值。纯 CPU 收尾在环境检查之前分派，VAE 仅在确有未完成模型调用时加载；同 kernel Run all 沿用已修复的旧包缓存清理。

**新实验使用独立逻辑目录**，拟定默认 `MyDrive/Video-WM/Local-Joint-State-Payload-V1-All-Bit-Continue/resumable-task-v1-2x010/`。旧任务的 `...All-Bit/resumable-task-v1/run` 只提供 BASE/START，不复用其完成状态或旧梯度来充当新两步。新目录首次创建四槽，再次 Run all 读取同一目录可用产物，从未完成阶段继续，不清空 raw/metrics/已完成槽。沿用 `notebook_attempts/<UTC>/`、`attempts/`、`calls/` 保留各次日志、状态和物理调用；全部完成或合法终局时只报告，不再开启新步骤。目录和源码位置只作定位，无 strict/override 模式；同一次进程不后台重试，跨次恢复由用户 Run all 触发。

**60秒 / 完成单元两层边界：**固定每60秒保存一次**最近已提交的可恢复状态及进度快照**，包含已完成单元、对应已落盘input/gradient/delta/terminal/raw/score的位置、下一恢复阶段、固定方法数值与各attempt的已知调用账；不能只写存活心跳。快照采用普通临时文件替换并**保留上一份可读快照**，实际写入失败保留原异常和旧可读状态。每个完整单元（gradient+输入、delta/terminal、float RGB、raw、score）完成即立即落盘，不等60秒；恢复以可读的完整单元为依据，不能因周期快照较旧而忽略已成功保存的gradient、重复其VJP。长decode/VJP尚未完成时，周期快照保存最近完成单元及当前运行中状态，不能凭它跨会话恢复半张autograd图。因此60秒是状态保存间隔，**不是最多丢失60秒计算的保证**；图未完成时损失窗口可覆盖整次梯度D+VJP。硬过期/实际Drive写入失败前来不及保存的区间按完成未知保留，不能伪造完成。无需将spool搬到Drive或每60秒重启单元。

每个新逻辑步及时保存：

- 一次普通原子保存的 `gradient_checkpoint.pt`，含**实际求导输入terminal与该masked gradient**（普通张量/字典），另存55值/q、loss/活动项及该次调用记录。只保存能读回的完整结果，不序列化autograd graph或64GiB spool。新方向只由该绑定输入和已保存gradient恢复；不是复用旧run的梯度。
- 由上述g作CPU归一化、原球投影后保存delta、实际terminal和step报告；普通decode后依次保存float RGB → raw → metrics/loss → quality/图片。两个新VJP都在自身finally清理cache/spool后再进入下一步，跨会话不保留中途cache。
- 恢复检查限于实际可读/shape/dtype/有限值、数值mask/原球语义和本阶段需要的数据结构；无哈希、签名、manifest、源码/依赖/Git身份相等。来源/环境仅定位。固定loss、key/message、mask及已采用步长仍是方法语义，恢复读取本逻辑任务保存的数值；不能用一次续跑暗改方法。

| 已持久化边界（此前 BASE/START 均完成，未提及的后续单元尚未保存） | 过期后应做什么 | 新会话剩余 load / D / VJP |
|---|---|---|
| g1 及实际输入已保存，MID RGB、raw、score 均未保存 | CPU 恢复第一候选；MID 普通 D、新 g2 梯度 D/VJP、FINAL 普通 D | 1 / 3 / 1 |
| g1 与 MID RGB 可读，MID raw/score 尚缺，g2 未保存 | CPU 补 MID 读数/评分，再做 MID 梯度 D/VJP 和 FINAL 普通 D | 1 / 2 / 1 |
| MID float/raw/score 已保存，g2 未保存 | MID 梯度 D/VJP 和 FINAL 普通 D | 1 / 2 / 1 |
| g2 及实际输入已保存，FINAL RGB、raw、score 均未保存 | CPU 恢复第二候选；FINAL 普通 D | 1 / 1 / 0 |
| g2 与 FINAL RGB 可读，FINAL raw/score 尚缺 | 只用已有像素补 CPU 读数/评分/质量 | 0 / 0 / 0 |
| 四行 RGB 均可读，raw/metrics/quality/PNG 尚缺 | 只补 CPU 后处理 | 0 / 0 / 0 |
| raw 已保存，只缺该阶段 metrics/loss | CPU truth reducer，不重解码、不重求 g | 0 / 0 / 0（此阶段） |
| 全完成，或已保存零梯度/活动 tie 合法终局 | 报告已完成/缺失槽；若可补质量，仅作 CPU 后处理 | 0 / 0 / 0 |

若delta/terminal已保存，则直接读取；没有时可由完整gradient checkpoint作CPU恢复。score已保存则不重评分；质量/图片未完成且像素可读时只补CPU工序。raw/score完整但RGB损坏/缺失时保留评分及明确缺失质量，不能伪造像素或为更好质量自动重解码。这类实际I/O问题单独报告，不清掉成功VJP。

**不可恢复的边界：**VJP还在执行、只有gradient-forward/q/部分spool、只有未完成临时文件，或调用已返回但完整gradient checkpoint未落盘，都不能从中间继续autograd。用户手工恢复时只能在同一保存输入重新做该逻辑步的整个梯度D+VJP；不得从q/旧梯度拼方向。模型D返回而float RGB尚未保存时，未完成评分也只能重新普通D。保留先前attempted/returned/完成未知与失败，不将“没有落盘结果”写成“从未调用”。已落盘的零梯度/活动tie是构造结果，恢复不能重抽直到非零/无tie。

6D+2VJP是一次**不中断完整任务**的账，不是任意多次重启后的物理调用上限。发生上述丢失窗口，每次手工续跑先列可复用阶段、剩余调用及必须重做的调用；所有实际尝试累计，VAE每个需要模型的新进程至多load一次，纯CPU恢复不load。例：g1返回但文件未保存，重新求g1再做g2会有3次实际已返回VJP，必须照实保留，不能仍报2。逻辑更新仍只两步，无后台重试、自动加步或择优，是否启动中断后的恢复由用户执行Run all决定。

## 5. 逐位、质量和资源交付

每行保留22 correlations、G、32 margins、loss及32/state系数、活动项。固定报告上述六位各自的两次变化、距0/μ缺口及过零；同时报告全32位增减、获位/丢位、正位数、最差margin，原26正位的保留和f3b5轨迹，以及每个新视图当时最弱正位。若无正位记无，不从分母删掉。报告G、距γ及是否状态项重新活动；μ/γ不是检测或质量PASS线。不能以较低L盖过丢位、G回落或平均/峰值质量代价。

所有质量相对本次BASE：四行共724条逐帧及720条相邻残差记录，含RMSE/PSNR/max_abs/像素位置、ROI内外能量与峰值、时序RMSE/max_abs。沿用现有定位规则：各非BASE视图的frame RMSE/max_abs和temporal RMSE/max_abs峰值，最早并列取索引，帧峰值±1、相邻对两端各扩1帧，边界截断；与固定帧[1,44,88,112,116,120,132,176]取并集，对全部可用视图输出相同索引及峰值crop。不能把上一轮峰值85/127/120→121/160→161或38张帧图当作本轮预定峰值/固定张数。RGB统一[0,1]、残差[-.25,.25]且报告显示截断数；数值不截断，无质量PASS阈值、无codec。

最新 `resumable-task-v1` 的资源记录为 **1022.9058 秒（17.05 分钟）**；allocated **19.01 GiB**、reserved **21.72 GiB**、进程 RSS **3.37 GiB**。本次普通 D 约 22.98–24.88 秒，两个 gradient-forward 约 229.13/228.00 秒，VJP 约 195.58/200.81 秒。每份 spool 峰值 64.53 GiB，两次累计写/读各 129.05 GiB，最终 live 均 0，CPU 传输 buffer 峰值 16 MiB。旧 v1 引用的 29.67 分钟属于上一日期运行，不作为本次资源记录。**17.05 分钟不是新实验耗时保证；小步不降低 decode/VJP 的张量规模，也不保证更快或峰值不变。** 沿用顺序独立 spool，累计 I/O 不当作同时驻留量。

最新保存的 START 质量记录（相对同次 BASE）为 RMSE `0.0026634106`、PSNR `51.4912 dB`、max_abs `0.2028902173`；ROI 内/外 RMSE `0.0079478835/0.0009290799`，时序残差 RMSE `0.0025617885`。局部 frame-RMSE 峰值在 85，max_abs 峰值在 127；时序 RMSE 峰值 120→121 为 `0.0047490583`，时序 max_abs 峰值 160→161 为 `0.2723852396`。这些是已保存质量记录及后续固定对照，当前审计未重新下载完整 RGB/PNG 像素，不能称作本轮像素级重测，也不把这些索引冻结为新结果峰值。

只复用实际成功的 [all-bit](../runtime/wan/local_joint_all_bit_v1.py)、[quality](../runtime/wan/local_joint_all_bit_quality_v1.py)、[disk-spooled VAE](../runtime/wan/local_joint_readout_checkpoint_v1.py) 与 [notebook builder](../scripts/build_local_joint_all_bit_v1_notebook.py) 路径。真正实施时同步新真实notebook和builder，lazy load仅在还有模型调用时发生；恢复不能因只差CPU报告而再要求一次VAE/CUDA加载。没有GPU白名单、磁盘配额硬停、完整性链或A线全套状态治理。

## 6. 本轮完成的必要算术与待采纳项

v2 [算术脚本](../../../diagnostics/b-line-all-bit-continue-proposal-20261012-v2/proposal_arithmetic.py) 与 [结果 JSON](../../../diagnostics/b-line-all-bit-continue-proposal-20261012-v2/proposal_arithmetic.json) 直接读取最新任务的 32 位记录、`inputs/terminal_latent.pt` 和 `step_2_unit.pt`，复核六负位、原 26 正位名单、f3b5、G/系数、实际范数及旧 step2 输入不等于新 START；只代入唯一 α=.10 的几何上界和明确 RGB 前提的恢复预算，并在人工二维/FP32 向量上复核投影/舍入。没有利用旧梯度构造新候选，没有新 loss 梯度、decoder、评分或步长扫描。少量变更记录见 [v2 差异说明](../../../diagnostics/b-line-all-bit-continue-proposal-20261012-v2/changes.md)。

人工边界例的最近FP32舍入会把范数1变成 `1.000000023841858`；规定的仅越界内侧修复将其变为 `0.9999999284744284`，位移舍入误差L2约9.70e−8，说明实际落盘预算不能仅看FP64计划值。这个数值例不是当前视频的新候选或方法成功证据。已发布 Journal/保存恢复测试本轮没有重跑。采纳后只覆盖新增的实际起点、0.10 更新/FP32 舍入和四视图接线所影响的必要 CPU 替身检查，不重做未变保存器的整套测试或数学审计。

请主审据本 v2 与上述具体差异集中呈报**一个采纳问题**：是否采用“保存ALLBIT_TWO实际起点、0.10×2、两次同目标新VJP、原BASE半径1实际累计投影及上述FP32边界处理、四评分视图/完整1load+6D+2VJP/352-5632-220、60秒进度与完成单元恢复边界”这一整组固定构造，在独立新逻辑目录中复用已发布恢复能力，并据此进入最小实现与用户 Run all 交付。旧μ/γ/权重/readout已采用，不重开参数菜单。
当前交付上限是“一个具体可审阅、可计价的续走构造及算术边界”，尚不是获采用的实现或可立即执行的新notebook。26/32不会自动触发codec、M1/M2、盲解码/FPR、main集成或科学PASS；真实效果必须由用户采纳后实施、运行并按完整固定分母判读。
