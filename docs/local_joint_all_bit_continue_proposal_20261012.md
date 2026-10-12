# B 线 All-Bit 两小步续走：具体提案

> **已搁置（2026-10-12 用户纠正）**：用户需要现有任务定时保存、Colab 到期后继续完成的能力，不采纳本提案的 0.10 两小步。本文件仅保留提案历史；以下参数、数值边界与新方法调用账均未实施。当前工作只为已采纳 All-Bit v1（原 BASE、0.5 × 2）增加持久化恢复，见 [恢复交付说明](local_joint_all_bit_resume_delivery_20261012.md)。

日期：2026-10-12；提案 v1。起草位置为 `dev/local-joint-state-payload-v1`，代码起点 `d83e46a5309537479b5648230b8cc0002f5aa254`，仅作定位记录。**首选一组：从保存 ALLBIT_TWO 的实际 terminal 开始，两次新 scalar VJP，各名义步长 0.10；原 BASE 中心、半径 1；四个评分视图，完整 1 load +6 decode +2 VJP。** 新步长、调用账及以下恢复/数值边界尚待用户集中采纳。本轮只有提案和 CPU 算术，没有修改 runtime/config/notebook，没有新梯度或模型调用。

## 1. 依据与本轮要回答的问题

已读同次 `20261011T162703237651Z` 的 [主审](../../../diagnostics/b-line-all-bit-20261011T162703237651Z-audit/root_audit.md)、[32位表](../../../diagnostics/b-line-all-bit-20261011T162703237651Z-audit/reviewer_bits_32.csv)、[实际张量预算](../../../diagnostics/b-line-all-bit-20261011T162703237651Z-audit/root_tensor_budget_audit.json)、[执行/质量审计](../../../diagnostics/b-line-all-bit-20261011T162703237651Z-audit/root_execution_quality_audit.json)，并核对现有 all-bit runner、quality、replay 和实际 notebook/builder。此轮不重做原实跑审计、不重跑原两步。

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

理由是保留一次中点刷新，同时缩短单次未重新观察软保护的距离。按实际起点范数 `r=0.8436085787762565`，第一步三角上界 `r+0.10=0.9436085787762565<1`，故精确算术下第一步不触cap；第二步投影前上界 `r+0.20=1.0436085787762566`，仍可能触cap，必须按真实方向投影，最外向几何情况的累计缩放约0.958213664。`1−r=0.1563914212`只是径向差，**不是方向无关的剩余路程**。本选择没有由新响应、扫描或最优性证明选出；不能据上界预报新梯度朝向。

输入采用同次保存的两个实际FP32 terminal：

```text
ROOT=/content/drive/MyDrive/Video-WM/Local-Joint-State-Payload-V1-All-Bit/20261011T162703237651Z/run
z_BASE = ROOT/BASE/terminal_latent.pt
z_START = ROOT/ALLBIT_TWO/terminal_latent.pt
可选辅助记录 = ROOT/ALLBIT_TWO/delta.pt
```

首个新VJP直接使用 `z_START` 原值，绝不以 `z_BASE+保存计划delta` 替换。后者的计划范数为0.8436085832765362，实际范数为0.8436085787762565；两个位移向量相差L2约9.6053e−6。保存delta仅辅助定位，不是预算状态，缺失它不阻断可用的两个terminal。旧 `gradients/step_2.pt` 在旧HALF处计算，不能当新START梯度。

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

BASE评分失败仍保留其槽，若两个输入terminal可用且START可评分，可执行新方向但BASE相对质量明确缺失；START评分/构造失败不进入新VJP；MID失败保留已形成第一方向且第二步缺失。FINAL变差或未评分都保留终点槽。失败不凑齐预算、不把已有方向改成未定义。

## 4. 只恢复本次逻辑任务的已完成工作

现有 [runner](../runtime/wan/local_joint_all_bit_v1.py) 要求新目录，现有notebook每次建UTC目录；其 `finalize_interrupted` 仅读报告，**不具备此续跑执行能力**。[ReplayLedger/BoundarySpool](../runtime/wan/local_joint_readout_checkpoint_v1.py) 保存的是当次autograd graph的临时storage/cache，不是跨Colab会话恢复点。采用后需做窄的阶段恢复接线，不能只改旧入口输出路径后声称能续跑。

拟用一个可编辑的逻辑目录，默认 `MyDrive/Video-WM/Local-Joint-State-Payload-V1-All-Bit-Continue/20261011T162703237651Z-2x010/`；首次创建四槽，用户再次Run all读取同一目录的可用产物，从未完成阶段继续，**不先清空raw/metrics或已完成槽**。每次进程日志/实际调用独立保留在 `attempts/<UTC>/`。目录是恢复定位，不是来源验真；没有strict/override模式或自动重试。全部完成后再次Run all只报告已完成任务，不再跑两个新步骤。

**60秒 / 完成单元两层边界：**固定每60秒保存一次**最近已提交的可恢复状态及进度快照**，包含已完成单元、对应已落盘input/gradient/delta/terminal/raw/score的位置、下一恢复阶段、固定方法数值与各attempt的已知调用账；不能只写存活心跳。快照采用普通临时文件替换并**保留上一份可读快照**，实际写入失败保留原异常和旧可读状态。每个完整单元（gradient+输入、delta/terminal、float RGB、raw、score）完成即立即落盘，不等60秒；恢复以可读的完整单元为依据，不能因周期快照较旧而忽略已成功保存的gradient、重复其VJP。长decode/VJP尚未完成时，周期快照保存最近完成单元及当前运行中状态，不能凭它跨会话恢复半张autograd图。因此60秒是状态保存间隔，**不是最多丢失60秒计算的保证**；图未完成时损失窗口可覆盖整次梯度D+VJP。硬过期/实际Drive写入失败前来不及保存的区间按完成未知保留，不能伪造完成。无需将spool搬到Drive或每60秒重启单元。

每个新逻辑步及时保存：

- 一次普通原子保存的 `gradient_checkpoint.pt`，含**实际求导输入terminal与该masked gradient**（普通张量/字典），另存55值/q、loss/活动项及该次调用记录。只保存能读回的完整结果，不序列化autograd graph或64GiB spool。新方向只由该绑定输入和已保存gradient恢复；不是复用旧run的梯度。
- 由上述g作CPU归一化、原球投影后保存delta、实际terminal和step报告；普通decode后依次保存float RGB → raw → metrics/loss → quality/图片。两个新VJP都在自身finally清理cache/spool后再进入下一步，跨会话不保留中途cache。
- 恢复检查限于实际可读/shape/dtype/有限值、数值mask/原球语义和本阶段需要的数据结构；无哈希、签名、manifest、源码/依赖/Git身份相等。来源/环境仅定位。固定loss、key/message、mask及已采用步长仍是方法语义，恢复读取本逻辑任务保存的数值；不能用一次续跑暗改方法。

| 已持久化边界（此前BASE/START均已完成） | 过期后应做什么 | 新会话剩余 load / D / VJP |
|---|---|---|
| g1及其实际输入已完整保存，MID未评分 | CPU恢复第一候选；MID普通D、新g2梯度D/VJP、FINAL普通D | 1 / 3 / 1 |
| MID的float/raw/score已保存，g2未保存 | 只做MID梯度D/VJP及FINAL普通D | 1 / 2 / 1 |
| g2及其输入已保存，FINAL未评分 | CPU恢复第二候选，FINAL普通D | 1 / 1 / 0 |
| 四行float RGB均保存，raw/metrics/quality/PNG尚缺 | 只用已保存像素/原始读出完成CPU后处理 | 0 / 0 / 0 |
| raw已保存，只有metrics/loss缺失 | CPU独立truth reducer；不重解码、不重求g | 0 / 0 / 0（此阶段） |

若delta/terminal已保存，则直接读取；没有时可由完整gradient checkpoint作CPU恢复。score已保存则不重评分；质量/图片未完成且像素可读时只补CPU工序。raw/score完整但RGB损坏/缺失时保留评分及明确缺失质量，不能伪造像素或为更好质量自动重解码。这类实际I/O问题单独报告，不清掉成功VJP。

**不可恢复的边界：**VJP还在执行、只有gradient-forward/q/部分spool、只有未完成临时文件，或调用已返回但完整gradient checkpoint未落盘，都不能从中间继续autograd。用户手工恢复时只能在同一保存输入重新做该逻辑步的整个梯度D+VJP；不得从q/旧梯度拼方向。模型D返回而float RGB尚未保存时，未完成评分也只能重新普通D。保留先前attempted/returned/完成未知与失败，不将“没有落盘结果”写成“从未调用”。已落盘的零梯度/活动tie是构造结果，恢复不能重抽直到非零/无tie。

6D+2VJP是一次**不中断完整任务**的账，不是任意多次重启后的物理调用上限。发生上述丢失窗口，每次手工续跑先列可复用阶段、剩余调用及必须重做的调用；所有实际尝试累计，VAE每个需要模型的新进程至多load一次，纯CPU恢复不load。例：g1返回但文件未保存，重新求g1再做g2会有3次实际已返回VJP，必须照实保留，不能仍报2。逻辑更新仍只两步，无后台重试、自动加步或择优，是否启动中断后的恢复由用户执行Run all决定。

## 5. 逐位、质量和资源交付

每行保留22 correlations、G、32 margins、loss及32/state系数、活动项。固定报告上述六位各自的两次变化、距0/μ缺口及过零；同时报告全32位增减、获位/丢位、正位数、最差margin，原26正位的保留和f3b5轨迹，以及每个新视图当时最弱正位。若无正位记无，不从分母删掉。报告G、距γ及是否状态项重新活动；μ/γ不是检测或质量PASS线。不能以较低L盖过丢位、G回落或平均/峰值质量代价。

所有质量相对本次BASE：四行共724条逐帧及720条相邻残差记录，含RMSE/PSNR/max_abs/像素位置、ROI内外能量与峰值、时序RMSE/max_abs。沿用现有定位规则：各非BASE视图的frame RMSE/max_abs和temporal RMSE/max_abs峰值，最早并列取索引，帧峰值±1、相邻对两端各扩1帧，边界截断；与固定帧[1,44,88,112,116,120,132,176]取并集，对全部可用视图输出相同索引及峰值crop。不能把上一轮峰值85/127/120→121/160→161或38张帧图当作本轮预定峰值/固定张数。RGB统一[0,1]、残差[-.25,.25]且报告显示截断数；数值不截断，无质量PASS阈值、无codec。

同次已跑通的资源记录：总1780.19秒（约29.67分钟）；普通D约22.99–24.84秒；两次gradient-forward约228.58/973.42秒，VJP约201.90/207.98秒。峰值allocated19.01GiB、reserved21.72GiB、RSS3.37GiB；每份spool峰值64.53GiB，两次累计写/读各129.05GiB，最终live均0，CPU传输buffer峰值16MiB。迟滞没有profiler归因。**新步长变小不降低解码/VJP张量规模或保证更快**；沿用顺序独立spool，不能承诺新峰值/耗时，更不将累计I/O当同时驻留量。

只复用实际成功的 [all-bit](../runtime/wan/local_joint_all_bit_v1.py)、[quality](../runtime/wan/local_joint_all_bit_quality_v1.py)、[disk-spooled VAE](../runtime/wan/local_joint_readout_checkpoint_v1.py) 与 [notebook builder](../scripts/build_local_joint_all_bit_v1_notebook.py) 路径。真正实施时同步新真实notebook和builder，lazy load仅在还有模型调用时发生；恢复不能因只差CPU报告而再要求一次VAE/CUDA加载。没有GPU白名单、磁盘配额硬停、完整性链或A线全套状态治理。

## 6. 本轮完成的必要算术与待采纳项

[算术脚本](../../../diagnostics/b-line-all-bit-continue-proposal-20261012/proposal_arithmetic.py) 与 [结果JSON](../../../diagnostics/b-line-all-bit-continue-proposal-20261012/proposal_arithmetic.json) 只做：读取保存32位表及两个terminal/辅助delta的CPU范数，确认六位/弱正位/现有系数；代入唯一α=.10的几何上界与四视图/恢复预算；在人工二维向量上检验同一步长的内向、外向、反向几何；在人工FP32向量上验证确定性舍入。没有读取旧梯度来试新候选、没有loss梯度、decoder或新评分，也没有步长扫描。

人工边界例的最近FP32舍入会把范数1变成 `1.000000023841858`；规定的仅越界内侧修复将其变为 `0.9999999284744284`，位移舍入误差L2约9.70e−8，说明实际落盘预算不能仅看FP64计划值。这个数值例不是当前视频的新候选或方法成功证据。将来实施只需覆盖新增的实际起点/数值舍入、两步恢复阶段和丢失窗口的CPU替身检查，不重做未变的checkpoint整套数学审计。

请主审在同版方法/执行审查后，只集中呈报**一个采纳问题**：是否采用“保存ALLBIT_TWO实际起点、0.10×2、两次同目标新VJP、原BASE半径1实际累计投影及上述FP32边界处理、四评分视图/完整1load+6D+2VJP/352-5632-220、60秒进度与完成单元恢复边界”这一整组固定构造，并据此进入最小实现与用户Run all交付。旧μ/γ/权重/readout已采用，不重开参数菜单。
当前交付上限是“一个具体可审阅、可计价的续走构造及算术边界”，尚不是获采用的实现或可立即执行的新notebook。26/32不会自动触发codec、M1/M2、盲解码/FPR、main集成或科学PASS；真实效果必须由用户采纳后实施、运行并按完整固定分母判读。
