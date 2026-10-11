# B 线全 32 位余量目标：具体提案

日期：2026-10-11；审阅版本：v2。当前 B 分支为 `dev/local-joint-state-payload-v1`，本轮起点 `77917657ddf1543b4a09a60a9437873dd78fef42`。这是已采纳方案的设计记录；用户已回复「采用这组参数，进入最小实现」。当前实现与验证见 [交付说明](local_joint_all_bit_v1_delivery.md)，未调用真实模型。

**首选：从同一原始 BASE 重算两次新标量梯度，以覆盖全部 32 位的正余量短缺损失替换“四个当前最弱位的平均”；达到目标余量的位不再被持续放大，接近零的正位提前进入保护；G 达到小正下限后不再主动增长，跌回下限才恢复惩罚。**保留同一 mask、readout、两个半步和相对原 BASE 的 L2≤1，主比较为已经完成的 TWO_MEAN。

“全位进入、保护近零正位、状态达标后以维持为主”的方向已采纳。下述 **μ=0.02、γ=0.01、payload/state 权重4/5与1/5、两步各0.5、6D+2VJP** 是一组已获用户完整采纳的固定数值。它们是设计选择，不是已有成功参数或新检测阈值；只提交这一组，不扫描备选。本次按采纳范围实施用户 Run all notebook；真实执行由用户完成。

## 1. 已完成结果指出的具体缺口

最新同次来源为 Mean Refresh run `20261010T145910398211Z`。已读 [主审结果](../../../diagnostics/b-line-mean-refresh-20261010T145910398211Z-audit/root_audit.md)、[独立数值审查](../../../diagnostics/b-line-mean-refresh-20261010T145910398211Z-audit/review_b_numeric_method.md)、[完整32位表](../../../diagnostics/b-line-mean-refresh-20261010T145910398211Z-audit/review_b_bits_32.csv) 与 [预算/质量审计](../../../diagnostics/b-line-mean-refresh-20261010T145910398211Z-audit/root_execution_budget_quality_audit.json)。[同次 Drive 目录](https://drive.google.com/drive/folders/1mQXPtrXmfQqsIP6PgHuHC3YITDig_KIR) 是用户结果入口。本轮只读本地已取回记录，未重新解码或补跑。

| 同次视图 | G | 正 signed margin /32 | 全部位最差 margin | BASE-relative RGB RMSE |
|---|---:|---:|---:|---:|
| BASE | -0.001996996 | 16 | -0.429410042 | 0 |
| ONE_MEAN | 0.014166796 | 17 | -0.364628717 | 0.002952199 |
| TWO_MEAN | 0.017267412 | 16 | -0.280612629 | 0.002448107 |

TWO_MEAN 对 ONE_MEAN 的 G 和四片段最弱 margin 全部提高，但32位只有17位变好、15位变差；15个原非正位没有一个过零，fragment3/bit7 由 `+0.006972974` 降到 `-0.000928505`。四个中点最弱位均发生转移，其中三个新瓶颈位在第二半步自身略变差。原目标只对4个当前最弱位和G反传，不能据此声称测过32个位梯度的冲突。

本次440窗口、7040 chips、275指标全部从原始能量重算一致，执行完整不是主要缺口。TWO_MEAN 报告的累计 BASE-relative L2 为 `0.8198027002`，没有把剩余半径补满。这个范数来自已保存 runtime 记录及实现核对；主审的大张量下载未完成，不能写成独立张量范数复算完成。

质量也存在取舍：TWO_MEAN 的 RMSE 较低，但全局 max_abs `0.226869702` 大于 ONE_MEAN 的 `0.198389888`；ROI外残差能量占约13.89%，全帧RMSE峰值从116移到92。新方案必须导出实际峰值及邻域，不能仅查看此前的116帧。上述正位数只描述已知消息下的 signed margin，不是盲解码准确率、BER或FPR。

## 2. 与历史 B2 全位损失逐项比较

直接核对历史 [B2 固定任务卡](../../RGB-DCT-B2-Terminal-Receiver-V1/docs/rgb_dct_b2_terminal_receiver_protocol.md)、[全组可微 reader](../../RGB-DCT-B2-Terminal-Receiver-V1/main/tube_state/rgb_dct_terminal_gradient.py)、[终端写入 backend](../../RGB-DCT-B2-Terminal-Receiver-V1/runtime/wan/rgb_dct_b2_terminal_receiver_backend.py) 和 [同次真实结果审计](../../../diagnostics/receiver-first-20260923/rgb_dct_b2_terminal_receiver_result_audit_20260927.md)。原始 JSON 为 [b2_terminal_receiver_result_20260926T155237439193Z.json](../../../diagnostics/receiver-first-20260923/b2_terminal_receiver_result_20260926T155237439193Z.json)。旧文件里的 hash/身份要求不进入本方案。

| 项目 | 历史 RGB-DCT B2 | 本提案 | 可以与不可以据此解释什么 |
|---|---|---|---|
| 优化对象 | 30组 q 全部进入 `mean(relu(-q)^2)`；不是只取最差组 | 当前B的32个消息 signed margins全部进入同一个损失 | “全位损失”与真实VAE梯度已存在，不构成新意；只检验当前五目标所漏掉的位覆盖 |
| reader | 时空分组的 DCT 特征相关 q，含中心化与能量归一 | 现有22×4局部窗口，512样本能量池化比值，再聚合成4×8 margins | 不能比较两者 q 的绝对数值、符号组数或成功率；不把30组判别当32位消息恢复 |
| 起点正位 | q>0 的自身损失和梯度均为0；没有正余量 | 0<m<μ 有连续保护梯度，m≥μ 才退出；所有位随当前位置重新计算 | 保护是软目标，不能保证其他项更新不会把一个正位推负 |
| 深负位影响 | 导数幅度随负缺口增长，`2(-q)/30` | 所有m≤0的单项导数封顶，同量级覆盖浅负与深负位 | 防止仅由最深缺口按标量系数主导；不同位的真实输入梯度范数仍可不同 |
| 状态 | 同一30组presence构造，无当前G/分片分离 | G 单独使用下限惩罚；达到γ后不主动继续增大 | 是当前B的资源分配假设，不保证有限步保持G≥γ |
| 更新/预算 | 同史native T49的一次方向，归一到 support RMS R*=0.0429433127；时序1:45全空间 | 从当前同一原BASE直接改终端，现局部mask；两次重算，每次0.5，累计L2≤1 | 同一B新旧比较可用；不能把B2约57.65的全支持L2与当前局部L2=1当等预算 |
| 真实代价与观测 | 真FP32 VAE VJP；float、RGB8、MP4；原判别C≥24 | 复用已跑通FP32 VAE缓存重放；只看float与当前原描述条件 | 本轮不声称codec、盲同步或轨迹链通过 |

B2 有效负证据必须保留：copperkettle/paperplane 的 float C 均为22，MP4为21/23；float损失分别是OFF的约1772.17/192.60倍，尽管记录的一阶 g·Δz 均预测下降。旧正组分别丢7/6个，终态R*实际到位且没有后续自由尾程。故“完整VAE梯度＋全位损失＋固定范数”从来不保证有限步效果，也不能把该失败单独归为尾程或codec。

本方案同时改变标量的覆盖/保护形状，并将当前已完成的两半步结构保留为对照。若有效，只能说明这组目标在当前终端/预算上比原五目标更合适；不能把收益单独归因于 μ、导数封顶或G权重中的某一项，更不能宣称换loss本身是创新或路线已成立。

## 3. 一个明确的首选标量

维持当前 `m_i`（i=0..31）与 `G=c0−max(c1..c21)`，不改读出、消息符号、q分母或证据池化。对任意正目标a定义正余量短缺函数：

```text
H_a(x) = 0                    , x >= a
         (a-x)^2 / (2a)       , 0 < x < a
         a/2 - x              , x <= 0

dH_a/dx = 0                  , x >= a
           -(1-x/a)          , 0 < x < a
           -1                , x <= 0
```

这是以a为过渡宽度的单边 Huber 形状；在0、a都一阶连续，不需另设温度、epsilon或任取边界导数。首选：

```text
μ = 0.02                       # payload优化目标余量；非检测阈值
γ = 0.01                       # state优化目标下限；非科学PASS阈值
L_payload = (1/32) * sum_i H_μ(m_i)
L_state = H_γ(G)
L = (4/5)*L_payload + (1/5)*L_state
```

32项始终在求和中；m≥μ时该项梯度为0，若后续降回μ以下便重新激活，并非永久锁定一批位。payload内部不按片段、正负历史或样本选择权重，每位一律1/32。每个当前非正位的标量梯度系数幅度是1/40；接近零的正位系数连续趋近1/40。例如m=0.006972974时单项保护系数为非正位的约65.14%，不会像B2的零余量平方hinge一样在起点完全不受自身项约束。这里只计算标量系数，未测真实位梯度或新候选。

μ=0.02是小正余量缓冲的首选设计，γ=0.01是把状态从无界增长改成小正下限的首选设计；二者没有由噪声分布、FPR或优化扫描校准，也不因为旧TWO_MEAN的G较大就把它宣布“通过”。权重4/5与1/5保留原“五项平均”中payload四项与state一项的名义块权重；现在payload份额平均分给32位。state在G≤0时系数最多为0.2，相当于8个处在封顶区的payload标量系数，但真实输入梯度强度仍取决于decoder Jacobian。这个选择可审阅，不是已经验证的最佳平衡。

G≥γ后 `L_state=0`，其梯度为0；若第二步起点G跌回γ以下，惩罚重新启用。这是“维持为主”的离散软实现，不是硬约束；若最后一步跌破下限，固定两步内不会自动追加修复。最终必须同时报告G与所有位，不能用较低L覆盖G跌落、丢位或质量代价。

已有数据提供一个直接限制：把这一个新L事后代入旧保存metrics，ONE_MEAN为`0.0497660613`、TWO_MEAN为`0.0413522028`，损失下降但正位仍17→16。两者活动payload项分别18（15非正＋3近零正）、16（16非正＋0近零正）。因此即便新L下降，也不能推出符号覆盖改善、G保持或原描述条件成立。见 [单组CPU算术脚本](../../../diagnostics/b-line-all-bit-proposal-20261011/proposal_arithmetic.py) 与 [完整数值](../../../diagnostics/b-line-all-bit-proposal-20261011/proposal_arithmetic.json)：只有既有指标代入、解析导数核对与预算算术，没有新梯度、候选、模型调用或参数扫描，也没有据此更换本组数值。

## 4. 两个新梯度步骤与同一 BASE 球

设同一保存原终端为z0，当前固定mask为M，δ0=0。首选固定k=0、1：

```text
zk = z0 + δk
Yk = actual_frozen_FP32_Wan_decode(zk)   # normalized缩放、内部clamp与RGB转换均在图内
gk = M * grad_z L(Yk)                   # 一次标量decoder VJP
uk = -gk / ||gk||2                     # 只归一化整个标量梯度
candidate = δk + 0.5 * uk
scale = 1                         if ||candidate||2 == 0
        min(1, 1/||candidate||2)   otherwise
δ(k+1) = candidate * scale
```

投影中心始终是原z0，绝不以中点为新预算中心；不把不足1的范数补满。两个长度0.5的增量理论上已满足三角不等式上限1，投影只保持累计球约束及数值实现一致。记录计划δ与实际保存terminal−z0的FP64范数、mask外分量、每步实际位移和投影系数；FP32加法/舍入如产生微小差异，如实报告，不能假称严格实数精确等式，也不能据记录差异添加身份门禁。

**第一步必须在z0对新L重新VJP，第二步必须在新z1对同一个新L再次VJP。**旧M0 MEAN或旧Mean Refresh首半步只属于旧目标，不可借作新目标的第一步后还称为完整两步比较。每步只求一个标量VJP，不做32次位梯度、COMMON凸包或逐目标单位化。

两步均按固定顺序；不根据L、过零位数、G或质量择优、回滚、提前接受、缩步或追加步骤。若第二步变差，仍交付第二步及中点。不产生后续方向的情况包括输入shape/数值语义无效、零pooled support或reader失败、非有限、真实读写/加载/OOM/decoder/VJP失败、零masked梯度，或有梯度作用的state max并列；按实际缺失/工程失败/构造未定义分别保留。不能拿零delta冒充未产生的方向或增加备用VJP。若两个有效非零梯度方向恰好相消使candidate=0，那是合法的累计位移结果，按上式正常保存和decode，不能误判成零梯度或UNDEFINED。

payload最弱位并列不再决定新目标可微性：32项H均一阶连续，“四个min”仅保留描述。对于G，只有G<γ且最大错误offset并列时状态项仍需要一个未唯一确定的梯度，此时保留新依赖行UNDEFINED；若G≥γ状态项平坦，wrong-offset并列只记录，不应误用旧五目标的全局`tied`开关阻断。0与μ、γ边界用上面解析一阶连续定义。该精确活动范围与新数值一并提交采纳，不静默沿用不适用的旧min规则。

## 5. 固定四视图与真实调用账

| 新run固定行 | 输入 | 用途 |
|---|---|---|
| BASE | z0 | 同次float基线与新第一步起点 |
| TWO_MEAN_SAVED | z0 + 已完成Mean Refresh的TWO_MEAN/delta.pt | 同一实际decoder下的两半步旧目标对照；读取保存delta原值 |
| ALLBIT_HALF | z0 + δ1 | 新目标第一半步；第二次VJP起点 |
| ALLBIT_TWO | z0 + δ2 | 新目标固定最终行；主比较对象 |

主比较为 ALLBIT_TWO 对 TWO_MEAN_SAVED，另存每行对BASE及 HALF→TWO。不再追加ONE_COMMON/COMMON refresh。历史五视图完整报告仍可链接，但不混进新run分母。旧对照累计半径约0.8198，新方案可能不同；都是同一个L2上限，不能称实际位移完全相等或只凭更高PSNR把原因归到新loss。

完整非退化预算首选为 **1次VAE load、6D、2 decoder VJP；0E/DiT/native/scheduler/codec**：四次普通评分decode，加z0/z1两次梯度decode。普通chunk=4×46=184；梯度原始chunk=2×46=92；名义replay另92。每次梯度decode的重复读数只是工程一致性诊断，不算新样本；普通评分decode不由梯度前向冒充。

新run固定分母为 **4×88=352 windows、5632 chips、220原指标**（每行22 correlations、G、32 margins），每chip512样本。新loss、四片段min、系数、质量、资源为额外描述字段，不改变原指标分母。BASE与保存旧对照先按各自输入独立执行；若旧delta缺失，该对照明确MISSING，新目标依赖仅z0仍可继续；不得重跑旧Mean Refresh来补对照。若新方向退化，保留四行和实际attempted/completed，不凑满调用。

已完成Mean Refresh本轮只用了1个新增VJP，是因为首方向复用了M0；那不是新目标完整两步可以只记1VJP的依据。两方案若比较额外资源，必须同时说明旧方向的前期成本没有包含在其6D+1VJP运行账里。

输入定位沿用原BASE路径与新对照路径：

```text
/content/drive/MyDrive/Video-WM/Local-Joint-State-Payload-V1/20261009T132316545817Z/run/joint/terminal_latent.pt
/content/drive/MyDrive/Video-WM/Local-Joint-State-Payload-V1-Mean-Refresh/20261010T145910398211Z/run/TWO_MEAN/delta.pt
```

路径可编辑；只核实际可读、shape/有限值、method key/message/mask/累计预算等真实语义。保存旧对照不重归一化、不补半径。可选来源路径记录不作为运行门禁。

## 6. 资源依据和不能承诺的量

复用当前 [Mean Refresh runtime](../runtime/wan/local_joint_mean_refresh_v1.py) 与 [cache/recompute/disk边界](../runtime/wan/local_joint_readout_checkpoint_v1.py)，参数冻结FP32、输入图和跨chunk cache保持连接，两次VJP顺序执行，每次结束释放图与暂存，不同时持有两份图。

当前同次Mean Refresh的 [result.json](../../../diagnostics/b-line-mean-refresh-20261010T145910398211Z-audit/result.json) 实测：

- 6D+1VJP+1load总runtime约610.876秒；五个普通decode各约22.90–24.89秒，梯度decode约230.799秒，VJP边界约209.562秒。这些边界不含所有后处理/I/O。
- CUDA allocated峰值20,417,041,408 bytes（约19.01GiB），reserved 23,318,233,088 bytes（约21.72GiB）；进程峰值RSS 3,462,172 KiB（约3.30GiB）。都是该次进程报告，不能与历史B2约38.59GiB RSS混用。
- 单次VJP暂存峰值69,284,167,680 bytes（约64.53GiB），disk_written/read与D2H/H2D各同该值；closed=true，结束disk/CPU packed live均0，CPU transfer buffer峰值16MiB。

新计划同为6D，却把一次普通decode换成一次完整梯度decode+VJP，代价必需按两次VJP完整记账。若边界规模与执行路径相同，两次顺序VJP意味着约两份上述读写流量，而非两份同时驻留；这只是以当前记录作容量/I/O解释，不是新测峰值、磁盘配额或耗时保证。不得用6D数量相同暗示耗时相同，也不把普通无梯度路径当成梯度内存证据。新loss虽只改reader标量，真实峰值仍须用户运行报告。

保留actual call与chunk attempted/completed、加载与各decode/VJP边界时间、GPU allocated/reserved、RSS、磁盘写读/peak/live、cleanup及失败阶段。没有GPU白名单、版本相等、磁盘配额预检硬停或自动重试；真实分配/写读错误按原槽保留。

两次梯度forward分别对照同输入的普通读出：z0对应BASE，z1对应ALLBIT_HALF。保留各自55个原指标与活动项、逐项差和最大绝对差，有限数值差只作工程诊断，不变成版本/身份相等门禁；真实非有限/支持失败仍按方法含义记录。每次VJP创建独立ReplayLedger与spool，在finally释放后再进入下一步。

## 7. 逐位、有限步和实际峰值质量输出

每行保存全部原始能量/q/support、22 correlations、G、32 signed margins和四片段min；保存L、L_payload、L_state及32个实际标量导数系数、状态系数。逐位表固定32行，列出对BASE、对旧TWO_MEAN和HALF→TWO的变化，所有非正位的过零缺口、负转正和正转负集合、最差margin及正位数；不得只展示最弱片段或均值。

每步记录masked scalar gradient范数、方向和实际累积delta/terminal，保存单一预测 `g_k·(δ(k+1)−δk)` 与实际 `L(z(k+1))−L(zk)`。这是该标量在当前步的一阶预测，不是32个分量预测，不声称获得各位梯度。新目标对照需同时列：是否新增过零、是否丢旧正位、最坏缺口、G与其状态下限、平均margin/L以及质量；低L本身不能宣布成功。

原描述条件仍是G>0且32/32 margins>0；μ、γ只是优化偏好，不替代该条件。即使达到μ/γ，也只支持这个已见保存终端点的float已知消息描述，不能晋级为盲解码、codec、FPR、轨迹或科学PASS。

质量始终相对同次BASE：全局/逐帧RMSE、PSNR、max_abs及峰值位置；ROI内外能量/峰值；180条相邻残差的RMSE/max_abs。补齐现有逐帧缺失的max_abs字段。保存181帧全部数值。

图片导出规则预先固定：每个非BASE视图分别定位“逐帧RMSE最大”“逐帧max_abs最大”“相邻残差RMSE最大”“相邻残差max_abs最大”。同值用最早索引，仅作为可视化定位；帧峰值输出t±1，时间峰值输出该相邻帧对及两端各一个邻帧，边界截断。把各视图定位集合取并集，再给所有四视图输出同一批帧对照，另保留原固定帧；零残差行不扩成181帧并列集合。输出最大像素坐标及对应局部crop，完整数值报告仍保留所有帧，图片定位不选择候选或排除数据。

RGB显示统一[0,1]，残差图统一固定[-0.25,0.25]并标注超显示范围像素数，数值度量始终用未截断float；此范围仅是图像显示比例，不是新增质量门槛。实际峰值可能仍大，平均PSNR改善不等于视觉通过。没有新增自动质量PASS或模型codec操作。

## 8. 已采纳的最小实现设计与必要检查

用户已采纳全部具体参数；本次增加一个窄的all-bit方法模块和独立runner/config/CLI，调用既有张量读出及VAE资源路径；保留旧M0/Mean Refresh结果和入口。明确绕开不再适用的payload-min `tied`门，不改原方法的既有行为。

只需覆盖新增语义的检查：H的分段值/边界一阶导数、32项均匀求和和近零正位系数、G平坦区及活动max并列；同一BASE两半步累计投影和新第一步不借旧方向；CPU小型因果decoder的两次标量VJP/独立reader对应与中断槽位；峰值帧/邻域并集和完整数值保留；全新notebook命名空间顺序替身。已证明的checkpoint alias/cache数学不因新增loss而重复整套审计，除非接线改动触及它。

用户Run all notebook同步真实builder与普通ZIP，沿用安装、source下载、loader、logger、子进程清理；输出新UTC目录；不引入B64/hash/manifest/签名、Git干净状态或精确依赖版本门禁。真实GPU仍由用户执行。

## 9. 已完成的一次集中采纳

主审已将以下作为**同一组具体方案**呈用户决定，用户回复「采用这组参数，进入最小实现」：

1. 采用第3节单边Huber余量短缺标量：μ=0.02、γ=0.01，payload/state=4/5与1/5，每位均匀；状态是软下限，不承诺硬维持。
2. 从原BASE开始两个新的标量VJP步骤，各0.5，累计L2≤1；采用第4节活动state并列/零梯度/失败保留规则，无扫描、回滚或补满。
3. 采用四视图固定矩阵和1load+6D+2VJP预算，普通184/梯度92/replay名义92；旧TWO_MEAN只读取保存delta作对照。接受第7节实际峰值及邻域输出，不新增质量阈值。

此前提案只支持“已把方向具体化为可审阅、可计价、与历史负例有明确区别的固定构造”；当前采纳授权实施、必要验证及工程交付，不意味着真实运行或方法成功。COMMON昂贵刷新、放大cap、M1/M2、学习读出、扫描择优和main集成均不在本方案内。
