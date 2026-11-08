> 2026-11-08 调度修订：运行顺序、真实checkpoint、分片参数和续跑边界以 [paper_execution_v1.md](paper_execution_v1.md) 为准。下文方法/统计协议保持，历史顺序执行说明不再作为当前运行指引。

# 低FPR主张、有限独立验证与统计口径

本增量不修改QMatch、DCT2＋CDC32、时间候选、旧H/L、载荷和旧正式输出。只补概率论证、48项保存记录派生统计及薄入口；不训练、不白化、不调半径、不删启动0、不复活F40–F44构造。

## 可证明的随机查询界

先固定全部接收证据E。每个view×key的最终输出，仅在32位全部合法可用时有一个可接受词，否则无词；技术输入缺失则词集合未知。一个source×arm只有五view×两key，接受词去重集合A的大小d最多10。三窗口、四phase、R0/R1、多个时间候选/引用都在不读取query的内部过程里聚合，最终每view×key只有一份32槽输出；U和P_PAIRED不进入QMatch族。

对在给定E后仍均匀的独立32位query Q：

`P(任一MATCH | E) = |A| / 2^32 <= 10 / 2^32 = 2.3283064365e-9`。

这是计数证明，不要求输出位均匀、位之间独立、视图或key之间独立。未知u个输入时，报告`[d/2^32, min(10,d+u)/2^32]`；只有u=0才报告精确概率。理论小界不表示缺失记录已经返回。

|随机查询事件|固定比较数|协议上界|
|---|---:|---:|
|单最终词、一个Q|1|1/2^32|
|单source×arm、同一个Q|10|10/2^32|
|同源三arm各一个Q|30|30/2^32|
|TEST-A原查询子集|120|120/2^32|
|TEST-B原查询子集|240|240/2^32≈5.58794e-8|
|原十二源随机查询整包|360|360/2^32≈8.38190e-8|

跨族项为union上界，不是整体精确概率，也不需要族间独立。它只覆盖独立随机查询包，不覆盖旧720条固定q0/q1声明、外部Baseline、任意后加身份/媒体/候选。对同一十词族另试J个仍独立于证据的均匀query，界扩为`min(1,10J/2^32)`；看过证据选query、无限试探或按query选候选不在此保证内。重读已存query不算新抽样。

1%、0.1%、10^-6三列仅比较同一个已证明上界，不构成三个新接收阈值。这里报告条件接受概率而非套用输出位二项模型的p值。上界低于10^-6可以如实报告，不需为了挂上0.1%再生成几千负例。

## 必须分开的概率目标

- 固定身份/固定key的内容FPR：`P_X(MATCH(X,key00,q0) | X无水印)`，需要指定内容分布；本项目当前只有固定面板实测，没有为任意自然总体证明0.1%。
- 独立随机query：上述严格条件计数界，适用于冻结OFF/P0/NEW证据。P0/NEW若Q恰等于真消息，保留抽样，但错误消息事件记N/A，不重抽；真消息未知单列技术未知。`|A去除M|/2^32`与条件于Q≠M的`|A去除M|/(2^32-1)`分开。
- 随机注册消息：若注册M独立于OFF及接收证据，同样可以对M取事前概率。当前C4固定q0=A6D39C5E、q1=26D39C5E、key00=watermark、key01=watermark-wrong，没有实际执行随机注册，不能追认。某个已实现固定身份或按该身份自适应构造内容不自动获此保证。
- 正例检出：NEW正确q0/key00的MATCH率、完全载荷恢复、正确时间/支撑归属分开。P0是身份有/时间无控制，其正确query不是身份假阳性负例。MATCH不证明精确时间认证。

## 有限独立验证与预算

2026-11-08只读Drive清单只有t01–t04目录；保存汇总t05–t12均NOT_STARTED。因方法未改，TEST-A仍是旧方案正式首批，标“既有已观察正式证据”；不作废其结果。本增量事前指定原TEST-B t05–t12八个固定prompt-seed为尚未观察的有限确认面板；不新增第13源、替换失败源或事后调阈值。

每源NEW正例、OFF无水印负例、P0身份有/时间无控制，各五视图双key双固定query。统计单位是来源，B为8，不是24个arm或120个视图。B继续原8×1114=8912操作、480固定声明、240份最终evidence；新增源/model/GPU/codec/receiver/post/query/位比较预算均为0。现有随机阶段仍整体一次36draw/360claim/最多11520比较；B的24draw/240claim只是其中子集，不能另建B查询包。

未消费原阶段由用户Colab执行；已消费或失败包只读，未知不补跑。有限验证检查正常检出、OFF实际接受、指定错误key/消息行为和接线假设；不凭位均值或两两相关宣称证明高阶iid。它不是以8个负例认证极小总体尾率：即使0/8，iid参考的单侧95%上限也约31.23%。预定prompt面板并非总体概率抽样，CP列只能标iid-reference。

48项新增确定性报表：ALL/A/B×12个固定arm/key/query条件=36项（各含五视图、ANY5/ALL5/至少4视图统计）；ALL/A/B×3随机arm=9项；三个面板random-only union=3项。固定声明输入720行、随机声明360行、随机族36行复用旧结果；这些派生项不是48次新实验，不再次匹配bits或后评。每次新报表快照保留全部缺失槽。

ANY5有一个MATCH则真；无MATCH但有技术未知则未知。ALL5全部MATCH才真，已知非MATCH则假，其余技术未知保持未知。至少4/5同理：已接受a、技术未知u，a>=4为真，a+u<4为假，否则未知。MISMATCH矛盾拒绝、UNRESOLVED逻辑未决、INPUT_UNRESOLVED技术未知始终分列。不接受不等于成功矛盾拒绝。

结束条件：固定八源及原查询包达到各自终态后交付全部成功/失败/未决，按原C4判据评估主张并转正式测量汇总，不因普通覆盖下降再无限加源。OFF固定身份反例直接计实测FP并限制该经验结论；随机query观察不验证或推翻另一概率空间。真实错误不得靠理论小界隐藏。

## Run-all和表格

- 完整论文入口：`notebooks/paper_suite_v1.ipynb`，主方法十二源、同Wan及原查询预算不变；public默认CAL32/TEST500×12，原v4独立入口保留。
- 有限确认入口：`notebooks/paper_fpr_validation_v1.ipynb`，共享runtime，仅cohort→tables→原queries→tables；不启动外部Wan或public模型。GPU与CPU均在Colab内完成，无下载回本地续接。
- `fpr_probability_models.json/csv`：概率空间、假设、实际执行与未采用模型。
- `fpr_measurements.json/csv`：48项，逐视图/ANY5/ALL5接受率、四态、N/A、源分母与区间、随机查询概率和union界。
- 时间和载荷正确性仍来自`main_time`、`main_time_strata`、`main_payload_contributions`已保存后评，不由MATCH替代。

结果继续落在`Video-WM_Result/paper_suite_v1/reports/<新快照>`，原包不覆盖、不删除、不清理。论文正文和图表应写“独立uniform32查询的条件接受上界＋固定内容面板经验表现”，不要合并为“固定身份在任意视频总体FPR已认证”。正在运行的public Baseline保持原预算和checkpoint。

## 文献依据与区别

[Stable Signature主文§3](https://openaccess.thecvf.com/content/ICCV2023/papers/Fernandez_The_Stable_Signature_Rooting_Watermarks_in_Latent_Diffusion_Models_ICCV_2023_paper.pdf)以公平独立提取位的零假设使用二项尾；[补充B.5–B.6](https://openaccess.thecvf.com/content/ICCV2023/supplemental/Fernandez_The_Stable_Signature_ICCV_2023_supplemental.pdf)讨论非理想分布和经验核验。不能把白化或低两两相关当严格高阶独立证明。

[VideoSeal v1 §3.1/A.1](https://arxiv.org/html/2412.09492v1#A1.SS1)显式使用iid公平位的理论p值作payload比较，并说明其解释边界。[Gaussian Shading §7.1](https://arxiv.org/html/2404.04956v1#S7.SS1)还利用流加密伪随机性作为统计模型依据；其计算假设不能直接移植为当前CDC输出已证明的性质。本增量采用独立query的计数证明，因此不需要添加这些未经核验的提取位假设。
