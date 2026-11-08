# 完整论文实验入口 V1

选择GPU后运行 `notebooks/paper_suite_v1.ipynb`。默认矩阵是完整十二源及四配置对照，不是单独TEST-B。所有模型、媒体、接收、CPU统计与后评都在Colab中接完；Notebook只负责挂载、输入、调用和展示，builder同步。真实成功与工程交付分开。

## 已接入的实验与预算

|部分|固定范围|预算与既有结果处理|
|---|---|---|
|主方法|C4十二源、三臂、五视图、双key双q|每源1114操作、全12源13368；已完成TEST-A读取、未启动TEST-B执行；不重跑消费源|
|公开自然视频|原v4 DEV64/CAL500/TEST500，四配置40视图|完整继承原budget/checkpoint及恢复额度；不重建已完成媒体/推理；详见独立v4文档|
|Wan同域外部对照|同一十二源、四配置、OFF/WM、五视图|每源534外层操作；model load3、writer49、reader350、encode20、probe26、decode26、quality20、post40。全12源6408操作；writer588、reader4200、codec encode240/probe312/decode312|
|主QMatch统计|十二源×OFF/P0/NEW，各一个uniform32查询，五视图×两key|36随机抽取、360声明、最多11520位比较；额外计算36个接受词集合的精确概率/未知界；无新model/codec/读位|
|贡献与表格|原B/U、P_PAIRED、OFF/P0/NEW及F42/relative保存输出|只读保存后评进行分栏汇总，不重跑旧后评、不新增身份分支、不重复已失败构造|

49和350是adapter调用机会，不冒称真实网络forward次数；模型hook实际调用与帧数单独保存。Wan物理reader为每源5754帧（7×822），全批69048；逻辑OFF帧复用不增加物理调用。Pixel OFF raw同时供Pixel与FrameIndex，不能读两次GPU。三个模型按源顺序加载释放，最多36次load；每源一次attempt，失败不补跑。新预算独立于旧C4和公开v4，不能转借。

## 同域比较的确切含义

四个外部配置共同输入已保存C4 OFF/first.mp4解码RGB；不重建已释放的precodec缓存。写入后直接作同五编辑并执行一次与C4相同命令的H26418编码，成为最终第二代媒体，不再先做一次WM首代编码。现成OFF五个received.mp4直接作为负例/质量参考。主方法使用已有NEW二代结果。

这是同源、共同编辑和最终媒体协议的pipeline比较。主方法在生成期/原生解码链写入，外部方法在首代后处理写入；不能称相同precodec像素干预或等容量排名。WAM32、PixelSeal256、VideoSeal256和ADAPTED FrameIndex（248owner+8index）保留原生位数、阈值、真实尾批和错误query。除FrameIndex外的原生时间列N/A；不冒充已发表原生时间定位方法。普通PSNR和成本足够覆盖本轮有限主张，不增加LPIPS/训练或感知优势主张。

## 主方法0.1%口径

QMatch四态算法不改：有合法矛盾MISMATCH，32有效位全等MATCH，无矛盾但不足32位UNRESOLVED，关键技术输入失效INPUT_UNRESOLVED。query不参与时间/候选/支持选择。

每个source×arm从已冻结的五视图×两key读取最多10个完整32位可接受词，去重为集合A。对独立均匀32位外部query，族任一MATCH概率为|A|/2^32≤10/2^32，远低于0.001。缺文件/技术失效不是空集合：保留已知下界、最多10词上界，exact_probability为未知。该族统计不改变单声明四态，也不选有利消息。

每source×arm一次独立os.urandom(4)，36次一次抽取并保存，不因碰巧等于真实q0重抽。360条声明全部保存后才读取实际writer信息；P0/NEW的正确query碰撞不计错消息。另报告|A去除真实消息|/2^32，以及条件于错误query时分母2^32−1的概率。OFF不存在自身消息。实际固定q0/q1/错误key控制按旧结果单列，不能套用随机query证明。

统计单位是来源，每arm分母12，视图/key/声明不扩大来源数。TRUE/FALSE/UNKNOWN及四态计数分开，未决是非接受但不是矛盾拒绝。Clopper–Pearson区间仅作为iid-Bernoulli参考；这12个预声明prompt是有限面板，不是自然总体随机抽样。0/12单侧95%上界约22.09%；本交付不声称自然总体FPR置信界≤0.1%，不擅自添加2995或416个新源。

## 结果和恢复

- 原主方法：`Video-WM_Result/main_method/paper_cohort_v1`，旧包保留。
- 原公开Baseline：`Video-WM_Result/baselines/public-native-and-frameindex-fpr001-v4/attempt01`，原checkpoint优先；已有独立runtime ACTIVE时不抢占，不改其包。
- 新Wan：`Video-WM_Result/baselines/wan-cohort-five-view-v1/<source>/attempt01`。
- 新查询：`Video-WM_Result/main_method/random-query-family-v1/attempt01`。
- 统一表格与阶段日志：`Video-WM_Result/paper_suite_v1`，每阶段新快照，latest_report.json仅索引最新快照。

新的Wan/query开始或失败消耗，重入只读；尚无上游终态的Wan源保持DEPENDENCY_PENDING，不提前消耗。C4任一源尚未明确终态时，查询统计保持DEPENDENCY_PENDING且不消耗attempt。旧C4已消费但缺文件不会被重新生成/补post。显式中断停止后续阶段；普通失败保留固定分母，独立工作可继续。

公开v4仍用原计算游标和原有限基础设施恢复预算。若其它runtime仍活动，统一流程报告EXTERNAL_RUNTIME_ACTIVE并保留其运行；实际停止后才允许用户设置PREVIOUS_RUNTIME_STOPPED。每阶段子进程隔离依赖和GPU资源。源码/权重在`/content/video-wm-paper-suite-v1`及原缓存区，均在结果树之外；本入口不清理正式包。

## 自动表格

每次表格快照含JSON+CSV：main_time、main_payload_contributions、main_fixed_claims、main_fixed_source_statistics、main_random_family/claims/source_statistics、wan_native_comparison、wan_source_statistics、quality、costs、components和public_baseline_summary。另含十二源汇总和公开Baseline原四张完整表的定位索引。缺失行保持TECHNICAL_UNKNOWN，不由报告层重新计算科学读出。报告不是“全部正结果”判定器。

纯方法只依赖main；runtime只依赖方法/共享适配，公开v4默认协议放在main/baselines/protocols，所有子进程入口在runtime；实验CLI与薄notebook可拆卸。没有SHA/manifest/Git-clean/精确版本准入门禁。


来源接受事件明确标为 `ANY5`，随机查询族为 `ANY10_VIEWS_KEYS`，不能误读为全部视图通过。正确消息碰撞保留 N/A，分列计划、eligible、技术未知，不重抽、不计错误接受。另分 ALL、TEST-A、TEST-B 来源汇总。查询失败或缺失时仍展开360声明和36族，已返回逐条声明直接读取，不重跑匹配。贡献表包含B/U/P_PAIRED绝对支撑、完整字面/归属消息、启动区和真实事件分层；成本连接已保存阶段资源、I/O、scratch记录。格式异常逐项列出并保持未知，其它来源仍可读。
