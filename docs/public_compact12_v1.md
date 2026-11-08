# 公开 Baseline 精简协议 V1

完整入口 `notebooks/paper_suite_v1.ipynb` 默认调用本协议；单独入口 `notebooks/native_baselines_public_compact12_v1.ipynb`。原 `native_baselines_public_fpr001_v4.ipynb` 保留，仍只继续原40视图包。本次不执行真实媒体、GPU或保存数据回放。

四配置不变：WAM32、PixelSeal256、VideoSeal256、ADAPTED FrameIndex（248 owner＋8 index）。同 Wan 十二源×四配置×五视图完全不变。

公开新执行名单：原 CAL500 的前32源（原全局序号65–96；28类），原 TEST500全部保留（565–1064）；DEV64只引用现有 post/质量结果，不启动新的DEV推理。原来源、全局序号、编辑种子和查询记录延用；DEV不会改名为TEST。名单在执行前固定，不看TEST成绩选视图。

|保留视图|含义|
|---|---|
|G1/T00、G2/T00|完整序列及再次编码对照|
|G2/T01–T04|四个相邻trim起点；覆盖mod4，不覆盖全部mod8|
|G2/T06、G2/T08|54帧随机删除、54帧连续删除|
|G2/T10|18帧块回放|
|G2/T14|1.25倍最近邻加速重采样|
|S/S00|H.264 CRF28|
|S/S02|中心裁剪后恢复原尺寸再H.26418|

每流1838帧、1826边；1个G1、11个G2。没有继续测：轻度删帧、点重复、重排、外源/合成插入、0.8倍慢速、H265、半尺寸、模糊、亮度、噪声、JPEG。只有T00成对比较G1/G2，不能为其它攻击单独归因编码影响。原40视图缺失/失败/未启动行留在原包，本协议不能把它变成完整。

阈值保持 WAM≥30/32、Pixel/Video≥161/256、FI≥165/248，不因族缩小而放宽。独立均匀query条件下每source×method×12视图族union界分别约0.00022638、0.00026618、0.00026618、0.00019399；四方法同源OFF族再union约0.00095273，不要求方法间独立，不是500源整包误接受界，也不是固定身份总体认证。CAL32只做独立null/实现sanity，不拟合阈值。若0/32，iid参考单侧95%上界约8.94%；0/500约0.597%。不能据此宣称总体0.1%尾部已被经验认证。

结果独立放在 `Video-WM_Result/baselines/public-native-frameindex-compact12-v1/attempt01`。缓存独立放在 `Video-WM_Cache/baseline-compact12-v1`，临时源码/权重在 `/content/video-wm-paper-suite-v1`。原v4结果和缓存不删除、不迁移。新cursor只解释新任务列表，不用原40视图游标索引。

只在当前任务需要时读旧记录，避免启动前全盘扫描。旧native按最后已保存epoch和原batch地址引用，不按分数选，不混epoch；完整raw无需模型/原媒体。旧partial只有原epoch输入仍可读且该batch从未启动时才可补未启动部分；旧已启动失败不因新目录恢复额度，丢失旧输入保留UNKNOWN。旧writer/已完成LPIPS不重算；旧partial质量只读保留，缺少旧precodec不会触发重写旧视频。canonical/G1可读则引用；新增decode按实际收费。进口记录独立列出旧STARTED，不充当新独立推理。

预算（有效旧记录复用前的最大新工作）：532次download/source_probe/normalize；30384次video_encode/probe；42088次decode；WAM/Pixel/Video reader243552/98048/66048，共407648调用、6609448帧；writer11500/500/1000/11500；模型load102；LPIPS72000pair调用（Alex144000forward）、6000阶段1086000帧对；物理43152视图、逻辑49536视图。JPEG/DiT/Wan/VAE/训练均0。基础设施恢复沿原规则：每单元最多2次额外，类别额外ceil(2%)，额外64 runtime/192 watermark loads/64 LPIPS loads；不能调阈值或换source。导入的旧失败不会获得新重试。

表格保留payload、ownership、time、quality/resources；FAMILY12表示任一接受，ALL12_ROBUST表示所有12视图接受且完整。错误query碰巧等于正确query时保留N/A而非假阳性。所有来源和缺失位置保留；DEV附录不参与新CAL/TEST完成判定，原生方法时间N/A。本包只汇总query/wrong_query与旧DEV saved参考，不新增旧v4 cross_source/DEV-donor重验证；旧表保留原包。

切换：本入口不会停止、接管或续跑原v4任务。请用户自行在已保存的视图/阶段边界停止旧notebook，再启动新入口；同一GPU不要并跑。`PREVIOUS_RUNTIME_STOPPED`仅指精简包自身前一runtime，不能替代实际停止。换runtime继续本包时依自身checkpoint续算。原完整v4可随时从其独立入口按原预算续接。

按2026-11-08保存的DEV任务/视图耗时（只读时间戳）推算，已覆盖主要成本约76.4小时；不含未测WAM写入、OFF下载/归一化、加载、最终I/O和恢复。可按约4–5天规划公开精简线，但这不是保证；质量成本约26小时，不能按12/40直接等比例缩总时间。整体suite还须另加未完成主方法TEST-B、固定同Wan对照和查询统计，当前无足够同Wan端到端实测支持给出严格总时长。
