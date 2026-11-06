# 四配置公开视频 Baseline v4

本入口实现已采用的 UCF101 DEV64、CAL500 OFF、TEST500 OFF＋四种 WM；每来源181连续真实帧、25fps、320×512。它是独立的公开自然视频对照包，不执行 Wan，也不替主方法定义接受规则。工程交付和真实效果分别报告。

## 运行入口

- `notebooks/native_baselines_public_fpr001_v4.ipynb`：选择GPU后 Run all，首代码cell只有Drive mount。
- `scripts/build_native_baselines_public_fpr001_v4_notebook.py`：生成同一实际交付 notebook。
- `python experiments/wan_state_clock/native_baselines_public_fpr001_v4_run.py --vendor /path/to/vendor`：完整独立CLI；vendor含官方videoseal和watermark-anything源码及WAM权重。
- `experiments/wan_state_clock/configs/public_native_fpr001_v4.json` 和相邻 `.sources.json`：固定语义与名单。

薄notebook下载普通可编辑源码、检查实际依赖导入、准备官方权重、调度所有来源与后评。源码位置、ref和依赖记录不作哈希或精确版本准入。真实格式、方法参数、数值、来源分组与盲读隔离保持必要检查。CLI不需要Git仓库或notebook状态。

## 数据与方法

既有CAL500/TEST500字段不改；在原1145合格(action,group)中剩余145组，按action/group排序后用PCG64(2026110601)选64个DEV。统一global_ordinal按DEV/CAL/TEST编号；失败不替换。头格式可读不等于完整181帧可解码；运行时实际检查。这个长片子集不代表完整101类别无偏总体或预训练未见。

| 方法 | 写入和原始读取 | 给定query接受 |
|---|---|---|
| WAM MIT32 | FP32逐帧B8，中心1/4矩形；官方semihard选mask像素，保存原生bits、valid、count、float64 sums/means | 至少一帧32位匹配≥30 |
| PixelSeal256 | 官方pixelseal权重，step8/chunk32；181帧23 anchors一次writer；FP32图像detector B32 | 完整收到序列256位logit均值>0，匹配≥161 |
| VideoSeal v1.0 256 | 官方y_256b_img权重，step4/chunk32；128＋53帧块对应32＋14 anchors | 同上，独立权重与单独行 |
| PixelSeal-FrameIndex256 ADAPTED | 同Pixel图像writer FP32 B8，每帧248随机owner＋8 MSB原index | owner匹配≥165；再满足index≤180才输出时间 |

每source的前三完整query用实际随机字节一次抽取并保存，FrameIndex用Pixel前248位。每WM独立wrong-query，碰撞不重抽。raw detector只收像素；保存全部实际批次数组后才验证query和读取编辑真值。没有额外ROI、投票、ECC、PRF、平滑或按分数重试。

Pixel OFF实际B32原始输出同时供Pixel与FrameIndex逻辑行；不得重复GPU。WAM自然空mask是已观察擦除/无候选。WAM/FrameIndex部分输出中真实阳性见证为TRUE、complete=false；没有见证为UNKNOWN。Seal缺任何计划收到帧时不算prefix均值。正常攻击删除后的短序列是完整输入。family任一已知TRUE即TRUE；all40鲁棒率要求40完整接受，任一完整拒绝即FALSE。二者不可混用。

## 攻击与统计

16时间条件各G1编辑RGB/G2再H264：identity、4×129帧trim、随机删18/54、burst删18/54、重复18、回放18、4块乱序、DEV外来源替换18、线性插18、1.25/0.8最近邻重采样。另8条件：H26428、H26528、中心crop/restore、halfsize/restore、Gaussian sigma1、亮度0.8、噪声2/255、JPEG75后H26418。时间编辑先作用于实际首代H26418解码RGB；medium/GOP250/bframes3/threads1。共40views、6664收到帧、6624邻边，B8=852、B32=232真实尾批。

每source固定PCG64 SeedSequence([2026110601,0,global_ordinal,operation_id])；全部arm共享编辑计划。T12供体只来自DEV，不选自身；T13两parent均为target且round-even，不假定唯一原index。G1/G2也复用同一计划。

名义0.1%是逐方法、单均匀随机query、固定40视图族的union bound：WAM约0.082079%、Pixel/VideoSeal约0.088726%、FrameIndex约0.070336%。不是自然总体的经验认证，不是四方法合并上界。CAL不调阈值，CAL缺失不禁用固定解析规则。TEST500来源族与各视图报告TRUE/FALSE/UNKNOWN、execution_incomplete、固定分母率区间、CP双侧95%及单侧95%上界，action-cluster bootstrap2000。0/500单侧上界约0.597355%。

## 四表与保存

`tables/A_payload.{json,csv}`逐来源/视图保存bit错误、有效覆盖、擦除、缺失、完整消息和完整owner。T12 foreign逐帧BER对供体真实消息，native视频聚合只称target query agreement。T13 native恒定消息与FI248 owner有定义，FI8 index位UNDEFINED。

`tables/B_ownership.{json,csv}`分DEV/CAL/TEST、OFF/WM、query/wrong-query/cross-source。跨源只复用固定DEV的已存WM raw；比较次数与真实不同donor数另列，绝不增加独立来源。`FAMILY40`是任一view，`ALL40_ROBUST`是全部view；G1/G2子族另列。主固定分母保留partial已知TRUE；`known_decision_subset`只表示判决已知，`complete_subset`要求该channel全部执行完整，二者分母、阳性和区间分别列出。

`tables/C_time.{json,csv}`保留三原生方法N/A；FI报告target exact/±1/±4、合法错误index、foreign错误target归属、synthetic虚假精确时间、擦除与未执行。query通过且index精确，与全部ownerbits正确且index精确分列。完整节点/邻边数组附于post容器；不从真值生成foreign/synthetic预测，也不把擦除当正确识别编辑。只恢复对应，不恢复删除图像。

`tables/D_quality.{json,csv}`和`D_resources.json`：仅precodec、G1、G2-identity，181帧RGB MSE/PSNR、Gaussian11/sigma1.5 SSIM、FP32 LPIPS Alex B16尾批5；180邻边水印残差差分RMS不是运动补偿tLPIPS。逐帧质量保留在`units/quality/`。实际forward hooks、峰值、调用秒数、下载/codec/存储日志分别保存；DEV执行后的剩余成本估计不额外试跑。

正式目录固定为`/content/drive/MyDrive/Video-WM_Result/baselines/public-native-and-frameindex-fpr001-v4/attempt01`，保留canonical、G1媒体、protocol/queries/editor、native批次、post/tables、resources与failures。正式目录及失败包不清理。临时源码、权重、writer块、JPEG和衍生媒体位于`/content/drive/MyDrive/Video-WM_Cache/baseline-v4`，仅对应读者和质量完成后清除。失败接收输入转存正式failures，并可从该实际输入继续。

## 调度与恢复

先准备DEV OFF和各方法WM供体库，再读DEV；之后CAL/TEST按16来源pack、三个backend顺序复用。DEV库同backend跨四个16来源子包共享一次loader，基础最多195次，低于协议204上限。无额外科学样本。供体解码缓存约28.5GB，以避免为每次T12增加解码；这比原20GB临时空间情景更大。总新增200–260GB仍仅规划，媒体实际字节、已有Drive占用未知，不保证400GB必够。

固定基础预算detector2273296 forwards、36305472帧；编码/探测各83000，decode上限144032，JPEG编码/解码各600920，normalize1064；LPIPS81216 pair-batches。每实际物理单位至多两次额外基础设施重算，各类累计最多ceil(基础上限×2%)。最多64个额外runtime，每个额外watermark loader≤3、LPIPS loader≤1。实际失败、已开始但未保存的未知成本都计账；方法失败/非有限/非法shape不自动重跑或改参数。预算用尽相关单位UNKNOWN，继续独立工作。

批次返回立即保存实际数组。native Pixel writer整181帧，VideoSeal128＋53，WAM/FI B8、reader B8/B32；不合并尾批、不pad。codec按完整clip恢复。相同输入缓存尚在则只补未提交批次；丢失可能改变RGB时保留旧partial，新epoch重建全view，全部批次计恢复预算，不混epochs或挑分数。首次完整有效返回采用。已存raw而post中断时只补CPU后评。

同一结果包一次只能有一个活动runtime。本机flock、owner记录和≤20秒心跳不构成跨VM原子锁；旧不同主机仍显示ACTIVE时，先实际停止后通过notebook的`PREVIOUS_RUNTIME_STOPPED`确认，没有TTL自动抢写。初始化query等包内输入也在取得运行所有权后写入。正常关闭后直接Run all。完成判断读取实际批次数组覆盖，不只信进度标记；当次runner缓存已核读状态，避免同一来源反复扫描。全部原始读数与画质已完成时，缺后评表只补CPU，不因临时缓存消失而重建媒体。空间不足暂停新单元；任何清理函数只允许独立cache的子路径。

## 工程验证与证据上限

`python -m pytest -q -o addopts= tests/test_native_baselines_public_fpr001_v4.py` 使用人工数组和fake backend/codec。内部fixture可缩小像素画布，保留181帧、40视图及真实批次数；该入口不暴露于正式CLI/config。检查共享OFF、各物理类算术、完整四表、二次续接不加推理、部分见证、真值边界和恢复失败。模型接口检查只用人工stub，不载入权重。

未执行真实媒体、codec、模型、GPU或Colab。发布可运行入口只表示工程交付；实际数据可用性、依赖组合、显存/速度、画质/鲁棒性及论文结论仍须由用户真实Run all和保存证据确认。
