# ADJ 单帧origin邻域纠错与四新源确认

唯一候选为RANK3R_ADJACENT_ORIGIN_V1（ADJ）。CPU固定开发回放全部判据满足，建议进入一次四新来源确认；尚未自动运行GPU，不据此宣称正式采用。

[直接在Colab打开](https://colab.research.google.com/github/RICHAAARC/SC-SSTW/blob/dev/local-joint-state-payload-v1/notebooks/b_adjacent_origin_confirmation_v1.ipynb)。Run all默认直接从B分支加载可用源码，无需手动上传ZIP。

## 方法

对完整H137先调用原RANK3R。所有非REJECT_GEOMETRY分支原样保留。仅几何拒绝时，每窗取唯一第二名；第二名并列则无候选。先过滤origin与第一名相差恰好1，再检查单窗替换后的三元组是否属于原4247合法路径类。对至多3个合法候选，以原始同窗S1-S2的唯一最小值选择，精确并列擦除。无第三名回退、双/三窗修改、别名求和、新q/tau门、分数重写或参数扫描。

选择只看匿名观测和公钥标识，不用载荷、消息、源标签、真映射或actual support。使用选中候选对应的实际phase/view缓存；全计划保存回读后读位，全决定保存回读后评价。旧RANK3R与旧次峰模块未改。

该开发候选在见过m04和一个null反例后形成。相邻origin不是VAE平滑保证，也不保证同local_row共享3/4帧；不提供独立FPR或IID概率界。

## 一次CPU开发结果

固定312检测=52正+260null；276读取/8832槽=46正/1472+230null/7360。正检测52/52，实际支持1469/1472；stage1 e02三例slot30原冲突保留。M3 20/20、640/640；m04两任务各恢复32槽。null检测/输出、字面错误、错误归属、无实际支持输出均0。原50DETECT选择及44个读取任务计划/决定/支持不变；1405原支持保留，新增64、丢失0。M3对CDC_M增125、丢0，净增125。共同输出BER两侧均0，未观察到共同BER改善。

已执行一次，592账本行RETURNED；19判据全满足，物理与质量新增调用0。36个stage1校准任务仅检测；旧阶段逐项分列，M2 CDC_M为N/A。旧M3与旧次峰负结论未覆盖。独立结果核验单独记录。

## 固定GPU四源名单

|ID|seed|a|内容|
|---|---:|---:|---|
|b-adj-v01|2026102701|2|湖面、帆船、植被|
|b-adj-v02|2026102702|11|日间市场、手部、果蔬|
|b-adj-v03|2026102703|17|雪景、电车、行进|
|b-adj-v04|2026102704|6|陶艺工坊、缓慢水平平移|

四个完整英文prompt预先写入配置，未按预跑成功率挑选或替换。原writer、γ=1、旧背景与CDC_M旧τ不变；5片段为CROP、左右delete8、左右repeat8，接缝49/102。ADJ、原RANK3R、CDC_M共享观测。

每读法120任务/3840槽=20正/640+100null/3200；PM wrong-key20任务/640槽是null子集。三读法360计划、360读取、11520槽。

物理预算：8轨迹、400native steps、800DiT、100local gradient、4state_construct；D18112；codec encode/probe/decode各72；E41720、Q1440；E41 DCT720、keybits1440；native Q8、DCT8、keybits16；codebooks8。native P0/PM位对2816、媒体位对9600；360个H族/49320分数；108质量记录、17148帧、17040相邻对。

## 预定判读

系统层沿原M3三层数值阈值：总体每源检测≥4/5、支持≥80/160且总支持≥480/640；重复总检测≥6/8、每源≥1/2，支持≥192/256、每源≥32/64；裁剪删除总检测≥9/12、每源≥2/3，支持≥288/384、每源≥48/96。null检测/输出、字面错误、错误归属、无支持均0；对原RANK3R及CDC_M逐任务逐槽零丢失。全部方法判据满足后，相对CDC_M净增≥32才达到系统贡献判据。

ADJ层单列相对原RANK3R新增、丢失、纠错分支、错误。若原法本批已全成功且ADJ无增益，明确“本批未提供新增纠错证据”；系统PASS不能代替纠错贡献。新增错误与恢复分列，不能抵销；不自动追加来源。

## 运行、保存与限制

复用已跑通M3安装、ftfy/torchao定点修复、ffmpeg、FP32 VAE、生成/释放与两代H264媒体链。首cell仅Drive挂载；HF/TORCH/模型和中间tensor在/content。Drive输出独立为Video-WM/B-Adjacent-Origin-Confirmation-V1/fixed-four-five-v1，必要JSON和72最终MP4；不再额外归档未来M4用大体积OFF RGB24。

一次尝试、失败、中断和未进入项保留；后续Run all只读已消费尝试。保存检查分开报告云端和挂载状态，不改变方法状态；下载发起不保证用户浏览器保存完毕。不建立B64、SHA、manifest、Gitclean、精确版本或GPU型号门禁。仅固定来源的描述性确认，不宣称IID、总体FPR、所有攻击或感知质量PASS。

本轮本地检查和针对性独审与实际CPU回放分列；GPU由用户运行。源码仅在B分支交付，不合并main。
