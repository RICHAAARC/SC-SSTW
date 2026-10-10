# B R-MEAN-2H：已采纳的两半步 MEAN 刷新

用户于 2026-10-10 采纳推荐方案 R-MEAN-2H。本文件说明该方案的固定实现；
旧 COMMON 刷新备选仍未采纳。它不包含 M1/M2、native 接线、方向扫描或自动 GPU 执行。
本次交付仅经过静态、CPU 数值和替身执行验证，真实 Colab 尚未执行。

## 输入和固定计算

复用同一 JOINT 原终端（run `20261009T132316545817Z`）和已跑通 M0
（run `20261010T072250346991Z`）保存的 MEAN/COMMON delta。SOURCE_REF 和输入路径可编辑；
不设置 SHA、manifest、硬件型号或版本身份门禁。实际文件可读性、张量形状/有限性、
已采纳的 key/message/rho/cap，以及保存方向的实际 mask/unit-L2 是计算所需条件。
unit 检查只容纳 FP32 舍入（8×dtype epsilon），不重归一化或悄悄换方向。

固定 carrier 为 `local-joint-state-payload-v1-first-mechanism`、message `8001a55a`、
rho=0.5。原 latent support、读出、五目标和分母保持不变。decoder 参数冻结 FP32；
内部 [-1,1] clamp 与外层 RGB clamp 均保留，读出池化 FP64。不学习或替换 reader。

设原终端为 z0，保存的单位方向为 d0=MEAN。固定：

1. delta1=0.5*d0，z1=z0+delta1。
2. 从实际 D(z1) 重读 G 和四段各自最弱 signed bit margin，记为 f0..f4。
3. 在 z1 对 F=(f0+...+f4)/5 做一次 decoder VJP，g=M*grad F。
   不逐目标归一化，也不计算或声称已获得五个独立梯度。
4. 若 g 非零且有限，d1=g/||g||2，候选 delta=delta1+0.5*d1。
   仅当 ||delta||2>1 时整体投影到以原 z0 为中心的单位球。
   TWO_MEAN=z0+delta。不会把小于1的半径补满，也不会轮轮追加 L2=1。

平均目标局部上升不保证五目标各自上升或32位均改善。保存的预测量仅为
`g dot (delta2-delta1)`；它是中点处平均目标对实际第二位移的一阶预测，
不是各分量预测，不是相对于原 BASE 的整步预测。
输出累计位移实际 FP64 范数；FP32输出的最后舍入可能在机器精度范围偏离理想球面。

## 五视图与资源

固定五视图 BASE、ONE_MEAN、ONE_COMMON、MID_MEAN、TWO_MEAN。
三个独立对照先解码；中点退化不会抹去它们。**主比较是 TWO_MEAN 对 ONE_MEAN**。
ONE_COMMON 是保存的一步比较，不会默认判断 COMMON 或 TWO 胜出。

完整非退化路径为一次 VAE load、6D、1 decoder VJP：
五次普通解码（230个 chunk）、一次梯度前向（46个 chunk），VJP的名义重放46个 chunk。
E/DiT/native/codec 均为0。实际 attempted/completed、各调用起止时间、原前向和
重放、缓存写/读字节、峰值内存、进程 RSS、清理状态分别留档。

直接复用 M0 已跑通的 checkpoint/disk boundary 实现；因果 cache 保持梯度连接，
不是 detach 或切断时间导数。边界存本地磁盘，梯度结束清理。一次 VJP 仍有显著成本。
原 M0 在 L4 的11D+5VJP总计约56.64分钟；各 VJP 前向+反向约7.07–24.52分钟，
单次缓存约64.53 GiB，不应据次数线性承诺新流程时间。这里没有新硬件门禁、
资源预测承诺或盲重试。新流程真实时间、显存和磁盘峰值尚未实测。

原窗口分母每视图88，每窗16chips，共440窗口/7040chips/275指标。
每视图包含22 offset correlations、32 signed bit margins、state gap。
comparison.json 提供所有32位、剩余非正位过零缺口、gained/lost bits、活跃最弱位和
竞争 offset 切换；主比较、相对 BASE 和实际第二半步分别保留。

## 退化与中断

中点普通读出或梯度前向读出发现活动 min/max 并列，TWO_MEAN 为 undefined，不选任意
子梯度。平均梯度严格为0时不执行第二步，也不补一个零位移“成功”视图。
非有限值、OOM、读写失败均留工程异常和实际完成数，没有回退、额外 VJP 或 retry。
单个对照缺文件/解码失败不阻止另一个独立对照。缺 MEAN 输入时没有中点路径；
缺 COMMON 输入仍可完成 MEAN 路径。质量写出失败不会抹去已存 raw/metrics 或阻断固定方法调用。

固定槽位始终保留。科学负结果或合理 undefined 不用作工程门禁；COMPLETE 不是科学 PASS。
notebook 子进程清理后，finalize_interrupted 仅从已持久 raw/metrics 补齐比较和显式缺失，
不读像素、不重算指标、不调用模型，也不根据观察文件推断某个 model call 已完成。
持久存储完全不可写时不能保证异常记录也能落盘；保留此前已存内容。

## 质量与交付

质量始终相对 BASE：全局/逐帧 RMSE、PSNR、max_abs、ROI内外能量和时间残差变化。
PNG 固定零基帧 [1,44,88,112,116,120,132,176]，覆盖此前 frame116 峰值邻域；
残差显示比例全视图一致。quality.json 的 temporal_framewise 固定180行，
第 i 行是相邻残差 r[t]-r[t-1] 的 RMSE/max_abs；这不是原视频运动量。
不设新质量 PASS 阈值，局部峰值和 ROI 外变化须与全局量一起解释。
固定帧图片不代表完整视频观感，更不代表 codec 结果。

Run all 文件：

- `notebooks/local_joint_mean_refresh_v1_colab.ipynb`
- 普通配套 ZIP：`notebooks/local_joint_mean_refresh_v1_portable_source.zip`
- builder：`scripts/build_local_joint_mean_refresh_v1_notebook.py`
- CLI：`experiments/wan_state_clock/local_joint_mean_refresh_v1_run.py`
- 配置：`experiments/wan_state_clock/configs/local_joint_mean_refresh_v1.json`

首个代码单元仅挂载 Drive。后续按顺序建立新 UTC 输出、下载普通 ZIP、检查实际依赖、
执行固定实验和汇总。缺依赖沿用 M0 正常安装路径；没有新的模式开关或强校验。

默认三个输入：

- `/content/drive/MyDrive/Video-WM/Local-Joint-State-Payload-V1/20261009T132316545817Z/run/joint/terminal_latent.pt`
- `/content/drive/MyDrive/Video-WM/Local-Joint-State-Payload-V1-Readout-M0/20261010T072250346991Z/run/MEAN/delta.pt`
- `/content/drive/MyDrive/Video-WM/Local-Joint-State-Payload-V1-Readout-M0/20261010T072250346991Z/run/COMMON/delta.pt`

新输出：`/content/drive/MyDrive/Video-WM/Local-Joint-State-Payload-V1-Mean-Refresh/<UTC>/`。
回传完整文件夹，包括缺失/失败视图和资源日志。此次源码和普通 ZIP 的验证不能代替
真实 Colab；单终端局部写入也不能解释或承诺原 step25..49 的 CFG/native 轨迹。
