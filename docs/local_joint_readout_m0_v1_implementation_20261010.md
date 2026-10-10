# B 线 M0 实施与验证交付

日期：2026-10-10。用户已明确采纳“B线路同意M0实施”。本地交付基于提案修订 `7c4b69f`，仅实现已采纳选项 A。以下为**本地交付时快照：本地实现和定向 CPU/替身验证完成**；当时尚未执行真实模型、完成真实结果审计、发布或 main 集成。发布由主会话依据已有授权在冻结后处理；本段不代表后续发布状态。

## 固定执行

入口为 `notebooks/local_joint_readout_m0_v1_colab.ipynb`，普通配套包为同目录 `local_joint_readout_m0_v1_portable_source.zip`。首代码单元仅两行 Drive mount，GPU metadata 已设，Run all 使用普通可编辑 `SOURCE_REF` 下载 ZIP。安装、VAE loader、子进程日志和进程组清理由已有 bridge notebook 复用。没有源码身份、摘要、manifest、精确版本或 GPU 型号准入；安全解压和真实导入/读写/模型错误保留。

默认输入为原 JOINT `20261009T132316545817Z/run/joint/terminal_latent.pt` 与桥接 `20261010T031603868487Z/run/tensors/capped.pt`。后者直接作为已保存 delta 读取，不再次 mask/cap。新输出进入 `MyDrive/Video-WM/Local-Joint-State-Payload-V1-Readout-M0/<UTC时间>/`；原输入目录不写入。notebook setup 中输入路径可编辑，保存后的 config 直接传入独立 CLI。

保持原 key、message `8001a55a`、DCT 配对能量统计、22×4 窗口、四个局部 latent ROI 和 cap1。五目标是 G 和四片段各自最弱 margin；梯度先 mask，绝不各自单位化。MEAN 为算术平均，COMMON 解无 ridge 的 5×5 Gram 单纯形问题，31 个面只是同一代数问题的求解。只对两条最终方向各做一次 L2=1 归一化，FD 为固定 ±1/64 COMMON，无扫描和重试。

六个固定行是 BASE、BRIDGE、MEAN、COMMON、FD_MINUS、FD_PLUS。活跃 min/max 并列使四个新方向行未定义；零 MEAN 只移除 MEAN；精确零或数值不可辨 COMMON 只移除 COMMON 与两条 FD。每行保留 `direction_status`、缺失原因与固定槽位。精确零只能排除五项全部严格一阶上升；数值近零/未解不提供排除证书。

完整非退化计划为 1 load、11 decode、5 decoder VJP，0 encode/DiT/native/codec。普通 chunk276、梯度原始 forward230、名义 replay230 分开报告实际 attempted/completed。五次 VJP 顺序执行，每次释放图和磁盘暂存。独立 BASE/BRIDGE 仍按其输入执行；失败或退化不补调用。

## 梯度与报告边界

- `main/tube_state/local_joint_readout_m0_v1.py` 只实现张量读出和五目标代数。q 是池化后 `(A-B)/(A+B)`，没有 epsilon、逐 tile q 均值、浮点序列化或 detach 断图。
- `runtime/wan/local_joint_readout_checkpoint_v1.py` 复用历史 B2 的 Tensor/None/Rep cache、cursor、first_chunk、输出 cache 挂图及磁盘边界。移除旧磁盘预检/配额；保留真实 I/O 错误与使用统计。别名恢复在单次 recompute 内共享 storage，在正常返回、提前停止、callback 异常后释放，不持有到整次 VJP 结束。
- `runtime/wan/local_joint_readout_m0_v1.py` 通过真实冻结 FP32 decoder 反传 normalized 缩放、Wan 内部 clamp、RGB 转换/clamp 和张量读出。普通观察仍使用原独立 raw reader。decode 的返回账记录在模型方法返回处，适配器/质量保存失败不会倒写为模型未返回。
- 每行 raw 在独立 truth reducer 之前持久化，保存 22 correlations、G、32 margins、四个 fragment minima、活跃 wrong offset/weakest bits。保存各次梯度 forward 相对 BASE 的数值/活跃项差异，报告有限变化、预测斜率、失去/新增正位及 FD 活跃项切换。
- 保存 delta、实际 terminal、float RGB、五个 masked gradient、Gram/weights/KKT 数值残差。质量包含 RGB RMSE/PSNR/max abs、逐帧误差、时间残差、ROI 内外能量与峰值。帧 `[1,44,88,132,176]` 使用共同 RGB `[0,1]` 和 residual `[-0.1,0.1]` 显示尺度；PNG 仅供查看，不是 codec 观测。
- FD 同时报告 FP64 减法可分辨性和已有 FP32 decoder 重复 forward 的观测波动。重复波动为零不代表 FP32 误差上界，不据此调步长或追加调用。
- 528 windows、8448 chips、330 metrics 始终保留分母；缺失 raw 不伪造 q。中断接管仅在子进程结束后按保存的 raw/metrics 重建计数，质量/图像保存状态单独报告，不恢复模型执行。

## 定向验证

在现有 WSL CPU 环境执行：

```text
python -m scripts.build_local_joint_readout_m0_v1_notebook
python -m pytest -q tests/test_local_joint_readout_m0_v1.py tests/test_local_joint_readout_m0_v1_notebook.py
15 passed in 20.77s
```

覆盖：原独立 reader 与张量 reader 数值等价、解析 q 导数和方向有限差分；单纯形 KKT、精确/数值零与弱协调反例；含内部 clamp、跨 chunk Tensor/None/Rep 与重叠非连续 alias 的小型因果 decoder 前向/VJP 精确等价；recompute 日志 I/O 失败后的 storage 清理；六行全部完成计数；梯度失败、适配器失败、并列退化、保存 metrics 后中断的证据保留；notebook 全新命名空间按顺序执行替身、环境失败和进程被杀接管；普通 ZIP 的独立无 Git CLI。

完整小模型路径实测 11 decode、5 VJP、普通276/梯度230/replay230 chunk，以及 8448 个有效 chip、330 个指标。此处是接口和生命周期工程证据，不是 Wan 实跑效果。主会话另做的完整 512 样本/每 chip CPU reader 交叉核对和历史 checkpoint alias 增量审查，未计为新增实验样本。

未加载真实 Wan 权重，未运行 GPU、VAE/Colab、生成、encode、codec 或新水印实验。真实 M0 的方向有效性、质量、显存/RSS/磁盘峰值和耗时仍待用户自运行。历史 B2/L4 资源记录不作为新 M0 的保证。M1/M2、轨迹控制、盲同步、FPR、学习读出和科学 PASS 均不由这次工程交付推出。
