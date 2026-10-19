# B 四相位盲时间对应与分段归位（卡58本地交付）

用户已采用卡57候选，卡58独立实现与必要合成 CPU/假 backend 验证完成；卡59将本运行包发布到现有 dev/local-joint-state-payload-v1。没有新真实运行结果。旧 clean35、写端与 A 线保持原样。

入口为 `notebooks/blind_temporal_segmented_v1.ipynb`，builder 为 `scripts/build_blind_temporal_segmented_v1_notebook.py`，普通源码包为同目录 `blind_temporal_segmented_v1_portable_source.zip`。
Notebook 与普通源码 ZIP 同步发布到 dev/local-joint-state-payload-v1。通过下方 Colab 入口直接 Run all，默认普通 URL 自动下载源码 ZIP，无需手工上传。也可把 SOURCE_LOCATION 设为实际可用源码目录、普通 ZIP 或 URL；不核对摘要、Git 状态或精确版本。

[打开 Colab](https://colab.research.google.com/github/RICHAAARC/SC-SSTW/blob/dev/local-joint-state-payload-v1/notebooks/blind_temporal_segmented_v1.ipynb)。真实 GPU 任务由人类运行；发布与静态检查不是实际执行成功证明。

独立 CLI：

```bash
python -m experiments.wan_state_clock.blind_temporal_segmented_v1_run
```

可用 `--config <path>` 改实际输入/输出/模型位置。默认配置是 `experiments/wan_state_clock/configs/blind_temporal_segmented_v1.json`。它加载同次 clean pair 保存的 P/S `media/*/mp4_rgb8.pt`，不会重新解码 MP4，也不重新生成源视频。输出为 Drive 的 `Video-WM/B-Blind-Temporal-Segmented-V1/fixed-four-phase-v1`。

固定编辑为 source[37:126]（89帧）和 source[:45]+source[88:]（138帧）。接收核心不接收这些名称、来源臂、真实坐标或消息；它只读取按收到帧建立的四相位真实 E bank、key 与公开源181协议。相位 p 的输入前补 p 份首帧后截到 1+4k。补帧只是上下文；完整真实 4/8 帧窗口才有载荷/pilot。原22方向、4ROI、两 latent 正负 pair 保持，奇偶 pair 都读。

路径族包含未知头尾裁剪和至多一个内部连续缺口，非循环、严格递增。每条候选使用其解释为完整源8帧块的全部 pilot 项，按 source-k 顺序 `math.fsum` 后除以公共 `floor(N/8)`，以最终 float64 分数取完整精确最大集合。原始和、分数、支持数、所有候选参数和最大索引存 NPZ。没有阈值、epsilon、top-K、half fallback 或按路径支持数归一化。任一必要 phase/pilot 失败使 EST INCOMPLETE。两种已知几何不可辨例按规范保持同分；不保证全帧唯一或精确接缝。

载荷计划显式列出 `(received_start, bank, latent_index, source_t, segment, context_flags, mapping_status)`。并列只接纳所有最大路径一致的完整4帧单位，每个源 t 最多一份票，t0/t45不计，t11/12与t33/34跨段拆开。上下文标记仅来自相位构造或最大集合共识，分歧记 unknown；真实接缝只在后评。真实修改后的因果 VAE 历史不被恢复，没有暖机剔除。

每 bank 一次 FFT real/ortho，两个 key 共享，strict >0 和 exact-zero mask 保留。归段票序为 channel → ascending received unit → keyed coordinate slot::8，平票采用 Counter 首遇。缺失 bank 的原始失败保留；baseline/oracle 可读剩余 bank，标 PARTIAL_FAILED，不能据此补 EST。净容量仍4×8位。

完整执行顺序：E16/32组 key-pilot → 保存全部8盲路径与计划 → FFT16/32组 key-votes → 保存 EST/UNALIGNED 16载荷表 → blind_completion → 打开消息/真实路径 → 同库 ORACLE8与评价24。UNALIGNED 固定 f(i)=i/p0。ORACLE只是同观测库已知对应后评，不保证消息准确率上界。

未来单次用户预算为 VAEload1、E16、FFT16；D/codec/probe/新媒体解码/生成/DiT/梯度/逐帧模型/路径后E均0。固定8 receiver行、852 pair、74976 ROI/18744 pooled数、848640 signed votes及同量zero mask。24表保留768计划位（576结构存在+192缺失）；每模式256位、176state、352cell、908收到帧。缺失段输出记 false coverage，不计正确恢复或规范 BER，即使值与消息相同。共同BER与有效覆盖限定结构存在位，错源单位/错段与bit碰巧正确分开记录。

attempt在backend构造前持久化；每个调用先写 STARTED。失败/中断消耗这次尝试，不重试。后续 Run all 先检查已有产物，跳过源码下载、依赖安装与模型；只有保存记录回读。缺失/损坏 checkpoint 会显示固定24表不可用占位，不能恢复计算。手动删除产物重跑不属于此包授权语义。

保存目录包含两个输入读取记录、4个匿名received PT、phase输入索引记录、normalized bank PT、pair/votes NPZ、全部 candidates NPZ（完整/partial/absent计数）、盲plan/path/payload、blind completion后才保存的独立oracle maps、后置ORACLE、ledger/checkpoint/result及逐帧/44cell/22state/32bit评价。计划中的完整pilot几何/边界单位推断和实际pair可用性分开；实际错票只按reducer明确used_units统计，失败表不凭计划虚构票。只读调用计数优先归约最新ledger，checkpoint不能掩盖较新的STARTED。未来GPU执行记录峰值和耗时；当前没有这些实测。失败与并列不是科学PASS。合成测试只验证数学/调用/故障接线；实际变长 VAE可运行性、显存、耗时、因果上下文影响与真实归位收益仍待人类未来运行。
