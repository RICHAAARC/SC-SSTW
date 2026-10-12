# All-Bit Continue v1：已采纳的 0.10 × 2 独立入口

2026-10-12，用户已集中采纳 c152150 提案 v2；本页记录 A1 最小实现与本地交付，发布由主审负责。没有真实 GPU/Colab/Drive 执行，也没有推送或 main 集成。

## 直接入口

- Notebook：`notebooks/local_joint_all_bit_continue_v1_colab.ipynb`
- Builder：`scripts/build_local_joint_all_bit_continue_v1_notebook.py`
- 普通源码包：`notebooks/local_joint_all_bit_continue_v1_portable_source.zip`
- CLI：`python -m experiments.wan_state_clock.local_joint_all_bit_continue_v1_run --config <config.json> --output <task/run>`
- 固定配置：`experiments/wan_state_clock/configs/local_joint_all_bit_continue_v1.json`

Notebook 第一格仍是两行 Drive mount，直接 Run all，无方法模式菜单。默认输入为旧完整任务：

```text
/content/drive/MyDrive/Video-WM/Local-Joint-State-Payload-V1-All-Bit/resumable-task-v1/run/inputs/terminal_latent.pt
/content/drive/MyDrive/Video-WM/Local-Joint-State-Payload-V1-All-Bit/resumable-task-v1/run/step_2_unit.pt['terminal']
```

只读取实际 BASE 和 START；旧 step2 的 input_terminal、gradient、计划 delta 都不是新求导起点。默认独立新输出：

`MyDrive/Video-WM/Local-Joint-State-Payload-V1-All-Bit-Continue/resumable-task-v1-2x010/`

## 方法与真实张量语义

四视图固定 BASE / START_SAVED / CONT_MID / CONT_FINAL。保留原 32 位 Huber loss：mu=.02、gamma=.01、权重.8/.2，以及 key/message、public mask、readout 和实际 frozen FP32 Wan/clamp/causal-cache 路径。新 scalar VJP 分别在实际 START 和实际 MID，名义步长各 .10。不会重新做旧 .5×2，不扫描、回滚择优、加第三步、扩 cap 或追加到成功。

新 core `main/tube_state/local_joint_all_bit_continue_v1.py` 在 CPU FP64 中每步计算 actual current−BASE，再沿 fresh masked gradient 更新并投影到原 BASE 半径1。落盘 FP32 的最近舍入只有实际越界时才逐坐标向 BASE 取一个相邻值；仍越界则明确数值失败。报告 FP64 scale ULP 修正、内侧舍入分量、计划/实际范数、舍入误差及投影前后和实际增量的 scalar 预测。

新步骤单元保存实际 input_terminal、实际 output terminal、FP64 actual delta=terminal−BASE 及 report；恢复以实际 tensor 传递下一输入，不用 BASE+旧计划 delta 重建。梯度检查点保留实际 input+gradient 原子对。旧 .5 入口保持其原 planned-FP32-delta 语义；新旧入口共用执行器但各自固定方法，不增加用户可切换模式。

原 scalar loss 数学直接复用，新 loss/readout/step/report/config 使用新 `step_size=.1`，没有遗留 `half_step=.5` 预算。完整不中断账为1 load +6D +2VJP，普通184/gradient-forward92/nominal-replay92 chunks；352 windows/5632 chips/220 metrics，缺失或失败槽保留。

## 恢复与质量

复用已发布 Journal 的原子单元、60秒状态/进度、上一可读状态、attempt/call账和 lazy model load。Journal 只将固定 view 名称作为窄参数；保存/计时机制没有另写一套。quality exporter 同样只接收本案四视图；实际峰值/邻域按新保存质量定位。

| 已完成边界，之前控制均完成 | 剩余 load/D/VJP |
|---|---|
| g1 已保存；MID RGB 尚未保存 | 1/3/1 |
| g1 和 MID RGB 已保存；MID raw/score 尚缺 | 1/2/1 |
| g2 已保存；FINAL RGB 尚未保存 | 1/1/0 |
| g2 和 FINAL RGB 已保存；FINAL raw/score 尚缺 | 0/0/0 |
| 全完成或零梯度/活动 tie 合法终局 | 0/0/0，报告/必要 CPU 收尾 |

中途 autograd、partial spool 或未提交梯度不会被当作恢复点；未完成整个单元可在下次用户 Run all 重算，保留累计物理调用和失败。旧成功任务不被作为新任务完成状态。CPU 尾段不调用 CUDA 或加载 VAE。BASE reader 失败但 BASE RGB 可读时，仍计算其他视图相对质量。完整 raw/score 存在但 RGB 损坏时保留读数并报告缺失质量，不静默重解码。

`comparison.json` 保留32位、22 correlations/G、逐步 loss 与 scalar 预测；`tracked_bits.json` 固定保留原六负位、原26正位、f3b5和各视图当前最弱正位。每视图181帧/180时序对、ROI内外、平均和局部峰值、邻域/crop全部保留。没有新增 PASS 阈值、codec、盲读、新源或 M1/M2。

## 定向检查与证据上限

检查覆盖新投影/FP32越界例、actual START 与两次直接 decoder VJP 参考、实际 terminal 的连续传递/FP64 delta 缓存、固定分母与调用、g1/g2/RGB落盘恢复、CPU-only完成再入、合法终局、BASE读数失败仍有质量、输入缺失、新旧 notebook/ZIP 同步与无.git独立CLI。

具体命令、结果和恢复账见主目录 `diagnostics/b-line-all-bit-continue-implementation-20261012/`。没有重跑未改 Journal 整套保存测试。旧普通入口的一项完整数学路径和受共享接线影响的 notebook/CPU尾段单独检查。真实 continuation 效果、Colab资源/到期恢复仍由用户运行后判读；不从 CPU/fake 检查外推32/32、质量或科学 PASS。
