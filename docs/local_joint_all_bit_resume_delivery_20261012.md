# 现有 All-Bit v1：Colab 到期后的定时保存与恢复

2026-10-12，A1 本地实现，交主审复核和发布。用户纠正后，`5957f17` 的 0.10 两小步提案已搁置。

现有 `notebooks/local_joint_all_bit_v1_colab.ipynb`、builder 和普通源码 ZIP 同步更新。原 BASE、0.5 × 2、既定 Huber loss、mask、半径、四视图与固定 352 windows / 5632 chips / 220 metrics 均不变。core/config 未改；没有实施新步长、向内舍入或新方法。

## 用户操作

打开交付 notebook，挂载同一 Drive 后 Run all。默认逻辑任务目录固定为：

`MyDrive/Video-WM/Local-Joint-State-Payload-V1-All-Bit/resumable-task-v1/`

Colab 到期后重新打开 notebook，再次 Run all，自动使用该目录已完成的单元继续。旧 UTC 完成目录不自动选择、不迁移、不重跑。仅在有意建立另一独立任务时另选 OUTPUT 目录。未完成 decode/VJP 需要 GPU；剩余评分、质量和报告可以在 CPU runtime 完成。全部完成或零梯度/活动状态并列等合法终局只返回报告，不开启新两步。

## 保存与恢复边界

- 输入 terminal 和旧 TWO_MEAN delta 在使用时复制到持久目录；新 VM 不依赖上一 VM 的临时输入或 activation spool。
- 每个完整梯度检查点原子保存实际输入 terminal、masked gradient、梯度读数。仅在实际输入相同的既定步骤复用；输入不匹配时保留原因并重新计算正确单元。
- 每步 delta、terminal 和步报告作为完整单元保存；RGB、raw、metrics、quality 和图像索引依次保存。raw 仍先于 message truth 评分。
- 单元完成即写入；独立线程每 60 秒保存最近已提交完整状态及进度。`result.json.previous` 与 `checkpoints/latest.json` / `previous.json` 提供可读回退。半写 `.tmp` 不作为完成单元。
- 定时器不会保存未完成 CUDA kernel/autograd graph。未提交完整结果的 decode/VJP 从最近保存输入重算；没有声称中途继续 backward。
- 已完成 raw/score 的 RGB 若损坏，不为补质量重解码；保留读数并报告具体缺失，防止新 RGB 与旧读数混用。BASE RGB 在 reader 失败后仍可作质量参考。
- 同一 kernel 再次 Run all 会清理旧工作区包缓存，载入这次可用源码。CPU 剩余阶段的环境探测不要求 CUDA 或加载 diffusers/VAE。

## 调用账和失败记录

普通完整任务仍为 1 load + 6 decode + 2 VJP。`counts` / `call_history` / `calls/` 保存跨进程物理尝试与返回；每次 gradient decode 的 replay/spool 记录随该次调用保留。中断后可能多一次 load，未完成 VJP 可能多一组 gradient decode/VJP，不能用普通预算遮掉。

`attempt_history`、`attempts/` 和 `notebook_attempts/<UTC>/` 保留历史状态、失败、资源和日志。本次成功恢复可以变为 COMPLETE；旧失败不会删除，也不会让已成功补完的任务永远 ENGINEERING_FAILURE。未完成或不可读材料继续保留固定分母。

## 工程检查与证据上限

针对定时快照/损坏回退、完整梯度中断、未完成 VJP 重算、错误输入梯度拒用、RGB 已存后的 CPU 评分、RGB 损坏时保持读数、跨进程硬退出且临时目录消失、全完成零 GPU、合法终局、builder/notebook/ZIP 同步、重复 Run all 环境和独立 CLI 做 CPU/fake/static 检查。具体命令及结果见主目录 `diagnostics/b-line-all-bit-resume-20261012/validation.md`。

未运行真实 GPU/Colab、未执行新实验、未推送或合并 main。Drive FUSE 原子替换和真实到期恢复仍需用户在真实环境观察；CPU 检查不证明 Colab 掉电期间未完成写入的持久性，也不产生新的科学结论。

参考实际代码：CEG-WM `src/cegwm/formal_experiment.py:FormalRunStore` 的完成单元复用，以及 A 线 `6d9f66b` 的 `experiments/paper_results_v1/resume_state.py`。只复用工程模式，没有导入旧身份/摘要/序列完整性门禁。B runtime 不依赖 experiments。
