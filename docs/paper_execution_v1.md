# 连续执行与阶段级续跑

六个预填入口默认持续完成各自固定队列，`MAX_ITEMS=None`、`MAX_HOURS=None`。不会按四小时、一源或一包自动暂停。保持 DCT2＋CDC32、旧 H/L、固定源/视图、消息、阈值和统计分母。

| 入口 | 固定分工 |
|---|---|
| [主方法 0](https://colab.research.google.com/github/RICHAAARC/SC-SSTW/blob/main/notebooks/paper_parallel_main0_v1.ipynb) | main/0，第1/3/5/7/9/11源及同 Wan 对照 |
| [主方法 1](https://colab.research.google.com/github/RICHAAARC/SC-SSTW/blob/main/notebooks/paper_parallel_main1_v1.ipynb) | main/1，第2/4/6/8/10/12源及同 Wan 对照 |
| [公开 Baseline 0](https://colab.research.google.com/github/RICHAAARC/SC-SSTW/blob/main/notebooks/paper_parallel_public0_v1.ipynb) | public/0，完整 pack 分片 |
| [公开 Baseline 1](https://colab.research.google.com/github/RICHAAARC/SC-SSTW/blob/main/notebooks/paper_parallel_public1_v1.ipynb) | public/1 |
| [公开 Baseline 2](https://colab.research.google.com/github/RICHAAARC/SC-SSTW/blob/main/notebooks/paper_parallel_public2_v1.ipynb) | public/2 |
| [公开 Baseline 3](https://colab.research.google.com/github/RICHAAARC/SC-SSTW/blob/main/notebooks/paper_parallel_public3_v1.ipynb) | public/3 |

每个入口选择独立 GPU runtime，挂载同一个原 Drive 后 Run all。同一片只开一份，不与完整入口或重叠子入口同时运行。断线后先实际停止旧 runtime，再在同一个入口 Run all；旧占用记录仍 ACTIVE 时设置 `PREVIOUS_RUNTIME_STOPPED=True`。这个参数用于确认旧任务已停止，不允许并发占用同片。

## 恢复粒度

采用 CEG-WM 的外层思路：任务完整返回立即提交实际输出，持续更新进度，重开跳过已完成任务。主方法分为准备/生成、求解、回放/物化、接收、身份读取五个阶段。完整阶段的实际输出和下一阶段输入保存在原 Drive 结果及独立缓存中；`stage_checkpoint.json` 指向最后完成阶段。中断后重新执行当前未完成阶段，不重做已完成阶段、不新增源或改变 seed/消息。生成、求解、VAE 内部不承诺任意位置无损恢复。

每次阶段尝试使用独立输出目录。前次中断的部分输出、未知调用、资源记录和失败状态保留。下一阶段只从已完成阶段复制输入；未完整返回或不可读的阶段不提交为完成。内部生成和求解算法使用原代码，未引入训练或新检测规则。

同 Wan 对照继续复用其已完成 writer/read/codec 批次；未完成任务使用单列基础设施恢复记录。公开 Baseline 沿用其已有 pack/视图 checkpoint 和原定恢复预算。PixelSeal/FrameIndex 的共享计算留在同一片。

每个源的主方法和同 Wan 对照共享最多两次基础设施恢复。此额度允许当前未完成阶段/任务重算，取代本轮曾讨论的单原子或100迭代精确恢复要求。原每次方法调用限制仍在各次尝试内执行；阶段重算是额外实际成本，原已用调用、未知返回与已知耗时不清零。累计成本按所有尝试报告，硬中断未观测尾部标为未知。数值失败、已触发的资源硬限或旧终态失败不自动重试。

## 结果与汇总

正式结果及失败包保留在 `MyDrive/Video-WM_Result`。主方法仍为 `main_method/paper_cohort_v1/<sid>/attempt01`；阶段尝试位于其 `stage_attempts/`，迁移前非终态记录保留在 `before_stage_resume/`。同 Wan 和公开结果仍在原 baselines 子目录。大数组、native 特征、源码与权重留在结果目录之外；本轮不删除中断缓存或旧结果。最终成本表列出所有阶段尝试与共享恢复记录，固定科学分母不重复累计。

所有计算 worker 停止后，单独运行[唯一 CPU 汇总入口](https://colab.research.google.com/github/RICHAAARC/SC-SSTW/blob/main/notebooks/paper_finalize_v1.ipynb)。它读取已存结果并执行依赖齐备的原定随机查询，不重抽已消费查询；缺失和未决保留。公开自然视频与 Wan 域分表，队列结束不等于科学 PASS。

本轮仅必要人工 CPU 工程检查。真实跨 Colab/Drive 的运行性能和 GPU 表现由用户验证。阶段重算耗时取决于中断位置；历史准备阶段约17分钟、求解约85分钟，整源约122–127分钟，不把整源时间称为单次生成时间。
