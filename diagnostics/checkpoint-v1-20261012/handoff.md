# A 线 checkpoint 续跑候选交付

2026-10-12。A1 实施完成，等待主审对冻结候选复核后发布。本地不 push、不 merge，不启动真实实验。

## 交付行为

首次 Run all 在 `MyDrive/Video-WM/Paper-Results-V1-Checkpoint-V1` 新建任务，之后使用同一 active pointer 自动续跑。旧轮目录不删除，旧 853 READ 快照迁移已不作为要求。实际 notebook、builder 与 71 文件的 companion ZIP 已统一生成。

逐单元结果立即提交；每 60 秒保存最近已提交的完整状态，每 32 次状态提交和阶段退出也发布 checkpoint/progress。保留上一可读全状态及两份快照、较新的单元记录。回放不增加未知固定行；缺 status/核心字段的记录只报告问题，不覆盖有效数据。动态物理 group 先于引用它的逻辑行提交。重开进程缓存原单元内容，避免重复写入所有微提交文件。

所有实际 confirmation 阶段支持重新进入和跳过已提交单元。技术错误、中断与额外调用保留。baseline PRE 留在任务目录；`/content` 丢失后从现存 MP4 恢复 RGB，含 FULL 别名和复用恢复缓存。payload latent 持久化后再读独立 K0/K1；key 完成后可回收该组 latent。latent/votes 临时文件关闭前 flush/fsync，再原子替换。

辅助日志损坏保留原文件并回退/继续，不阻断状态恢复；active pointer 的损坏 JSON、null/list、非法目录名均可回退。同一 kernel 的旧 experiments 包缓存会在初始化时清理，新源码父包路径已通过模拟测试。

固定方法、8 源 roster、2880 主行/92160 bits、240 baseline/56 quality、盲读顺序和 key 独立证据不变。0..2880 表示唯一物理输入，技术重试额外计数。科学负结果、UNSUPPORTED/UNRESOLVED 及已评估行不因续跑重做。没有新摘要、版本、manifest、严格/override 门禁。

## 精确验证

工作目录 `/home/richar/projects/Video-WM/worktrees/Paper-Results-V1`；使用已有 CPU 环境，未安装依赖。所有命令设置 `CUDA_VISIBLE_DEVICES=''`、`PYTHONDONTWRITEBYTECODE=1`。

```sh
/home/richar/projects/CEG-WM/alive/CEG-WM/.venv/bin/python -B -m unittest tests.test_paper_results_v1_checkpoint tests.test_paper_results_v1_stage_resume tests.test_paper_results_v1_confirmation tests.test_paper_results_v1_confirmation_notebook -v
/home/richar/projects/CEG-WM/alive/CEG-WM/.venv/bin/python -B -m pytest tests/test_paper_results_v1_real_eval.py -q --tb=short -p no:cacheprovider
/home/richar/projects/CEG-WM/alive/CEG-WM/.venv/bin/python -B -m unittest tests.test_paper_results_v1_attack_eval -v
python3 -B diagnostics/checkpoint-v1-20261012/validate_artifacts.py
git diff --check
```

- 18/18 checkpoint/confirmation/notebook 测试通过，36.346s 命令耗时。包含 10 阶段实际控制路径的 `os._exit(23)`、删除模拟 `/content`、新进程续跑；VAE 前/中/后、K0 后退出；已评估行保持不变；周期可读快照、全状态损坏、微提交先于主状态替换；固定分母/损坏记录；旧父包缓存；辅助日志损坏；真实 CPU 小张量及 NPZ 的 fsync 顺序和读取。
- real_eval：33/33 通过，2.191s 命令耗时；attack：8/8 通过，6.558s。
- 最终 notebook 7 个代码 cell AST 合法、输出清空；ZIP 71 个源文件与当前源文件内容一致。无 `.git` 解压目录中三套 CLI init 全部成功；confirmation 固定 3600 计划行中 2880 为本次执行行。
- `test_results.json` 保存精确 argv、退出码、时间和日志路径；`artifact_validation.json` 保存最终构建结果。初次 CEG 环境检查的四个 fixture 失败已保留在 `targeted_tests-before-fixture-repair.log`：原因是 fake NumPy 被真实 Torch 清理导入，已为故障 worker 补充 Torch cleanup stub；真实 CPU 张量保存测试独立执行。最终 59 项均通过。

## 恢复粒度和证据上限

从最后已提交的 trajectory、臂输出、codec、clock/key、baseline 观测或质量指标继续。未落盘的整个单元可能再执行一次；不恢复未保存的 CUDA kernel、denoising step 或模型内部 batch。部分报告标记 PARTIAL_RESUMABLE，不能将过程退出码当科学成功。

CEG-WM 参考为 `alive/CEG-WM/src/cegwm/formal_experiment.py:588-715`：仅借固定 JOB_ID、单元微提交和按时间/数量 checkpoint，不借其身份或连续性准入。这里没有 GPU、模型、VAE、codec、媒体、Colab、Drive、网络下载或安装执行；真实 Drive I/O 性能与云端持久性、Colab 到期恢复仍需用户挂载 Drive 后对最终已发布 notebook 执行 Run all 验证。CPU/stub 通过仅支持工程恢复行为，不产生新的科学结果。
