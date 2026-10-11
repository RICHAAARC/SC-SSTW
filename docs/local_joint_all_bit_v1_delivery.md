# B 线全 32 位余量目标：用户 Run all 交付

日期：2026-10-11。用户已采纳 [v2 整组具体参数](local_joint_all_bit_proposal_20261011.md)，原话为「采用这组参数，进入最小实现」。实现位于 `dev/local-joint-state-payload-v1`；旧 M0 / Mean Refresh 入口、checkpoint 和历史结果保持原样。当前仅 CPU/static/stub 验证；真实 GPU、模型、Colab、Drive 或媒体执行均由用户完成。分支发布由主审处理。

## 固定构造与入口

- 目标：`0.8 mean_32 H_0.02(m_i) + 0.2 H_0.01(G)`；全部32位均匀入式，G为软下限。每步只对这一个标量做VJP，mask后全向量归一化下降0.5；累计投影中心始终是原BASE，L2≤1，不补满半径。
- 从原BASE和新HALF分别计算两个新VJP。旧 TWO_MEAN/delta.pt 只作保存对照，不重归一化、不重mask、不重跑旧任务。零梯度或活动state max并列停止依赖方向；payload-min并列、平坦state并列不阻断。两个有效方向相消为零仍是正常最终候选。
- 固定四行：BASE、TWO_MEAN_SAVED、ALLBIT_HALF、ALLBIT_TWO。主比较是 ALLBIT_TWO 对 TWO_MEAN_SAVED，另保留每行对BASE及HALF→TWO。无扫描、择优、回滚、缩步、额外步、COMMON刷新或M1/M2。
- 完整非退化预算：1 load +6 decode +2 decoder VJP；0 encode/DiT/native/scheduler/codec。普通184 chunks、梯度原始92、名义replay92另记。固定352 windows /5632 chips /220原指标；梯度前向重复55值与loss、质量均不增加分母。

直接入口：[local_joint_all_bit_v1_colab.ipynb](../notebooks/local_joint_all_bit_v1_colab.ipynb)，普通源包：[local_joint_all_bit_v1_portable_source.zip](../notebooks/local_joint_all_bit_v1_portable_source.zip)。[builder](../scripts/build_local_joint_all_bit_v1_notebook.py) 与真实 notebook 同步。

用户选择CUDA runtime后 **Run all**，授权Drive挂载。首代码单元严格只有Drive import/mount两行；随后准备依赖、下载可编辑 `SOURCE_REF` 的普通ZIP、启动独立CLI。既有loader/logger/进程组清理路径复用，不增加精确版本、GPU型号、B64/hash/manifest/Git状态或源码身份准入。

默认输入路径可编辑：

```text
/content/drive/MyDrive/Video-WM/Local-Joint-State-Payload-V1/20261009T132316545817Z/run/joint/terminal_latent.pt
/content/drive/MyDrive/Video-WM/Local-Joint-State-Payload-V1-Mean-Refresh/20261010T145910398211Z/run/TWO_MEAN/delta.pt
```

新输出位于 `MyDrive/Video-WM/Local-Joint-State-Payload-V1-All-Bit/<UTC>/`。成功、构造未定义或工程失败均回传**整个UTC目录**，不只回传summary。旧对照缺失仍运行新路径并保留其MISSING行；BASE失败仍独立尝试保存旧对照，但停止依赖BASE的新路径。

无需 notebook / scripts / .git 的CLI：

```bash
python -m experiments.wan_state_clock.local_joint_all_bit_v1_run \
  --config experiments/wan_state_clock/configs/local_joint_all_bit_v1.json \
  --output /a/fresh/output/run
```

## 输出及失败含义

`result.json` 保留固定四槽、实际load/decode/VJP attempted/completed和时间、普通chunks、两份独立replay/spool记录、GPU allocated/reserved、RSS、cleanup与失败阶段。VAE decode返回即记录完成，不把后续adapter/reader失败错写成模型未返回。每次VJP在finally释放spool后再开始下一步；没有自动补满调用或重试。

每行保存计划delta、实际terminal、float RGB、raw与独立metrics（raw先落盘后读消息truth）、loss及系数、质量JSON。`comparison.json` 保留32行逐位表、负转正/正转非正集合、所有缺口、最差/平均margin、G、loss以及每步单标量预测与实际loss变化。`step_1/2.json` 报告投影尺度、计划累计/增量范数和 `g·(δ_next−δ)`；同时记录实际FP32 terminal相减后的FP64范数与预测，避免把加法舍入说成精确等式。

两次 `gradients/step_*_readout.json` 分别保存对应BASE/HALF的55个原值、活动项、loss、逐项差及最大绝对差。有限差异是诊断，不是相等准入。零梯度/活动state tie保留构造未定义且缺失依赖行；实际输入、I/O、reader、模型、VJP或非有限错误保留ENGINEERING_FAILURE。`COMPLETE` 只表示固定流程结束，仍可能含构造未定义行，永远不是科学PASS。

`quality.json` 保留全181帧与180相邻残差记录，含逐帧max_abs/坐标及ROI内外质量。每个非BASE行的frame RMSE、frame max_abs、temporal RMSE和temporal max_abs实际峰值按最早并列位置定位，取预定邻域并与旧固定帧并集；所有可用视图输出同一组帧。`visuals.json` 索引共享帧与峰值像素crop；每行的 `frames/` 和 `crops/` 输出对应RGB及残差图片。统一RGB [0,1]、残差 [-0.25,0.25]，报告超显示范围像素/分量数量，数值不截断。图片失败不阻断固定方法调用。外部中断后finalizer仅重建durable报告，不再读大张量或调用模型。

## 资源与证据上限

复用已实跑的 FP32 VAE输入梯度/cache/disk重放路径，不把无梯度offload当成梯度资源证据。旧Mean Refresh单VJP记录约64.53GiB暂存峰值，新任务有两个顺序VJP，会增加读写流量；不承诺新任务峰值、时长或磁盘可用容量。新运行报告才是实际资源证据。

新L降低不保证全位过零、G≥γ、质量改善或原描述条件成立。原描述条件仍为G>0且32/32 margins>0；μ/γ不是检测阈值。同一L2上限不等于两方案实际使用的扰动范数相同。本交付只检验一个已见保存终端的float已知消息指标，不能推出盲解码、codec、FPR、CFG/native因果、轨迹或科学成功。

## 必要检查及审查收据

A1定向检查包括：Huber分段/边界导数、32项系数及state活动并列；原reader标量方向差分；BASE球与相消/零梯度；两次新VJP分别对独立CPU因果decoder梯度逐值核对；6D/2VJP/184/92/92账及固定分母；保存对照原值/缺失、BASE或梯度decode失败、第二VJP非有限、质量I/O失败和报告接管；92/116峰值与邻域/crop；builder同步、普通ZIP闭包、全新notebook命名空间complete/killed/environment-failure顺序替身和无.git独立CLI。

本轮21项定向检查通过（41.73秒）；记录见 `diagnostics/b-line-all-bit-implementation-20261011/a1_targeted_tests.txt`，同目录保留 `cpu_full_run/` 与 `cpu_peak_fixture/` 替身产物。A2/A3同版独立审查与A4综合完成后在此追加收据；主审统一复用A5做A/B最终交付里程碑核验，B线不重复启动A5。没有真实模型或Colab执行；CPU替身不得写成真实视频效果。


A2 对首个实现候选 `f45d3c9` 给出有范围接受。A3 同版发现两项机械证据问题：目标普通decode/readout失败会覆盖已形成方向的状态；图片索引漏了子目录。已将目标观测失败与方向形成分开，保留DEFINED及对应视图的原始异常；HALF失败使第二步明确MISSING_DEPENDENCY。索引改为相对每个view目录的 `frames/...`、`crops/...`。

修复后仅做受影响验证，**11项通过（26.44秒）**：含新增4个HALF/TWO decode与readout失败边界、完整CPU两步、所有图片索引存在、实际峰值输出与5项notebook/普通ZIP/CLI检查。未重复未变的公式/reader/checkpoint全套。记录见 [修复复检](../../../diagnostics/b-line-all-bit-implementation-20261011/a1_targeted_repair_tests.txt)，修复后的完整CPU产物在同目录 `cpu_repaired_full_run/`、`cpu_repaired_peak_fixture/`；4个失败案例记录在 `cpu_target_failure_cases/`。首版21项检查及首版产物仍保留，未覆盖原始审查证据。等待A2/A3增量确认及A4综合。
