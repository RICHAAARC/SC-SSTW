# DISTINCT_REF_U P2 独立来源确认运行包

2026-10-17：P2已采纳，运行包已准备并完成必要CPU合成/静态检查；真实四源生成、GPU/模型、媒体及跨Colab恢复尚未执行。默认CENTERED不变。主会话核验并发布当前dev后，使用 [实际notebook](../../notebooks/paper_results_v1_distinct_confirmation_p2_colab.ipynb) 在GPU Colab直接Run all。

固定四个新prompt/seed与八视图详见 [采用协议](../../experiments/paper_results_v1/distinct_confirmation_p2.adopted.json)：CROP_MID、CROP_DELETE、CROP_DUPLICATE、DELETE_S1、SHORT17、SHORT17_HOLD_EXACT、SHORT17_MEDIA、SHORT17_HOLD_MEDIA。四观察臂RAW_U/CENTERED_U/REF_U/DISTINCT_REF_U，加UNALIGNED/ORACLE；每源另有首次candidate发布FULL received的CLEAN/ORACLE，禁止用已含payload的NATIVE_PRE替代。

196逻辑行/6272计划位固定为3968结构存在位、2304结构缺失位；128次selector、8128臂路径位置。所有盲计划先保存，再读clean/oracle/消息。clean不完整精确时该源48攻击行保留BLOCKED_CLEAN，不换源。失败、UNSUPPORTED、UNRESOLVED、缺失、未读与误归属分开保留，完整存在分母未读时BER为null。

普通六视图直接索引首次读回RGB与其逐帧Y。两个MEDIA视图分别以实际17/35帧固定libx264/8fps/CRF18/yuv420p再发布一次，再framewise编码；不伪填181。DC/增益/完全相等类等权列中位/原U公式不变，背景只用当前收到视图的行。每个视图候选优先实际读取，encode依据完整输入映射共享；read还要求完整plan/cells/key/投票语义一致，单独记录alias与物理调用。失败的同一物理read不能借另一臂名重试。

rowmax只有诊断输出：精确并列UNRESOLVED_ROW_TIE；非单调保留原序列ROWMAX_NONMONOTONE，不排序、不新设载荷臂。重复组保存RGB、Y、完整Q_ref的精确类与全pair最大绝对/RMS差；MEDIA仍完全相同时保留NOT_EXERCISED_EXACT，不更改codec。普通/HOLD二次发布与各自一次发布另列逐帧差量，不当接收器效果。

| 新源预算 | 上限 |
|---|---:|
| generation load / trajectory | 各4 |
| native step / DiT双分支合计 / 新局部payload gradient | 200 / 400 / 100 |
| native Wan load / decode | 1 / 4 |
| framewise load / encode clips、batches / decode clips、batches | 1 / 16、216 / 8、184 |
| clock write / residual compose | 各4 |
| codec save、probe、readback | 各12 |
| U selector | 128 |
| reader Wan load / encode / physical FFT read | 1 / ≤196 / ≤196 |

100次新局部梯度是四条新轨迹固有工作，不重算旧P1/旧28/84或旧生成。没有OFF轨迹、感知模型、外部baseline、扫描、增幅或调参。NATIVE_PRE质量只支持新增clock/codec变化，总writer质量仍缺。

固定输出：

```
/content/drive/MyDrive/Video-WM/Paper-Results-V1-Distinct-Confirmation-P2/resumable-distinct-confirmation-p2
```

同Drive再次Run all读取该目录，复用可读已完成产物，只继续从未开始且预算可用的单元。STARTED但未完成、失败、中断、产物损坏均保留且不自动重算；native/framewise/reader各1次load，无额外恢复load额度。因此某些中断会耗尽加载预算并保留阻断，不能承诺任意断点都能续跑。模型依赖只按剩余可执行工作准备；源码下载失败可使用同目录可读ZIP，不设摘要、精确版本、Git干净状态或strict/override门禁。实际不安全路径、不可读文件、形状/数值/模型调用错误仍直接报告。

CLI与notebook是同一入口，无运行模式菜单：

```sh
python -m experiments.paper_results_v1.distinct_confirmation_p2_cli --output /path/to/p2 --wan-model /path/to/wan --framewise-model /path/to/framewise
```

结果回收整个输出目录，主要文件包括summary.json、row_results.json、distinct_confirmation_p2_state.json、records/plan、records/read、votes、observations、rowmax_diagnostics.json、repeat_layer_diagnostics.json、media_pair_diagnostics.json、paired_rows.json和execution.log。保留所有196行，不能只交成功行。真实运行完成后再按这些记录核对科学结果；本地合成与静态通过不等于真实成功。
