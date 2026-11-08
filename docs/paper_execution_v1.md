# 分批、checkpoint 与最多六运行时

这是已采用论文实验的执行调度修订，未改变 DCT2＋CDC32、旧 H/L、QMatch、固定源、视图、阈值、科学调用或分母。旧结果与失败包保留；本次工程交付没有运行真实数据、模型或 GPU。

## 推荐：静态六片

六个独立 Colab runtime 都打开 `paper_parallel_worker_v1.ipynb`，选择 GPU，分别填写：

| runtime | ROLE | SHARD | 唯一任务 | 默认本次边界 |
|---|---|---:|---|---|
| 1 | main | 0 | 固定第1/3/5/7/9/11源，每源 C4 后同 Wan 四方法 | 1个完整源或软4小时 |
| 2 | main | 1 | 固定第2/4/6/8/10/12源，每源 C4 后同 Wan 四方法 | 同上 |
| 3 | public | 0 | 完整 pack 的全局序号 modulo4=0 | 1个完整pack或软4小时 |
| 4 | public | 1 | modulo4=1 | 同上 |
| 5 | public | 2 | modulo4=2 | 同上 |
| 6 | public | 3 | modulo4=3 | 同上 |

可用更少 runtime，依次跑这些固定组合即可，不更改分片数。每个组合只能有一个活动 worker。CAL有两个16源pack；TEST有32个pack（最后4源）。同一pack的三物理backend、PixelSeal/FrameIndex共享OFF读取、writer和质量依赖都留在本片。分片只划分原预算，不增加实验。main队列返回 `WORKER_TASKS_RETURNED`，public队列返回 `SHARD_RETURNED`，公开片结束后再次运行返回 `READ_ONLY_COMPLETED`。队列返回不代表测量完整：本片result仍可能为 `INCOMPLETE_RETAINED`，以本片结果与最终报告为准；安全暂停返回 `PAUSED_BOUNDARY`，保持参数重新Run all。

所有计算worker已停止或暂停后，释放其中一个runtime，选CPU运行 **`paper_finalize_v1.ipynb`**。它是唯一全局query/表格/latest写入者。主方法十二源终态齐后才执行原36次随机query/360声明；完成或已消费查询只读，不重抽。公开或同Wan未完成时仍输出全分母部分报告，不能把报告保存解释为全部实验成功。六个GPU计算worker不更新共享latest。

## 入口用途与互斥关系

| notebook | 是否必跑 | 依赖/用途 | 可并行 | 断开后 |
|---|---|---|---|---|
| paper_parallel_worker_v1 | 推荐计算入口 | 静态2+4片；GPU任务和相应CPU步骤在同Colab内完成 | 不同ROLE/SHARD最多六个 | 同组合Run all |
| paper_finalize_v1 | 最终必跑 | worker停止/暂停后CPU统计 | 只运行一份，不与计算worker并行 | 只读已有query，不补抽 |
| paper_suite_v1 | 单runtime替代方案 | 顺序轮转六片；每片MAX_ITEMS，整次软MAX_HOURS | 不与任何子入口同时跑 | 同入口Run all轮转，未完成物理工作按checkpoint续接 |
| paper_fpr_validation_v1 | 可选子集 | 两片C4及最终query；包含GPU，非CPU-only | 计算阶段可与public worker并行；汇总要等待其停止 | 同入口续接；不跑同Wan对照 |
| native_baselines_public_compact12_v1 | public单runtime替代 | 顺序轮转四公开片 | 可与main0/1并行；不能与public分片/完整入口同时跑 | 同入口续接 |
| b_paper_cohort_v1 | 可选C4单独入口 | TEST-A/TEST-B原名单；默认1源软4小时 | 不与main/完整/FPR争用同源 | 同入口续接 |
| native_baselines_public_fpr001_v4 | 本轮不必跑 | 旧40视图独立协议/结果只读引用 | 需要引用前先停旧worker | 保留原入口，非新compact续跑入口 |

单runtime方案完成计算后同样运行CPU汇总入口。完整/FPR入口在时限前可能自动进入finalizer；如果时限已到或public仍在运行，则留待单独finalizer。不得同时打开完整入口与它的子入口。跨VM不存在可信的分布式锁承诺：依靠明确静态分工与操作者停止旧runtime，本地锁/ACTIVE记录用于阻止已知争用，不以心跳超时擅自接管。

## 实际恢复边界

- C4保存生成上下文、终端latent、native特征、CPU求解输入/返回、接收数组、身份中间返回及RNG；同Wan保存writer像素、decode返回、native read批次、query及RNG。逐调用先记STARTED，再保存实际值，最后记RETURNED。复用真正保存值，不只跳过状态行。
- 在源、阶段/调用或public视图/任务边界暂停。MAX_HOURS是软调度边界，不杀正在运行的生成/求解/模型调用，单次可能超出4小时。不能承诺任意中途断电无损。
- 已STARTED但返回文件不完整的C4/同Wan调用标为不可准确恢复，不隐性补跑。算法失败不重试；旧无checkpoint失败/消费包只读，不改成新attempt。完整返回已落盘但状态未完成时可恢复其返回，不重复物理调用。
- 接收/身份累计时间上限不重置。未闭合阶段没有可读返回/时间记录时如实停止，不凭空补零。恢复模型上下文是额外实际加载，写入 `restore_overhead.jsonl` 并进入成本表，**不声称包含在原C4每源1114或同Wan每源534次逻辑调用内**。
- 新context恢复受每源/owner最多65次运行记录上限；不能因此新增科学writer/reader。公开线四片基础预算总和仍等于原预算；固定恢复额度分摊，原unsharded compact已花调用/运行次数先扣除。另三片初次LPIPS加载占原恢复额度，不复制四份额度。
- 正常 `PAUSED_BOUNDARY` 已关闭owner，可直接再Run all。换runtime且旧owner仍ACTIVE时，先实际停止旧runtime，再设置 `PREVIOUS_RUNTIME_STOPPED=True`。此参数不是允许并发/算法补跑。若耗尽额度或不可恢复，保留缺失/失败行，最终报告仍保留原分母。

## 目录、缓存和报告

正式结果都在 `/content/drive/MyDrive/Video-WM_Result`：

- 主方法原 `main_method/paper_cohort_v1/<sid>/attempt01` 不换目录。
- 同Wan原 `baselines/wan-cohort-five-view-v1/<sid>/attempt01` 不换目录。
- 新公开 `baselines/public-native-frameindex-compact12-v1/attempt01/shards/part0..3`，独立cursor、日志、query、预算与返回。原compact父包和旧v4只读复用；全局收集按固定source所属片→原compact→旧v4读实际存在的记录，现有坏记录不以较好的旧值替代。
- `paper_execution_v1/workers/main0..1`、`owners/`记录调度；每次 `paper_suite_v1/invocations/<uuid>`独立日志；最终 `paper_execution_v1/final_reports/<uuid>`包含主方法和public表。只有finalizer更新共享latest。
- 大数组与native特征在独立 `MyDrive/Video-WM_Cache/paper-checkpoints-v1`；公开cache在原compact cache根的 `shards/partN`。源码、权重与缓存均不放正式结果树。暂停不清缓存；C4/同Wan完整成功后仅释放对应自有cache，全部正式返回/原始最终读量/媒体/失败保留。

Drive需有能容纳当前活动源native特征及checkpoint的实际空间；新实现会增加序列化与Drive读写成本，尚未测量。没有摘要、manifest、版本相等或Git干净门禁。

## 时间规划和结论边界

先前C4四源共29733.889秒，剩余八源原估18–24小时总工作量；两main片理想约9–12小时，另加尚未测量的同Wan对照和恢复存储开销。公开compact旧规划约76.4小时计算量；四片理想约19小时/片，实际GPU/内容/Drive差异可能更长。默认每次1源/1pack、软4小时旨在分批，不保证Colab分配或单次时长；建议观察本片进度，在额度内续接。

工程检查仅人工CPU数据。真实跨runtime、Drive持久缓存及GPU运行尚待用户Colab验证。主张仍是时间/载荷/消息匹配与未决测量，不是完整时间认证；公开自然视频与Wan域分表；随机查询界与固定身份总体FPR分开。
