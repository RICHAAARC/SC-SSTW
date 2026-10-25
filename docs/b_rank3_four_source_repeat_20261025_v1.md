# B-RANK3 独立复跑 repeat-20261025-v1

这是同四来源、相同prompt/seed/a和同一固定方法的独立执行记录，不是新增独立样本确认。使用同一已修复 RANK3 runtime；γ=1、旧背景/τ、五读法、全部固定预算和失败分母保持不变。

新入口：[Colab Run all](https://colab.research.google.com/github/RICHAAARC/SC-SSTW/blob/dev/local-joint-state-payload-v1/notebooks/b_rank3_four_source_repeat_20261025_v1.ipynb)，由 `scripts/build_b_rank3_four_source_repeat_20261025_v1_notebook.py` 生成，随 B 分支的缓存/临时张量修改一同发布。发布检查未实际认证、读写Drive、运行模型或codec；真实执行由用户 Run all 启动。旧确认Notebook及旧结果目录保留。

默认输入仍为 `/content/drive/MyDrive/Video-WM/B-Terminal-Three-Group-V1/fixed-four-new-v1`。本次唯一输出为 `/content/drive/MyDrive/Video-WM/B-Rank3-Four-Source-Confirmation-V1/repeat-20261025-v1`。Run all先挂载Drive，再用Colab auth与google.auth.default获取Drive API会话，按MyDrive目录路径逐级唯一定位。实时storageQuota只报告实际返回值，不设置未经采用的容量阈值。

云端或挂载端已有任何执行痕迹，或API检查未知，就不下载源码、不setup、不启动worker。身份/路径错误、重复同名目录直接说明，不自动选择其他账户或目录。普通setup.log/runner.log不算执行预算。仅由原Notebook的OUTPUT.mkdir建立目录；不通过API另建一个同名文件夹。

成功源码和环境准备后，再次API检查云端执行痕迹。挂载建立的目录尚未可见时，仅作最多六次、间隔五秒的目录可见性读取；仍不可见即不启动。目录唯一可见且无执行记录时，API只发出一次repeat_start.json创建请求并完整读取解析该启动收据，然后启动原CLI。创建或读回响应不确定则停止并保留START_UNKNOWN，不自动重试创建。该普通收据不伪装成runtime的attempt.json，也不宣称提供同时启动会话的原子互斥。

后续Run all只读取本次结果，并可重新做保存检查与备份。临时模型/张量缺失、Drive同步滞后、文件缺失均不触发物理恢复或重算；旧首失败/中断耗尽已STARTED预算、未进入行保留的规则不变。模型/HF缓存和临时张量沿用当前/content会话策略。

保存确认与计算完成分开：

- Drive API通过files.get alt=media完整读取并解析15个核心根JSON：result、evaluation、ledger、planned_outputs、plan、protocol、background、thresholds、physical_counts、physical_events、plans_complete、blind_complete、post_truth_access、attempt、progress。
- 对48个固定最终MP4流式读至EOF，检查实际读取字节数；不把挂载文件存在、close或fsync当作云端完成，不执行解码、ffprobe或补编码。
- 若计算记录声明完成，进一步检查既有完成语义：POST_RECORDED、72任务/76质量行完成、固定ledger返回、实际物理计数以及plan/blind回读标记。云端仍是中间态或与本地完成字段冲突时，保存保持未确认。对最终失败attempt也比较既有finished/spent进度；云端落后仍未确认，同等失败终态则允许确认保存，不要求失败变成功。attempt仍为STARTED、progress仍为post是原运行器允许的最终形态，不被误判。
- 成功状态仅为CORE_RESULTS_AND_48_MEDIA_CLOUD_READABLE，不能据此说所有附属JSON都已逐一API验证。保存状态不会改写method_status、contribution_status或原科学结论。

备份位于/content，包含本地所有可读JSON、云端已读取的核心JSON以及48视频。JSON按已有完成/进度语义优先保留更完整的副本，不用摘要或整文件相等比较；每项记录实际备份来源。云端旧路径可读与本地最终内容已云端保存是两个不同字段。损坏JSON原件可另存unparsed_local，但备份标不完整。媒体fallback不得为空；已有唯一云端size时长度不符也标不完整，不补跑。云端目录列出但本地缺失的附属JSON列入缺失清单，不追加数千次内容GET。

备份完整仅指当前已列出JSON集合和48视频均成功装入ZIP；目录/文件读取失败、缺项、解析失败或磁盘写入失败全部列明。它不是不可篡改或逐文件完整性证明。具体报告与ZIP一同放在/content；files.download只能报告已生成和已发起下载，用户电脑是否保存完成需看浏览器下载结果。

必要CPU检查仅使用API/文件系统stub，覆盖隐藏云端attempt、未知创建不重试、真实流式读取接口、云端中间态与本地最终备份优先、损坏JSON/空视频/缺项、Notebook在source/setup/worker前只读。原RANK3七项检查和根侧15项缓存/存储改动保留，不扩展物理预算或方法路线。
