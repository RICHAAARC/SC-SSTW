# B-RANK3R M3：四新源、五片段固定确认

本入口随 B 分支发布，Notebook 默认直接读取该分支可用源码。本轮完成 CPU 假后端、格式和接口检查及独立审查，没有执行真实 M3 Run all、模型、VAE、codec 或 Drive 操作；方法结论仍须由真实保存结果判定。

入口：`notebooks/b_rank3r_new_source_confirmation_v1.ipynb`；生成器：`scripts/build_b_rank3r_new_source_confirmation_v1_notebook.py`。Notebook 首个代码单元只有两行 Drive 挂载，随后先通过 Drive API 检查执行痕迹。普通源码由可用 B 分支下载到本次会话临时隔离目录，无需源码 ZIP、摘要或版本相等门。显式 SOURCE_DIRECTORY 也复制后使用，不修改用户源目录。

默认输入为 `B-Rank3-Four-Source-Confirmation-V1/repeat-20261025-v1`，只要求可读、格式和数值合法的旧背景。历史 protocol/threshold 单列定位记录；当前 protocol 明确记录 M3 的四个 m 来源、gamma1、名单、固定预算和判据。输出为 `B-Rank3R-New-Source-Confirmation-V1/fixed-four-five-v1`，不覆盖旧阶段结果。

## 固定方法与观测

配置文件 `experiments/wan_state_clock/configs/b_rank3r_new_source_confirmation_v1.json` 固定四个 m 来源的 prompt、seed、a，以及 181 帧、320×512、8fps、50 步、CFG5、32 位 A6D39C5E、gamma1。五片段为 CROP、左右 delete8、左右 repeat8；seam 为49/102。沿成功链由完整视频第一代 H264 回读创建151帧片段，然后第二代 H264 回读进入三窗、四 phase E41。

RANK3R 使用独立4247类完整重复域，原 RANK3 保持961类。两排名读法不使用阈值；CDC_M 保持原 full-C 读法和旧 tau=0.49221603782894896，不校准、不扫描。三读法共享物理观测；不修改旧 PATHS、全局来源或旧入口。域规模和 seam aliases 不是物理调用；有唯一峰时两排名读法最多240次任务级 membership，精确并列保留方法擦除，不补执行以凑数。

每法120任务、3840槽：20正例/640槽，100负例/3200槽，其中 PM wrong-key 为20任务/640槽。360 reader plans 全部保存并回读后才打开位缓存，360 decisions 全部保存回读后才打开当前真值。接收器只见 key 与匿名观测地址，不见来源/臂/攻击真值；实际支持与错误归属只作后评。

| 调用或记录 | 固定完整成功分母 |
|---|---:|
| OFF/P0 原生轨迹、采样步、DiT、局部梯度、终态构造 | 8 / 400 / 800 / 100 / 4 |
| D181 | 12 |
| codec encode / probe / decode | 72 / 72 / 72 |
| E41 / 双 key Q / H family / H候选槽 | 720 / 1440 / 360 / 49320 |
| E41 DCT / 双 key位解码 | 720 / 1440 |
| native Q / DCT / 双 key位解码 | 8 / 8 / 16 |
| native / media 配对bit | 2816 / 9600 |
| 质量记录 / 帧 / 邻接帧 | 108 / 17148 / 17040 |

生成、FP32 VAE、ffmpeg、缓存与安装均复用成功四来源实现，保留 missing-ftfy 和 torchao/FqnToConfig 定点修复。VAE 重载计数器归零单独记录，不产生负调用差值。模型、HF 和暂存 tensor 位于 `/content`；Drive 保存必要 JSON、72个最终 MP4 及下面的归档，运行结束移除本次暂存 tensor。

## 方法状态、贡献与一次预算

总体、重复、裁剪删除三层检测/实际支持判据及逐源门槛均固定在配置中。主法 null DETECT 即失败，无输出也不豁免；字面错误、误归属、无实际支持输出均须为0。对 CDC_M 与原 RANK3 必须逐任务逐槽零丢失，不能以净增掩盖局部丢失。全部方法判据满足后，实际支持相对 CDC_M 净增至少32才贡献达标；贡献与方法状态分别报告。共同输出 BER 保留实际分母，分母0记未定义。摘要给三读法逐源检测/支持、paired added/lost/net，完整任务在 evaluation.json。

attempt 与固定计划先持久化，真实失败/中断消耗该次尝试，后续单元保留 NOT_ENTERED；缺测总体为 TECHNICAL_UNRESOLVED，同时保留已见方法违反。已开始的 Run all 再次运行在源码下载/环境/model 前只读，不自动补跑。Notebook 准备成功、worker前 API 写读普通启动收据；未知创建响应停止，不重试创建，且不宣称并发原子互斥。普通 CLI 也拒绝已有执行痕迹。null DETECT 后只完成已保存数据的规定逻辑诊断，不再物理工作。

## 保存与可选未来 M4 原始帧

OFF 量化后、第一代 H264 前的原始 uint8 RGB，以 THWC C-contiguous RGB24 写入 `m4_inputs/<source>/OFF/precodec.rgb24` 和可读 JSON sidecar。每份88965120字节，共355860480字节（339.375MiB），无额外编码。归档失败单列 ARCHIVE_MISSING、保留具体错误，不改变 M3 方法状态、不重新生成或重试；当前不执行 M4。

末尾 API 完整实读15个核心根 JSON 与72个最终 MP4，核对既有最终进度语义；4份 RGB24/sidecar 保存可用性单列。`CORE_RESULTS_AND_72_MEDIA_CLOUD_READABLE` 只说明这些核心结果与媒体，不暗示所有附属 Q/bit JSON 都逐个云端实读。归档缺失不覆盖核心保存状态或 method_status。完整本地 ZIP 包含全部可读 JSON、72媒体及4归档；不足明确标 INCOMPLETE_BACKUP，损坏本地 JSON 保留原件但不计完整。只声明备份已生成/发起下载，不声称用户电脑下载完成。旧云端中间态不覆盖更完整本地结果。质量仅记录 RGB距离/时序代价，没有新增感知 PASS 阈值。

该固定非 IID 小样本确认只支持本配置/来源/编辑范围的结论，不证明一般重复鲁棒、全攻击优越或外部同负载基线胜出。
