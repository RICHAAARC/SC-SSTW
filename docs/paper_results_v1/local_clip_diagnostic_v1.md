# 当前残差 A 的固定局部片段诊断

本诊断已经采用，固定方法和预算于 2026-10-16 经主审核定。复用 `resumable-independent-four-v1` 的四个 residual candidate：`independent_20261015_01..04`，不重新生成或写入，不重跑 RAW8。它检查一个具体中间片段及其删除/重复视图，不代表任意长度、起点、全部块相位或广泛时域鲁棒性。

## 固定输入与分母

每个来源取零起点源帧 `[37:126]`，即 37–125（含端点），共 89 帧，37 mod 4=1。

| 视图 | 操作 | 收到帧数 |
|---|---|---:|
| CROP_MID | 保留该片段 | 89 |
| CROP_DELETE | 删除局部位置 `[0,11,23,35,47,59,71,88]` 的帧各一次 | 81 |
| CROP_DUPLICATE | 在上述局部位置的原帧后各增加一份 | 97 |

三个视图共用同一来源已发布 MP4 的 received RGB，编辑后不再视频编码。四个独立源、十二个相关视图，RAW_U/CENTERED_U 各十二行/384 位，合计固定 **24 行/768 位**。所有失败、缺失、UNSUPPORTED、UNRESOLVED 保留；相关视图、两接收器和重复帧不增加独立源数。

## 复用观察和真实读取的边界

接收端使用固定 181 源时钟、K0 `watermark`，保持原 RAW_U/CENTERED_U 和公开 U 语法。先对保存的 clean `signed_projection` 与 `rho` 同步取行，再在收到的局部矩阵上重新计算列中位数；不把整段的中位数直接裁下来。完成固定 24 行 prepare 遍历并保存选择或具体失败后才做路径真值评价；目标消息只在固定真实读取遍历结束、所有行均为读取或失败终态后加入报告。中断期间对应真值字段为 `NOT_JOINED`。

普通 `fixed_method_semantics.json` 记录实际 crop、编辑位置、key、接收器、消息目标和固定分母，防止已知不同方法在同一任务中混用；它不记录或比较源码摘要、版本、路径和依赖。旧可选方法记录缺失、不可读或损坏时允许续跑，并记录既往字段不可核对的具体原因；可读的已知方法冲突仍具体报告。消息目标只存在于方法记录和事后评价，不进入执行 state 或 selector。来源 schema 标签只作定位记录；实际解析检查 rows 结构、必需字段、固定 key 和每个 case 的唯一 clean 行，缺失保留 MISSING，重复或非法结构保留 FAILED。

两个可选方法记录的写入若发生 OSError，仅把具体原因保存为记录警告并继续。必要执行 state、观察、latent、read 的持久化失败仍直接报告，中断仍向上传递。

操作沿用原 `decode_visible_span`：起点向下对齐四帧、最近收到帧填补、等距选更早收到帧、尾部截为 `1+4k`，R=min(44,k)，最低支持 R=1。报告保留完整 operation，包括 `front_boundary_copies`、`tail_discarded_source_positions`、`synthetic_output_indices`、`repeated_output_indices`、`dropped_received_indices` 和两个完整映射。填补不是找回被删画面，重复也不是新增独立支持。不存在为短片段新增的编码或接收器。

路径报告逐行给出起止位置及误差、区间/全路径精确性、支持误差和/均值/最大值、正确/错误对应、不同接收帧与不同真实源帧数、gap 和状态；不设 gap 阈值。载荷报告独立给出实际 32 位结果、错误数和未读位数。两臂十二对全部可评估才给总体配对 BER 差，部分配对不伪装完整收益。

## 调用预算与已完成 CPU 预分析

无中断名义上限为 1 次 Wan VAE load、24 encode、24 原载荷 read、4 次既有 MP4 CPU decode。新生成、DiT、writer、framewise encode、新视频 encode 均为 0。

四份 clean 原始分数可复用。仅在分数不可读时，可由同次保存的 framewise latent 用原函数做 CPU 分数重算；两者均缺则 MISSING，不加载新 framewise 模型。优先使用源 received RGB；不可读时每源最多解码一次既有候选 MP4并持久缓存，三个视图/两接收器共用。同源/同视图且完整 received_index_map 相同的输入可共享 Wan 编码；完整 operation 相同时才共享 read，否则各自保留时间注释与读取记录。R 相等不能代替输入相等。

采用设计后完成的保存分数 CPU 预分析固定为 24 次 selector：RAW_U 0/12 精确路径、误差和 13160；CENTERED_U 12/12、误差和 0。RAW_U 的 source04/CROP_DELETE 仅输出 1 帧/R0，按现有定义为 UNSUPPORTED；其余 23 行 supported 且没有相同物理输入。因此若用户 Drive 上相同观察可读，本次实际计划为 **23 次新 Wan encode/read**，修正后输入共 2299 帧、569 个 R 位置；24 行分母不变。分数、选择及统计只来自保存观察，**短片段真实 Wan 读取尚未执行，载荷结果未知**，不能从路径结果预告载荷收益。一个 unsupported 对意味着总体十二对 BER 差将保持不可完整计算，仍逐对报告其余结果。

## 用户 Run all

实际 notebook 为 `notebooks/paper_results_v1_local_clip_diagnostic_v1_colab.ipynb`；主会话完成必要文件发布后，入口为 [Colab](https://colab.research.google.com/github/RICHAAARC/SC-SSTW/blob/dev/paper-results-v1/notebooks/paper_results_v1_local_clip_diagnostic_v1_colab.ipynb)。本地交付阶段该链接仅指向拟发布路径，需以主会话远端检查为可用性确认。

1. 选择可执行 Wan VAE 的 GPU，直接 Run all；第一格挂载原有 Drive。
2. 自动获取当前 A dev 分支的普通源码 ZIP，无手动上传、摘要/版本/Git 状态门禁；实际下载、解压路径、导入、数值或文件错误具体报告。
3. 固定输出为 `Video-WM/Paper-Results-V1-Local-Clip-Diagnostic-V1/resumable-local-clip-diagnostic-v1`。先复用保存分数形成局部选择，再准备尚需的 Wan VAE，遍历固定逻辑行。
4. 中断后同一 Drive 再 Run all；完成 read、本任务自身 latent、精确相同输入及已有 CPU 解码缓存可复用。旧失败、中断和累计实际尝试保留；UNSUPPORTED/UNRESOLVED 不因结果不好而重算。
5. 返回完整输出目录，包含 `summary.json`、`evaluation_report.json`、`local_clip_state.json`、`fixed_diagnostic_spec.json`、`fixed_method_semantics.json`、`observations/`、`selections/`、`payload/`、attempt/checkpoint 和 `execution.log`。

本地 CLI：

```bash
python -m experiments.paper_results_v1.local_clip_diagnostic_v1_cli \
  --phase all --source-run /path/to/resumable-independent-four-v1 \
  --output /path/to/local-clip-output --model-path /path/to/wan-model --device cuda
```

实际 notebook 与 builder `experiments.paper_results_v1.build_local_clip_diagnostic_v1_colab` 同步。定向检查为 `python -m pytest tests/test_local_clip_diagnostic_v1.py -q`；CPU/fake/模拟 Colab 验证不等于真实 GPU、Drive 断线或实际载荷结果。
