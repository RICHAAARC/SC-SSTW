# Paper Results V1 confirmation plan

状态：**固定方案，尚未执行**（2026-10-11）。本文件把已经采用的接收器决定和既有十源配置展开为八源 confirmation 运行计划。它不授权本地或代理执行 GPU、模型、VAE、codec、Colab 或 Drive 工作。

## 1. 冻结结论与证据上限

同批两 pilot 的真实开发结果为 90/90 行可评价：`RAW_U` 为 25/30 完整恢复、8/960 bit error；`CENTERED_U` 与 `CENTERED_D4` 均为 30/30、0/960。34 行复用、56 条新逻辑读和 30 次新物理 Wan 编码仅描述该开发批，不能按比例外推到八个 confirmation source。

本轮冻结：

- **主接收器：`CENTERED_U`**，即先计算收到的 `q=signed_projection/rho`，再对 q 的每列做中位数居中，并使用无 D4 约束的单调路径选择。
- **原版选路对照：`RAW_U`**，即未居中的原 U 选择器；这里的 RAW 指原始 score，不是固定源索引读取。
- **路径约束对照：`CENTERED_D4`**。D4 只用于判断额外驻留约束是否改变路径或载荷，不是成功条件，也不因开发批 30/30 被升级为必要机制。
- 载荷仍为固定 32-bit `A6D39C5E`；K0=`watermark`，K1=`watermark-wrong`。严格零值、Counter 平票、FFT 坐标、`R=min(44,k)`、1+4k 输入以及盲接收 truth 隔离均不变。
- confirmation 只报告冻结方案，不回用于选择接收器、改变 D4、攻击、阈值、消息、key、强度或运行预算。

证据上限是八个预声明 source、十五个固定时序条件、固定 writer/receiver/baseline/codec 下的 confirmation。八个 source 是报告的 source-level 聚类单位；本计划不声称它们的统计独立性。同一 source 的 15 条条件、多个臂、key 和接收器也不是独立样本。K1 与 OFF 是小规模负对照，不能称总体 FPR。

## 2. 固定 confirmation roster

这些条目来自 `experiments/paper_results_v1/real_eval.adopted.json` 和 `attack_eval.adopted.json`。它们与两 pilot 分开，名单和 seed 不因运行结果替换。

| case | seed | prompt |
| --- | ---: | --- |
| confirm_01 | 2026110201 | a white porcelain fox figurine beside a single fern frond on a dark oak shelf, locked camera, gentle leaf motion, no people, no cuts |
| confirm_02 | 2026110202 | a violet glass sphere drifting in a shallow clear acrylic tray, locked camera, soft side light, one object, no cuts |
| confirm_03 | 2026110203 | a green tin robot walking slowly across a plain cork board, locked camera, steady warm light, no people, no cuts |
| confirm_04 | 2026110204 | a folded orange paper crane turning slowly on a black felt surface, locked camera, steady neutral light, no hands, no cuts |
| confirm_05 | 2026110205 | a silver toy submarine gliding through a clear rectangular water tank, locked camera, steady cool light, no bubbles, no cuts |
| confirm_06 | 2026110206 | a purple wool ball rolling slowly along a pale concrete channel, locked camera, constant daylight, no people, no cuts |
| confirm_07 | 2026110207 | a yellow wooden spinning top rotating on a matte blue tabletop, locked camera, constant soft light, no people, no cuts |
| confirm_08 | 2026110208 | a black ceramic cat figurine on a slowly rotating white platform, locked camera, diffuse daylight, no people, no cuts |

当前身份描述保持 `USER_ADOPTED_LIMITED_LOCAL_NO_MATCH_UNPROVEN`：已有有限本地文本/归档检索未发现这些 seed/prompt，但这不证明全局未见、统计独立或语义未见，也不阻断固定 roster。近期使用历史由单独的 `confirmation_source_history.md` 记录，本文件不重复或改写该审计。

## 3. 固定 writer、baseline 与媒体链

每个 source 使用同一 prompt、negative prompt、seed 和 source noise：

1. `OFF_NATIVE` 与 `PAYLOAD_NATIVE` 各运行一条 50-step Wan 轨迹；P0 仅在 step 25..49 启用固定 payload 控制。
2. P0 的 181 帧只做一次 framewise encode；共享 latent 分别 decode 为 `PAYLOAD_FRAMEWISE_RECON`（P1）和写入后 decode 为 `PAYLOAD_FRAMEWISE_M05`。两条 decode 互不串接。
3. VideoSeal 与 RivaGAN 都从同一 `OFF_NATIVE` PRE RGB 视频嵌入固定有效 32-bit 任务。VideoSeal 使用 256 通道 `j mod 32` 重复（rate 1/8）；RivaGAN 使用披露的社区 32-bit checkpoint。两者都不得读取 truth 来选帧或选 bit。
4. 四个主臂先通过相同的 8 fps / libx264 / CRF 18 / yuv420p 完整视频发布；VideoSeal/RivaGAN 各自在 OFF PRE 上 embed 后，再分别用同一 codec 发布。RivaGAN tensor 接口不得暗含其旧 `mp4v/20fps` 路径。
5. 固定 15 条时序条件为：FULL、crop37:126、speed 0.75/1.25、mean3/5、三组 10% delete、三组 repeat insert、三组 adjacent equal insert。group seed 与 positions 完全沿用 `attack_eval.adopted.json`，不重抽。
6. FULL 直接使用各方法 POST；其余 14 条从该 POST 的 decoded RGB 做固定编辑，再按相同 codec 二次发布。条件内所有方法使用相同 recipe；二次发布影响必须与编辑共同解释。

VideoSeal 独立 256 信息位 native-K 模式、额外强度、额外 baseline、阈值/FPR、攻击扫描和 edited-video 第三次编码均不在本轮固定计划内。

## 4. 固定接收与结果分母

### 4.1 主方法路径和 payload

每个 `source × main arm × attack` 先对收到的视频做一次 framewise VAE encode；同一媒体的 framewise latent 在 K0/K1 间共享，key 只进入随后 clock/scoring。clock/scoring 用 latent、当前 key 和公开坐标生成 `signed_projection`、`rho` 及 `q=signed_projection/rho`。三个 selector 只消费各自的 `q` 和公开 grammar；消息与攻击 truth 只在选择结果持久化后进入报告。

固定逻辑表为：

| 项目 | 每 source | 8 sources | 说明 |
| --- | ---: | ---: | --- |
| 同步/路径行 | 4 arms × 15 attacks × 2 keys × 3 receivers = 360 | 2,880 | 每行保存 selector 状态、path、runner-up/tie、truth 后验路径误差；失败/UNRESOLVED/UNSUPPORTED 保留 |
| payload 行 | 360 | 2,880 | 每行固定 32 bit，共 92,160 planned bits；路径不可用或读失败不能从分母删除 |
| 冻结主结果 | 15 | 120 | `PAYLOAD_FRAMEWISE_M05 × K0 × CENTERED_U`；其中 FULL 8 行单列 control，非 FULL 112 行为攻击条件 |
| 原选择器对照 | 15 | 120 | `M05 × K0 × RAW_U` |
| D4 路径对照 | 15 | 120 | `M05 × K0 × CENTERED_D4`，不作为成功必要条件 |
| OFF 负对照 | 2 keys × 3 receivers × 15 = 90 | 720 | 不汇入主恢复率 |
| K1 负对照 | 4 arms × 3 receivers × 15 = 180 | 1,440 | 不称总体 FPR；与 K0 分表 |

OFF 与 K1 是交叉切片：`OFF × K1` 的 360 行同时属于上表两项，因此 720 与 1,440 不能相加成 2,160 个不同逻辑行。总表始终只有固定 2,880 行。

主表必须分别给 source-level 固定 15 条（及 14 条非 FULL）分母、成功、bit error、missing/failed/unresolved；不能把 120 view rows 当 120 个独立 source。同步路径指标与实际 payload exact/BER 分开报告，避免用路径吻合替代载荷结果。

### 4.2 外部 baseline

每个 source、每个 baseline、每个 attack 形成一个固定 K0 effective-32 row：2 × 15 × 8 = **240 rows / 7,680 effective bits**。VideoSeal 先在全部收到 frame/空间位置和 256 native channel 上保留完整 soft，再按固定 `j mod 32` 合并、`>0` 判位；RivaGAN 对全部预声明收到 frame 的逐 bit logits 等权均值、`>=0` 判位。native bit、soft、tie、shape/frame 错误和失败仍完整保留。

VideoSeal repeat32 是同 32 个有效信息位、不同 native 通道冗余/开销的对照，不是 256 独立信息容量，也不是同 code rate、同强度或同冗余。外部 baseline 与主方法按 `source × attack` 左连接；任一侧缺失保留，不做 success-only 平均。
成功评价的 baseline 行必须显式保存 `missing_bits=0`；失败、非有限、shape/frame 不符或 sidecar 不可读则保留固定行和相应 missing bits，不能沿用别的旧 source 状态补值。

### 4.3 质量与成本

质量只在未做时序编辑的固定 POST 上评价，不为 15 条攻击各造一套质量结果。每 source 固定七对：P0/OFF、P1/P0、P1/OFF、M05/P1、M05/OFF、VideoSeal/OFF、RivaGAN/OFF；8 sources 共 **56 quality rows**。每行独立保存 PSNR_RGB、LPIPS、reference-flow residual warp fluctuation 状态，某一指标失败不抹掉其他指标。LPIPS 覆盖完整 181 帧；Farneback 覆盖全部 180 个相邻 transition。`8 frames / 7 transitions` 只定义最差滑窗摘要，不是全指标只计算八帧。P0 含 payload，不是 OFF；PSNR 差不能当作可加性分解。M05/P1 与 M05/OFF 各保存一个并排视频，共 16 个展示视频，展示编码不算科学质量指标。

## 5. 确定调用量与物理复用边界

以下是没有中断重试时的固定方案调用量，不是 GPU 时间或峰值显存估计。2026-10-12 采用的续跑机制允许补做未提交的技术单元；额外实际尝试单独保留，不能把本表当作重试后的实际调用上限。

| 层 | 每 source | 8 sources | 计数口径 |
| --- | ---: | ---: | --- |
| Wan trajectories | 2 | 16 | OFF + P0，各 50 scheduler steps |
| scheduler steps | 100 | 800 | 50 × 2 |
| conditional/unconditional transformer forwards | 200 | 1,600 | 每 step 两个 branch |
| Wan native video decode | 2 | 16 | OFF、P0 |
| framewise writer encode | 1 video / 23 batches | 8 videos / 184 batches | 181 frames，batch 8 |
| framewise writer decode | 2 videos / 46 batches | 16 videos / 368 batches | P1 与 M05 各 23 batches |
| baseline embed adapter entries | VS 1 + Riva 1 | 16 | 同一 OFF PRE；无 hidden codec；整视频高层入口 |
| initial full-video codec | 6 | 48 | 4 main + 2 baseline |
| temporal edit transforms | 6 × 14 = 84 | 672 | CPU RGB recipe；不含 FULL |
| temporal edit codec | 84 | 672 | 编辑后第二次发布 |
| total scientific video codec | 90 | 720 | 48 initial + 672 edited；不含 16 个展示视频 |
| main clock/framewise encodes | 4 × 15 = 60 observations | 480 | 同一 observation encode 后供 K0/K1 score；逐 observation batch 总计 1,376/source、11,008 batches |
| baseline extract adapter entries | 2 × 15 = 30 | 240 | VS 120、Riva 120；整视频高层入口，失败仍占固定行 |
| quality row evaluations | 7 | 56 | 168 metric attempts；各指标独立状态 |
| LPIPS batch forwards | 7 × ceil(181/8) = 161 | 1,288 | 全 181 帧；batch 8 |
| CPU Farneback transitions | 7 × 180 = 1,260 | 10,080 | 全部相邻 transition；无 RAFT |
| quality display codec | 2 | 16 | 仅并排视频，单独列账 |

高层 adapter 行数不等同于底层 GPU forward 数。现有 VideoSeal 接口对应 8 次整视频 embed 与 120 次整视频 detect；其内部 batching 由真实 backend 回执记录。现有 RivaGAN tensor backend 逐帧调用：8×181 = **1,448 次 encoder**，15 个条件每 source 合计 2,701 个收到帧，因此 8×2,701 = **21,608 次 decoder**。这些是代码路径展开的确定调用数，不是耗时或显存实测。

Wan payload 物理 encode 按完整 `(received artifact, selected index map)` 共享；key/receiver 标签本身不能阻止相同 map 共享，也不能强迫不同 map 合并。共享仅止于 VAE latent：同 map 的 K0/K1 必须分别执行各自 key 的 FFT coordinate、votes 和 bit read，并分别持久化 keyed logical evidence。逻辑分母固定为 360/source，不随共享改变：

- 唯一物理输入范围为 **0..360/source，0..2,880 total**。媒体、clock、路径或支持不足可以令某些甚至全部 Wan payload encode 不发生，固定逻辑行仍保留。运行中断后的重复尝试另计。
- 条件下界：只有在每个 `arm × attack` 至少产生一个可读 map、且三接收器和两 key 全部共享该 map 时，才是 **60/source，480 total**；这不是无条件预算下界。
- 无共享保守上界：每个逻辑行 map 不同，为 **360/source，2,880 total**。
- FULL 的盲路径不假定不同 key/receiver 得到同一 map，因此不据此收紧保守上界。
- 开发批 26/30 centered 共享、30 次新物理编码不用于估计 confirmation。真实运行必须在模型调用开始/返回/异常时逐项落 actual ledger，报告 attempted/completed/failed/unfinished、每个物理 group 服务的逻辑行，以及实际共享数。

保存量按 181×320×512×3 RGB8 估算：六臂 PRE+POST 为约 1.0676 GB/source、8.54 GB/8 sources，不含 MP4、攻击输出、latent、soft、sidecar、cache 和临时工作空间。Wan latent、framewise latent 和模型缓存按现有 runner 逐阶段释放；峰值显存、wall time 和实际 MP4 总量未测，不设 GPU 型号门槛。

## 6. 执行顺序与固定失败保留

Run all 只执行 confirm_01..08；两 pilot 保留为开发证据，不加入 confirmation 分母。每个 source 分阶段运行，以便释放显存和局部恢复：plan/init → OFF/P0 generation → framewise P1/M05 → four main codecs → baseline embed → two baseline native codecs → temporal edits/codecs → main clock → main payload reads → baseline extracts → quality → report。

任何 source/arm/attack/key/receiver 缺失、失败、歧义或不支持均保留固定行。独立阶段失败不得抹掉已完成的其他方法；已有 COMPLETE 或有 durable sidecar 的昂贵调用不因 Run all/restart 静默重做。阶段 COMPLETE 只表示遍历结束，不能代替逐行科学成功。confirmation 输出与 pilot 输出分目录、分 cohort 汇总。

### 2026-10-12：Colab 到期后的固定任务续跑

新版真实 notebook 默认在 `MyDrive/Video-WM/Paper-Results-V1-Checkpoint-V1` 下创建任务，并以 `active_confirmation_run.json` 保持运行目录。之后挂载同一 Drive、Run all 自动接续。旧 confirmation 目录保留；旧轮 853 READ 的迁移不作为此次交付条件。

每个单元提交后立即保存结果与状态，模型执行阶段及 selector 准备阶段每 60 秒保存最近已提交的完整状态；每 32 次状态提交、阶段退出也保存 checkpoint。保留上一可读状态、两份快照和逐单元记录，`progress.json` 指向可实际恢复的快照。全状态损坏时读取可读快照并回放较新的单元；未知固定行 ID、缺少核心字段的单元只报告问题，不扩大固定分母或覆盖有效记录。动态物理 group 先于引用它的逻辑行提交。

恢复粒度：完整生成 trajectory、每臂 decode、共享 framewise latent 与各臂输出、每次媒体发布、clock 观测/已提交 key、baseline 嵌入/逐观测提取、逐项质量指标。payload VAE latent 先保存，再分别读取 K0/K1；全部 key 完成后可回收该 latent。latent/votes 临时文件先 flush/fsync 再替换。不能从未保存的 CUDA kernel、denoising step 或编码器内部 batch 接续；未提交单元可能重做，`unit_attempts`、`encode_attempts` 和阶段历史保留额外尝试及失败。

baseline PRE 和其他必要阶段输入保存于任务目录。`/content` 丢失时从已发布 MP4 恢复 RGB 缓存，沿 FULL 别名找到来源，避免重新生成或再次发布已提交的视频。已读、已评估、UNSUPPORTED/UNRESOLVED 及科学负结果保持不变；上游仍缺失的读取阶段为 PARTIAL，不能误标 COMPLETE。辅助日志损坏保留原文件并继续；同一 kernel 重跑时清理旧 experiments 包路径缓存，加载新源码。没有摘要、manifest、版本或严格模式门禁。

只借鉴 CEG-WM `formal_experiment.py:588-715` 的固定 JOB_ID、逐单元提交与按时间/数量 checkpoint 做法，不引入该参考的身份、精确一致性或连续序列限制。本地交付通过 CPU/stub 强退与真实 CPU 小张量持久化验证；Drive 挂载实际 I/O、Colab 运行时到期、模型执行和耗时仍待用户真实 Run all 验证。

## 7. 当前待决项与明确不变项

本计划没有新增阈值、攻击、receiver scan、source 替换、baseline、消息或质量指标，也没有需要在 notebook 实现前另行裁定的方法事项。

1. Notebook 沿用已成功准备路径的可编辑默认 cache/source/weight 位置；缺失时按既有下载/准备逻辑处理。路径和版本只作定位，差异不作准入门槛，真实 import/load 失败仍保留。
2. 本地任务只准备代码和 Run-all notebook，不代替用户运行。用户自行 Run all 与代理获得外部执行权限是两件不同的事，不把这一边界实现成 notebook 的审批门禁。
3. roster 的本地历史见 `confirmation_source_history.md`。它记录有限搜索与已知执行投影，不证明未见或统计独立；不换 source，也不阻断固定运行。

VideoSeal native-K=256 独立信息位模式、任何新阈值/FPR、额外攻击、强度匹配或预算扫描都需要新的明确方法决定；它们不作为本轮 notebook 的隐藏选项。
