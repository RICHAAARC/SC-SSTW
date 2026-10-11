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

每个 `source × main arm × attack × key` 先对收到的视频做一次 framewise VAE encode，产生 key 对应的 signed projection/rho；同一媒体的 framewise latent 在 K0/K1 间共享，key 只进入随后 score 计算。三个接收器仅消费 received score、rho、当前 key 和公开规则；消息与攻击 truth 只在选择完成后进入报告。

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

主表必须分别给 source-level 固定 15 条（及 14 条非 FULL）分母、成功、bit error、missing/failed/unresolved；不能把 120 view rows 当 120 个独立 source。同步路径指标与实际 payload exact/BER 分开报告，避免用路径吻合替代载荷结果。

### 4.2 外部 baseline

每个 source、每个 baseline、每个 attack 形成一个固定 K0 effective-32 row：2 × 15 × 8 = **240 rows / 7,680 effective bits**。VideoSeal 先在全部收到 frame/空间位置和 256 native channel 上保留完整 soft，再按固定 `j mod 32` 合并、`>0` 判位；RivaGAN 对全部预声明收到 frame 的逐 bit logits 等权均值、`>=0` 判位。native bit、soft、tie、shape/frame 错误和失败仍完整保留。

VideoSeal repeat32 是同 32 个有效信息位、不同 native 通道冗余/开销的对照，不是 256 独立信息容量，也不是同 code rate、同强度或同冗余。外部 baseline 与主方法按 `source × attack` 左连接；任一侧缺失保留，不做 success-only 平均。

### 4.3 质量与成本

质量只在未做时序编辑的固定 POST 上评价，不为 15 条攻击各造一套质量结果。每 source 固定七对：P0/OFF、P1/P0、P1/OFF、M05/P1、M05/OFF、VideoSeal/OFF、RivaGAN/OFF；8 sources 共 **56 quality rows**。每行独立保存 PSNR_RGB、LPIPS、reference-flow residual warp fluctuation（8 frames/7 transitions）状态，某一指标失败不抹掉其他指标。P0 含 payload，不是 OFF；PSNR 差不能当作可加性分解。M05/P1 与 M05/OFF 各保存一个并排视频，共 16 个展示视频，展示编码不算科学质量指标。

## 5. 确定调用量与物理复用边界

以下是固定方案的调用预算，不是 GPU 时间或峰值显存估计。

| 层 | 每 source | 8 sources | 计数口径 |
| --- | ---: | ---: | --- |
| Wan trajectories | 2 | 16 | OFF + P0，各 50 scheduler steps |
| scheduler steps | 100 | 800 | 50 × 2 |
| conditional/unconditional transformer forwards | 200 | 1,600 | 每 step 两个 branch |
| Wan native video decode | 2 | 16 | OFF、P0 |
| framewise writer encode | 1 video / 23 batches | 8 videos / 184 batches | 181 frames，batch 8 |
| framewise writer decode | 2 videos / 46 batches | 16 videos / 368 batches | P1 与 M05 各 23 batches |
| baseline embed | VS 1 + Riva 1 | 16 | 同一 OFF PRE；无 hidden codec |
| initial full-video codec | 6 | 48 | 4 main + 2 baseline |
| temporal edit transforms | 6 × 14 = 84 | 672 | CPU RGB recipe；不含 FULL |
| temporal edit codec | 84 | 672 | 编辑后第二次发布 |
| total scientific video codec | 90 | 720 | 48 initial + 672 edited；不含 16 个展示视频 |
| main clock/framewise encodes | 4 × 15 = 60 observations | 480 | 同一 observation encode 后供 K0/K1 score；逐 observation batch 总计 1,376/source、11,008 batches |
| baseline extracts | 2 × 15 = 30 | 240 | VS 120、Riva 120；失败仍占固定行 |
| quality row evaluations | 7 | 56 | 168 metric attempts；各指标独立状态 |
| quality display codec | 2 | 16 | 仅并排视频，单独列账 |

Wan payload 物理 encode 按完整 `(received artifact, selected index map)` 共享；key/receiver 标签本身不能阻止相同 map 共享，也不能强迫不同 map 合并。逻辑分母固定为 360/source，不随共享改变：

- 无条件实际范围为 **0..360/source，0..2,880 total**。媒体、clock、路径或支持不足可以令某些甚至全部 Wan payload encode 不发生，固定逻辑行仍保留。
- 条件下界：只有在每个 `arm × attack` 至少产生一个可读 map、且三接收器和两 key 全部共享该 map 时，才是 **60/source，480 total**；这不是无条件预算下界。
- 无共享保守上界：每个逻辑行 map 不同，为 **360/source，2,880 total**。
- FULL 的盲路径不假定不同 key/receiver 得到同一 map，因此不据此收紧保守上界。
- 开发批 26/30 centered 共享、30 次新物理编码不用于估计 confirmation。真实运行必须在模型调用开始/返回/异常时逐项落 actual ledger，报告 attempted/completed/failed/unfinished、每个物理 group 服务的逻辑行，以及实际共享数。

保存量按 181×320×512×3 RGB8 估算：六臂 PRE+POST 为约 1.0676 GB/source、8.54 GB/8 sources，不含 MP4、攻击输出、latent、soft、sidecar、cache 和临时工作空间。Wan latent、framewise latent 和模型缓存按现有 runner 逐阶段释放；峰值显存、wall time 和实际 MP4 总量未测，不设 GPU 型号门槛。

## 6. 执行顺序与固定失败保留

Run all 只执行 confirm_01..08；两 pilot 保留为开发证据，不加入 confirmation 分母。每个 source 分阶段运行，以便释放显存和局部恢复：plan/init → OFF/P0 generation → framewise P1/M05 → four main codecs → baseline embed → two baseline native codecs → temporal edits/codecs → main clock → main payload reads → baseline extracts → quality → report。

任何 source/arm/attack/key/receiver 缺失、失败、歧义或不支持均保留固定行。独立阶段失败不得抹掉已完成的其他方法；已有 COMPLETE 或有 durable sidecar 的昂贵调用不因 Run all/restart 静默重做。阶段 COMPLETE 只表示遍历结束，不能代替逐行科学成功。confirmation 输出与 pilot 输出分目录、分 cohort 汇总。

## 7. 当前待决项与明确不变项

本计划没有新增阈值、攻击、receiver scan、source 替换、baseline、消息或质量指标，也没有需要在 notebook 实现前另行裁定的方法事项。

1. Notebook 沿用已成功准备路径的可编辑默认 cache/source/weight 位置；缺失时按既有下载/准备逻辑处理。路径和版本只作定位，差异不作准入门槛，真实 import/load 失败仍保留。
2. 本地任务只准备代码和 Run-all notebook，不代替用户运行。用户自行 Run all 与代理获得外部执行权限是两件不同的事，不把这一边界实现成 notebook 的审批门禁。
3. roster 的本地历史见 `confirmation_source_history.md`。它记录有限搜索与已知执行投影，不证明未见或统计独立；不换 source，也不阻断固定运行。

VideoSeal native-K=256 独立信息位模式、任何新阈值/FPR、额外攻击、强度匹配或预算扫描都需要新的明确方法决定；它们不作为本轮 notebook 的隐藏选项。
