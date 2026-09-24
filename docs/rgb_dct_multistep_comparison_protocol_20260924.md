# RGB-DCT 同源多步控制比较：冻结实施协议

状态：用户已采纳的下一轮固定方法，交 A1 在隔离候选分支实施并交付用户自行 Run all 的 Colab。当前没有本轮 GPU、真实媒体或多步增益结果。原时间组一致性运行 `20260924T071328194840Z` 的六槽观察和旧负结果保持原样；本轮不复用旧媒体作分母。

## 机制问题与不变量

问题：在同一冻结 RGB-DCT key、正向载体和盲接收器下，把总的逐步原生响应 RMS 预算从终步 T49 移到 T46，或平均分给 T44/T46，最终 MP4 的时间组判别和质量怎样变化？这比较写入时刻及多步轨迹，不宣称预算等于终态扰动或能量。

固定 key `WanProjection-first-validation-key-v1`，key ID `785b91ae6b23bfc9`；接收器 spec SHA-256 `cee183c3b64a6de23dfa9879810488d320985409db464cb0b70dbfb1709f1caa`，对完整 MP4 的 30 个时间组计算 `C=sum(q_g>0)`，`C>=24` 判 H1，其余 H0，不校准、不调阈值。模型 `Wan-AI/Wan2.1-T2V-1.3B-Diffusers` revision `0fad780a534b6463e45facd96134c9f345acfa5b`；320×512、181 帧、8 fps、50 原生 UniPC 步、CFG 5、同一 negative prompt `text, watermark, logo, camera motion, cuts, multiple objects, flicker`。媒体固定 libx264/CRF18/yuv420p，FFmpeg RGB24 读回。载体为现有 RGB-DCT 正向同通道 `A=0.1*c_t*s_b*K`，VAE 为冻结 FP32 posterior mode，latent 仅保留时间索引 `1:45`，不引入像素臂、反号选择、梯度、幅度/时刻扫描或失败后回退。

每来源从同一新 noise 和完整 scheduler 历史分叉四臂：`OFF`、`SINGLE49`、`SINGLE46`、`MULTI44_46`。OFF 正常走完 0–49，并保存 T44/T46/T49 的 `z_t`、CFG `v_t` 和完整 UniPC snapshot。SINGLE49 严格复用已验证 T49 构造：`X0_49=decode(normal_step_49(z49,v49,history49))`。早步 T44/T46 则固定实时 clean proxy `x0hat_t=z_t−sigma_t*v_t`，`X0_t=decode(x0hat_t)`；该代理不等同于仍含噪的正常下一状态，也不声称与 T49 的 `X0_49` 数值等价。每次均独立计算 `P+=clip(X0_t+A,0,1)`、`u_raw=E(P+)−E(X0_t)`，蒙住 latent 时间端点；MULTI 在受控 T44、正常 T45 后，必须从它自己的 `z46`、新 CFG `v46`、完整已演化 history 重新计算 `X0_46` 与 lift，不能借 OFF/SINGLE46 的 T46 方向或 history。

每个受控时刻在当前臂同一 `(z_t,v_t,history_t)` 上计算正常下一状态 shadow，独立 unit native-response probe，按该 probe 选 epsilon，以 `v_t−epsilon*u_unit/sigma_t` 做一次原生受控步。原 history 在 shadow/probe 后不得污染；正式受控步更新该臂自己的完整 history。SINGLE46 和 MULTI 的其余步正常使用实时 CFG velocity 与 scheduler，MULTI 的 T47–49 全为自由步，不在终步补写。

统一预算 `R*=0.042943312697648145`，指标是当步受控下一状态减同史正常 shadow 的 latent `[:,:,1:45]` support RMS。SINGLE49/SINGLE46 各一次目标 `R*`；MULTI T44、T46 各目标 `R*/2=0.021471656348824072`。逐步记录目标与实际 support/global RMS、peak、响应误差和方向身份；每臂记录实际 `sum RMS(D_t)`、`sum RMS(D_t)^2`、逐步 peak、终态与同源 OFF 的 latent 净位移，以及成对完整 MP4 质量诊断。相同 `sum RMS` 不等于相同平方和、终态净位移或视觉质量。

## 冻结来源、分母与停机

仅有以下两项新来源，各四臂，固定 **2 来源、8 MP4/评分槽、1448 帧**。按名单尝试，不据第一来源结果换第二来源或跳过后续臂。

| 来源 | seed | prompt |
| --- | ---: | --- |
| `eval_kite_s2421` | `2026092421` | `locked camera, a single red kite drifting steadily across a clear blue sky, stable daylight, no people, no cuts` |
| `eval_pottery_s2422` | `2026092422` | `locked camera, a blue ceramic vase rotating slowly on a plain display turntable, stable indoor light, no people, no cuts` |

所有八槽在运行前写为 `PENDING`。每槽保存 attempt/status、失败原因、MP4 身份和 SHA-256、181 帧验证、C 和 `C−24`。有效的 OFF 误判或 H1 漏判是本轮观察负例，不得改阈值或重选视频。有限零 lift/零 unit-response 是方法负结果，关联媒体槽明确保留为未生成；OOM、超时、非有限、scheduler/history 错误、视频保存或读回不完整是工程无效，逐槽保留。MULTI 的 T44 失败不能让 T46 独立补写替代；独立的其他臂和第二来源仍按预定名单尝试。无自动换种子、降尺寸、重试或结果驱动的补跑。只有不可恢复的全局 setup 失败才结束执行，仍保留八槽及原因。

同源质量只作诊断，不设事后 gate；报告三条 H1 与 OFF 的 C、差值、MP4 配对质量、全部失败与缺失。即使八槽完整且 MULTI 优于单步，也只支持这两个指定来源、此固定 key/模型/codec 的机制比较；不宣称低 FPR、总体检出率、载荷、视觉质量通过或多步普遍优越。

## 调用和资源边界

计划每来源共享一条正常 50 步 OFF 轨迹及其 T44/T46/T49 实时状态：100 次 Transformer CFG 前向；SINGLE46 续跑 47–49 加 6 次，MULTI 续跑 45–49 加 10 次，SINGLE49 不增生成用 Transformer。另预留每源 3 次**同输入数值恢复核验**（共享 T44/T46 两次及 MULTI 自身 T46 一次），每次 CFG 的 conditional/unconditional 共 2 次前向，核验调用不得推进 scheduler 或代替正式生成调用。因此生成 116 + 核验 6 = **122 次/源、244 次/轮**；生成与核验计数分列。每源 50 个 OFF 正式 scheduler 步，SINGLE49 加 1，SINGLE46 加 4，MULTI 加 6，共 **61 正式步/源**；四个受控事件各做一次显式同史 shadow 和 unit probe，共 4 shadow、4 probe/源，故上限 **69 scheduler 调用/源、138/轮**。每源计划 7 次 VAE decode（OFF/T49 基点复用、三早步基点、三新增终态）、8 次 VAE encode、4 次 MP4 保存/读回/评分；两源合计 14/16/8，0 backward。attempted/completed 分开计；若 A1 选用经哈希验证的 shadow 复用，必须在源码固定并相应重算、审计调用账，不能暗减。

已验证的 T49 路径在 Transformer 全程结束后释放模型才加载 VAE。早步 VAE-lift 必须显式保存当前 latent、CFG embeddings 和完整 scheduler history，在无梯度条件下让 Transformer 与 FP32 VAE 分相驻留（例如模型 CPU/GPU 迁移或重新加载同 revision）；恢复后核验相同输入的 CFG velocity、noise/source 身份、scheduler cursor/history 与未切换路径一致。记录 CUDA 分配/保留峰值、驻留、相位耗时和切换失败。旧 L4 成功仅证明末步顺序加载可行，**不证明**本轮早步反复切换可行；不得把 GPU 型号或 torch 本地 CUDA 后缀写成无关硬门槛，也不得以 `empty_cache()` 当作卸载。

先复用已成功的 group-consistency notebook 的安装、模型、codec、独立 worker 与回执路径，做定向 CPU/fake 的状态恢复、完整 history、各臂方向隔离、预算、计数、失败留槽验证。A1 为隔离分支唯一代码写入者；A2/A3 对同一固定版本独立只读复核，A4 汇总，A5 交付审计。源码 S 先固定发布，再生成只绑定 S 的固定 Run-all notebook N；首代码 cell 为独立两行 Drive mount。用户运行 GPU/Colab；审计任务不代跑。远端候选分支、S/N/notebook 字节与 GitHub default `main` 均需读回核验。
