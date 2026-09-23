# RGB-DCT 接收器可控性：一次固定 T49 VAE-lift 开发实验

状态：**待实施的固定开发协议，不是已执行结果或正式 H0/H1 检出协议。** 目标只判断当前固定钥 RGB-DCT 分数能否通过一个明确、非梯度的 Wan T49 原生写入，在实际保存 MP4 上产生稳定的同源 OFF 增量。用户执行 Colab；实施与本审计不运行模型/GPU。

## 历史差异与选择

旧 public-statistic run08 已完成三臂、两次 backward 和三段 MP4；after47/48 的梯度提案均使终端损失升高、接受更新为零，三个 MP4 字节相同。其控制为“终端 2D 亮度目标的完整剩余 solver→VAE 梯度，按微小 RMS 更新 latent，若损失不降则回退”。仅把目标换成 RGB-DCT 后再对 terminal latent 求梯度仍属同一机制家族，不作本次实验。旧 min-RGB 已证明另一公共亮度统计的直接像素写入在 MP4 上有响应；本次不能把一般的像素统计编码存活重新报告成新发现。

本次方向改为**固定钥的显式 RGB-DCT 载体经 VAE 确定性编码差 lift，再作一次 T49 原生步控制**。不反向传播、不优化损失、不做终端梯度/Transformer 尾程、不扫描方向、符号、幅度或时刻。直接 RGB± 仅用于核对**这个特定新码**在真实内容/相同 MP4 编码链上的方向，不能代替 Wan 写入结果。

## 全部冻结输入与公式

- 基线源码为 RGB-DCT 候选 `c967db03767768871ba0cd3b49208aa308f51b48`，评分规范 SHA-256 `0a8cc13d8f99fa87fa2ad8f6e34b953e091261763877b5e39863f179f317f00f`。Key 原样使用 `b"WanProjection-first-validation-key-v1"`，新接收器 key ID `785b91ae6b23bfc9`。所有 MP4 固定用同一无阈值的 `score_mp4(path,key)`。
- Wan 模型 `Wan-AI/Wan2.1-T2V-1.3B-Diffusers`，revision `0fad780a534b6463e45facd96134c9f345acfa5b`。320×512、181 帧、8 fps、50 原生 scheduler 步、CFG 5，固定 negative prompt：`text, watermark, logo, camera motion, cuts, multiple objects, flicker`。MP4 用项目既有 `encode_rgb`（libx264、CRF18、yuv420p、8 fps），实际读回仍用既有 FFmpeg RGB24 接口。
- 两个预先指定的**新 prompt/seed 组合**，均为开发来源而非独立正式评价：`truck_new_s11`，seed `2026092311`，prompt `locked camera, a single turquoise cargo truck moving slowly along a straight empty road, stable daylight, no people, no cuts`；`dog_weak_new_s12`，seed `2026092312`，prompt `locked camera, a single black dog walking slowly across an empty park lawn, stable cloudy light, no people, no cuts`。后二者内容类别来自已见旧开发病例，后一类在旧 writer 中反应弱；新 seed 不使本轮成为正式独立评价。两源均必须保留，不根据第一源更换第二源。
- 对每源，共同正常轨迹至 T49 前，保存 `z49`、`v49` 与完整 scheduler history。正常 T49 得 OFF terminal `z0`；VAE 无梯度解码为浮点 `X0`。由现有 `key_codes` 得空间 `s_b∈{−1,+1}`、181 帧时间权重 `c_t`；由现有 DCT 差核 `K=DCT(2,1)−DCT(1,2)`，`||K||²=2`。固定 RGB 载体 `A[t,b,i,j]=0.1*c_t*s_b*K[i,j]`，**同量加至 R/G/B 三通道**。`P+=clip(X0+A,0,1)`，`P−=clip(X0−A,0,1)`。不裁剪时亮度 DCT 差的标称变化为 `±0.2*c_t*s_b`；记录 clipping，并在浮点预编码 RGB 上调用同一 `score_rgb`，与 MP4 读回分数并列报告以描述**分数变化**。这里不要求二次 MP4 读取或像素 RMSE，不将分数变化称为视频质量或物理编码误差。0.1 是本轮单个预声明像素幅度，不调参。
- 直接媒体对照物理保存 `OFF=encode_rgb(X0)`、`PIXEL_PLUS=encode_rgb(P+)`、`PIXEL_MINUS=encode_rgb(P−)`；三者同源、同一编码参数，各恰好一次保存读回。score 接口只收到实际 MP4 和 key，不能收到臂名、OFF 分数、writer 状态或真值。全部连续分数先持久化，随后才计算同源差。
- 仅对固定正向载体构造 native writer：`E` 为现有冻结 Wan VAE 的 posterior `.mode()` 并恢复 **normalized latent** 坐标；`u_raw=E(P+)−E(X0)`，禁止改为 `E(P+)−z0`，且 `P+` 在 VAE encode 前**不经过 RGB8 或 MP4**。两次 encode 使用相同 VAE 对象和确定性输入。仅保留 latent 时间索引 `1:45`，索引 0/45 强制为零，避免未计预算的端点更新；记录 raw/masked RMS、峰值和方向身份。
- 在原 `z49,v49,history` 上用既有同 history unit probe 与 `v49−u/sigma49` 原生控制，固定 T49 单步实际响应目标：`target_D_support_rms=0.042943312697648145`，support 为 `latent[:,:,1:45]`。该数值在本协议中**重新明确冻结**，不把历史 R* 的授权或成功结论自动移来；验证 `sigma49>0`、sigma50=0、cursor、unit response、实际响应和 history 不污染。只有正向一条 native 候选；无反号选择或回退 OFF。控制后 VAE decode、按同 codec 保存 `NATIVE_PLUS` MP4，读回打分。

## 固定分母、停机与判别

每源四个物理媒体槽 `OFF/PIXEL_PLUS/PIXEL_MINUS/NATIVE_PLUS`，总分母 **2 新来源、8 MP4/评分槽、1448 帧**。先在任何媒体工作前持久化全部 8 个 `PENDING` 槽；逐槽记录路径、SHA-256、181 帧、解码和分数状态。缺失/非有限/异常/资源失败留原槽，不删分母、换来源、自动重试或降低帧数。预期上限为 2 次 generation、200 次 Transformer、104 次 native scheduler（含 2 次 unit probe）、4 次 VAE decode、4 次 VAE encode、0 backward、8 次 MP4 save/read/score；attempted/completed 分开落盘，VAE cache、内存、耗时另记录。实际流程若因预声明停机而少调用，未运行槽保留原因。

每源在三条直接媒体分数落盘后，先核方向：`S(PIXEL_PLUS)>S(OFF)>S(PIXEL_MINUS)`。不满足或任一不可评分时，该源 `NATIVE_PLUS=NOT_RUN_CHANNEL_NEGATIVE/INVALID`，仍处理另一预定源；这只是**特定载体的通道门**。满足者继续唯一 lift 与 T49。若两次确定性 encode 有效，但 masked `u_raw` 有限且为零，则记 `ZERO_LIFT_DIRECTION`；若合法固定 native49 的 unit probe 有限且响应为零，则记 `ZERO_NATIVE_RESPONSE`。这两种均是本固定构造的**方法负结果**，`NATIVE_PLUS` 保留对应未生成状态，不当作 OOM/工程无效。非有限、shape/history 错误、预算执行失败、OOM/超时或 MP4 无法完整读回才标工程无效；不自动换配置。

当两源 8 槽都完整时，开发可控性判据为每源 `Δ_direct=S(PIXEL_PLUS)−S(OFF)>0` 且 `Δ_native=S(NATIVE_PLUS)−S(OFF)≥0.1*Δ_direct`；负向对照还须低于 OFF。`0.1` 是预声明的最低**归一化接收分数增量比**，不是物理信号/能量留存率，也**不是检测阈值或 FPR 规则**。任何完整来源未满足，或发生前述有限零方向/零响应，即停止该固定 native writer；不能选最好来源、调符号/预算/幅度或改阈值救结果。若两源均满足，只能说新接收器对这一个 T49 写入候选有开发可控性证据；下一阶段才另冻独立 OFF 校准、全新 H0/H1 来源、阈值与失败分母。当前不声称盲二元存在检测、低 FPR、质量、载荷、归因或多步增益。

## 交付与审计

在当前 RGB-DCT 隔离候选分支开发新 runner/config/纯方法载体及必要测试；A1 唯一代码写手，A2/A3 同版复核，A4 综合，A5 交付核验。不能修改远端 `main` 或旧结果。CPU/fake/静态验证与真实 GPU 结果严格分开。源码 S 先发布，再生成只绑定 S 的单一 Run all Colab notebook N；首代码单元仅为 `from google.colab import drive` 与 `drive.mount('/content/drive')` 两行。新时间戳 Drive 输出、setup/source/environment 回执、固定 config/hash、逐源日志、全部媒体/hash、原生响应与连续分数、调用及失败槽均持久化。最终远端核对 S/N、notebook 字节、GitHub `default_branch/main` 与直接 Colab URL 后交付用户执行。
