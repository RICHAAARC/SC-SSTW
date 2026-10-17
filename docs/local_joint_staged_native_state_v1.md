# B 分阶段生成期时间状态写入 V1

2026-10-18，卡42实施说明。当前为待审用户运行包；本地只做静态、CPU合成与桩验证。真实Wan/Colab由用户运行。

## 方法身份与既有证据

保留成功的多步载荷生成，在采样结束、首次普通VAE解码前的原生normalized terminal latent上，单独加位置相关时间状态，然后普通解码与H.264保存。这是**分阶段生成期写入**，不是载荷与状态在同一多步轨迹共同形成，不修改VAE内部特征、不训练。首包复用一个缓存原生终端点；该点过去已解码不妨碍本次配对控制，但本次不是新在线完整生成验证。

唯一源为GROW MULTI，run `20260929T124706736859Z`，`fixed_reference/MULTI/terminal.pt`。生成调用链是 `experiments/wan_state_clock/grow_video_reference_run.py` 调用 `runtime/wan/grow_video_reference.py::run_trajectory`：50步完成、MULTI控制25–49、cursor50后保存原生terminal，再进入普通媒体。形状 `[1,16,46,40,64]`，181帧320×512，8fps。旧审计同源MULTI在terminal/float/RGB8/MP4均0/32错误；LAST也成功，故没有多步必要性证据。旧MULTI相对OFF的MP4 RMSE为0.0240673534、PSNR32.3714，观察过可见纹理差异，不能称不可感知。错key的12/32错误不构成FPR。

用户运行时只读上述固定Drive点。缺失、不可读或格式错误记失败，不换点、不重采样、不用RGB经E重造terminal。源码/模型位置仅为定位记录，无摘要、版本、manifest或Git状态门禁。

保留旧B负结果：terminal bridge同一JOINT点为RGB候选32/32→posterior28/32→raw reinject26/32→masked22/32→cap16/32；VAE往返是首个明显损失，cap非唯一原因。其11观察视图保留，每视图88 windows/1408 chips，base/preclip缺失仍缺失。同点双臂同S的MEAN20/32、ALLBIT21/32，全部32位与22状态行、下降位及普通4端点352窗口/5632 chips/220 metrics保留；历史26/32在另一点/预算。P48原MP80连续证书已通过，但其FP32候选被拒绝；随后两次600秒联合FP32构造/舍入尝试均未完成，新native尝试固定7234、visited7241、buffer7，RESOURCE_UNRESOLVED。连续证书与可执行FP32构造不能混称；原36+0及C缺失不变。本包不继续P48、旧GPU或梯度计算，不把这些点当作成功payload源。

## 单一固定构造

- 输入为normalized coordinate `z=(raw-mean)/std` 的FP32原生terminal；直接扰动只在通道4–15，载荷通道0–3不变。普通VAE混合通道后仍可能损伤载荷，必须实读。
- 时间支持 `[1,45)`，22个不重叠双时刻单元 `[1+2s,3+2s)`，s=0..21。空间四ROI `(8:16,12:20)`、`(8:16,44:52)`、`(28:36,12:20)`、`(28:36,44:52)`。每单元1536维，88单元合计135168坐标。
- 每ROI由key派生固定PRNG，取512维非DC Walsh行中的22行，重复3次并作统一坐标置换、逐行符号，得到1536维零均值、单位L2正交码。公开候选全集是0..21；各位置只写相应状态码，四ROI冗余。
- 固定加法：每单元增量 `a*c_s/sqrt(88)`，**a=0.0199**，为原0.02上限预留固定FP32舍入余量；不按内容或结果调节。上限是整段新增normalized-latent实际FP32差分的全局L2≤0.02，不是RMS、每块上限，也不等同旧P48两球/轨迹预算。记录名义及实际L2、RMS、最大绝对差、支持外差分和载荷通道差分。超限即失败，不缩放重试。
- 不清除host投影、不做梯度、投影、候选挑选或补偿。host背景可能淹没这个小幅码；两臂实读保留该失败可能，不增幅。

## Wan相位与盲读

已读本地 `diffusers/models/autoencoders/autoencoder_kl_wan.py` 的 `_encode`、`_decode`、`WanCausalConv3d` 和 `WanResample`。首latent对应首帧；后续逐latent解码，两个时间×2层生成4帧。名义相位：j≥1对应帧 `[4j-3,4j]`，双时刻s对应 `[8s+1,8s+8]`，末单元169–176。帧0及177–180不在直接状态支持。空间stride8给出每ROI名义64×64px。**这是输出分组，不是独立感受野**：因果卷积/缓存向后传播，空间卷积和注意力扩散；未测完整有效感受野，不承诺局部无串扰或4帧独立支持。编码从首帧及后续四帧组作posterior-mode，并恢复normalized coordinate。普通decode可能内置clamp，观测层不称pre-clamp内部特征。

这是完整181帧clean、已知公开网格的最小诊断，无裁剪相位搜索或状态空间路径恢复。接收器只接收观测feature、key、公开固定几何；对每个接收双时刻及每ROI计算22个码的内积，以四ROI平均分盲选。每块先减自身均值（零均值码下不改变理想内积）。唯一有限最高分产生label；精确并列为UNRESOLVED。没有水印存在阈值，低分仍可被迫唯一argmax，因此标签不是存在性认证。输入非有限/缺失保留失败。接收索引只切接收网格，不用其编号选择或偏置候选。

raw分数、全候选及盲label先保存，之后评价函数才取得目标状态0..21与消息OKOK。报告逐位置真分数、最大错状态分数、差值、真状态rank、盲标签及时间位置误差，全部22行保留；正确/错误key各读一次。载荷保持用原GROW全46时刻FFT-real/ortho、strict-zero和Counter首次并列规则，32位均保留。错key为固定对照，不能校准FPR。无message驱动重选或路径oracle。

状态与载荷配对分别记录有效性：两臂payload有效就报告32位丢失/新增，不因state失败遮蔽；state有效就报告22状态的配对变化，不因payload失败遮蔽。各自未知槽及分母保留，单类可计分时总状态为PARTIAL；原raw/evaluation不删除。

## 固定预算与输出

一源两臂PAYLOAD_ONLY、PAYLOAD_STATE；每臂terminal、float重编码、MP4重编码三观察层。完整预算：

| 项目 | 固定最多调用 |
|---|---:|
| terminal文件加载 / 构造 | 1 / 1 |
| VAE加载 | 1 |
| 普通decode | 2（名义92 latent chunks） |
| posterior-mode encode | 4（名义184 causal组；无replay） |
| H.264保存 / ffprobe / MP4 RGB24 decode | 2 / 2 / 2 |
| 状态读出 | 12（22×22×4，共23232个ROI分数；池化5808） |
| 载荷读出 / FFT2批调用 | 12 / 12（各32位，共384位；每批736个40×64平面） |
| float及MP4配对质量 | 2（各181帧/180相邻差；复用已有RGB） |
| DiT、native采样、scheduler、梯度decode、VJP、重试 | 0 |

真实计数在调用之前落盘，返回/失败后分别记录；未进入保留NOT_ENTERED，能捕获的中断记INTERRUPTED，硬中止可能留下STARTED未决记录，不自动重复。Notebook在源码/依赖前定义固定OUTPUT并检查attempt/result/checkpoint/ledger/evaluation及其.tmp，以及blind/features/media/quality中的科学产物；有任一项就只读展示，完全跳过源码下载、解压、依赖与runner。Runtime也在新建attempt/Journal之前作相同检查，marker缺失或小JSON不可读不使预算恢复。仅setup.log、源码ZIP或空目录不算科学开始。有未完成ledger时不猜测进程是否仍在运行，不继续物理调用。此首包不是断点补跑器；模型/媒体实际失败仍保留12个state和12个payload槽位及质量槽位。

输出 `result.json`、`ledger.json`、先于评价保存的 `blind/*.json`、`evaluation.json`、`quality/*.json`、两臂MP4、两个native terminal与四个reencoded feature，便于同次审计。调用耗时、CUDA峰值（若可用）、失败及固定分母完整保存。模型使用现有Wan2.1 VAE普通FP32路径，冻结、无tiling替代、无GPU型号门禁。

首次环境准备复制A/P2已成功使用的实际import probe及定点修复文本到B builder，没有A工作树运行或构建依赖。只导入AutoencoderKLWan（另检查实际需要的tensor/模型支持包）；明确缺包才安装，只在实际日志同时出现FqnToConfig导入错误和torchao.quantization时移除本实验不用的可选torchao，重检一次该修复，未知错误直接报告。ffmpeg/ffprobe实际缺失才apt准备，不比较版本。metadata建议GPU但无型号硬停。这些准备是imports/安装，不执行模型或科学codec往返。

唯一VAE load之前优先定位同Drive `Paper-Results-V1-Cache/models/Wan2.1-T2V-1.3B-Diffusers` 的available、根、已有子目录（依次检查普通vae配置和权重文件是否可用定位）。也允许显式本地模型目录。只选一个位置并load一次，失败不换缓存；没有可定位缓存时使用原HF模型ID，可能下载VAE配置/权重，其准备与加载耗时包含在该一次load中，不能声称零下载。实际load校验可读性和模型调用；缓存位置、版本不作强绑定门禁。

## 判读及后续边界

首先核对PAYLOAD_STATE在float和MP4的22位置全表及payload相对PAYLOAD_ONLY退步、真/错key对照、两臂质量。仅描述22/22是否唯一命中，不设置科学PASS/存在阈值、不自动进入下一阶段。干净媒体真实可读之后才另提裁剪/删除/重复及分段用途。本次单源成败不能证明泛化、新颖性或完整开题目标。

旧archive `SyncTube_paper.zip/paper.md` 为逐帧VAE局部tubelet+候选同步证据融合的机制参考，旧检测率不是当前逐块时间路径恢复；旧sample_id、原长、攻击参数不得输入当前接收器。相关工作包括 [SPDMark](https://arxiv.org/abs/2512.12090) 的帧位置相关消息及二分匹配、[SIGMark](https://arxiv.org/abs/2603.02882) 针对因果3D VAE的Segment Group-Ordering。这里只核对原文摘要，尚无公平实跑比较，不能以移到terminal本身宣称新颖。

历史依据：根目录 `diagnostics/grow-video-reference-real-run-audit-20260929/audit.md`、`diagnostics/b-line-terminal-bridge-20261010T031603868487Z-audit/audit_summary.txt`、`diagnostics/b-line-same-point-dual-result-20261015/REPORT.md`、`diagnostics/b-line-p48-native-real-cpu-result-20261017/REPORT.md`；授权卡42与根开题文件。根路线A结果不改。
