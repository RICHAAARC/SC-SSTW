# B 线：真实读出驱动的终端方向对照与最小提案

日期：2026-10-10。状态：**研究提案，尚未采纳；未实现新 writer/reader，未执行模型。**

建议先保留 B 当前的局部 DCT 配对能量读出，在同一个保存 Wan 终端点上，比较原桥接方向、五个关键读数的平均梯度方向、以及同时改善这些读数的一阶共同方向。关键读数是 **state gap 和四个 fragment 各自最弱的 signed margin**。先查真正的 decoder 输入梯度能否形成有用的有限扰动，再讨论 CFG/native、自由尾程和学习读出。不是宣称“加 VAE 梯度就能成功”，也不以平均 loss 下降代替原描述条件。

本轮只有来源核验、已存数值复算、CPU 小张量数学检查和本文。以下目标、求解方式、矩阵、预算、停止条件均为待用户集中采纳的建议；本文件不授予执行或方法修改权限。当前两个 notebook、方法/config、已存结果和 main 保持原状。

## 1. 当前证据决定了问题的范围

审计目录：`/home/richar/projects/Video-WM/diagnostics/b-line-terminal-bridge-20261010T031603868487Z-audit/`。本文读取其已修正的 `audit_summary.txt`、`audit_numeric.json`、`payload_signed_means.csv`、`result.json`；没有重新解码、编码或读取新像素。

| 同次 postclip 阶段 | state gap | 正向 payload / 32 | 全部 bit 最弱 margin | 对保存 base 的 RGB RMSE |
|---|---:|---:|---:|---:|
| base | -0.001996996 | 16 | -0.429410042 | 0 |
| RGB candidate | +0.499234359 | 32 | +0.473996079 | 0.007300736 |
| D(E(candidate)) | +0.081882775 | 28 | -0.124352201 | 0.013983870 |
| D(z + E(candidate) - E(base)) | +0.070578525 | 26 | -0.207397484 | 0.010816440 |
| masked reinjection | +0.045948590 | 22 | -0.325318893 | 0.011992959 |
| cap=1 reinjection | -0.001209894 | 16 | -0.427410935 | 0.000155277 |

固定读数为 88 windows、1408 chips；每 chip 512 个样本。704 state evidence、32 项 payload signed means；四片段每 bit 的证据分母为 24/24/20/20。正向 bit 是已知真值诊断，不是盲准确率。11 个已观测视图完整，base/preclip 原本缺失；保留该缺失。

首个显著损失已在 VAE 往返发生。raw→mask 还有 5 位丢失、1 位恢复；posterior→raw 为 3 位丢失、1 位恢复。32→28→26→22→16 不是同一批 bit 的单调包含关系。mask 保留约 92.29% 范数也不代表保留同等水印信号。cap 把 masked L2=159.5750 缩为约 1，保留 0.6267% 幅度；它加重问题，但未 cap 也未达到全部 32 位正向。

新的 CSV 复算显示，base 的四片段最弱项分别为 `(0,4), (1,4), (2,2), (3,6)`，margin 为 `-0.4294100/-0.3746443/-0.3935664/-0.2724950`。cap 后 fragment 2 最弱 margin 反而略降至 `-0.3936457`，虽然总均值和 state gap 都较 base 提高。这是本次数据里“总量改善不能替代最弱项”的直接例子，不需要再造实验。

**clamp 的解释限制：**同次实际依赖为 Diffusers 0.40.0；其 `AutoencoderKLWan._decode`、`tiled_decode` 在返回前均 clamp 到 [-1,1]。项目 `runtime/wan/vae.py` 的 preclip 是 `vae.decode` 返回后再转 [0,1] 的观察点。因此后四次 decode 的 pre/post 相同仅排除**外层** RGB clamp 的新增影响；内部 clamp 贡献未知。本文推荐保留内部 clamp，不能静默绕过。

此次真实调用只有 2E+4D；没有 DiT、native step、backward 或 codec。它定位一个保存 JOINT 终端点的桥接损失，不能归因为原 step25..49 任一具体环节，也不能证明增大 cap 足以修复生成轨迹。

## 2. 论文、官方源码与可迁移边界

### 2.1 Guidance Watermarking

[论文 v2 §4.3–4.5、附录 B.2](https://arxiv.org/html/2509.22126v2) 的实用简化是前向完成余下生成，再用终端梯度近似早期梯度；扩散链的 Jacobian 被 identity 近似。VAE/reader 仍反传，不能称为完整尾链 VJP。附录讨论全反传消融；其中简化编号及梯度替换方向的文字有不一致，本文以 §4.5 的明确公式和实际代码路径解释，不据表中编号推导 Wan 结论。

[官方 README](https://github.com/EnoalG/Guidance-Watermarking-for-Diffusion-Models) 的演示模型是 SD2、FLUX、Sana，使用 VideoSeal 图像 decoder；不是 Wan、8-frame 局部时间身份或分片 payload 的验证。

[watermark.py](https://github.com/EnoalG/Guidance-Watermarking-for-Diffusion-Models/blob/main/guidance-watermarking-for-diffusion-models/watermarking/watermark.py) 的 `ProbGuidanceWatermarker.update_noise`：前向 forecast final latent；对 final latent 的全 1 cotangent 求 `dx`；独立对 reader 求 RGB cotangent，再用 `autograd.functional.vjp(vae_decode, ..., v=dl)` 得 `dy`；最后 `grad=dx*dy`。实际 [FLUX loop](https://github.com/EnoalG/Guidance-Watermarking-for-Diffusion-Models/blob/main/guidance-watermarking-for-diffusion-models/watermarking/modified_diffusers/flux.py) 把 denoiser 放在 no_grad，scheduler step 留在图内。故该乘积不能当作一般 `J_tail^T dy`：只有特定对角/identity 近似下才可这样解释。CPU 2×2 反例给出真 VJP `[23,34]`，乘积 `[20,36]`。不把这个代数反例升级为论文实用 Euler 配置的运行 bug。

代码还有按误码选 `best_latents`、augmentation/PCGrad 和幅度控制。本文不移植这些搜索、选优或超参数；也不把其 detector 的 whitening/FPR 假设移给局部视频读出。后续 Wan 路径若只有前向尾链，图上必须标明“无 tail VJP”。

```mermaid
flowchart LR
  Z[当前 latent] -->|余下生成仅前向| T[forecast terminal]
  T --> D[真实 VAE decode 含内部 clamp]
  D --> R[reader]
  R -->|reader VJP| D
  D -->|VAE 输入 VJP| T
  T -. identity 近似 不是尾链反传 .-> Z
```

### 2.2 GROW 的实际 FFT 路径

[正式论文入口](https://openaccess.thecvf.com/content/CVPR2026/html/Luo_GROW_Watermark_Generation_with_Progressive_Guidance_for_Diffusion_Models_CVPR_2026_paper.html) 与项目 [grow_video_reference_v1.md](grow_video_reference_v1.md) 中的论文 DCT 说明，应与执行路径分开。本轮核验了官方代码，CVF PDF 全文抓取返回 403；不声称在本轮重核了论文全部公式/超参数。

官方 [`utils.py::dct2d_torch`](https://github.com/luopengchen/GROW/blob/main/GROW/grow/utils.py) 实际是 `torch.fft.fft2(x, norm="ortho").real`；[`watermark.py`](https://github.com/luopengchen/GROW/blob/main/GROW/grow/watermark.py) 对 conditional clean 的 masked mean MSE 求局部梯度，映回 conditional noise 后做 CFG。该局部梯度没有 VAE 或 denoiser VJP。

项目 Wan 参考实现保持 FFT-real 和 actual-mask mean scaling，4 channels、32 bits 在 46 latent times 重复，MULTI25..49 与 LAST49 的同一固定源 MP4 都曾 0/32 错。它支持该重复 payload 构造；不证明 22 个局部时间状态、四种分片或任意片段同步。当前 B 是 RGB 8×8 真 DCT 的配对能量统计，不能把 FFT band 或 GROW 学习率直接搬来，仍称“原 B”。历史文档中的摘要/manifest 规则不进入本提案。

## 3. 为什么这不是历史路线换名

| 路线与已读证据 | 实际梯度/控制路径 | 已得到的负证据 | 本提案的可检验区别及仍共有的风险 |
|---|---|---|---|
| RGB-DCT B2，20260926T155237439193Z | 实际 OFF z50→完整 FP32 Wan VAE→30 个 q；`mean relu(-q)^2`；归一到 R*=0.0429433 support RMS，T49 native 写入 | float C22/22，MP4 C21/23；实际 loss 为 OFF 的约 1772/193 倍；丢旧正组 7/6，已无自由尾程 | **真实 VAE 梯度不是新意。**改成当前 B 的 state gap+四个分片最弱 margin；显式检查目标梯度冲突和有限步结果。预算为当前局部 mask L2=1；依然可能过大、局部线性失效，不能保证成功 |
| RAW420 forward-accept，20261002T143942015107Z | decoder+encoder VJP，媒体 STE 的 composite loss，实际 raw420 gap 决定保留 | composite 7.69588e-6→6.54419e-6，真实 gap -5.47056e-7→-7.98984e-7；cap 未激活；原规则拒绝 | M0 不含 encoder/quantization STE；直接真实 D(z) 和当前 RGB q；用五个最弱指标逐项判断，不由复合平均值接管最终判据。后续 codec 仍须实际读回 |
| Structured 网格，20260928T053626644222Z | 三个 VAE-lift 时间基、有限尾程响应代理、离散组合与实测验收 | 六处早期零接受；MULTI=ONLY49，FREE49=OFF | 不再用 OFF 上三个固定方向估计长尾 J；先在 actual terminal 的 reader 输入处求导。零接受不能称“已写入后抹除” |
| Structured 连续六点，20260929T002119133184Z | 同三方向，连续求解后固定 18 个实际候选 | 18/18 执行、0 接受，全部丢旧正组；16 个真实 loss 恶化；预测误差/预测变化 1.194–17.868 | 不把凸求解收敛当真实响应；M0 同样保存预测与实际五指标，以及一次预定中心差分。不能靠再缩步或放宽条件把历史负例改成成功 |
| Local observer gradient 44/46/48 与 B1 terminal46 | 局部 clean proxy 或真实 tail VJP；30-group presence 目标 | local MP4 C16/19，B1 C19/15；SINGLE49 同批 C30/30 | 视频时间局部码/片段目标与 presence 不同；先无 tail 比方向，后单独加入 native/tail，不混淆两条时间轴 |
| 旧二维 observer G1 / public-statistic run08 | G1 中间 VAE observer 后两个 free steps；run08 有受限真实终端反传与 baseline fallback | G1 轨迹残差/质量失败；run08 工程完成但两个候选 loss 上升、接受更新为零；二维 affine 接口另有反例 | 不重用二维亮度目标或宣称其 affine 保证；维持当前 RGB paired-energy reader，优化五个实际读出短板。工程反传完成不等于方法成立 |

本提案真正新增的是**当前局部协议下五目标的共同上升方向与平均方向同半径对照**，而不是首次用梯度、首次保护正组、首次 forward acceptance，或首次让 Wan VAE 可微。共同上升只是一阶局部命题；实际有限步、内部 clamp、最弱 bit/offset 切换、mask 可控性仍可能使方案失败。

历史来源：

- `/home/richar/projects/Video-WM/diagnostics/receiver-first-20260923/rgb_dct_b2_terminal_receiver_result_audit_20260927.md`
- `/home/richar/projects/Video-WM/diagnostics/raw420-forward-accept-real-run-audit-20261002/audit_result.json`
- `/home/richar/projects/Video-WM/diagnostics/method-mechanism-audit-20260928/structured_feedback_review/review.md`
- `/home/richar/projects/Video-WM/diagnostics/six-point-audit-20260929/audit.md`
- `/home/richar/projects/Video-WM/diagnostics/b-line-time-signal-root-cause-20261010/evidence_matrix.json`
- `/home/richar/projects/Video-WM/diagnostics/run08_result_audit/result.json`

## 4. 推荐 M0：同保存终端点、同 reader/mask/半径的方向检验

### 4.1 固定输入、真实梯度链

采用已有 `20261009T132316545817Z/run/joint/terminal_latent.pt`，以及已完成 bridge 的 `tensors/capped.pt` 作桥接对照。两者是实际方法输入；文件不可读记缺失，不补 encode 或重新生成。来源只记录 URL/path。key、message `8001a55a`、22×8-frame 协议、四 RGB ROI、DCT 对、所有固定分母不变。此点本身已含旧 JOINT 控制，因此本轮是**同点增量诊断**，不是从无水印源到成功 watermark 的验证。

设保存 normalized latent 为 z，真实 decoder 为

`Y(z) = clamp01( permute( D_Wan(z * std + mean) ) / 2 + 0.5 )`。

`D_Wan` 包含其内部 clamp[-1,1]，使用真实 FP32 参数、原始 causal cache、完整 46 latent times→181 RGB frames；参数 `requires_grad=False`，但 z 是有梯度的新 leaf。均值/标准差来自实际 VAE config，不能借 SD scaling_factor。不使用 `inference_mode/no_grad` 包住梯度 decode，不把 decoded RGB detach；普通评分仍可无梯度。

现 `decode_normalized_latent` 使用 inference_mode，现 raw reader 把能量变为 Python float，**两者不能原样直接反传**。若采纳，需在独立候选路径增加保留张量图的同数学 reader / decode 入口，先验证与现独立 raw reader 的数值语义；不把现评分器替换掉。raw extraction 只接 RGB、公共协议、key；writer 可用消息目标求梯度，评价仍先保存 raw 再连接真值。

对任一 chip，令 DCT 配对系数为 a_i、b_i：

`A=sum a_i², B=sum b_i², q=(A-B)/(A+B)`。

必须先池化 512 样本能量再作比值，不能平均逐 tile q。正分母上，`dq/da_i=4 a_i B/(A+B)²`、`dq/db_i=-4 b_i A/(A+B)²`。零能量/非有限按真实支持失败保存，不能加未采纳 epsilon 后冒充同一 reader。DCT、RGB luma、pooling、normalized-latent 缩放和 clamp 均在梯度链内；没有 posterior encoder、codec STE、denoiser 或 tail VJP。

### 4.2 五个目标，而非单一平均 loss

保持原定义：`c_o=(1/704) sum state_code[(slot+o)%22,chip] * q_state`；
`G=c_0-max(c_1..c_21)`；`m_fb=sign(message[f,b]) * mean(q_payload over that fragment/bit)`。

取 `s(z) = [G(z), min_b m_0b(z), min_b m_1b(z), min_b m_2b(z), min_b m_3b(z)]`。全局最弱 margin 是后四项的最小值；原描述条件仍是 G>0 且所有 32 个 m>0。本次不会新增盲接收阈值。

五个输入梯度先限制到**当前 mask**：`g_j = M ∇_z s_j(Y(z))`。mask 是全部 16 通道、time[1,45)、四个 8×8 latent ROI，180224 个元素；不把 VAE 感受野等同于 mask 几何支持。五个标量按原量纲使用，不各自做梯度单位化，不调权重、温度、margin 或 ridge。

对照方向为 `g_mean=(g_0+...+g_4)/5`。建议方向解一个五变量凸问题：

`alpha* = argmin_{alpha>=0, sum alpha=1} ||sum_j alpha_j g_j||²`，`g*=sum_j alpha*_j g_j`。

这是输入梯度凸包的最小范数点，只需 5×5 Gram 矩阵的 CPU 数值解。解的 KKT 条件给出 `g_j·g* >= ||g*||²`；非零时沿 g* 对五个当前活跃目标具有共同一阶改善。Gram 只做公共尺度缩放改善数值条件，不逐目标重标定；未来实现应报告简单数值最优性残差，不能借求解器失败换目标。该求解没有调用 Wan 的迭代优化、参数扫描或多次候选选择。

严格限制：min/max 在并列点不可微；本保存数值的四个最弱 bit 和最大错误 offset 唯一，但新 forward 必须按其真实读数确认。任何活跃 min/max 并列都使本固定五梯度方向构造未定义，不采用 autograd 的默认 tie 规则、不任取子梯度，也不自动增加目标/VJP；固定槽位处置见下文。

在有效、唯一活跃项的梯度下，精确 g*=0 只排除当前 mask 内使五项**全部严格一阶上升**的方向，不排除“均不降、至少一项上升”的弱协调方向，也不排除有限步或高阶改善。例如五个梯度 `(1,0),(-1,0),(0,1),(0,1),(0,1)` 的凸包含零，但方向 `(0,1)` 的五个斜率为 `0,0,1,1,1`。本提案仍停止当前 COMMON 构造，不回退平均方向后称共同方向成功；接近零且数值不可分辨只报告不确定，不援引精确零的排除结论。

有限候选为 `delta_mean=g_mean/||g_mean||₂`、`delta_common=g*/||g*||₂`，**各一次、L2=1**。这是待采纳的等半径方向实验，并非声称单位归一化稳定。沿用当前 cap=1 的实际半径以避免把方向改进与增大预算混在一起；真实步长仍可能过大。旧 B2 的全空间时间支持 L2 约 57.6503，不能把旧 R* RMS 与当前局部 L2=1 当同一预算，也不能因当前较小就推定安全。

### 4.3 最小固定矩阵与调用预算（建议，未采纳）

| 行 | 实际输入 | 用途 |
|---|---|---|
| BASE | z | 本次新 decode 的同点基线；历史 base 仅辅助比较 |
| BRIDGE | z + 已保存 capped delta | 既有真实方向，在同次 decoder 下对照 |
| MEAN | z + delta_mean | 五目标简单平均的方向对照 |
| COMMON | z + delta_common | 唯一推荐新方向 |
| FD_MINUS | z - (1/64) delta_common | 一次固定中心差分探针 |
| FD_PLUS | z + (1/64) delta_common | 同一中心差分另一侧 |

4 个主行 + 2 个诊断行，共 6 个固定视图。中心差分半径 1/64 也是待采纳数值，只有这一对，不据它选步长、换方向、缩步重试或替代主行。对每个 s_j 比较 `(s_j(FD_PLUS)-s_j(FD_MINUS))/(2/64)` 与 `g_j·delta_common`；若活跃项切换或 FP32 数值噪声无法分辨，明确报告。此差分核对不保证 L2=1 有限步效果。

完整非退化路径的调用计划与上限建议：**1 次 VAE load、11 次 decode、5 次 decoder VJP、0 encode、0 DiT、0 scheduler、0 codec**。11D=1 基线普通 decode+5 个顺序梯度 decode+5 个其他普通视图 decode。采用顺序 VJP，每次释放该图和暂存后再求下一项；不同时留五份 decoder 图。五个梯度 decode 的 causal chunk 原始 forward 计划 5×46=230；重算名义上另 230 chunk，单列实际 attempted/completed，不把重算偷藏到“5D”。普通 6D 的 chunk forward 另 276；不能用 checkpoint 重算代替一个原定评分 decode。并列、退化、缺失或失败导致实际调用不足时，按真实 attempted/completed 报告，不为凑满预算新增调用。

固定证据分母为 6×88=528 windows、8448 chips、6×55=330 个描述汇总。5 次梯度内部重复读数另标 engineering，不作额外样本。方向未定义的行保留 `direction_status=UNDEFINED`，未产生的观测记 `MISSING` 并附具体原因；不填零、不当负判定、不从分母删除。处置如下：

- 任何活跃 min/max 并列：五梯度方向构造未定义，MEAN、COMMON、FD_MINUS、FD_PLUS 均按上述方式保留，不依赖 autograd 默认 tie 规则选择方向。
- g_mean=0：MEAN 未定义，不做除零归一化。
- g*=0 或数值不可辨：COMMON、FD_MINUS、FD_PLUS 未定义；若 MEAN 独立有效，仍只按原定对照行执行，不作为 COMMON 的替代。
- BASE/BRIDGE 具有独立输入，保留并继续其原定读取；其自身的 OOM、缺文件或失败按实际情况记录。上述退化均不新增 VJP、不改目标或固定分母。

没有新来源、种子、消息、mask、cap 或观察候选择优。

每个主行同时报告 G、四个 fragment min、全部 32 margins、最弱 bit、最大错误 offset、正→非正/非正→正集合；均值/MSE只能作辅助。预测与实际变化并排，保存实际 delta、terminal、float RGB 和 raw q，避免再次只剩摘要而无法定位。

### 4.4 判读与停止条件

以下是待采纳的**方法诊断判读**，不是运行门禁或科学 PASS：

1. 无有效 VJP、非有限、缺失、OOM：工程未完成，不解释为方法负或零信号，不自动重试。
2. 有效梯度且精确 g*=0：只排除当前五目标/当前 mask 下五项全部严格一阶上升，不排除弱协调或有限步/高阶改善。保留结果和未定义槽位，停止当前 COMMON 构造，不扫描目标权重或解锁 mask；数值不可辨则保留不确定状态。
3. 中心差分不支持预测方向：先定位梯度路径、内部饱和、min/max 切换或尺度可分辨性；不能先用更大预算运行整条轨迹。
4. COMMON 的任一五指标实际下降，或丢失原本正向 bit：记录有限步取舍/失败；平均值提高不能覆盖该事实。不得只展示最好的片段。
5. 五指标均不降且至少一项严格提升、无原正 bit 丢失：只支持“该同点有限扰动协调改善”的描述。若 G≤0 或任一 margin≤0，当前原描述条件仍未达到。只有 G>0 且 32/32>0 才可说该保存终端点满足原描述条件，仍不是盲解码、FPR 或轨迹成功。

完整报告各行 RGB RMSE/PSNR、max abs、每帧误差、时间残差 `RMSE((Y'-Y)[t]-(Y'-Y)[t-1])`、ROI 内外能量、局部峰值，以及预定帧 `[1,44,88,132,176]` 的相同显示尺度比较。局部 latent mask 不保证 RGB 局部化。没有已采纳的视觉质量阈值，因此没有自动“质量通过”；即使数值方向有用，也需用户查看质量后才讨论下一阶段。本轮不生成这些模型图像。

## 5. 可微性与资源：已有真实梯度先例，不以无梯度 L4 成功推断

已核官方 [Diffusers v0.40.0 Wan 源码](https://github.com/huggingface/diffusers/blob/v0.40.0/src/diffusers/models/autoencoders/autoencoder_kl_wan.py)：内部 decode 是可微张量操作；causal cache 使用 clone，未 detach，时序依赖跨 chunk 保留；`_supports_gradient_checkpointing=False`。**46 次分块 forward 不表示反向图小，也不能直接调用通用 enable_gradient_checkpointing 就宣称解决。**外层 inference_mode 和 Python float 是当前 B 入口的接线限制，模型本身并非不可微。

与其援引本次无梯度 L4，更合适的先例是实际 B2：181×320×512 FP32 Wan decoder VJP 在 L4 完成，46 forward+46 replay/来源。其独立历史 `runtime/wan/gradient_checkpointing.py::checkpoint_decode` 显式携带 Tensor/None/Rep cache、cursor、first_chunk，并将边界 storage 暂存到磁盘；不能直接复制整个历史 runner 或其身份校验。未来采纳时只复用必要 cache/replay 数学和生命周期，适配当前实际运行接口，保留真实 I/O/内存错误。

同次 B2 JSON 记录的全 worker 最大值：GPU allocated **19.04 GiB**、reserved **20.66 GiB**、主机 RSS **38.59 GiB**、VAE boundary spool peak **64.53 GiB**。这些是历史工作进程记录，包含其阶段与实现，**不是新 M0 的已测峰值或下/上界**。它比“无梯度在 L4 跑过”更相关，但新 paired-energy reader 的 FP64 图和五次 VJP 尚未实测。run08 的 A100 经历也说明 CPU offload、活跃 causal cache 和历史图必须分别计算。

静态存储量：一个 FP32 normalized latent 7.1875 MiB；一份全 RGB FP32 0.3314 GiB，FP64 0.6628 GiB。五个 latent 梯度约 35.94 MiB；这些都不包括 decoder activation/workspace/allocator。实现时只对读取 ROI 的块转 FP64，逐目标释放，避免无意义复制全视频 FP64。五次顺序 VJP 增加时间和暂存 I/O，不应声称总峰值为五倍或自动等于历史值；缓存寿命没有实测前不给精确新峰值。

资源建议是沿用已验证的显式缓存 checkpoint/磁盘边界策略，报告实际峰值、磁盘写读量和重算次数；不得添加 GPU 型号白名单、精确版本门槛、配额硬停或身份门禁。真实磁盘不足、模型 OOM/接口错误按失败保留。没有梯度实跑计时，不能拿本次 2E4D 的速度预测 5 VJP 时长。

## 6. 终端之后怎么走：逐层前向验证，不默认 50 步全反传

M0 的成功只提供终端方向证据。以下均需后续另行采纳，不能因本文存在就开始执行。

**M1：一个 T49、同 history 的 native 映射。**原 JOINT 保存包没有 step49 的 conditional/unconditional、latent 和 scheduler history；不能从 z50 反推。若用户之后授权一次确定前缀，则冻结原 prompt/seed/50步/CFG5，保存完整 history，先算同史无控制终端 n0，再在 n0 求当前五目标 VAE 梯度。不得把 M0 的具体向量直接移植到另一个基点。

Flow 条件 clean 的 delta d 满足 `c'=c-d/sigma`、`v'=u+5(c'-u)=v-5d/sigma`。同 history native map 为 `n=S(z,v,h)`，应比较真实 `S(z,v-5d/sigma,h)`，而不是用 `D(x_c+d)` 替代。若 native map 对当前 v 的实测标量增益为 k，则终端增益为 `gamma=-5k/sigma`；这一系数属于实际 scheduler/history，不能默认 1 或无条件只除 CFG5。

建议 M1 明确采用**terminal L2=1 对照**，通过一次固定 native unit probe 量出方向响应，回映 d 并报告 clean/native 两套范数。这是新预算语义的待决项；若用户选保持旧 clean cap1，则实际 terminal 能量可能不同，必须按不同实验解释。只做 BASE/COMMON 两个实际终端，均保存读数，不进行参数搜索。预期参考账：49 步前缀+当前49 conditional/unconditional共100次 branch DiT forward；49 prefix scheduler+baseline/probe/controlled共52次 scheduler；2普通D+5梯度D=7D、5VJP，0E/codec。冻结真实调用细节后再交付 notebook；无 denoiser/tail backward。

**M2：单个提前点及一段固定自由尾程。**在 M1 后另选明确的一个点，例如48，记录“直接 decode 的中间 native 状态”和“原生 free49 后终端”两层同一 reader/质量，加入同 history OFF。可复用终端 VAE 梯度作待检验方向，但若将其用于早期点，必须标成未验证的 identity/transport 假设；不能把 M0 成功当精确尾链梯度。是否改为真实单步 scheduler/DiT VJP 是新的选择，其额外模型调用、activation/replay另算；本文不预先采纳。原 step25..49 的因果解释仍不能由这个新点替代。

**随后才是多步、RGB8/真实 MP4、盲同步。**每层先分开报告损失位置；不得把最后一步存在当非末步贡献，不要求 MULTI 必须优于 LAST，但须证明实际提前控制及其留下的媒体信号。最终 receiver 只接收视频/key/public protocol，盲路径先落盘再接 truth；局部码、分片、同步、存在判别、质量和误报分别完成。当前已知网格数值不能提前称为这些层的成功。

## 7. 可选学习读出：具体但不推荐本轮同时改

| 选项 | 明确构造 | 新增代价/不能沿用的结论 |
|---|---|---|
| A，推荐 | 保留当前 DCT/32-chip state/4×8 payload；只做 M0 的方向与五目标对照 | 不训练，不换 reader/mask/cap；仍可能无共同可用方向或有限步质量差 |
| B，后备研究方案 | 冻结官方演示使用的 VideoSeal 256-bit **图像** decoder；同22×4窗口，每窗口8个64×64 ROI frame 逐个双线性 resize到256²，输入转[-1,1]；选定 logits0..7 对应该ROI八个 state chips、8..15 对应片段八位，其他240维不用；每窗8帧先对 `tanh(logit)` 平均，再按原24/24/20/20分母汇总 | 该16维选取、tanh、局部 resize、目标映射均是**新协议**，未采纳；不是现DCT数值等价。每个完整读出有704张crop的decoder前向，五梯度方案若顺序执行有5份reader计算及VJP；VAE成本仍在。权重未下载/加载，新显存未知 |

选项 B 的归一化依据是官方 [`VideoSealDetector.preprocess`](https://github.com/EnoalG/Guidance-Watermarking-for-Diffusion-Models/blob/main/guidance-watermarking-for-diffusion-models/detector/detector.py)，而局部 crop/tanh/通道用途是本文提出的方案，不能归为作者已验证设计。若采纳 B，需冻结 weights/config、目标映射与局部支持语义作为方法定义，来源仅路径记录，无完整性门禁。先在同保存终端点验证局部读出本身；预训练图像 decoder 的 whole-image robustness、whitening 和 p-value 不自动适用于局部视频。没有额外训练授权，不新增训练集或 fine-tuning。

保留 DCT 能隔离“posterior difference 方向失配/多目标冲突”而不同时改变接收器和信号。学习读出可能提供更适于图像统计的方向，但也可能只在完整画面上读出全局重复消息；后者不能替代 local time ID/shard。若未来采用 B，需要重新定义接收质量与盲搜索误报，不把两个 reader 的 margin 大小直接排名。

## 8. 本轮已完成的 CPU 检查及集中待决项

检查文件：`/home/richar/projects/Video-WM/diagnostics/b-line-readout-proposal-20261010/cpu_feasibility.py` 和 `cpu_feasibility.json`。只调用现有 DCT 的小型 CPU 运算，不实例化 Wan/VideoSeal，也未实现上述优化 writer。

- 同次 11 视图 payload CSV 的四片段最弱值和表格复算；未重新解释为盲准确率。
- 现 DCT→池化 q 的 CPU autograd 与中心差分误差 `6.64e-12`；解析 q 导数相符。
- 池化能量 q 与 tile-q 均值的反例为 0.207792 vs 0.057646；说明不可偷换目标。
- 两目标凸包小例子验证共同一阶方向条件，简单平均对其中一项斜率为 -4，共同方向对两项均约 +0.819672。只证明数学构造可区别目标冲突，不证明当前 Wan 有这样的方向。
- Jacobian 收缩乘积不等于一般 VJP 的小反例；v0.40.0 内部 clamp、当前 inference_mode、reader float 序列化及 checkpoint 支持标志的静态检查。
- B2 原始 JSON 的历史资源最大值与当前几何字节量复算。

建议用户集中决定三项，而不是逐项运行中追加：

1. 是否采纳**选项 A 的 M0**：五目标凸包方向、MEAN/BRIDGE对照、现mask和等L2=1、固定±1/64差分、上述无扫描判读；或改选学习读出 B（需要先冻结新协议）。
2. 是否接受 **11D+5VJP、0E/DiT/native/codec** 作为一次用户自运行诊断完整非退化路径的调用计划与上限，以及顺序显式缓存 checkpoint 方案。退化或失败按实际调用报告；当前结果不证明其资源峰值。
3. 是否将“同点五指标改善、原描述条件是否达到、用户看过质量”作为讨论 M1 的依据；**本次不提前采纳 M1–M2、多步、codec、盲同步或新的质量阈值**。

无论选择如何，完整保留失败/缺失和原固定分母。源码普通下载、可编辑引用、URL/path即可；不建立 B64、hash、manifest、Git状态或精确版本的准入要求。实施、真实执行、方法结果、发布与 main 集成仍分别报告。
