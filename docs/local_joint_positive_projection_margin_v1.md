# B：固定正裕量终态投影与真实媒体观察包

卡47已采用的用户运行包。固定γ_s=1/√2、γ_p=1/2：状态差法向范数1/√2、载荷法向范数1/2，对应相同单位法向距离1。该选择只基于公开几何，未校准媒体；不保证画质或可读性。已取消0.02终态、累计及每步硬上限，无隐藏替代cap。

输入为卡44同次 `MyDrive/Video-WM/B-Staged-Native-State-V1/fixed-clean-pair-v1/features/PAYLOAD_ONLY/terminal.pt`。两臂为原PAYLOAD_ONLY与一个PROJECTION_MARGIN候选；源不替换。输出为 `MyDrive/Video-WM/B-Positive-Projection-Margin-V1/fixed-clean-pair-v1`。writer在实际宿主上完成一次FP64闭式最小投影，一次物化FP32；保存真实改变量和每条实际margin。FP32目标短差与严格正gap分别报告，不以epsilon判成功，不加幅、补码或选择第二个candidate；合法有限候选照常完成媒体观察。

读端使用B-JPM-v1的key/public方向、状态相关π与mask，不包含OKOK。每个位置仅从22个pilot取唯一最大state，再按估计state解扰局部32位；并列擦除。窗口位置不能充当state。按22位置等权软和/22聚合，精确零或无支持擦除，逐bit保留支持和全部局部输出。盲JSON先保存，评价再接真实state/OKOK。分别保存true-state下704chip几何和inferred-state下704个实际局部解码bit，二者不替换。旧GROW单独保留all46、FFT real/ortho、strict-zero与Counter首遇平票语义；新添首票字段用于复核，没有改变判决。

| 本包预算 | 固定最多调用 |
| --- | ---: |
| 源加载 / 候选构造 / 两key公共码本 | 1 / 1 / 2 |
| VAE加载 / 普通D / posterior-mode E | 1 / 2 / 6 |
| RGB8量化 / MP4写 / probe / 读 | 2 / 2 / 2 / 2 |
| 新局部54方向表 / 旧GROW FFT读 | 16 / 16 |
| DiT、采样、梯度、VJP、候选重试 | 0 |

P/J×terminal、float重编码、RGB8重编码、MP4重编码×correct/wrong key，共16表。每表22state/462竞争约束/704chip/32聚合bit；总76032 ROI分数、19008 pooled、352状态、7392竞争约束、11264chip、512新消息bit及512旧消息bit，全部失败/未进入/擦除保留。每层/每key的P/J对比独立保留state与两种载荷的退步、共同bit BER、增添覆盖和丢失正确位。wrong key不建立总体FPR或存在性阈值。

同一VAE、FP32普通D、posterior.mode E，完整调用前后清cache；优先使用既有Drive VAE缓存，只选择一个位置load一次，无失败换源。GPU推荐而不设型号门禁。普通D一次同时保留preclip及clipped float，不增加D；RGB8只量化一次，MP4直接消费保存的uint8数据。固定libx264、CRF18、yuv420p、8fps；E分别观察float/RGB8/MP4。模块hooks记录实际encoder/decoder forward块数（可观察时），不以名义46代替实测。

质量是相对P的增量，不是相对OFF的总损伤或感知合格判断。终态记录L2/RMS/Linf、支持内外、46时间片及通道0–3。preclip/float/RGB8/MP4各保留181帧、180相邻帧残差及名义ROI×8像素、[1,177)帧mask外统计；该mask不界定VAE真实影响范围。两臂裁剪、量化、codec各自前后变化另表。preclip允许超出[0,1]；显示PSNR使用range=1并注明。没有质量PASS阈值。

产物包括每层blind/evaluation、系数与实际差值、8份latent feature、preclip/float及RGB8/MP4读回张量、两份MP4、质量和调用账。原始RGB张量合计约1.66GiB，另有媒体与小记录。耗时、RSS/CUDA峰值与失败照实记录，未继承卡46stdlib筛查的60秒/1GiB限制。新包只证明固定终态构造的clean通道；不证明原生多步联合写入、未知起点/编辑恢复、完整状态空间同步或科学PASS。

用户使用同步notebook直接Run all；首代码单元只mount Drive。首次加载可用源码，按实际import错误准备依赖，安全解压，不建立源码摘要/manifest/Git/精确版本门禁。发现任何旧科学记录或残留就仅展示已有JSON及媒体，跳过下载、安装和运行；失败/marker缺失/中断不自动继续。开发CPU合成/静态检查不是VAE/媒体实跑。当前包先在本地审查，远端URL在获得新发布授权并发布后才可首次使用。

可从可编辑无.git源码目录独立运行：

```bash
python -m experiments.wan_state_clock.local_joint_positive_projection_margin_v1_run
```

`--config`仅用于指定可用输入/输出/模型位置；正裕量、key、消息、几何、双臂和预算固定于本包，无试参模式。GPU/Colab由用户运行，实施阶段不调用模型、不push、不合并main。
