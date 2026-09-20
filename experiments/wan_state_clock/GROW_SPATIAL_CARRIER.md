# 原空间频域载体的完整媒体配对

范围：独立dev/grow-spatial-carrier工作树，基底b6e470c（paired源码51b774a及其notebook发布），不改旧枝。新入口补齐旧空间载体MULTI/LAST完整媒体，不称新载体、不否定整条GROW、不称官方忠实复现。

## 官方与既有方法对照

官方仓库 https://github.com/luopengchen/GROW ，读取固定commit `6aa69a9c5d4a9e75df457fcca8dfc71b64a6b870`，`GROW/grow/watermark.py`：254–256 UNet no_grad；263–264 conditional pred_x0及requires_grad；275 masked F.mse_loss默认mean；278求导对象pred_x0；279 clean梯度更新；282–287反推conditional epsilon后做CFG。没有UNet Jacobian反向。代码未显式detach pred_x0、外层也未全no_grad，不能把它的图生命周期说成与本适配完全相同。论文官方页：https://openaccess.thecvf.com/content/CVPR2026/html/Luo_GROW_Watermark_Generation_with_Progressive_Guidance_for_Diffusion_Models_CVPR_2026_paper.html 。官方PDF已通过PowerShell下载并核对§4.2.1–4.2.3/Algorithm1：keyed midfrequency重复编码、conditional clean梯度、反推conditional噪声后CFG、VAE encode后的sign+majority。Eq5写masked squared L2且正文称MSE；具体归一化以官方代码的mean为准，不能视公式未写分母为half-sum实现。固定Wan30..49按本实验index定义，不直接等同论文时间索引。

本适配保留历史Wan/Flow/native UniPC：先完成conditional/unconditional CFG，再FP32 v；clean=z−sigma*v成为detach叶子。目标L=.5Σ(DCT_spatial(clean)_mask−.5payload)²，g=grad_clean L，u=−.1g，v'=v−u/sigma。因此z−sigma*v'=clean+u。不涉及Transformer或未来轨迹反向。native actualD由同状态/完整history的OFF shadow与live step差测得，中间actualD不能等同u。末步sigma>0、nextsigma0、原native阶数1，记录末步实际输出与controlled-clean误差。CPU测解析梯度、mask外零、局部loss×.81、post-CFG位置、native真实预算及历史MULTI逐tensor等价。

官方与适配还在SD/DDIM对Wan3DVAE/UniPC、mask/通道、损失mean对half-sum、eta及归一化、payload/key组织、tie规则上不同；官方仓库README示例eta200，而论文§5.2写eta100；都不能与本half-sum的.1直接对照。main/tube_state/grow_frequency.py保持未改：channel0、每时间片whole40×64正交DCT-II、2<=u,v<=9、4频repeat×46时间=184票。无时间差分、无奇偶配对92票、无时序同步或攻击结论。184票高度相关，不是184独立样本。

## 历史实际证据与复用边界

G盘原始 `G:/我的云端硬盘/Video-WM/GROWVideoFrequency/grow_video_frequency_20260918T012616870889Z/result.json` SHA256 `9287ef9036739bd3b540a22625b8f3cb051f0d8cfef078694d0bfbd7da0411ea`：sourcee04f48e，10..29原空间12video/48layer完整，四层完整消息均0/8。`G:/我的云端硬盘/Video-WM/GROWLateControl/grow_late_control_20260918T080310904782Z/result.json` SHA256 `019d3a453829e18158de0b0923abaad72cc62f9287af03dad687b29ec8743e43`：source482222f，30..49终态3/8，未自动媒体。对照档案在旧GROW工作树evidence/raw同run；此处只引用不复制大文件、不新评分。时间差分paired的MP4 MULTI4/8、LAST7/8不能冒充空间载体结果。

旧late已保存initial/prompt/negative、OFF/A/B终态及last49前态/clean/after、每marked20步actualD预算。缺的是可重建OFF49完整UniPC history（model_outputs/last_sample等）和直接v49快照；只有config/timesteps/sigmas不足。并非“无原输入或预算”，modelrevision缺失也不是唯一理由。要用旧MULTI需另加OFF重播及逐tensor等价复用验证，本轮最小方案不增加该分支，因此固定新20全链。同配置MULTI终态有意重测，并非新贡献。

## 冻结协议

原4dev（2内容×2seed），每case OFF/MULTI_A/B/LAST_A/B。MSE/eta.1/target.5/key/A/B/空间reader不变；MULTI30..49、LAST49。E_MULTI=Σ_t mean((native_next_controlled−native_next_OFF_samehistory)²)，全latent分母；LAST同case/message以唯一unit probe测Eunit，scale=sqrt(E_MULTI/Eunit)，只一次live更新，记录实际E、相对/绝对误差。不扫参、不重试调scale、不以终态/接收器选预算。零响应按原matching_scale明确失败/零目标语义，保留固定槽。

这个预算依赖同message MULTI轨迹，是配对诊断oracle，不是统一在线强度；预算相同不代表peak、终态净差或感知质量相同。记录u/dv/实际D的peak、sum RMS、sum RMS²、末步损失和native响应，净终态OFF差单列。

全轮20video/80layer，MULTI和LAST各8marked/128bit。每视频terminal、float RGB VAE roundtrip、uint8 VAE roundtrip、真实MP4保存回读VAE roundtrip；失败不gate其他媒体。无真值reader只收latent/book，truth后处理；hard与rawsoft固定并列，不择优。输出184票/bit、含擦除BER、完整payload、A/B二候选连续排名/gap。A/B归因、16bit恢复、存在检测彼此不同；无校准存在阈值，无FPR声明。OFF单列。PSNR只相对OFF残差，感知质量未测，须人工看片。

成本每case500TF/250native/42local-grad/42OFF-shadow/2unit-probe；媒体5decode/15encode/5MP4save/read。全轮2000/1000/168/168/8和20/60/20/20。生成与媒体独立子进程释放Transformer；原失败分母/时间46槽保持。固定RunAll notebook首cell独立Drive挂载，源码发布后绑定。新输出GROWSpatialCarrier，旧树/历史结果不写。仅CPU/fake/static，本地不实模/GPU/新实验评分。
