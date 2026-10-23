# LFG公开公式 Post-hoc K0 阶段1参考包

这是四个VIEWED pre-tag P0来源、全181帧的普通固定消息通道诊断。用户授权本地隔离实现和必要CPU/静态/stub检查；真实模型/GPU/Colab由用户运行，当前新包没有发布/commit/push/merge授权。没有获得作者源码，不能称LFG-WM等价复现、用户432位增容、time-ID/局部化或同步成功。A/default/原用户净32位与历史QIM/RANDOM/BLOCK2源码不改。

论文来源：本地ssrn-7467115.pdf物理18-20、35-36、43、48页，相关定义/配置见16-17、23-24、40-42。论文N为coded bits512，而M为LLL coefficient总数。公式W=sum((2*b_i-1)*c_i)/sqrt512；LLL'=LLL+alpha*sigma_c*W；r_i=mean(LLL*c_i)，r>0为1、否则0。p23的BitAcc默认在BCH-decoded序列上计算，不能直接当raw512准确率或完整消息成功率；BCH失败计分细节未详，本包分列raw512、BCH状态、decoded432位与整432 exact。

固定alpha=.15、K=0；normalized Wan latent[1,16,46,40,64]经正交Haar T/H/W（low=(even+odd)/sqrt2, high=(even-odd)/sqrt2），LLL[16,23,20,32]、M235520。偶数轴，无padding/crop/reflect；所有16通道和旧guards均参与。sigma_c在原P0 LLL各channel14720个系数上作population std/ddof0，不是Wan latents_std，zero sigma不加floor。其它7band理想保持，实际FP32后的微小漂移另做Haar保存，不强行投影回补。writer/reader固定在同一normalized坐标，reader不接P0、写端sigma或消息，不以writer的per-channel affine关系宣称global correlation仿射不变。

一次FP64变换/叠加/逆变换后单次FP32物化，实际latent energy/L2/RMS/Linf、通道/guards/原payload通道变更及FP32误差都保存。每construct有initial Haar1、FP32 receipt Haar1、inverse1，四源full Haar共8（4+4），不漏记；读取24份LLL各共享给两key，不反复变换。原P0不能in-place更改。

PN固定：新domain LFG-WM-Posthoc-Reference-V1，以key UTF8作HMAC-SHA256，label为domain|C16|T23|H20|W32|i{i}，每bit seed喂SHAKE256，按MSB-first展开±1并按C,T',H',W'行优先reshape；每key512行、仅缓存packed数据。无DC平衡/方向筛选/重抽，无sqrtM归一化或W单位化。key00/key01沿用watermark/watermark-wrong文本，但不复用QIM book/参数。

BCH补定：GF2^10 primitive polynomial0x409，母码primitive narrow-sense[1023,943,t8]，roots alpha^1..16、generator degree80；标准缩短511个前导0信息位至[512,432]，无padding/extension/puncture；54 data bytes+10 ECC bytes、全MSB-first。生产是独立pure-stdlib Berlekamp-Massey/Chien实现，bchlib仅受控CPU oracle，不是生产依赖。普通54-byte固定ASCII测试消息单独准备，绝不是替换或增加用户净载荷；BCH decoder只吃512 received bits，不用已知ASCII/真消息筛码或救码。>8可误纠；BCH_SUCCESS与真实整432恢复/认证分开，失败432位为未决，不把raw前432当fallback。

BCH通用构造参考：[galois作者文档](https://mhostetter.github.io/galois/latest/api/galois.BCH/)；独立oracle接口：[python-bchlib作者源码](https://github.com/jkent/python-bchlib)。这些仅支持编码选择/工程核验，不补成论文作者未公开的具体参数。

固定四源b101-s01/s03/s02/s04，各P0/MSG两臂同runtime新D181→同一次RGB8量化。clean-E用未压缩RGB8，MP4-E用同RGB8送H264 CRF23/yuv420p/8fps encode/probe/decode后的readback；都只/255一次，复用原Wan posterior-mode/mean-std helper。无float-clean E、历史媒体替代、新DiT/梯度/训练、E41/phase/path。181/320x512不同于论文21/832x480，M与处理增益不同，不能挪用论文结果。

规划预算VAE1/D8/codec各8/E18116（RGB8-clean8+MP48）/FFT16/双钥原payload32读/Counter12288，预计decoder368/encoder736 chunks。消息native/clean/mp4共48双钥表、24576 raw位/20736 info位，MSG/key00正例12个消息；原32位两个media层都读，固定256group/2816cell配对，不因BCH失败跳过。质量全局四层16组、2896逐帧/2880相邻残差，clipping8；不沿用旧ROI“支持外不变”断言或旧局部mask。

python -m experiments.wan_state_clock.b_lfg_posthoc_reference_v1_run 是完整固定CLI，仅允许source/output定位。attempt独占创建、每call STARTED先落盘，失败/中断消费；已有科学残留即只读saved JSON，不续算、重试、fallback或生成新host。全部raw scores/code、BCH消息与原payload保存并回读后，才打开新固定message truth和旧protocol.config.writer.bits/quality后评。中间保存失败不能以预写占位冒充盲完成；真实上游未进入槽保留未决。BCH有效负结果不冒充工程失败。

Notebook首codecell只有Drive mount，随后直接GitHub clone可用代码，不内嵌SOURCE_TEXT/B64或hash/manifest/版本/Git-clean门禁，不要求手工ZIP。ffmpeg/ffprobe缺少才处理依赖。新源码尚未发布，当前本地候选并非已可运行的远程发布版；CPU验证与真实运行计数分别记录，无科学PASS。
