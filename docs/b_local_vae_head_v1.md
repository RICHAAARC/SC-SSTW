# 局部 VAE 条件写入头固定候选 V1

本候选复用已跑通的 Wan/VideoSeal 环境和 RGB8→H264 CRF18/yuv420p 媒体链。本次 B 线发布包含14个候选文件，已完成必要 CPU/static 检查、A2/A3同版针对性独审和A5发布前核对。直接 [Colab Run all](https://colab.research.google.com/github/RICHAAARC/SC-SSTW/blob/dev/local-joint-state-payload-v1/notebooks/b_local_vae_head_v1.ipynb)，默认从B分支下载可用源码，无需手动上传ZIP。真实入口为 `notebooks/b_local_vae_head_v1.ipynb`，builder 为 `scripts/build_b_local_vae_head_v1_notebook.py`。本候选真实训练与验证由用户运行，当前没有真实效果结果。历史 B 负结果保持原义。

## 方法与固定实现

冻结 Wan2.1 T2V1.3B 的 DiT/VAE。181帧、320×512、8fps、50步、CFG5。小头挂在原 decoder.conv_out 后、原最终 clamp 前，输入为 detach 的 norm_out+SiLU 特征；原 conv_out 与因果缓存不改。每次完整 decode 重置帧计数，首1帧、随后45×4帧。C读取实际 in_channels。

三层3×3 Conv2d C→64→64→3，两个隐藏层由96位 FiLM 调制，SiLU，最后卷积零初始化。L为 y32:288,x0:256；R为 y32:288,x256:512。L边界16像素 raised cosine，RGB残差8/255×tanh；转换原[-1,1]坐标时乘2。原生编码前 outside-L 严格相等；编码后泄漏单列。零头等于OFF不是额外真实decode预算：用CPU接口检查和训练step0的缓存合成比较。

Pilot为[32,8,12]，基1,x0,x1,x2,x3,x4,x0x1⊕x2x3,x0x2⊕x1x4。label系数LSB优先，position的x0为LSB。key置换与XOR；0..180合法。信息32位经BCH(63,36,d≥11)前导4零缩短到(59,32)，27校验位、5 padding，t=5。源块k=floor(t/8)控制payload置换与mask。同视频同消息，不同源块码样不同。VideoSeal原生256位里前32为pilot、后64为payload，剩160为固定公开交替填充；head/BCE仅96位。ECC不提供来源认证、256位容量或逐位物理支持证明。

TRAIN只有既存n01–n04 OFF/full，共724帧。4次输入probe/decode，4E181、4D181。缓存L特征FP16和完整原生pre-clamp RGB坐标输出FP32，仅/content。C=96时约9.11GB特征+1.42GBpreclip；不保存模型/特征/terminal到Drive。teacher接clamped base，先clip其残差±8/255并施加同窗，再与preclip相加clamp；蒸馏实际post-clamp残差。

一次3000 Adam步，lr1e-4、batch8，首20步资源观察包含在3000内；只取最终小头。每样本独立训练key/message，排除eval key。固定随机源选择和start0..173；连续8帧可跨源块，条件始终用实际t，覆盖最后部分块。teacher使用image batch而非video单消息传播。冻结VideoSeal detector参数但保留输入梯度。损失为BCE96 + 残差MSE/(teacher残差能量+1e-6)+100×ROI MSE。偶数零起始step clean；奇数step两次完整320×512真实H264往返，ST仅是梯度代理，不宣称codec梯度。像素与蒸馏损失取clean postclamp，BCE取clean/ST输入。

训练缓存来自H264后再编码，而验证来自native DiT terminal特征，存在域差异；本次实验正要测量这条链，不以工程检查保证其效果。

## 同步与读取

只向纯接收器提供L的logits、key、公开协议。e=Σ tanh(logit/2)×signed pilot。151帧完整域9275条：31无跳、3450删除(+9步长)、5794重复(−7步长)。保留score≥best−8的全部路径；所有保留路径对源帧一致且e≥16才支持。源帧去重取最早合格received帧。存在条件：best均值≥8，≥24不同源帧，span≥64。只有全保留路径为edit时，逐路径要求jump两侧各≥8支持；mixed no-edit/edit保留不确定类型，不单凭这一点拒整片。

提取时归档完整256 logits，并分别存pilot32与payload64。计划阶段只打开pilot文件；先保存本phase所有main和I计划并回读，再打开payload文件。按支持源帧块undo后，59位等权soft均值；任何精确零整消息ERASURE，但其余非零信息位仍保留为raw观测，按共同索引计算BER。非有限为TECHNICAL_UNREAD；至多一次t5 bounded-distance BCH，验证再编码与距离≤5。单列DIRECT_OBSERVED与ECC_RECONSTRUCTED，以及所有引用的真实时间对应。I为逐帧唯一argmax、相同阈值/去重/存在/ECC规则，无路径侧支持条件。

全blind后才用真值做O/W_GROUP/U：共30正例×3=90reads/2880槽，三个读法用相同true-source-first去重。O用真k，W用(k+1)%23，U用floor(received/8)。不重提取、不额外GPU，错BCH字不是认证成功。保留逐帧256 logits、逐8帧窗口均值与原始消息位BER。

## 固定分母与停止

CAL=m01/m02，DEV=m03/m04/v02/v04，沿用既定prompt/seed和a。每源一条OFF轨迹，同terminal解码OFF与WM。先full编码，再CROP/左右DELETE8/左右REPEAT8五个151帧fixture编码，准确两代媒体。五种fixture同源任务不可当独立来源。

| 项目 | 固定计划 |
|---|---:|
| 主读法 | 150任务 / 4800信息槽 |
| CAL | 10正例 / 320槽；40null / 1280槽 |
| DEV | 20正例 / 640槽；80null / 2560槽 |
| 独立I | 另150任务 / 4800槽 |
| 后评O/W/U | 另90reads / 2880槽 |
| 质量 | 42对 / 6702帧 / 6660相邻对 |
| E181 / D181 | 4 / 16 |
| codec encode / decode / probe | 3072 / 3076 / 76 |
| 其中训练encode / decode | 3000 / 3000，24000编码帧次 |
| 其中验证encode / decode / probe | 72 / 72 / 72 |
| native轨迹 / steps / nominal DiT forward | 6 / 300 / 600 |
| 验证ROI frame inputs | 18120（batch32时600网络调用） |

每source/fixture四null：OFF/L/key00、OFF/L/key01、WM/L/key01、WM/R/key00。CAL必须10/10全消息正确且40null零存在/输出才开始DEV；否则保留DEV全部NOT_STARTED。DEV需20/20全消息正确、全部使用引用准确、80null零存在/输出并满足24/64支持。未读null的零判据为未知，不能计为满足。主method判据与O writer诊断分开：O使用全部真源去重帧，可与主读法高pilot支持子集有不同结果；不将O强加到主方法总门槛。按既定停止边界，O不能恢复时不扩展writer候选。只有主方法、完整观测/预算、质量记录及writer诊断支持时，才提出独立新源确认；不自动启动。没有感知质量PASS门槛，没有总体FPR或来源认证结论。

首技术失败停止后续物理工作，固定分母和失败/擦除/未启动仍保存。科学工作前持久化390个主/I/OWU读取状态及42质量记录；逻辑操作前后更新状态。SIGTERM尽量正常封口；即使SIGKILL来不及写最终汇总，下一次仍只读恢复全部既存状态，不重新评分、读payload或BCH。没有后评的错误数和时间归属保持未知。一次尝试在启动前普通Drive收据读回后消费；中断、OOM、错误不会触发重试。收据不是并发原子锁，请勿同时启动两个runtime。后续Run all只读保存结果。

## 存储与入口

结果：`/content/drive/MyDrive/Video-WM/B-Local-VAE-Head-V1/fixed-six-source-v1`。运行中本地写入并镜像到Drive；结束后通过Drive API上传并实际读取结果。云端保存状态与方法状态分开，保留具体读取/上传错误。仅必要72媒体、小头、原始logits和JSON证据持久保存。

完整CLI：`python -m experiments.wan_state_clock.b_local_vae_head_v1_run`，默认使用可用ADC凭据和相同固定配置。Notebook与CLI共用官方VideoSeal源码准备和实际依赖探针；CLI也在收据及模型调用前发现缺依赖。Notebook另外沿用已跑通的按实际缺失安装与可选torchao兼容修复。路径在入口解析成绝对路径，避免模型临时cwd影响。CLI不依赖.git或Notebook隐藏方法变量。Notebook声明GPU偏好，无GPU型号硬门禁；无摘要/版本相等或Git-clean门禁。

实施检查仅CPU合成：独立bchlib校验BCH，pilot最小距离，路径与payload、局部hook与输入梯度、固定停止及全分母、Notebook AST/nbformat/builder同步。真实VAE/模型/codec接口与资源是否跑通，仍需用户Run all确认。
