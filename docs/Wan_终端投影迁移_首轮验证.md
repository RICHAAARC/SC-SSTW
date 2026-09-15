# Wan 终端投影写读迁移：首轮固定验证

本轮替换协方差状态/公共原型写读，不再重排公共点或重复补帧诊断。
用户确认旧 SyncTube 方法可工作，但原始实验记录已丢失；保留该历史事实，
不据此推定 Wan 成功，也不以恢复旧记录为执行前提。

## 旧机制与迁移适配

核对来源：RICHAAARC/SyncTube 的
`main/methods/temporal_tubelet_watermark/{embedding,codebook,tubelet_partition,evidence,synchronization}.py`，
`configs/method/real_video_tubelet_sync_candidate_runtime.json`，以及
`main/vae/diffusers_autoencoder_kl_framewise.py`。
实际旧候选为 tubelet=4、patch=4×4、projection margin=1.0，全通道空间平铺。
旧 direction 按参考 support 独立生成 uniform[-1,1] 向量，L2 归一化；
payload sign 与 temporal sync sign 的乘积决定写入方向。不是只有管状重复。

旧后端是逐帧 AutoencoderKL，latent 为 T,C,H,W，mode×scaling_factor；
本轮是 Wan 因果时间 VAE，终端 latent 为 [1,16,46,40,64]。
写入发生在去噪完成、第一次 VAE 解码之前；不先输出 RGB 再用旧 VAE 嵌入。
解码/重编码继续复用 `runtime/c2a/chain.py`：解码 z×std+mean，
读取 posterior.mode 后 (z−mean)/std。不开启 Transformer，不生成新内容。

使用已落盘共享终端，三个臂 OFF、MESSAGE_0、MESSAGE_1。
保留特殊组0、末尾普通组45；写普通组1..44，分11个四组管状块，
每块160个4×4空间区域，全部16通道，共1760个不重叠support。
每个方向1024维，内部展平顺序为 T,C,H,W。CPU确定性生成后使用，
不依赖 CPU/CUDA 随机实现一致。写入保存为 FP32。

令参考块向量 x、单位方向 d、码 c=payload(message,i)×sync(time_start)，
采用 x'=x+c*d*max(0,1−c<x,d>)/||d||²。显式范数补偿归一化舍入；
落盘全部写前/写后投影和写后实际 margin，无门限扫描。

旧 sample_id 实际参与方向和码种子，迁移替换成固定公开协议 namespace。
首轮只比较消息0、1两条预声明模板，每个support独立派生payload sign；
没有64位任意消息恢复或扩容主张。角色分离HMAC、namespace、种子实现
是明确的迁移适配，不保证与旧仓库逐比特相同。

检测器仅接收重编码latent、预共享key和固定协议几何/候选；不接收原始terminal、
写入记录、视频身份、真实消息、真实删除位置或攻击参数。
旧 `embedding_projection_support_weight=0.09` 使用写端投影改善量，本轮移除该加分。
旧 evidence 中 speed_ratio 会开启/过滤scale、reference_shape会改变几何，
本轮统一固定几何与完整候选网格，不由攻击标签改变搜索。
旧 synchronization 的 GT offset/scale 本身只在排名后算误差；不将其误述成排名泄漏。
旧 aligned-payload/rescue/coverage 数值门限不迁入 Wan；这里只报告载荷一致率和覆盖。

## 首轮运行与调用量

从 `C2T1/c2t1_20260915T092031Z/shared_terminal_normalized.pt` 读取固定终端。
每臂一次 Wan 解码，以已跑通的 RGB8→H.264 CRF18/yuv420p、8fps 保存181帧正常MP4。
从各消息正常MP4的读回RGB派生：

| 接收条件 | 数量 | 帧数 | 保存链 |
| --- | ---: | ---: | --- |
| OFF/消息0/消息1正常 | 3 | 181 | 第一次编码 |
| 消息0/消息1未删除重存 | 2 | 181 | 相同第二次编码 |
| 消息0/消息1删除第138帧（零基） | 2 | 180 | 删除后相同第二次编码 |

删除与未删除重存比较，控制第二次有损保存；正常与重存区分保存影响。
七条视频各自独立重编码 g=0,1,2,3，共28次 VAE encode，3次 decode，0次生成/Transformer。
读取支持为起点g之后一个特殊首组及全部完整普通组，末尾不足4帧丢弃并记录，不补帧。
正常组数45/44/44/44，删除44/44/44/44。仍有实际候选覆盖损失，全部保留。
一次运行不根据正常结果自动跳过删除；按固定分母保留全部结果和失败。

## 显式同步、载荷检查与判读

每条接收视频统一搜索 g∈{0,1,2,3}、a∈{4/5,1,5/4}、b∈[-8,8]整数，
204个时间候选×2消息，复用28个观察，不再调用VAE。
映射 source_center=a×received_center+b；b的单位为原始RGB帧。
接收普通组j中心 g+2.5+4j，参考普通组r中心2.5+4r。
对各参考组按中心最近、误差≤2帧选接收组，同距取较早、每组最多使用一次。
四个参考组都找到观察才保留完整tubelet；部分tubelet不截短方向。
此中心映射是由逐帧后端向 Wan 组观察的明确适配，不是对时间VAE的精确逆。
固定单帧删除只检验这种显式仿射读出能否容忍局部编辑，不声称它具有局部删除模型。

每个参考support读取对应方向投影，再乘sync和候选payload。
总分=所有匹配support的clip(c×projection,−1,1)之和/固定1760，越大越好；
缺失贡献0且报告覆盖率、匹配数量、未截断平均投影、对齐载荷正符号比例。
这是旧投影关联与对齐载荷检查的迁移，未照搬依赖写端量和旧阈值的rescue fusion。
等价路径提前定义为相同g和同一完整参考→接收组选择列表，包含缺失位置。
输出所有参数行、等价类、每个消息的最优分数、身份路径及排除该类的最佳其他路径。
列出最高分并列项，不能用确定性排序掩盖消息并列。
身份路径只是固定比较项；对删除条件不将它标为真实完整同步路径。

正常失败：先检查写后margin、保存后真实投影与方向承载，不再修协方差原型。
正常可读但重存或删除失败：根据匹配保存链区分保存损伤和时间读出问题。
全部可读：仅支持下一次新内容确认；单OFF不能估计可靠FPR。
没有新增阈值、PASS门、状态空间搜索或GPU环境硬门。

## 落盘与验证边界

Drive输出包含固定配置、运行源码SHA、codebook.npz、3条写后终端、3条解码前编码RGB、
7条接收MP4、28条观察状态及成功latent、全部候选表、写后测量和失败列表。
正式统计分母仍为0，同时显式保留本轮诊断分母3臂/7视频/28观察，不抹去实验结果。
launcher日志位于run目录同级，runner独占创建新run目录。

本地仅运行NumPy构造数组测试、AST及nbformat schema校验；没有模型/GPU/Colab执行。
用户在Colab选择GPU运行时后从首格依次运行全部单元，实际验证以Drive落盘数据为准。
