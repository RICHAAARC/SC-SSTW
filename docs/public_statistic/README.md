# 二维图像统计与持续终端反馈：CPU 准备

root 已只读核对 `0e92dff..1698fbf` 的 aisb/calibration/sync/linalg 四核心文件无差异，未漏迁旧分支后续核心修复。基线为纯 AISB/calibration/sync 的 `0e92dff`，本分支新增独立统计与 callback 控制内核。当前已做小张量 CPU 测试，未处理两个真实视频、未生成任何 Phase1 MP4、未运行模型或 GPU。science_denominator=0。

## 公共读出与旧路线差异

输入是 [0,1] RGB **码值**，Y=.2126R+.7152G+.0722B；F=(mean(Y hx),mean(Y hy))。h 为最低 DCT 半周期 cos(pi*(i+.5)/N)，去均值并归一单位 RMS。F1 正值代表左亮右暗，F2 正值代表上亮下暗。X/Y±只是这两个特征轴的正负，绝不是主体平移或物理位置。读出接口只收 RGB，不读 OFF、命令或 key。

旧 G1 在8步零基5后直接 VAE 解码 noisy latent，对全周期色度余弦统计做 −mean(q·u) 线性目标，以 .005 RMS(z) 归一梯度更新一次，再正常续6/7。旧 lighthouse 残差 .9760651005、glass .6885504813；二维 rank 可用但轨迹和质量失败。旧质量是 [-1,1] RGB RMSE/RMS(OFF1)，不能与这里 [0,1] 绝对预算比较。原始结果位于交付目录 cpu_preparation/old_g1_result.json。

本提案是输出统计控制路线的实质机制修订。色度换亮度及增加次数本身不构成解决失败的证据。真正可证伪差异是：完整剩余 solver → FP32 VAE 浮点终端 RGB 的有限 target MSE；梯度穿过整个剩余链；对同一候选 forecast 同时要求终端误差下降和固定 RGB 质量预算才接受。学习率梯度步 Δ=−min(eta,cap/RMS(g))*g，cap 是上限，不强制近目标仍走固定幅度。目标和质量基准全程固定，累计接受更新 RMS 受总预算约束。拒绝后原分支正常前进一步，无线搜索或重试。

## Phase1：待一次批准的有限码值干预

固定两个现有输入：P50 saved.mp4（512×320、8fps、6.125s），JumpingJack/v_JumpingJack_g01_c01.avi（320×240、30000/1001fps、3.1031s）。只取 [0,2) 按8fps输出16帧，原分辨率。路径在 configs/public_luma_phase1.json；可迁移路径但不可换内容、身份、尺寸或参数。时轴是 FFmpeg 规则输出轴，不假定 source index=固定整数倍，保存原 PTS。

每输入固定 OFF1/OFF2/X_PLUS/X_MINUS/Y_PLUS/Y_MINUS，共12原片、192固定行。每帧 RGB 同加 ±(2/255)h，clip[0,1]，保存截断通道比例、RMS和最大幅度，再四舍五入8位。理想未截断 ΔF=±a、交叉0是构造代数，不是独立发现。固定 libx264 CRF18/yuv420p/threads1、8fps；不选编码质量救结果。编码后实际 RGB 重新盲读 F；preencode 与 MP4 都保存完整16行。每个动态臂分别相对两个 OFF 报告响应，不能用平均OFF隐去重复差异。OFF完全相同不提供科学 noise bound。

v2 拟议方法门不要求原始坐标对角传递。对每内容 s、固定时刻 t，D_st 的两列是 (F_X+−F_X−)/2 与 (F_Y+−F_Y−)/2。公共 C 是两个内容全部32个 D 的算术均值，不能删帧、按内容重估、求逆、拟合偏移或校准读出。C/a 的最小奇异值≥.5且条件数≤3。

每内容至少共同15/16时刻同时满足：该时刻 D_st/a 最小奇异值≥.5且κ≤3；四动态臂对两个OFF的全部8个向量误差 ||F_arm−F_OFF−sign*C_col||≤a/4；四动态臂逐帧相对原source RGB RMSE≤3/255。不能各臂分别挑15帧后合称共同通过。source 是扰动及编码前从原视频固定采样得到的 RGB。整片 RMSE 是所有16帧、全部像素及RGB通道平方误差的平均再开方；相对同编码OFF的RMSE另报，不能扣掉OFF编码损失或取代source硬门。另外全部12臂（含OFF）整片编码后相对原source总RMSE≤3/255硬AND，任何一臂总质量超预算都不可豁免。两个内容共同门都要满足，最坏帧及所有失败原行全保留。每个 D/SVD、8误差、共同mask、C与OFF重复差值都输出。

半名义最小增益 .5 排除绝对响应很弱但κ=1的情况；a/4容差允许名义四分之一偏差；κ3是预定各向异性工程预算。3/255为2/255构造幅度加量化编码余量。全部人为开发阈值，无科学、噪声或感知保证。稳定旋转、反射和可逆混合可满足门；不同内容分别稳定但混合不同由公共C误差拒绝。结论仅为相对各帧OFF的共同加性二维线性混合相容，不是自然内容F的仿射模型，也不是heldout校准。

外评使用逐帧OFF构造加性响应，公共reader始终只读RGB。本门不证明单个未知内容F可盲恢复命令，不构成训练/留出校准性能。X/Y是输入特征轴，混合后不要求读出同坐标方向。门失败仅说明本冻结共同稳定/质量目标未达，不能推广为任意仿射可校准方法NO_GO。

原 own signed≥1/255、cross≤.5/255完整保留为 raw_diagonal_diagnostic_only，不进入方法门。preencode使用相同v2评价但仅诊断，最终门只用MP4读回；缺臂、非16×2或非finite则INCOMPLETE，不nanmean或改分母估C。人类可见质量仍待审，低RMSE不自动感知合格。v1→v2是尚未媒体执行的提案修订，旧v1材料保存在 cpu_preparation/history_v1_raw_diagonal。

一次建议批准仅覆盖固定2输入、12编码读回、192行外评，CPU单进程，600秒墙时（含所有子进程），每命令120秒，零自动重试。正常方法失败仍保留其余固定有信息输出；不足/失败不扩样或调幅度阈值。当前只是准备，命令运行本身需要用户批准。所有输入只读、输出新目录，不上传Drive。

## Phase2 / Phase3 条件缺口

CPU callback 测试验证完整 toy history clone、多次 normal advance、固定目标/质量参考、预算和拒绝语义；**不证明真实 Wan 可微路径成立**。拟 after44..47/50 的四反馈仍是候选频率。提案真实反馈目标为 F*=F(固定OFF1终端RGB)+a u，u=恒定±e1或±e2，正常49帧逐帧同量；不是把自然内容统计抹成0。OFF只在生成端/外评使用，不进入公共读出。可沿P50同prompt/seed/model，真实 eta、a、逐次/累计预算、终端质量预算必须在 Phase1 与 adapter profile 后再冻结。

旧 velocity().detach()、decode_rgb().round().uint8() 会断梯度，不能移植作可微 adapter。需要浮点 decoded RGB 统一F、完整历史内的梯度链、冻结权重/prompt而仅候选latent求导；真实 Wan checkpoint/offload、显存及反向成本尚未知，不能拿 no-grad forward 数量充 GPU 预算。adapter 不存在，当前没有 GPU ready 入口。VAE forecast 下降不保证最终 MP4 读回下降。持续反馈失败则定位该路线，不自动扫层/换参数。

Phase1可读且质量可接受才提交真实adapter与明确预算；Phase2最终六臂双向与质量成立后，Phase3才研究 AISB/pilots/sync及时间状态闭合。届时仍须冻结时序模板、帧轴到solver状态映射、公共pilot名单、完整歧义集合、固定分母和判据阈值；现在均仅条件计划。恒量二维统计成功不能直接迁移旧PASS、keys或宣称水印成立。两内容仅开发因果证据，不是泛化、盲检或科学验证。

## 使用与交付

本地优先：在源码根执行 `PYTHONPATH=. python experiments/public_statistic/run_phase1.py --config configs/public_luma_phase1.json --output /absolute/new/output`。依赖 numpy、ffmpeg/ffprobe；Phase1不依赖torch或模型。新输出目录、同名日志/退出记录已存在则拒绝。CPU单测依赖现有torch2.5.1+cpu，不改共享环境。

薄 notebook 使用本地源码zip及已可访问的两个原始文件；不含假远程链接。修改的仅文件路径，固定身份仍须由执行者核对。当前未运行 notebook 媒体单元；批准后按顺序执行才能验证真实codec和依赖兼容性。所有失败保留，不将 CPU mock 当真实编码测试。
