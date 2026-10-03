# Local Fourier RM 唯一16×16低带宽局部基 V1

本轮只实施2026-10-03已采纳的一个局部空间载体构造。此前同raster实测证明原8×8局部状态在DIRECT差分存在有限正信号，而RAW420/MP4仍选错误路径；重复payload存活不能替代state路径。没有把失败归因单一H264、chroma或VAE环节，也不重新尝试历史4×4 Walsh terminal-STE路线。

新支持固定为(4:20,8:24)、(4:20,40:56)、(24:40,8:24)、(24:40,40:56)。中心与原8×8相同，四支持互不重叠；底部触latent h40边界，不事后移动。对16×16网格，取(h,w)≤((-h)%16,(-w)%16)的非DC共轭代表，按min(h,16-h)^2+min(w,16-w)^2及原(h,w)lex排序，固定前32。每列为sign×sqrt2×cos(2π(hy+wx)/16)/16；所选均非自共轭，列正交且零DC。密钥置换/符号使用原VLFRM1/basis/order及sign域；RM32、45state、4age、age×8+chip分配、5568active／192boundary、channel4、alpha=1/sqrt5568、eta696与cap1不变。cap是只缩小的上界，不把所有梯度强制放大到1。更大支持与更低频率作为单一固定carrier设计，不能分别宣称因果；同系数能量可能降低每像素幅度。

真实generation四臂OFF、PAYLOAD_MULTI、STATE_OLD_MULTI、STATE_LOWBAND_MULTI。原state臂保留原8×8 pilot函数，payload与真实CFG/native history同原协议。每臂完整50步、MULTI25–49，common initial与pristine scheduler逐臂fresh deepcopy；核对before25 z/history/conditional/unconditional指纹。保存200实际step记录、100writer sidecars（conditional_clean、pilot_delta、z_pre、z_post、cfg_clean）、实际payload/pilot/mergednorm及终态。sidecars只属写入诊断，不进入接收快照。生成与VAE/media分子进程，退出后释放模型内存；不调用旧实验orchestration。

每臂仅一次现有VAEdecode（含原clamp）与np.rint RGB8。完整181×320×512×3 uint8、88965120bytes保存并核SHA；三通道分别从同一个文件重开。DIRECT为FP32 /255直接encode；RAW420为原materialized RGB24→YUV420→RGB24；MP4为原libx264 CRF18/yuv420p/8fps与独立RGB24读取。共4decode／12encode。完整保存raster、YUV420、RGB24 readbacks、MP4、commands/stdout/stderr/原ffprobe和实际VAE环境。RAW420与MP4可能有不同color conversion/metadata，差异不单独确认H264因果。

接收器的两个公开family OLD8及LOWBAND16在全部四臂、三通道、正确/错误key上各自独立提取，不能按arm选择或根据truth择优。两family得到同形q44×4×4×8，再调用完全未改的ABSOLUTE_CONTROL和ADJACENT_DIFFERENCE。g0/R44、174合法路径／3915总槽、完整45局部states及43×132有界transitions、整数模板exact等价类、tie1e-12、lexcanonical、歧义/拒绝不变，无新phase/编辑族/罚项/阈值。payload每channel/key只一次，共24次新真实read，与两family数量无关；48消息后评在封存raw/snapshots后加入，不选family或path。

| 固定项 | 数量 |
| --- | ---: |
| generated / native steps / controlled steps / writer sidecars | 4 / 200 / 75 / 100 |
| underlying normalized observations | 12 |
| OLD8+LOWBAND16 keyed q / ABS+DIFF scores | 48 / 96 |
| path costs / catalog slots / structural exclusions | 16704 / 375840 / 359136 |
| difference edge / absolute local state costs | 272448 / 95040 |
| path / edge posthoc | 96 / 48 |
| genuine repeated payload / message evaluations | 24 / 48 |
| VAE decode / encode | 4 / 12 |
| same-format quality pairs | 18 |

两个family/两个模式、43edges、三通道和重叠payload votes均不视为独立样本。结果分别报告原构造和新构造，不能把不同family的低cost相加或择优宣称检测。完整receiver/payload snapshots不含writer sidecars、生成日志、原终态或truth；truth1..44只后评加入。全部acceptance=false；执行完整只确认工程全分母齐全，不自动scientificPASS。每个臂/媒体/VAE/观测失败独立保留，其他可用支继续；源与依赖失败也有完整setup矩阵。

quality以已存同格式RGB24 bytes按短frame chunks计算RMSE/PSNR，全4臂6成对×3channel=18，零新增VAE/媒体调用，无质量PASS阈值。写端继续记录实际更新和投影闭合；terminal loss或可读重复payload不等于MP4 state路径存活。

Notebook固定Run all、cell0精确Drive mount两行。当前Python与已可用Torch优先；实际AutoencoderKLWan/WanPipeline import及FFmpeg/ffprobe不可用才必要repair，无venv、GPU型号、旧包环境或版本门禁。agent仅CPU/fake/static验证与发布准备，真实模型运行由用户执行。

本轮针对开题中的真实多步写入与媒体稳定局部状态观测，同方法MP4路径若出现正信号也只说明本固定候选与有限时间family。时间相关片段payload未实现，独立盲裁剪片段、声明删/重复/速度恢复、片段—序列归因聚合与必要拒绝仍是后续闭合部分。不能以多步引导执行、writer sidecar或重复payload零误码替代这些完成条件。
