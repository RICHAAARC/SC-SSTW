# Local Fourier RM 同一RGB8媒体链诊断 V1：冻结计划

复用原20261003T050759442813Z三臂终态，原source5d0284feecfe8e29bf9d1cc55afff2cf623d257d。新runner不运行生成或writer，不改MULTI25–49、local support、RM码本、alpha=1/sqrt5568、eta/cap/budget。VAE与原revision一致，当前实际环境如实记录，不复原旧包环境、不增加GPU或版本硬门槛。

每臂恰一次现有VAE解码（含原[0,1]clamp），一次原np.rint RGB8量化，保存完整181×320×512×3 uint8文件及SHA256。DIRECT_RGB8、RAW420、MP4每支从同一文件重新打开，验证哈希。DIRECT转换FP32 /255送现有VAE编码；RAW420通过原 materialized RGB24→YUV420→RGB24，无H264；MP4通过原io.encode_rgb的libx264/CRF18/yuv420p/8fps/threads1命令，独立ffprobe和原io.read_mp4的noautorotate/RGB24读回命令。保存raw420、各RGB24读回bytes及commands/stdout/stderr/ffprobe、每支源raster身份，再送同一VAE编码。原raw420纯媒体helper复用，历史zero-mean方法receiver/orchestration不引用。

全轮3 decode、9 encode、9 normalized、18keyed局部q；同q按当前冻结ABSOLUTE_CONTROL/ADJACENT_DIFFERENCE两式评分：36records、6264path costs、102168difference edge costs、35640absolute局部状态成本。R44、g0、174有效路径／3915总catalog、完整classes/ties/feasible sets不变。18次新真实payload_read与36消息后评为重复载荷，不能参与path选择或当时间相关恢复。36path后评、18edge后评只在所有receiver raw和快照封存后加入truth1..44及消息。

所有表预建，缺臂、load/decode/单支媒体或encode/read失败独立保留，其他支继续；实际调用与固定分母分别报告，全部acceptance=false，工程完整不自动scientificPASS。RAW420与MP4可能使用不同内部color metadata/conversions，二者差异只能归到实际新增媒体链，不能单独宣称H264因果。新DIRECT若已错误，保留该不稳定，不选择旧正raster或调参。

新媒体helper仅处理原命令捕获与完整byte持久化；检测层直接复用当前Local Fourier RM绝对／差分receiver及payload_read。实际VAE/媒体与CPU接收均在子进程执行，封存后的truth/message后评在父进程执行，失败仍保留固定行。Notebook首单元exact Drive2lines、Run all固定输入/output Video-WM/Video-Local-Fourier-RM-Same-Raster-V1，无venv或runmode控件。新模型运行由用户执行，agent仅做CPU/fake与固定工程raster媒体验证。

该轮仍服务于开题的媒体稳定局部状态观测→盲状态路径→时间相关片段载荷→声明时域编辑→片段/序列聚合链，未替代任何剩余完成条件。历史MP4失败与前轮RGB8有限正信号各自保留，当前三通道才是同一解码raster的受控对照。
