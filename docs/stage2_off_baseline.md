# OFF 生成基线：一次四输出诊断

当前仅完成现有证据只读核查、真实运行代码和CPU接线测试；本协议尚未运行GPU。旧六臂run01保留，不改observer、强度、v2门或旧产物。没有发现足以把旧结果归因于某个机械错误的证据，不预先认定“8步不足”。

## 已有证据和缺口

主任务run01_readonly_audit已核六臂×两流31灰帧：所有严格>12差分支持均0，OFF原RGB非饱和且相邻最大差4；编码前也如此，不能单独归因MP4。当前prefix/final latent都是[1,16,13,40,64]有限非零，RMS .85077→1.35299，时间相邻RMS .70799→.20631，未发现latent被复制成相同时间帧或全部归零。

旧G1实际archive的两段OFF也不是健康画质证明。灯塔有显著彩色横纹，玻璃花大面积模糊并有异常颜色；其49帧RGB相邻max均255，平均分别7.455/.883。5Hz灰度31帧中灯塔各对>12像素数7154–45901，玻璃花36–1607。仅说明旧8步不是一概无像素动态，不证明主体运动或优质生成。只读图片/统计在 diagnostics/阶段二-生成二维位置原语/off_baseline_diagnosis/，未运行旧observer/v2。

旧G1与新run01相同model revision、8步、320×512/49帧/8fps、CFG5以及BF16 VAE；prompt/seed不同，旧来源配置max_sequence_length226而本次512。旧runtime Python3.12.13，新3.13.15；新记录transformers5.16.1等，旧没有这些包的完整freeze，不能据此归因。官方Wan示例使用FP32 VAE、默认50步且尺寸/帧数更大；这是已知配置差异，不是已证根因。本诊断不同时变更尺寸/prompt/seed来混淆比较。

## 固定四输出

共同锁定Wan-AI/Wan2.1-T2V-1.3B-Diffusers revision 0fad780a534b6463e45facd96134c9f345acfa5b、diffusers0.35.2；当前方块prompt/negative/seed1275/maxlen512/320×512/49帧/8fps/CFG5，详见configs/stage2_off_baseline.json。Transformer和text encoder一直BF16，去噪latent一直原FP32路径。

| ID | 去噪入口 | 步数 | VAE | 实际Transformer调用预算 |
|---|---|---:|---|---:|
| M8 | 现manual前6步+完整UniPC fork续6/7，无任何平移 | 8 | 原BF16 | 16 |
| P8 | 官方WanPipeline output_type=latent | 8 | 同BF16 | 16 |
| V8 | 不再去噪，严格复用P8 final latent | 0新增 | 从原checkpoint重新FP32加载 | 0 |
| P50 | 官方WanPipeline output_type=latent | 50 | 同V8 FP32 | 100 |

总3次生成、132次真实forward、4次VAE、4个MP4。另存每个输出相同RGB的无损诊断文件，不属于额外生成。V8的FP32模型通过AutoencoderKLWan.from_pretrained原revision/subfolder=vae/torch_dtype=float32重新加载，绝不对已经BF16量化的实例.float()冒充原权重FP32加载。P50的FP32仅指VAE，不改变Transformer/text encoder/latent精度。

只生成一个显式共同initial latent并保存；M8直接clone，P8/P50官方prepare_latents的实际返回值捕获保存且要求逐值等于共同initial，不把命令输入当已核实实际输入。各次生成均从同一scheduler config fresh创建；50步仅让官方set_timesteps产生对应网格，不特别调shift/历史。M8保持真实的零基step5后fork，既不简化成无fork也不平移历史。保存timesteps、actual_initial、final latent与真实逐forward记录。

M8/P8：若不同，只定位本配置manual/fork整体与stock入口存在差异，尚不能定位某一内部bug；若相同，仅排除本配置可见差异，不证明manual普遍正确。P8/V8只改变VAE加载/计算精度；V8/P50改变去噪步数设置（及其网格），两者用同FP32 VAE、同initial和其他配置。改善只属于本prompt/seed的有限效应，不能反推旧G1或所有失败的唯一原因；本链不穷尽所有精度×步数交互。

## 输出与判断

保存全部四身份、失败/未执行行、源initial/final、preencode_rgb.npy、无损RGB/MP4、首中末0/24/48预览。两流继续通过原production decode_video的5Hz/300帧/2073600像素/60秒限制，同一实际灰frame数组交原observer（delta12/support .0001–.5），原31左右实际轴及缺失不删除，不新建q注入或v2扫描。

描述全片RGB范围/均值/标准差、逐帧对比度、空间相邻绝对差、逐帧时间差；全灰轴保存相邻max/均值/>12支持和原observer全部行。没有用Sobel或任意清晰度阈值自动“通过”；需要查看四输出首中末及实际视频，人工判断是否能辨认方块与预期运动。清晰画面、较大像素变化、有效q三者分开记录；背景噪声/条纹也可能产生支持。

整体执行完整只标OFF_DIAGNOSTIC_EXECUTED_REQUIRES_VISUAL_REVIEW；输入/模型/资源缺口保留OPERATIONAL_BLOCKED，不伪装为observer失败或生成方法通过。比较后才提出一个有证据的下一步，不在本轮继续六臂、增强度/降门或扫prompt/seed。

## 工程上限和入口

建议一次L4 24GB或更高BF16 GPU，进程组硬墙钟3600秒包含模型加载/3次去噪/4次VAE及读回；依赖安装和clone在此前，实际模型缓存命中/下载时间仍记录在预算内。可先确保已有官方模型缓存，但不因此扩大运行范围。超时kill整个worker进程组；OOM停止，其他局部失败保留后只继续仍有意义的固定输出；无自动重试，原输出或execution.log/exit已存在则拒绝覆盖。保存实际显存峰值与外层真实退出码。历史L4跑通不保证本轮FP32 VAE/50步都能在限内完成。

- 真实模块：runtime/stage2/off_baseline.py，复用现wan_translation.py的既定decode/observer/编码帮助函数，不改旧方法。
- 入口：experiments/stage2/run_off_baseline.py，按单进程组3600秒运行。
- 薄Colab：notebooks/stage2_off_baseline.ipynb，首cell仅独立两行mount，无force；自动clone当前SC-SSTW分支，不依赖未提交的本机路径。GPU cell直接运行真实入口，无额外False锁；**只有在一次GPU预算已获批准后才执行该cell，本次准备没有运行**。
- Notebook启动前拒绝已有输出/执行记录；source_commit只对新输出用open('x')记录，不能篡写旧run版本。

实际计数采用register_forward_pre_hook(with_kwargs=True)，跨Accelerate对.forward的移除/重挂载仍保留，finally移除。此为本次新诊断准备期修复；旧run01使用显式call()计数，所以不是其低动态的已证原因。

CPU仅检查接线：真实CPU tensor、nn.Module prehook（模拟forward重挂载）及mock模型/调度，不运行真实Wan/UniPC求解、不制造新视频。A2已静态APPROVE方法与hook修正版；最终CPU日志及notebook静态结果置于delivery目录。原stage2/v2历史包和结果不更新。

官方版本依据：[Wan v0.35.2 pipeline](https://github.com/huggingface/diffusers/blob/v0.35.2/src/diffusers/pipelines/wan/pipeline_wan.py)；[官方模型卡](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B-Diffusers)。scope始终science_denominator=0，无科学、AISB或独立泛化结论。

最终交付检查：9/9 CPU接线测试实际通过、0skip、exit0（cpu_wiring_tests_final.log），包含实际nn.Module hook跨forward替换与旧输出/commit不覆盖回归。六个notebook代码cell AST通过且全部未执行。A2静态终审APPROVE；root已核计数hook和薄入口保护。此结论只覆盖代码/CPU准备，不是四输出GPU实证。
