# M4 VideoSeal 固定外部参考

本入口随 B 分支发布，默认使用该分支可用源码。实现和 CPU/fake/static 检查不构成真实 VideoSeal 运行结果。原 B M3 三读法的负结果保持原样；本入口没有模型优劣、质量 PASS 或 FPR 判据。

直接打开 `notebooks/b_m4_videoseal_reference_v1.ipynb` 并 Run all。首可执行 cell 仅挂载 Drive。输入固定为 `MyDrive/Video-WM/B-Rank3R-New-Source-Confirmation-V1/fixed-four-five-v1`，输出独立为 `MyDrive/Video-WM/B-M4-VideoSeal-Reference-V1/fixed-four-five-v1`。无需上传源码 ZIP；已存在 attempt、启动收据或执行痕迹时，在源码下载/setup/model 前只读。云端未知不启动；准备成功后通过 Drive API 写读一次普通启动收据，不能视为并发原子锁。失败或中断消耗一次尝试，后续不补跑。

使用官方 `videoseal_1.0`、原生 256 位、共同 owner256（seed 202610260101 的冻结字面值），4 个来源使用同一条消息。保持 scaling_w=.2、scaling_i=1、JND、chunk32、step4、repeat、clamp、FP32。embed 显式 bilinear/align_corners=False/antialias=True；detect 显式 antialias=False。一次 detect 保存原生 logits，排除 mask 列后在原设备原生 float32 上按帧 mean，再 >0；不再调用 extract_message。嵌入结果用 rint(clip(x,0,1)*255) 量化。

4 份既有 OFF precodec RGB24 经 embed 后产生第一代 WM H264；从其实际回读生成 CROP、左右 delete8、左右 repeat8，再第二代 H264。OFF 对照直接读取既有对应 MP4；缺失不使用另一代输入替代。ffprobe 每个 OFF/WM 片各一次，解码使用该 probe 的帧数与尺寸，不隐藏额外 probe。

固定预算：4 embed、4 quantize、24 encode、48 probe、48 decode、20 edit、48 extract；网络内部预期 8 embedder 与 248 extractor 前向，另 8 个非网络 JND 块处理 724 帧，分别计账。Wan/VAE/生成/B 接收器重跑均为 0。48 提取任务含 12288 位；每臂 6144 位（full 1024、fixture 5120）。首技术失败停止后续物理调用，全部 48 提取结果或明确未决先保存回读，之后才使用消息后评；未读不计作正确或错误。

质量为 4 precodec、4 full/readback、20 fixture/readback，共 28 条、4468 帧、4440 相邻帧对。RMS/PSNR 是归一化 RGB 距离；时序值来自相邻帧残差差。RMS=0 的 PSNR 用 null+POSITIVE_INFINITY 表达。没有 ROI/感知质量阈值。B 的同定义质量只引用原 M3 已保存 4 条 full OFF→PM/readback；B fixture/precodec 同定义项为 N/A，不把两个距离相加。

B32 与 VideoSeal256 容量、写入载体及读出链不同，完整分母分列。OFF 匹配率不是 false detection；没有新增 DETECT、FPR 或 wrong-key 定义。path/exact support/wrong-key 为 N/A。共同 owner 消息使跨来源位准确率在消息维度相关，不能宣称四条独立消息实验。

模型加载 CWD 固定 `/content/video-wm-m4/vendor/videoseal`，其 ckpts/HF/源码/日志/临时张量均在 `/content`。只按实际缺失补依赖，不盲重装 torch/torchvision、不使用版本门禁。setup 深层导入 `videoseal.utils.cfg`，不加载模型。Drive 仅本次必要 JSON 与 24 个 WM MP4，不复制 OFF/权重/PT。

末尾 Drive API 完整读取核心 JSON 与 24 个 WM MP4，保存状态与计算/参考状态分开；本地 ZIP 保留全部可读结果 JSON 和 24 媒体，缺失明确列出。发起 files.download 不代表浏览器已保存完成，保存检查不触发任何模型/codec 补跑。CLI 可在普通无 .git 源码目录导入与 `--help`；真实运行仍需要上述实际模型及媒体环境。

2026-10-27 启动修复：首轮在源码导入探针的嵌套 Python 字符串处发生 SyntaxError，尚未到启动收据及 worker；云端目录为空，下载 ZIP 仅有缺失报告。本版修复该转义错误并直接回显探针错误；结果单元显示准备失败，空结果 ZIP 标明仅含缺失报告且不自动下载。修复由实际执行嵌套探针的无 .git CPU 检查覆盖，不改变方法、预算或已有尝试只读规则。先前已打开的 Colab 标签页需重新从发布链接加载新版 Notebook，不能只重跑旧单元。

审查修复后的内容级边界：恢复同时识别数字 receipt 的 `.json` 与 `.json.tmp`，pending/不可读记录保守计一次已消费，状态与 attempts 不匹配不算未进入。配置与加载对象核固定实际方法参数（FP32、.2/1、32/4、repeat/clamp/JND存在），失败观测保存在 model.json；不核源码或依赖版本。B 引用核已保存后评四源/三读法/120任务×32槽及4条SCORED full质量，缺失标 B_REFERENCE_UNRESOLVED；原 method/contribution 字段原样保留，不要求匹配预期负结论字符串，prompt/seed等来源文字不作门禁。
