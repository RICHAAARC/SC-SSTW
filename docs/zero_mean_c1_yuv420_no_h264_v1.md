# Zero-mean C1：无 H.264 的 raw YUV420 往返控制

本阶段仅冻结设计和参数证据，未实现runner/notebook，未执行颜色转换、视频解码、VAE或评分。输入固定run20260930T165530863126Z/source d4f5d73c506c20f4dfa5e1f74065836c23432064，result SHA3561a99003047ec89bda90b813dc712af3dc6ee2db6ec4907d009b7bf138d117。复用已保存RGB8及原零均值reader；不改writer、码、预算、loss、路径族、排序或阈值。

## 实际盘点与参数证据

三臂OFF/PAYLOAD_MULTI/OVERLAP_MULTI的RGB8_NO_CODEC.pt均已在Drive逐文件stat确认存在，大小88,966,697B；shape181×320×512×3、dtype uint8与期望SHA来自已hash绑定result。本阶段不再读取或重新哈希大Q8/float；OVERLAP的Q8已被主审上一轮核验。正式执行时三份输入均须实际SHA匹配后加载。

| arm | Q8容器SHA256 |
|---|---|
| OFF | be3f0eadb746f76136365e1e59dce79b963d759e4e16daefd9474e317782af30 |
| PAYLOAD_MULTI | 4adc9fcadd3b627e7d6300efcab481db8e91716b27e39b592aed6dd37421d78e |
| OVERLAP_MULTI | d20ee27689e43568ad5a295393aba256ed98b31a0fd9dcbe9d8739bd3d10ccc3 |

仅复制三份小SOURCE MP4（486707/469507/455277B）与小日志/收据至独立临时目录，实际媒体SHA均匹配result。ffprobe仅使用show_streams/show_format，没有show_frames/read_intervals或像素解码。三者都是H264 High、yuv420p、512×320、181帧、8/1fps；chroma_location=left。color_range/color_space/color_transfer/color_primaries均未报告，准确记为unknown，不能由分辨率推断BT601/709或limited/full。

原execution.log明确FFmpeg/ffprobe6.1.1-3ubuntu5，libswscale7.5.100；当前WSL非转换version探针相同。原命令只有-v error，没有原auto_scale详细日志，不能追认完整原转换上下文。未来执行仍保存实际binary版本、build/library行、完整命令及verbose stderr；版本相同不自动证明数值链相同。没有读取大float或重算上一轮5220cost/30投影。

## 唯一固定控制

每臂从同一已保存Q8直接取uint8连续RGB24字节。第一段真实物化为raw yuv420p文件，确认退出成功及长度后第二段重新打开该文件转RGB24，再重新读回RGB24实际文件供VAE。不能用一条保持内部metadata的filter链替代。原始Q8不经过float再量化、不重新decode、不调用H264。

第一段固定argv（文件占位符由独立输出目录解析）：

    ffmpeg -v verbose -threads 1 -f rawvideo -pix_fmt rgb24 -s 512x320 -r 8 -i pipe:0 -an -c:v rawvideo -pix_fmt yuv420p -f rawvideo -n {yuv420_output}

第二段：

    ffmpeg -v verbose -threads 1 -noautorotate -f rawvideo -pix_fmt yuv420p -s 512x320 -r 8 -chroma_sample_location left -i {yuv420_input} -map 0:v:0 -f rawvideo -pix_fmt rgb24 -n {rgb24_output}

保留原输入线程参数位置和自动像素格式转换，不加scale/zscale、BT709/BT601、range、sws_flags、bitexact或accurate_rnd。第二段显式带回原SOURCE已知left标签；本地-help证明该AVOption可用于解码，但未以真实转换试跑。它是codec-context属性，实际scaler采样位置还有独立控制项；left是输入metadata解释约定，不证明第一段物理格点与原x264前buffer或插值完全一致。第一段不额外加left输出标签。unknown色域/范围等字段不虚构，继续依赖实际版本auto fallback并记录其可观察信息；verbose未显示的内容仍是unknown。

FFmpeg官方说明rawvideo不保存尺寸/格式，必须显式提供demux参数；滤镜可插入格式转换，scaler行为涉及多个可配置选项。因此相同pix_fmt不足以证明完整处理相同。本候选叫“无H264 raw420控制”，不宣称严格纯H264单因素因果隔离。[rawvideo](https://ffmpeg.org/ffmpeg-formats.html#rawvideo)、[filters](https://ffmpeg.org/ffmpeg-filters.html)、[scaler](https://ffmpeg.org/ffmpeg-scaler.html)。当前官网不被用来证明旧6.1.1的全部自动默认值。

## 固定分母、接收与参考

单臂原始YUV420应44,482,560B=181×512×320×3/2；回读RGB24应88,965,120B=181×512×320×3。三臂新增两类原始字节总400,343,040B，另存3完整normalized tensor和6raw成本文件。每段命令、退出码、stderr、实际SHA/字节和按精确长度确定的帧数都保留。长度不符不能裁齐、补帧或继续伪称完成。

未来仅0Transformer、0VAEdecode、3VAEencode、3RGB→raw420、3raw420→RGB；0H264编码、0MP4解码。VAE同Wan revision0fad780a534b6463e45facd96134c9f345acfa5b、FP32、posterior.mode、Wan mean/std及前后cache清理。编码完整181帧，不裁177；保存[1,16,46,40,64]，主支持仅regular1..44，extra45保留不评分。固定g0不称恢复相位。

3臂×2key只有6次新infer、6次原payload read，合计23490目录/1044有效成本/22446结构排除、6path后评/12message后评。原完整family/ties及UNCALIBRATED、accepted_payload=False保持。盲API只接实际新normalized张量、公开key和mask，不接arm、truth、旧观察或路径名。

已有RGB8_NO_CODEC与MATCHED_SOURCE_MP4的6norm/12raw按SHA原样引用，不重新编码或评分，不能把引用算成18次新infer。主审audit_result/assessment仅为既有证据，不重跑原矩阵。新raw全部提交后再与两参考层比较同R44的真实路径rank/unique、起点、虚构edit、已定赢家及固定同一对手成本/局部time-block-age贡献。固定历史对手直接来自hash绑定父result.path_posthoc[arm/RGB8_NO_CODEC/key].fixed_historical_phase0_opponent的raw_sha256/catalog_index/path，明确是原FULL的phase0对手，非global其他phase；不用新增远古Drive根依赖。缺失只标该后评比较MISSING_REFERENCE，不改变盲排名或主分母。

## 失败、范围与下一步

初始化所有新观察/转换/normalized/read/evaluation槽和完整公开catalog，独立臂继续。保留失败、部分输出和未完成；canonical结果先于盲投影原子提交。输出目录不得等于或位于输入根后代，已存在旧输入不可覆写。new/reference/package完整性分列：参考不可用不抹掉新结果，新失败也不抹掉已有参考。任何执行完成都不自动升级科学PASS、FPR、质量或一般同步结论。

当前仅新增本config/doc、input_inventory、parameter_evidence及design_freeze。待主审确认才新增独立runtime/runner/builder/notebook/tests；旧runtime/method/builder/notebook完全不动。Notebook后续仍未发布SOURCE_SHA=None，current Python/fresh child，不设GPU型号或精确Python门槛。本阶段无commit/push/merge，无自动真实转换、VAE、媒体或Colab执行；不扩第二控制矩阵。
