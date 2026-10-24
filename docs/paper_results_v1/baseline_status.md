# 被动原生 baseline 接口

`experiments/paper_results_v1/native_adapters.py` 仅接收调用方已经加载的 backend。`VideoSealNativeAdapter` 与 `RivaGANNativeAdapter` 的 embed/extract 只有调用方主动调用才执行；导入和报告均不触发模型。没有 loader、下载、设备选择、训练、codec、评测或 32 位公平归约器。

`model_version/source_version/weight_identity/environment` 是可选定位记录；不因缺项、变化或可选记录回调失败拒绝接口。实际消息长度、输出形状、dtype、有限性、颜色、归一化与 transport 语义继续检查。没有自动媒体 hash，也没有预期摘要相等检查。

VideoSeal 要求显式 `native_message_length` 和 `detect_output_layout`（`T,1+K,H,W`、`T,K,H,W` 或 `T,K`）；检测返回 `preds` 的通道/维数必须匹配。原生 soft 输出完整保留，小输出 inline，大输出必须由显式 `native_output_store` 保存并返回 `{lossless:true,uri,format}`，有限性与形状检查在 sidecar 前执行。sidecar 回执是调用方的保真承诺，本接口不声称验证了其磁盘内容。`VideoSealLoadedBackend` 调用已加载模型，接收调用方准备的 `[T,3,H,W]` RGB 浮点 `[0,1]` 张量和严格保位的 `[1,K]` 消息；不增加颜色/消息映射。

单独 embed 返回前，`imgs_w` 必须保持输入形状、浮点 dtype 与有限数值；无需先调用 extract 才能发现非法输出。该检查不隐式裁剪 native 输出或修改其颜色。

RivaGAN 的 `CallableTensorBackend` 接受明确 embed/extract 回调，必须声明 transport mode、color layout、decoder normalization、codec comparability。`RivaGANLoadedTensorBackend` 接受注入的 tensor/array 模块和已加载 encoder/decoder，按声明 RGB8→BGR 或 BGR8 读取 `[T,H,W,3]`，归一化 `uint8/127.5-1`，每帧重复同一 32 位；输出逐帧 32 soft logits，保存 native `>=0` 的逐帧位，不做全片 reducer。隐藏 OpenCV mp4v/20fps path backend 未纳入。

历史比较定义只作说明：VideoSeal v1.0 256 通道卡/`y_256b_img.pth` 的每通道 j 写净位 j mod32，再平均非通道轴和每位八通道，严格 `>0`；RivaGAN 采用 Peachypie98 社区 32 位 checkpoint（非 DAI-Lab 官方权重），全帧逐位平均，`>=0`。这些 reducer 不在本公共接口中，二者时间冗余均不等于四段各八位。相同净位数不表示能量、质量、码率和推理成本匹配。

历史 two-pilot 中两个模型各两个源已完成 embed，四次共享 libx264/8fps/CRF18/yuv420p codec 完成。后续 60 评估行因媒体状态登记 bug FAILED；八源 confirmation 未执行。不是模型从未加载，也不是有效 baseline 性能结果。当前 P2 没有同源公平外部 baseline，旧四源队列未采用、不自动成为分母。保存定位见[结果索引](results_index.md)。

当前接口没有完整有效的 32 位公平 baseline；checkpoint 当前可用性、真实模型输出、完整恢复率、质量与 FPR 均未由本轮验证。新的四段八位 baseline、阈值、拒读或公平匹配方案仍是待用户采用的方法/协议决策。
