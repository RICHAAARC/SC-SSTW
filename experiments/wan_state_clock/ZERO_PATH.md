# 原四个开发样本的同路径 no_grad 零对照

固定输入：`MyDrive/Video-WM/VelocityCalibration/velocity_calibration_20260917T011448482341Z`。
原运行源码 `5db00fa60e1a3cad88b6736e1a402381afad3623`；dev_p0_s0、dev_p0_s1、
dev_p1_s0、dev_p1_s1 共4案例。新输出位于独立 `MyDrive/Video-WM/VelocityZeroPath/<run-id>`，
不写入/覆盖原run，不重算候选选择，不补媒体，也不运行holdout。

每case复用原完整config、codebook和pre_intervention_state.pt（latent+完整scheduler）。
先检查source/软件版本、latent形状与FP32、scheduler配置/50步日程、step_index44、
begin_index0、this_order/lower_order_nums、model_outputs/last_sample/timestep_list完整性。
原torch2.11.0+cu128与diffusers0.40.0由Notebook恢复；匹配torch不重装，无GPU型号门。
原scheduler对象是用户自己的可信保存文件，以map_location=cpu反序列化；不是通用外部pickle入口。

调用原prepare_generation(load_vae=False)重新编码原prompt/negative，丢弃新建随机latent，
绝不重跑0..43。模型设备上的model_outputs/last_sample、schedule设备上的timesteps/
timestep_list，以及原生CPU sigmas/训练schedule表分别恢复；其余字段、类型和dtype保持，
迁移后逐字段核对CPU值一致，不将所有scheduler tensor统一搬GPU、不清零历史。
原tail在torch.no_grad下用全零3×1760系数、use_checkpoint=True及原responses/R续采样44..49一次；
其无梯度分支本身不启用checkpoint。A/B共享这一次Z0，不执行两次零尾程。

**调用预算：** 每case12 Transformer、6 live scheduler、6原shadow；总48/24/24。
前缀、backward、梯度、response probes、VAE、MP4全部0。模型/文本编码加载是恢复开销，
不混入Transformer续采样计数。每case独立子进程；缺失必需状态则记录具体失败及最小恢复建议，
不静默补前缀、梯度或旧基线。4个case分母固定；比较也保留缺失的原ZERO/FLOW行。

**输出统计：** 保存Z0；分别对原ZgA/B求RMS(Z0−Zg)，对原3rho×AB求RMS(Zrho−Z0)，
均先tensor相减再使用原support/global RMS定义。复用endpoint_metrics的原float64投影与loss定义，
给出相对Z0的loss绝对/相对下降、正确码字有符号投影增益及正确-竞争margin增益。
原FLOW绝对margin等record原样透传；重算绝对值另列，不改写原值。
原媒体盲读出和NO_SELECTION原样透传，不根据更新基线重选、不新增事后PASS阈值。

**恢复证据边界：** RESTORED_WITH_REENCODED_CONDITIONING只说明保存状态的结构/值/设备恢复
及原配置条件重编码。原prompt embeddings、resolved模型revision/hash未保存，model.revision=null，
无法事前证明原模型条件位级相同；Z0−Zg不能独占归因于grad/no_grad，更不能自动称bug。
复用原receiver证据但没有Z0媒体零基线，不能彻底归因编解码损失。

工程验证：小型真实Wan BF16+真实UniPC保存/加载/恢复后完整六步终点逐位一致；Fake完整case
只有一次12TF零尾程，原run全部文件hash不变；同范数相反张量差RMS=2反例；缺历史和首case失败
不扩大调用且保留4case；Notebook首mount/AST/固定路径与版本恢复静态检查。均为CPU工程证据。
未执行真实模型/GPU/Colab/Drive。当前Notebook固定运行源码 `b9b7e89486370be1b98255b7c9741b0bfefffca5`；
Notebook交付提交与该运行源码提交分开，原输入证据仍属于5db00fa，不改写原run。
