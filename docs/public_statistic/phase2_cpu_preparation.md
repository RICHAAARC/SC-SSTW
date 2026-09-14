# Phase2 可微终端反馈 CPU 工程实现

实际代码为 runtime/public_statistic/phase2.py：直接调用 Wan1 风格 transformer(hidden_states,timestep,encoder_hidden_states) 与 FP32 VAE.decode；不调用被no_grad装饰的WanPipeline.__call__。单transformer、普通batch timestep、CFG公式、完整UniPC.step正常续程；显式拒绝transformer_2/boundary_ratio/expand_timesteps以及未经验证offload/持久transformer缓存。prompt及权重冻结，浮点VAE z*latents_std+mean，输出/2+.5 clamp[0,1]后同F，不round、uint8或detach horizon。VAE cache在finally清除。仅支持单batch。

每反馈在正常after-index后，以当前latent为唯一求导变量。完整剩余solver历史通过clone_graph_state递归克隆；tensor.clone保留nonleaf图，不遗漏model_outputs/timestep_list/last_sample/lower_order_nums/this_order/_step_index/_begin_index。外部next_index与内部step_index不一致即拒绝。候选和baseline forecast各自独立完整state；候选仅终端target MSE严格下降且固定OFF1终端RGB质量预算满足才接受。固定目标F(OFF1)+a*u及质量reference不随反馈更新。eta梯度步受单次RMS上限和累计接受RMS和约束，无线搜索；拒绝照常推进原solver，不重置历史。

第一次实际CPU故障已保留：真实UniPC带nonleaf history时copy.deepcopy抛RuntimeError。修复为递归tensor.clone，旧内核在phase2_cpu_preparation/initial_snapshot保留。CPU测试用torch2.5.1+cpu/diffusers0.39.0的真实UniPC，6步、小可微transformer/VAE替身；两反馈after2/3。有限差分、完整history隔离、OFF与正常链精确相等、严格接受/质量拒绝、detach断链、缓存异常清理及步索引守卫均检查。该配置仅CPU接线，不迁移为真实GPU参数。

真实Wan大权重、BF16梯度数值、50步after44..47、FP32真实VAE反向、checkpoint/offload/显存和墙时尚未实测。旧detach/uint8路径不复用。CPU替身通过不证明真实Wan可微adapter资源可行或模型可控。未来GPU一次小试验的a/eta/逐次与累计budget/质量budget/反馈频率及资源需单独批准；不能以no-grad forward数虚报反向成本。当前无模型下载/GPU/媒体/科学晋级/commit/push。

CPU执行：PYTHONPATH=. CUDA_VISIBLE_DEVICES= python -m experiments.public_statistic.phase2_cpu_check --output NEW_DIAGNOSTIC_DIRECTORY。每次输出必须新目录，保留测试日志、实际计数和实际源码快照。公开reader不读OFF/命令，控制与外评使用固定OFF不等盲恢复；终端forecast改善也不保证最终codec读回成功。
