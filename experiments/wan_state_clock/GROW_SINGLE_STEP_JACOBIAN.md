# 第30步共享状态的输入 Jacobian 对照

准备状态：只做CPU/合成工程验证；真实模型、显存、视频和科学效应均待用户Colab执行。

固定4case、5arm OFF/LOCAL_A/B/JAC_A/B、20video/80层；每方向组8marked/128位/层，缺失完整保留。原差分DCT载体23pair×4repeat=92票/16bit、MSE目标amplitude.5、硬接收器、CFG5、50步native UniPC均不变。

共享无梯度前缀0..29，保存进入step30的latent与完整scheduler，所有arm克隆同一状态/历史。LOCAL取得g_clean=∇clean L；实现为leaf z对z−σv_detached求导，完全等价。JAC完整双分支CFG前向，直接autograd.grad(L(z−σv(z)),z)取得g_z=(I−σJ_CFG)^Tg_clean，包括实际dtype cast和CFG运算链。无分离VJP近似、无整段轨迹反向、无VAE反向。权重eval/frozen，但输入梯度保留。

两种方向都使用u=−eta*g、v_control=v−u/σ，相同速度注入接口，JAC不是对z直接做梯度下降。因此不能要求预测clean局部损失下降，也不预设Jacobian更好。

LOCAL eta=.1固定；E_LOCAL=mean((native_next_LOCAL−native_next_OFF)^2)，其中OFF与LOCAL为同状态、同历史、同step30。JAC只做一次unit native probe，scale=sqrt(E_LOCAL/E_unit)，一次live更新，报告实际E及误差；不scan/cap/retry、不读消息恢复选scale。E是离散单步状态响应能量，不是终态或物理时间积分能量；相等不意味着相同峰值/视觉质量。零响应无法匹配保留失败。

每case先完成全部single-step可行性，再逐arm执行no_grad tail31..49；输入VJP graph在函数退出后释放。候选独立启用官方enable_gradient_checkpointing hook/nonreentrant checkpoint，shared loader与旧runtime不变。checkpoint block forward/recompute计数代表context入口，不是完整Transformer调用，也不代表全部成功重算；真正OOM由方向失败及attempt/completed识别。每方向CUDA synchronize/reset peak，保存baseline/after/peak allocated/reserved与耗时。CPU资源为null。OOM不降级local或改变模型路线。

预算每case256 Transformer显式前向、130 live scheduler、2local gradient、2input VJP、2unit native probe；4case总1024/520/8/8/8。checkpoint内部block replay另计。prefix30步共享一次，单步5arm与5个19步tail；不能以20完整独立轨迹计算其调用数。媒体复用旧独立VAE子进程，20decode/60encode/20MP4；不以terminal恢复筛选。四层为terminal、float RGB重编码、uint8RGB重编码、MP4重编码。OFF仅参照不是FPR；PSNR不设质量门槛，人工检查语义、伪影、连贯性。

保存源SHA/manifest/初始与prefix/历史哈希、梯度与step30/terminal、每层盲恢复及truth后评估。所有文件新入口隔离；不声称与Frame Guidance某版完全相同，实验仅判别当前单步输入Jacobian方向的作用，无法直接证明多步轨迹最优性或时序同步鲁棒性。

同场一致性用既有固定check_delta容差ATOL2e-5/RTOL2e-4核对带图预测与共享no_grad速度；失败为SAME_FIELD_FORWARD_MISMATCH，不偷换公共速度。after_allocated/reserved为direction函数退出前，不应解读为graph-free释放后占用；runner另存allocated_after_direction_release。
