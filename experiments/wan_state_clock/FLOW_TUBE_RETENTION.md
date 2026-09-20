# 固定单次写入保留效率诊断

本轮仅准备，未执行真实模型、GPU或旧数据新评分。现有分支内独立入口，不改已发布多步实现或evidence索引。

固定原dev_p0_s0/dev_p1_s0（原2内容各种子0），OFF与T44/T46/T49各A/B，14视频/56内部RGB相位编码。不是独立holdout。载体1760管状块、state×polarity×sync、margin1/key/消息/native50步及盲receiver保持原样。

先沿唯一OFF完成50步，保留44..49节点、44/46/49完整scheduler快照。每消息LAST49沿u=P(clean)-clean写一次，实测同history native D49的support RMS作统一尺度R。44/46沿该时刻原full correction方向，以一次unit native probe测Runit，一次缩放u*=R/Runit，再一次live native更新；无cap、/3、扫描、重试或末态择优。分母为[:, :, 1:45]全部元素的RMS，与原flow_step.measures一致。记录global RMS、u、dv、真实D、R误差与scale，不能假设浮点native响应完全线性。unit零/非有限或R非正保留失败。LAST actualD已是锚点，不重复执行。

R依赖未来OFF末态及消息，因此这是诊断性匹配尺度，不是在线可用的时间选择规则、统一强度校准或公平感知质量证明。

每arm先记当前clean投影前后变化（u的预测作用）；native后state响应D与之分开。利用tail本来需要的CFG(t+1)，记录更新后节点t+1的clean/state相对OFF响应；完成一个自由native步后利用CFG(t+2)记录短期响应。44/46均自然存在这两个节点，49标N/A。最终记录net terminal delta及clean投影、完整MP4原五模式best-correct minus other与相对OFF同消息参考间隔增益，truth只在reporting join出现。局部clean分数是固定码投影诊断，不代替盲receiver。只给配对表，不把12marked当独立内容样本或做伪显著结论。

成本每case：OFF100TF/50native；T49AB共2native；T44AB共20TF/12native/2unit probes；T46AB共12TF/8native/2unit probes。合计132TF/72native/4unit probes；全轮264/144/8，14decode/14save/56encode。计数逐path另存；原始calls_by_path生成成本与media累计不同，media在独立子进程释放Transformer后执行。所有可用终态均入媒体，固定失败槽不删。

源码/manifest/初始/embedding/OFF节点与完整history/控制u与D/终态/媒体/盲检测保存。当前生成无自动旧input复用，因此不存在未经核验的模型历史复用；只做一次固定OFF路径，无环境硬GPU型号门槛。控制u与D真实张量保存，短期节点的配对数值及clean诊断落盘，OFF参考节点可追溯。

PSNR及残差仅相对OFF扰动，感知质量未测；请观看OFF与T44/T46/T49视频的内容、伪影和连贯性。不得把PSNR当新质量门槛。未来holdout接口固定enabled=false/selection_rule=null，先冻结共享规则后另行独立验证，本轮不在视频内择优时刻。

预定predictive_diagnostics只取T44/T46的8个case×message×time槽，current_clean_gain及short_clean_gain分别与terminal_projection_gain、主local_state MP4间隔gain比较符号；报告有效/缺失、同号/异号、单方及双方零值。T49独立参考不进入预测性分母。两个basecase内高度相关，不作p值/显著性，不根据摘要自动选择时刻。response_norm_ratios是响应support RMS相对actualD的比值，不代表消息保留率或方向一致性。
