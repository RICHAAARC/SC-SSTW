# 局部 Fourier RM 相邻窗口差分接收诊断 V1

本轮是在当前已存局部状态观测上新增最小接收对照，服务于开题设计的局部软观测与状态路径估计；不另立正式方法、不替换管状块＋状态空间同步主线。多步生成写入仍是原 MULTI25–49，载体、局部支持、RM码本、alpha、写预算与真实通道观测均不改。历史绝对SSE与新差分SSE分别报告，不能把不同分支或通道的成功拼接成整体闭合。

输入固定为 Video-Local-Fourier-RM-Channel-V1/20261003T064701431228Z/fixed_reference，source b394a307320b1b3fb6f5cf684f1adfc08a077833。24个keyed raw含3臂×4阶段×2key的44×4×4×8 q，来自12个底层观测。新receiver只接受q、key、公开availability。仅读raw JSON，不重读latent张量、不生成、不VAE、不MP4、不新phase，不依赖Torch、HF或模型。

设窗口投影q_j，相邻差分d_j=q_(j+1)−q_j，j=1..43；公开edge支持是m_j与m_(j+1)逐块交集。对原有限catalog中任一合法44窗源路径tau，预测模板为alpha[C_(tau_(j+1))−C_(tau_j)]，C为原composite_signs，alpha=1/sqrt5568。总成本是所有可用edge维数上的联合SSE除以有效维数，完整支持43×4×4×8=5504。必须按候选tau计算，repeat取同一源状态、skip取跳跃状态；不能在这些事件下仍用连续源差分。43个差分共享原窗口，不作为独立投票，路径坐标仍是44个源窗。

完整原catalog3915槽，174有效、3741结构排除。按公开mask后的整数差分模板逐byte精确等价分组，保留全部成员及44窗tau feasible sets。数值tie_atol=1e-12；canonical按原路径taus、event_type、event_i字典规则仅用于稳定展示，不解释为置信度。差分若消去偏置也可能消去信息，保留歧义、不自由仿射拟合、不编辑惩罚、不强制唯一峰。局部源转移u→v为stay/+1/+2、1..45内共45+44+43=132种，完整保留43×132局部成本。45个stay局部模板都为零，canonical stay绝不能当源位置定位。零支持、零差分energy及异常观测拒绝，canonical=None、全部acceptance=false；正常argmin仍是未校准诊断。

同一q上的ABSOLUTE_CONTROL重算原family完整path/class/projection/local_state，与同文件的旧infer逐字段核对；不重新提取q。24absolute＋24difference共48path records、8352path costs、187920catalog槽、179568结构排除；新difference局部edge成本136224。固定48path后评、24edge后评、24cached payload、48cached消息后评，不删失败。payload仅标REUSED，未产生新payload读取/恢复，消息答案不会进入path选择。绝对与差分不同归一化，原数值大小不能直接作机制优劣阈值。

先密封receiver_channel_readouts（新FLOAT/RGB8两通道）和receiver_reference_readouts（原terminal44/旧MP4g0），以及cached_payload_readouts，再做truth1..44与REGISTERED/WRONG_MESSAGE后评。旧MP4来自前轮VAE环境，仍只作跨运行参考，不能单独确认codec因果。所有原窗口证据、候选成员、ties、错误密钥与OFF控制完整保留。某raw缺失/损坏单独失败，其他inputs继续；cache payload坏不阻断state分析。EXECUTION_COMPLETE只表示固定诊断完整且absolute controls与原一致，不给scientificPASS。

Notebook固定首单元Drive mount，直接Run all用固定输入和输出Video-WM/Video-Local-Fourier-RM-Difference-V1/<stamp>/fixed_reference；当前Python仅探测NumPy，缺失时尝试修复，无GPU/venv/版本硬门槛。SOURCE_SHA由主会话发布后绑定；本地CPU既存观察回放可检验此接收诊断，但不替代新的独立盲视频或时域编辑验证。

研究后续目标仍是同链完成：局部状态可辨识、状态路径与歧义/拒绝、时间相关片段载荷、裁剪/删除/重复及声明范围速度变化、片段—序列聚合。重复载荷零误码不能证明同步的实际作用；receiver对照本身不完成轨迹水印全部方法机制。

本地验证（2026-10-03）：9项针对性NumPy测试通过（14.72s），覆盖实际repeat/skip差分、两端支持交集、稀疏精确等价、零支持/零energy/溢出拒绝、失败分母和truth后评封存、notebook绑定。完整既存观察CPU回放EXECUTION_COMPLETE：24inputs、48mode records、8352path costs、136224edge costs、48path后评、24edge后评、24cache REUSED及48cached消息评价；24个absolute controls与原infer完整精确MATCH。

独立公式同版核对通过：全部观测、dq、路径/class成本、模板成员/feasible sets、top/tie/canonical、path排名与后评一致。仅局部edge成本存在FP64归约顺序最大6.94e-17差异，edge后评Delta最大3.82e-17；均远低于原1e-12 tie容差且不改变任何离散结果。没有修改构造、阈值或科学判据。完整记录在diagnostics/local-fourier-rm-difference-v1-cpu-validation-20261003。

STATE正确密钥同观察对照的真路径名次：绝对为终态1、FLOAT3、RGB8 2、旧MP4 53；差分为终态3、FLOAT3、RGB8 1、旧MP4 70。差分只在当前RGB8对照选中了正确路径，旧MP4仍失败；结果完整保留，不能据此认定媒体盲同步完成。此CPU回放不是新增生成或新独立视频，Colab挂载与固定源码入口由用户Run all核验。
