# 同载体多步机制：完整投影与末步补偿

分支dev/flow-tube-multistep自ae9b03a隔离，仅新增文件。当前CPU/fake/static实现，不跑真实模型/GPU。A裁剪和成功LAST baseline入口不改。

旧实现与原记录已核：Flow-Tube-State的44/45/46每步请求deficit*c*d/(3||d||²)，u与same-history D各R/3、累计R预算，47/48/49自由tail。原run flow_tube_state_20260916T111114159525Z/source fca6f1f4a447da8f3b725425f0db960541cb6a74，原result Drive id1cWF_zjt1WJhepyVyK-Os3ZpH5bZvsMCi：124TF/62native/12shadow/5media/20encode全部完成、failures=[]、first_round_criteria=false；四项saved quality比较均true。TERMINAL A/B五mode均正确；FLOW_A只有global_state/local_without_update正确，FLOW_B则这两项错而matched/global-local及local_state正确。因此历史是归因不稳定，不是本次oldFlow质量或OOM失败。G1和后续true-terminal梯度OOM是别的路线。

新方法复用同1760管状块/state×polarity×sync编码、margin1、key/A/B、原receiver/native scheduler。EARLY仍固定44/45/46，但请求完整u=P(clean)−clean，即deficit*c*d/||d||²，去/3和Rcap，无新增eta/强度/搜索。对照LAST49、EARLY_LAST44/45/46/49、EARLY_ONLY44/45/46加47..49自由续采样。改变dose与末步补偿，不把它包装成新梯度算法或相对旧路线纯时间因果证明。EARLY_ONLY是无final补偿消融，不是等预算公平性能对比。

共享tree：prefix0..43=88TF44native；baseline44..48=10TF5native，49共享CFG2TF，OFF/LASTA/LASTB三native。EARLY A/B各44..48=10TF5native，再49各CFG2TF后EARLY_ONLY及EARLY_LAST两native。合计每case124TF66native。每早期控制有一个同history未控shadow，共6/case；final49复用同z/v/history已经执行的OFF或EARLY_ONLY作响应基线，无额外shadow。4原dev共496TF264native24shadow，28实际视频28decode/28MP4save/112真实RGB phaseencode。共享早期数据在两个arm统计中复用，但真实调用只记一次。

中间必须scheduler.step真实传播history，不能用最终projection代替。记录每步clean预测的两消息hinge/min/mean signed projection与nominal matched score，after native noisy-state投影仅机制诊断，不是终态/MP4消息证据。每controlled step记录u、实际delta_v及D=native(controlled)−samehistory native(uncontrolled)的support/global RMS。每armsum(D RMS)、max(D RMS)、sum(D RMS²)及u sum/peak，与net_terminal−OFF分开；sum不是累计终态位移，不等于公平预算。49前margin、49补偿u/D和final margin单列。EARLY_ONLY/EARLY_LAST的49输入/历史fingerprint必须相同。

LAST baseline要与成功runtime在同initial/scheduler的CPU原生路径数值一致；native49/Pclean等价仅是调度器桥接。EARLY各控制步没有这个终态等价替代。所有28视频都独立完整MP4链+原五mode盲read，写端diagnostic不进reader。每scheme8marked固定分母，OFF只排名不FPR、不宣称任意payload恢复。原H264CRF18/yuv420p和四origin181/177/177/177不变。质量只诊断，无旧1.5阈值或新科学门槛；用户需看内容、伪影、连贯性。

新Run all notebook固定FlowTubeMultistep输出，独立两行Drive挂载和发布后sourceSHA；无mode/参数菜单。工程验证不是GPU/科学结果，所有失败28/112槽及计数/文件/source哈希保留。
