# 冻结末步方法的独立完整视频验证

当前仅CPU/fake/静态实现，未跑真实GPU/模型/旧数据重评分，也未完成真实视频人工观看检查。

使用仓库velocity_calibration.json已有holdout：白天鹅内容seed20261001、蓝色缆车内容seed20261002。两prompt与两seed集合均同本路线4dev无交集；独立性仅针对这些开发组合和冻结方法，这两条名单曾用于另一反演候选的holdout与裁剪运行。本轮是相对当前tube/state开发集的预先固定跨内容/种子验证，不声称全项目未见样本。

新入口flow_tube_state_holdout_run/config/notebook独立，旧开发入口不改。复用原纯method/runtime和媒体/接收函数；编排固定2case×OFF/LAST_A/B=6实际视频24内部接收phase。每case prefix49步+共享CFG100Transformer，OFF/LASTA/LASTB3native末步合计52native。全轮200/104、6VAEdecode/6MP4save/24VAEencode。CPU原carrier.write(OFF)仅作数值参照，不产生TERMINAL独立媒体。无裁剪、多步、Jacobian/VAE梯度、强度扫描、候选择优或新阈值。

冻结原1760管状块、state×polarity×sync codes、margin1、full projection u=P(clean)−clean、v−u/sigma/native49和原五mode state_clock.read(obs,publicbook)。权重/媒体精度和H264 CRF18/yuv420p不变。共享prefix/速度/history源一致及三个末步数值等价量继续保存。写端projection只诊断，不进接收得分。

每mode固定4marked（两case×A/B），所有失败保留6视频24phase槽；partial/missing和缺候选不伪造0间隔。消息间隔=各自最优clock候选下best_correct.score−best_other.score，truth仅read之后加入。OFF仅两候选排名、score0−score1与并列，不作正确/错误/FPR断言。这里是二候选归因，不是16bit或任意payload解码。

coverage报告4相位完成数、nominal1760支持分母、最佳候选matched_supports；四phase是真实RGB内部起点181/177/177/177帧，不是未知裁剪攻击或时间同步成功证明。输出完整rankings/消息间隔、质量、实际u/native响应、等价摘要。RGB/时域质量不设门槛；用户需观看每case received_videos/OFF.mp4、LAST_A.mp4、LAST_B.mp4检查语义、伪影、连贯性。

新目录MyDrive/Video-WM/FlowTubeStateHoldout；Run all固定执行，无菜单。首codecell独立Drive两行，source SHA发布后绑定。完整source/config/book/张量/媒体哈希和两stage退出码保留。未宣称generalization、FPR、payload容量或真实裁剪鲁棒性成功。
