# 固定真实RGB裁剪与接收同步诊断

状态：仅实现/CPU合成/静态检查，不运行真实模型/GPU/旧数据新评分。用户Run all执行VAE-only实验。

固定源run `FlowTubeStateGuidance/flow_tube_state_guidance_20260919T202545864571Z`，source commit `892013f44a5ab1596386f5e7046548ca4150d44e`。4原dev组合×OFF/LAST_A/B=12源MP4；每源start0/4/5、长度129帧，36片段×4origin=144VAEencode。8个marked源×3相关裁剪=24marked片段，不是24个独立视频。start0同样是截短控制；start4/5在同长度下比较偏移/phase变化，不是对比未裁完整181帧。

读取原MP4真实RGB，乘255 round恢复uint8，裁真实数组[start:start+129]，保存lossless uint8.npy和source range端点（左闭右开）。实际重读npy/hash作为接收输入，不二次MP4转码。四origin依实际RGB起点取1+4k长度129/125/125/125；VAE latent长度33/32/32/32。不是latent裁剪替代。源SHA/config/book/各MP4 hash核验，公开key重建book并与已存book核对。独立输出，禁止写入源目录。没有Transformer/VAEdecode/MP4save调用。

原state_clock.read只得到obs和publicbook，真实起点、arm、消息truth、source latent均不进其接口。固定11窗、1760名义support分母、missing窗innovation=1与原scale/offset/event表不变，无新阈值/补帧/候选扫描。原五mode保留。新no_search_matched/state/without_update固定g0,scale1,offset0,delta0,boundary11；从已经盲算的candidate/class原matched_score/state_score/without_update_score直接提取，不再改变分母或按真值选路径。global/local本身均搜索，不冒称no-search。

裁剪后处理不用原report(delta/edit)。allocation将received中心g+2.5+4j先乘scale再加offset映射source；start=S参考offset=+S和phaseg=(-S)%4。因此三参考(g,offset)=(0,0)/(0,4)/(3,5)，完整几何窗分别[0..7]/[1..7]/[1..7]，结构参考8/7/7窗、1280/1120/1120支持；partial窗不算完整。评分始终保持原11窗分母，只有时间评价分母使用结构完整且VAE观测实际可用的reference窗。VAE边界context变化使该几何映射不构成精确逆VAE保证。

事后分别报告selected best的逐窗origins+selected group匹配比例、selected observation class是否等于reference、top ties是否含reference类和不同observation class数。缺失==缺失不计匹配；实际参考phase缺失则无可评估时间比例而不是0。不同g使用不同RGB VAE输入，不因center相近就合并。deterministic best不是唯一clock，message_unique不代表时间路径唯一，不以best offset==start作为主要成功条件。不宣称逐帧定位或完整时间鲁棒性。

固定36fragment/144phase槽；每start8marked×8mode（原5+no-search3）都报告完成/归因/正确减另一候选的间隔/coverage/clock/ties。truth只在read后join。OFF排名/score0−score1，不作FPR。完整记录在case/result及crop/detection.json；顶层保留逐行结果，notebook只打印24行compact摘要（fixed分母、完成、正确、间隔min/max、窗匹配计数/比例、OFF完成数）。失败不会隐藏为成功或0间隔。

模型与VAE精度沿既有版本。源model revision若未固定，resolved VAE revision仅记录比较限制；不把source SHA误认为模型revision。新目录FlowTubeCrop，首独立Drive两行、source SHA发布后pin、固定Run all，无菜单。保留必要crop数组/144obs和检测记录，不复制源大张量或重新生成。
