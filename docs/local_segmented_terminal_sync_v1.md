# B 分片多步载荷 + 原生终态仅状态：固定 clean 用户运行包

本轮已采纳 Card54 单一缓存宿主与固定35槽（VAEload1/D1/E2）方案，并授权在现有dev分支发布，供人类在GPU上直接Run all一次。真实结果待运行；当前已有验证仅为合成CPU/静态证据，不是新组合的科学结果。agent不执行GPU或真实35槽。

B使用自身GROW native50、sampling25..49局部控制；视频潜时间1..44分4×8，四通道同时间写同一段。alpha.5/mask44160/eta5520不变，t0/t45分别保护首末段而不参与净读取。main/tube_state/b_segmented_payload_v1.py的target、full-cell几何与vote reducer局部借鉴A已验证实现；无A residual clock、framewise VAE、CENTERED/U/去重receiver依赖。runtime/wan/b_segmented_payload_v1.py有可调用50步适配；当前clean runner/notebook不调用它，仅有stub分派验证，不支持新B真实轨迹或真实梯度验证结论。

首包仅源01缓存terminal及native uint8，缓存宿主不是独立新来源。主审确认terminal FP32/finite；native由实际读取槽核对，坏输入保留失败不补D(P)。state-only使用B-JPM-v1前22方向、gamma_s=1/sqrt2、4 ROI、ch4..15双latent相反符号、一次FP32落盘，无payload/parity修正。writer与observer都不接受message；state5和16跨payload段边界，clean局部reader逐Wan单元归段。

35槽：source terminal1/native RGB8 1/book2/construct1/VAEload1/D1/quantize1/codec2/probe2/readback2/E2/state8/segment FFT8/quality2/latentdelta1。固定8表176状态3696竞争15488ROI分数3872池化分数，32段256计划位，两质量各181帧/180相邻行。全部状态表和clean计划先落盘；payload signed-votes/exact-zero masks与计划先落盘，再打开writer真值评价。失败、中断、未知消耗尝试，后续Run all只读已有记录。

质量是cached-P/new-S pipeline增量：可能混入原P与当前S的VAE环境差异，不能归因纯state效果。仅RGB8和MP4，无float P配对，不重解P、不设质量PASS。状态/局部payload、共同位BER、覆盖、错误/擦除与票裕量退步分开报告。clean使用公开固定网格，无估计路径，不构成盲同步闭环，不声称同步增益；phase2盲编辑路径及预算尚未采用。

历史固定小幅state-only为1/22状态、旧GROW32/32，保留有效负结果；最新22/22+32/32来自终态state+payload，不能改写为此新组合正例。ECC支线已收束，不在此包重复终态用户消息。

入口notebooks/local_segmented_terminal_sync_v1.ipynb与scripts/build_local_segmented_terminal_sync_v1_notebook.py同步。配套普通ZIP仅含必要B源码，与notebook一同发布到现有dev分支；默认SOURCE_LOCATION通过普通URL自动下载该ZIP，无需手传。SOURCE_LOCATION也可填写可用目录/ZIP/URL；直接Run all，无执行模式开关、hash/manifest/版本/Git门禁。CLI同样可从无.git目录执行。
