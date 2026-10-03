# 局部 Fourier RM 通道定位 V1

本轮仅定位上一轮 STATE_MULTI 从生成终态真实路径第1名，到 MP4真相位第53名的退化层。复用 20261003T050759442813Z 已保存 OFF / PAYLOAD_MULTI / STATE_MULTI 三臂 `terminal.pt`；原生成 source SHA 为 5d0284feecfe8e29bf9d1cc55afff2cf623d257d。终态不修补、不重生成，不改变 MULTI25–49、载体支持、RM码本、alpha、状态目标能量、eta、cap、载荷或接收器。

每臂加载终态一次、使用原 FP32 frozen VAE revision 和现有 adapter 解码一次。FLOAT_VAE_ROUNDTRIP 直接将原解码所得 [0,1] FP32 RGB 送现有 deterministic posterior-mode encode，包含原 decode 的 clamp；不是无裁切纯 VAE。RGB8_VAE_ROUNDTRIP_WITHOUT_CODEC 从同一 RGB 使用原 `np.rint(clamp(rgb,0,1)*255).astype(uint8)`，再转 FP32 /255，送同一个 encode。没有 FFmpeg、MP4 保存、MP4 读取、YUV转换、去噪或梯度修补。新调用总数3 decode、6 encode，完整RGB不必保存，仅保存形状、范围、fingerprint、量化误差和6个重编码normalized张量。

当前 FLOAT 与 RGB8 共用同次解码，因此两者差异可定位到增加的 RGB8 量化。终态到 FLOAT 包含 VAE 解码、[0,1]裁切与编码复合层。旧 MP4 是上一进程的 VAE解码结果再经原 yuv420p保存和读回；旧解码RGB未保留，不能证明本次与旧次解码bitwise相同。因此 RGB8 到旧 MP4 只作跨运行一致性参考，不能将差异单独归因 codec。新旧环境、同一 frozen model revision 均记录；环境差异不会引入新的硬门槛。

所有几何观测统一 g0 / R44：新FLOAT、RGB8各3个张量；既存terminal（仅规则latent1..44）与旧 `FULL_SOURCE181.phase0.pt` 各3个参考，共12个观测张量。每个张量2把key，12个新read加12个参考read共24。接收器保留每read 44×45局部状态SSE、3915有限路径catalog槽中的174有效路径SSE、结构排除、精确等价类与数值tie，完整总数47520局部成本、4176有效路径成本。无全局phase重搜索，也不按真值选择路径。

新channel接收函数只有actual normalized observations、key、公开支持，保存到 `blind_channel_readouts.json`；原终态和旧MP4另存 `reference_readouts.json`，明确是离线参考。终态参考本身不是盲视频检测。重复32-bit载荷对所有R44窗口读取，仍不依赖状态路径，不能当作同步有效性的证明。全部 state_path_accepted 与 accepted_payload 固定 false。保存receiver记录之后才加入已知未编辑g0真实路径和REGISTERED/WRONG_MESSAGE两条消息后评，48条评价、24条局部后评不变。旧MP4 raw只在重放之后精确比对，对新channel不作为路径输入。

所有表在模型load前预建。参考在主进程CPU回放，新VAE在fresh child运行；child启动、load、单臂decode、单channel encode或read失败保留对应missing/failed行，其他臂继续，不删失败、不扫描、不以参考成功覆盖新channel失败。baseline使用固定目录与固定arm相对文件名重定位，原result中的绝对Colab路径只当记录，不用来寻址。每个输入hash核对对应原result记录，缺单臂不阻断其他臂。科学判定与EXECUTION_COMPLETE分开。

Notebook `notebooks/video_local_fourier_rm_channel_v1_colab.ipynb` 使用当前Python、优先可用Torch、serial fresh VAE child；不建立venv、不硬要求GPU型号，也不强制导入WanPipeline、transformer、scheduler或codec。用户直接Run all，固定Drive输入不需填写路径。输出目录 `Video-WM/Video-Local-Fourier-RM-Channel-V1/<stamp>/fixed_reference`。源码需由主会话发布并绑定SHA；本地CPU/fake验证不是新真实VAE证据。

该诊断服务于开题设计的局部载体、真实通道存留及盲状态空间同步主线。若本轮定位到首个状态信息退化层，后续另行决定具体载体/读出改动；本轮不改方法，不引入LAST正式替代。时间相关片段载荷、时域编辑、片段—序列聚合与拒绝仍是统一链路剩余项。

CPU验证（2026-10-03）：新6项＋旧local-Fourier 8项共14 passed in49.17s；子进程cleanup／dependency-repair机械修正后新6项再验6 passed in9.76s。覆盖同次decode两分支输入、单臂缺失、单channel encode失败、VAE load失败、child spawn失败、固定成本和调用计数、posthoc不改receiver记录、notebook source binding与VAE-only依赖。fake结果不计为真实VAE证据。

同次真实既存数据CPU回放输出 `/home/richar/projects/Video-WM/diagnostics/local-fourier-rm-channel-v1-cpu-validation-20261003/20261003T063028862260Z/result.json`：6参考张量、12参考read、2088有效路径成本、23760局部状态成本、24后评；6个MP4-g0 inference与payload都和原raw精确MATCH。STATE_MULTI正确key终态44真实路径rank1、Delta=+2.1369121984207706e-6；原MP4-g0 rank53、Delta=-6.056593117471226e-5。无新的decode、encode或生成，所有调用仅CPU reference_path_read / reference_payload_read各12。
