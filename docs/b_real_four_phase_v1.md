# Card89 实际四相位补齐运行包

Notebook默认直接浅克隆GitHub现有`dev/local-joint-state-payload-v1`源码及保存JSON，Run all无需手动上传ZIP。若`/content/SC-SSTW-b-real-four-phase-v1`已有可用源码目录，则直接使用；本地user_bundle.zip仅作可选副本。Notebook无需嵌入源码或改变模式。

输入保持原P0/PM各181帧实际MP4 readback缓存，位于`/content/drive/MyDrive/Video-WM/B-Legacy-Position-Margin-V1/fixed-clean-transfer-v1/media/{P0,PM}/readback.pt`。消息真值只在全部盲结果保存后，从`B-Real-Local-Wan-V1/fixed-45-frame-phase01-v1/references/truth.json`读取。

旧Card88本地保存数据作为普通ZIP sidecar：`experiments/wan_state_clock/data/b_real_four_phase_v1_saved.zip`。24个blind Q/votes/tables成员只读复用；旧280候选不重新打分、不重新归约。另一个post/card88_evaluation.json成员仅marker后读取作对照。没有hash、manifest、精确版本、Git干净或预期摘要门禁。

固定新增VAE load1、cached RGB loads2、E6、FFT6、encoder名义66、两钥新Q/票各12。仅新起点62/63/64每arm各一次E41，latent0不读、r1..10全读。同E共享FFT，两钥独立；D/DiT/writer/codec/新媒体decode/gradient均0。

旧4个E45同reset前缀加新6个E41供16个逻辑相位视图；20份不同Q/票绑定32逻辑表，每查询137合法(p,s)。仅408新候选产生1224新直接均值/12240项、408新共享归约。688不同候选/22016位行与1096逻辑状态/35072位绑定分开，不当作新独立容量。

单个查询只使用其自身4相位和key。任一相位Q/票/归约技术失败都使族未决，不能删项选峰。精确最大并列全部保留，不取首项、不引入epsilon。无tau、无标定或拒读认证。每个p按真实名义帧支撑后评，不能跨p比较同local_row；group-only覆盖若无共同帧定义标未决，消息按固定group/bit、单源按canonical cell比较。MIXED精确身份增损null；源ABSENT/PARTIAL与消息级假覆盖、present-unread分开。

输出固定`/content/drive/MyDrive/Video-WM/B-Real-Four-Phase-V1/fixed-seen-prefix-completion-v1`。attempt在数据/模型操作前保存；失败/中断消耗尝试，后续Run all或CLI只读，不重入/补跑。所有固定盲表/候选/条件载荷及失败先保存，再marker，之后才读真值/源角色/旧评价。

CLI（从可用源码根目录，无需.git）：

```sh
python -m experiments.wan_state_clock.b_real_four_phase_v1_run
```

代理本轮仅必要CPU fake后端/假FFT和结构/入口检查，不运行真实模型、FFT、媒体或GPU，也不执行真实保存数据的新四相位科学回放。用户最终需在有上述Drive数据的Colab中Run all，核验真实load1/E6/FFT6及结果。没有新质量测量，不能声称画质重新验证；没有独立宿主、FPR或完整消息信任结论。
