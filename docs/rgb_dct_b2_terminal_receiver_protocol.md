# RGB-DCT B2 终端接收器梯度固定任务卡（2026-09-25）

本卡由授权审计会话在 B2 代码和 GPU 运行前冻结。B1 的同次完整结果是
`RGB-DCT-Terminal-Gradient-V1/20260925T114509787453Z/result.json`
（Drive 文件 ID `18cPG8R0wgQumIMNrxDnGIbuniQwD6izD`）：
源码 `dc116f192a0780d5f7f61545a83790968c1794ec`，
配置 SHA-256 `6dcefab2fb6e113ab790c028f1d9349bd2419c31340007a219dd760ce1c0b596`，
协议 SHA-256 `a37fcbfa7b27250701f383bf60fdf648d4ef48838381749b61e9e0ac9a67b7c1`。
B1 是 6/6 SCORED、0 invalid、1086 帧的有效负结果：clock 的 OFF/SINGLE49/TERMINAL46
C 为 12/30/19；umbrella 为 18/30/15。两例 T46 完整 VJP、MP4、181 帧读取均成功。
T46 即时 z47 响应 support RMS 均为 `R*=0.042943312697648145`，自由尾程后的
终态相对 OFF 只为 0.0260305655 / 0.0249805402。T46 在 float RGB 层 C 仅 16/12，
RGB8 为 16/15，MP4 为 19/15；故不是 MP4 才首次失败。clock 连续代理损失从
2.33183e-6 降至 6.36857e-7，但 C 仍不达 24；umbrella 从 4.11905e-6
升至 5.16206e-6。B1 原始行、30 个 q、失败/资源账保留，不在原两来源上改阈值、
预算、步点、强度、目标或补写候选。

## 唯一 B2 机制假设

B1 的同史全尾程梯度有效，但有限 T46 更新经过 T47–T49 的非线性传输后，
无法稳定把接收器组符号送过判决门。B2 **只改变梯度的作用位置**：用与 B1 完全相同的
冻结 FP32 VAE、完整 181 帧 RGB-DCT 可微代理
`L=mean_g relu(-q_g)^2`，在 OFF 终端 `z50` 计算 `g=dL/dz50`，
于同史原生 T49 最后一步写入负梯度方向。不得更换损失、添加 margin、
选组、扫描、额外写入步或因结果选择符号/候选。这样直接检验消除 T46 后续自由尾程
传输是否足够；并不预先假定只是写入强度不足。

冻结模型、revision、320×512、181 帧、8 fps、50 原生 UniPC 步、CFG 5、
negative prompt、固定 key、30 组 q、`C=sum(q_g>0)`、
`C>=24:H1`、libx264 CRF18 yuv420p、FFmpeg RGB24 完整读回，均与 B1 相同。
终端梯度为完整 float RGB 的真 VAE 输入梯度，先按 B1 容差与 NumPy receiver
核对 30 个 q；VAE 参数冻结但输入图不断。将 `g` 的 latent 时间端点 0、45
置零，仅支持 1:45；非有限或零梯度保留工程无效。以 `-g` 为**终端 latent 期望方向**，
复用 SINGLE49 同史 T49 native scheduler 的方向到 velocity 映射及 unit-response
probe；单次归一至终态 support RMS `R*=0.042943312697648145`，
随后按原路径 FP32 VAE 解码、MP4 保存和盲接收器读取。梯度臂仅一次 T49 写入，
不执行 T46 VJP，不使用解析 lift 替代梯度。SINGLE49 是原无梯度阳性参照，
两臂同源、同 R*、同 native T49 history，OFF 为阴性参照。

在实施前锁定两个此前未用于该系列判读的新来源：

- `eval_copperkettle_s2501`，seed 2026092501，prompt:
  `locked camera, a copper kettle resting on a plain wooden table, steady soft indoor light, no people, no cuts`
- `eval_paperplane_s2502`，seed 2026092502，prompt:
  `locked camera, a white paper airplane gliding slowly across a plain blue background, steady soft light, no people, no cuts`

每来源固定 OFF、原 SINGLE49、B2 TERMINAL49_RECEIVER，共 6 个 MP4 槽、
1086 计划帧；先持久化六槽。新来源、顺序、prompt、seed 不得按结果替换。
同 noise/prefix/history 从同源分叉，不复用 B1 的 MP4，也不把 B1 旧来源计入
B2 分母。任何失败、超时、缺失保留原槽及原始阶段，不重试、不补样本。

## 必留的区分性读数与判读

每臂记录 float RGB、确切 RGB8 raster、MP4 RGB24 的全部 30 个 q、C、
连续代理损失和 181 帧状态；仅最终 MP4 的冻结 C 产生 H0/H1。
记录 `g` 范数/指纹、原生 T49 probe 与实际终态响应、OFF/新终态指纹、
实际 `g·(z50_B2-z50_OFF)` 一阶预测、实际 float RGB
`L_B2-L_OFF`、组符号翻转、FFmpeg 读回、同源 MP4 RGB RMSE/PSNR、
调用/重算账、峰值 GPU/主机内存、磁盘暂存生命周期与所有失败。
这些诊断不得决定候选选择或重运行。

完整六槽且 OFF 均 H0、SINGLE49 均 H1、B2 均 H1，才可报告
“两新来源、固定 key 的功能观察”；任一 B2 的 MP4 C<24 是此固定构造的负结果。
若 B2 在 float RGB 已 C<24，定位在终端接收器梯度/固定预算机制；
若 float RGB 达 24 而 MP4 失效，定位在量化/媒体链；
若无完整六槽则标记 INCOMPLETE。即使全部通过，也不声称低误报率、
总体泛化、载荷或视频质量达标；RMSE/PSNR 仅作诊断，没有视觉 PASS 门。

复用 B1 已验证的 Run-all Colab 环境安装、模型、子进程、FP32 VAE 梯度暂存和
媒体链。独立 B2 分支/工作树实施，保留 B1 不变；源码 S 审查发布后 notebook N
锁定 S。A1 唯一写入，A2/A3 对同一冻结版本独立审查，A4 综合，A5 里程碑复核。
CPU/静态验证只证明执行结构；真实 Colab/GPU 由用户运行，随后由审计会话核验。
