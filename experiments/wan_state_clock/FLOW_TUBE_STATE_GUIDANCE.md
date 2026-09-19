# 原tube/state载体的末步guidance桥接

状态：新分支dev/flow-tube-state-guidance自Flow-Tube-State d5877c5隔离；只CPU/fake/static工程验证。未跑真实模型/GPU/媒体、旧真实数据重评分。原4开发组合的内容与种子已同GROW开发名单逐字段核对，不是holdout。

| 历史路线 | 编码/控制 | 本候选区别与证据边界 |
|---|---|---|
| 恢复的SyncTube ZIP | framewise diffusers AutoencoderKL；最终sync_rescue_fusion配置嵌入enable_sync=False，实际codes=payload；sample_id进入码种子 | 已有source/results/paper包，不能说记录丢失；不是Wan3D VAE或生成轨迹。旧evidence以非零.09权重引入写端mean_projection_after-before，不能继承成严格仅视频盲检测成绩 |
| 当前Wan state_clock/projection_margin | 1760非重叠4×16×4×4管状块；state×polarity×sync替代原book.payload形成codes；margin1 | 原样复用这套后续Wan编码、公开book和五mode接收器；非通用PRC/可扩展任意payload容量论证 |
| 旧Flow-Tube-State | step44/45/46各deficit*c*d/(3 norm²)，R/3逐步和累计R预算；47/48/49自由tail | 新仅49完整deficit*c*d/norm²，无/3、预算分配、自由tail或新强度 |
| VelocityDirection/Calibration/ClippedMargin | 44..46系数、44..49真终态图/强度开发；旧NO_SELECTION等失败保留 | 不复做终态反向/强度扫描；当前是原闭式margin算法的末步实现桥接，非新梯度算法 |

旧session把SyncTube描述为可工作的显式同步子系统；恢复ZIP的具体编码/信息依赖以上述源码审计限定。该历史不能证明本候选完整同步、严格盲FPR或Flow trajectory成功。未在旧Flow分支runtime/experiments找到完全相同的1760 LAST49+TERMINAL媒体验证；没有据此声称所有历史仓库穷尽。

固定方案：4case×OFF/TERMINAL_A/B/LAST_A/B=20视频。共享0..48与同step49 CFG/历史。OFF=原生final(z,v)。TERMINAL_m=P_m(OFF)，LAST_m=原生final(z,v−u/σ49)，u=P_m(z−σ49v)−(z−σ49v)。carrier.write保持原float64块投影、实际norm²与FP32落盘。解析上u_b=relu(1−c_b p_b)c_b d_b/||d_b||²；对mean hinge²损失它等于−N/(2||d_b||²)预条件梯度，不是固定eta=.1。FP32减加的误差保留。

native要求flow_prediction/predict_x0、thresholding false、solver_p None、50步、lower_order_final true且sigma_next=0；最后native order1。逐项记录OFF−clean49、LAST−P(clean49)、LAST−TERMINAL maxabs和固定容差ATOL2e-5/RTOL2e-4，实际signed投影和margin另保存。不要求bitwise相等、不以margin达标代替视频成功。不调用旧controlled_step（其隐含预算上限）。若两写法等价，结果只证明实现/调度器桥接，不是新的生成约束优势。

每视频VAEdecode→H264 CRF18/yuv420p MP4→真实RGB回读→4origin VAE重编码→原state_clock.read(obs,book)。g0使用181帧；g1/2/3使用177帧，各origin尾部不足丢弃并记录；不是latent裁剪替代。四origin属于接收器内部相位假设，不等价真实未知偏移裁剪攻击。下一阶段未知偏移/裁剪仅列为计划，本轮无攻击执行。

读取器不接收写端projection、sample_id、消息truth或attack/reference元数据。原五mode全量报告，每组8marked固定分母；truth只read后report。OFF仅排序不算FPR，编码内容非任意用户消息容量实验。媒体不由terminal筛选。失败/缺失20视频、80origin槽保留。

每case100Transformer（prefix98+共享last2）、52native步骤（prefix49+OFF/LASTA/LASTB），全轮400/208，20decode/20save/80encode。生成与媒体独立子进程，不共驻liveTransformer/VAE。保留native source/config/initial/prefix/history/velocity/codebook/终态/媒体/观测及哈希。

质量为RGB MSE/PSNR、时域差分能量比、残差时域MSE；不沿用旧1.5门槛，不把PSNR当感知成功。需用户人工审阅提示词一致、伪影与连续性。Notebook沿用已工作torch2.11cu128/diffusers.40依赖，独立Drive首cell、SOURCE_COMMIT待发布绑定、Run all固定入口，无参数菜单。
