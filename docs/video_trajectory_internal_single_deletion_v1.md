# 固定内部单帧删除诊断

本入口采用 proposal-v1，仅交付用户手动 Run-all。固定复用已接受的 1B M05 FULL181 MP4 读回 RGB，SHA `db63d12e3a767c78d93e25ebf420222d4d96cdb5e41c8b607f34a6f82b12a5bd`。不生成、写水印、读历史 latent、再 codec 或增加质量臂；fresh receiver latent 仅由收到 RGB 新编码产生。该来源已用于开发，本轮不是独立留出。

共同范围 X[2:180) 含 178 帧。C 删除尾帧成为 X[2:179)；D 删除源帧 90，成为 X[2:90)+X[91:180)，均为 177 帧。D 相比 C 缺少源 90、增加源 179，不隐称同长序列直接删一帧。准备配置可早读这些构造索引；blind score、selector、plan 和校正仅使用收到 RGB、公开 roster 及本 key 估计。

| 层 | 固定规则 / 分母 |
|---|---|
| 时间证据 | 每观察一个 fresh framewise encode；两观察、两 key，共 4 reads。缓存 r∈[0,176]、d∈[0,4] 的原完整方向 age 切片 numerator/rho，885/read、共 3540。普通帧合 160 patch rho=40；源 180 的末块 rho=160，不逐帧重归一 |
| 全局与单跳路径 | H0: m(r)=b+r，b=0…4；H1: m(r)=b+r+1[r≥k]，b=0…3、k=1…176。每读 709 路径，共 2836；H0 与原 score 代数等价。分别选完整有限 H0 与联合集合的唯一最大值，沿用 tie_atol=1e-12；tie/缺失/非有限无 fallback |
| 局部展示 | 公开接收网格 44 个 4 帧 bin 加尾 1 帧，每 bin 五个 d，共 900 格；保存完整 top/tie，仅展示，不选路径，不等于 Wan 独立感受野 |
| 接收臂 | RAW / GLOBAL_ALIGN / PATH_ALIGN 为 12 个 blind 逻辑槽；TRUTH_PATH 为封存后的 4 个 oracle 槽。全部 R44，共 675840 票、22528 time-bit、512 final-bit；失败保留 |
| 调用预算 | FULL 读取 1、CPU 构造 2；framewise load 1、encode 2 / 354 帧 / batch8 为 46 批，sync 4。Wan load 1；blind 最多 10 encode / 12 read，含 oracle 总计最多 12 encode / 16 read。完整映射及 key 决定实际去重，aliases 不增证据 |

GLOBAL_ALIGN 按 H0 的 p=b mod4 前插首帧、截至 177。PATH_ALIGN 的 H1 先在 k 复制前一收到帧作为占位，再 prepend p 并截至 177；不还原缺失 RGB，不分段 VAE，不用真值重置 VAE 状态。原每输入编码前后清 cache、FP32 posterior mode 保留。每 mode 分列估计的 nominal map、实际 output→received map、synthetic 与 dropped；RAW 无 source nominal map。真 source map 只在 posthoc 联结。C/D 真路径校正后的 source 索引仅 output90 不同（90 与复制的 89），这不意味着 Wan 影响只限一个 4 帧块。

顺序为 fresh sync → blind sync seal → blind plan seal → 所有 blind payload reads/seal → oracle-map 配置语义解析/plan → oracle reads/seal → message/posthoc 配置语义解析。配置身份 hash 不等于 receiver 消费真值；oracle/posthoc 由 public 配置 SHA 绑定。源构造真值合法早读，但不传给盲选择器。CPU latent 按同一观察的完整输入 index-map 缓存；跨 key 仍分别 read。已失败的 encode/read 不因 oracle 重试；成功 encode 的新 key 可封后补一次 read。已有 blind 文件不改写。

Posthoc 分别报告 family/b、177 帧对应正确数与误差、精确 k 及 signed/absolute k 误差；C 的 k 为 NA。每 mode 保留完整 final-bit 和 time-bit、通道统计、K1 角色、零系数/平票、相对 RAW 的描述性差分。名义 receiver index 不视为匹配源感受野。H1 较多自由度、邻 k 仅差一帧，联合得分较高或 payload 全对都不等于缺口检测/精确定位成立；C 选 H1 仅称本输入假跳变，不称 FPR。没有存在阈值、科学 PASS 或按 BER 选路。

运行入口：`python -m experiments.wan_state_clock.video_trajectory_internal_single_deletion_v1_run --output <new-dir>`；Notebook：`notebooks/video_trajectory_internal_single_deletion_v1_colab.ipynb`。草稿 SOURCE_SHA=None，在输出目录创建前停止；根发布已审 source S 后才绑定。只做本地 CPU/fake/static 验证，不执行真实模型或 Notebook。