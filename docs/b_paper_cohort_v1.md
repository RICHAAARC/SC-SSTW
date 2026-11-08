# Paper Cohort V1：固定 12 内容的 Colab 入口

默认 TEST-A 为 `paper-joint-t01..t04`，TEST-B 为 `t05..t12`。完整 prompt/seed 固化在 `main/tube_state/b_paper_cohort_v1.py`；两个批次均为正式 TEST，描述目标不控制后批执行。C4已完成工程审查，TEST-A四源也已产出真实结果；本次main选择性整合保留该证据，不重跑、不重置。TEST-B尚未执行。

打开 [完整 Colab notebook](https://colab.research.google.com/github/RICHAAARC/SC-SSTW/blob/main/notebooks/b_paper_cohort_v1.ipynb)，选择GPU并 Run all，保留默认 `BATCH = 'TEST-A'` 即运行首4源。首代码 cell 仅挂载 Drive；入口自动从 `main` 加载可用源码并准备依赖。也可通过 `SOURCE_DIRECTORY` 指向Colab内可读的完整源码。源码位置用于加载，不校验 SHA/manifest/精确版本/Git 状态。

真实入口文件为 `notebooks/b_paper_cohort_v1.ipynb`，由 `scripts/build_b_paper_cohort_v1_notebook.py` 同步生成。只有用户修改为固定 `TEST-B` 才运行后8源；TEST-A不会自动启动TEST-B。

同一完整源码内的命令行入口：

```text
python -m experiments.wan_state_clock.b_paper_cohort_v1_run --batch TEST-A
python -m experiments.wan_state_clock.b_paper_cohort_v1_run --batch TEST-B
```

这些命令会启动真实物理链，只应在用户授权的 Colab 运行。本地工程测试不执行它们的真实 batch。worker-stage 是同一次预留内的子进程入口，不是续跑功能。

每源普通 OFF 与 DCT2 P0 使用同一 context、initial clone 和 fresh scheduler；generation 两轨迹后释放。prepare 共享一个 VAE 完成 OFF 普通 D181 和 P0 一次 capture。NEW 复用该 FP32 capture 的原 CPU solver/head replay，identity 第三次 VAE load 全量执行 180 E41、180 project、360 key decode。原 codec/native-Y/BCH/F42/relative/CDC32/QMatch 参数保持，未调用旧 H137/RANK3R/PM，也未新增 W/q2。

每源固定 1114 方法操作＝113 time＋926 identity＋60 QMatch＋15 quality。共享40 identity final/1280槽；60声明最大1920比较。全部15盲时间终态与40套计划先保存，随后读位；全部40身份终态后计算声明，联合盲终态后才打开后评 truth/writer。OFF/P0自身空时间未决不叫成功拒绝。P_PAIRED单独借用NEW映射，只作身份保持；U_NEW保留原四phase收到时钟。

每源先持久预留 attempt。失败、中断或缺少结果均消耗；重复调用只读取已保存结果，不补物理或后评。普通来源失败后可继续固定名单中的未启动来源；显式取消会保存 `interrupted` 并停止当前批次启动后续来源。普通低覆盖/未决/错误不引发参数重试或增加来源；目标是描述统计，没有自动方法更改。单源记录不可读、JSON结构错误或后评摘要数值不合法时逐源报告并保留未知，其余来源继续展示，固定汇总分母不缩小。

正式结果存 `/content/drive/MyDrive/Video-WM_Result/main_method/paper_cohort_v1/<source>/attempt01`。永久保留首/二代媒体、native-Y原始readout/packets/relative、bank projection/keyed原始读量、引用计划、全部槽位和判词、后评、运行/失败回执。原权重/源码缓存在 `/content/video-wm-paper-cohort-v1`；每源另建只属于本次调用的 `paper-cohort-owned-*` scratch，包含临时terminal/FP32 features/preclip。时间阶段结束或失败后释放自有 scratch，再进入identity；不删除结果树或任何旧缓存。`scratch_release.json`记录已释放路径、文件字节和阶段边界样本，样本不冒充真实磁盘峰值；中断若来不及收尾，遗留临时目录不由重入清理。

原 solver 7200s/4000 iter/16GiB工作数组估算不变。单codec300s，hard/relative各600s、共享累计9450s及8GiB RSS；identity含闭账/后评的worker为18900s/8GiB。生成/capture/replay只记录真实wall/RSS/CUDA峰值，未新增整阶段wall门槛。生成阶段从模型加载前单独开始CUDA峰值范围，在OFF/P0返回、加载失败和进入VAE重置前保存 `generation_resources.json`；部分执行记录其完成状态，无CUDA时不填虚假零值。它与后续VAE峰值分开报告。JSON辅助I/O计数限定于新记录helper，不冒充全部tensor/array读写或OS调用。

后评保留NEW时间822节点/817边、正确FULL支撑与字面BER、错归属、60条件声明、质量15pair-view及成本。OFF无自身消息，其载荷正确性为不适用；OFF/P0未写时间，不把碰巧编号记成合法时间恢复。技术不可评槽为[0,1]，实际可评的方法擦除为[0,0]；B/U固定160槽形成每源Δ界，再按全部4/12源等权。收益界跨0即证据不足，不能把U失败当0。所有盲读/后评的缺失都保留，已消费但后评失败的来源只显示未知，不补后评。

本地人工测试覆盖接线、预算、隔离、失败语义及薄notebook结构；不能证明真实Wan数值表现、Colab资源可行性、画质、泛化、FPR或时间认证。本次集成仅产生工程证据，不把原开发分支的真实结果改称main重新验证。

## 分层与已有 TEST-A 证据

- `main/tube_state/`：纯方法与固定协议；不依赖runtime、实验、notebook或治理层。
- `runtime/wan/`：模型/媒体适配、记录与阶段执行。C4子进程直接调用 `runtime.wan.b_paper_cohort_worker_v1`，拆掉实验与notebook层后worker入口仍可用。
- `experiments/wan_state_clock/b_paper_cohort_v1_run.py`：批次/配置CLI；只保留这一正式C4入口，没有迁入旧诊断或r02实验CLI。
- notebook与builder：挂载、选择固定批次、调用bootstrap、展示；依赖准备复用已验证的实际导入修复与安全解压函数，放在runtime的独立helper中。

C4原开发版本的TEST-A只读核验：4/4源完整，4456/4456操作RETURNED；NEW节点3284/3288正确、4未决、0错误，关系3260/3268正确、8未决、0错误。B正确归属支撑640/640，U224/640；240声明为20 MATCH、60 MISMATCH、160 UNRESOLVED。OFF/P0的160未决来自空时间入口，不能称为成功拒绝；匹配也不证明完整时间认证或总体FPR。整合后没有重新执行这些科学计算。

源码定位：原交付开发分支 `4921b88568a23227cd4cc87079eca89395df4a42`；该定位仅用于解释原证据，不是运行门禁。正式结果目录、12源名单、逐源1114操作、固定分母和失败消费语义均不变。
