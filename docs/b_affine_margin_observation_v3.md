# V3 s01整链观察运行包：授权与实现清单

2026-10-23用户已授权准备runner/Notebook/builder及必要无模型CPU核验；真实Run all由用户运行。固定预选s01、P0/OLD/NEW、key00/key01、O33/crop104与delete13 seam52、3window×8phase。预算：一次CPU构造，VAE1，D181×3，E181×3+E45×144=147，FFT147、读取294，codec encode/probe/decode各9。sourceprep/DiT/旧梯度0。用户后续已明确授权将本包10个文件提交并push到现有dev/local-joint-state-payload-v1；不合并main，不自动执行GPU/模型/FFT/codec/Colab/Drive。

接口：main/tube_state/b_affine_margin_v3.py只含单次native构造与收到端全bank估计/扣候选c_hat的共同支撑；runtime/wan/b_affine_margin_observation_v3.py负责一次尝试账本及物理编排；独立backend接现有Wan接口；post evaluator只能在blind矩阵、H/c_hat/row支撑及payload引用保存回读后读取protocol/private truth。CLI和Notebook调用同一runner。

先持久化attempt/STARTED再构造与物理调用。失败/中断消费尝试，后续Run all只读，缺失/未决留固定槽；无重求、重试、host替换或P0 fallback。NEW未保存过，不伪称可直接复用。固定max-t、原per(g,c) energy/peak与precast/actual FP32验收；不调参/加幅/扫描/放rho。

实际仅s01；历史4host/84unit、s02/s03未决为背景记录，不能写成另外host已物理执行。toy tau/rho仅固定描述性诊断参考，真实MODEL_DOMAIN_UNVALIDATED，无认证/PASS。Notebook直接获取GitHub可用分支源码或已放置的可编辑目录；无B64/大内嵌/摘要/manifest/版本/Git干净门禁。

## 用户运行入口与源码状态

实际Notebook：notebooks/b_affine_margin_observation_v3.ipynb；builder：scripts/build_b_affine_margin_observation_v3_notebook.py；CLI：python -m experiments.wan_state_clock.b_affine_margin_observation_v3_run。Notebook直接获取RICHAAARC/SC-SSTW的dev/local-joint-state-payload-v1可用源码，Run all执行固定任务，无需手动上传ZIP；实际下载或入口缺失会直接报告。也可用可编辑源码目录，不要求.git。

Colab：https://colab.research.google.com/github/RICHAAARC/SC-SSTW/blob/dev/local-joint-state-payload-v1/notebooks/b_affine_margin_observation_v3.ipynb

P0默认是B-Background-Normalized-Match-V1/fixed-four-unseen-v1/hosts/b101-s01/private/P0_terminal.pt；OLD默认是B-Lowrate-Local-Time-Label-V1/fixed-viewed-development-v1/native/b101-s01/TAG.pt；均在MyDrive/Video-WM下。truth默认同P0 source的protocol.json，仅post读取config.writer.bits。输入只能检查实际可读、shape/dtype/finite及原保护支持/预算，不要求摘要/固定源码身份。

输出固定MyDrive/Video-WM/B-Affine-Margin-Observation-V3/fixed-s01-once-v1。attempt记录创建后，任何后续Run all直接只读；STARTED代表完成情况可能未知，不能自动重求/补物理。中断通知同worker进程组，但不声称已证明外部所有任务停止。setup仅处理真实缺依赖/导入故障，不重试科学调用。

## 同一方法与物理层范围

native每role全21g满足r_g=aB_g+c，最大化t且a>=t*.15*sbar/sqrtN，在原per(g,c) energy/terminal峰及预先浮点reserve内单次CLARABEL。新收据保存primal/逐约束slack以定位此次未决，不改历史缺记录。任何precast/actual未决不缩放不重求；仅label的remaining_hard_error_budget有RM-r7语义，pilot置None。原A、净32位ch0..3和保护时间均不变。

E181三次只读取full MP4 readback；未独立对uncompressed RGB8作fullE。每臂full MP4经crop/delete104后，再各编码/探测/解码一次，local144是该真实received MP4/readback上3window×8phase的reset E45。不能声称已将VAE、codec、reset全部因果拆开。FFT每E一次，key00/key01共用；没有oracle新FFT。质量只做同runtime P0→OLD/NEW full/readback和两received/readback六记录（778帧/772相邻对），clipping三臂另报。RGB8/preclip/floatRGB保存，不冒称它们各自有独立E181测量。

## 盲保存、失败与后评

raw/和payload/含固定全部147view×2key的原始48维和投票，失败/未进入明确TECHNICAL_UNRESOLVED并绑定journal实际status/receipt；不是以物理成功作为全局后评门。blind/每query保存全部161994分数/增益/c_hat、所有H、候选校正pilot/label/pure IDs、96行支撑、冻结sourcecell/ref及32位U/E；H0明确NO_COMPATIBLE_REFERENCE。全部文件实际回读、必要方法几何/状态一致后才markers/blind_complete，再开放truth。同源P0不参与NEW的盲决定。

post补U/E/O_geom/O_admitted固定各384bit槽；分别报告物理缺失、无完整观测、技术未读、无共同来源、无相容参考和读擦。post_true_path_corrected_rows保留H空时真θc_hat下的pilot/label/64码距，post_full_readback_corrected_rows是full公共g位置的校正描述，均不能改变盲输出或宣称实际认证。full payload正确key比较64 group-mode bit槽/704cell-mode bit槽，wrong-key另列；local原始比较8448bit槽/钥。原32位退步/common-bit BER、coverage及false attribution（含同bit异source）分列。当前几何整payload组完全无source-frame仍未测试。
