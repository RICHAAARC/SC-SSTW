# B固定检出／候选集定位与拒读（Card94）

这是已采纳固定2cal+2confirm协议的隔离用户运行包。默认receiver、原writer、gamma1.0、净32与A均不改。源码不依赖旧科学观测、历史terminal或Drive cache产生新实验；四个新source都还未生成。当前交付是工程候选，不是真实校准或科学结果。

## 固定运行

在源码目录运行：

```bash
python -m experiments.wan_state_clock.b_fixed_detection_confidence_v1_run
```

本地可用源码、不含.git的编辑副本也能运行；可用`--output`指定输出目录、`--config`指定实际可读配置。默认固定输出为`/content/drive/MyDrive/Video-WM/B-Fixed-Detection-Confidence-V1/fixed-2cal-2confirm-v1`。参数不提供阶段、扫描、恢复／重试或strict/override模式。实际文件／模型／数值错误直接保留，不设摘要、版本相等、Git-clean或GPU型号准入。

Notebook为`notebooks/b_fixed_detection_confidence_v1.ipynb`，builder为`scripts/build_b_fixed_detection_confidence_v1_notebook.py`。首代码cell仅mount Drive，之后优先使用`/content/SC-SSTW-b-fixed-detection-confidence-v1`的可用源码；无目录时从现有`dev/local-joint-state-payload-v1`分支clone。无需人工ZIP上传。可选`b_fixed_detection_confidence_v1_user_bundle.zip`是普通源码副本；Notebook没有内嵌源码字典。远端尚须后续获准发布此候选才含新入口；本轮不push。

任何attempt、临时执行记录或科学产物存在时，再次Run all只读原记录，不重新下载源码、安装依赖、加载模型或补算失败。该协议不做单元恢复：首次中断／失败耗用对应槽，根attempt也使整次后续只读。全局attempt和每slot STARTED在实际调用之前持久保存；独立receipts让进程被kill时仍可读到未返回调用，ledger固定保留所有未进入／失败行。确认不因校准失败换宿主。

## 数据与物理路径

配置明确列b93-c01/c02校准、b93-v01/v02确认的固定prompt/seed，来源均NOT_GENERATED；名单手选，不建立总体iid假设。每host同一初态、fresh scheduler各跑OFF/P0原生50步，200条件/无条件分支前向、100native更新；P0在25..49步的原四段8位local_delta，25次全局部FFT梯度，无denoiser/tail backward。OFF从不写payload/sync。PM仅在同P0 terminal上构造gamma1的176约束，native observe一次含7744个长度768点积，不重写消息。

每host生成pipeline1次（无VAE），完成两链并释放；随后FP32 VAE1次负责D3/E72。三臂各普通D181、preclip、clamp/round uint8、H.264 CRF18/yuv420p/8fps、probe与完整媒体decode。只从各自实际MP4 RGB cache取R45六起点[0,4,17,62,107,136]；每R四实际因果reset E41、仅读r1..10。24个reset origin每臂互异。每E一次FFT供两钥票，不跨臂／跨实际view复用。

完整四host预计：pipeline4，DiT800，native400，local FFT grad100，PMconstruct4／nativeQ30976；VAE4，D12，E288，receiver FFT288，MP4编码／probe／decode各12。writer FFT前向100及对应autograd100独立计数。每host Q/votes各144、36query×137=4932候选；共享条件归约4932、157824消息行、70560组位Counter；全预算另见`configs/b_fixed_detection_confidence_v1_budget.json`。次数是成功路径计划，不是实测收据或资源硬上限；内部hook报告实际encoder/decoder块数，未观测时保持null。

质量保留OFF→P0与P0→PM在preclip/rgb8/MP4三个层共6pairs，每pair181逐帧delta及180相邻残差，whole及outside nominal/outside union nominal分别保存。nominal mask不是VAE因果区域。不增D/E/codec或LPIPS，不新增质量阈值。

## 检出、定位与消息

H137为p0/s1..35及p1..3/s2..35；a_hat=4(s−1)−p。四ROI分别双中心化，F为ROI0/1的10行均值、V为ROI2/3；T=min(F,V)，M=max_H T。旧raw/centered pooled仅伴随记录，不据确认结果择优或改规则。所有exact ties保留，无epsilon。

完整保存两cal host全部盲Q、原位票、分数及候选归约和blind_complete，再监督打开arm/origin（不读消息选择），每host N为30个null query全H最大值、A为六PM正确钥真态gap最坏值。tau=max(0,N_c01,N_c02)，lambda=max(A_c01,A_c02)，各计算一次、数值持久冻结以后才进入确认任何物理sourceprep。calibration再使用不是独立性能结果。

M<=tau为REJECT_SYNC；M>tau暂检出；C={h:M−T(h)<=lambda}，边界保留，不再加T>tau筛C。singleton为暂定位置，多个C不声称确切源身份。组位只有全C都有完整名义支持且译值一致才条件读取，其余擦除。技术评分缺失／非有限不删候选救分；分数完整但票／归约失败保留M/ties/score-only数值，最终PAYLOAD_UNRESOLVED、无正式组位输出，不算成功拒绝。cal未完成则tau/lambda未定义、两确认物理槽保留NOT_ENTERED。

每PM正例host物理组消息存在80/缺失112，真态名义支持72、另8位存在但未被E41支持（a4完整cell12在观测之外）。确认两PM正确钥384槽=160存在+224缺失，名义144+未观测16。每host60名义cell视图只覆盖42种不同标记源cell，净容量仍32。cell45是guard，不能成为第5组消息；MIXED／PARTIAL／guard真值未定义保留，不能靠后看真值擦掉错误。

## 保存顺序与评价

writer必须在生成时知道消息；这不等于receiver或后评价器可提前读取。数值函数只接自身E/公开key/Q/票/表，不接arm/origin/真消息。preparation与writer trace在private/preparation路径，未传入数值函数。calibration是显式监督阶段，只使用新cal的角色／真态几何，不读消息定阈值。

确认所有盲family和正式decision保存后写`markers/confirmation_blind_complete.json`，之后首次将writer bits传给post evaluator，并留`post_truth_access.json`。时间标记记录进程内执行顺序，非不可篡改证明。

后评保留每query32输出槽、137候选、44个源位置与技术失败：同步null误检、单例误接受、真态被C排除、错误phase/归位、源假覆盖、MIXED支持未定义、组消息物理支持／名义支持／读擦、缺失组假读分别计数。OFF无嵌入目标，正式消息读出一律OFF_NO_TARGET_OUTPUT，偶合writer bits不算恢复，BER未定义；P0/key00有payload但sync null，可另列真实载荷对照；P0/PM错钥读出为WRONG_KEY_ATTRIBUTION，不因偶合而称归因成功。全部三臂双钥合并的1920/2688只是几何support槽，不是全部实际存在嵌入消息。

同证据旧强取min作为未认证对照，报告共同可读BER、正确保留/损失/新增及错误／假覆盖变化，跨phase不能按相同local_row做源身份差。旧MIXED则确切源覆盖差为null。另为同host同origin P0/key00→PM/key00保存同**真实phase**原位80-bit共同frame-support BER、保留/损失/新增与gated组位比较；不以选中位置的消息差代替真实物理载荷退步。该配对只读已存原位位，无重新FFT或Counter。

全拒读／全137候选集合擦除不算成功；统计population FPR、固定校准条件保证、认证位置、科学PASS保持null。手选2+2只能支持名单内经验可证伪结论，不能将相关裁剪/keys/137候选当独立样本。连续裁剪六点不能扩大为内部删帧、重复、变速或任意位置已确认。

## 本轮验证边界

仅运行`tests/test_b_fixed_detection_confidence_v1.py`的tiny/fake CPU、序列化/阶段/失败/预算接线与Notebook结构/源码ZIP检查。没有启动真实sourceprep、校准、确认、模型/VAE/FFT/codec/GPU/Drive，也没有回放旧观测、重算原票或真实tau/lambda。Notebook实际Run all仍待用户在发布后的包运行。工程通过不代表科学成功。
