# Card99：绝对兼容集合四新来源用户运行包

当前：本地实现准备，未实际抽角色、生成来源或运行物理任务；尚未发布，不提供可用远端Colab链接。

## 运行入口

- Notebook：notebooks/b_absolute_confirmation_v1.ipynb
- 同步builder：scripts/build_b_absolute_confirmation_v1_notebook.py
- 普通源码ZIP：notebooks/b_absolute_confirmation_v1_user_bundle.zip（依赖闭包，无源码摘要/manifest）。
- 直接CLI：`python -m experiments.wan_state_clock.b_absolute_confirmation_v1_run`
- 固定Drive目录：`/content/drive/MyDrive/Video-WM/B-Absolute-Compatibility-V1/fixed-four-unseen-v1`

Notebook首代码单元只有Drive mount。后续直接clone可用GitHub dev/local-joint-state-payload-v1源码；候选未发布时缺入口会报告具体FileNotFoundError，不能把本地就绪当作远端可用。无需上传ZIP。构建ZIP和notebook不运行实验。

Notebook遇既存attempt/result/角色/收据等执行残留会跳过源码下载/依赖/角色/worker，只读自身记录。CLI也在配置读取和backend导入前判断既存attempt。失败或中断消耗本次，不能换目录救跑。无reset、strict或override，没有资源硬上限或摘要/版本门禁。

## 采用规格与接线

Card99以及Card97已采用SOURCES/BUDGET为权威；Card98 D1源码、结果冻结，不补救9位。四来源b97-s01..04、seed2026102201..04和完整prompt在configs/b_absolute_confirmation_v1.json。writer gamma1/净32/默认接收器/A不变。

未来首次用户Run all：先持久attempt，再从6个无序二元pair均匀一次抽校准角色并保存roles，之后才允许任何pipeline/source观察。准备测试只用明确固定的注入假pair，未调用真实随机抽样。角色失败/中断无重抽。

两个cal源采用同生成/D/codec/E41/Q/T，只保存137等支撑finite score，不需要payload/FFT或全候选组聚合；独立纯calibrator仅使用role/origin选择真态T。tau=max(0,N1,N2)、L=min(K1,K2)，无lambda、无D1阈值继承。阈值实际保存后才进入任何confirm sourceprep；失败则保留确认NOT_ENTERED。

确认真实E41每相位一次FFT供双钥原位票，保存signed_votes/zero_mask及每cell的原120票Counter结果。新读器直接复用Card98纯decision方程：C={T>=L}独立于检测M>tau；全C共同cell交集，每组最早j先定，再逐h读取自己的phase/r，逐bit可用且一致才读。空C/空交集/分歧是正常擦除，记录缺失是技术未决；不换j、不删C、不回退max。

B0仅对每query唯一max的一候选调用原票reduce_all，保持c/r/k encounter order、strict >0、Counter tie。至多72候选/1152组位Counter；不对137H全部归约，不用cell多数位重建组票。该一次归约同时产生ungated和同tau gated版本，两者都在truth前实际落盘。后评只重载二者评价。

确认72份新决定及72对B0输出全部成功保存后才写blind marker；再重读实际文件，之后才开放writer bits给后评。任一保存失败则不进入post。接收纯函数仅收score/已有raw bits/阈值，不接收源ID、真实origin或消息。配置writer bits只用于writer和最后post，不参与cal/C/j/一致性。

## 预算和固定分母

完整预算在configs/b_absolute_confirmation_v1_budget.json。完整路径：pipeline4、fresh50轨迹8、DiT800、native400、新host梯度100；writerFFT100/autograd100；PM4/704约束/native点积30976；VAE4/D12/E288、receiverFFT144；MP4 encode/probe/decode各12；Q576、T19728、receiverROI点积1013760。cal不FFT/votes，confirm原位Counter23040、B0组位Counter至多1152、新规则bit比较上界157824。上界用于操作预算，不是资源硬停机。质量24层配对、4344frame/4320adjacent及12preclip，复用同新三臂媒体。

确认72查询=60null+12正例，9864H/2304消息槽；正例384=物理有组160+缺失224，trueE41名义144/存在未观测16；120cell-view/960原位bit，84唯一sourcecell/672bit。全部失败/未进入/未决保留。共享cell支撑、实际单cell lookup/输出、点定位和消息覆盖分别报。未选原位单元保持未读，不把组位扩张成全组单元读出。

后评保留OFF、错钥、结构缺失假覆盖、unsupported source/wrong-group/cross-group undefined，不按真态入C过滤输出；每h实际条件几何都保存。备选窗口是互斥条件解释，不计作多条真实路径错误；不能隐藏它们只报真分支。物理P0-PM真phase配对与B0/new逻辑共同位BER、保留/损失/新增分开。零错误不能掩盖覆盖牺牲或未读槽。

有限四tuple随机角色不是从无限prompt总体抽样，不证明总体FPR或条件于accepted的风险；需完整role-invariant潜在观测才有有限名单秩解释。完整执行、全拒或全擦除均不自动是科学PASS。名义单cell不证明VAE/D/codec无跨组贡献。

## 分层与准备验证

main仅纯机制；runtime复用Card94成功物理适配并隔离编排；experiments只有CLI/config。没有改默认或旧D1。新builder与实际notebook同步，普通依赖源码包可在no-.git目录独立导入/CLI使用。

准备仅允许定向CPU/static/stub：score-only校准、随机角色持久化前置（注入假pair）、阈值保存前不得confirm、call ledger/失败、盲输出保存失败隔离post、既存attempt只读，以及notebook/包/预算接线。真实notebook Run all未执行，需后续用户在Colab实际运行；任何测试结果都不是新实验结果。
