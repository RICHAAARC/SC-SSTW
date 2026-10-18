# Card51 固定局部载荷SECDED用户运行包

已按card50采纳规格实施，真实结果尚未产生。只准备包；本地验证限于合成CPU与静态检查。

方法：每窗4ROI池化，systematic缩短扩展Hamming(39,32,4)，22窗重复同一净32消息。原22pilot+32data方向、原data排列/掩码不变；同一756基排序追加7个parity方向。γ_s=1/√2，数据/校验γ_p=.5，无范数cap或补幅。J_ECC从保存的FP32 P计算连续61方向更新，最后一次FP32落地；冻结J0直接加载，绝不在J0上叠加。

选路仅22×22pilot，未知初态，22条+1模22路径，22项math.fsum等权和，精确并列保留且UNRESOLVED。先保存raw分数和selection，再按保存路径解扰39原始符号并保存，然后公共syndrome译码。单错规则只翻转一次并复核；双错/非法syndrome拒绝，任意擦除整窗32位擦除，不找最近码字或改选state。三错可误纠、四错可未检出；例如校验坐标1/2/4同时翻转会误纠原本正确的数据坐标7。校验一致不证明水印存在或真实性。

保留原32data软和聚合；另列ECC分支，用译码符号乘原abs(soft)，拒绝窗无证据，仍固定22分母。所有raw/ECC/aggregate输出先落盘，再读取writer真值评价。实际inferred-state读出与事后truth-state几何分列；P/J0的parity是未编码负控参考，不合成传输BER。报告局部修复/新增错误、误纠/未检出、共同覆盖BER与丢失覆盖，原GROW投票裕量/BER退步和全部缺失保留。

## 固定运行

三臂P→J0→J_ECC，唯一新候选；只读terminal和MP4潜变量端点，VAE load1/D3/E3。P/J0同场重放D2/E2是预算内的新成本。preclip/float/RGB8/MP4像素质量均保存，无额外E。

69核心槽：terminal load2、book2、construct1、VAE1、D3、E3、量化3、MP4写/probe/读各3、局部12、旧GROW FFT12、质量21。另304个CPU槽：message encode1、path12、ECC264、aggregate24、latent差值3。路径264/5808pilot项，syndrome最多528次。

12表/264状态/5544竞争/10296 raw符号观测槽（8448data+1848parity）/8448 ECC后局部消息位/每聚合分支384位/旧GROW384位；J_ECC的MP4正确key为22块、858raw符号、704净消息位。10296包含未编码及错钥对照，不是总传输BER。每项质量保留181帧、180相邻残差、名义支持外变化。三个实际FP32差J0−P、J_ECC−P、J_ECC−J0完整记录46时间片；同符号γ不意味着等扰动或免费校验。

质量记录另附`peaks`：逐帧whole/outside_nominal_support和相邻残差whole/outside_union_nominal_support的RMS、Linf各自峰值，明确给出frame或(frame_before,frame_after)。归约只读取已有标量行，不增加质量、模型或媒体调用。以精确`==`保留全部最大值并列，按帧/帧对升序排列，无epsilon或首遇挑选。未执行、失败或缺失指标行不生成伪峰值：值为null、位置为空，附UNAVAILABLE原因。Notebook摘要直接展示这8组峰值，完整逐行记录继续保留。

输入保持卡47同次：

- `MyDrive/Video-WM/B-Positive-Projection-Margin-V1/fixed-clean-pair-v1/features/PAYLOAD_ONLY/terminal.pt`
- `MyDrive/Video-WM/B-Positive-Projection-Margin-V1/fixed-clean-pair-v1/features/PROJECTION_MARGIN/terminal.pt`

输出：`MyDrive/Video-WM/B-Local-Payload-Protection-V1/fixed-clean-triple-v1`。保存三臂RGB张量约2.5GiB，另加latent、媒体和JSON。已有科学记录或临时残片即只读返回，不重跑未完成调用；不存在自动重试、调参、换源、加减幅或删输出重启。

## Colab入口与用户运行

notebook默认从现有dev/local-joint-state-payload-v1分支的普通raw.githubusercontent.com源码ZIP自动下载，安全解压后直接Run all，无需手动上传ZIP。真实模型、媒体与GPU验证由用户运行；CPU/静态检查不替代真实结果。第一代码单元独立mount Drive。SOURCE_LOCATION也接受可用普通源码目录/ZIP/URL；实际读取或导入失败直接报告，无摘要、manifest、Git状态、固定版本或完整性准入链。ZIP仍检查越界路径和符号链接。

现有尝试时notebook跳过源码、依赖和runner，标准库读取保存记录。首次依赖只按真实import失败定向补包；无GPU型号硬门槛。真实Run all只由用户启动，不属于本地CPU验收。

CLI在可编辑、无.git源码目录中同样可用：

```bash
python -m experiments.wan_state_clock.local_joint_payload_protection_v1_run
# 使用另一合法路径配置时：
python -m experiments.wan_state_clock.local_joint_payload_protection_v1_run --config /path/to/config.json
```

盲文件在blind/；原分数、路径、raw39、ECC及两聚合各有记录。core ledger和cpu ledger分开保存实际开始/返回/失败/中断与耗时。result/checkpoint保留分母、评价与三对比较；不存在科学PASS。源/模型路径用于定位，实际不可读、非FP32/非法几何、非有限值和模型/媒体失败按真实问题处理。

## 验证与边界

固定785个ECC CPU案例：1无错、39单错、741双错、3个高阶边界、1擦除；另有公共carrier前54精确保持/正交性、从P一次FP32构造、盲依赖/先落盘后评价、合成三臂69+304账、失败保留与不重跑、notebook和源码ZIP/无.git CLI检查。合成结果不证明新媒体每窗至多一错或实际画质可接受。

峰值摘要补丁另以小合成标量行验证8组峰值/精确并列、零峰值与不可用语义，以及原质量函数只调用一次；交付同步检查覆盖builder、真实notebook和ZIP。该窄补丁不重复785个已通过ECC案例。

不以704/704作为反复试验门槛；ECC恢复不抹去物理符号翻转。只支持固定clean终态方法诊断，不验证真实多步生成期联合写入、删帧/重复/裁剪盲同步或总体误报率。旧固定删除保存观测提案未实施，旧card47–49源码、数据与结论不覆盖。
