# B P0：保存全尺寸输入的两分支 VJP

2026-10-16；只采纳 P0。此包准备用户运行，本地检查不证明真实资源可行性。P48/P49、writer、质量门槛不在本轮。

固定 `CONTROL_TAIL_after48` FP32 输入，使用其原 BF16 转换、保存 prompt/negative 与 index49 timestep。读取原 `units/gradient.pt[direction]` 为**未缩放的输出测试 cotangent d**；保存的 d 所属 input_terminal 必须数值等于原 ZERO，旧 step49 prediction 的 input 必须数值等于 CONTROL_TAIL_after48。没有差分方向代理、重新编码或新模型基线。输入 finite/shape/support/unit norm 与实际输入点检查服务于方法语义；来源版本/schema/status 标签不作为准入条件。

每分支从独立 FP32 leaf 经过原 BF16 cast、原 Transformer、FP32 output，以 `autograd.grad(output, input, grad_outputs=d)` 求输入 cotangent。保留 native attention processor/参数；`from_pretrained(torch_dtype=bfloat16)` 的原生 keep-in-FP32 层仍保留，不强制重铸。模型 eval、参数 frozen，不求参数梯度。

两个分支串行，使用原 Transformer 的 block checkpoint 接口，non-reentrant、preserve_rng_state=True。每block至多一次重算，外层 saved-tensor hooks 用已跑通的边界 storage/view 存盘机制；16MiB分块传输，释放 pack dedup 的原GPU storage引用，VJP结束/异常后释放独立spool。当前活block、非block激活和workspace仍可能超出资源；没有显存可行性保证，也不靠empty_cache或换精度/kernel重试。

预算为1 preparation、2 forwards、2输入VJP、60原始block进入、≤60block重算。0 native/VAEload/decode/VAE-VJP/writer/newvideo/encode/codec。每个调用进入前保存RUNNING；成功返回/失败/未知分别保留。每block记录原始与recompute状态；early-stop重算属于完成所需重算，不冒充又完成一个顶层forward。GPU阶段allocated/reserved及峰值、.5秒采样host RSS峰值、process RSS高水位、spool峰值和读写/传输分开记录。临时spool数量不是显存；GPU资源值本地未实测。

固定分母只有conditional/unconditional两行、两个forward和两个VJP单元；不建立接收器读出/视频窗口分母。输出保存为各分支`output.pt`，VJP保存为`input_cotangent.pt`，包括零和非有限值。`forward.json`与`vjp.json`区分forward-only；`result.json`、`report.json`和`calls/`保存全部失败/缺失。

原始失败或中断不重试。1 preparation预算意味着跨进程不能重建丢失活图，也不能加载第二个模型来补做未进入分支。相同输出目录再次Run all只读取已完成单元和恢复报告；消耗过预算的未完成部分保持FAILED/UNKNOWN/MISSING。读不到已保存单元会报告具体文件错误，不暗中重算。要增加一次真实尝试属于新物理预算决定。

同进程的普通单分支工程错误只终结该分支及依赖VJP，释放后继续另一个预定独立分支，仍不超过固定预算。按已采用P0源定义，OOM、任何非有限forward/VJP或进程中断则停止后续分支、保留MISSING，不变更精度/内核重试。完成/失败恢复会核验已保存input、exact d、forward及cotangent的实际形状/dtype/数值；非有限cotangent保留NONFINITE，零cotangent是有效零观测。

保存context的dtype标签缺失或与旧声明不同只作定位记录；实际prompt/negative必须是BF16，模型普通参数路径必须是BF16，原生keep-in-FP32例外可保持FP32。输入和输出也直接检查实际dtype，不以标签替代数值检查。

旧prediction比较输出exact-equal、差L2/max/relative L2，**不设数值相等门槛**。跨preparation比较不能单独归因于gradmode；有限或非零VJP不证明实际离散映射导数、可写关系或writer成功。零VJP也是完整观测，不自动重跑。保留完整结果，不生成科学PASS。

## 用户运行

打开 [P0 Colab](https://colab.research.google.com/github/RICHAAARC/SC-SSTW/blob/dev/local-joint-state-payload-v1/notebooks/local_joint_p0_input_vjp_v1_colab.ipynb)，选择可用GPU后 Run all。Notebook首格仅挂载 Drive，随后从现有 dev 分支自动下载普通源码 ZIP；无需手动上传。真实GPU实验由用户运行。

固定输入来自既有两个任务：

- `MyDrive/Video-WM/Local-Joint-State-Payload-V1-Paired-Transfer/resumable-paired-transfer-v1/run`
- `MyDrive/Video-WM/Local-Joint-State-Payload-V1-Fixed-Paired-Tail/resumable-fixed-paired-tail-v1/run`

固定输出：`MyDrive/Video-WM/Local-Joint-State-Payload-V1-P0-Input-VJP/resumable-p0-input-vjp-v1/`。请交回整个目录，包括`run/`、`notebook_attempts/`和失败记录。

独立CLI从普通解压源码目录使用：

```bash
python -m experiments.wan_state_clock.local_joint_p0_input_vjp_v1_run \
  --config experiments/wan_state_clock/configs/local_joint_p0_input_vjp_v1.json \
  --output /path/to/new-p0-output
```

非Colab环境可按实际挂载位置修改输入/output/spool路径；固定方法不变。CLI不依赖notebook、Git、manifest或固定版本。不需要模型时，完整/失败任务报告恢复不加载diffusers模型或要求GPU。
