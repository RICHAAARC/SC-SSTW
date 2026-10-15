# 新四源 RAW_U 实际载荷读取

这项已采用的对照固定读取四个新源的 CLEAN_MP4 和 DELETE_G0，共 8 行、256 位。输入是同次运行保存的视频及已保存的 RAW_U 操作。每个操作实际构造自己的 RGB 输入，调用 Wan VAE 编码，再调用原载荷读取器。已有 CENTERED_U 读取只用于最后配对报告。

运行代码、实际 notebook、源码 ZIP 和定向 CPU/fake 检查已完成，R2 获 A2/A3 独审及主审 A5 本地交付接受。当前交付从现有 dev 分支直接下载源码；真实 Wan/GPU/Colab 尚未执行，RAW 误码栏保持未知。

## 直接运行

1. 打开 [直接运行的 Colab notebook](https://colab.research.google.com/github/RICHAAARC/SC-SSTW/blob/dev/paper-results-v1/notebooks/paper_results_v1_raw8_payload_v1_colab.ipynb)，选择可运行 Wan VAE 的 GPU，执行 Run all。
2. 第一格挂载原有 Drive。随后自动从当前 dev 分支下载[普通 companion ZIP](https://raw.githubusercontent.com/RICHAAARC/SC-SSTW/dev/paper-results-v1/notebooks/paper_results_v1_raw8_payload_v1_companion.zip)，无需手动上传。每次 Run all 加载可用源码，已完成 RAW 单元按原状态恢复。
3. notebook 读取原有 `Paper-Results-V1-Independent-Four-V1/resumable-independent-four-v1`。结果写入 `Video-WM/Paper-Results-V1-RAW8-Payload-V1/resumable-raw8-payload-v1`，不会重跑源视频生成。
4. 断开后重新挂载同一 Drive 并 Run all，读取已完成的 RAW 记录或 RAW latent，从未完成单元继续。最后提供整个输出目录，至少包含 `summary.json`、`evaluation_report.json`、`raw8_state.json`、`saved_raw_operations.json`、`raw_reads/` 和执行日志。

源码链接使用现有 `dev/paper-results-v1` 分支，可用源码直接加载，没有摘要、固定版本、Git 状态或 GPU 型号门禁。下载/保存失败、坏 ZIP、不安全解压路径或实际导入失败会报告具体错误；完成的实验单元不会因源码加载而重算。

## 固定几何与成本

| 视图 | 源 | 收到帧数 | RAW 输出帧数 | 有效载荷时间位置 R |
|---|---|---:|---:|---:|
| CLEAN_MP4 | 01–04，各一行 | 181 | 181 | 44 |
| DELETE_G0 | 01–03，各一行 | 163 | 181 | 44 |
| DELETE_G0 | 04 | 163 | 149 | 37 |

无中断名义预算是 1 次 Wan VAE 加载、8 个不同 RGB 输入的 Wan encode、8 次原载荷读取，共 1416 个修正后输入帧、345 个载荷时间位置。R 相等不意味着输入相等，不共享 CENTERED_U latent。R 不是独立样本数。

已有 received RGB 可读时直接使用；否则最多解码四个已发布 candidate MP4，clean/delete 共用同源解码。0 生成、writer、framewise 模型、selector、新视频编码或新攻击。VAE 内部分块数及真实 GPU 时间/峰值显存未测；中断可能增加尝试，累计调用和本次调用分别记录。

## 结果含义

选择器结果和操作在读取真值前保存。真实 RAW 遍历终止后才加入固定目标和已有 centered bits。全部 8 行及失败、缺失、unsupported、unresolved 始终保留；只有 8 对都可评估时才计算总配对误差减少量。

- `CENTERED_REMOVES_RAW_ERRORS_THIS_BATCH`：RAW 有错、已有 CENTERED_U 全零，支持这批四源/两视图的载荷收益。
- `BOTH_READERS_ZERO_ERROR_PATH_IMPROVEMENT_ONLY`：两者都零，支持路径改善和载荷保持，不支持误码收益。
- `INCOMPLETE_FIXED_EIGHT_PAIR_COMPARISON`：保留缺失对，不将部分完成写成成功。
- 其他完整结果逐行报告，不自动升级结论，不改路径、消息、key、攻击或源样本重跑。

失败单元可以按原操作恢复技术执行。已完成记录不因结果不好而重算。原始 signed votes、零掩码及时间优先的 Counter 决策一起保存，精确票数平局沿用原读取器的首次遇到规则。

## 本地入口和检查

从 worktree 或无 `.git` 的源码目录执行：

```bash
python -m experiments.paper_results_v1.raw8_payload_v1_cli \
  --phase all --output /path/to/raw8-output \
  --source-run /path/to/resumable-independent-four-v1 \
  --model-path /path/to/readable-wan-model --device cuda
```

构建交付使用 `python -m experiments.paper_results_v1.build_raw8_payload_v1_colab`。定向检查使用 `python -m pytest tests/test_raw8_payload_v1.py -q`。13 项检查覆盖不同实际输入映射、149 帧分支、原票序、成功复跑零模型调用、失败/中断恢复、真实来源字段、MP4 fallback复用、来源索引缺失时已有RAW latent继续读取、unsupported/unresolved、缺失配对参考、ZIP 可移植及 notebook 的缺失输入路径。它们是 CPU/fake/静态工程验证，不是实际模型或云运行结果。

上述13项为R2运行包的已通过检查。免上传发布只复核受影响的源码下载、坏ZIP/不安全路径、实际notebook缺失输入Run all及无`.git` portable CLI，共4项；不重复方法全套检查。远端下载和导入可用性也不等于实际Colab模型运行。
