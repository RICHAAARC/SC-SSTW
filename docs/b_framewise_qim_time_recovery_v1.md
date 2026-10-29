# 剪辑后时间关系 CPU 验证

已采用2026-10-30规格的独立接收验证。实际入口为 `notebooks/b_framewise_qim_time_recovery_v1.ipynb`；builder同名，默认普通CPU Colab。依赖仅Python、NumPy、psutil及FFmpeg/ffprobe；builder另用nbformat。没有模型下载或GPU设置。

独立CLI（普通源码目录，无需`.git`、notebook、scripts或report）：
```
python -m experiments.wan_state_clock.b_framewise_qim_time_recovery_v1_run \
  --qim-media /path/to/saved/QIM.mp4 --base-media /path/to/original/BASE.mp4 \
  --output /path/to/new-time-recovery
```

QIM与BASE各执行FULL181、CROP[38,139)101、删除44为180、44后复制44为182、删除[43,46)178。每个视图均在既有媒体上再次H264编码，CRF18/yuv420p/8fps/320×512；包括FULL。预算encode10/probe12/decode12/时间读10/接收10/码表1；A及所有模型、writer、E41调用为0。固定1644帧、1035720位槽、8220lane槽、1634相邻关系；未执行保留，不补零错误。

接收器只取实际RGB/raw和公共协议，不接收arm/view、真实181终点、编辑映射或类别。每帧保留q/d0/d1/soft/mask并重新读取后才融合。两有效副本soft相加，精确零或全无效擦除；全公共域92672码字按lane，在1024块中检查2e+u≤14，不设seed、幅度门或top-K。非冲突标签给[j,j+4]交集，prefix下界/suffix上界精确表示全部非降兼容路径。冲突本身及相邻关系保持未决；首尾自由，包括startup歧义。至少两个j、两个lane的非冲突观测并有共同路径才有结构支撑。BASE支撑即有限负例错误接受。

独立输出先保存attempt和全计划；调用前started、成功后returned。中断/失败消耗，独立视图继续，其依赖保留未执行。再次CLI/Run all优先只读，不下载、安装、导入科学依赖或补领预算；缺报告缓存仍显示原始记录。预检、日志及cache同级，原输入目录只读。

码表300秒、接收每view600秒/总6300秒、真实进程与子进程RSS峰值8GiB。codec300秒，超时terminate等30秒后kill；超限记RESOURCE_UNRESOLVED，不降域或重试。原始盲输出全部终结后才打开攻击truth后评。局部位/标签、原始图、资格后对应与关系分列；细表留JSON，raw/fused留NPZ，报告紧凑。

完整/部分恢复、错归属、BASE错误接受、全拒及运行失败分别记录。原native精确落点失败与最优性未证保持。即使未来固定项全满足，也仅是同一来源固定剪辑、同二代H264时间关系；不证明一般null、跨内容/长视频、任意编辑或剪辑后CDC32身份。当前交付检查仅人工fixture、fake服务和CPU工程检查，未读取真实视频/raw、系数/features或执行真实FFmpeg媒体。
