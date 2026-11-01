# c02 固定9raw CPU分层诊断
一次 Run all 读取既有五view received raw 与 QIM preclip/clip/rgb8/h264 四层 raw，不重生媒体。CPU，仅 numpy/psutil；不加载torch/模型/GPU/codec，不重读payload。源码普通下载或可编辑目录，无摘要/版本门禁。

固定67逻辑项：raw_load9、saved_view_load5、codebook1、arithmetic9、fusion9、compatibility27、path5、blind_save1、post1。接收822位置/817边/4110lane/517860原始bit；辅助724位置/3620lane/456120原始bit，无图。总1546位置/7730lane/973980原始bit，F/R0/R1共23190lane-role、1460970bit决策槽；兼容比较上界429812736不是物理采样数。内部复用fuse可能重复加法，逻辑fusion9不是浮点操作计数。

原FP64低tie算式只复算并对比保存量，本轮三角色兼容诊断仍用保存soft/mask，全扫公域92672、lane模5、2e+u<=14、block1024。只对五接收F运行原interval_paths；不改H/L，不重算C/L/G，不选好副本或阈值。四层不是可选追加，也不是H接受门。

所有盲派生和失败账本先保存，唯一post才读原time_evaluation的truth。启动未写lane保持UNWRITTEN。全817边检查精确端点和缺失/重复索引；真假GAP/REPEAT均保留。缺层的首见/转变为UNKNOWN；preclip已错不证明唯一物理根因。旧H/原融合L、B160及native/solver失败不被诊断覆盖。

首计算前消费固定attempt，失败/中断不补跑，再Run all只读本次记录、不重post。RSS8GiB、总墙钟6300秒（含post/主要保存）、码本300秒、每raw600秒；阻塞调用只能返回后观察超限，不保证硬中断。全局资源停止不暗算剩余项，局部失败仍保留分母并可继续独立项。

关闭资源监视器的迟报错误也进入总状态。恢复保留STARTED中已知started/returned回执计数，未持久返回标为UNFINISHED_RECEIPT；缺失/非法回执为UNRESOLVED_RECEIPT。默认摘要显示已保存post/进度中的诊断统计、读盘错误和67项状态计数，失败不抹掉已知发现。

默认仅打印紧凑摘要；逐bit数组、全部hit/距离、旧/重算路径、完整post逐行记录在OUTPUT。本地工程交付不等于已完成真实诊断；发布后由用户自行运行。完整writer/reader92672码字核对仅交付前一次工程检查，不是用户第68项。
