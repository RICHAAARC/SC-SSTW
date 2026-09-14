# 单 Y_MINUS 最小RGB能量 writer 试验

单次新编码，旧 phase1_cpu_run01/JumpingJack 的 source_sampled_rgb.npy、旧Y_MINUS preencode/decoded、双OFF decoded及旧result.aggregate.mp4公共C全部只读复用；不重估C、不重编码旧臂。

w=(.2126,.7152,.0722)，deltaRGB=−(2/255) hy w/(w·w)，clip[0,1]后round uint8。理想未clip亮度增量仍−a hy，RGB均方能量比等通道writer为1/(3 w·w)，仅构造代数。固定16×240×320RGB、8fps、libx264 CRF18 yuv420p线程1。

两OFF的全向量 ||Fnew−Foff+C_y||≤a/4 与逐帧sourceRMSE≤3/255在同一至少15/16时刻同时成立；另整片sourceRMSE≤3/255硬AND。只叫single_arm_gate，不是Phase1 PASS。source是旧扰动/编码前的float[0,1]采样RGB，不能用OFF扣除编码损失替代总预算。保存W=P−S、K=D−P的逐帧及全片平方项和2mean(WK)，编码前/后F及clip原始记录，旧新逐帧比较。

一次尝试、仅一个encoder调用，父进程120秒硬截止含子进程，各命令60秒；失败和timeout保留16固定slots，新目录及同名日志/退出记录存在即拒绝。无重试、模型或GPU。当前实现者只做小数组CPU测试，真实编码由已授权主任务唯一执行。

入口：PYTHONPATH=. python -m experiments.public_statistic.minimum_rgb_yminus --old OLD_RUN --output NEW_RUN
