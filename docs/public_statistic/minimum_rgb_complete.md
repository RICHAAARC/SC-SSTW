# 完整 minimum-RGB Phase1：固定7新＋5复用

新编码仅：P50 X_PLUS/X_MINUS/Y_PLUS/Y_MINUS，以及JumpingJack X_PLUS/X_MINUS/Y_PLUS，共112新帧、7MP4。复用原run01两内容双OFF，以及已见单Y_MINUS试验的Jump Y_MINUS，共5臂80行。完整12臂192行均保留；5复用明确源路径，不复制旧媒体冒新实验，单Y_MINUS是已见开发数据。

每源固定原source_sampled_rgb.npy全部16帧及原尺寸；delta=sign*(2/255)*h_axis*w/(w·w)，w=(.2126,.7152,.0722)，clip后rounduint8。reader与libx264 CRF18 yuv420p、8fps、线程1不变。入口验证复用OFF preencode与source量化相同、复用Y_MINUS preencode与指定公式相同，缺失/不一致保留失败而不换源。

主方法门只调用冻结v2 aggregate，以完整32个D形成新公共C，逐内容共同15/16及全部12臂总source质量规则不变。并列诊断原run01旧公共C双OFF兼容误差、逐时刻mask和实际旧等RGB→新minimum-RGB响应变化，不与主门偷偷AND；旧C不兼容如实报告。源/双OFF RGB误差另报；保存W=P−S、K=D−P及2mean(WK)逐帧全片分解。

一次尝试、7个encoder调用上限、不默认重试或扫参。父600秒总墙含子进程，每命令120秒，失败保留固定192slots及112new分母。新目录/同名日志已存在拒绝。CPU准备仅小数组/数学检查，不真实编码；主任务已授权唯一执行者运行本轮。无GPU/模型/commit/push，原reader、v2及单Y_MINUS入口不修改。

入口：PYTHONPATH=. python -m experiments.public_statistic.minimum_rgb_complete --old OLD --single SINGLE_Y_MINUS --config configs/public_luma_phase1.json --output NEW
