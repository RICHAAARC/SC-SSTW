"""Build and statically validate the actual user-run v4 notebook."""
import ast
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def build(root=ROOT):
    cells=[]
    def code(text):
        ast.parse(text);cells.append(dict(cell_type='code',id='code-'+str(len(cells)),metadata={},execution_count=None,outputs=[],source=text.splitlines(keepends=True)))
    def note(text):cells.append(dict(cell_type='markdown',id='note-'+str(len(cells)),metadata={},source=text.splitlines(keepends=True)))
    code("from google.colab import drive\ndrive.mount('/content/drive')\n")
    note('# 四配置公开视频 Baseline v4\n\n选择 GPU 后 Run all。固定 UCF101 DEV64 / CAL500 / TEST500，181 连续真实帧、25fps、320×512；40 视图。WAM32、PixelSeal256、VideoSeal v1.0 256 和 ADAPTED PixelSeal-FrameIndex256（248 owner＋8 index）分行报告。原生三个方法的时间列为 N/A。\n\n0.1% 是逐方法、单随机 query、固定40视图族的名义上界，不是 TEST500 对自然总体的经验认证；0/500 单侧95%上界约0.597%。这批不生成 Wan，也不代表主方法公平排名已经完成。\n')
    code("SOURCE_REF = 'dev/baselines-paper-v4'\nSOURCE_DIRECTORY = None  # 可选：可编辑的本地源码目录\nPREVIOUS_RUNTIME_STOPPED = False  # 仅旧的不同主机 runtime 已实际停止时设 True\n")
    note('## 执行与续接\n\n同一 Drive 结果包重开 Run all，从缺失/中断部分继续。启动只使用轻量完成记录定位工作，不全量复读旧原始数组，也不预读全部编辑计划；实际需要的批次数组才读取验证，最终统计仍依据真实读数。每次从分支下载新的临时源码，避免旧 project 副本遮蔽更新。旧临时源码仅在确认没有运行进程使用时清理，正式结果与恢复记录不动。先保存原生读数，再运行 query 验证及 CPU 后评。每实际批次数组即时保存；已有完整批次不重复推理。若同一 view 的接收像素缓存丢失，旧 partial 保留，该 view 在固定恢复预算内整份重建，不混合两份输入。每物理单位最多2次额外基础设施重算、各类累计最多基础上限2%；方法失败不换参数重跑。最多64次新 runtime 续接。\n\n同一包只运行一个实例。正常关闭后无需修改设置；若旧不同主机仍标记活动，请先实际停止它再确认续接。心跳不是跨主机锁，不按超时抢写。监视器只把本次 worker 的进度标为当前；旧行明确显示为历史，无变化会显示停留时长。旧任务仍在运行时先停止并断开旧运行时；新运行时续接时将 PREVIOUS_RUNTIME_STOPPED 设为 True，再 Run all。\n\n正式及失败结果位于 MyDrive/Video-WM_Result/baselines/public-native-and-frameindex-fpr001-v4/attempt01。源码、权重和衍生缓存位于其外。已有 Drive 空间未知，新增空间约200–260GB只是规划；DEV供体解码缓存约28.5GB、实际媒体大小另计，空间不足暂停未开始单元，保留正式包。没有预先测速，不承诺GPU小时或单次Colab完成。\n')
    code("from pathlib import Path\nimport urllib.request\nbootstrap_path = 'runtime/wan/native_baseline_colab_v4.py'\nif SOURCE_DIRECTORY:\n    bootstrap_text = (Path(SOURCE_DIRECTORY) / bootstrap_path).read_text()\nelse:\n    url = f'https://raw.githubusercontent.com/RICHAAARC/SC-SSTW/{SOURCE_REF}/{bootstrap_path}'\n    bootstrap_text = urllib.request.urlopen(url).read().decode('utf-8')\nentry = {'__name__': 'native_baseline_v4_colab_entry'}\nexec(compile(bootstrap_text, bootstrap_path, 'exec'), entry)\nRUN = entry['run_colab'](source_directory=SOURCE_DIRECTORY, source_ref=SOURCE_REF, previous_runtime_stopped=PREVIOUS_RUNTIME_STOPPED)\n")
    note('## 读取四类结果\n\nA：原生 payload BER、完整消息、coverage 与擦除；B：给定query验证、OFF FP/区间、wrong-query及固定DEV跨源读量复用；C：FrameIndex位置/邻边/错误归属与未决；D：precodec、G1、G2-identity画质和实际资源。全计划分母及失败不删除。partial WAM/FrameIndex阳性见证保留为TRUE但未完整；Seal缺一计划帧时视频均值验证为UNKNOWN。\n\n此 notebook 交付时未启动真实模型、codec、GPU或Colab实验。工程检查不等于科学效果。\n')
    code("print('入口状态:', RUN['status'])\nprint('保留结果:', RUN['output'])\nentry['show_results']()\n")
    notebook=dict(nbformat=4,nbformat_minor=5,metadata=dict(kernelspec=dict(name='python3',display_name='Python 3',language='python'),accelerator='GPU'),cells=cells)
    path=Path(root)/'notebooks/native_baselines_public_fpr001_v4.ipynb';path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(notebook,ensure_ascii=False,indent=1)+'\n',encoding='utf-8');return path


if __name__=='__main__':print(build())
