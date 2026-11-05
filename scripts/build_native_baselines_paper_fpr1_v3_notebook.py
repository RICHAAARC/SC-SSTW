"""Build the actual thin v3 delivery notebook without executing it."""
import ast
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]


def build(root=ROOT):
    cells = []
    def code(s):
        ast.parse(s)
        cells.append(dict(cell_type='code', id=f'code-{len(cells)}', metadata={}, execution_count=None,
                          outputs=[], source=s.splitlines(keepends=True)))
    def note(s):
        cells.append(dict(cell_type='markdown', id=f'note-{len(cells)}', metadata={}, source=s.splitlines(keepends=True)))
    code("from google.colab import drive\ndrive.mount('/content/drive')\n")
    note('# WAM32 / PixelSeal256：正式自然视频载荷 baseline\n\n选择 GPU 后 Run all。实际 UCF101 名单为 500 个 CAL 来源组与 500 个 TEST 来源组；25fps、181 连续真实帧，中心8:5裁剪后缩放到320×512。2525组文件头盘点确认1145组合格，从中固定抽取1000组，覆盖73个动作（CAL 69、TEST 71）；这是已确认可读取长片子集，不是完整101类别或所有自然视频的无偏样本。头读取错误保留，真实解码失败不替补。每源11个时间条件×两代媒体。原生32/256位及空间支持不同；本轮先完成公开视频正式Baseline，Wan生成域同源比较另列。\n')
    code("SOURCE_REF = 'dev/local-joint-state-payload-v1'\nSOURCE_DIRECTORY = None  # 可选的本地可编辑源码目录\n")
    note('## 自动执行与续接\n\n入口自动下载真实来源、准备OFF、CAL定标、TEST盲读取及后评。每方法一条给定完整消息验证规则；目标为22视图source-family nominal 1% FPR，实际结果与区间另报。同一Drive重连后Run all只继续原计划中从未开始的单位，失败/中断单位不补跑。100个10源分片自动调度，请勿并发启动两个实例。不承诺单次Colab会话完成。\n')
    code("from pathlib import Path\nimport urllib.request\nbootstrap_path = 'runtime/wan/native_baseline_colab_v3.py'\nif SOURCE_DIRECTORY:\n    bootstrap_text = (Path(SOURCE_DIRECTORY) / bootstrap_path).read_text()\nelse:\n    url = f'https://raw.githubusercontent.com/RICHAAARC/SC-SSTW/{SOURCE_REF}/{bootstrap_path}'\n    bootstrap_text = urllib.request.urlopen(url).read().decode('utf-8')\nentry = {'__name__': 'native_baseline_v3_colab_entry'}\nexec(compile(bootstrap_text, bootstrap_path, 'exec'), entry)\nRUN = entry['run_colab'](source_directory=SOURCE_DIRECTORY, source_ref=SOURCE_REF)\n")
    note('## 读取结果\n\n结果固定保留在 MyDrive/Video-WM_Result/baselines/native-wam-pixelseal-paper-fpr1-v3/attempt01，失败包也保留。原生盲读先保存，再作单query验证与真值后评。CAL缺失时对应方法标CALIBRATION_INCOMPLETE，TEST原生读取仍执行；工程缺失不计零误报，正常WAM空mask保留无消息状态。帧BER/coverage、原生视频消息、给定消息验证、画质/资源分别报告。阈值在TEST前固定，严格大于才接受；tau退化也不重新调整。wrong-query仅描述，不另声称1%。源码/权重/临时缓存位于/content；正式结果目录不参与清理。\n')
    code("print('入口状态:', RUN['status'])\nprint('保留结果:', RUN['output'])\nentry['show_results']()\n")
    nb = dict(nbformat=4, nbformat_minor=5, metadata=dict(kernelspec=dict(name='python3', display_name='Python 3', language='python'), accelerator='GPU'), cells=cells)
    destination = Path(root)/'notebooks/native_baselines_paper_fpr1_v3.ipynb'
    destination.write_text(json.dumps(nb, ensure_ascii=False, indent=1)+'\n', encoding='utf-8')
    return destination


if __name__ == '__main__':
    print(build())
