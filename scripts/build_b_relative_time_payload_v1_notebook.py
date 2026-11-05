"""Build the thin complete user Run-all entry; never run the experiment here."""
import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def build(root=ROOT):
    cells = []
    def code(text):
        ast.parse(text)
        cells.append(dict(cell_type='code', id=f'code-{len(cells)}', metadata={}, execution_count=None,
                          outputs=[], source=text.splitlines(keepends=True)))
    def note(text):
        cells.append(dict(cell_type='markdown', id=f'note-{len(cells)}', metadata={}, source=text.splitlines(keepends=True)))
    code("from google.colab import drive\ndrive.mount('/content/drive')\n")
    note('# 冻结时间到身份载荷：同批有限策略比较\n\n选择GPU后 Run all。自动读取原 r01 的 NEW/BASE 五视图媒体与已存条件时间，执行120 E41、120投影、180 keyed decode、75局部读取与25聚合；551主操作，2400局部/800最终槽。不会重新生成视频或重跑时间 reader。真实结果尚未运行。\n')
    code("SOURCE_REF = 'dev/local-joint-state-payload-v1'\nSOURCE_DIRECTORY = None  # 可选已有可编辑源码目录\n")
    note('## 读取已有输入并执行一次\n\n输入为 MyDrive/Video-WM/B-Joint-Time-Relative-Consistency-V1/fresh-r01-v1/attempt01；输出另存 B-Relative-Time-Payload-CDC32-V1/fixed-r01-v1/attempt01。整 attempt 消耗后再次 Run all 只读，包括不重做后评；失败/未进入槽保留。请只开一个运行实例。科学 worker 由外部进程组 watchdog 限制12600秒与进程树8GiB RSS，CUDA峰值实记，无GPU型号白名单。源码与依赖准备另计时。\n')
    code("""from pathlib import Path
import json
output = Path('/content/drive/MyDrive/Video-WM/B-Relative-Time-Payload-CDC32-V1/fixed-r01-v1/attempt01')
entry = None
RUN = {'status': 'NOT_STARTED', 'output': str(output)}

def show_saved_results():
    print('保留结果目录:', output)
    for name in ('attempt.json', 'result.json', 'counts.json', 'resources.json', 'quality.json',
                 'raw_terminal_summary.json', 'evaluation_summary.json', 'preparation_failure.json'):
        path = output / name
        if path.is_file():
            try:
                value = json.loads(path.read_text())
                if name == 'attempt.json':
                    value = {k: value.get(k) for k in ('status', 'consumed', 'reserved_at', 'denominators', 'limits')}
            except (OSError, ValueError, AttributeError) as error:
                value = {'read_error': str(error)}
            print(name, json.dumps(value, ensure_ascii=False, indent=2))
    print('完整保存记录: state.json、raw_terminal_slots.json、plans/、local/、final/、post/、bank/。')

if output.exists() and any(output.iterdir()):
    RUN = {'status': 'READ_ONLY', 'output': str(output)}
    show_saved_results()
else:
    import urllib.request
    bootstrap_path = 'runtime/wan/b_relative_time_payload_colab_v1.py'
    if SOURCE_DIRECTORY:
        bootstrap_text = (Path(SOURCE_DIRECTORY) / bootstrap_path).read_text()
    else:
        url = f'https://raw.githubusercontent.com/RICHAAARC/SC-SSTW/{SOURCE_REF}/{bootstrap_path}'
        bootstrap_text = urllib.request.urlopen(url).read().decode('utf-8')
    entry = {'__name__': 'relative_payload_colab_entry'}
    exec(compile(bootstrap_text, bootstrap_path, 'exec'), entry)
    RUN = entry['run_colab'](source_directory=SOURCE_DIRECTORY, source_ref=SOURCE_REF)
""")
    note('## 完整结果与解释范围\n\nB使用冻结时间，U使用四phase收到时钟，W固定时间+1，K固定另一实际key；P_BASE借NEW时间但读取BASE自己的媒体与缓存。先保存全部计划和盲终态，后打开truth/消息，报告共同位BER、支持增失、错归属、擦除和context。B/U/W是有限策略比较，不能归因成纯phase效果；重复载荷不认证组内精确位置。BASE带DCT2，五组画质仅是时间包增量，无clean OFF总体质量或FPR结论。原native精确目标未达与solver最优性未证保留。\n')
    code("print('运行状态:', RUN['status'])\nprint('结果目录:', RUN['output'])\nif entry is not None and callable(entry.get('show_results')):\n    entry['show_results']()\nelse:\n    show_saved_results()\n")
    path = Path(root)/'notebooks/b_relative_time_payload_v1.ipynb'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(nbformat=4, nbformat_minor=5, metadata=dict(
        kernelspec=dict(name='python3', display_name='Python 3', language='python'), accelerator='GPU'), cells=cells),
        ensure_ascii=False, indent=1)+'\n', encoding='utf-8')
    return path


if __name__ == '__main__': print(build())
