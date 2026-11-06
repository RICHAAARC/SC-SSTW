"""Build the complete thin r02 Run-all notebook without executing real science."""
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
    note('# r02 单新源：联合时间与身份载荷验证\n\n选择 GPU 后 Run all。从固定海浪源生成开始，自动串接原 DCT2 写入、时间包 CPU 求解、原生 VAE 回放、两臂五视图盲时间、冻结 refs、E41/CDC32 盲载荷、联合盲终态、两项后评和展示。固定 seed 2026110602，保留原 Joint-A/F42/630位条件严格最小与 DCT2/CDC32 方法。工程检查不代表真实 GPU 或科学结果已完成。\n')
    code("SOURCE_REF = 'dev/local-joint-state-payload-v1'\nSOURCE_DIRECTORY = None  # 可选已有可编辑源码目录\n")
    note('## 一次尝试与自动执行\n\n629 是时间78＋载荷551外层操作，并非模型调用数。预登记1644节点/1634边、28800 raw位、2400局部/800最终槽。整尝试失败或中断也消费；再次 Run all 只读，不续阶段或重做后评。请只开一个实例。临时源码/权重/缓存放 /content，结果永久保留。\n\n求解器4000迭代/7200秒/16GiB live-array估算；时间接收8GiB、单次600秒/累计6300秒；codec单次300秒。载荷子进程含联合闭合与两项后评，限12600秒/8GiB进程树RSS。此上限不覆盖生成、捕获或原生回放全链；这些阶段耗时/峰值记录，CUDA峰值实记。\n')
    code("""from pathlib import Path
import json
output = Path('/content/drive/MyDrive/Video-WM/B-Joint-Time-Payload-R02-V1/fresh-r02-v1/attempt01')
entry = None
RUN = {'status': 'NOT_STARTED', 'output': str(output)}

def show_saved_results():
    print('保留结果目录:', output)
    for name in ('attempt.json', 'result.json', 'posts.json', 'preparation_failure.json',
                 'time_chain/evaluation.json', 'payload/evaluation_summary.json',
                 'payload/resources.json', 'payload/quality.json', 'payload/raw_terminal_summary.json'):
        path = output/name
        if path.is_file():
            try:
                value = json.loads(path.read_text())
                if name == 'attempt.json':
                    value = {k: value.get(k) for k in ('status','consumed','reserved_at','denominators')}
            except (OSError, ValueError, AttributeError) as error:
                value = {'read_error': str(error)}
            print(name, json.dumps(value, ensure_ascii=False, indent=2))

if output.exists() and any(output.iterdir()):
    RUN = {'status': 'READ_ONLY', 'output': str(output)}
    show_saved_results()
else:
    import urllib.request
    bootstrap_path = 'runtime/wan/b_joint_time_payload_r02_colab_v1.py'
    if SOURCE_DIRECTORY:
        bootstrap_text = (Path(SOURCE_DIRECTORY)/bootstrap_path).read_text()
    else:
        url = f'https://raw.githubusercontent.com/RICHAAARC/SC-SSTW/{SOURCE_REF}/{bootstrap_path}'
        bootstrap_text = urllib.request.urlopen(url, timeout=120).read().decode('utf-8')
    entry = {'__name__': 'joint_r02_colab_entry'}
    exec(compile(bootstrap_text, bootstrap_path, 'exec'), entry)
    RUN = entry['run_colab'](source_directory=SOURCE_DIRECTORY, source_ref=SOURCE_REF)
""")
    note('## 结果范围\n\nB冻结时间；U收到时钟四phase；W固定+1且不wrap；K固定另一实际key；P_BASE借NEW映射、读BASE自己的身份缓存，并非clean OFF。两项后评均在联合盲终态之后；时间成绩不决定是否执行载荷。分别报告共同位BER、支持增失、时间错误、错归属、擦除、wrong-key与质量。旧native精确目标未达/solver最优性未证、W组内位置和K拒绝局限保留。单新内容只能检验迁移，不证明总体泛化、FPR或科学PASS。\n')
    code("print('运行状态:', RUN['status'])\nprint('结果目录:', RUN['output'])\nif entry is not None and callable(entry.get('show_results')):\n    entry['show_results']()\nelse:\n    show_saved_results()\n")
    path = Path(root)/'notebooks/b_joint_time_payload_r02_v1.ipynb'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(nbformat=4, nbformat_minor=5, metadata=dict(
        kernelspec=dict(name='python3', display_name='Python 3', language='python'), accelerator='GPU'), cells=cells),
        ensure_ascii=False, indent=1)+'\n', encoding='utf-8')
    return path


if __name__ == '__main__': print(build())
