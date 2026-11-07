"""Build a thin frozen-batch Run-all; never execute the physical experiment here."""
import ast
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def build(root=ROOT):
    cells=[]
    def code(text):
        ast.parse(text);cells.append(dict(cell_type='code',id=f'code-{len(cells)}',metadata={},
            execution_count=None,outputs=[],source=text.splitlines(keepends=True)))
    def note(text): cells.append(dict(cell_type='markdown',id=f'note-{len(cells)}',metadata={},source=text.splitlines(keepends=True)))
    code("from google.colab import drive\ndrive.mount('/content/drive')\n")
    note('# 正式 TEST：12 内容、三臂时间与条件载荷\n\n选择 GPU 后 Run all。默认首4源 TEST-A，后8源是固定 TEST-B；普通低覆盖/未决不会要求换源或调参。真 OFF 与 DCT2 P0 两条配对轨迹，NEW 复用同一P0 capture；三臂五视图、双key双q、U与借映射P_PAIRED辅助、质量及后评由同一入口完成。入口自动从 dev/local-joint-state-payload-v1 加载可用源码并准备依赖。C4已完成工程审查；真实新源、GPU资源与Colab执行效果由本次用户运行验证，尚无新源实验结果。\n')
    code("BATCH = 'TEST-A'  # 固定另批为 TEST-B，无自由挑源列表\nSOURCE_REF = 'dev/local-joint-state-payload-v1'\nSOURCE_DIRECTORY = None  # 可选Colab内已有可编辑源码目录\n")
    note('每源一次attempt，开始/失败/中断都消费。再次Run all只读已消费源，未启动源仍按固定名单执行；不补阶段或后评。结果永久保存到 Video-WM_Result/main_method/paper_cohort_v1；本次自有临时缓存在根外且逐源释放。每源1114方法操作、60声明（最多1920比较），不是1114模型调用。solver保持原7200s/4000iter/16GiB工作数组估算；hard与relative共享9450s，identity18900s/8GiB。生成/capture/replay无新增整阶段时限，实际成本记录。\n')
    code("""from pathlib import Path
import json
OUTPUT_ROOT = Path('/content/drive/MyDrive/Video-WM_Result/main_method/paper_cohort_v1')
if BATCH not in ('TEST-A', 'TEST-B'):
    raise ValueError('选择固定 TEST-A 或 TEST-B')
source_ids = [f'paper-joint-t{i:02d}' for i in (range(1, 5) if BATCH == 'TEST-A' else range(5, 13))]
entry = None
if all((OUTPUT_ROOT / sid / 'attempt01').exists() and any((OUTPUT_ROOT / sid / 'attempt01').iterdir()) for sid in source_ids):
    RUN = {'status': 'READ_ONLY', 'batch': BATCH, 'output': str(OUTPUT_ROOT)}
    for sid in source_ids:
        path = OUTPUT_ROOT / sid / 'attempt01' / 'result.json'
        try:
            saved = json.loads(path.read_text()) if path.is_file() else '已消费，未保存完整result；不补跑'
            if path.is_file() and not isinstance(saved, dict):
                raise ValueError('result需要JSON对象；现有记录格式无效')
        except (OSError, ValueError) as error:
            saved = {'status': 'SAVED_RESULT_UNREADABLE', 'read_error': str(error), 'rerun': 'READ_ONLY'}
        print(sid, saved)
    summary_path = OUTPUT_ROOT / (BATCH + '_summary.json')
    if summary_path.is_file():
        try:
            print('已有批次汇总（纯只读）:', json.loads(summary_path.read_text()))
        except (OSError, ValueError) as error:
            print('已有批次汇总不可读；不补后评:', str(error))
else:
    import urllib.request
    bootstrap_path = 'runtime/wan/b_paper_cohort_colab_v1.py'
    if SOURCE_DIRECTORY:
        text = (Path(SOURCE_DIRECTORY) / bootstrap_path).read_text()
    else:
        text = urllib.request.urlopen(f'https://raw.githubusercontent.com/RICHAAARC/SC-SSTW/{SOURCE_REF}/{bootstrap_path}', timeout=120).read().decode('utf-8')
    entry = {'__name__': 'paper_cohort_colab_entry'}
    exec(compile(text, bootstrap_path, 'exec'), entry)
    RUN = entry['run_colab'](batch=BATCH, source_directory=SOURCE_DIRECTORY, source_ref=SOURCE_REF)
""")
    note('时间正确率、正确物理/时间归属支撑、字面BER、条件QMatch、质量和成本分开。OFF/P0空时间路径不是成功拒绝，P0本来有身份；P_PAIRED只用于身份保持。技术未知用收益区间保留，失败源不从分母剔除。重复载荷MATCH不证明精确时间认证，12源是有限Wan面板，不是总体FPR或任意内容保证。\n')
    code("print('状态:', RUN['status'])\nprint('结果目录:', RUN['output'])\nif entry is not None:\n    entry['show_results'](BATCH)\n")
    path=Path(root)/'notebooks/b_paper_cohort_v1.ipynb';path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(dict(nbformat=4,nbformat_minor=5,metadata=dict(
        kernelspec=dict(name='python3',display_name='Python 3',language='python'),accelerator='GPU'),cells=cells),ensure_ascii=False,indent=1)+'\n')
    return path


if __name__=='__main__': print(build())
