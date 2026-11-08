"""Thin finite validation entry reusing the complete suite orchestration."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def build(root=ROOT):
    def code(source):return dict(cell_type='code',execution_count=None,outputs=[],metadata={},source=source.splitlines(keepends=True))
    def md(source):return dict(cell_type='markdown',metadata={},source=source.splitlines(keepends=True))
    cells=[code("from google.colab import drive\ndrive.mount('/content/drive')\n"),
        md("# 有限独立验证与低FPR统计\n\n选择GPU后Run all：复用原十二源入口，已消费A只读、原未启动B八源执行；随后运行原36draw/360claim查询包并生成48项只读派生统计。无新增视频或查询预算。此入口不启动Wan外部方法或public Baseline；完整论文入口仍保留。\n"),
        code("SOURCE_REF = 'main'\nSOURCE_DIRECTORY = None\n"),
        md("随机查询独立于冻结证据时，单族最多10个完整32位词，条件接受概率不超过10/2**32；现有36族的union界不超过360/2**32。这不是固定q0/key的自然总体FPR认证。TEST-B是有限8源，NEW正例、OFF负例、P0身份有/时间无控制分列；A的正式结果保留。普通未决和技术缺失均保留，不调阈值或加源。\n"),
        code("from pathlib import Path\nimport urllib.request\nbootstrap_path = 'runtime/wan/paper_suite_colab_v1.py'\nif SOURCE_DIRECTORY:\n    bootstrap = (Path(SOURCE_DIRECTORY) / bootstrap_path).read_text()\nelse:\n    bootstrap = urllib.request.urlopen(f'https://raw.githubusercontent.com/RICHAAARC/SC-SSTW/{SOURCE_REF}/{bootstrap_path}', timeout=120).read().decode('utf-8')\nentry = {'__name__': 'paper_fpr_colab_entry'}\nexec(compile(bootstrap, bootstrap_path, 'exec'), entry)\nRUN = entry['run_fpr_colab'](source_directory=SOURCE_DIRECTORY, source_ref=SOURCE_REF)\n"),
        md("新表fpr_probability_models明确概率空间/假设，fpr_measurements分列逐视图、ANY5、ALL5、至少4/5的来源统计、四态计数、参考区间和随机查询族界。旧结果不覆盖；query失败或消费后不补抽。名义1%/0.1%/10^-6只是同一界的展示，不改变接收器。\n"),
        code("print('入口状态:', RUN['status'])\nprint('执行记录:', RUN['invocation'])\nentry['show_results']()\n")]
    for i,cell in enumerate(cells):cell['id']='paper-fpr-'+str(i)
    nb=dict(nbformat=4,nbformat_minor=5,metadata=dict(kernelspec=dict(name='python3',display_name='Python 3',language='python'),accelerator='GPU'),cells=cells)
    path=Path(root)/'notebooks/paper_fpr_validation_v1.ipynb';path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(nb,ensure_ascii=False,indent=1)+'\n');return path


if __name__=='__main__':print(build())
