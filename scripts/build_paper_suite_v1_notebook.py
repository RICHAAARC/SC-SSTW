"""Build the complete, thin user-run Colab notebook; never execute it here."""
import ast
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def build(root=ROOT):
    cells=[]
    def add(kind,text):
        cell=dict(cell_type=kind,id='paper-suite-'+str(len(cells)),metadata={},source=text.splitlines(keepends=True))
        if kind=='code':ast.parse(text);cell.update(execution_count=None,outputs=[])
        cells.append(cell)
    add('code',"from google.colab import drive\ndrive.mount('/content/drive')\n")
    add('markdown',"# 完整论文实验与结果包\n\n选择 GPU 后 Run all。默认完整十二源：已消费 TEST-A 只读，未启动 TEST-B 按原名单执行；之后完成四配置 Wan 五视图对照、主方法随机查询统计、原公开视频 Baseline v4 断点续接和统一论文表格。所有 GPU 与 CPU 媒体/接收/后评均在 Colab 内完成，不需要下载回本地接续。\n\n这是一套有固定预算的完整执行入口，不承诺一次 Colab 会话足够或结果全部为正。每个新 Wan 源只有一次 attempt，失败不补；public v4 保留原计算 checkpoint 与其原有恢复预算。再次 Run all 读取已消费新源，并继续未启动源及 public 原断点。\n")
    add('code',"SOURCE_REF = 'main'\nSOURCE_DIRECTORY = None  # 可选 Colab 内可读源码目录\nPREVIOUS_RUNTIME_STOPPED = False  # 仅 public Baseline 的旧 runtime 已实际停止时改 True\n")
    add('markdown',"若公开 Baseline 仍在其它 runtime 执行，本入口只报告 EXTERNAL_RUNTIME_ACTIVE 并继续独立部分，不抢占。只有实际停止旧任务后才设置上面的确认。正式结果全部在 Video-WM_Result；旧结果不删除、覆盖或重算。新同域对照与查询统计使用独立目录和预算。\n")
    add('code',"from pathlib import Path\nimport urllib.request\nbootstrap_path = 'runtime/wan/paper_suite_colab_v1.py'\nif SOURCE_DIRECTORY:\n    bootstrap = (Path(SOURCE_DIRECTORY) / bootstrap_path).read_text()\nelse:\n    bootstrap = urllib.request.urlopen(f'https://raw.githubusercontent.com/RICHAAARC/SC-SSTW/{SOURCE_REF}/{bootstrap_path}', timeout=120).read().decode('utf-8')\nentry = {'__name__': 'paper_suite_colab_entry'}\nexec(compile(bootstrap, bootstrap_path, 'exec'), entry)\nRUN = entry['run_colab'](source_directory=SOURCE_DIRECTORY, source_ref=SOURCE_REF, previous_runtime_stopped=PREVIOUS_RUNTIME_STOPPED)\n")
    add('markdown',"每个阶段后自动保存论文表格快照：时间（原 F42/relative）、B/U及P_PAIRED贡献、载荷、四态声明/误接受/未决、随机查询解析界与来源统计、同域比较、质量和成本；公开自然视频完整四表保留在原包，统一报告提供索引。\n\n同域对照共享 OFF 首代解码输入和五种最终二代媒体条件，但主方法是生成期写入、外部方法是首代后写入；原生位数分列，不声称等容量排名。FrameIndex 是 ADAPTED，其他原生方法时间列为 N/A。QMatch 32位全匹配的随机查询族解析界与十二源经验统计分开，不声称自然总体 FPR 的95%上界≤0.1%。OFF/P0空时间未决不叫成功矛盾拒绝。\n")
    add('code',"print('入口状态:', RUN['status'])\nprint('执行记录:', RUN['invocation'])\nentry['show_results']()\n")
    path=Path(root)/'notebooks/paper_suite_v1.ipynb';path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(dict(nbformat=4,nbformat_minor=5,metadata=dict(kernelspec=dict(name='python3',display_name='Python 3',language='python'),accelerator='GPU'),cells=cells),ensure_ascii=False,indent=1)+'\n')
    return path


if __name__=='__main__':print(build())
