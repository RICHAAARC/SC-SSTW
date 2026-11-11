"""Thin Run-all entry: saved recovery, minimal physical continuation, global tables."""
import json
from pathlib import Path


def build(root=Path(__file__).resolve().parents[1]):
    cells=[]
    def add(kind,text):
        row=dict(cell_type=kind,id='compact-repair-'+str(len(cells)),metadata={},source=text.splitlines(True))
        if kind=='code':row.update(execution_count=None,outputs=[])
        cells.append(row)
    add('code',"from google.colab import drive\ndrive.mount('/content/drive')\n")
    add('markdown','# Compact12 四片定向恢复\n\n选择 GPU 后直接 Run all。先恢复已保存原始观测及 CPU 后评，再补限定视图/配对素材，最后按原分母汇总四片。原缺口为 87 个逻辑视图和 11 个质量阶段；经授权 ucf-0643 PixelSeal 正例完整替代，另测原 9 个完整视图及 3 个质量阶段，因此本次执行范围为 96 视图、14 质量阶段、79 个共享原始视图组，来自 6 个源。原始视图组数不等于 GPU 调用数，全局分母不增加。原四片已停止，旧成功、缺失、失败、预算和 v4 结果全部保留。\n')
    add('code',"SOURCE_REF = 'main'\nSOURCE_DIRECTORY = None\nPREVIOUS_RUNTIME_STOPPED = True  # 已确认原四片 runtime 均 CLOSED\nCONFIG = dict(\n    parent='/content/drive/MyDrive/Video-WM_Result/baselines/public-native-frameindex-compact12-v1/attempt01',\n    output='/content/drive/MyDrive/Video-WM_Result/baselines/public-native-frameindex-compact12-v1/attempt01/recovery/repair01',\n    cache='/content/drive/MyDrive/Video-WM_Cache/baseline-compact12-v1',\n    old='/content/drive/MyDrive/Video-WM_Result/baselines/public-native-and-frameindex-fpr001-v4/attempt01',\n    old_cache='/content/drive/MyDrive/Video-WM_Cache/baseline-v4')\n")
    add('markdown','自动导入同目录的 `local_cpu_recovery.zip`，仅补缺失文件并将本地 CPU 成本计入一次；写入前检查原分片和修复目录的运行状态，采用完整临时写入后原子发布，已有 Colab 记录不覆盖，不需要手工上传。已恢复的 3 视图和 5 质量阶段直接跳过。PixelSeal 与 FrameIndex 的 OFF 原始读取共享，分别执行原逻辑后评。缺 epoch 时仅在回执能唯一关联时恢复，绝不混 epoch 或按分数择优。ucf-0642 的两个正例臂各 12 视图全缺，按原查询整臂重建；ucf-0643 PixelSeal 按原方法/消息完整重建 12 视图和配套 3 质量，新臂每槽均只认新结果，失败槽保持未知，绝不回填旧 9 个成功视图。旧臂仍保留，替代关系记录在 `shards/part2/replacements/`。其余部分 epoch 输入丢失、已提交数组不可读、关联不清等单列未决。质量必要 codec 成本独立记录；LPIPS 不补算。\n')
    add('code',"from pathlib import Path\nimport urllib.request, importlib.util\nname = 'native_baseline_compact_recovery_v1_entry.py'\nif SOURCE_DIRECTORY:\n    path = Path(SOURCE_DIRECTORY)/'scripts'/name\nelse:\n    path = Path('/content')/name\n    urllib.request.urlretrieve(f'https://raw.githubusercontent.com/RICHAAARC/SC-SSTW/{SOURCE_REF}/scripts/{name}', path)\nspec = importlib.util.spec_from_file_location('compact_repair_entry', path)\nentry = importlib.util.module_from_spec(spec)\nspec.loader.exec_module(entry)\nRUN = entry.execute(CONFIG, SOURCE_DIRECTORY, SOURCE_REF, PREVIOUS_RUNTIME_STOPPED)\n")
    add('markdown','新恢复包位于正式结果的 `recovery/repair01`，旧分片仍只读。`cpu_recovery.json` 保存逐槽恢复/剩余清单，`physical_recovery.json` 保存补算返回与失败，`global/tables` 保存全局 A/B/C/D 表。运行完成、测量完整和科学结论分别报告；保留原阈值、CAL32/TEST500、12 视图、DCT2＋CDC32 和旧 H/L。\n')
    add('code',"print('返回:', RUN)\nprint('全局表目录:', CONFIG['output']+'/global/tables')\n")
    value=dict(nbformat=4,nbformat_minor=5,metadata=dict(accelerator='GPU',kernelspec=dict(name='python3',display_name='Python 3',language='python')),cells=cells)
    path=Path(root)/'notebooks/native_baseline_compact_recovery_v1.ipynb';path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=1)+'\n');return path


if __name__=='__main__':print(build())
