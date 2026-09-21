"""Build a fixed-source, JSON-only CPU audit notebook. Does not run it."""
from pathlib import Path
import argparse,json,re,subprocess
ROOT=Path(__file__).resolve().parents[1]

def build(source_commit=None):
    if source_commit is not None:
        if not re.fullmatch('[0-9a-f]{40}',source_commit):raise ValueError('full source SHA required')
        subprocess.run(['git','-C',str(ROOT),'cat-file','-e',source_commit+':experiments/wan_state_clock/flow_tube_sync_diagnostics_run.py'],check=True)
    cells=[]
    def cell(kind,name,text):
        row=dict(cell_type=kind,id=name,metadata={},source=text.splitlines(keepends=True))
        if kind=='code':row.update(execution_count=None,outputs=[])
        cells.append(row)
    cell('code','drive-mount',"from google.colab import drive\ndrive.mount('/content/drive')\n")
    cell('markdown','scope',"""# 既有 LAST 检测结果：CPU 时域同步诊断

固定读取 flow_tube_detection_20260920T113113319009Z（原生成源码 f561fd58）。无需 GPU。
只读原 JSON：6 eval 来源 ×7视图=42视图/462窗口，另保留6个原sequence决策，不混9个calibration来源。
原阈值、存在性、A/B、排名、候选不变；不生成、不读视频/张量、不VAE编码、不重新搜索。
报告选中完整路径、观测等价类与各top-tie支持、缺失/重复、原state更新一致性；真值映射仅事后加入。
删除边界partial、变速遗漏源帧是几何支持限制，不能将exact-support=0误报为同步成功率0。
nominal RGB支持不等于VAE感受野/latent等价；精确同步和编辑帧定位保持不可判定。
逐份读取原blind JSON并验证SHA；缺失保留固定42分母。原始文件不改，新审计输出另存。
这只是旧结果复核，不增加独立实验样本。源码SHA："""+str(source_commit)+"\n")
    cell('code','source',f"""from pathlib import Path
from datetime import datetime, timezone
import json, subprocess, sys
SOURCE_COMMIT = {source_commit!r}
if SOURCE_COMMIT is None:
    raise RuntimeError('Pending publication: rebuild with fixed source SHA')
RUN_ID = 'flow_tube_sync_diagnostics_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
SOURCE = Path('/content') / (RUN_ID + '_source')
subprocess.run(['git', 'init', str(SOURCE)], check=True)
subprocess.run(['git', '-C', str(SOURCE), 'remote', 'add', 'origin', 'https://github.com/RICHAAARC/SC-SSTW.git'], check=True)
subprocess.run(['git', '-C', str(SOURCE), 'fetch', '--depth', '1', 'origin', SOURCE_COMMIT], check=True)
subprocess.run(['git', '-C', str(SOURCE), 'checkout', '--detach', SOURCE_COMMIT], check=True)
assert subprocess.check_output(['git', '-C', str(SOURCE), 'rev-parse', 'HEAD'], text=True).strip() == SOURCE_COMMIT
subprocess.run([sys.executable, '-m', 'pip', 'install', 'numpy'], check=True)
""")
    cell('code','fixed-cpu-audit',"""INPUT = Path('/content/drive/MyDrive/Video-WM/FlowTubeDetection/flow_tube_detection_20260920T113113319009Z')
OUTPUT = Path('/content/drive/MyDrive/Video-WM/FlowTubeSyncDiagnostics') / RUN_ID
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
LOG = OUTPUT.parent / (RUN_ID + '.log')
command = [sys.executable, '-u', '-m', 'experiments.wan_state_clock.flow_tube_sync_diagnostics_run', '--source', str(INPUT), '--output', str(OUTPUT)]
with LOG.open('w') as log:
    code = subprocess.run(command, cwd=SOURCE, stdout=log, stderr=subprocess.STDOUT, check=False).returncode
print('Original fixed input:', INPUT, 'Audit output:', OUTPUT, 'Log:', LOG)
if (OUTPUT / 'result.json').exists():
    result = json.loads((OUTPUT / 'result.json').read_text())
    print(json.dumps({k:result.get(k) for k in ('status','source_denominator','view_denominator','window_denominator','by_view','claim')}, indent=2))
if code:
    raise subprocess.CalledProcessError(code, command)
""")
    nb=dict(cells=cells,metadata={'kernelspec':{'display_name':'Python3','language':'python','name':'python3'}},nbformat=4,nbformat_minor=5)
    path=ROOT/'notebooks/flow_tube_sync_diagnostics_colab.ipynb';path.write_text(json.dumps(nb,indent=1)+'\n');return path

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source-commit');print(build(p.parse_args().source_commit))
