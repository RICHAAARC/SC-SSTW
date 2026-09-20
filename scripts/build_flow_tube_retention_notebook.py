"""User-run fixed tube terminal/native bridge and blind full-video receiver; no execution here."""
from pathlib import Path
import argparse
import json
import re
import subprocess

ROOT=Path(__file__).resolve().parents[1]


def build(source_commit=None):
    if source_commit is not None:
        if not re.fullmatch('[0-9a-f]{40}',source_commit):raise ValueError('full source SHA required')
        subprocess.run(['git','-C',str(ROOT),'cat-file','-e',source_commit+':experiments/wan_state_clock/flow_tube_retention_run.py'],check=True)
    cells=[]
    def cell(kind,name,text):
        row={'cell_type':kind,'id':name,'metadata':{},'source':text.splitlines(keepends=True)}
        if kind=='code':row.update(execution_count=None,outputs=[])
        cells.append(row)
    cell('code','drive-mount',"from google.colab import drive\ndrive.mount('/content/drive')\n")
    cell('markdown','scope',"""# 同载体单次写入：控制保留效率诊断（准备版）

原dev_p0_s0/dev_p1_s0；固定OFF及44/46/49各A/B，共14视频、56个内部接收phase编码。
1760管状块/state×polarity×sync、margin1、A/B、key、原receiver和native UniPC不变。
每case先完成一条OFF并保存各必要节点和完整history；每marked仅从对应节点分叉写一次，后续自由采样。
49原full correction的实际native D support RMS，作为同case/message的44/46匹配参考。
一次unit probe测响应，再一次缩放、一次live update，记录实际匹配误差，不重试不挑终态。
该尺度用到未来OFF末态，只是匹配尺度的诊断，不是可在线部署的统一规则，也不保证同感知质量。

测量控制clean响应；t+1预测clean；再自由一步的t+2 clean；终态net响应；完整MP4盲读消息间隔增益。
44/46短期CFG均复用自然tail调用，49短期N/A。所有14视频均入媒体，不依据终态筛选。
每case132TF/72native/4unit probes，7decode/7MP4/28encode；全轮264/144/8和14/14/56。
失败保留固定分母，OFF排名不是FPR。比较是12marked配对描述，非显著性检验/独立泛化结论。
PSNR仅相对OFF残差；感知质量未测，请自行观看received_videos中的OFF与各时刻A/B。
未来holdout接口冻结为disabled、selection_rule=None：本轮不逐视频选时刻或择优规则。
仅CPU/fake/static验证，本地不运行实模/GPU/旧数据评分；用户Run all实测。
输出独立 `MyDrive/Video-WM/FlowTubeRetention/flow_tube_retention_<UTC>`，不改旧证据。
源码SHA："""+str(source_commit)+"""。
""")
    cell('code','source-and-software',f"""from pathlib import Path
from datetime import datetime, timezone
import importlib.metadata, json, os, signal, subprocess, sys
SOURCE_COMMIT = {source_commit!r}
if SOURCE_COMMIT is None:
    raise RuntimeError('Pending source publication: rebuild with the published full SHA')
SOURCE_URL = 'https://github.com/RICHAAARC/SC-SSTW.git'
RUN_ID = 'flow_tube_retention_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
SOURCE = Path('/content') / (RUN_ID + '_source')
subprocess.run(['git', 'init', str(SOURCE)], check=True)
subprocess.run(['git', '-C', str(SOURCE), 'remote', 'add', 'origin', SOURCE_URL], check=True)
subprocess.run(['git', '-C', str(SOURCE), 'fetch', '--depth', '1', 'origin', SOURCE_COMMIT], check=True)
subprocess.run(['git', '-C', str(SOURCE), 'checkout', '--detach', SOURCE_COMMIT], check=True)
if subprocess.check_output(['git', '-C', str(SOURCE), 'rev-parse', 'HEAD'], text=True).strip() != SOURCE_COMMIT:
    raise RuntimeError('Source checkout differs from pinned SHA')
def version(name):
    try: return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError: return None
if version('torch') != '2.11.0+cu128':
    subprocess.run([sys.executable, '-m', 'pip', 'install', 'torch==2.11.0', 'torchvision', '--index-url', 'https://download.pytorch.org/whl/cu128'], check=True)
subprocess.run([sys.executable, '-m', 'pip', 'install', 'diffusers==0.40.0', 'transformers', 'accelerate', 'ftfy', 'sentencepiece', 'safetensors', 'huggingface_hub', 'numpy', 'Pillow'], check=True)
subprocess.run([sys.executable, '-c', "import torch,diffusers; assert str(torch.__version__) == '2.11.0+cu128', torch.__version__; assert diffusers.__version__ == '0.40.0', diffusers.__version__"], check=True)
""")
    cell('code','fixed-fourteen-videos',"""OUTPUT = Path('/content/drive/MyDrive/Video-WM/FlowTubeRetention') / RUN_ID
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
ARCHIVE = OUTPUT.parent / (RUN_ID + '.source.zip')
subprocess.run(['git', '-C', str(SOURCE), 'archive', '--format=zip', '--output', str(ARCHIVE), SOURCE_COMMIT], check=True)
LOG = OUTPUT.parent / (RUN_ID + '.launcher.log')
command = [sys.executable, '-u', '-m', 'experiments.wan_state_clock.flow_tube_retention_run', '--output', str(OUTPUT)]
with LOG.open('w') as log:
    process = subprocess.Popen(command, cwd=SOURCE, start_new_session=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    try:
        for line in process.stdout:
            print(line, end=''); log.write(line); log.flush()
        code = process.wait()
    except BaseException:
        try:
            os.killpg(process.pid, signal.SIGTERM); process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL); process.wait()
        except ProcessLookupError:
            pass
        raise
print('Output:', OUTPUT, 'Launcher log:', LOG, 'Source archive:', ARCHIVE)
if (OUTPUT / 'result.json').exists():
    result = json.loads((OUTPUT / 'result.json').read_text())
    print(json.dumps({k:result.get(k) for k in ('status','video_denominator','receiver_encode_denominator','fixed_calls','actual_calls_observed','recovery_summary','equivalence_summary','mechanism_summary','predictive_diagnostics','quality_summary','OFF_summary')}, indent=2))
if code:
    raise subprocess.CalledProcessError(code, command)
""")
    notebook={'cells':cells,'metadata':{'kernelspec':{'display_name':'Python3','language':'python','name':'python3'},'language_info':{'name':'python'}},'nbformat':4,'nbformat_minor':5}
    target=ROOT/'notebooks/flow_tube_retention_colab.ipynb';target.write_text(json.dumps(notebook,indent=1)+'\n');return target


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--source-commit')
    print(build(parser.parse_args().source_commit))
