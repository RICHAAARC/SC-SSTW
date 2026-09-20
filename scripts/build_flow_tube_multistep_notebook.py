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
        subprocess.run(['git','-C',str(ROOT),'cat-file','-e',source_commit+':experiments/wan_state_clock/flow_tube_multistep_run.py'],check=True)
    cells=[]
    def cell(kind,name,text):
        row={'cell_type':kind,'id':name,'metadata':{},'source':text.splitlines(keepends=True)}
        if kind=='code':row.update(execution_count=None,outputs=[])
        cells.append(row)
    cell('code','drive-mount',"from google.colab import drive\ndrive.mount('/content/drive')\n")
    cell('markdown','scope',"""# 同管状块完整投影：早期写入与末步补偿机制（准备版）

仅CPU/fake/static验证，未在本地运行真实模型或GPU。原4dev×7arms=28视频/112相位encode。
固定OFF、LAST_A/B、EARLY_LAST_A/B、EARLY_ONLY_A/B。载体1760块、state×polarity×sync、
margin1、key、A/B、原接收器及native UniPC不变，不扫描强度。
EARLY=44/45/46，full projection u=P(clean)−clean；EARLY_LAST再49投影，EARLY_ONLY在47..49自由续采样。
LAST只49控制。共享0..43；每消息early两臂共用到49前状态/history再分叉。

旧44..46用/3修正与R预算cap；其原run工程全完成、质量规则通过但消息归因不稳定。
本轮去掉/3与cap，改变dose及末步补偿；不是仅时间差异的因果证明，也不是等预算性能比较。
没有新梯度算法。中间步骤走真实native scheduler，不用终态投影替代轨迹。
每控制步同history OFF shadow测actualD；49复用已有无控制分叉，不多跑shadow。
记录实际D的sum/peak RMS、净终态位移、每步预测clean投影/消息诊断、49前margin和末步u/D补偿。
中间noisy-state投影仅诊断，不是终态消息；质量与消息判断仍看完整MP4盲读。

共享预算每case124Transformer/66native/6shadow；全轮496/264/24，28decode/28MP4/112encode。
每scheme8marked，所有视频均入媒体，失败不隐藏。媒体五mode/质量等设置原样，无新门槛。
先看mechanism_summary，再看recovery_summary/quality_summary；OFF不作FPR。
请人工观看各case received_videos中的OFF和六marked内容、伪影、连贯性，不能只看PSNR。
新目录 `MyDrive/Video-WM/FlowTubeMultistep/flow_tube_multistep_<UTC>`；原baseline与裁剪入口不改。
保存源ZIP、配置、prefix/history、每step诊断、终态和接收张量。源码SHA："""+str(source_commit)+"""。
""")
    cell('code','source-and-software',f"""from pathlib import Path
from datetime import datetime, timezone
import importlib.metadata, json, os, signal, subprocess, sys
SOURCE_COMMIT = {source_commit!r}
if SOURCE_COMMIT is None:
    raise RuntimeError('Pending source publication: rebuild with the published full SHA')
SOURCE_URL = 'https://github.com/RICHAAARC/SC-SSTW.git'
RUN_ID = 'flow_tube_multistep_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
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
    cell('code','fixed-twentyeight-videos',"""OUTPUT = Path('/content/drive/MyDrive/Video-WM/FlowTubeMultistep') / RUN_ID
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
ARCHIVE = OUTPUT.parent / (RUN_ID + '.source.zip')
subprocess.run(['git', '-C', str(SOURCE), 'archive', '--format=zip', '--output', str(ARCHIVE), SOURCE_COMMIT], check=True)
LOG = OUTPUT.parent / (RUN_ID + '.launcher.log')
command = [sys.executable, '-u', '-m', 'experiments.wan_state_clock.flow_tube_multistep_run', '--output', str(OUTPUT)]
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
    print(json.dumps({k:result.get(k) for k in ('status','video_denominator','receiver_encode_denominator','fixed_calls','actual_calls_observed','recovery_summary','equivalence_summary','mechanism_summary','quality_summary','OFF_summary')}, indent=2))
if code:
    raise subprocess.CalledProcessError(code, command)
""")
    notebook={'cells':cells,'metadata':{'kernelspec':{'display_name':'Python3','language':'python','name':'python3'},'language_info':{'name':'python'}},'nbformat':4,'nbformat_minor':5}
    target=ROOT/'notebooks/flow_tube_multistep_colab.ipynb';target.write_text(json.dumps(notebook,indent=1)+'\n');return target


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--source-commit')
    print(build(parser.parse_args().source_commit))
