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
        subprocess.run(['git','-C',str(ROOT),'cat-file','-e',source_commit+':experiments/wan_state_clock/flow_tube_state_guidance_run.py'],check=True)
    cells=[]
    def cell(kind,name,text):
        row={'cell_type':kind,'id':name,'metadata':{},'source':text.splitlines(keepends=True)}
        if kind=='code':row.update(execution_count=None,outputs=[])
        cells.append(row)
    cell('code','drive-mount',"from google.colab import drive\ndrive.mount('/content/drive')\n")
    cell('markdown','scope',"""# 原1760管状块：终态投影与末步native guidance（准备版）

仅CPU/合成测试；真实模型与媒体结果待用户Colab运行。首轮固定原4个开发case，非新holdout。
每case共享0..48前缀、step49速度与scheduler历史，OFF/TERMINAL_A/B/LAST_A/B共20视频。
原state_clock码本codes=state×polarity×sync，margin1、1760管状块及五种接收模式不变。
TERMINAL直接投影OFF终态；LAST对clean=z−sigma*v做相同完整投影u=P(clean)−clean，
通过v−u/sigma走native UniPC最后一步。无/3分配、强度参数、Jacobian、VAE梯度或扫描。
这是原margin闭式预条件梯度的完整修正；LAST与TERMINAL预期数值等价，非新梯度算法或轨迹创新。
报告OFF vs clean、LAST vs P(clean)、LAST vs TERMINAL的maxabs/固定容差及实际投影margin。

所有20视频独立decode/H264 CRF18/yuv420p保存回读，每视频4个真实RGB内部origin重新编码共80次。
181帧完整视频：g0输入181，g1/2/3输入177。这些是接收器相位假设，不是真实裁剪攻击。
state_clock.read仅接收公开码本和视频VAE观测；写端投影/真值不进得分，真值后报。
TERMINAL/LAST各8marked，原五mode分别报告，OFF不作FPR；未知偏移真实裁剪留下一阶段。
不沿用旧1.5质量门槛；RGB/时域残差仅诊断，请人工检查内容、伪影、连贯性。

每case100 Transformer、52 native step；全轮400/208，20decode/20MP4/80encode。
无terminal筛选，失败固定槽保留。模型阶段与VAE媒体阶段分独立子进程。
保存源码ZIP/日志/manifest/码本/prefix与history/终态/媒体/接收张量和哈希。
旧SyncTube是framewise VAE路线，与当前Wan3D VAE及state编码不同；不继承其成功率或盲检测声明。
输出 `MyDrive/Video-WM/FlowTubeStateGuidance/flow_tube_state_guidance_<UTC>`。
源码SHA："""+str(source_commit)+"""。本notebook尚未执行真实验证。
""")
    cell('code','source-and-software',f"""from pathlib import Path
from datetime import datetime, timezone
import importlib.metadata, json, os, signal, subprocess, sys
SOURCE_COMMIT = {source_commit!r}
if SOURCE_COMMIT is None:
    raise RuntimeError('Pending source publication: rebuild with the published full SHA')
SOURCE_URL = 'https://github.com/RICHAAARC/SC-SSTW.git'
RUN_ID = 'flow_tube_state_guidance_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
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
    cell('code','fixed-twenty-videos',"""OUTPUT = Path('/content/drive/MyDrive/Video-WM/FlowTubeStateGuidance') / RUN_ID
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
ARCHIVE = OUTPUT.parent / (RUN_ID + '.source.zip')
subprocess.run(['git', '-C', str(SOURCE), 'archive', '--format=zip', '--output', str(ARCHIVE), SOURCE_COMMIT], check=True)
LOG = OUTPUT.parent / (RUN_ID + '.launcher.log')
command = [sys.executable, '-u', '-m', 'experiments.wan_state_clock.flow_tube_state_guidance_run', '--output', str(OUTPUT)]
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
    print(json.dumps({k:result.get(k) for k in ('status','video_denominator','receiver_encode_denominator','fixed_calls','actual_calls_observed','recovery_summary','equivalence_summary','control_summary','quality_summary','OFF_summary')}, indent=2))
if code:
    raise subprocess.CalledProcessError(code, command)
""")
    notebook={'cells':cells,'metadata':{'kernelspec':{'display_name':'Python3','language':'python','name':'python3'},'language_info':{'name':'python'}},'nbformat':4,'nbformat_minor':5}
    target=ROOT/'notebooks/flow_tube_state_guidance_colab.ipynb';target.write_text(json.dumps(notebook,indent=1)+'\n');return target


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--source-commit')
    print(build(parser.parse_args().source_commit))
