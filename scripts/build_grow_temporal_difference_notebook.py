"""User-run fixed terminal-only late-control candidate; no execution here."""
from pathlib import Path
import argparse
import json
import re
import subprocess

ROOT=Path(__file__).resolve().parents[1]


def build(source_commit=None):
    if source_commit is not None:
        if not re.fullmatch('[0-9a-f]{40}',source_commit):raise ValueError('full source SHA required')
        subprocess.run(['git','-C',str(ROOT),'cat-file','-e',source_commit+':experiments/wan_state_clock/grow_temporal_difference_run.py'],check=True)
    cells=[]
    def cell(kind,name,text):
        row={'cell_type':kind,'id':name,'metadata':{},'source':text.splitlines(keepends=True)}
        if kind=='code':row.update(execution_count=None,outputs=[])
        cells.append(row)
    cell('code','drive-mount',"from google.colab import drive\ndrive.mount('/content/drive')\n")
    cell('markdown','scope',"""# GROW 固定相邻时间差分：终态验证

选择 GPU 后“全部运行”。固定新生成 2 内容 × 2 种子 × OFF/A/B = 12 条轨迹，
8 条携带消息；不复用历史 OFF。仅终态，不执行 VAE、MP4、质量或时间攻击测试。

唯一候选改动：在原有 channel0、空间 DCT-II 频率2..9 上，将46片恒定重复
改为23对相邻时间正交差 d[k]=(c[2k]-c[2k+1])/sqrt(2)。写入损失与读出均使用d，
保持16位消息、每对4重复、幅度.5、eta.1、控制30..49、CFG5及50步原生UniPC。
读出每位92个符号票，零/平票擦除；原始时间46片与23对分别报告。
该固定配对需要完整视频已知时间起点，不构成时间同步或抗裁剪方案。

全部8条消息16/16正确且零擦除才标media_eligible；不自动进入媒体阶段。
记录same-history无控制shadow的实际响应、差分域OFF偏置、局部loss、末步贡献。
末步输出可能等于受控predicted-clean，终态成功不证明此前累积或MP4存活。
固定1200 Transformer、600 live step、160局部梯度、160 shadow step。

独立输出 `MyDrive/Video-WM/GROWTemporalDifference/grow_temporal_difference_<UTC>`；
父级source.zip/launcher.log，子目录配置、调度、终态张量、结果与日志。
环境沿用torch2.11.0+cu128/diffusers0.40.0，无新增GPU型号限制。
本交付只完成CPU/Fake与静态检查，未运行预训练模型、GPU或Colab。
固定源码 SHA："""+str(source_commit)+"""。
""")
    cell('code','source-and-software',f"""from pathlib import Path
from datetime import datetime, timezone
import importlib.metadata, json, os, signal, subprocess, sys
SOURCE_COMMIT = {source_commit!r}
if SOURCE_COMMIT is None:
    raise RuntimeError('Pending source publication: rebuild with the published full SHA')
SOURCE_URL = 'https://github.com/RICHAAARC/SC-SSTW.git'
RUN_ID = 'grow_temporal_difference_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
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
    cell('code','fixed-twelve-videos',"""OUTPUT = Path('/content/drive/MyDrive/Video-WM/GROWTemporalDifference') / RUN_ID
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
ARCHIVE = OUTPUT.parent / (RUN_ID + '.source.zip')
subprocess.run(['git', '-C', str(SOURCE), 'archive', '--format=zip', '--output', str(ARCHIVE), SOURCE_COMMIT], check=True)
LOG = OUTPUT.parent / (RUN_ID + '.launcher.log')
command = [sys.executable, '-u', '-m', 'experiments.wan_state_clock.grow_temporal_difference_run', '--output', str(OUTPUT)]
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
    print(json.dumps({'status': result['status'], 'video_denominator': result['video_denominator'], 'media_gate': result.get('media_gate'), 'fixed_calls': result['fixed_calls'], 'actual_calls_observed': result.get('actual_calls_observed'), 'cases': {k: v['status'] for k, v in result['cases'].items()}}, indent=2))
if code:
    raise subprocess.CalledProcessError(code, command)
""")
    notebook={'cells':cells,'metadata':{'kernelspec':{'display_name':'Python3','language':'python','name':'python3'},'language_info':{'name':'python'}},'nbformat':4,'nbformat_minor':5}
    target=ROOT/'notebooks/grow_temporal_difference_colab.ipynb';target.write_text(json.dumps(notebook,indent=1)+'\n');return target


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--source-commit')
    print(build(parser.parse_args().source_commit))
