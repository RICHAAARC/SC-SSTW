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
        subprocess.run(['git','-C',str(ROOT),'cat-file','-e',source_commit+':experiments/wan_state_clock/flow_tube_crop_run.py'],check=True)
    cells=[]
    def cell(kind,name,text):
        row={'cell_type':kind,'id':name,'metadata':{},'source':text.splitlines(keepends=True)}
        if kind=='code':row.update(execution_count=None,outputs=[])
        cells.append(row)
    cell('code','drive-mount',"from google.colab import drive\ndrive.mount('/content/drive')\n")
    cell('markdown','scope',"""# 原管状块接收器：固定真实RGB裁剪诊断（准备版）

仅实现/CPU合成检查，本地未执行真实科研评分或GPU。Run all只加载VAE，不运行Transformer。
固定复用开发run `flow_tube_state_guidance_20260919T202545864571Z` 的4case OFF/LAST_A/B共12源MP4，
验证原source SHA与文件hash。原8marked源各3个相关裁剪，非24个独立生成样本。
每源真实RGB起点0/4/5、长度129，共36片段。start0也是129帧截短对照，不是完整181帧baseline。
裁剪存uint8.npy并实际回读；不再MP4压缩，避免额外codec混杂。
每片4内部origin真实RGB输入129/125/125/125，固定144VAEencode；0Transformer/0decode/0MP4save。

原receiver/publicbook和11窗分母、missing窗惩罚、搜索表均不改。保留原五mode，另从盲算candidate/class
提取固定g0/scale1/offset0/delta0/boundary11的无搜索matched/state/without_update三对照。
原global vs local并不是无搜索vs搜索。真起点/消息只在攻击生成和read之后报告，不进接收分数。

offset正号把received中心映射回source。start0/4/5的几何参考phase0/0/3、offset0/4/5，
结构完整窗8/7/7（1280/1120/1120支持）；实际可用参考窗另报。
只比较可观测完整窗的origin/组索引与等价类，缺失不计匹配；不声称逐帧定位/逆VAE精确对齐。
同时报告best映射、top ties是否含参考类、不同observation class数；消息唯一不等于时间路径唯一。
OFF只排名不FPR；每start固定8marked，失败完整保留。间隔是二候选归因，不是payload比特恢复。

新输出 `MyDrive/Video-WM/FlowTubeCrop/flow_tube_crop_<UTC>`；源run只读不覆写。
保存source路径/hash、裁剪raw range、npy/144观测、盲detection与完整逐行结果。
stdout仅24行start×mode紧凑摘要，细节在result.json；无新门槛/补帧/扫描。
源码SHA："""+str(source_commit)+"""。
""")
    cell('code','source-and-software',f"""from pathlib import Path
from datetime import datetime, timezone
import importlib.metadata, json, os, signal, subprocess, sys
SOURCE_COMMIT = {source_commit!r}
if SOURCE_COMMIT is None:
    raise RuntimeError('Pending source publication: rebuild with the published full SHA')
SOURCE_URL = 'https://github.com/RICHAAARC/SC-SSTW.git'
RUN_ID = 'flow_tube_crop_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
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
    cell('code','fixed-thirtysix-crops',"""OUTPUT = Path('/content/drive/MyDrive/Video-WM/FlowTubeCrop') / RUN_ID
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
ARCHIVE = OUTPUT.parent / (RUN_ID + '.source.zip')
subprocess.run(['git', '-C', str(SOURCE), 'archive', '--format=zip', '--output', str(ARCHIVE), SOURCE_COMMIT], check=True)
LOG = OUTPUT.parent / (RUN_ID + '.launcher.log')
command = [sys.executable, '-u', '-m', 'experiments.wan_state_clock.flow_tube_crop_run', '--output', str(OUTPUT)]
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
    print(json.dumps({k:result.get(k) for k in ('status','source_video_denominator','fragment_denominator','marked_denominator','receiver_encode_denominator','fixed_calls','actual_calls_observed','compact_summary')}, indent=2))
if code:
    raise subprocess.CalledProcessError(code, command)
""")
    notebook={'cells':cells,'metadata':{'kernelspec':{'display_name':'Python3','language':'python','name':'python3'},'language_info':{'name':'python'}},'nbformat':4,'nbformat_minor':5}
    target=ROOT/'notebooks/flow_tube_crop_colab.ipynb';target.write_text(json.dumps(notebook,indent=1)+'\n');return target


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--source-commit')
    print(build(parser.parse_args().source_commit))
