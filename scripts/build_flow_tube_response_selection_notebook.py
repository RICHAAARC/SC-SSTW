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
        subprocess.run(['git','-C',str(ROOT),'cat-file','-e',source_commit+':experiments/wan_state_clock/flow_tube_response_selection_run.py'],check=True)
    cells=[]
    def cell(kind,name,text):
        row={'cell_type':kind,'id':name,'metadata':{},'source':text.splitlines(keepends=True)}
        if kind=='code':row.update(execution_count=None,outputs=[])
        cells.append(row)
    cell('code','drive-mount',"from google.colab import drive\ndrive.mount('/content/drive')\n")
    cell('markdown','scope',"""# T46 有限幅度响应选择（准备版）

2原开发内容 × OFF/LOCAL_A/B/RESPONSE_A/B = 10正式视频 / 40编码 / 4固定配对。
载体、状态码、margin1、native50及MP4盲读保持。固定T46，测量 +epsilon / -epsilon 两条无梯度真实续程的终态hinge loss。
两符号共用一次unit-native probe得到的epsilon，R49来自同case/message未来LAST oracle，仅机制诊断，不是可部署统一预算。
只选择 -epsilon、0、+epsilon：超过固定数值deadband才选改善方向，均不改善则正式零控制；双方合格且平局取+。
缺任一探测则保留RESPONSE失败；LOCAL独立继续。ZERO方向仍执行完整零路径。无梯度、无扫描、无额外放大。
所有正式尾程从原始完整history新clone重放，绝不拿probe成品替代。全部正式媒体固定落盘读回，无loss筛选门。

全轮296TF / 168native / 4unit probe / 0backward / 10decode / 10MP4save / 40encode。
看selection_summary（正/负/skip/zero/invalid）、paired_summary（实际终态/MP4 gain vs LOCAL）及mechanism_summary。
全部选+意味着退化为LOCAL；skip/zero偶然判对不是水印写入成功，失败仍留固定分母。
secant是有限幅度诊断而非梯度定步长；同幅度probe/formal相符仅工程重放，不是独立预测成功。
OFF差异质量不是感知通过，OFF排名不是FPR，两开发内容不是独立泛化；请人工检查视频内容/伪影/运动。
仅完成CPU/fake/static实现验证，尚未真实模型运行。各探测/正式路径独立记录calls及perf_counter耗时。
新输出FlowTubeResponseSelection。旧identity notebook及方法未改。
源码SHA："""+str(source_commit)+"""。
""")
    cell('code','source-and-software',f"""from pathlib import Path
from datetime import datetime, timezone
import importlib.metadata, json, os, signal, subprocess, sys
SOURCE_COMMIT = {source_commit!r}
if SOURCE_COMMIT is None:
    raise RuntimeError('Pending source publication: rebuild with the published full SHA')
SOURCE_URL = 'https://github.com/RICHAAARC/SC-SSTW.git'
RUN_ID = 'flow_tube_response_selection_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
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
    cell('code','fixed-ten-videos',"""OUTPUT = Path('/content/drive/MyDrive/Video-WM/FlowTubeResponseSelection') / RUN_ID
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
ARCHIVE = OUTPUT.parent / (RUN_ID + '.source.zip')
subprocess.run(['git', '-C', str(SOURCE), 'archive', '--format=zip', '--output', str(ARCHIVE), SOURCE_COMMIT], check=True)
LOG = OUTPUT.parent / (RUN_ID + '.launcher.log')
command = [sys.executable, '-u', '-m', 'experiments.wan_state_clock.flow_tube_response_selection_run', '--output', str(OUTPUT)]
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
    print(json.dumps({k:result.get(k) for k in ('status','video_denominator','receiver_encode_denominator','fixed_calls','actual_calls_observed','selection_summary','recovery_summary','equivalence_summary','mechanism_summary','paired_summary','quality_summary','OFF_summary')}, indent=2))
if code:
    raise subprocess.CalledProcessError(code, command)
""")
    notebook={'cells':cells,'metadata':{'kernelspec':{'display_name':'Python3','language':'python','name':'python3'},'language_info':{'name':'python'}},'nbformat':4,'nbformat_minor':5}
    target=ROOT/'notebooks/flow_tube_response_selection_colab.ipynb';target.write_text(json.dumps(notebook,indent=1)+'\n');return target


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--source-commit')
    print(build(parser.parse_args().source_commit))
