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
        subprocess.run(['git','-C',str(ROOT),'cat-file','-e',source_commit+':experiments/wan_state_clock/grow_temporal_difference_media.py'],check=True)
    cells=[]
    def cell(kind,name,text):
        row={'cell_type':kind,'id':name,'metadata':{},'source':text.splitlines(keepends=True)}
        if kind=='code':row.update(execution_count=None,outputs=[])
        cells.append(row)
    cell('code','drive-mount',"from google.colab import drive\ndrive.mount('/content/drive')\n")
    cell('markdown','scope',"""# GROW 时间差分：已存终态的媒体验证

选择 GPU 后点击“全部运行”。固定读取已完成的
`MyDrive/Video-WM/GROWTemporalDifference/grow_temporal_difference_20260918T142825573989Z`。
四case OFF/A/B 共12个终态全部保留；不会生成视频潜变量或加载Transformer。
先核原文件SHA256、配置和终态读出，源目录不被修改，缺失/失败不缩减分母。

每个终态只decode一次，分别测 terminal、float RGB无压缩重编码、RGB8量化重编码、
CRF18 MP4保存回读重编码，共48层。每层使用同一23对/92票/16位固定差分读出。
预算12 VAE decode、36 encode、12 MP4保存和12回读，Transformer调用为0。
不会修改强度、筛选样本、自动补跑或进入其他实验。

新输出：`MyDrive/Video-WM/GROWTemporalDifferenceMedia/grow_temporal_difference_media_<UTC>`。
保存原终态副本、中间重编码latent、float RGB与两种uint8栅格、MP4及文件哈希，
便于区分VAE、量化和视频编解码影响。完整RGB证据约6.4GB，另有latent和MP4；
保留足够Drive空间即可，不设置额外设备或空间硬门槛。

相对于同case OFF的PSNR/MSE/时序残差仅诊断，不是质量通过阈值。
请人工观看OFF/A/B：核提示词内容、细节伪影、闪烁、运动和跨帧连贯性。
执行完成、消息恢复和视觉质量是不同结论；OFF巧合记录不构成FPR校准。
来源缺model revision时会记录精确版本比较限制，不因此重新生成或阻断。

依赖沿用已工作torch2.11.0+cu128/diffusers0.40.0，无GPU型号限制。
本交付仅静态和CPU/Fake核验，尚未运行真实媒体链。
固定媒体源码 SHA："""+str(source_commit)+"""。
""")
    cell('code','source-and-software',f"""from pathlib import Path
from datetime import datetime, timezone
import importlib.metadata, json, os, signal, subprocess, sys
SOURCE_COMMIT = {source_commit!r}
if SOURCE_COMMIT is None:
    raise RuntimeError('Pending source publication: rebuild with the published full SHA')
SOURCE_URL = 'https://github.com/RICHAAARC/SC-SSTW.git'
RUN_ID = 'grow_temporal_difference_media_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
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
    cell('code','fixed-twelve-videos',"""OUTPUT = Path('/content/drive/MyDrive/Video-WM/GROWTemporalDifferenceMedia') / RUN_ID
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
ARCHIVE = OUTPUT.parent / (RUN_ID + '.source.zip')
subprocess.run(['git', '-C', str(SOURCE), 'archive', '--format=zip', '--output', str(ARCHIVE), SOURCE_COMMIT], check=True)
LOG = OUTPUT.parent / (RUN_ID + '.launcher.log')
INPUT_RUN = Path('/content/drive/MyDrive/Video-WM/GROWTemporalDifference/grow_temporal_difference_20260918T142825573989Z')
command = [sys.executable, '-u', '-m', 'experiments.wan_state_clock.grow_temporal_difference_media', '--output', str(OUTPUT), '--source-root', str(INPUT_RUN)]
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
    print(json.dumps({'status': result['status'], 'video_denominator': result['video_denominator'], 'layer_denominator': result['layer_denominator'], 'recovery_summary': result.get('recovery_summary'), 'fixed_calls': result['fixed_calls'], 'actual_calls_observed': result.get('actual_calls_observed'), 'cases': {k: v['status'] for k, v in result['cases'].items()}}, indent=2))
if code:
    raise subprocess.CalledProcessError(code, command)
""")
    notebook={'cells':cells,'metadata':{'kernelspec':{'display_name':'Python3','language':'python','name':'python3'},'language_info':{'name':'python'}},'nbformat':4,'nbformat_minor':5}
    target=ROOT/'notebooks/grow_temporal_difference_media_colab.ipynb';target.write_text(json.dumps(notebook,indent=1)+'\n');return target


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--source-commit')
    print(build(parser.parse_args().source_commit))
