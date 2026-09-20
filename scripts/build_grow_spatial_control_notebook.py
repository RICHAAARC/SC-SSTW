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
        subprocess.run(['git','-C',str(ROOT),'cat-file','-e',source_commit+':experiments/wan_state_clock/grow_spatial_control_full.py'],check=True)
    cells=[]
    def cell(kind,name,text):
        row={'cell_type':kind,'id':name,'metadata':{},'source':text.splitlines(keepends=True)}
        if kind=='code':row.update(execution_count=None,outputs=[])
        cells.append(row)
    cell('code','drive-mount',"from google.colab import drive\ndrive.mount('/content/drive')\n")
    cell('markdown','scope',"""# 原空间DCT载体：MULTI/LAST完整媒体配对（准备版）

不是官方GROW直接复现，也不是首次尝试空间载体。旧10..29空间载体完整媒体0/8；旧30..49终态3/8未自动媒体。
本轮补齐30..49空间载体的MULTI/LAST对照；原MULTI终态部分是明确重复，不包装新方法。
固定4dev×OFF/MULTI_A/B/LAST_A/B=20视频，每视频terminal/float RGB回编码/uint8回编码/真实MP4回编码=80层，全部执行不gate。

每latent时间片channel0的40×64空间DCT，频率[2..9]²，16bit、4频repeat×46time=184票，无相邻时间差分，无新同步。
原MSE half-sum、amplitude.5、eta.1。MULTI固定30..49，LAST仅49。
继续采用旧Wan的post-CFG clean leaf梯度，u=-eta*grad，v'=v-u/sigma。
官方先conditional-clean局部梯度（masked mean MSE），反推conditional噪声后才CFG；参数/归一化/CFG位置与本候选不同。

E_MULTI=sum RMS(native_controlled_next−same_history_shadow_OFF_next)^2；LAST单位probe测Eunit，再一次scale=sqrt(E_MULTI/Eunit)匹配。
这是同case/message事后配对预算，不是统一部署强度、终态能量或等感知质量；实测误差保留，不扫参重试或依读出挑选。
保留peak/sum/RMS²、净终态差、每步loss/实际native响应/末步clean和实际输出。
旧late存在initial/embedding/terminal和逐步预算，但缺完整OFF49 UniPC history和直接v49快照，不能零生成无损恢复LAST。
本轮固定重新生成20视频，不引入OFF重播验证复用分支。

固定hard和rawsoft都报告，不择优；候选A/B排名不是16bit完整恢复，也不是存在检测或FPR。
PSNR只相对OFF残差，感知质量未测，请观看生成视频内容/伪影/连贯性。
全轮2000TF/1000native/168local-gradient/168OFF-shadow/8unit-probe；20decode/60encode/20MP4save/read。
新路径 `MyDrive/Video-WM/GROWSpatialCarrier/grow_spatial_control_<UTC>`。保存日志/config/source/hash/张量，约10.7GB RGB中间结果另加latent。
仅CPU/fake/static验证，未自动运行GPU或真实模型。固定源码："""+str(source_commit)+"""。
""")
    cell('code','source-and-software',f"""from pathlib import Path
from datetime import datetime, timezone
import importlib.metadata, json, os, signal, subprocess, sys
SOURCE_COMMIT = {source_commit!r}
if SOURCE_COMMIT is None:
    raise RuntimeError('Pending source publication: rebuild with the published full SHA')
SOURCE_URL = 'https://github.com/RICHAAARC/SC-SSTW.git'
RUN_ID = 'grow_spatial_control_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
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
    cell('code','fixed-twenty-videos',"""OUTPUT = Path('/content/drive/MyDrive/Video-WM/GROWSpatialCarrier') / RUN_ID
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
ARCHIVE = OUTPUT.parent / (RUN_ID + '.source.zip')
subprocess.run(['git', '-C', str(SOURCE), 'archive', '--format=zip', '--output', str(ARCHIVE), SOURCE_COMMIT], check=True)
LOG = OUTPUT.parent / (RUN_ID + '.launcher.log')
command = [sys.executable, '-u', '-m', 'experiments.wan_state_clock.grow_spatial_control_full', '--output', str(OUTPUT)]
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
    print(json.dumps({'status': result['status'], 'video_denominator': result['video_denominator'], 'reader_factorial_summary': result.get('reader_factorial_summary'), 'budget_comparison':result.get('budget_comparison'), 'candidate_attribution_summary':result.get('candidate_attribution_summary'), 'quality_summary':result.get('quality_summary'), 'stage_status': {k:v['status'] for k,v in result['stages'].items()}, 'fixed_calls': result['fixed_calls'], 'actual_calls_observed': result.get('actual_calls_observed'), 'layer_denominator':result['layer_denominator']}, indent=2))
if code:
    raise subprocess.CalledProcessError(code, command)
""")
    notebook={'cells':cells,'metadata':{'kernelspec':{'display_name':'Python3','language':'python','name':'python3'},'language_info':{'name':'python'}},'nbformat':4,'nbformat_minor':5}
    target=ROOT/'notebooks/grow_spatial_control_colab.ipynb';target.write_text(json.dumps(notebook,indent=1)+'\n');return target


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--source-commit')
    print(build(parser.parse_args().source_commit))
