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
        subprocess.run(['git','-C',str(ROOT),'cat-file','-e',source_commit+':experiments/wan_state_clock/grow_single_step_jacobian_full.py'],check=True)
    cells=[]
    def cell(kind,name,text):
        row={'cell_type':kind,'id':name,'metadata':{},'source':text.splitlines(keepends=True)}
        if kind=='code':row.update(execution_count=None,outputs=[])
        cells.append(row)
    cell('code','drive-mount',"from google.colab import drive\ndrive.mount('/content/drive')\n")
    cell('markdown','scope',"""# GROW 第30步同状态：局部方向与完整输入 Jacobian（准备版）

本轮仅实现及CPU合成测试，未执行真实科研实验。请用户选择GPU后全部运行。
固定4case × OFF/LOCAL_A/B/JAC_A/B =20视频，全部四层媒体链=80层，不按terminal筛选。
所有arm共享同一个已保存prefix0..29、step30状态和scheduler历史。
LOCAL固定eta.1，g_clean来自原始差分DCT载体MSE；JAC是当前一步完整CFG图对z_t的输入梯度，
模型eval且权重冻结，官方nonreentrant checkpoint重算。图仅含step30；无tail或VAE反向。
JAC仍通过同速度接口 v-u/sigma 注入，因此不是直接更新z_t，局部loss下降不是门槛。
JAC一次unit native probe，以sqrt(E_LOCAL/E_unit)匹配同历史实际state response平方均值，
无扫描、裁剪、重试或读出择优。能量相等不等于画质、终态位移或峰值相等。

先完成该case所有arm的single-step finite可行性记录，再无图tail31..49。
OOM/不匹配等失败完整保留，不改为local、不降精度冒充Jacobian。
记录梯度范数/cos、带图与无图速度差、每方向walltime/显存baseline/peak及checkpoint block重算入口数。
checkpoint重算不是完整Transformer调用。原载体、消息、硬接收器和native UniPC不变。
总计划1024Transformer前向（不含内部block replay）、520live steps、8local gradient、
8input VJP、8unit probe；媒体20decode/60encode/20MP4。每组8marked/128位/层；OFF不作FPR。
保存共享prefix latent+完整scheduler、方向、step30/terminal与媒体产物及哈希。
媒体floatRGB/uint8RGB/MP4层用于定位传递；PSNR仅诊断，需人工看提示词内容、伪影与连贯性。
中间RGB约10.7GB加其他产物，无新增硬件型号门槛。
输出 `MyDrive/Video-WM/GROWSingleStepJacobian/grow_single_step_jacobian_<UTC>`。
源码SHA："""+str(source_commit)+"""。CPU测试不证明GPU可行或科学效果。
""")
    cell('code','source-and-software',f"""from pathlib import Path
from datetime import datetime, timezone
import importlib.metadata, json, os, signal, subprocess, sys
SOURCE_COMMIT = {source_commit!r}
if SOURCE_COMMIT is None:
    raise RuntimeError('Pending source publication: rebuild with the published full SHA')
SOURCE_URL = 'https://github.com/RICHAAARC/SC-SSTW.git'
RUN_ID = 'grow_single_step_jacobian_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
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
    cell('code','fixed-twelve-videos',"""OUTPUT = Path('/content/drive/MyDrive/Video-WM/GROWSingleStepJacobian') / RUN_ID
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
ARCHIVE = OUTPUT.parent / (RUN_ID + '.source.zip')
subprocess.run(['git', '-C', str(SOURCE), 'archive', '--format=zip', '--output', str(ARCHIVE), SOURCE_COMMIT], check=True)
LOG = OUTPUT.parent / (RUN_ID + '.launcher.log')
command = [sys.executable, '-u', '-m', 'experiments.wan_state_clock.grow_single_step_jacobian_full', '--output', str(OUTPUT)]
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
    print(json.dumps({'status': result['status'], 'video_denominator': result['video_denominator'], 'recovery_summary': result.get('recovery_summary'), 'direction_diagnostics':result.get('direction_diagnostics'), 'budget_comparison':result.get('budget_comparison'), 'quality_summary':result.get('quality_summary'), 'stage_status': {k:v['status'] for k,v in result['stages'].items()}, 'fixed_calls': result['fixed_calls'], 'actual_calls_observed': result.get('actual_calls_observed'), 'layer_denominator':result['layer_denominator']}, indent=2))
if code:
    raise subprocess.CalledProcessError(code, command)
""")
    notebook={'cells':cells,'metadata':{'kernelspec':{'display_name':'Python3','language':'python','name':'python3'},'language_info':{'name':'python'}},'nbformat':4,'nbformat_minor':5}
    target=ROOT/'notebooks/grow_single_step_jacobian_colab.ipynb';target.write_text(json.dumps(notebook,indent=1)+'\n');return target


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--source-commit')
    print(build(parser.parse_args().source_commit))
