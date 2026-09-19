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
        subprocess.run(['git','-C',str(ROOT),'cat-file','-e',source_commit+':experiments/wan_state_clock/grow_paired_control_full.py'],check=True)
    cells=[]
    def cell(kind,name,text):
        row={'cell_type':kind,'id':name,'metadata':{},'source':text.splitlines(keepends=True)}
        if kind=='code':row.update(execution_count=None,outputs=[])
        cells.append(row)
    cell('code','drive-mount',"from google.colab import drive\ndrive.mount('/content/drive')\n")
    cell('markdown','scope',"""# GROW 同实际控制预算：多步轨迹与仅末步对照（准备版）

本轮仅实现与合成工程测试，未执行真实科研实验。之后由用户选择GPU并“全部运行”。
固定4case，每case OFF、MULTI_A/B、LAST_A/B，共20视频，全部执行四层媒体测量共80层。
原始MSE写入目标固定，差分载体/16位/92票/eta.1不变，不使用单侧margin候选。

MULTI在30..49共20步控制。匹配预算 E=sum(mean((native_controlled_next-
same_history_OFF_next)^2))；LAST此前无控制，仅49处用同历史unit probe测E_unit，
取唯一scale=sqrt(E_multi/E_unit)，一次live更新，不扫描、不重试调整、不根据读出选择。
预算目标由同case同消息MULTI实际响应给出，不是统一校准超参或终态能量。
预算相等不代表相同峰值、终态位移或视觉质量；过强末步结果完整保留。
零响应不可匹配时保留失败，不跳其他样本。

每层同时保存固定hard与rawsoft分数/恢复，以及无真值A/B连续候选得分和top/tie/gap。
二候选归因不等于16位完整恢复或存在检测。禁止择优切换读出、挑pair、调阈值。
原始历史soft负例保留，不把soft预设为改进。

预算：2000Transformer、1000live step、168局部梯度、168same-history OFF shadow、
8unit probe；媒体20decode/60encode/20MP4保存回读。各stage计数分开。
MULTI/LAST各8marked/128位；不按terminal gate筛媒体，缺失保留80层固定槽。
摘要列8对目标/实际E、误差、scale及峰值控制量。PSNR为诊断，请人工看内容/伪影/连贯性。

独立输出 `MyDrive/Video-WM/GROWPairedControl/grow_paired_control_<UTC>`。
保存source.zip、日志、generation/media配置/张量/哈希。中间RGB约10.7GB加其他产物，
不新增型号或空间硬门槛。依赖沿用现有已工作版本，CPU测试不代表GPU/效果成功。
固定源码 SHA："""+str(source_commit)+"""。
""")
    cell('code','source-and-software',f"""from pathlib import Path
from datetime import datetime, timezone
import importlib.metadata, json, os, signal, subprocess, sys
SOURCE_COMMIT = {source_commit!r}
if SOURCE_COMMIT is None:
    raise RuntimeError('Pending source publication: rebuild with the published full SHA')
SOURCE_URL = 'https://github.com/RICHAAARC/SC-SSTW.git'
RUN_ID = 'grow_paired_control_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
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
    cell('code','fixed-twelve-videos',"""OUTPUT = Path('/content/drive/MyDrive/Video-WM/GROWPairedControl') / RUN_ID
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
ARCHIVE = OUTPUT.parent / (RUN_ID + '.source.zip')
subprocess.run(['git', '-C', str(SOURCE), 'archive', '--format=zip', '--output', str(ARCHIVE), SOURCE_COMMIT], check=True)
LOG = OUTPUT.parent / (RUN_ID + '.launcher.log')
command = [sys.executable, '-u', '-m', 'experiments.wan_state_clock.grow_paired_control_full', '--output', str(OUTPUT)]
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
    print(json.dumps({'status': result['status'], 'video_denominator': result['video_denominator'], 'reader_factorial_summary': result.get('reader_factorial_summary'), 'budget_comparison':result.get('budget_comparison'), 'stage_status': {k:v['status'] for k,v in result['stages'].items()}, 'fixed_calls': result['fixed_calls'], 'actual_calls_observed': result.get('actual_calls_observed'), 'layer_denominator':result['layer_denominator']}, indent=2))
if code:
    raise subprocess.CalledProcessError(code, command)
""")
    notebook={'cells':cells,'metadata':{'kernelspec':{'display_name':'Python3','language':'python','name':'python3'},'language_info':{'name':'python'}},'nbformat':4,'nbformat_minor':5}
    target=ROOT/'notebooks/grow_paired_control_colab.ipynb';target.write_text(json.dumps(notebook,indent=1)+'\n');return target


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--source-commit')
    print(build(parser.parse_args().source_commit))
