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
        subprocess.run(['git','-C',str(ROOT),'cat-file','-e',source_commit+':experiments/wan_state_clock/grow_media_offline.py'],check=True)
    cells=[]
    def cell(kind,name,text):
        row={'cell_type':kind,'id':name,'metadata':{},'source':text.splitlines(keepends=True)}
        if kind=='code':row.update(execution_count=None,outputs=[])
        cells.append(row)
    cell('code','drive-mount',"from google.colab import drive\ndrive.mount('/content/drive')\n")
    cell('markdown','scope',"""# GROW 已存媒体张量：CPU离线逐位传递诊断

直接“全部运行”，无需GPU。固定读取已完成媒体run
`MyDrive/Video-WM/GROWTemporalDifferenceMedia/grow_temporal_difference_media_20260918T171544536253Z`。
不生成、不VAE、不MP4重编码、不改读出、强度或阈值。

读取12视频×4层共48已存latent，核原SHA并重算23pair/92系数投票。
新增逐系数连续值、每对/频率符号翻转、逐位正确→错误及错误→正确转移、
连续传递拟合与残差；保存全部128位含soft负例，不仅挑六个坏位。
OFF只作参考，不视为误码真值或FPR。signed均值变化不是VAE bug或画质判据。

已知结果hard完整恢复8→8→8→4；MP4六错误含三擦除。
六坏位soft为正，但全量soft另错10位、仅3/8，所以本次不替换接收器。
输出独立 `MyDrive/Video-WM/GROWMediaOffline/grow_media_offline_<UTC>`，
保存诊断JSON、源配置/codebook/哈希、source.zip和日志。CPU读取约362MB已存latent，
不复制大RGB或MP4。仅使用Colab已有torch，不安装diffusers或重装GPU栈。
固定诊断源码 SHA："""+str(source_commit)+"""。
""")
    cell('code','source-and-software',f"""from pathlib import Path
from datetime import datetime, timezone
import importlib.metadata, json, os, signal, subprocess, sys
SOURCE_COMMIT = {source_commit!r}
if SOURCE_COMMIT is None:
    raise RuntimeError('Pending source publication: rebuild with the published full SHA')
SOURCE_URL = 'https://github.com/RICHAAARC/SC-SSTW.git'
RUN_ID = 'grow_media_offline_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
SOURCE = Path('/content') / (RUN_ID + '_source')
subprocess.run(['git', 'init', str(SOURCE)], check=True)
subprocess.run(['git', '-C', str(SOURCE), 'remote', 'add', 'origin', SOURCE_URL], check=True)
subprocess.run(['git', '-C', str(SOURCE), 'fetch', '--depth', '1', 'origin', SOURCE_COMMIT], check=True)
subprocess.run(['git', '-C', str(SOURCE), 'checkout', '--detach', SOURCE_COMMIT], check=True)
if subprocess.check_output(['git', '-C', str(SOURCE), 'rev-parse', 'HEAD'], text=True).strip() != SOURCE_COMMIT:
    raise RuntimeError('Source checkout differs from pinned SHA')
import torch
print('CPU-only saved-tensor diagnosis; torch:', torch.__version__)
""")
    cell('code','fixed-twelve-videos',"""OUTPUT = Path('/content/drive/MyDrive/Video-WM/GROWMediaOffline') / RUN_ID
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
ARCHIVE = OUTPUT.parent / (RUN_ID + '.source.zip')
subprocess.run(['git', '-C', str(SOURCE), 'archive', '--format=zip', '--output', str(ARCHIVE), SOURCE_COMMIT], check=True)
LOG = OUTPUT.parent / (RUN_ID + '.launcher.log')
INPUT_RUN = Path('/content/drive/MyDrive/Video-WM/GROWTemporalDifferenceMedia/grow_temporal_difference_media_20260918T171544536253Z')
command = [sys.executable, '-u', '-m', 'experiments.wan_state_clock.grow_media_offline', '--output', str(OUTPUT), '--result', str(INPUT_RUN / 'result.json'), '--tensor-root', str(INPUT_RUN)]
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
if (OUTPUT / 'diagnosis.json').exists():
    result = json.loads((OUTPUT / 'diagnosis.json').read_text())
    print(json.dumps({'status': result['status'], 'tensor_summary': result['tensor_summary'], 'actual_calls':result['actual_calls'], 'layers': result['layers']}, indent=2))
if code:
    raise subprocess.CalledProcessError(code, command)
""")
    notebook={'cells':cells,'metadata':{'kernelspec':{'display_name':'Python3','language':'python','name':'python3'},'language_info':{'name':'python'}},'nbformat':4,'nbformat_minor':5}
    target=ROOT/'notebooks/grow_media_offline_colab.ipynb';target.write_text(json.dumps(notebook,indent=1)+'\n');return target


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--source-commit')
    print(build(parser.parse_args().source_commit))
