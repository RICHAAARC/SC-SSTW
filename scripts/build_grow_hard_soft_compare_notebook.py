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
        subprocess.run(['git','-C',str(ROOT),'cat-file','-e',source_commit+':experiments/wan_state_clock/grow_hard_soft_compare.py'],check=True)
    cells=[]
    def cell(kind,name,text):
        row={'cell_type':kind,'id':name,'metadata':{},'source':text.splitlines(keepends=True)}
        if kind=='code':row.update(execution_count=None,outputs=[])
        cells.append(row)
    cell('code','drive-mount',"from google.colab import drive\ndrive.mount('/content/drive')\n")
    cell('markdown','scope',"""# GROW 同载体hard/rawsoft固定读出对照（准备版）

仅准备，当前没有对任何真实数据执行新评分。用户之后“全部运行”才读取固定旧MSE媒体run
`MyDrive/Video-WM/GROWTemporalDifferenceMedia/grow_temporal_difference_media_20260918T171544536253Z/result.json`。
不读新margin实验，不生成、不VAE、不GPU，只使用JSON已有相同系数汇总。

hard=sign(sum(sign(d)))；soft=sign(mean(d))，阈值均0，零/平票擦除。
全部128位和OFF分别保留，两种固定规则都报告，不根据正确答案择优切换。
另报告各分数向量与公开A/B消息的均匀相关得分、top/tie/gap；之后再结合真值评价。
A/B二候选归因不等于完整16位恢复或水印存在性判断。历史rawsoft曾3/8低于hard4/8，
这份准备不会删除负例，也不会自动选择新接收器。

独立输出 `MyDrive/Video-WM/GROWHardSoftCompare/grow_hard_soft_compare_<UTC>`，
保存完整48行比较、输入/评分代码哈希、原source_commit及日志/source.zip。
无需安装torch/diffusers，不读取大tensor；标准库CPU执行。
固定诊断源码 SHA："""+str(source_commit)+"""。
""")
    cell('code','source-and-software',f"""from pathlib import Path
from datetime import datetime, timezone
import importlib.metadata, json, os, signal, subprocess, sys
SOURCE_COMMIT = {source_commit!r}
if SOURCE_COMMIT is None:
    raise RuntimeError('Pending source publication: rebuild with the published full SHA')
SOURCE_URL = 'https://github.com/RICHAAARC/SC-SSTW.git'
RUN_ID = 'grow_hard_soft_compare_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
SOURCE = Path('/content') / (RUN_ID + '_source')
subprocess.run(['git', 'init', str(SOURCE)], check=True)
subprocess.run(['git', '-C', str(SOURCE), 'remote', 'add', 'origin', SOURCE_URL], check=True)
subprocess.run(['git', '-C', str(SOURCE), 'fetch', '--depth', '1', 'origin', SOURCE_COMMIT], check=True)
subprocess.run(['git', '-C', str(SOURCE), 'checkout', '--detach', SOURCE_COMMIT], check=True)
if subprocess.check_output(['git', '-C', str(SOURCE), 'rev-parse', 'HEAD'], text=True).strip() != SOURCE_COMMIT:
    raise RuntimeError('Source checkout differs from pinned SHA')
print('CPU-only JSON comparison; no model or tensor execution')
""")
    cell('code','fixed-twelve-videos',"""OUTPUT = Path('/content/drive/MyDrive/Video-WM/GROWHardSoftCompare') / RUN_ID
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
ARCHIVE = OUTPUT.parent / (RUN_ID + '.source.zip')
subprocess.run(['git', '-C', str(SOURCE), 'archive', '--format=zip', '--output', str(ARCHIVE), SOURCE_COMMIT], check=True)
LOG = OUTPUT.parent / (RUN_ID + '.launcher.log')
INPUT_RUN = Path('/content/drive/MyDrive/Video-WM/GROWTemporalDifferenceMedia/grow_temporal_difference_media_20260918T171544536253Z')
command = [sys.executable, '-u', '-m', 'experiments.wan_state_clock.grow_hard_soft_compare', '--output', str(OUTPUT), '--input-result', str(INPUT_RUN / 'result.json')]
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
if (OUTPUT / 'comparison.json').exists():
    result = json.loads((OUTPUT / 'comparison.json').read_text())
    print(json.dumps({'status': result['status'], 'actual_calls':result['actual_calls'], 'summary': result['summary']}, indent=2))
if code:
    raise subprocess.CalledProcessError(code, command)
""")
    notebook={'cells':cells,'metadata':{'kernelspec':{'display_name':'Python3','language':'python','name':'python3'},'language_info':{'name':'python'}},'nbformat':4,'nbformat_minor':5}
    target=ROOT/'notebooks/grow_hard_soft_compare_colab.ipynb';target.write_text(json.dumps(notebook,indent=1)+'\n');return target


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--source-commit')
    print(build(parser.parse_args().source_commit))
