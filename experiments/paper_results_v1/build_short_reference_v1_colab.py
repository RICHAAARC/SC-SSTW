"""Build the fixed CPU-only user notebook and its ordinary source ZIP."""
import json
from pathlib import Path
import textwrap
import zipfile
from experiments.paper_results_v1.build_residual_controls_v1_colab import cell

ROOT=Path(__file__).resolve().parents[2]
NAME='paper_results_v1_short_reference_v1'


def companion_files():
    return [ROOT/p for p in (
        'main/__init__.py','main/tube_state/__init__.py','experiments/__init__.py','experiments/paper_results_v1/__init__.py',
        'main/tube_state/video_trajectory_payload_framewise_sync_v1.py',
        'main/tube_state/video_trajectory_temporal_edit_receiver_v1.py','main/tube_state/receiver_controls_v1.py',
        'experiments/paper_results_v1/resume_state.py','experiments/paper_results_v1/short_reference_v1.py',
        'experiments/paper_results_v1/short_reference_v1_cli.py')]


def build():
    folder=ROOT/'notebooks';folder.mkdir(exist_ok=True)
    archive_path=folder/f'{NAME}_companion.zip'
    with zipfile.ZipFile(archive_path,'w',zipfile.ZIP_DEFLATED) as archive:
        for path in companion_files():archive.write(path,path.relative_to(ROOT).as_posix())
    intro='''# A：固定短片与多数重复的接收参考分析

直接在 CPU Colab **Run all**。读取原独立四源任务保存的真实 framewise Y，不加载模型或读取载荷；同 Drive 输出可恢复。CENTERED 仍为默认，本入口是固定压力条件分析，不预设 REF 更好。

四源全部保留。起点固定37；长度9/17各一档，另在各自几何中点额外重复10/18帧，形成19/35帧多数重复视图。四源×四视图×RAW_U/CENTERED_U/原唯一DC参考=48行、48次CPU selector、960个臂×接收位置。只有4个独立来源，重复不增加独立证据。没有长度/位置/系数扫描。

REF 使用同patch逐通道均值及公开方向逐patch补偿，未再叠加列居中。多数重复是预声明机制压力，不代表普遍真实视频失败；数学单帧复制例子不计为实测。输出全部精确/错误/未解决/不支持/缺失行、支持和CPU调用数；没有新BER或GPU结论。
'''
    setup=textwrap.dedent('''\
        import json, os, subprocess, sys, urllib.request, uuid, zipfile
        from pathlib import Path
        SOURCE_RUN=Path('/content/drive/MyDrive/Video-WM/Paper-Results-V1-Independent-Four-V1/resumable-independent-four-v1')
        OUTPUT=Path('/content/drive/MyDrive/Video-WM/Paper-Results-V1-Short-Reference-V1/resumable-short-reference-v1')
        OUTPUT.mkdir(parents=True,exist_ok=True)
        SOURCE_URL='https://raw.githubusercontent.com/RICHAAARC/SC-SSTW/dev/paper-results-v1/notebooks/paper_results_v1_short_reference_v1_companion.zip'
        ARCHIVE=OUTPUT/'companion_source.zip'
        CODE=Path('/content')/('short-reference-'+uuid.uuid4().hex[:10]);CODE.mkdir()
        def readable_archive(path):
            with zipfile.ZipFile(path) as archive:
                archive.read('experiments/paper_results_v1/short_reference_v1_cli.py')
                for member in archive.infolist():
                    target=(CODE/member.filename).resolve()
                    if target!=CODE.resolve() and CODE.resolve() not in target.parents:
                        raise ValueError('unsafe source archive path')
                if archive.testzip() is not None: raise ValueError('unreadable source ZIP member')
        try:
            pending=ARCHIVE.with_suffix('.download.zip')
            with urllib.request.urlopen(SOURCE_URL,timeout=60) as response:
                pending.write_bytes(response.read())
            readable_archive(pending)
            pending.replace(ARCHIVE)
        except Exception as exc:
            print('Source download unavailable:',type(exc).__name__,str(exc))
            readable_archive(ARCHIVE)
            print('Using readable source ZIP already saved in this Drive output.')
        with zipfile.ZipFile(ARCHIVE) as archive:
            archive.extractall(CODE)
        try: import numpy
        except ImportError: subprocess.run([sys.executable,'-m','pip','install','numpy'],check=True)
        print('Available source loaded. Fixed source:',SOURCE_RUN,'Output:',OUTPUT)
    ''')
    run=textwrap.dedent('''\
        command=[sys.executable,'-u','-m','experiments.paper_results_v1.short_reference_v1_cli',
                 '--source-run',str(SOURCE_RUN),'--output',str(OUTPUT)]
        with (OUTPUT/'execution.log').open('a') as log:
            log.write('COMMAND '+repr(command)+'\\n');log.flush()
            completed=subprocess.run(command,cwd=CODE,env={**os.environ,'PYTHONPATH':str(CODE)},stdout=log,stderr=subprocess.STDOUT)
        if completed.returncode:
            raise RuntimeError('CPU analysis failed; inspect '+str(OUTPUT/'execution.log'))
        summary=json.loads((OUTPUT/'summary.json').read_text())
        print(json.dumps(summary,ensure_ascii=False,indent=2))
        print('Full fixed rows:',OUTPUT/'row_results.json')
    ''')
    notebook=dict(nbformat=4,nbformat_minor=5,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python'}},cells=[
        cell('code',"from google.colab import drive\ndrive.mount('/content/drive')\n",'drive-mount'),
        cell('markdown',intro,'scope'),cell('code',setup,'load-source'),
        cell('markdown','## 固定48行CPU计算与同Drive恢复\n完成的方法结果不重跑；缺失输入保留MISSING，不自动编码补造Y。\n','run-title'),
        cell('code',run,'run-and-report')])
    nb=folder/f'{NAME}_colab.ipynb';nb.write_text(json.dumps(notebook,ensure_ascii=False,indent=1)+'\n')
    return {'notebook':str(nb),'companion':str(archive_path),'files':len(companion_files())}


if __name__=='__main__':print(json.dumps(build(),indent=2))
