"""Build the directly runnable segmented-payload notebook and ordinary source ZIP."""
import json
from pathlib import Path
import textwrap
import zipfile
from experiments.paper_results_v1.build_residual_controls_v1_colab import cell

ROOT=Path(__file__).resolve().parents[2]
NAME='paper_results_v1_distinct_confirmation_p2'


def companion_files():
    # Runtime closure plus an explicit outer-layer list; do not package P1,
    # development comparisons, paper WIP or arbitrary experiments/*.py.
    paths=[ROOT/f'{name}/__init__.py' for name in ('main','runtime','experiments')]
    for folder in ('main/tube_state','runtime/wan'):
        paths.extend((ROOT/folder).rglob('*.py'))
    names=('__init__.py','resume_state.py','short_reference_v1.py','distinct_confirmation_p2.py',
           'distinct_confirmation_p2_diagnostics.py','distinct_confirmation_p2_cli.py',
           'distinct_confirmation_p2.adopted.json','real_backends.py','native_adapters.py','attack_matrix.py')
    paths.extend(ROOT/'experiments/paper_results_v1'/n for n in names)
    return sorted(set(paths))


def build():
    output=ROOT/'notebooks';output.mkdir(exist_ok=True)
    archive_path=output/(NAME+'_companion.zip')
    with zipfile.ZipFile(archive_path,'w',zipfile.ZIP_DEFLATED) as archive:
        for path in companion_files():
            info=zipfile.ZipInfo(path.relative_to(ROOT).as_posix(),date_time=(1980,1,1,0,0,0))
            info.compress_type=zipfile.ZIP_DEFLATED;info.external_attr=0o100644<<16
            archive.writestr(info,path.read_bytes())
    setup=textwrap.dedent('''\
        import json, os, signal, subprocess, sys, urllib.request, uuid, zipfile
        from pathlib import Path
        OUTPUT=Path('/content/drive/MyDrive/Video-WM/Paper-Results-V1-Distinct-Confirmation-P2/resumable-distinct-confirmation-p2')
        OUTPUT.mkdir(parents=True,exist_ok=True)
        ARCHIVE=OUTPUT/'companion_source.zip'
        URL='https://raw.githubusercontent.com/RICHAAARC/SC-SSTW/dev/paper-results-v1/notebooks/paper_results_v1_distinct_confirmation_p2_companion.zip'
        CODE=Path('/content')/('distinct-confirmation-p2-'+uuid.uuid4().hex[:10]);CODE.mkdir()
        def readable_archive(path):
            with zipfile.ZipFile(path) as archive:
                archive.read('experiments/paper_results_v1/distinct_confirmation_p2_cli.py')
                for member in archive.infolist():
                    target=(CODE/member.filename).resolve()
                    if target!=CODE.resolve() and CODE.resolve() not in target.parents:
                        raise ValueError('unsafe source archive path')
                if archive.testzip() is not None: raise ValueError('unreadable source ZIP member')
        try:
            pending=ARCHIVE.with_suffix('.download.zip')
            with urllib.request.urlopen(URL,timeout=60) as response: pending.write_bytes(response.read())
            readable_archive(pending);pending.replace(ARCHIVE)
        except Exception as exc:
            print('Source download unavailable:',type(exc).__name__,str(exc))
            readable_archive(ARCHIVE)
            print('Using readable source ZIP already in this Drive output.')
        with zipfile.ZipFile(ARCHIVE) as archive: archive.extractall(CODE)
        ENV={**os.environ,'PYTHONPATH':str(CODE)}
        CLI=[sys.executable,'-u','-m','experiments.paper_results_v1.distinct_confirmation_p2_cli']
        LOG=OUTPUT/'execution.log'
        def command(args,label,check=True):
            print(label,flush=True)
            with LOG.open('a') as stream:
                stream.write('COMMAND '+repr(args)+'\\n');stream.flush()
                child=subprocess.Popen(args,cwd=CODE,env=ENV,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
                try: code=child.wait()
                except BaseException:
                    os.killpg(child.pid,signal.SIGTERM)
                    try:child.wait(timeout=10)
                    except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
                    raise
            if code and check:raise RuntimeError(label+' failed; see '+str(LOG))
            return code
        def ensure_imports(probe,packages,label):
            import re
            module_packages={package.replace('-','_'):package for package in packages}
            repaired=set()
            while True:
                offset=LOG.stat().st_size if LOG.exists() else 0
                if not command([sys.executable,'-c',probe],('Recheck ' if repaired else 'Probe ')+label,check=False):return
                with LOG.open('rb') as stream:
                    stream.seek(offset);failure=stream.read().decode('utf-8',errors='replace')
                missing=re.search("ModuleNotFoundError: No module named ['\\\"]([^'\\\"]+)['\\\"]",failure)
                if "cannot import name 'FqnToConfig'" in failure and 'torchao.quantization' in failure:
                    # This experiment does not use torchao quantization. Remove only
                    # this observed optional-package conflict, then probe afresh.
                    action='remove incompatible optional torchao'
                    args=[sys.executable,'-m','pip','uninstall','-y','torchao']
                elif missing and missing.group(1) in module_packages:
                    package=module_packages[missing.group(1)]
                    action='install missing '+package
                    args=[sys.executable,'-m','pip','install',package]
                else:
                    raise RuntimeError(label+' failed without a known targeted repair; see '+str(LOG))
                if action in repaired:raise RuntimeError(label+' still fails after '+action+'; see '+str(LOG))
                repaired.add(action)
                command(args,'Repair '+label+': '+action)
        ensure_imports('import numpy',['numpy'],'array dependency')
        command([*CLI,'--output',str(OUTPUT),'--worker','init'],'Initialize fixed 196 rows')
        summary=json.loads((OUTPUT/'summary.json').read_text())
        NEED=summary['setup_requirements']
        CACHE=Path('/content/drive/MyDrive/Video-WM/Paper-Results-V1-Cache/models')
        def present(path,name):
            if name=='framewise':return (path/'config.json').is_file() and (path/'diffusion_pytorch_model.safetensors').is_file()
            if NEED['wan_full']:
                return (path/'model_index.json').is_file() and all((path/p).is_dir() for p in ('transformer','text_encoder','tokenizer','scheduler','vae')) and all(list((path/p).glob('*.safetensors')) for p in ('transformer','text_encoder','vae'))
            return (path/'vae/config.json').is_file() and bool(list((path/'vae').glob('*.safetensors')))
        MODELS={}
        for name,folder,repo in [('wan','Wan2.1-T2V-1.3B-Diffusers','Wan-AI/Wan2.1-T2V-1.3B-Diffusers'),('framewise','sd-vae-ft-mse','stabilityai/sd-vae-ft-mse')]:
            base=CACHE/folder
            candidates=[base/'available',base]+([p for p in base.iterdir() if p.is_dir()] if base.is_dir() else [])
            path=next((p for p in candidates if present(p,name)),base/'available')
            MODELS[name]={'path':path,'repo':repo}
        print('Remaining dependency needs:',NEED)
    ''')
    environment=textwrap.dedent('''\
        if NEED['torch']:ensure_imports('import torch',['torch'],'tensor dependency')
        classes=[]
        if NEED['wan_full']:classes.append('WanPipeline')
        if NEED['wan_vae']:classes.append('AutoencoderKLWan')
        if NEED['framewise']:classes.append('AutoencoderKL')
        if classes:
            probe='import safetensors,huggingface_hub; from diffusers import '+','.join(classes)
            packages=['diffusers','accelerate','safetensors','huggingface-hub']
            if NEED['wan_full']:
                probe+='; import transformers,sentencepiece,ftfy'
                packages+=['transformers','sentencepiece','ftfy']
            ensure_imports(probe,packages,'remaining model imports')
        import shutil
        if NEED['ffmpeg'] and (not shutil.which('ffmpeg') or not shutil.which('ffprobe')):
            command(['apt-get','-qq','update'],'Prepare media dependency')
            command(['apt-get','-qq','install','-y','ffmpeg'],'Install FFmpeg')
        targets=[]
        if NEED['wan_full'] or NEED['wan_vae']:targets.append('wan')
        if NEED['framewise']:targets.append('framewise')
        for name in targets:
            model=MODELS[name]
            if not present(model['path'],name):
                patterns=None if name=='wan' and NEED['wan_full'] else ['model_index.json','vae/*'] if name=='wan' else ['config.json','*.safetensors']
                script='from huggingface_hub import snapshot_download; snapshot_download(repo_id='+repr(model['repo'])+', local_dir='+repr(str(model['path']))+', allow_patterns='+repr(patterns)+')'
                command([sys.executable,'-c',script],'Prepare '+name+' model files')
        if not targets:print('No remaining model loads required; saved results can be read directly.')
    ''')
    run=textwrap.dedent('''\
        args=[*CLI,'--output',str(OUTPUT),'--wan-model',str(MODELS['wan']['path']),'--framewise-model',str(MODELS['framewise']['path'])]
        try:command(args,'Run fixed P2 confirmation task')
        finally:command([*CLI,'--output',str(OUTPUT),'--worker','report'],'Save complete fixed-denominator report',check=False)
        summary=json.loads((OUTPUT/'summary.json').read_text())
        print(json.dumps(summary,ensure_ascii=False,indent=2))
        print('Return this full directory:',OUTPUT)
        print('All 196 rows:',OUTPUT/'row_results.json')
    ''')
    intro='# A P2：固定独立来源确认\n\n在GPU Colab中直接 **Run all**。四个预列新来源distinct_confirm_20261017_01..04（风铃、锦鲤、骑行者、旗帜）与seed 2026101701..04不替换。固定八视图CROP_MID/CROP_DELETE/CROP_DUPLICATE/DELETE_S1/SHORT17/SHORT17_HOLD_EXACT/SHORT17_MEDIA/SHORT17_HOLD_MEDIA。四观察臂RAW_U/CENTERED_U/REF_U/DISTINCT_REF_U，另UNALIGNED和ORACLE；每源还有首次candidate发布FULL读回的CLEAN/ORACLE，不能用NATIVE_PRE替代。\n\n完整196逻辑行/6272计划位=3968结构存在+2304结构缺失；128次selector/8128路径位置。先保存全部盲计划，再读clean/oracle/消息；clean未完整精确时该源48攻击行保留BLOCKED_CLEAN。17/35帧使用真实N帧Y，普通视图取行，MEDIA各固定libx264/8fps/CRF18/yuv420p再发布一次；完全重复未被codec打破就保留“近似重复边界未触及”，不重编或调参。\n\n固定writer、强度、gain、U、key、分段cell与strict-zero/Counter投票不变。每视图候选先读；只有同一收到输入的完整映射相同才共享encode，完整plan/cells/key/votes规则相同才共享read，两类alias分别计数。rowmax只作事后诊断，不是载荷臂；并列保留未解决，非单调不排序。\n\n本新源预算：4轨迹/200native steps/400DiT/100新局部payload梯度；不重算旧源梯度。native decode4；framewise encode16clips/216batches、decode8clips/184batches；12codec往返；Wan encode/read各最多196。生成每源load1，native Wan/framewise/reader Wan各只load1，无恢复load预留。未使用预算不可转给新来源或其他条件。\n\n同一Drive目录保存逐单元状态，调用前记录STARTED；Run all复用已完成可读产物，只继续从未开始且预算可用的单元。失败、中断、完成未知或产物损坏不自动重算，模型load预算耗尽会保留缺失/阻断，不承诺所有断点都可继续。本包只做CPU合成与静态验证，真实模型/媒体与跨Colab恢复尚待用户执行。\n\n默认CENTERED不替换。无OFF生成、感知质量模型、外部baseline或扫描；native已含payload，质量只说明clock/codec增量。来源ZIP直接加载可用源码，无摘要/版本/干净状态门禁。首个单元只挂载Drive；源码下载失败可使用同目录已有可读缓存。\n'
    notebook=dict(nbformat=4,nbformat_minor=5,metadata={'accelerator':'GPU','kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python'}},cells=[
        cell('code',"from google.colab import drive\ndrive.mount('/content/drive')\n",'drive-mount'),
        cell('markdown',intro,'scope'),cell('code',setup,'source-and-fixed-plan'),
        cell('markdown','## 准备剩余实际依赖\n','dependencies-title'),cell('code',environment,'dependencies'),
        cell('markdown','## 直接执行并保存全部固定行\n','run-title'),cell('code',run,'run-and-report')])
    path=output/(NAME+'_colab.ipynb');path.write_text(json.dumps(notebook,ensure_ascii=False,indent=1)+'\n')
    return dict(notebook=str(path),companion=str(archive_path),files=len(companion_files()))


if __name__=='__main__':print(json.dumps(build(),indent=2))
