"""Build the directly runnable segmented-payload notebook and ordinary source ZIP."""
import json
from pathlib import Path
import textwrap
import zipfile
from experiments.paper_results_v1.build_residual_controls_v1_colab import cell

ROOT=Path(__file__).resolve().parents[2]
NAME='paper_results_v1_segmented_payload_v1'


def companion_files():
    paths=[ROOT/f'{name}/__init__.py' for name in ('main','runtime','experiments')]
    for folder in ('main/tube_state','runtime/wan','experiments/paper_results_v1'):
        paths.extend((ROOT/folder).rglob('*.py'))
    paths.append(ROOT/'experiments/paper_results_v1/independent_four_v1.adopted.json')
    return sorted(set(paths))


def build():
    output=ROOT/'notebooks';output.mkdir(exist_ok=True)
    archive_path=output/(NAME+'_companion.zip')
    with zipfile.ZipFile(archive_path,'w',zipfile.ZIP_DEFLATED) as archive:
        for path in companion_files():archive.write(path,path.relative_to(ROOT).as_posix())
    setup=textwrap.dedent('''\
        import json, os, signal, subprocess, sys, urllib.request, uuid, zipfile
        from pathlib import Path
        OUTPUT=Path('/content/drive/MyDrive/Video-WM/Paper-Results-V1-Segmented-Payload-V1/resumable-segmented-payload-v1')
        OUTPUT.mkdir(parents=True,exist_ok=True)
        ARCHIVE=OUTPUT/'companion_source.zip'
        URL='https://raw.githubusercontent.com/RICHAAARC/SC-SSTW/dev/paper-results-v1/notebooks/paper_results_v1_segmented_payload_v1_companion.zip'
        CODE=Path('/content')/('segmented-payload-'+uuid.uuid4().hex[:10]);CODE.mkdir()
        def readable_archive(path):
            with zipfile.ZipFile(path) as archive:
                archive.read('experiments/paper_results_v1/segmented_payload_v1_cli.py')
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
        CLI=[sys.executable,'-u','-m','experiments.paper_results_v1.segmented_payload_v1_cli']
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
        command([*CLI,'--output',str(OUTPUT),'--worker','init'],'Initialize fixed 36 rows')
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
        try:command(args,'Run fixed segmented payload task')
        finally:command([*CLI,'--output',str(OUTPUT),'--worker','report'],'Save complete fixed-denominator report',check=False)
        summary=json.loads((OUTPUT/'summary.json').read_text())
        print(json.dumps(summary,ensure_ascii=False,indent=2))
        print('Return this full directory:',OUTPUT)
        print('All 36 rows:',OUTPUT/'row_results.json')
    ''')
    intro='''# A：真实时间分段载荷的固定用途验证

在 GPU Colab 中 **Run all**。四个既有prompt/seed各新生成一条分段载荷候选。净32bit固定A6D39C5E，四个时间段各8bit；有效Wan层1..44每11层一段，每层在原4通道重复该段字节。边界0/45写S0/S3但不计净位。原mask44160、幅度0.5、eta5520、50步后25步局部FFT写入和clock margin0.5保持。完整每位1320票、有效总票42240不增加；这次确实改变生成目标。

固定接收条件：CROP_MID=[37:126]；DELETE_S1删除45..88。三盲臂CENTERED_U/RAW_U/UNALIGNED，共24盲行；另8攻击oracle、4clean oracle。始终保留36行/144段/1152计划位；盲行192位、攻击oracle64位在真值上结构缺失，不能当正确位、错误位或已恢复信息。合格名义四帧单元必须全部由估计路径中的真实收到索引支持；填帧仅是VAE上下文。错误盲定位可能误报已删段，单列错误归属。

先持久保存全部盲计划，再读各源clean正确对应；该源四字节完整且精确才执行其后8行。不可读源8行固定保留NOT_RUN_SOURCE_UNREADABLE。clean答案只决定是否执行，不能改路径/权重/组段。Wan名义时间分组不保证VAE感受野严格局部，跨段串扰是本次真实验证内容。

同Drive恢复：每个物理计算单元只尝试一次，调用前记录STARTED。失败、中断、完成未知和已开始但产物损坏均消耗该单元；不会自动重跑。复用可读完成产物，仅继续从未开始的单元。模型load总上界：generation4；native Wan阶段2、reader Wan阶段2；framewise2。后三者各含1次恢复预留，仅用于继续未开始单元，耗尽则保留BUDGET_EXHAUSTED。正常计算预算：4条生成/200native step/400DiT/100局部FFT、4native decode、8framewise encode/8decode各184批、4codec往返、16selectors、最多36Wan encode/36reader。相同源/接收视图/完整索引映射可共享encode并标记相关性，不按R相同共享。

质量参照为本轮新分段NATIVE_PRE；candidate_POST差包含clock和codec。全部失败、缺失、错误顺序与错误位保留。本交付只做工程静态/CPU fixture验证；GPU、媒体可读性、恢复跨Colab和科学结论待用户真实运行。无参数扫描、替换来源或科学PASS。普通可用源码ZIP直接加载，下载失败可使用同Drive可读缓存。
'''
    notebook=dict(nbformat=4,nbformat_minor=5,metadata={'accelerator':'GPU','kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python'}},cells=[
        cell('code',"from google.colab import drive\ndrive.mount('/content/drive')\n",'drive-mount'),
        cell('markdown',intro,'scope'),cell('code',setup,'source-and-fixed-plan'),
        cell('markdown','## 准备剩余实际依赖\n','dependencies-title'),cell('code',environment,'dependencies'),
        cell('markdown','## 直接执行并保存全部固定行\n','run-title'),cell('code',run,'run-and-report')])
    path=output/(NAME+'_colab.ipynb');path.write_text(json.dumps(notebook,ensure_ascii=False,indent=1)+'\n')
    return dict(notebook=str(path),companion=str(archive_path),files=len(companion_files()))


if __name__=='__main__':print(json.dumps(build(),indent=2))
