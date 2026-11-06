"""Thin stdlib bootstrap: consumed outputs need neither source nor scientific imports."""
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time
import types
import urllib.request

REPOSITORY = 'RICHAAARC/SC-SSTW'
SOURCE_REF = 'dev/local-joint-state-payload-v1'
OUTPUT = Path('/content/drive/MyDrive/Video-WM/B-Joint-Time-Payload-R02-V1/fresh-r02-v1/attempt01')
WORK = Path('/content/video-wm-joint-time-payload-r02')


def source_text(path, source_directory, source_ref):
    if source_directory: return (Path(source_directory)/path).read_text(encoding='utf-8')
    return urllib.request.urlopen(f'https://raw.githubusercontent.com/{REPOSITORY}/{source_ref}/{path}',
                                  timeout=120).read().decode('utf-8')


def standalone(path, source_directory, source_ref):
    module = types.ModuleType(Path(path).stem)
    exec(compile(source_text(path, source_directory, source_ref), path, 'exec'), module.__dict__)
    return module


def show_results(output=OUTPUT):
    output = Path(output); print('保留结果目录:', output)
    for name in ('result.json','posts.json','preparation_failure.json','chain_resources.json',
                 'time_chain/evaluation.json','payload/evaluation_summary.json',
                 'payload/resources.json','payload/quality.json','payload/raw_terminal_summary.json'):
        if (output/name).is_file():
            try: value = json.loads((output/name).read_text())
            except (OSError, ValueError) as error: value = dict(read_error=str(error))
            print(name, json.dumps(value, ensure_ascii=False, indent=2))
    print('逐槽证据: time_terminal_slots.json、time_chain/、payload/；两阶段固定分母均保留。')


def install_dependencies(source, output):
    import importlib.util
    path = source/'scripts/b_framewise_qim_v1_setup.py'
    spec = importlib.util.spec_from_file_location('joint_r02_dependency_setup', path)
    setup = importlib.util.module_from_spec(spec); spec.loader.exec_module(setup)
    logs = output/'preparation'; logs.mkdir(parents=True, exist_ok=True)
    setup.ensure_imports('import numpy,torch,psutil,accelerate,safetensors,huggingface_hub; '
        'from diffusers import WanPipeline,AutoencoderKLWan; '
        'import runtime.wan.b_joint_time_relative_physical_v1,runtime.wan.b_relative_time_payload_backend_v1',
        ['numpy','torch','psutil','accelerate','safetensors','huggingface-hub','diffusers','transformers','sentencepiece','ftfy'],
        'r02 actual generation/time/payload imports', source, logs/'setup.log')
    if not shutil.which('ffmpeg') or not shutil.which('ffprobe'):
        setup.command(['apt-get','-qq','update'], 'Prepare media dependency', source, logs/'setup.log')
        setup.command(['apt-get','-qq','install','-y','ffmpeg'], 'Install media dependency', source, logs/'setup.log')


def run_colab(source_directory=None, source_ref=SOURCE_REF):
    if OUTPUT.exists() and any(OUTPUT.iterdir()):
        show_results(OUTPUT); return dict(status='READ_ONLY', output=str(OUTPUT))
    # Small stdlib reservation modules only; all 629 operations/slots precede preparation and science.
    tp = standalone('main/tube_state/b_joint_time_relative_validation_v1.py', source_directory, source_ref)
    payload = standalone('runtime/wan/b_relative_time_payload_records_v1.py', source_directory, source_ref)
    records = standalone('runtime/wan/b_joint_time_payload_r02_records_v1.py', source_directory, source_ref)
    config = json.loads(source_text('experiments/wan_state_clock/configs/b_joint_time_payload_r02_v1.json',
                                    source_directory, source_ref))
    config['output'] = str(OUTPUT)
    if not records.reserve(OUTPUT, config, time_protocol=tp, payload_records=payload):
        show_results(OUTPUT); return dict(status='READ_ONLY', output=str(OUTPUT))
    start = time.monotonic()
    try:
        WORK.mkdir(parents=True, exist_ok=True)
        if source_directory: source = Path(source_directory).resolve()
        else:
            from urllib.parse import quote
            helper = standalone('runtime/wan/b_relative_time_payload_colab_v1.py', source_directory, source_ref)
            stage = Path(tempfile.mkdtemp(prefix='available-source-', dir=WORK))
            archive = stage/'source.zip'
            urllib.request.urlretrieve(f'https://codeload.github.com/{REPOSITORY}/zip/refs/heads/{quote(source_ref,safe="/")}', archive)
            helper.safe_extract(archive, stage/'unpacked')
            choices = [p for p in (stage/'unpacked').iterdir()
                       if (p/'runtime/wan/b_joint_time_payload_r02_v1.py').is_file()]
            if len(choices) != 1: raise ValueError('one usable source directory required')
            source = choices[0]
        sys.path.insert(0, str(source))
        os.environ.update(HF_HOME=str(WORK/'cache/huggingface'), TORCH_HOME=str(WORK/'cache/torch'))
        install_dependencies(source, OUTPUT)
        records.write(OUTPUT/'preparation.json', dict(status='RETURNED', seconds=time.monotonic()-start,
                      source_directory=str(source), source_ref=source_ref, informational_only=True))
        from runtime.wan.b_joint_time_payload_r02_v1 import execute_reserved
        result = execute_reserved(OUTPUT, WORK/'work')
    except BaseException as error:
        failure = f'{type(error).__name__}: {error}'
        records.write(OUTPUT/'preparation_failure.json', dict(failure=failure, seconds=time.monotonic()-start))
        records.close_blind(OUTPUT, failure, payload_records=payload)
        if not (OUTPUT/'posts.json').is_file():
            records.write(OUTPUT/'posts.json', {name: dict(status='NOT_STARTED', started=0, returned=0,
                dependency_status='DEPENDENCY_UNRESOLVED', reason=failure) for name in ('time','payload')})
        result = records.finish(OUTPUT, failure)
    show_results(OUTPUT)
    return dict(result, output=str(OUTPUT))
