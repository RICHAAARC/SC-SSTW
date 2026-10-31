"""CPU-only dependency preparation; reuse actual-import repair without version gates."""
from pathlib import Path
import importlib.util
import os
import shutil
import sys


def setup_dependencies(source_root, log_directory, entry):
    if entry not in ('semantic', 'raw', 'media'):
        raise ValueError('one configured input entry required')
    root = Path(source_root)
    spec = importlib.util.spec_from_file_location('conditional_setup_common', root/'scripts/b_framewise_qim_v1_setup.py')
    common = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(common)
    log = Path(log_directory)/'setup.log'
    previous = os.environ.get('PYTHONDONTWRITEBYTECODE')
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    try:
        common.ensure_imports('import numpy,psutil', ['numpy', 'psutil'], 'CPU receiver dependencies', root, log)
        if entry == 'media':
            if not shutil.which('ffmpeg') or not shutil.which('ffprobe'):
                common.command(['apt-get', '-qq', 'update'], 'Prepare media dependency', root, log)
                common.command(['apt-get', '-qq', 'install', '-y', 'ffmpeg'], 'Install FFmpeg', root, log)
            if not shutil.which('ffmpeg') or not shutil.which('ffprobe'):
                raise RuntimeError('ffmpeg/ffprobe unavailable; see '+str(log))
        probe = ('import main.tube_state.b_conditional_time_output_v1; '
                 'import runtime.wan.b_conditional_time_output_v1; '
                 'import experiments.wan_state_clock.b_conditional_time_output_v1_run')
        common.command([sys.executable, '-B', '-c', probe], 'Probe actual conditional receiver imports', root, log)
    finally:
        if previous is None:
            os.environ.pop('PYTHONDONTWRITEBYTECODE', None)
        else:
            os.environ['PYTHONDONTWRITEBYTECODE'] = previous
