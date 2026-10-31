"""Reuse exercised Wan materialization setup, then actual CPU receiver imports."""
from pathlib import Path
import importlib.util
import os


def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value);return value


def setup_dependencies(source_root,log_directory):
    root=Path(source_root)
    previous=os.environ.get('PYTHONDONTWRITEBYTECODE')
    os.environ['PYTHONDONTWRITEBYTECODE']='1'
    try:
        material=load(root/'scripts/b_framewise_qim_v1_setup.py','conditional_material_setup')
        material.setup_dependencies(root,log_directory)
        cpu=load(root/'scripts/b_conditional_time_output_v1_setup.py','conditional_receiver_setup')
        cpu.setup_dependencies(root,log_directory,'media')
        probe=('import experiments.wan_state_clock.b_conditional_time_validation_v1_run; '
               'import runtime.wan.b_minimal_l_materialization_v1')
        import sys
        material.command([sys.executable,'-B','-c',probe],'Probe full materialization/validation imports',root,Path(log_directory)/'setup.log')
    finally:
        if previous is None:os.environ.pop('PYTHONDONTWRITEBYTECODE',None)
        else:os.environ['PYTHONDONTWRITEBYTECODE']=previous
