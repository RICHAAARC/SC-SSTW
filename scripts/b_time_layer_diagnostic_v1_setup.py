"""Repair only actual numpy/psutil imports; never install/load model or codec."""
from pathlib import Path
import importlib.util
import sys

def setup_dependencies(root,logs):
    root=Path(root);logs=Path(logs)
    spec=importlib.util.spec_from_file_location('layer_setup_common',root/'scripts/b_framewise_qim_v1_setup.py')
    common=importlib.util.module_from_spec(spec);spec.loader.exec_module(common)
    common.ensure_imports('import numpy,psutil',['numpy','psutil'],'CPU diagnostic imports',root,logs/'setup.log')
    common.command([sys.executable,'-B','-c','import runtime.wan.b_time_layer_diagnostic_v1; import experiments.wan_state_clock.b_time_layer_diagnostic_v1_run'],
                   'Probe actual CPU diagnostic imports',root,logs/'setup.log')
