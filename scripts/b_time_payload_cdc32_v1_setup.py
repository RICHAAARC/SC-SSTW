"""Reuse ordinary Wan import repair and media tools, without loading the model."""
from pathlib import Path
import importlib.util
import sys


def setup_dependencies(root,logs):
    root=Path(root);logs=Path(logs)
    spec=importlib.util.spec_from_file_location('time_payload_dependencies',root/'scripts/b_framewise_qim_v1_setup.py')
    value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value)
    value.ensure_imports('import psutil',['psutil'],'RSS monitor dependency',root,logs/'setup.log')
    value.setup_dependencies(root,logs)
    value.command([sys.executable,'-B','-c','import runtime.wan.b_time_payload_cdc32_v1; import experiments.wan_state_clock.b_time_payload_cdc32_v1_run'],
                  'Probe time to payload receiver imports',root,logs/'setup.log')
