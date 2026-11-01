"""Ordinary-source CPU diagnostic launcher; consumed outputs recover before setup."""
from pathlib import Path
import importlib.util
import json
import tempfile
spec=importlib.util.spec_from_file_location('layer_cpu_entry',Path(__file__).with_name('b_conditional_time_output_v1_entry.py'))
cpu=importlib.util.module_from_spec(spec);spec.loader.exec_module(cpu)
STEM='b_time_layer_diagnostic_v1'
CACHE_FILES=cpu.CACHE_FILES+('scripts/'+STEM+'_entry.py','experiments/wan_state_clock/'+STEM+'_records.py','experiments/wan_state_clock/configs/'+STEM+'.json')
def saved_status(output):
    records=cpu._common().load_saved_module(Path(__file__).resolve().parents[1],STEM+'_records')
    return records.recover(output)
def summarize(result,output):
    records=cpu._common().load_saved_module(Path(__file__).resolve().parents[1],STEM+'_records')
    return records.summary(result,output)
class Session(cpu.Session):
    stem=STEM
    cache_files=CACHE_FILES
    recover=staticmethod(saved_status)
    def prepare_dependencies(self,setup,root):setup.setup_dependencies(root,self.logs)
def launch(source_attempt,output,source_directory=None,source_ref=cpu.SOURCE_REF,scratch='/content/video-wm-time-layer-diagnostic-v1'):
    existing=cpu._existing(output,saved_status)
    if existing is not None:return existing
    cfg=dict(source_attempt=str(Path(source_attempt).expanduser().resolve()),output=str(Path(output).expanduser().resolve()))
    with tempfile.TemporaryDirectory(prefix='time-layer-config-') as temporary:
        path=Path(temporary)/'config.json';path.write_text(json.dumps(cfg))
        return cpu.launch(source_directory,path,output,source_ref,scratch,session_type=Session)
