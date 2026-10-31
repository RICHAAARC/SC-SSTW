"""Thin ordinary-source notebook launcher; no physical work inside the notebook."""
from pathlib import Path
import importlib.util
import json
import tempfile

spec=importlib.util.spec_from_file_location('time_payload_common',Path(__file__).with_name('b_conditional_time_output_v1_entry.py'))
cpu=importlib.util.module_from_spec(spec);spec.loader.exec_module(cpu)
STEM='b_time_payload_cdc32_v1'
CACHE_FILES=cpu.CACHE_FILES+('scripts/'+STEM+'_entry.py','experiments/wan_state_clock/'+STEM+'_records.py',
                           'experiments/wan_state_clock/configs/'+STEM+'.json')


def saved_status(output):
    records=cpu._common().load_saved_module(Path(__file__).resolve().parents[1],STEM+'_records')
    return records.recover(output)


class Session(cpu.Session):
    stem=STEM
    cache_files=CACHE_FILES
    recover=staticmethod(saved_status)
    def prepare_dependencies(self,setup,root):setup.setup_dependencies(root,self.logs)


def launch(source,output,source_directory=None,source_ref=cpu.SOURCE_REF,model_id=None,
           scratch='/content/video-wm-time-payload-cdc32-v1'):
    existing=cpu._existing(output,saved_status)
    if existing is not None:return existing
    try:
        cfg=cpu._load(Path(__file__).resolve().parents[1]/'experiments/wan_state_clock/configs'/(STEM+'.json'))
        cfg.update(source=str(Path(source).expanduser().resolve()),output=str(Path(output).expanduser().resolve()))
        if model_id is not None:cfg['model_id']=str(model_id)
        with tempfile.TemporaryDirectory(prefix='time-payload-config-') as directory:
            path=Path(directory)/'config.json';path.write_text(json.dumps(cfg,ensure_ascii=False))
            return cpu.launch(source_directory,path,output,source_ref,scratch,session_type=Session)
    except (OSError,ValueError,TypeError) as error:
        return dict(status='PREPARATION_FAILED',error=str(error),physical_calls_this_invocation=0)
