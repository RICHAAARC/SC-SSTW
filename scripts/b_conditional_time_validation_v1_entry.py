"""Full user-run notebook entry, sharing ordinary source/setup/worker preparation."""
from pathlib import Path
import importlib.util
import json
import tempfile

spec=importlib.util.spec_from_file_location('conditional_cpu_entry',Path(__file__).with_name('b_conditional_time_output_v1_entry.py'))
cpu=importlib.util.module_from_spec(spec);spec.loader.exec_module(cpu)
STEM='b_conditional_time_validation_v1'
CACHE_FILES=cpu.CACHE_FILES+('scripts/'+STEM+'_entry.py',
    'experiments/wan_state_clock/'+STEM+'_records.py',
    'experiments/wan_state_clock/b_minimal_l_materialization_v1_records.py',
    'experiments/wan_state_clock/b_framewise_qim_v1_records.py',
    'experiments/wan_state_clock/b_framewise_qim_saved_candidate_v1_records.py',
    'experiments/wan_state_clock/configs/'+STEM+'.json')


def saved_status(output):
    try:
        common=cpu._common()
        records=common.load_saved_module(Path(__file__).resolve().parents[1],STEM+'_records')
        return records.recover(output)
    except (OSError,ValueError,ImportError) as error:
        result=cpu.saved_status(output)
        result.update(status='READ_ONLY_EXISTING_PACKAGE',report_error=str(error))
        return result


class Session(cpu.Session):
    stem=STEM
    cache_files=CACHE_FILES
    recover=staticmethod(saved_status)

    def required_files(self):
        return ('experiments/wan_state_clock/'+STEM+'_run.py','scripts/'+STEM+'_setup.py',
                'runtime/wan/b_minimal_l_materialization_v1.py')

    def prepare_dependencies(self,setup,root):
        setup.setup_dependencies(root,self.logs)


def launch(source,output,source_directory=None,source_ref=cpu.SOURCE_REF,model_id=None,
           scratch='/content/video-wm-conditional-time-validation-v1',config_path=None):
    existing=cpu._existing(output,saved_status)
    if existing is not None:return existing
    try:
        path=Path(config_path) if config_path is not None else Path(__file__).resolve().parents[1]/'experiments/wan_state_clock/configs'/(STEM+'.json')
        config=cpu._load(path)
        config.update(source=str(Path(source).expanduser().resolve()),output=str(Path(output).expanduser().resolve()))
        if model_id is not None:config['model_id']=str(model_id)
        # Ordinary temporary launch configuration, not an identity or expected digest chain.
        with tempfile.TemporaryDirectory(prefix='conditional-validation-config-') as temporary:
            launch_path=Path(temporary)/'input.json';launch_path.write_text(json.dumps(config,ensure_ascii=False))
            return cpu.launch(source_directory,launch_path,output,source_ref,scratch,session_type=Session)
    except (OSError,ValueError,TypeError) as error:
        return dict(status='PREPARATION_FAILED',error=f'{type(error).__name__}: {error}',physical_calls_this_invocation=0)
