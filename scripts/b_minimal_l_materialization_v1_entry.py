"""Optional stdlib entry for two independent e02 stages; CPU receiver stays local."""
from pathlib import Path
import importlib.util
import json
import sys

_spec=importlib.util.spec_from_file_location('minimal_l_common',Path(__file__).with_name('b_framewise_qim_v1_entry.py'))
common=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(common)
STEM='b_minimal_l_materialization_v1'
CACHE_FILES=('scripts/'+STEM+'_entry.py','scripts/b_framewise_qim_v1_entry.py',
             'experiments/wan_state_clock/'+STEM+'_records.py',
             'experiments/wan_state_clock/b_framewise_qim_v1_records.py',
             'experiments/wan_state_clock/b_framewise_qim_saved_candidate_v1_records.py')


def pending(output):
    output=Path(output)
    if common.existing_output(output/'B'): return False
    if not common.existing_output(output/'A'): return True
    try:
        result=json.loads((output/'A/result.json').read_text())
        complete=isinstance(result,dict) and result.get('stage_complete') is True and result.get('finite_candidate_saved') is True
        names=['coefficients.pt','fixed_targets.npy','BASE_preclip.npy']+[f'features/{s:03}.pt' for s in (0,*range(1,181,4))]
        return complete and all((output/'A'/name).is_file() for name in names)
    except (OSError,ValueError): return False


class Session(common.Session):
    def __init__(self,source,output,model_id=None,source_directory=None,source_ref='dev/local-joint-state-payload-v1'):
        super().__init__(source,output,source_ref,source_directory,scratch='/content/video-wm-minimal-l-v1')
        self.model_id=model_id

    def execute(self):
        if not pending(self.output): return
        try:
            source=self.source.expanduser().resolve();output=self.output.expanduser().resolve()
            writable=[output,self.cache,self.logs,output.with_name(output.name+'-preflight-v1'),
                      Path(self.source_directory) if self.source_directory is not None else self.scratch]
            for path in writable:
                path=path.expanduser().resolve()
                if path==source or path.is_relative_to(source) or source.is_relative_to(path):
                    raise ValueError('write location overlaps read-only source')
            root=common.prepare_source(self.source_directory,self.source_ref,self.scratch,
                required_files=('experiments/wan_state_clock/'+STEM+'_run.py',
                                'runtime/wan/'+STEM+'.py','scripts/b_framewise_qim_v1_setup.py'))
            records=common.load_saved_module(root,STEM+'_records')
            if not records.needs_work(output): return
            stage='A' if records.output_state(output/'A')=='NEW' else 'B'
            check=records.preflight(source,output,stage)
            print('阶段预检:',stage,check)
            if check['status']!='PREFLIGHT_READY': return
            try: common.cache_entry(root,self.cache,CACHE_FILES)
            except OSError as error: print('可选入口缓存未写入:',str(error))
            self.logs.mkdir(parents=True,exist_ok=True)
            setup=common.load_file(root/'scripts/b_framewise_qim_v1_setup.py','minimal_l_setup')
            setup.setup_dependencies(root,self.logs)
            if not records.needs_work(output): return
        except Exception as error:
            self.preparation_error=f'{type(error).__name__}: {error}';print(self.preparation_error);return
        self._worker(root)

    def _command(self,root):
        args=[sys.executable,'-B','-u','-m','experiments.wan_state_clock.'+STEM+'_run',
              '--source',str(self.source),'--output',str(self.output)]
        if self.model_id is not None: args+=['--model-id',str(self.model_id)]
        return args

    def display(self):
        for stage in ('A','B'):
            print('阶段',stage)
            for name in ('result','counts','resources'):
                try:
                    value=json.loads((self.output/stage/(name+'.json')).read_text())
                    if not isinstance(value,dict): raise ValueError('saved object required')
                    print(name,value)
                except (OSError,ValueError) as error: print(name,'未知:',str(error))
        print('本地CPU确认未由该notebook执行；媒体路径:',self.output/'A/BASE.mp4',self.output/'B/QIM.mp4')
