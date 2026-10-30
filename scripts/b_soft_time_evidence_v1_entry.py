"""Optional ordinary notebook entry. Only numpy/psutil setup; no media/model."""
from pathlib import Path
import importlib.util
import subprocess
import sys
import json

spec=importlib.util.spec_from_file_location('soft_common_entry',Path(__file__).with_name('b_framewise_qim_v1_entry.py'))
common=importlib.util.module_from_spec(spec);spec.loader.exec_module(common)
STEM='b_soft_time_evidence_v1'
CACHE_FILES=('scripts/'+STEM+'_entry.py','scripts/b_framewise_qim_v1_entry.py',
             'experiments/wan_state_clock/'+STEM+'_records.py','experiments/wan_state_clock/'+STEM+'_report.py',
             'experiments/wan_state_clock/b_framewise_qim_time_recovery_v1_records.py')
REQUIRED=tuple('experiments/wan_state_clock/'+STEM+'_'+s+'.py' for s in ('run','experiment','records','evaluation'))+(
    'main/tube_state/'+STEM+'.py','runtime/wan/'+STEM+'.py',
    'main/tube_state/b_framewise_qim_time_recovery_v1.py','runtime/wan/b_framewise_qim_time_recovery_v1.py',
    'experiments/wan_state_clock/b_framewise_qim_time_recovery_v1_records.py')


def setup_dependencies(logs):
    missing=[name for name in ('numpy','psutil') if importlib.util.find_spec(name) is None]
    if missing:
        with (logs/'setup.log').open('a') as stream:subprocess.run([sys.executable,'-m','pip','install',*missing],stdout=stream,stderr=subprocess.STDOUT,check=True)


class Session(common.Session):
    def __init__(self,source_attempt,output,source_directory=None,source_ref='dev/local-joint-state-payload-v1'):
        super().__init__(source_attempt,output,source_ref,source_directory,scratch='/content/video-wm-soft-time-evidence-v1')
    def _saved_report(self):
        try:self.report=common.load_report(self.cache,STEM+'_records',STEM+'_report')
        except Exception as error:print('详细报告不可用:',str(error))
    def execute(self):
        self.source=self.source.expanduser().resolve();self.output=self.output.expanduser().resolve()
        self.cache=self.output.with_name(self.output.name+'-entry-cache');self.logs=self.output.with_name(self.output.name+'-logs')
        if (self.output/'attempt.json').exists():self._saved_report();return
        if common.existing_output(self.output):self.preparation_error='OUTPUT_CONFLICT';return
        try:
            writes=[self.output,self.cache,self.logs,self.output.with_name(self.output.name+'-preflight-v1'),Path(self.source_directory) if self.source_directory is not None else self.scratch]
            for path in writes:
                p=path.expanduser().resolve()
                if p==self.source or p.is_relative_to(self.source) or self.source.is_relative_to(p):raise ValueError('write path overlaps old input attempt')
            root=common.prepare_source(self.source_directory,self.source_ref,self.scratch,REQUIRED)
            records=common.load_saved_module(root,STEM+'_records');preflight=records.preflight(self.source,self.output)
            print('预检:',preflight['status'],preflight['errors'],preflight['inputs'])
            if preflight['status']!='PREFLIGHT_READY':return
            try:common.cache_entry(root,self.cache,CACHE_FILES)
            except OSError as error:print('可选缓存未写入:',str(error))
            try:self.report=common.load_report(root,STEM+'_records',STEM+'_report')
            except Exception as error:print('可选报告不可用:',str(error))
            if records.output_state(self.output)!='NEW':return
            self.logs.mkdir(parents=True,exist_ok=True);setup_dependencies(self.logs)
            if records.output_state(self.output)!='NEW':return
        except Exception as error:self.preparation_error=f'{type(error).__name__}: {error}';print(self.preparation_error);return
        self._worker(root)
    def _command(self,root):
        return [sys.executable,'-B','-u','-m','experiments.wan_state_clock.'+STEM+'_run','--source-attempt',str(self.source),'--output',str(self.output)]
    def display(self):
        if self.report is None:self._saved_report()
        if self.report is not None:
            try:return self.report.display(self.output)
            except Exception as error:print('报告未知:',str(error))
        for name in ('attempt','result','resources'):
            try:
                value=json.loads((self.output/(name+'.json')).read_text())
                if not isinstance(value,dict):raise ValueError('JSON object required')
                print(name,{k:v for k,v in value.items() if k!='plan'})
            except (OSError,ValueError) as error:print(name,'未知:',str(error))
