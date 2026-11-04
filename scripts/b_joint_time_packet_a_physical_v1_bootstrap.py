"""Editable source download following the existing successful thin entry path."""
from pathlib import Path
import importlib.util
import tempfile
import urllib.request


def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path);value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value);return value


def launch(cpu_handoff,input_packages,output,source_directory=None,source_ref='dev/local-joint-state-payload-v1',model_id=None):
    pending=any(not (Path(output)/stage/'attempt.json').exists() for stage in ('gpu1','cpu_media','gpu2','post'))
    if not pending:
        import json
        stages={}
        for stage in ('gpu1','cpu_media','gpu2','post'):
            folder=Path(output)/stage
            item=dict(status='READ_ONLY_CONSUMED_STAGE' if (folder/'attempt.json').exists() else 'WAITING_INPUT',
                      physical_calls_this_invocation=0,saved={},read_errors={})
            if (folder/'attempt.json').exists():
                for name in ('attempt','state','counts','resources','blind_terminal','result'):
                    try:item['saved'][name]=json.loads((folder/(name+'.json')).read_text())
                    except (OSError,ValueError) as error:item['read_errors'][name]=str(error)
            stages[stage]=item
        return dict(status='RUN_ALL_STATUS',stages=stages)
    with tempfile.TemporaryDirectory(prefix='joint-a-source-') as temporary:
        if source_directory is None:
            path=Path(temporary)/'common.py'
            urllib.request.urlretrieve('https://raw.githubusercontent.com/RICHAAARC/SC-SSTW/'+source_ref+'/scripts/b_framewise_qim_v1_entry.py',path)
            common=load(path,'joint_a_source_common')
            root=common.prepare_source(None,source_ref,Path(temporary)/'checkout',required_files=(
                'scripts/b_joint_time_packet_a_physical_v1_entry.py',
                'scripts/b_joint_time_packet_a_physical_v1_report.py',
                'experiments/wan_state_clock/b_joint_time_packet_a_physical_v1_run.py'))
        else:root=Path(source_directory).expanduser().resolve()
        entry=load(root/'scripts/b_joint_time_packet_a_physical_v1_entry.py','joint_a_entry')
        return entry.launch(cpu_handoff,input_packages,output,root,model_id)
