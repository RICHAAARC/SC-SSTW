"""Ordinary editable-source bootstrap; no science or integrity verification."""
from pathlib import Path
import importlib.util
import tempfile
import urllib.request


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def launch(source, output, source_directory=None, source_ref='dev/local-joint-state-payload-v1', model_id=None):
    # Notebook handles consumed output before loading/downloading this source.
    # Direct bootstrap callers also get local metadata-only recovery.
    if (Path(output)/'attempt.json').exists():
        import json
        saved, errors = {}, {}
        for name in ('attempt','state','counts','resources','fixed_slots','blind_summary','post_status','result'):
            try: saved[name] = json.loads((Path(output)/(name+'.json')).read_text())
            except (OSError, ValueError) as error: errors[name] = str(error)
        return dict(status='READ_ONLY_EXISTING_PACKAGE', saved=saved, read_errors=errors,
                    physical_calls_this_invocation=0, post_calls_this_invocation=0)
    if Path(output).exists() and (not Path(output).is_dir() or any(Path(output).iterdir())):
        return dict(status='OUTPUT_CONFLICT', physical_calls_this_invocation=0)
    with tempfile.TemporaryDirectory(prefix='native-y-source-entry-') as temporary:
        if source_directory is None:
            path = Path(temporary)/'common.py'
            urllib.request.urlretrieve('https://raw.githubusercontent.com/RICHAAARC/SC-SSTW/'+source_ref+
                '/scripts/b_framewise_qim_v1_entry.py', path)
            common = load(path, 'native_y_source_common')
            root = common.prepare_source(None, source_ref, Path(temporary)/'checkout', required_files=(
                'scripts/b_native_y_source_confirmation_v1_entry.py',
                'experiments/wan_state_clock/b_native_y_source_confirmation_v1_run.py'))
        else:
            root = Path(source_directory).expanduser().resolve()
        entry = load(root/'scripts/b_native_y_source_confirmation_v1_entry.py', 'native_y_source_entry')
        return entry.launch(source, output, root, source_ref, model_id)
