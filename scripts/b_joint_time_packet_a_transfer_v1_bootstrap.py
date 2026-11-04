"""Load the existing editable source path for a standard-library byte transfer."""
from pathlib import Path
import importlib.util
import tempfile
import urllib.request


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def launch(project, source_directory=None, source_ref='dev/local-joint-state-payload-v1'):
    with tempfile.TemporaryDirectory(prefix='joint-a-transfer-source-') as temporary:
        if source_directory is None:
            common_path = Path(temporary) / 'common.py'
            print(f'Loading transfer source from branch: {source_ref}', flush=True)
            urllib.request.urlretrieve(
                'https://raw.githubusercontent.com/RICHAAARC/SC-SSTW/' + source_ref
                + '/scripts/b_framewise_qim_v1_entry.py', common_path)
            common = load(common_path, 'joint_a_transfer_source_common')
            root = common.prepare_source(None, source_ref, Path(temporary) / 'checkout',
                required_files=('scripts/b_joint_time_packet_a_transfer_v1.py',))
        else:
            root = Path(source_directory).expanduser().resolve()
        transfer = load(root / 'scripts/b_joint_time_packet_a_transfer_v1.py', 'joint_a_transfer')
        return transfer.run(project)
