"""Detachable C4 main entry checks; synthetic CPU values, no models or codecs."""
import ast
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import patch
import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_runtime_worker_without_outer_layers(tmp_path):
    source = tmp_path / 'source'
    for name in ('main', 'runtime'):
        shutil.copytree(ROOT / name, source / name, ignore=shutil.ignore_patterns('__pycache__'))
    env = dict(os.environ, PYTHONPATH=str(source))
    help_result = subprocess.run([sys.executable, '-m', 'runtime.wan.b_paper_cohort_worker_v1', '--help'],
        cwd=tmp_path, env=env, capture_output=True, text=True, timeout=30)
    assert help_result.returncode == 0, help_result.stderr
    assert '--worker-stage' in help_result.stdout
    imports = subprocess.run([sys.executable, '-c',
        'from runtime.wan import b_paper_cohort_runner_v1, b_paper_cohort_backend_v1, b_colab_support_v1; '
        'assert b_paper_cohort_runner_v1.CLI == "runtime.wan.b_paper_cohort_worker_v1"'],
        cwd=tmp_path, env=env, capture_output=True, text=True, timeout=30)
    assert imports.returncode == 0, imports.stderr
    # The pure method package remains importable after removing even runtime.
    shutil.rmtree(source / 'runtime')
    core = subprocess.run([sys.executable, '-c',
        'from main.tube_state import b_paper_cohort_v1 as p; '
        'assert len(p.operation_plan()) == 1114; '
        'from main.tube_state import b_local_identity_dct_v1, b_cdc32_aggregation_v1, '
        'b_joint_time_packet_a_relative_v1, qmatch_v1'],
        cwd=tmp_path, env=env, capture_output=True, text=True, timeout=30)
    assert core.returncode == 0, core.stderr


def test_layer_imports_and_real_stage_command(tmp_path):
    for folder, forbidden in [('main/tube_state', {'runtime','experiments','governance','scripts'}),
                              ('runtime/wan', {'experiments','governance','scripts'})]:
        for path in (ROOT / folder).rglob('*.py'):
            for node in ast.walk(ast.parse(path.read_text())):
                names = [a.name for a in node.names] if isinstance(node, ast.Import) else (
                    [node.module or ''] if isinstance(node, ast.ImportFrom) and not node.level else [])
                assert not any(name.split('.')[0] in forbidden for name in names), str(path)
    from runtime.wan import b_paper_cohort_runner_v1 as runner, b_worker_process_v1 as process
    output = tmp_path / 'out'; result = output / 'stages/prepare/result.json'
    result.parent.mkdir(parents=True); result.write_text('{"status":"COMPLETED"}')
    with patch.object(process, 'stage_process', return_value={'returncode': 0}) as call:
        assert runner.subprocess_stage('prepare', output, tmp_path / 'scratch')['status'] == 'COMPLETED'
    command = call.call_args.args[0]
    assert command[command.index('-m') + 1] == 'runtime.wan.b_paper_cohort_worker_v1'
    assert command[command.index('--worker-stage') + 1] == 'prepare'


def test_legacy_vae_default_and_observer_are_same_output():
    import torch
    from runtime.wan.vae import decode_normalized_latent
    class FakeVAE:
        config = SimpleNamespace(latents_mean=[0], latents_std=[1])
        def decode(self, latent, return_dict=False):
            return (latent.repeat(1,3,1,1,1),)
    z = torch.tensor([[[[[-2., 0., 2.]]]]])
    base = decode_normalized_latent(FakeVAE(), z)
    seen = []
    def observe(value):
        seen.append(value.clone()); value.zero_()
    actual = decode_normalized_latent(FakeVAE(), z, diagnostic_observer=observe)
    assert torch.equal(base, actual)
    assert seen[0].min() < 0 and seen[0].max() > 1
    assert base.min() == 0 and base.max() == 1


def test_prequantized_encoder_passes_bytes_without_codec(tmp_path):
    import torch
    from runtime.wan import io
    pixels = torch.arange(24, dtype=torch.uint8).reshape(2,2,2,3)
    with patch.object(io.subprocess, 'run') as call:
        io.encode_rgb8(pixels, tmp_path / 'fake.mp4', 8, 18)
        assert call.call_args.kwargs['input'] == pixels.numpy().tobytes()
        with pytest.raises(ValueError, match='uint8'):
            io.encode_rgb8(pixels.float(), tmp_path / 'invalid.mp4', 8, 18)
        assert call.call_count == 1


def test_standalone_archive_helper_keeps_safe_extraction(tmp_path):
    import zipfile
    from runtime.wan.b_colab_support_v1 import safe_extract
    safe = tmp_path / 'safe.zip'
    with zipfile.ZipFile(safe, 'w') as archive: archive.writestr('source/a.txt', 'ok')
    safe_extract(safe, tmp_path / 'ok')
    assert (tmp_path / 'ok/source/a.txt').read_text() == 'ok'
    bad = tmp_path / 'bad.zip'
    with zipfile.ZipFile(bad, 'w') as archive: archive.writestr('../escape', 'bad')
    with pytest.raises(ValueError, match='unsafe'): safe_extract(bad, tmp_path / 'bad')
    assert not (tmp_path / 'escape').exists()
