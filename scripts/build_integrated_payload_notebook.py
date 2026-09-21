"""Build the fixed Run-all Colab notebook bound to the integrated source SHA."""
from __future__ import annotations

import json
from pathlib import Path

SOURCE_SHA = "8da1687c05a86bb320bcee67bb084bcf020068a7"
REPOSITORY = "https://github.com/RICHAAARC/SC-SSTW.git"
OUTPUT = Path(__file__).parents[1] / "notebooks" / "integrated_payload_v1_colab.ipynb"


def build():
    metadata = {
        "accelerator": "GPU",
        "colab": {"name": "SC-SSTW Integrated Payload V1", "provenance": []},
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
        "source_commit": SOURCE_SHA,
        "notebook_binding_kind": "immutable_source_commit_pending_publication",
    }
    def code(source):
        return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": source.splitlines(keepends=True)}

    def markdown(source):
        return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(keepends=True)}

    cells = [
        code("from google.colab import drive\ndrive.mount('/content/drive')"),
        markdown(
            "# SC-SSTW Core Integration V1\n\n"
            "Run all cells once. This fixed candidate creates four fresh Wan prompt/seed cases, "
            "freezes calibration from two independent OFF sources, then evaluates OFF, SINGLE46, "
            "and MULTI44_46 on two new sources carrying payloads `0x5` and `0xa`. It writes 56 "
            "saved views and performs 224 four-phase receiver encodes with the bound Partial3-V2 receiver. "
            "No mode or strength scan is exposed."
        ),
        code(
            "import subprocess, sys\n"
            "subprocess.run(['apt-get', 'update', '-qq'], check=True)\n"
            "subprocess.run(['apt-get', 'install', '-y', '-qq', 'ffmpeg'], check=True)\n"
            "subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', "
            "'diffusers==0.40.0', 'transformers>=4.49,<5', 'accelerate', 'ftfy', "
            "'sentencepiece', 'safetensors', 'huggingface_hub', 'Pillow'], check=True)"
        ),
        code(
            "import shutil, subprocess\n"
            "from pathlib import Path\n\n"
            f"SOURCE_SHA = '{SOURCE_SHA}'\n"
            f"REPOSITORY = '{REPOSITORY}'\n"
            "REPO = Path('/content/SC-SSTW-Core-Integration')\n"
            "if REPO.exists():\n"
            "    shutil.rmtree(REPO)\n"
            "subprocess.run(['git', 'clone', '--filter=blob:none', REPOSITORY, str(REPO)], check=True)\n"
            "subprocess.run(['git', '-C', str(REPO), 'checkout', '--detach', SOURCE_SHA], check=True)\n"
            "actual = subprocess.check_output(['git', '-C', str(REPO), 'rev-parse', 'HEAD'], text=True).strip()\n"
            "assert actual == SOURCE_SHA\n"
            "print('source commit:', actual)"
        ),
        code(
            "import torch, diffusers\n"
            "print('torch:', torch.__version__, 'cuda:', torch.version.cuda)\n"
            "print('diffusers:', diffusers.__version__)\n"
            "if not torch.cuda.is_available():\n"
            "    raise RuntimeError('This fixed real Wan run requires a CUDA runtime')\n"
            "print('device:', torch.cuda.get_device_name(0))"
        ),
        code(
            "import datetime, os, subprocess, sys\n"
            "from pathlib import Path\n\n"
            "DRIVE_ROOT = Path('/content/drive/MyDrive/Video-WM/SC-SSTW-Core-Integration')\n"
            "DRIVE_ROOT.mkdir(parents=True, exist_ok=True)\n"
            "stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')\n"
            "OUTPUT = DRIVE_ROOT / ('integrated_payload_v1_' + stamp)\n"
            "CONFIG = REPO / 'experiments/wan_state_clock/configs/integrated_payload_v1.json'\n"
            "env = os.environ.copy()\n"
            "env['PYTHONUNBUFFERED'] = '1'\n"
            "command = [sys.executable, '-u', '-m', 'experiments.wan_state_clock.integrated_payload_run', "
            "'--config', str(CONFIG), '--output', str(OUTPUT)]\n"
            "print('output:', OUTPUT)\n"
            "subprocess.run(command, cwd=REPO, env=env, check=True)"
        ),
        code(
            "import json\n"
            "result = json.loads((OUTPUT / 'result.json').read_text())\n"
            "print('status:', result['status'])\n"
            "print('fixed denominator:', result['fixed_denominator'])\n"
            "print('calibration:', result['calibration'])\n"
            "for case_id, case in result['cases'].items():\n"
            "    print('\\n', case_id, case['status'])\n"
            "    for arm, item in case['videos'].items():\n"
            "        print(arm, item.get('decision'), item.get('reporting_only'))\n"
            "print('full result:', OUTPUT / 'result.json')"
        ),
    ]
    notebook = {"cells": cells, "metadata": metadata, "nbformat": 4, "nbformat_minor": 5}
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return OUTPUT


if __name__ == "__main__":
    print(build())
