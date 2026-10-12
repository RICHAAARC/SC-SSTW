"""Build the fixed eight-source confirmation Run-all notebook.

The accepted two-pilot setup cells remain the source for environment and model
preparation.  This builder changes the execution scope and adds the adopted
attack/receiver stages; it never imports a model package.
"""
from __future__ import annotations

import json
import re
import textwrap
from pathlib import Path

from experiments.paper_results_v1.companion_source import build_companion_zip


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "notebooks/paper_results_v1_confirmation_v1_colab.ipynb"
BASE = ROOT / "notebooks/paper_results_v1_two_pilot_colab.ipynb"


def _cell(kind, source, ident):
    row = {"cell_type": kind, "metadata": {}, "source": source.splitlines(keepends=True), "id": ident}
    if kind == "code":
        row.update(execution_count=None, outputs=[])
    return row


def _base_cell(notebook, ident):
    return "".join(next(row["source"] for row in notebook["cells"] if row.get("id") == ident))


def build_notebook():
    build_companion_zip()
    base = json.loads(BASE.read_text(encoding="utf-8"))
    setup = _base_cell(base, "portable-setup")
    setup = setup.replace(
        'PORTABLE_ROOT = Path("/content/paper-results-v1-source")',
        'PORTABLE_ROOT = Path("/content/paper-results-v1-checkpoint-source")',
    )
    setup = setup.replace('PILOT_CASES = ("pilot_01", "pilot_02")', 'PILOT_CASES = ()')
    setup = setup.replace(
        'OUTPUT_ROOT = Path("/content/drive/MyDrive/Video-WM/Paper-Results-V1-Two-Pilot") / RUN_ID\nOUTPUT_ROOT.mkdir(parents=True, exist_ok=False)',
        'OUTPUT_PARENT = Path("/content/drive/MyDrive/Video-WM/Paper-Results-V1-Checkpoint-V1")\n'
        'OUTPUT_PARENT.mkdir(parents=True, exist_ok=True)\n'
        'ACTIVE_RUN_POINTER = OUTPUT_PARENT / "active_confirmation_run.json"\n'
        'def saved_run_id(parent, pointer):\n'
        '    for candidate in (pointer, pointer.with_suffix(".previous.json")):\n'
        '        if candidate.is_file():\n'
        '            try:\n'
        '                saved = json.loads(candidate.read_text())["run_id"]\n'
        '                if not isinstance(saved, str) or saved in ("", ".", "..") or Path(saved).name != saved: raise ValueError("invalid run directory name")\n'
        '                return saved\n'
        '            except (OSError, ValueError, KeyError, TypeError) as exc:\n'
        '                print("Run pointer unreadable:", candidate, str(exc))\n'
        '    candidates = [p for p in parent.iterdir() if p.is_dir() and any((p / name).is_dir() for name in ("run_state", "attack_run", "confirmation_receiver"))]\n'
        '    if candidates:\n'
        '        restored = max(candidates, key=lambda p: p.stat().st_mtime)\n'
        '        print("Recovering existing task directory:", restored)\n'
        '        return restored.name\n'
        '    return None\n'
        'saved_id = saved_run_id(OUTPUT_PARENT, ACTIVE_RUN_POINTER)\n'
        'RESUMED_EXISTING_RUN = saved_id is not None\n'
        'if RESUMED_EXISTING_RUN: RUN_ID = saved_id\n'
        'OUTPUT_ROOT = OUTPUT_PARENT / RUN_ID\n'
        'OUTPUT_ROOT.mkdir(parents=True, exist_ok=RESUMED_EXISTING_RUN)',
    )
    setup = re.sub(
        r"FIXED_ATTEMPT_SCOPE = \{.*?\n\}\nSTAMP =",
        textwrap.dedent('''\
            FIXED_ATTEMPT_SCOPE = {
                "cases": 8, "case_ids": list(CONFIRMATION_CASES),
                "main_receiver_rows": 2880, "main_receiver_bits": 92160,
                "baseline_rows": 240, "quality_rows": 56,
                "wan_physical_encode_unconditional_range": [0, 2880],
                "preferred_m05_k0_centered_u_rows": 120,
            }
            STAMP ='''),
        setup,
        flags=re.S,
    )
    setup = re.sub(
        r"def load_companion_source\(\):.*?\n\ntry:\n",
        textwrap.dedent('''\
            def load_companion_source():
                required_relatives = (
                    "experiments/paper_results_v1/real_cli.py",
                    "experiments/paper_results_v1/attack_cli.py",
                    "experiments/paper_results_v1/confirmation_v1_cli.py",
                    "experiments/paper_results_v1/confirmation_v1.py",
                    "experiments/paper_results_v1/resume_state.py",
                    "experiments/paper_results_v1/resume_media.py",
                    "experiments/paper_results_v1/confirmation_v1.adopted.json",
                    "main/tube_state/receiver_controls_v1.py",
                    "runtime/wan/video_trajectory_temporal_edit_receiver_v1.py",
                )
                def missing_required():
                    missing = []
                    for name in required_relatives:
                        try:
                            with (PORTABLE_ROOT / name).open("rb") as stream: stream.read(1)
                        except OSError:
                            missing.append(name)
                    return missing
                archive_path = OUTPUT_ROOT / "companion_source.zip"
                missing_before = missing_required()
                status = "REFRESHED_AVAILABLE_SOURCE"
                try:
                    status = "DOWNLOADED_AND_EXTRACTED_AVAILABLE_SOURCE"
                    urllib.request.urlretrieve(COMPANION_ZIP_URL, archive_path)
                except Exception as exc:
                    if missing_before:
                        raise
                    print("Source refresh unavailable; using readable local source:", type(exc).__name__, str(exc))
                    status = "REUSED_READABLE_SOURCE_AFTER_FETCH_FAILURE"
                else:
                    PORTABLE_ROOT.mkdir(parents=True, exist_ok=True)
                    root_resolved = PORTABLE_ROOT.resolve()
                    with zipfile.ZipFile(archive_path) as archive:
                        for member in archive.infolist():
                            target = (PORTABLE_ROOT / member.filename).resolve()
                            if root_resolved not in target.parents and target != root_resolved:
                                raise RuntimeError("unsafe companion archive member")
                        archive.extractall(PORTABLE_ROOT)
                missing_after = missing_required()
                if missing_after:
                    raise FileNotFoundError("companion source is missing required files: " + repr(missing_after))
                receipt = {
                    "status": status, "source_repository": SOURCE_REPOSITORY,
                    "source_ref": SOURCE_REF, "companion_zip_url": COMPANION_ZIP_URL,
                    "source_root": str(PORTABLE_ROOT), "required_source_files": list(required_relatives),
                    "missing_before_fetch": missing_before, "manifest_required": False,
                    "digest_required": False, "git_clean_required": False,
                }
                atomic_json(OUTPUT_ROOT / "companion_source_receipt.json", receipt)
                return receipt

            try:
        '''),
        setup,
        flags=re.S,
    )
    setup = setup.replace('"status": "FIXED_TWO_PILOT_ATTEMPT_SCOPE"', '"status": "FIXED_EIGHT_SOURCE_CONFIRMATION_SCOPE"')
    setup = setup.replace('"attempted_case_ids": list(PILOT_CASES)', '"attempted_case_ids": list(CONFIRMATION_CASES)')
    setup = setup.replace('"confirmation_case_ids": list(CONFIRMATION_CASES)', '"pilot_case_ids": ["pilot_01", "pilot_02"]')
    setup = setup.replace('"confirmation_status": "NOT_EXECUTED_BY_NOTEBOOK"', '"pilot_status": "NOT_EXECUTED_BY_NOTEBOOK"')
    setup = setup.replace(
        '"reason": "Run all is fixed to the two adopted pilot cases. Full-denominator evaluation later projects absent confirmation receiver/baseline/comparison evidence to failure/unevaluable rows while quality retains its recorded state; none are attempted runs.",',
        '"reason": "Run all is fixed to the eight adopted confirmation cases; the two development pilots are not attempted or pooled into confirmation.",',
    )
    setup = setup.replace(
        'print("Fixed execution scope:", PILOT_CASES, "; confirmation is not executed by this notebook.")',
        'print("Fixed execution scope:", CONFIRMATION_CASES, "; pilots are not executed by this notebook.")',
    )
    setup = setup.replace(
        'stage_receipts = []',
        textwrap.dedent('''\
            EVENT_LOG_TARGETS = {}
            def read_event_log(path):
                path = Path(path)
                def read_list(candidate):
                    value = json.loads(candidate.read_text())
                    if not isinstance(value, list) or any(not isinstance(row, dict) for row in value):
                        raise ValueError("event log must be a list of objects")
                    return value
                if not path.is_file():
                    previous = path.with_name(path.name + ".previous")
                    if previous.is_file():
                        try: return read_list(previous)
                        except (OSError, ValueError, TypeError) as exc:
                            print("Previous event log unreadable:", previous, str(exc))
                    return []
                try:
                    return read_list(path)
                except (OSError, ValueError, TypeError) as exc:
                    damaged = path.with_name(path.name + ".unreadable." + str(time.time_ns()))
                    try:
                        path.replace(damaged)
                        print("Preserved unreadable event log:", damaged, type(exc).__name__, str(exc))
                    except OSError as preserve_error:
                        # Leave the original untouched and continue in a new segment.
                        alternate = path.with_name(path.name + ".continued." + str(time.time_ns()))
                        EVENT_LOG_TARGETS[str(path)] = alternate
                        print("Event log remains at", path, "; continuing at", alternate,
                              type(exc).__name__, str(exc), str(preserve_error))
                    previous = path.with_name(path.name + ".previous")
                    if previous.is_file():
                        try: return read_list(previous)
                        except (OSError, ValueError, TypeError) as previous_error:
                            print("Previous event log unreadable:", previous, str(previous_error))
                    return []
            def write_event_log(path, value):
                path = EVENT_LOG_TARGETS.get(str(path), Path(path))
                if path.is_file():
                    try:
                        prior = json.loads(path.read_text())
                        if isinstance(prior, list): atomic_json(path.with_name(path.name + ".previous"), prior)
                    except (OSError, ValueError, TypeError) as exc:
                        print("Previous event-log copy unavailable:", path, str(exc))
                atomic_json(path, value)
            stage_receipts = read_event_log(STAGE_RECEIPTS)
        ''') +
        'atomic_json(ACTIVE_RUN_POINTER.with_suffix(".previous.json"), {"run_id": RUN_ID, "output_root": str(OUTPUT_ROOT)})\n'
        'atomic_json(ACTIVE_RUN_POINTER, {"run_id": RUN_ID, "output_root": str(OUTPUT_ROOT)})',
    )
    setup = setup.replace('atomic_json(STAGE_RECEIPTS, stage_receipts)', 'write_event_log(STAGE_RECEIPTS, stage_receipts)')

    identity = _base_cell(base, "identity-and-init").replace(
        'print("Initialized full 10-case denominator. Run all will attempt only:", PILOT_CASES)',
        'print("Initialized fixed ten-case source plan. Run all will attempt confirmation only:", CONFIRMATION_CASES)',
    )
    identity = identity.replace(
        'logged([PYTHON, "-u", "-m", "experiments.paper_results_v1.real_cli", "--config", str(EFFECTIVE_CONFIG), "--output", str(RUN_OUTPUT), "--phase", "init"], "RUN_STATE_INIT", cwd=PORTABLE_ROOT, env=portable_env)',
        'if not state_exists(RUN_OUTPUT / "run_state.json"):\n'
        '    logged([PYTHON, "-u", "-m", "experiments.paper_results_v1.real_cli", "--config", str(EFFECTIVE_CONFIG), "--output", str(RUN_OUTPUT), "--phase", "init"], "RUN_STATE_INIT", cwd=PORTABLE_ROOT, env=portable_env)\n'
        'else:\n'
        '    record_stage("RUN_STATE_INIT", "REUSED_DURABLE_RUN_STATE", path=str(RUN_OUTPUT / "run_state.json"))',
    )
    identity = textwrap.dedent('''\
        import sys, importlib
        sys.path.insert(0, str(PORTABLE_ROOT))
        # A reused kernel can retain the old source package's __path__.
        for cached_name in tuple(sys.modules):
            if cached_name == "experiments" or cached_name.startswith("experiments."):
                del sys.modules[cached_name]
        importlib.invalidate_caches()
        from experiments.paper_results_v1.resume_state import state_exists, read_state
    ''') + identity
    environment = _base_cell(base, "environment-and-models")

    attack_init = textwrap.dedent('''\
        # Initialize the adopted confirmation attack and receiver plans.
        ATTACK_CONFIG = OUTPUT_ROOT / "effective_confirmation_config.json"
        attack_cfg = json.loads((PORTABLE_ROOT / "experiments/paper_results_v1/confirmation_v1.adopted.json").read_text())
        attack_cfg["source_run"]["run_dir"] = str(RUN_OUTPUT)
        attack_cfg["models"]["wan"]["local_snapshot_path"] = str(WAN_SNAPSHOT)
        attack_cfg["models"]["framewise"]["local_snapshot_path"] = str(FRAMEWISE_SNAPSHOT)
        attack_cfg["models"]["videoseal"].update(source_root=str(VS_ROOT), card_path=str(VS_CARD), checkpoint_path=str(VS_WEIGHT))
        attack_cfg["models"]["rivagan"].update(source_root=str(RIVA_ROOT), checkpoint_path=str(RIVA_WEIGHT))
        for model_name, optional_fields in {"videoseal": ("card_sha256", "checkpoint_sha256"), "rivagan": ("checkpoint_sha256",)}.items():
            for field in optional_fields: attack_cfg["models"][model_name].pop(field, None)
        atomic_json(ATTACK_CONFIG, attack_cfg)
        ATTACK_OUTPUT = OUTPUT_ROOT / "attack_run"
        ATTACK_TEMP = Path("/content") / ("paper-results-v1-confirmation-attack-" + RUN_ID)
        CONFIRM_OUTPUT = OUTPUT_ROOT / "confirmation_receiver"
        ATTACK_CLI = [PYTHON, "-u", "-m", "experiments.paper_results_v1.attack_cli", "--config", str(ATTACK_CONFIG)]
        CONFIRM_CLI = [PYTHON, "-u", "-m", "experiments.paper_results_v1.confirmation_v1_cli", "--config", str(ATTACK_CONFIG)]
        logged([*ATTACK_CLI, "--output", str(OUTPUT_ROOT / "attack_plan.json"), "--phase", "plan"], "ATTACK_FIXED_PLAN", cwd=PORTABLE_ROOT, env=portable_env)
        from experiments.paper_results_v1.resume_state import state_exists, read_state
        if not state_exists(ATTACK_OUTPUT / "attack_run_state.json"):
            logged([*ATTACK_CLI, "--output", str(ATTACK_OUTPUT), "--temp-root", str(ATTACK_TEMP), "--phase", "init"], "ATTACK_RUN_INIT", cwd=PORTABLE_ROOT, env=portable_env)
        else:
            record_stage("ATTACK_RUN_INIT", "REUSED_DURABLE_RUN_STATE", path=str(ATTACK_OUTPUT / "attack_run_state.json"))
        logged([*CONFIRM_CLI, "--output", str(OUTPUT_ROOT / "confirmation_plan.json"), "--phase", "plan"], "CONFIRMATION_FIXED_PLAN", cwd=PORTABLE_ROOT, env=portable_env)
        if not state_exists(CONFIRM_OUTPUT / "confirmation_state.json"):
            logged([*CONFIRM_CLI, "--output", str(CONFIRM_OUTPUT), "--attack-output", str(ATTACK_OUTPUT), "--temp-root", str(ATTACK_TEMP), "--phase", "init"], "CONFIRMATION_RUN_INIT", cwd=PORTABLE_ROOT, env=portable_env)
        else:
            record_stage("CONFIRMATION_RUN_INIT", "REUSED_DURABLE_RUN_STATE", path=str(CONFIRM_OUTPUT / "confirmation_state.json"))
    ''')

    execute = textwrap.dedent('''\
        # Resume durable units in the same task after Colab expiry.
        from experiments.paper_results_v1.resume_state import read_state
        ATTEMPTS_PATH = OUTPUT_ROOT / "phase_attempts.json"
        attempts = read_event_log(ATTEMPTS_PATH)
        GENERATION_PHASES = ("generate", "decode", "framewise", "codec")
        ATTACK_BEFORE_READ = (
            "import-main", "baseline-embed-videoseal", "baseline-embed-rivagan",
            "baseline-codec", "attack-media", "receiver-clock",
        )
        ATTACK_AFTER_READ = ("baseline-extract-videoseal", "baseline-extract-rivagan", "quality")
        ATTACK_PHASES = (*ATTACK_BEFORE_READ, *ATTACK_AFTER_READ)
        interpreters = {
            "baseline-embed-videoseal": baseline_pythons["videoseal"],
            "baseline-extract-videoseal": baseline_pythons["videoseal"],
            "baseline-embed-rivagan": baseline_pythons["rivagan"],
            "baseline-extract-rivagan": baseline_pythons["rivagan"],
        }
        def retain_attempt(case_id, family, phase, status, **fields):
            attempts.append({"case_id": case_id, "family": family, "phase": phase, "status": status, **fields})
            write_event_log(ATTEMPTS_PATH, attempts)
        def phase_status(family, case_id, phase):
            try:
                if family == "generation":
                    state = read_state(RUN_OUTPUT / "run_state.json")
                    return state["phases"][phase]["cases"][case_id]["status"]
                if family == "attack":
                    state = read_state(ATTACK_OUTPUT / "attack_run_state.json")
                    return state["phases"][phase]["cases"][case_id]["status"]
                state = read_state(CONFIRM_OUTPUT / "confirmation_state.json")
                return state["phases"][phase][case_id]["status"]
            except Exception as exc:
                print("State read requires CLI recovery:", family, phase, type(exc).__name__, str(exc))
                return None
        def invoke_owned(command, case_id, family, phase, *, interpreter=PYTHON):
            prior = phase_status(family, case_id, phase)
            if prior == "COMPLETE":
                retain_attempt(case_id, family, phase, "SKIPPED_DURABLE_COMPLETE")
                return 0
            return invoke(command, case_id, family, phase, interpreter=interpreter)
        def invoke(command, case_id, family, phase, *, interpreter=PYTHON):
            command = [interpreter, *command]
            try:
                rc = logged(command, f"{case_id}:{family}:{phase}", cwd=PORTABLE_ROOT, env=portable_env, check=False)
                retain_attempt(case_id, family, phase, "COMPLETE" if rc == 0 else "FAILED_RETAINED", returncode=rc)
                return rc
            except KeyboardInterrupt:
                retain_attempt(case_id, family, phase, "INTERRUPTED", reason="KeyboardInterrupt")
                raise
            except BaseException as exc:
                retain_attempt(case_id, family, phase, "FAILED_RETAINED", reason=f"{type(exc).__name__}: {exc}")
                return None

        interrupted = None
        real_prefix = ["-u", "-m", "experiments.paper_results_v1.real_cli", "--config", str(EFFECTIVE_CONFIG), "--output", str(RUN_OUTPUT)]
        attack_prefix = ["-u", "-m", "experiments.paper_results_v1.attack_cli", "--config", str(ATTACK_CONFIG), "--output", str(ATTACK_OUTPUT), "--temp-root", str(ATTACK_TEMP)]
        confirm_prefix = ["-u", "-m", "experiments.paper_results_v1.confirmation_v1_cli", "--config", str(ATTACK_CONFIG), "--output", str(CONFIRM_OUTPUT), "--temp-root", str(ATTACK_TEMP)]
        try:
            invoke([*real_prefix, "--phase", "preflight"], None, "generation", "preflight")
            for case_id in CONFIRMATION_CASES:
                for phase in GENERATION_PHASES:
                    invoke_owned([*real_prefix, "--phase", phase, "--case-id", case_id], case_id, "generation", phase)
            for case_id in CONFIRMATION_CASES:
                for phase in ATTACK_BEFORE_READ:
                    interpreter = interpreters.get(phase, PYTHON)
                    invoke_owned([*attack_prefix, "--phase", phase, "--case-id", case_id], case_id, "attack", phase, interpreter=interpreter)
            for case_id in CONFIRMATION_CASES:
                for phase in ("prepare", "read"):
                    invoke_owned([*confirm_prefix, "--phase", phase, "--case-id", case_id], case_id, "confirmation", phase)
            for case_id in CONFIRMATION_CASES:
                for phase in ATTACK_AFTER_READ:
                    interpreter = interpreters.get(phase, PYTHON)
                    invoke_owned([*attack_prefix, "--phase", phase, "--case-id", case_id], case_id, "attack", phase, interpreter=interpreter)
            invoke([*attack_prefix, "--phase", "evaluate"], None, "attack", "evaluate")
            invoke([*confirm_prefix, "--phase", "report"], None, "confirmation", "report")
        except KeyboardInterrupt as exc:
            interrupted = f"{type(exc).__name__}: {exc}"
            record_stage("CONFIRMATION_SEQUENCE", "INTERRUPTED", reason=interrupted)
            try:
                invoke([*attack_prefix, "--phase", "evaluate"], None, "attack", "evaluate-after-interrupt")
                invoke([*confirm_prefix, "--phase", "report"], None, "confirmation", "report-after-interrupt")
            finally:
                atomic_json(OUTPUT_ROOT / "interruption_handoff.json", {"status": "INTERRUPTED", "reason": interrupted, "attempts": attempts})
            raise
        record_stage("CONFIRMATION_SEQUENCE", "COMPLETE_WITH_RETAINED_ITEM_STATUSES", attempts=len(attempts))
    ''')

    summary = textwrap.dedent('''\
        # Compact handoff; status counts come from final retained rows, not process return codes.
        def optional_json(path):
            try: return json.loads(Path(path).read_text())
            except Exception as exc: return {"status": "UNAVAILABLE", "reason": f"{type(exc).__name__}: {exc}"}
        confirmation_report = optional_json(CONFIRM_OUTPUT / "evaluation_report.json")
        attack_report = optional_json(ATTACK_OUTPUT / "evaluation_report.json")
        confirmation_baselines = [row for row in attack_report.get("baseline_rows", []) if row.get("case_id") in CONFIRMATION_CASES]
        confirmation_quality = [row for row in attack_report.get("quality_rows", []) if row.get("case_id") in CONFIRMATION_CASES]
        handoff = {
            "schema": "paper-results-v1-confirmation-v1-handoff",
            "run_id": RUN_ID, "execution_scope": list(CONFIRMATION_CASES),
            "pilots": {"status": "NOT_EXECUTED_BY_NOTEBOOK", "case_ids": ["pilot_01", "pilot_02"]},
            "method": {"preferred_receiver": "CENTERED_U", "controls": ["RAW_U", "CENTERED_D4"]},
            "fixed_denominator": FIXED_ATTEMPT_SCOPE,
            "confirmation_report_status": confirmation_report.get("status"),
            "receiver_status_counts": confirmation_report.get("receiver_status_counts"),
            "baseline_status_counts": ({row.get("status"): sum(1 for x in confirmation_baselines if x.get("status") == row.get("status")) for row in confirmation_baselines}),
            "quality_status_counts": ({row.get("status"): sum(1 for x in confirmation_quality if x.get("status") == row.get("status")) for row in confirmation_quality}),
            "attempts": attempts,
            "checkpoint_interval_seconds": 60,
            "resume_granularity": "durable unit; unfinished trajectory/codec/encode/metric may repeat with a new attempt record",
            "progress_files": [str(folder / "progress.json") for folder in (RUN_OUTPUT, ATTACK_OUTPUT, CONFIRM_OUTPUT)],
            "files": {
                "generation_state": str(RUN_OUTPUT / "run_state.json"),
                "attack_state": str(ATTACK_OUTPUT / "attack_run_state.json"),
                "attack_report": str(ATTACK_OUTPUT / "evaluation_report.json"),
                "confirmation_state": str(CONFIRM_OUTPUT / "confirmation_state.json"),
                "confirmation_report": str(CONFIRM_OUTPUT / "evaluation_report.json"),
                "receiver_csv": str(CONFIRM_OUTPUT / "receiver_rows.csv"),
                "execution_log": str(LOG_PATH), "stage_receipts": str(STAGE_RECEIPTS),
            },
            "evidence_ceiling": "Eight predeclared source-level clusters under the fixed method; no independence, unseen, FPR, or threshold claim.",
        }
        atomic_json(OUTPUT_ROOT / "handoff_summary.json", handoff)
        print("Handoff:", OUTPUT_ROOT / "handoff_summary.json")
        print("Receiver status counts:", handoff["receiver_status_counts"])
    ''')

    cells = [
        _cell("code", "from google.colab import drive\ndrive.mount('/content/drive')\n", "drive-mount"),
        _cell("markdown", textwrap.dedent('''\
            # Paper Results V1 — fixed eight-source confirmation Run-all

            Run this notebook in a Colab GPU runtime. It attempts only `confirm_01` through `confirm_08`; the two development pilots remain outside confirmation. `CENTERED_U` is the frozen primary receiver, while `RAW_U` and `CENTERED_D4` are controls. Every failed, missing, unresolved, or unsupported row remains in the fixed denominator.

            The first Run all starts a task under `MyDrive/Video-WM/Paper-Results-V1-Checkpoint-V1`. After Colab expires, mount the same Drive and Run all again: `active_confirmation_run.json` selects that task automatically. Prior result directories remain available.

            Every completed unit is saved immediately. During model phases a full checkpoint of the last committed state is saved every 60 seconds, and again at phase exit; `progress.json` names that recoverable state. Completed results and unsupported outcomes are reused. A whole unsaved trajectory, codec call, encoder call, or metric may repeat, with a new attempt recorded. This does not restore an unsaved CUDA kernel or denoising step. Saved VAE latent supports completing remaining keys without another encode; missing local RGB is decoded from the saved MP4.

            `SOURCE_REF` in the next cell is an ordinary editable GitHub locator. Every Run all refreshes the public companion ZIP; a failed fetch may reuse readable local source. Source digests, manifests, clean checkouts, exact package versions, and GPU model names are not admission gates.
        '''), "goal"),
        _cell("code", setup, "portable-setup"),
        _cell("code", identity, "identity-and-init"),
        _cell("code", environment, "environment-and-models"),
        _cell("code", attack_init, "confirmation-plans"),
        _cell("code", execute, "fixed-confirmation-execution"),
        _cell("code", summary, "handoff-summary"),
        _cell("markdown", textwrap.dedent('''\
            ## Return the complete unique Drive directory

            Return `handoff_summary.json`, all three run-state/report trees, CSV files, NPZ sidecars, media receipts, `execution.log`, and `stage_receipts.json`. The deterministic plan comprises 16 Wan trajectories, 8 shared writer encodes, 16 writer decodes, 48 initial and 672 temporal-edit codec roundtrips, 480 clock encodes, 2,880 logical receiver rows, 240 whole-video baseline extract entries, 56 quality rows, and 16 display codecs. RivaGAN's backend expands its eight embeds into 1,448 frame encoder calls and its 120 extracts into 21,608 frame decoder calls; VideoSeal uses eight video-level embeds and 120 video-level detects. These are planned calls, not measured time or GPU memory.

            Quality covers 1,288 LPIPS batch forwards and 10,080 Farneback transitions. Unique Wan receiver inputs range from zero to 2,880; 480 is only a conditional shared-map lower bound. Interrupted uncommitted units can add attempts, which are retained separately in `unit_attempts` and `encode_attempts`. K0/K1 may share a VAE latent for the same artifact/map, but their FFT coordinates, votes, decoded bits, and durable records remain separate.

            This notebook and companion were validated locally with CPU/static/stub checks only. No real confirmation model, codec, GPU, Colab, or Drive execution occurred during delivery. Peak GPU memory and wall time remain unmeasured.
        '''), "handoff"),
    ]
    return {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU", "colab": {"name": OUTPUT.name, "provenance": []},
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3"},
        },
        "nbformat": 4, "nbformat_minor": 5,
    }


def main():
    OUTPUT.write_text(json.dumps(build_notebook(), indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(OUTPUT)


if __name__ == "__main__":
    main()
