"""Build the fixed two-pilot temporal-attack recovery notebook."""
from __future__ import annotations

import json
import textwrap
from pathlib import Path

from experiments.paper_results_v1.companion_source import build_companion_zip


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "notebooks/paper_results_v1_temporal_attack_recovery_colab.ipynb"


def _cell(kind, source, ident):
    row = {"cell_type": kind, "metadata": {}, "source": source.splitlines(keepends=True), "id": ident}
    if kind == "code":
        row.update(execution_count=None, outputs=[])
    return row


def _accepted_cells():
    notebook = json.loads((ROOT / "notebooks/paper_results_v1_temporal_attack_two_pilot_colab.ipynb").read_text())
    return {cell["id"]: "".join(cell["source"]) for cell in notebook["cells"]}


def build_notebook():
    companion = build_companion_zip()
    accepted = _accepted_cells()
    setup = accepted["plan-init"]
    setup = setup.replace(
        '"import-main", "baseline-embed-videoseal", "baseline-embed-rivagan",\n    "baseline-codec", "attack-media", "receiver-clock", "receiver-read",',
        '"index-saved-media", "receiver-clock", "receiver-read",',
    )
    setup = setup.replace(
        'OUTPUT_ROOT = Path("/content/drive/MyDrive/Video-WM/Paper-Results-V1-Temporal-Attack") / RUN_ID\n'
        'RUN_OUTPUT = OUTPUT_ROOT / "run_state"\n'
        'TEMP_ROOT = Path("/content") / ("paper-results-v1-attack-temp-" + RUN_ID)',
        'RECOVERY_PARENT = Path("/content/drive/MyDrive/Video-WM/Paper-Results-V1-Temporal-Attack-Recovery")\n'
        'RECOVERY_POINTER = RECOVERY_PARENT / "source-20261010T031617441784Z-2536172d.json"\n'
        'RECOVERY_PARENT.mkdir(parents=True, exist_ok=True)\n'
        'if RECOVERY_POINTER.is_file():\n'
        '    OUTPUT_ROOT = Path(json.loads(RECOVERY_POINTER.read_text())["output_root"])\n'
        '    RUN_ID = OUTPUT_ROOT.name\n'
        'else:\n'
        '    OUTPUT_ROOT = RECOVERY_PARENT / RUN_ID\n'
        'RUN_OUTPUT = OUTPUT_ROOT / "run_state"\n'
        'TEMP_ROOT = Path("/content") / ("paper-results-v1-attack-recovery-temp-" + RUN_ID)',
    )
    setup = setup.replace(
        'SOURCE_RUN = Path("/content/drive/MyDrive/Video-WM/Paper-Results-V1-Two-Pilot/20261009T123350716100Z-f5e1f880/run_state")',
        'SOURCE_RUN = Path("/content/drive/MyDrive/Video-WM/Paper-Results-V1-Two-Pilot/20261009T123350716100Z-f5e1f880/run_state")\n'
        'SOURCE_ATTACK_RUN = Path("/content/drive/MyDrive/Video-WM/Paper-Results-V1-Temporal-Attack/20261010T031617441784Z-2536172d")',
    )
    setup = setup.replace(
        'OUTPUT_ROOT.mkdir(parents=True, exist_ok=False); CACHE_ROOT.mkdir(parents=True, exist_ok=True)',
        'OUTPUT_ROOT.mkdir(parents=True, exist_ok=True); CACHE_ROOT.mkdir(parents=True, exist_ok=True)',
    )
    setup = setup.replace(
        'receipts = []\ndef record(stage, status, **fields):',
        'receipts = json.loads(RECEIPTS.read_text()) if RECEIPTS.is_file() else []\ndef record(stage, status, **fields):',
    )
    setup = setup.replace(
        'required_source = PORTABLE_ROOT / "experiments/paper_results_v1/attack_cli.py"',
        'required_source = PORTABLE_ROOT / "experiments/paper_results_v1/attack_recovery_cli.py"',
    ).replace(
        'companion source is missing experiments/paper_results_v1/attack_cli.py',
        'companion source is missing experiments/paper_results_v1/attack_recovery_cli.py',
    )
    old_tail = (
        'CLI = [sys.executable, "-u", "-m", "experiments.paper_results_v1.attack_cli", "--config", str(CONFIG)]\n'
        'logged([*CLI, "--output", str(OUTPUT_ROOT / "fixed_plan.json"), "--phase", "plan"], "fixed-plan", env=ENV, check=True)\n'
        'logged([*CLI, "--output", str(RUN_OUTPUT), "--temp-root", str(TEMP_ROOT), "--phase", "init"], "initialize", env=ENV, check=True)\n'
        'print("Initialized", RUN_ID, "with all 10 cases; only pilot_01/pilot_02 will be attempted.")'
    )
    new_tail = (
        'CLI = [sys.executable, "-u", "-m", "experiments.paper_results_v1.attack_recovery_cli", "--config", str(CONFIG), "--source-run", str(SOURCE_ATTACK_RUN)]\n'
        'if not (RUN_OUTPUT / "attack_run_state.json").is_file():\n'
        '    logged([*CLI, "--output", str(RUN_OUTPUT), "--temp-root", str(TEMP_ROOT), "--phase", "init"], "initialize-recovery", env=ENV, check=True)\n'
        '    atomic_json(RECOVERY_POINTER, {"source_run": str(SOURCE_ATTACK_RUN), "output_root": str(OUTPUT_ROOT)})\n'
        'state = json.loads((RUN_OUTPUT / "attack_run_state.json").read_text())\n'
        'atomic_json(OUTPUT_ROOT / "fixed_recovery_plan.json", {"fixed_denominator": state["fixed_denominator"], "recovery_budget": state["recovery"]["budget"], "confirmation": "NOT_EXECUTED_BY_NOTEBOOK"})\n'
        'print("Recovery directory", OUTPUT_ROOT, "for", SOURCE_ATTACK_RUN)'
    )
    if old_tail not in setup:
        raise RuntimeError("accepted attack notebook setup changed")
    setup = setup.replace(old_tail, new_tail)

    execute = textwrap.dedent('''\
        # Resume only missing pilot work. Completed item receipts are skipped.
        attempts = json.loads((OUTPUT_ROOT / "attempts.json").read_text()) if (OUTPUT_ROOT / "attempts.json").is_file() else []
        interpreters = {"baseline-extract-videoseal": VS_PYTHON, "baseline-extract-rivagan": RV_PYTHON}
        interrupted = None
        try:
            for case_id in PILOTS:
                for phase in PHASES:
                    python = interpreters.get(phase, sys.executable)
                    command = [python, "-u", "-m", "experiments.paper_results_v1.attack_recovery_cli", "--config", str(CONFIG), "--source-run", str(SOURCE_ATTACK_RUN), "--output", str(RUN_OUTPUT), "--temp-root", str(TEMP_ROOT), "--phase", phase, "--case-id", case_id]
                    try:
                        rc = logged(command, case_id + ":" + phase, env=ENV, check=False)
                        attempts.append({"case_id": case_id, "phase": phase, "returncode": rc}); atomic_json(OUTPUT_ROOT / "attempts.json", attempts)
                        if rc:
                            logged([sys.executable, "-m", "experiments.paper_results_v1.attack_recovery_cli", "--config", str(CONFIG), "--source-run", str(SOURCE_ATTACK_RUN), "--output", str(RUN_OUTPUT), "--phase", "record-failure", "--case-id", case_id, "--reason", phase + ":subprocess exited " + str(rc)], case_id + ":retain-failure", env=ENV)
                    except KeyboardInterrupt as exc:
                        interrupted = exc; attempts.append({"case_id": case_id, "phase": phase, "status": "INTERRUPTED"}); atomic_json(OUTPUT_ROOT / "attempts.json", attempts)
                        try: logged([sys.executable, "-m", "experiments.paper_results_v1.attack_recovery_cli", "--config", str(CONFIG), "--source-run", str(SOURCE_ATTACK_RUN), "--output", str(RUN_OUTPUT), "--phase", "record-failure", "--case-id", case_id, "--reason", phase + ":KeyboardInterrupt"], case_id + ":retain-interrupt", env=ENV)
                        except BaseException: pass
                        raise
                    except BaseException as exc:
                        attempts.append({"case_id": case_id, "phase": phase, "status": "FAILED_RETAINED", "reason": f"{type(exc).__name__}: {exc}"}); atomic_json(OUTPUT_ROOT / "attempts.json", attempts)
        finally:
            try: logged([sys.executable, "-m", "experiments.paper_results_v1.attack_recovery_cli", "--config", str(CONFIG), "--source-run", str(SOURCE_ATTACK_RUN), "--output", str(RUN_OUTPUT), "--phase", "evaluate"], "final-evaluate", env=ENV)
            except BaseException as exc: record("final-evaluate", "FAILED", reason=f"{type(exc).__name__}: {exc}")
            state = json.loads((RUN_OUTPUT / "attack_run_state.json").read_text())
            records = state.get("records", {})
            summary = {"run_id": RUN_ID, "recovery_source": str(SOURCE_ATTACK_RUN), "scope": {"attempted": list(PILOTS), "confirmation": "NOT_EXECUTED_BY_NOTEBOOK"}, "run_output": str(RUN_OUTPUT), "report": str(RUN_OUTPUT / "evaluation_report.json"), "recovery_phase_status": state["recovery"]["phases"], "item_status_counts": state["recovery"].get("item_status_counts"), "source_actual_calls": state["recovery"].get("source_actual_calls"), "new_actual_calls": {"receiver_by_case": records, "baseline_edit_codecs": state["recovery"].get("actual_calls", {}).get("baseline_edit_codecs", []), "baseline_extracts_by_case": {case_id: records.get(case_id, {}).get("baseline_extract_calls", {"attempted": 0, "completed": 0, "failed": 0, "unfinished": 0, "records": []}) for case_id in PILOTS}}, "interrupted": interrupted is not None}
            atomic_json(OUTPUT_ROOT / "handoff_summary.json", summary); print(json.dumps(summary, indent=2))
        if interrupted is not None: raise interrupted
    ''')
    handoff = textwrap.dedent('''\
        print("Return recovery directory:", OUTPUT_ROOT)
        print("Run all reuses this source-run recovery directory through", RECOVERY_POINTER)
        print("Required: handoff_summary.json, execution.log, attempts.json, fixed_recovery_plan.json, run_state/attack_run_state.json, run_state/evaluation_report.json, CSV files, records/, reused_records/, and media/.")
        print("The source failure history is retained; confirmation remains NOT_EXECUTED_BY_NOTEBOOK.")
    ''')
    intro = """# Two-pilot temporal-attack recovery\n\nRun all. This creates one new Drive recovery directory for the saved failed run and reuses it on later Run all attempts. It never changes the source run. It decodes 116 saved MP4 files, reuses FULL receiver evidence and ten quality rows, creates only 56 missing baseline temporal publications, and attempts only missing receiver/baseline/quality work. Real GPU/model/codec execution has not been run while building this notebook.\n"""
    notebook = {
        "nbformat": 4, "nbformat_minor": 5,
        "metadata": {"accelerator": "GPU", "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}},
        "cells": [
            _cell("code", "from google.colab import drive\ndrive.mount('/content/drive')\n", "drive-mount"),
            _cell("markdown", intro, "intro"), _cell("code", setup, "recovery-init"),
            _cell("code", accepted["environment"], "environment"),
            _cell("code", execute, "execute-recovery"), _cell("code", handoff, "handoff"),
        ],
    }
    OUTPUT.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return {"notebook": str(OUTPUT), "companion": companion}


if __name__ == "__main__":
    print(json.dumps(build_notebook(), indent=2))
