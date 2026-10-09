"""Small, standard-library helpers for the fixed two-pilot Colab handoff."""
from __future__ import annotations

import shutil
from collections import Counter
from pathlib import Path


ROW_KEYS = {
    "receiver": "receiver_rows",
    "baseline": "baseline_rows",
    "comparison": "comparison_rows",
    "quality": "quality_rows",
}
STATE_KEYS = {
    "receiver": "receiver_slots",
    "baseline": "baseline_slots",
    "comparison": "comparison_slots",
    "quality": "quality_rows",
}

BASELINE_RUNTIME_PROBE = (
    "import sys; from pathlib import Path; import numpy, torch; "
    "actual = Path(sys.prefix).resolve(); expected = Path(sys.argv[1]).resolve(); "
    "assert actual == expected, (actual, expected); "
    "assert sys.prefix != sys.base_prefix, (sys.prefix, sys.base_prefix); "
    "print('BASELINE_VENV_MAIN_STACK_READY', sys.prefix, numpy.__version__, torch.__version__)"
)


def _counts(rows):
    return dict(sorted(Counter(row.get("status", "MISSING_STATUS") for row in rows).items()))


def prepare_baseline_environment(*, install, probe):
    """Probe first and install only when the actual entry import is unavailable."""

    receipt = {
        "dependency_install_status": "NOT_NEEDED_IMPORT_READY",
        "entry_import_probe_status": "NOT_ATTEMPTED",
        "model_compatibility_status": "NOT_VALIDATED_REQUIRES_REAL_EMBED_EXTRACT",
    }
    try:
        probe()
        receipt["entry_import_probe_status"] = "IMPORT_READY_NO_MODEL_OR_WEIGHT_LOADED"
        return receipt
    except Exception as exc:
        receipt["initial_entry_import_probe_reason"] = f"{type(exc).__name__}: {exc}"
    try:
        install()
        receipt["dependency_install_status"] = "COMPLETE_AFTER_IMPORT_FAILURE"
    except Exception as exc:
        receipt.update(
            dependency_install_status="FAILED",
            dependency_install_reason=f"{type(exc).__name__}: {exc}",
            entry_import_probe_status="BLOCKED_DEPENDENCY_INSTALL_FAILED",
        )
        return receipt
    try:
        probe()
        receipt["entry_import_probe_status"] = "IMPORT_READY_NO_MODEL_OR_WEIGHT_LOADED"
    except Exception as exc:
        receipt.update(
            entry_import_probe_status="FAILED",
            entry_import_probe_reason=f"{type(exc).__name__}: {exc}",
        )
    return receipt


def ensure_baseline_interpreter(*, method, venv, main_python, run, remove_tree=shutil.rmtree):
    """Create or recover a system-site venv without invoking ensurepip.

    A leftover ``bin/python`` is not proof that ``venv`` creation completed:
    CPython writes an initially isolated ``pyvenv.cfg`` before ensurepip and
    only restores system-site visibility after ensurepip succeeds.  Probe the
    dedicated interpreter by importing the main NumPy/Torch stack, rebuild a
    stale partial directory with ``--without-pip``, then probe again.
    """

    method_label = str(method).upper()
    root = Path(venv)
    python = root / "bin/python"

    def probe(stage):
        if not python.is_file():
            return None
        return run(
            [str(python), "-c", BASELINE_RUNTIME_PROBE, str(root)],
            stage,
            check=False,
        )

    initial_probe_attempted = python.is_file()
    initial_probe_error = None
    try:
        initial_returncode = probe(method_label + "_VENV_REUSE_PROBE")
    except OSError as exc:
        initial_returncode = None
        initial_probe_error = f"{type(exc).__name__}: {exc}"
    if initial_returncode == 0:
        return str(python), {
            "status": "REUSED_VALIDATED",
            "python": str(python),
            "runtime_probe": "NUMPY_TORCH_IMPORTED_FROM_DEDICATED_INTERPRETER",
            "ensurepip_used": False,
        }

    if root.exists():
        remove_tree(root)
    run(
        [
            str(main_python), "-m", "venv", "--without-pip",
            "--system-site-packages", str(root),
        ],
        method_label + "_VENV_CREATE_WITHOUT_PIP",
    )
    if not python.is_file():
        raise RuntimeError(f"{method} venv creation did not produce {python}")
    created_returncode = probe(method_label + "_VENV_CREATED_RUNTIME_PROBE")
    if created_returncode != 0:
        raise RuntimeError(
            f"{method} dedicated interpreter cannot import the main NumPy/Torch stack "
            f"(returncode={created_returncode})"
        )
    return str(python), {
        "status": "REBUILT_AND_VALIDATED" if initial_probe_attempted else "CREATED_AND_VALIDATED",
        "python": str(python),
        "initial_probe_returncode": initial_returncode,
        "initial_probe_error": initial_probe_error,
        "runtime_probe": "NUMPY_TORCH_IMPORTED_FROM_DEDICATED_INTERPRETER",
        "ensurepip_used": False,
    }


def build_scope_summary(state, report, *, pilot_ids, confirmation_ids):
    """Summarize attempted pilots and unattempted confirmation without conflating them."""

    pilots = set(pilot_ids)
    confirmation = set(confirmation_ids)
    has_report = isinstance(report, dict)
    source = report if has_report else state
    source_label = "evaluation_report_final_rows" if isinstance(report, dict) else "run_state_fallback_not_evaluated"
    pilot_rows = {}
    confirmation_projection = {}
    confirmation_plan_counts = {}
    for label, report_key in ROW_KEYS.items():
        state_key = STATE_KEYS[label]
        source_rows = source.get(report_key if has_report else state_key, [])
        state_rows = state.get(state_key, [])
        pilot_rows[label] = [row for row in source_rows if row.get("case_id") in pilots]
        confirmation_projection[label] = [
            row for row in source_rows if row.get("case_id") in confirmation
        ]
        confirmation_plan_counts[label] = sum(
            row.get("case_id") in confirmation for row in state_rows
        )
    return {
        "pilot_status_source": source_label,
        "pilot_status_counts": {
            label: _counts(rows) for label, rows in pilot_rows.items()
        },
        "pilot_comparison_source_summaries": (
            [
                row for row in report.get("comparison_source_summaries", [])
                if row.get("case_id") in pilots
            ]
            if isinstance(report, dict) else []
        ),
        "pilot_comparison_cohort_summaries": (
            {
                key: value for key, value in report.get("comparison_cohort_summaries", {}).items()
                if isinstance(value, dict)
                and value.get("cohort") == "PILOT_EXCLUDED_FROM_CONFIRMATION"
            }
            if isinstance(report, dict) else {}
        ),
        "confirmation": {
            "execution_scope_status": "NOT_EXECUTED_BY_NOTEBOOK",
            "immutable_plan_counts": confirmation_plan_counts,
            "report_projection_status_counts": (
                {
                    label: _counts(rows)
                    for label, rows in confirmation_projection.items()
                }
                if isinstance(report, dict) else None
            ),
            "report_projection_semantics": (
                "The complete fixed-denominator evaluation projects absent confirmation receiver, "
                "baseline, and comparison evidence to FAILED/UNEVALUABLE rows, while quality retains "
                "its recorded state; none are attempted model failures."
                if isinstance(report, dict)
                else "Evaluation report unavailable; run-state rows are unevaluated planning state."
            ),
        },
    }


def execute_fixed_sequence(*, pilot_ids, phases, preflight, invoke, evaluate, persist):
    """Run fixed attempts once, retain launch failures, and always try report-only evaluation.

    ``invoke`` and ``evaluate`` return process return codes.  Ordinary launch
    exceptions are retained and independent work continues.  KeyboardInterrupt
    stops later heavyweight attempts, persists the interrupted row, attempts
    report-only evaluation in ``finally``, then is re-raised.
    """

    progress = {
        "preflight": {"status": "NOT_ATTEMPTED", "returncode": None},
        "phase_attempts": [],
        "evaluate": {"status": "NOT_ATTEMPTED", "returncode": None},
        "interrupted": False,
    }
    interrupted = None
    try:
        try:
            code = preflight()
            progress["preflight"] = {
                "status": "COMPLETE" if code == 0 else "FAILED_RETAINED",
                "returncode": code,
            }
        except KeyboardInterrupt as exc:
            progress["preflight"] = {
                "status": "INTERRUPTED", "returncode": None,
                "reason": f"KeyboardInterrupt: {exc}",
            }
            progress["interrupted"] = True
            interrupted = (exc, exc.__traceback__)
        except Exception as exc:
            progress["preflight"] = {
                "status": "FAILED_RETAINED", "returncode": None,
                "reason": f"{type(exc).__name__}: {exc}",
            }
        persist(progress)

        if interrupted is None:
            for case_id in pilot_ids:
                for phase in phases:
                    row = {"case_id": case_id, "phase": phase}
                    try:
                        code = invoke(case_id, phase)
                        row.update(
                            returncode=code,
                            status="COMPLETE" if code == 0 else "FAILED_RETAINED",
                        )
                    except KeyboardInterrupt as exc:
                        row.update(
                            returncode=None, status="INTERRUPTED",
                            reason=f"KeyboardInterrupt: {exc}",
                        )
                        interrupted = (exc, exc.__traceback__)
                        progress["interrupted"] = True
                    except Exception as exc:
                        row.update(
                            returncode=None, status="FAILED_RETAINED",
                            reason=f"{type(exc).__name__}: {exc}",
                        )
                    progress["phase_attempts"].append(row)
                    persist(progress)
                    if interrupted is not None:
                        break
                if interrupted is not None:
                    break
    finally:
        try:
            code = evaluate()
            progress["evaluate"] = {
                "status": "COMPLETE" if code == 0 else "FAILED_RETAINED",
                "returncode": code,
            }
        except KeyboardInterrupt as exc:
            progress["evaluate"] = {
                "status": "INTERRUPTED", "returncode": None,
                "reason": f"KeyboardInterrupt: {exc}",
            }
            progress["interrupted"] = True
            if interrupted is None:
                interrupted = (exc, exc.__traceback__)
        except Exception as exc:
            progress["evaluate"] = {
                "status": "FAILED_RETAINED", "returncode": None,
                "reason": f"{type(exc).__name__}: {exc}",
            }
        persist(progress)
    if interrupted is not None:
        raise interrupted[0].with_traceback(interrupted[1])
    return progress
