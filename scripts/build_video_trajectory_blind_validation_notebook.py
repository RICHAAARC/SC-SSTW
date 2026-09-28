"""Reuse the successful ten-slot notebook for six saved-state tail diagnostics.

Without --source-sha, produce an explicit unpublished draft. After the candidate
source is published, bind its verified immutable SHA with this same builder.
"""
from __future__ import annotations
import argparse
import ast
import copy
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "notebooks/rgb_dct_structured_terminal_feedback_v1_colab.ipynb"
OUTPUT = ROOT / "notebooks/video_trajectory_blind_validation_v1_colab.ipynb"


def build(source_sha=None, output=OUTPUT):
    if source_sha is not None and not re.fullmatch(r"[0-9a-f]{40}", source_sha):
        raise ValueError("published immutable source SHA required")
    notebook = json.loads(TEMPLATE.read_text())
    cells = copy.deepcopy(notebook["cells"])
    def get(i): return "".join(cells[i]["source"])
    def put(i, text): cells[i]["source"] = text.splitlines(keepends=True)
    put(1, """# Video Trajectory Blind Validation V1: six saved-state tail points

This notebook reuses the environment/setup/worker approach of the successfully
executed 20260928 structured-feedback notebook. It loads the original saved
states and complete UniPC histories for the same two sources at T46/47/48.
Each point is independent with spent=0. The only optimizer revision is the
continuous certified solver; directions, budgets, scales and acceptance guards
are unchanged. All six outcomes survive setup, missing-artifact and worker failures.

Publication status: """ + ("immutable candidate SHA bound.\n" if source_sha else
"**DRAFT — candidate source not yet published; SOURCE_SHA is intentionally unset.**\n") + """
After publication, run all once in Colab. The user executes GPU work. Rebuilding
the basis must match the original full raw/masked hashes. This stage reports
real-tail acceptance only, with no MP4, sequential control or blind-detection
success claim. Results are in the new output's `six_point_run/result.json`.
""")
    setup = get(2).replace("SOURCE_SHA = '2cfb709d7980068e00c5c3796198fa8960d6bc54'", f"SOURCE_SHA = {source_sha!r}")
    setup = setup.replace("ARMS = ('OFF', 'SINGLE49', 'ONLY49', 'MULTI46_47_48_49', 'MULTI46_47_48_FREE49')",
                          "ARMS = ('FIXED_T46', 'FIXED_T47', 'FIXED_T48')")
    setup = setup.replace("{'sources': 2, 'arms_per_source': 5, 'mp4_score_slots': 10, 'frames': 1810}",
        "{'sources': 2, 'independent_points_per_source': 3, 'real_tail_points': 6, 'mp4_slots': 0}")
    setup = setup.replace("Video-WM/RGB-DCT-Structured-Terminal-Feedback-V1')",
                          "Video-WM/Video-Trajectory-Blind-Validation-V1')")
    setup += "\nSAVED_RUN = Path('/content/drive/MyDrive/Video-WM/RGB-DCT-Structured-Terminal-Feedback-V1/20260928T053626644222Z')\nRUN_OUTPUT = OUTPUT / 'six_point_run'\n"
    setup += "if SOURCE_SHA is None:\n    raise RuntimeError('Unpublished candidate draft: bind the published source SHA with the notebook builder before Run all.')\n"
    put(2, setup)
    checkout = get(5).replace("SC-SSTW-RGB-DCT-STRUCTURED-TERMINAL-FEEDBACK-", "SC-SSTW-VIDEO-TRAJECTORY-BLIND-VALIDATION-")
    checkout = checkout.replace("'dev/rgb-dct-structured-terminal-feedback-v1'", "'dev/video-trajectory-blind-validation-v1'")
    checkout = checkout.replace("'docs/rgb_dct_structured_terminal_feedback_v1_protocol.md'", "'docs/video_trajectory_blind_validation_v1.md'")
    checkout = checkout.replace("'experiments/wan_state_clock/configs/rgb_dct_structured_terminal_feedback_v1.json'", "'experiments/wan_state_clock/configs/video_trajectory_blind_validation_v1.json'")
    checkout = checkout.replace("'experiments/wan_state_clock/rgb_dct_structured_terminal_feedback_run.py'", "'experiments/wan_state_clock/video_trajectory_blind_validation_run.py'")
    checkout = checkout.replace("'main/tube_state/rgb_dct_structured_feedback.py'", "'main/tube_state/rgb_dct_continuous_solver.py'")
    checkout = checkout.replace("    import hashlib", "    source_files['shared_real_loop'] = REPO / 'experiments/wan_state_clock/rgb_dct_structured_terminal_feedback_run.py'\n    import hashlib")
    put(5, checkout)
    run = get(7).replace("'experiments.wan_state_clock.rgb_dct_structured_terminal_feedback_run'", "'experiments.wan_state_clock.video_trajectory_blind_validation_run'")
    run = run.replace("'--output', str(OUTPUT),", "'--saved-run', str(SAVED_RUN),\n    '--output', str(RUN_OUTPUT), '--execute-real-tail',")
    run = run.replace("RESULT_PATH = OUTPUT / 'result.json'", "RESULT_PATH = RUN_OUTPUT / 'result.json'")
    put(7, run)
    put(8, """result = json.loads(RESULT_PATH.read_text(encoding='utf-8'))
if result.get('source_receipt', {}).get('git_head') != SOURCE_SHA:
    raise RuntimeError('result source SHA mismatch')
if result.get('fixed_denominator') != FIXED_DENOMINATOR:
    raise RuntimeError('fixed six-point denominator mismatch')
if tuple(result.get('cases', {}).keys()) != CASE_IDS:
    raise RuntimeError('fixed source roster mismatch')
print('status:', result.get('status'))
print('fixed point counts:', result.get('point_counts'))
for case_id in CASE_IDS:
    case = result['cases'][case_id]
    if tuple(case['controls']) != ARMS:
        raise RuntimeError('fixed point roster mismatch')
    print(case_id, case.get('status'), case.get('stage'))
    for arm in ARMS:
        rows = case['controls'][arm]
        for row in rows:
            print(arm, row.get('outcome'), row.get('accepted_scale'), row.get('error'))
            for trial in row.get('candidates', []):
                print('  scale', trial['scale'], trial['decision'], trial.get('C'),
                      trial.get('lost_groups'), trial.get('true_hinge_improvement'))
print('evidence ceiling:', result['evidence_ceiling'])
print('retained result:', RESULT_PATH)
""")
    for i, cell in enumerate(cells):
        cell["id"] = f"video-trajectory-blind-validation-{i}"
        if cell["cell_type"] == "code":
            cell["execution_count"], cell["outputs"] = None, []
            ast.parse("".join(cell["source"]))
    assert "".join(cells[0]["source"]) == "from google.colab import drive\ndrive.mount('/content/drive')\n"
    notebook["cells"] = cells
    notebook.setdefault("metadata", {})["candidate_binding"] = dict(
        status="PUBLISHED_SHA_BOUND" if source_sha else "UNPUBLISHED_DRAFT",
        source_sha=source_sha,
        template_sha256=hashlib.sha256(TEMPLATE.read_bytes()).hexdigest(),
        template_real_run="20260928T053626644222Z",
        validation="AST and structure only; GPU execution is user-run")
    output = Path(output)
    output.write_text(json.dumps(notebook, ensure_ascii=False, indent=1)+"\n")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-sha")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    print(build(args.source_sha, args.output))
