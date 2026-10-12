"""Fixed adopted continuation notebook; reuses published source/setup/resume cells."""
from __future__ import annotations
import json
from scripts import build_local_joint_all_bit_v1_notebook as previous

ROOT=previous.ROOT
NAME="local_joint_all_bit_continue_v1"
CONFIG_PATH="experiments/wan_state_clock/configs/local_joint_all_bit_continue_v1.json"
PORTABLE_FILES=previous.PORTABLE_FILES+(
    "main/tube_state/local_joint_all_bit_continue_v1.py",
    "runtime/wan/local_joint_all_bit_continue_v1.py",
    "experiments/wan_state_clock/local_joint_all_bit_continue_v1_run.py",CONFIG_PATH)


def portable_archive():
    return previous.previous.portable_archive(PORTABLE_FILES)


def build_notebook():
    notebook=previous.build_notebook()
    old_config=json.loads((ROOT/previous.CONFIG_PATH).read_text())
    config=json.loads((ROOT/CONFIG_PATH).read_text())
    for cell in notebook["cells"]:
        if cell["cell_type"]!="code":
            continue
        source="".join(cell["source"])
        source=source.replace("FIXED_CONFIG = "+repr(old_config),"FIXED_CONFIG = "+repr(config))
        for old,new in (
            ("local_joint_all_bit_v1",NAME),
            ("Local-Joint-State-Payload-V1-All-Bit'","Local-Joint-State-Payload-V1-All-Bit-Continue'"),
            ("OUTPUT = OUTPUT_PARENT / 'resumable-task-v1'","OUTPUT = OUTPUT_PARENT / 'resumable-task-v1-2x010'"),
            ("Video-WM-Local-Joint-All-Bit-","Video-WM-Local-Joint-All-Bit-Continue-"),
            ("('BASE','TWO_MEAN_SAVED','ALLBIT_HALF','ALLBIT_TWO')","('BASE','START_SAVED','CONT_MID','CONT_FINAL')"),
            ("ALLBIT-2H","ALLBIT-CONTINUE-2x010"),
            ("FIXED_ALL_BIT","FIXED_ALL_BIT_CONTINUE"),
        ):
            source=source.replace(old,new)
        cell["source"]=source.splitlines(keepends=True)
    notebook["metadata"]["colab"]["name"]=NAME+"_colab.ipynb"
    notebook["cells"][1]=previous.previous._markdown('''
    # B: adopted All-Bit continuation, 0.10 × 2

    **Run all** executes the fixed new experiment or resumes its own unfinished units.
    Input BASE is the latest resumable-task-v1/run/inputs/terminal_latent.pt.
    START is the actual `terminal` field from that task's step_2_unit.pt (ALLBIT_TWO).
    The old HALF input, old gradient and planned delta are not new experiment inputs.
    The original 0.5 × 2 updates are not repeated. Input paths and SOURCE_REF are editable
    locators, without source/version/Git identity gates.

    The new independent Drive folder is:
    MyDrive/Video-WM/Local-Joint-State-Payload-V1-All-Bit-Continue/resumable-task-v1-2x010/.
    The old completed task stays a saved input. Reopening this notebook and Run all
    returns to the new task; each attempt's logs and physical calls remain available.
    Completed tasks and method-defined zero-gradient/active-tie terminal outcomes only report.

    Fixed views: BASE / START_SAVED / CONT_MID / CONT_FINAL.
    Primary comparison: CONT_FINAL against START_SAVED, with MID and original BASE retained.
    Complete uninterrupted budget: **1 VAE load, 6 decode, 2 decoder VJP**;
    184 ordinary chunks, 92 gradient-forward chunks, 92 nominal replay chunks;
    **352 windows, 5632 chips, 220 original metrics**, retaining all missing/failed slots.
    No encode, DiT, native writer, scheduler, codec, new source or M1/M2.

    L = 0.8 mean_32 H_0.02(margin) + 0.2 H_0.01(G).
    H_a(x)=0 above a, (a-x)^2/(2a) for 0<x<a, and a/2-x at x<=0.
    All 32 terms remain; weak positives and state effort reactivate below their targets.
    These targets are optimization preferences, not detection/quality PASS thresholds.
    Each new scalar VJP uses its actual START or MID terminal and the existing frozen
    FP32 Wan decoder/clamp/readout gradient path. Mask once, normalize in FP64, step 0.10.
    Recompute displacement from actual current terminal minus original BASE in FP64,
    project into radius <=1, and save the FP32 terminal. Only if FP32 rounding escapes
    the ball, round overshooting coordinates one FP32 neighbor toward BASE. Record
    projection, rounding, actual radius and scalar predictions. Never refill the radius.
    No old gradient reuse, scan, rollback selection, third step, cap expansion or retry to success.

    Track all 32 margins, the six original negative bits, the original 26 positive bits,
    f3b5 and each view's current weakest positive, G, losses, gained/lost bits and deficits.
    Lower loss does not guarantee new positive bits, preserve G or improve quality.
    Full 181-frame/180-pair quality, ROI inside/outside residuals, actual peak neighborhoods
    and crops are recomputed for these views. Images use RGB[0,1] and residual[-.25,.25];
    numeric quality remains unclipped. A BASE reader failure does not remove usable BASE RGB.

    Reuse complete input/gradient pairs, delta/actual terminal units, saved RGB, raw,
    score and quality. Units commit immediately; every 60 seconds the last committed
    recoverable state and progress are saved, retaining the previous readable snapshot.
    In-flight decode/VJP cannot resume inside autograd and may need a whole-unit recompute;
    every physical attempt, return and failure remains counted beyond the ordinary budget.
    Saved MID RGB is scored on CPU before remaining g2/final work. Saved FINAL RGB needs
    only CPU postprocessing. Completed raw/score is retained if its RGB is damaged;
    missing quality is reported without silently decoding a new replacement observation.
    CPU-only tails do not require CUDA or load a VAE. GPU is needed for unfinished model work.

    The preceding full source task took 17.05 minutes, peak allocated19.01GiB/reserved21.72GiB;
    this is a historical resource observation, not a runtime or memory guarantee for this task.
    Current delivery validation is CPU/fake/static. Real Colab/GPU execution is user-run.
    One known-message saved-terminal diagnostic establishes no blind decoding, FPR,
    codec survival, native/CFG causality or scientific PASS. Return the entire task folder.
    ''')
    return notebook


def main():
    (ROOT/"notebooks"/(NAME+"_colab.ipynb")).write_text(json.dumps(build_notebook(),ensure_ascii=False,indent=1)+"\n")
    (ROOT/"notebooks"/(NAME+"_portable_source.zip")).write_bytes(portable_archive())


if __name__=="__main__":
    main()
