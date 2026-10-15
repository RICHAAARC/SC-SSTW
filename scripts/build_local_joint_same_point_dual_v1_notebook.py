"""Build fixed adopted dual-arm notebook from the proven process/resume shell."""
from __future__ import annotations
import io
import json
import zipfile
from scripts import build_local_joint_gradient_sum_v1_notebook as previous
from runtime.wan.local_joint_same_point_dual_v1 import EXPECTED, FINISHED
from main.tube_state.local_joint_same_point_dual_v1 import VIEWS

ROOT=previous.ROOT
NAME="local_joint_same_point_dual_v1"
CONFIG_PATH="experiments/wan_state_clock/configs/"+NAME+".json"
PORTABLE_FILES=previous.PORTABLE_FILES+(
    "main/tube_state/"+NAME+".py", "runtime/wan/"+NAME+".py",
    "experiments/wan_state_clock/"+NAME+"_run.py",CONFIG_PATH,"docs/"+NAME+".md")


def portable_archive():
    data=io.BytesIO()
    with zipfile.ZipFile(data,"w",compression=zipfile.ZIP_DEFLATED) as archive:
        for name in dict.fromkeys(PORTABLE_FILES):
            info=zipfile.ZipInfo(name,date_time=(2026,10,15,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED
            archive.writestr(info,(ROOT/name).read_bytes())
    return data.getvalue()


def code(text):
    return dict(cell_type="code",execution_count=None,metadata={},outputs=[],source=text.strip().splitlines(keepends=True))


def build_notebook():
    notebook=previous.build_notebook()
    config=json.loads((ROOT/CONFIG_PATH).read_text())
    old=json.loads((ROOT/previous.CONFIG_PATH).read_text())
    for cell in notebook["cells"]:
        if cell["cell_type"]!="code": continue
        text="".join(cell["source"]).replace("FIXED_CONFIG = "+repr(old),"FIXED_CONFIG = "+repr(config))
        if "EXPECTED_COST = " in text:
            a,b=text.index("EXPECTED_COST = "),text.index("OUTPUT_PARENT = ")
            text=text[:a]+"EXPECTED_COST = "+repr(EXPECTED)+"\n"+text[b:]
        for a,b in ((previous.NAME,NAME),
            ("Local-Joint-State-Payload-V1-Gradient-Sum'","Local-Joint-State-Payload-V1-Same-Point-Dual'"),
            ("resumable-gradient-sum-v1'","resumable-same-point-dual-v1'"),
            ("Video-WM-Local-Joint-Gradient-Sum-","Video-WM-Local-Joint-Same-Point-Dual-"),
            (repr(previous.VIEWS),repr(VIEWS)),("GRADIENT-SUM","SAME-POINT-DUAL"),("FIXED_GRADIENT_SUM","FIXED_SAME_POINT_DUAL"),
            ("unfinished ordinary decode","unfinished fixed model work")):
            text=text.replace(a,b)
        cell["source"]=text.splitlines(keepends=True)
    notebook["cells"][1]=dict(cell_type="markdown",metadata={},source=[
        "# B: adopted same-S .10-step dual comparison\n\n",
        "Run all executes NEXT_MEAN and NEXT_ALLBIT in the same durable Drive task. Both use saved SUM_STEP, original BASE radius1 and the same support/key/message/rho/readout.\n\n",
        "MEAN maximizes original A16 (including31)+trueG with raw scalar1:1. ALLBIT minimizes historical .8*mean32 H_.02+.2*H_.01(trueG). Each masks and normalizes its one combined gradient, then takes one .10 step (positive MEAN, negative ALLBIT). No old .5 updater, projection, scan, rollback, extra step or selection.\n\n",
        "MEAN exact max ties are undefined. ALLBIT stops for a max tie only when its state loss is active (G<.01); inactive ties do not block payload descent. Zero/undefined gradients retain completed observations and fixed missing endpoint rows. Mu/gamma are ALLBIT loss parameters, never result PASS gates.\n\n",
        "Nominal budget: 1load+2gradient decodes/2VJPs+2ordinary decodes;184 forward chunks (92gradient+92ordinary),92 nominal replay. No encode/DiT/native/scheduler/codec. Serial graphs;2 new endpoints,0 independent sources. All32 bits, six hard bits, all22 state correlations/trueG, lost/gained positives and original BASE distances are retained.\n\n",
        "Four quality pairs compare each new RGB against saved BASE/S RGB, with181 frame/180 temporal rows, actual peaks and neighborhoods. Missing old RGB is MISSING without replacement decode. Historical26/32 is retained as a saved reference. State weights also differ, so the comparison does not isolate payload aggregation.\n\n",
        "60-second committed-state checkpoints and immediate unit persistence support same-Drive Run all resume; completed units skip and report-only completion loads no model. Unfinished autograd graphs do not survive; rework/failure/replay is recorded per attempt and cumulatively. No real cross-Colab recovery claim. No source/version/Git identity gate.\n\n",
        "This package is prepared locally; publish its companion before using the default GitHub source URL. GPU/Colab is user-run. Return the whole task folder including failures. No scientific PASS or combined A/B conclusion.\n"])
    text="".join(notebook["cells"][5]["source"])
    text=text.replace("result['status'] != 'COMPLETE'","result['status'] not in "+repr(FINISHED))
    text=text.replace("ordinary_chunks=result['ordinary_chunks'])","ordinary_chunks=result['ordinary_chunks'], units=result['units'], steps=result['steps'], quality=result['quality'])")
    notebook["cells"][5]=code(text)
    notebook["cells"][6]=code('''
result = json.loads((RUN_OUTPUT / 'result.json').read_text())
comparison = json.loads((RUN_OUTPUT / 'comparison.json').read_text())
print('Task status:', result['status'], '; not scientific PASS')
print('Budget:', result['expected_cost'])
print('Cumulative calls:', result['counts'])
print('This attempt:', result.get('current_attempt_counts', {}))
print('Replay:', result.get('physical_replay_chunks', {}))
for name, row in comparison['views'].items():
    summary = row.get('summary') or {}
    print(name, row['status'], 'positive=', summary.get('positive_count'),
          'G=', summary.get('state_gap'), 'deepest=', summary.get('deepest_negative_indices'),
          'hard bits=', summary.get('hard_bits'), 'reason=', row.get('reason'))
for name, row in comparison['arms'].items():
    print(name, 'construction=', row['construction'], 'all-bit/state comparisons=', row['comparisons'])
print('Four quality pairs:', comparison['quality'])
print('Failures retained:', result['failure_history'])
print(result['ceiling'])
print('Return whole directory:', OUTPUT)
''')
    notebook["metadata"]["colab"]["name"]=NAME+"_colab.ipynb"
    for i,cell in enumerate(notebook["cells"]): cell["id"]="same-point-dual-"+str(i)
    return notebook


def main():
    (ROOT/"notebooks"/(NAME+"_colab.ipynb")).write_text(json.dumps(build_notebook(),ensure_ascii=False,indent=1)+"\n")
    (ROOT/"notebooks"/(NAME+"_portable_source.zip")).write_bytes(portable_archive())


if __name__=="__main__": main()
