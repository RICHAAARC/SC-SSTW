"""Build the reviewable fixed paired-transfer notebook and editable source ZIP."""
from __future__ import annotations
import io
import json
import zipfile
from scripts import build_local_joint_same_point_dual_v1_notebook as previous
from runtime.wan.local_joint_paired_transfer_v1 import EXPECTED
from main.tube_state.local_joint_paired_transfer_v1 import VIEWS
ROOT=previous.ROOT
NAME="local_joint_paired_transfer_v1"
CONFIG_PATH="experiments/wan_state_clock/configs/"+NAME+".json"
PORTABLE_FILES=previous.PORTABLE_FILES+(
    "main/tube_state/"+NAME+".py","runtime/wan/"+NAME+".py",
    "runtime/wan/local_joint_paired_transfer_native_v1.py","runtime/wan/grow_video_reference.py",
    "runtime/wan/trajectory.py","main/tube_state/grow_video_reference.py",
    "experiments/wan_state_clock/"+NAME+"_run.py",CONFIG_PATH,"docs/"+NAME+".md")
def portable_archive():
    data=io.BytesIO()
    with zipfile.ZipFile(data,"w",compression=zipfile.ZIP_DEFLATED) as archive:
        for name in dict.fromkeys(PORTABLE_FILES):
            archive.writestr(name,(ROOT/name).read_bytes())
    return data.getvalue()
def build_notebook():
    notebook=previous.build_notebook()
    config=json.loads((ROOT/CONFIG_PATH).read_text());old=json.loads((ROOT/previous.CONFIG_PATH).read_text())
    for cell in notebook["cells"]:
        if cell["cell_type"]!="code":continue
        text="".join(cell["source"]).replace("FIXED_CONFIG = "+repr(old),"FIXED_CONFIG = "+repr(config))
        if "EXPECTED_COST = " in text:
            a,b=text.index("EXPECTED_COST = "),text.index("OUTPUT_PARENT = ")
            text=text[:a]+"EXPECTED_COST = "+repr(EXPECTED)+"\n"+text[b:]
        for a,b in ((previous.NAME,NAME),("Local-Joint-State-Payload-V1-Same-Point-Dual","Local-Joint-State-Payload-V1-Paired-Transfer"),
            ("resumable-same-point-dual-v1","resumable-paired-transfer-v1"),("Video-WM-Local-Joint-Same-Point-Dual-","Video-WM-Local-Joint-Paired-Transfer-"),
            (repr(previous.VIEWS),repr(VIEWS)),("SAME-POINT-DUAL","PAIRED-TRANSFER"),("FIXED_SAME_POINT_DUAL","FIXED_PAIRED_TRANSFER"),
            ("from diffusers import AutoencoderKLWan","from diffusers import AutoencoderKLWan, WanPipeline, UniPCMultistepScheduler")):
            text=text.replace(a,b)
        cell["source"]=text.splitlines(keepends=True)
    # Environment notes are optional. Only actual probe imports/capability failures
    # may trigger dependency repair or stop execution.
    environment="".join(notebook["cells"][4]["source"])
    environment=environment.replace("probe_path = OUTPUT / 'dependency_probe.json'",
                                    "probe_path = ATTEMPT_OUTPUT / 'dependency_probe.json'")
    environment=environment.replace(
        'probe_lines += ["pathlib.Path(" + repr(str(probe_path)) + ").write_text(json.dumps(row))"]',
        'probe_lines += ["try:", "    pathlib.Path(" + repr(str(probe_path)) + ").write_text(json.dumps(row))",\n'
        '                    "except Exception as exc:", "    print(\'Optional dependency note unavailable:\', repr(exc), flush=True)"]')
    environment=environment.replace(
        "    environment_receipt = json.loads(probe_path.read_text())",
        "    try:\n"
        "        environment_receipt = json.loads(probe_path.read_text())\n"
        "        if not isinstance(environment_receipt, dict): raise ValueError('environment note must be an object')\n"
        "    except Exception as exc:\n"
        "        print('Optional dependency note unavailable:', repr(exc), flush=True)\n"
        "        environment_receipt = dict(environment_note_error=repr(exc))")
    environment=environment.replace(
        "    write_json(ATTEMPT_OUTPUT / 'environment_receipt.json', environment_receipt)",
        "    try:\n"
        "        write_json(ATTEMPT_OUTPUT / 'environment_receipt.json', environment_receipt)\n"
        "    except Exception as exc:\n"
        "        print('Optional environment receipt unavailable:', repr(exc), flush=True)")
    notebook["cells"][4]=previous.code(environment)
    notebook["cells"][1]=dict(cell_type="markdown",metadata={},source=[
        "# B: terminal/native paired transfer\n\n",
        "**User-adopted recipe (2026-10-15): historical ALLBIT, terminal/CFG-clean amplitude .10, conditional-clean .02 at CFG5, with the complete fixed budget below.** Run all directly executes this recipe in the user's GPU runtime. The local package passed bounded A2/A3/A5 review; real execution had not occurred at publication.\n\n",
        "One original yellow-sailboat seed/prompt, fresh ZERO-control50-step baseline with CFG5. Historical JOINT BASE and S are not reused or rebuilt. Fresh terminal VAE loss gradient d=-Pgrad/||Pgrad||. TERMINAL=z_ZERO+.10d; NATIVE injects conditional-clean .02d at update48 (zero based) with complete original UniPC history and then reevaluates the real final update49. No cold start, posterior lift, scan, projection, extra optimization or full32 gate. Record actual clean/CFG/native displacement; old cap1 and radius1 are not equivalent.\n\n",
        "Nominal cost:1generation preparation,102 ordinary DiT calls,52 native updates,2VAE loads,1gradient decode/VJP+2ordinary decodes.138 forward chunks and46 nominal checkpoint replays;0VAE encode/DiT VJP/DiT recompute/codec. ZERO gradient RGB is reused.3endpoints retain264windows/4224chips/165metrics, all32 payload and22state rows. Gradient-objective reader adds88windows/1408chips/55metrics using the same ZERO RGB, without an extra decode or independent endpoint.2quality pairs retain181frames/180transitions and actual peak neighborhoods. No new independent sample.\n\n",
        "Transformer offloads to CPU around VAE work; VAE is released before the native tail. Temporary boundary spool uses the proven checkpoint path; prior real geometry required roughly64.53GiB temporary disk, an estimate, not a gate. Actual calls, chunks, spool and storage are recorded. Extra loads/work after interruption remain visible.\n\n",
        "Every completed native step saves full history; gradient/RGB and fixed readers commit as durable units.60-second snapshots preserve last completed work. Same-Drive Run all skips complete units and repairs reports without models or rescoring saved reader/quality metrics; only missing images are redrawn. Failed/interrupted VJP forward artifacts are labeled forward-only. Counts separate completed, failed and completion-unknown interruptions. Saved effective task parameters remain in use on resume. Missing committed evidence is reported. Real cross-Colab recovery remains untested.\n\n",
        "Outcomes: terminal improves/native loses -> reject this transport construction; both improve -> limited support; terminal finite step fails -> cannot blame tail. Missing/undefined stays unresolved. No generalization, robustness or multi-step joint-method PASS.\n\n",
        "This notebook loads its ordinary companion ZIP from the existing dev/local-joint-state-payload-v1 branch. Validation at publication is CPU/fake/static; the user performs the real top-to-bottom Colab Run all. Return the whole task folder including failures. Publication does not merge main or imply scientific PASS.\n"])
    notebook["cells"][6]=previous.code('''
result = json.loads((RUN_OUTPUT / 'result.json').read_text())
comparison = json.loads((RUN_OUTPUT / 'comparison.json').read_text())
print('Status:', result['status'], '; scientific PASS: False')
print('Selection:', result['selection_status'])
print('Nominal:', result['expected_cost'])
print('Cumulative:', result['counts'], 'current attempt:', result.get('current_attempt_counts'))
print('Replay:', result.get('physical_replay_chunks'), 'ordinary chunks:', result['ordinary_chunks'])
for name, row in comparison['views'].items():
    s = row.get('summary') or {}
    print(name, row['status'], 'positive=', s.get('positive_count'), 'trueG=', s.get('state_gap'),
          'allbit=', s.get('allbit_loss'), 'hard bits=', s.get('hard_bits'), 'reason=', row.get('reason'))
print('Coordinate receipts:', comparison['transport'])
print('Bounded outcome:', comparison['bounded_outcome'], comparison['changes'])
print('Quality:', comparison['quality'])
print('Failures retained:', result['failure_history'])
print(result['ceiling'])
print('Return whole directory:', OUTPUT)
''')
    notebook["metadata"]["colab"]["name"]=NAME+"_colab.ipynb"
    for i,cell in enumerate(notebook["cells"]):cell["id"]="paired-transfer-"+str(i)
    return notebook
def main():
    (ROOT/"notebooks"/(NAME+"_colab.ipynb")).write_text(json.dumps(build_notebook(),ensure_ascii=False,indent=1)+"\n")
    (ROOT/"notebooks"/(NAME+"_portable_source.zip")).write_bytes(portable_archive())
if __name__=="__main__":main()
