"""Build fixed saved-fork Run-all notebook and its ordinary editable source ZIP."""
import io,json,zipfile
from scripts import build_local_joint_paired_transfer_v1_notebook as previous
from runtime.wan.local_joint_fixed_paired_tail_v1 import EXPECTED
from main.tube_state.local_joint_fixed_paired_tail_v1 import VIEWS
ROOT=previous.ROOT
NAME="local_joint_fixed_paired_tail_v1"
CONFIG_PATH="experiments/wan_state_clock/configs/"+NAME+".json"
PORTABLE_FILES=previous.PORTABLE_FILES+("main/tube_state/"+NAME+".py","runtime/wan/"+NAME+".py",
    "experiments/wan_state_clock/"+NAME+"_run.py",CONFIG_PATH,"docs/"+NAME+".md")

def portable_archive():
    data=io.BytesIO()
    with zipfile.ZipFile(data,"w",compression=zipfile.ZIP_DEFLATED) as z:
        for name in dict.fromkeys(PORTABLE_FILES):z.writestr(name,(ROOT/name).read_bytes())
    return data.getvalue()

def build_notebook():
    nb=previous.build_notebook();config=json.loads((ROOT/CONFIG_PATH).read_text())
    old=json.loads((ROOT/previous.CONFIG_PATH).read_text())
    for cell in nb["cells"]:
        if cell["cell_type"]!="code":continue
        source="".join(cell["source"]).replace("FIXED_CONFIG = "+repr(old),"FIXED_CONFIG = "+repr(config))
        if "EXPECTED_COST = " in source:
            a,b=source.index("EXPECTED_COST = "),source.index("OUTPUT_PARENT = ")
            source=source[:a]+"EXPECTED_COST = "+repr(EXPECTED)+"\n"+source[b:]
        for a,b in ((previous.NAME,NAME),
                    ("OUTPUT_PARENT = Path('/content/drive/MyDrive/Video-WM/Local-Joint-State-Payload-V1-Paired-Transfer')",
                     "OUTPUT_PARENT = Path('/content/drive/MyDrive/Video-WM/Local-Joint-State-Payload-V1-Fixed-Paired-Tail')"),
                    ("OUTPUT_PARENT / 'resumable-paired-transfer-v1'","OUTPUT_PARENT / 'resumable-fixed-paired-tail-v1'"),
                    ("Video-WM-Local-Joint-Paired-Transfer-","Video-WM-Local-Joint-Fixed-Paired-Tail-"),
                    (repr(previous.VIEWS),repr(VIEWS)),("PAIRED-TRANSFER","FIXED-PAIRED-TAIL"),
                    ("FIXED_PAIRED_TRANSFER","FIXED_PAIRED_TAIL"),("WanPipeline","WanTransformer3DModel")):
            source=source.replace(a,b)
        cell["source"]=source.splitlines(keepends=True)
    nb["cells"][1]=dict(cell_type="markdown",metadata={},source=[
        "# B: fixed paired short tail\n\n",
        "**User-adopted recipe (2026-10-16). Run all directly runs the fixed pair in your GPU runtime.** It reads the completed paired-transfer task under `MyDrive/Video-WM/Local-Joint-State-Payload-V1-Paired-Transfer/resumable-paired-transfer-v1/run`; outputs go to a separate fixed-paired-tail task. No manual ZIP upload.\n\n",
        "Reuse saved fork48 z/c/u and complete UniPC history, saved context prompt/negative/dtype, saved ZERO latent/RGB, and the exact direction in units/gradient.pt. No TERMINAL-difference or older BASE substitute, new gradient, or regenerated 50-step baseline. Unreadable/invalid numerical fields report their actual error. Source URLs/versions are optional location notes, not admission gates.\n\n",
        "ZERO_TAIL and CONTROL_TAIL restore independent copies of the same fork. Both use the same frozen model, arithmetic and CPU/device round trip before restoring history. At48 use saved c/u; at49 run real conditional and unconditional predictions and native update. CONTROL_TAIL uses .02 conditional-clean at CFG5, .10 nominal CFG-clean; ZERO_TAIL uses zero. Preserve actual per-coordinate effects and final c/u predictions. No amplitude change, scan, automatic retry or selection.\n\n",
        "Nominal total:1 preparation (frozen Transformer weights only, no new text encoding/noise),4DiT,4native updates,1VAE load,2ordinary decodes=92forward chunks.0VJP/gradient decode/replay/encode/codec/newsource/baseline. Both tails finish before the one VAE load. Two new endpoints retain176windows/2816chips/110metrics, all32payload/22state and raw components. Saved ZERO is reused without another decode.\n\n",
        "Two comparisons: zero tail vs saved ZERO; controlled tail vs zero tail. Each quality pair retains181frames/180transitions, local/temporal peaks and images. Results remain descriptive: no new full32/quality PASS threshold or unique Jacobian attribution.\n\n",
        "Full history checkpoints after each update, persisted prediction/decode/readout units,60-second snapshots and separate cumulative/current/failed/completion-unknown counts support same-Drive Run all. Completed units are reused; missing committed evidence errors instead of silently rerunning. Interrupted unfinished physical work can recur on a later user Run all and stays in cumulative accounting. Per-arm attempt/preparation records expose cross-attempt pairs without silently rerunning completed arms. Real cross-Colab recovery is not established by CPU fixtures. GPU execution belongs to the user; no main merge.\n"])
    # The previous concise report cell works with the new paired changes and receipts.
    nb["metadata"]["colab"]["name"]=NAME+"_colab.ipynb"
    for i,c in enumerate(nb["cells"]):c["id"]="fixed-paired-tail-"+str(i)
    return nb

def main():
    (ROOT/"notebooks"/(NAME+"_colab.ipynb")).write_text(json.dumps(build_notebook(),ensure_ascii=False,indent=1)+"\n")
    (ROOT/"notebooks"/(NAME+"_portable_source.zip")).write_bytes(portable_archive())
if __name__=="__main__":main()
