"""Build the actual Run-all notebook and its directly downloadable source ZIP."""
from pathlib import Path
import ast
import json
import zipfile

ROOT = Path(__file__).resolve().parents[1]
STEM = "local_joint_positive_projection_margin_v1"
SOURCE_FILES = [
    "main/__init__.py", "main/tube_state/__init__.py",
    "main/tube_state/grow_video_reference.py",
    f"main/tube_state/{STEM}.py",
    "runtime/__init__.py", "runtime/wan/__init__.py", "runtime/wan/vae.py", "runtime/wan/io.py",
    f"runtime/wan/{STEM}.py", "runtime/wan/local_joint_positive_projection_margin_quality_v1.py",
    "experiments/__init__.py", "experiments/wan_state_clock/__init__.py",
    f"experiments/wan_state_clock/{STEM}_run.py",
    f"experiments/wan_state_clock/configs/{STEM}.json", f"docs/{STEM}.md",
    f"scripts/build_{STEM}_notebook.py",
]


def build(root=ROOT):
    root = Path(root)
    destination = root/"notebooks"
    destination.mkdir(parents=True, exist_ok=True)
    cells = []

    def markdown(text):
        cells.append(dict(cell_type="markdown", metadata={}, source=text.splitlines(keepends=True)))

    def code(text):
        ast.parse(text)
        cells.append(dict(cell_type="code", metadata={}, source=text.splitlines(keepends=True),
                          execution_count=None, outputs=[]))

    # Embed the small filesystem-only helper so reopening never imports source,
    # torch or another worktree. This copy is generated from the B runtime.
    runtime_text = (root/f"runtime/wan/{STEM}.py").read_text(encoding="utf-8")
    helper_node = next(node for node in ast.parse(runtime_text).body
                       if isinstance(node, ast.FunctionDef) and node.name=="execution_artifacts")
    artifact_helper = ast.get_source_segment(runtime_text, helper_node)

    markdown("""# B：固定正裕量终态投影 · VAE/MP4 两臂观察

本 notebook 是用户运行包，尚无该新构造的真实结果。选择可用运行环境后直接 **Run all**。
固定复用卡44同次PAYLOAD_ONLY terminal；一个宿主感知最小投影候选，与P配对观察。
正裕量固定 gamma_s=1/sqrt(2)、gamma_p=1/2，对应单位法向距离1，未校准媒体效果；无0.02或其他范数cap。
完整预算：1 VAE load、2普通decode、6posterior-mode encode、2RGB8量化、2MP4写入、2probe、2MP4读回；
16新局部读表、16原GROW FFT读出，preclip/float/RGB8/MP4四层增量质量。零DiT、梯度或VJP。
FP32目标短差单独记录，合法有限候选不因短差跳过通道，也不补码重试。没有GPU型号限制，CPU可能非常慢。
本包仅固定终态的clean通道诊断，不证明多步生成期联合写入或未知起点同步。质量相对P，未设置质量PASS。

固定输入：`MyDrive/Video-WM/B-Staged-Native-State-V1/fixed-clean-pair-v1/features/PAYLOAD_ONLY/terminal.pt`。
固定输出：`MyDrive/Video-WM/B-Positive-Projection-Margin-V1/fixed-clean-pair-v1`。
保存逐层原始RGB张量约1.66GiB，另加媒体与记录。当前本地审查包的源码URL需正式发布后才可首次下载。
输入缺失记失败，不换源。已有科学记录（含临时残留）即只读展示，marker缺失也不重进；未完成尝试不自动重试。
""")
    code("from google.colab import drive\ndrive.mount('/content/drive')\n")
    markdown("""### 下载当前源码

先检查固定输出；已有记录时跳过全部下载、解压、依赖与runner。首次直接安全下载随 notebook 发布的源码包，不需要手动上传。
""")
    code('''from pathlib import Path
import json

OUTPUT = Path("/content/drive/MyDrive/Video-WM/B-Positive-Projection-Margin-V1/fixed-clean-pair-v1")
''' + artifact_helper + '''

EXISTING_ARTIFACTS = execution_artifacts(OUTPUT)
if EXISTING_ARTIFACTS:
    print("Existing scientific records: read only; no download, setup or runner.", EXISTING_ARTIFACTS)
else:
    import stat, urllib.request, zipfile
    OUTPUT.mkdir(parents=True, exist_ok=True)
    SOURCE_URL = "https://raw.githubusercontent.com/RICHAAARC/SC-SSTW/dev/local-joint-state-payload-v1/notebooks/local_joint_positive_projection_margin_v1_portable_source.zip"
    SOURCE_ROOT = Path("/content/b_positive_projection_margin_v1_source")
    SOURCE_ROOT.mkdir(parents=True, exist_ok=True)
    archive = Path("/content/b_positive_projection_margin_v1_source.zip")
    urllib.request.urlretrieve(SOURCE_URL, archive)
    with zipfile.ZipFile(archive) as package:
        for member in package.infolist():
            target = (SOURCE_ROOT/member.filename).resolve()
            if not target.is_relative_to(SOURCE_ROOT.resolve()) or stat.S_ISLNK(member.external_attr >> 16):
                raise ValueError(f"Unsafe ZIP path: {member.filename}")
        package.extractall(SOURCE_ROOT)
    print("Source:", SOURCE_ROOT)
''')
    markdown("""### 准备依赖并执行一次

模型采用可用 Wan2.1 VAE，普通FP32解码与posterior-mode编码。先做实际import probe，仅修复明确缺包或已观察到的可选torchao/FqnToConfig冲突。
优先使用本次 Colab 会话 `/content/video-wm-cache/models/Wan2.1-T2V-1.3B-Diffusers` 中available、根或已有子目录的VAE。没有可定位缓存时，唯一load可经HF模型ID下载权重，耗时计入load；选择后不换位置重试。
只在ffmpeg/ffprobe实际缺失时准备媒体依赖。GPU为环境建议，没有型号或版本门禁。
输入、模型或媒体调用失败会留在固定分母中。已有记录时直接显示，不重复执行。
""")
    code(r'''import subprocess, sys

EXISTING_ARTIFACTS = execution_artifacts(OUTPUT)
if not EXISTING_ARTIFACTS:
    LOG = OUTPUT/"setup.log"
    def command(args, label, check=True):
        print(label, flush=True)
        with LOG.open('a') as stream:
            stream.write('COMMAND '+repr(args)+'\n'); stream.flush()
            completed = subprocess.run(args, stdout=stream, stderr=subprocess.STDOUT)
        if completed.returncode and check:
            raise RuntimeError(label+' failed; see '+str(LOG))
        return completed.returncode

    # Copied from the already-used A/P2 setup; no A import/build dependency.
    def ensure_imports(probe, packages, label):
        import re
        module_packages = {package.replace('-', '_'): package for package in packages}
        repaired = set()
        while True:
            offset = LOG.stat().st_size if LOG.exists() else 0
            if not command([sys.executable, '-c', probe], ('Recheck ' if repaired else 'Probe ')+label, check=False):
                return
            with LOG.open('rb') as stream:
                stream.seek(offset); failure = stream.read().decode('utf-8', errors='replace')
            missing = re.search("ModuleNotFoundError: No module named ['\"]([^'\"]+)['\"]", failure)
            if "cannot import name 'FqnToConfig'" in failure and 'torchao.quantization' in failure:
                action = 'remove incompatible optional torchao'
                args = [sys.executable, '-m', 'pip', 'uninstall', '-y', 'torchao']
            elif missing and missing.group(1) in module_packages:
                package = module_packages[missing.group(1)]
                action = 'install missing '+package
                args = [sys.executable, '-m', 'pip', 'install', package]
            else:
                raise RuntimeError(label+' failed without a known targeted repair; see '+str(LOG))
            if action in repaired:
                raise RuntimeError(label+' still fails after '+action+'; see '+str(LOG))
            repaired.add(action)
            command(args, 'Repair '+label+': '+action)

    ensure_imports('import numpy,torch', ['numpy', 'torch'], 'tensor dependencies')
    ensure_imports('import accelerate,safetensors,huggingface_hub; from diffusers import AutoencoderKLWan',
                   ['diffusers', 'accelerate', 'safetensors', 'huggingface-hub'], 'Wan VAE imports')
    def ensure_media_tools():
        import shutil
        if not shutil.which('ffmpeg') or not shutil.which('ffprobe'):
            command(['apt-get', '-qq', 'update'], 'Prepare media dependency')
            command(['apt-get', '-qq', 'install', '-y', 'ffmpeg'], 'Install FFmpeg')
        if not shutil.which('ffmpeg') or not shutil.which('ffprobe'):
            raise RuntimeError('ffmpeg/ffprobe still unavailable; see setup.log')
    ensure_media_tools()

    with (OUTPUT/"runner.log").open("w") as log:
        process = subprocess.Popen([sys.executable, "-u", "-m",
            "experiments.wan_state_clock.local_joint_positive_projection_margin_v1_run"],
            cwd=SOURCE_ROOT, stdout=log, stderr=subprocess.STDOUT)
    import time
    try:
        while process.poll() is None:
            time.sleep(30)
            ledger_path = OUTPUT/"ledger.json"
            if ledger_path.exists():
                ledger = json.loads(ledger_path.read_text())
                returned = sum(row["status"] == "RETURNED" for row in ledger.values())
                active = [name for name, row in ledger.items() if row["status"] == "STARTED"]
                print(f"Completed fixed units: {returned}/{len(ledger)}; active: {active}")
    except KeyboardInterrupt:
        process.terminate()
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        print("Stopped; saved STARTED calls remain unresolved and will not be replayed.")
        raise
    print("Runner return code:", process.returncode)
else:
    print("Existing attempt: displaying saved records; no physical calls repeated.")
print("Output:", OUTPUT)
''')
    markdown("""### 读取同次结果

EXECUTION_COMPLETE只表示调用完成。下面显示实际状态命中、原载荷保持、质量与失败；不自动给科学PASS或启动攻击实验。
未完成记录也保留全部分母。将整个固定输出目录交回主会话核验。
""")
    code('''# This display cell needs only the standard library for all saved JSON.
saved_records = {}
for base in ("result.json", "checkpoint.json", "ledger.json", "evaluation.json", "attempt.json"):
    for suffix in ("", ".tmp"):
        name = base+suffix
        path = OUTPUT/name
        if path.is_file():
            try:
                saved_records[name] = json.loads(path.read_text())
                print("Read saved record:", name)
            except (OSError, ValueError) as exc:
                print("Unreadable saved record:", name, type(exc).__name__, str(exc))
result = next((saved_records[name] for name in ("result.json", "checkpoint.json", "result.json.tmp", "checkpoint.json.tmp")
               if isinstance(saved_records.get(name), dict)), None)
if result is not None:
    print("Status:", result.get("status", "UNKNOWN"))
    print("Fixed denominator:", result.get("fixed_denominator"))
    construction = result.get("construction", {})
    print("Construction:", {k:v for k,v in construction.items() if k not in ("rows", "actual_fp32")})
    print("Actual FP32 delta:", {k:v for k,v in construction.get("actual_fp32", {}).items() if k != "times"})
    print("Calls:", json.dumps(result.get("counts", {}), ensure_ascii=False))
    for slot, row in result.get("paired_readout", {}).items():
        print("Pair", slot, {kind: {k:v for k,v in row.get(kind, {}).items() if k != "rows"}
                            for kind in ("local_payload", "legacy_payload")})
    for slot, row in result.get("evaluation", {}).items():
        local = row.get("local", {})
        print("Readout", slot, {k:local.get(k) for k in ("status", "exact_states", "unresolved_states",
              "min_state_gap", "min_payload_margin", "state_target_shortfalls", "payload_target_shortfalls",
              "local_decoded_correct", "local_decoded_errors", "local_decoded_erasures")})
    for layer, row in result.get("quality", {}).items():
        print("Quality", layer, {k:v for k,v in row.items() if k not in ("frames", "adjacent")})
else:
    print("No readable result/checkpoint; displaying any ledger/evaluation. No replacement attempt.")
saved_calls = saved_records.get("ledger.json", saved_records.get("ledger.json.tmp", (result or {}).get("calls", {})))
for name, row in saved_calls.items():
    if row.get("status") != "RETURNED":
        print(name, row)
evaluation = saved_records.get("evaluation.json", saved_records.get("evaluation.json.tmp", {}))
if result is None:
    for slot, row in evaluation.items():
        print(slot, {kind: item.get("status", "UNKNOWN") for kind,item in row.items()})
if not (OUTPUT/"result.json").exists():
    print("Unfinished attempt: a saved STARTED call may be active or interrupted; no automatic replay.")
for arm in ("PAYLOAD_ONLY", "PROJECTION_MARGIN"):
    path = OUTPUT/"media"/f"{arm}.mp4"
    if path.exists():
        try:
            from IPython.display import display, Video
            print(arm)
            display(Video(str(path), embed=True, width=640))
        except Exception as exc:
            print("Preview unavailable:", type(exc).__name__, str(exc))
print("Return this directory for audit:", OUTPUT)
''')
    for i, cell in enumerate(cells):
        cell["id"] = f"positive-projection-{i:02d}"
    notebook = dict(nbformat=4, nbformat_minor=5, cells=cells,
                    metadata=dict(accelerator="GPU", kernelspec=dict(display_name="Python 3", language="python", name="python3"),
                                  language_info=dict(name="python")))
    path = destination/f"{STEM}.ipynb"
    path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1)+"\n", encoding="utf-8")
    archive = destination/f"{STEM}_portable_source.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as package:
        for relative in SOURCE_FILES:
            package.write(root/relative, relative)
    return path, archive


if __name__ == "__main__":
    for path in build():
        print(path)
