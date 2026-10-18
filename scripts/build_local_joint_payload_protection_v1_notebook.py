"""Card51 actual notebook and editable portable source; no identity gates."""
from pathlib import Path
import ast
import json
import zipfile

ROOT=Path(__file__).resolve().parents[1]
STEM="local_joint_payload_protection_v1"
SOURCE_FILES=[
    "main/__init__.py","main/tube_state/__init__.py",
    "main/tube_state/grow_video_reference.py",
    "main/tube_state/local_joint_positive_projection_margin_v1.py",
    "main/tube_state/local_payload_secded_39_32_v1.py",
    "main/tube_state/local_joint_payload_protection_v1.py",
    "runtime/__init__.py","runtime/wan/__init__.py","runtime/wan/vae.py","runtime/wan/io.py",
    "runtime/wan/local_joint_positive_projection_margin_v1.py",
    "runtime/wan/local_joint_positive_projection_margin_quality_v1.py",
    "runtime/wan/local_joint_payload_protection_v1.py",
    "runtime/wan/local_joint_payload_protection_evaluation_v1.py",
    "experiments/__init__.py","experiments/wan_state_clock/__init__.py",
    "experiments/wan_state_clock/local_joint_payload_protection_v1_run.py",
    "experiments/wan_state_clock/configs/local_joint_payload_protection_v1.json",
    "docs/local_joint_payload_protection_v1.md",
    "scripts/build_local_joint_payload_protection_v1_notebook.py",
]

SETUP = r'''
import subprocess, sys

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
            "experiments.wan_state_clock.local_joint_payload_protection_v1_run"],
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
                print(f"Completed core units: {returned}/{len(ledger)}; active: {active}")
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
'''


def build(root=ROOT):
    root=Path(root); cells=[]
    def md(value):
        cells.append(dict(cell_type="markdown",metadata={},source=value.splitlines(keepends=True)))
    def code(value):
        ast.parse(value)
        cells.append(dict(cell_type="code",metadata={},source=value.splitlines(keepends=True),execution_count=None,outputs=[]))
    runtime=(root/f"runtime/wan/{STEM}.py").read_text()
    helper_node=next(n for n in ast.parse(runtime).body if isinstance(n,ast.FunctionDef) and n.name=="execution_artifacts")
    helper=ast.get_source_segment(runtime,helper_node)
    md("""# B Card51：固定局部载荷保护 · 三臂用户运行包

本包只完成CPU/静态验证，尚无新真实媒体结果。按固定方案运行P、冻结J0、唯一J_ECC；
每窗systematic SECDED(39,32,4)，净32位，原54方向保持并追加7parity，所有gamma不提高。
单错纠正/双错检测只适用于正确状态且无擦除；3错可误纠、4错可未检出。校验一致不是存在性证明。
仅terminal/MP4潜变量两端：VAE load1、D3/E3；69核心槽，另304 CPU槽。
12表、264状态、10296原始符号观测槽（含未编码/错钥对照）、8448 ECC后局部消息位。
原raw软聚合与ECC聚合分别保留，不按结果挑分支。三对画质仍保存preclip/float/RGB8/MP4。

首次运行默认自动下载同分支的普通源码ZIP，无需手动上传。
直接Run all会从现有dev分支自动下载源码；真实模型、媒体和GPU验证由用户运行。
输入为卡47同次features/PAYLOAD_ONLY/terminal.pt及features/PROJECTION_MARGIN/terminal.pt。
输出为 `MyDrive/Video-WM/B-Local-Payload-Protection-V1/fixed-clean-triple-v1`，需要约2.5GiB RGB保存空间另加记录。
然后直接Run all。无GPU型号硬门槛，CPU可能很慢。源不可读或实际调用失败保留，不换源、不重试。
已有科学记录时全程只读展示；未完成尝试不自动再跑。此包不验证多步生成期或真实编辑同步。
""")
    code("from google.colab import drive\ndrive.mount('/content/drive')\n")
    md("### 加载可用源码\n先检查固定输出；已有结果时跳过源码与安装。SOURCE_LOCATION直接接受源码目录、ZIP或普通URL。")
    code('''from pathlib import Path
import json
OUTPUT=Path("/content/drive/MyDrive/Video-WM/B-Local-Payload-Protection-V1/fixed-clean-triple-v1")
'''+helper+'''
EXISTING_ARTIFACTS=execution_artifacts(OUTPUT)
if EXISTING_ARTIFACTS:
    print("Existing scientific records: read only",EXISTING_ARTIFACTS)
else:
    import stat,urllib.request,zipfile
    SOURCE_LOCATION="https://raw.githubusercontent.com/RICHAAARC/SC-SSTW/dev/local-joint-state-payload-v1/notebooks/local_joint_payload_protection_v1_portable_source.zip"
    supplied=Path(SOURCE_LOCATION)
    if supplied.is_dir():
        SOURCE_ROOT=supplied
    else:
        archive=supplied
        if SOURCE_LOCATION.startswith(("https://","http://")):
            archive=Path("/content/b_local_payload_protection_v1_source.zip")
            urllib.request.urlretrieve(SOURCE_LOCATION,archive)
        SOURCE_ROOT=Path("/content/b_local_payload_protection_v1_source")
        SOURCE_ROOT.mkdir(parents=True,exist_ok=True)
        with zipfile.ZipFile(archive) as package:
            for member in package.infolist():
                target=(SOURCE_ROOT/member.filename).resolve()
                if not target.is_relative_to(SOURCE_ROOT.resolve()) or stat.S_ISLNK(member.external_attr>>16):
                    raise ValueError("Unsafe ZIP path: "+member.filename)
            package.extractall(SOURCE_ROOT)
    OUTPUT.mkdir(parents=True,exist_ok=True)
    print("Source:",SOURCE_ROOT)
''')
    md("""### 准备依赖并执行固定一次
采用已跑通的Wan VAE/cache和媒体适配；只对实际缺包或已知可选torchao导入冲突定向修复。
不要求版本相等、摘要、manifest或Git状态。模型选择后不换位置重试。
两份保存终态先检查真实格式/FP32；冻结J0不重建。运行中保存core和CPU调用账与逐阶段盲结果。
""")
    code(SETUP)
    md("### 展示保存结果\nEXECUTION_COMPLETE只表示调用完成；raw符号/ECC消息/两聚合/旧GROW/质量分别查看，不自动给科学PASS。质量摘要显式列出whole/outside逐帧及相邻帧RMS/Linf峰值和全部精确并列位置；失败或未执行项保持不可用。")
    code('''saved={}
for name in ("result.json","checkpoint.json","ledger.json","cpu_ledger.json","evaluation.json","attempt.json"):
    for suffix in ("",".tmp"):
        path=OUTPUT/(name+suffix)
        if path.is_file():
            try:
                saved[name+suffix]=json.loads(path.read_text())
            except (OSError,ValueError) as exc:
                print("Unreadable record",path.name,type(exc).__name__,str(exc))
result=next((saved[n] for n in ("result.json","checkpoint.json","result.json.tmp","checkpoint.json.tmp") if isinstance(saved.get(n),dict)),{})
print("Status",result.get("status","NO_READABLE_CHECKPOINT"))
print("Fixed denominators",result.get("fixed_denominator"))
print("Core calls",result.get("counts"))
print("CPU calls",result.get("cpu_counts"))
print("CPU arithmetic",result.get("cpu_arithmetic"))
for slot,row in result.get("evaluation",{}).items():
    local=row["local"]
    names=("state","raw_symbols","raw_data","raw_parity","ecc_data","raw_aggregate","ecc_aggregate")
    print(slot,{n:{k:local[n][k] for k in ("denominator","correct","errors","erasures","unavailable")} for n in names})
    print("ECC outcomes",local["decoder_status_counts"],"miscorrections",local["miscorrections"],"undetected",local["undetected_wrong_codewords"])
    print("Legacy",{k:v for k,v in row["legacy"].items() if k not in ("rows","votes")})
for name,row in result.get("latent_difference",{}).items():
    print("Latent difference",name,{k:v for k,v in row.items() if k!="times"})
for name,row in result.get("quality",{}).items():
    print("Quality",name,{k:v for k,v in row.items() if k not in ("frames","adjacent","peaks")})
    print("Quality peaks",name,row.get("peaks",{"status":"UNAVAILABLE","reason":"NO_SAVED_PEAK_REDUCTION"}))
for name in ("ledger.json","cpu_ledger.json"):
    for slot,row in saved.get(name,{}).items():
        if row.get("status")!="RETURNED": print("Retained slot",name,slot,row)
if not (OUTPUT/"result.json").exists(): print("Incomplete/active attempt retained; no automatic replay.")
for arm in ("P","J0","J_ECC"):
    path=OUTPUT/"media"/(arm+".mp4")
    if path.is_file():
        try:
            from IPython.display import display,Video
            print(arm);display(Video(str(path),embed=True,width=640))
        except Exception as exc:
            print("Preview unavailable",type(exc).__name__,str(exc))
print("Return complete output directory for independent audit:",OUTPUT)
''')
    for i,c in enumerate(cells):c["id"]=f"payload-protection-{i:02d}"
    notebook=dict(nbformat=4,nbformat_minor=5,cells=cells,
        metadata=dict(accelerator="GPU",kernelspec=dict(display_name="Python 3",language="python",name="python3"),language_info=dict(name="python")))
    target=root/"notebooks";target.mkdir(parents=True,exist_ok=True)
    path=target/f"{STEM}.ipynb"
    path.write_text(json.dumps(notebook,ensure_ascii=False,indent=1)+"\n")
    archive=target/f"{STEM}_portable_source.zip"
    with zipfile.ZipFile(archive,"w",compression=zipfile.ZIP_DEFLATED) as bundle:
        for relative in SOURCE_FILES:bundle.write(root/relative,relative)
    return path,archive


if __name__=="__main__":
    for path in build():print(path)
