"""CPU fake components, but unchanged pinned upstream writer/reader bodies."""
from __future__ import annotations
import copy
import dataclasses
import json
import os
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from PIL import Image
from experiments.image_trajectory_reference import official_bridge as bridge
from experiments.image_trajectory_reference import run

pytestmark=pytest.mark.unit
UPSTREAM=Path(os.environ.get("GROW_REFERENCE_SOURCE_ROOT",
    "/tmp/image-trajectory-reference-a1-official-20260929/grow-tree/luopengchen-GROW-6aa69a9"))


@pytest.fixture(scope="session")
def official_source():
    if not UPSTREAM.is_dir():
        pytest.skip("Pinned upstream source absent; set GROW_REFERENCE_SOURCE_ROOT to its verified extraction")
    bridge.verify_source(UPSTREAM)
    return UPSTREAM


@pytest.fixture(autouse=True)
def one_cpu_thread():
    import torch
    old=torch.get_num_threads();torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def fake_pipeline():
    import torch
    from diffusers import DDIMScheduler
    class UNet(torch.nn.Module):
        config=SimpleNamespace(in_channels=4)
        def forward(self,x,t,encoder_hidden_states):
            return SimpleNamespace(sample=.05*x)
    class Encoder(torch.nn.Module):
        def forward(self,x):
            pooled=torch.nn.functional.avg_pool2d(x,8)
            return torch.cat((pooled,pooled.mean(1,keepdim=True)),1)
    class Decoder(torch.nn.Module):
        def forward(self,x):
            return torch.nn.functional.interpolate(x[:,:3].tanh(),scale_factor=8,mode="nearest")
    class VAE:
        config=SimpleNamespace(scaling_factor=.18215)
        encoder=Encoder();decoder=Decoder()
        def encode(self,x):return SimpleNamespace(latent_dist=SimpleNamespace(mean=self.encoder(x)))
        def decode(self,x):return SimpleNamespace(sample=self.decoder(x))
    class Tokenizer:
        model_max_length=77
        def __call__(self,*_,**__):return SimpleNamespace(input_ids=torch.zeros((1,77),dtype=torch.long))
    return SimpleNamespace(unet=UNet(),vae=VAE(),tokenizer=Tokenizer(),
        text_encoder=lambda ids:(torch.zeros((1,77,8)),),
        scheduler=DDIMScheduler(beta_start=.00085,beta_end=.012,beta_schedule="scaled_linear",
                                clip_sample=False,set_alpha_to_one=False))


@pytest.mark.usefixtures("official_source")
def test_exact_official_defaults_layout_and_truth_free_reader(tmp_path):
    cfg=run.load_config(UPSTREAM)
    assert cfg["official_config"]["guidance_scale"]==200 and cfg["public_payload_bits"]==32
    receipt=bridge.layout_receipt("watermark","watermark-wrong")
    assert receipt["correct"]["seed"]==974
    assert receipt["correct"]["bit_assignment_sha256"]!=receipt["wrong"]["bit_assignment_sha256"]
    with pytest.raises(ValueError,match="collides"):
        bridge.layout_receipt("watermark","kramretaw")
    image_path=tmp_path/"input.png"
    Image.fromarray(np.random.default_rng(42).integers(0,256,(512,512,3),dtype=np.uint8)).save(image_path)
    public=bridge.PublicReaderConfig(device="cpu",dtype="float32",secret_key="watermark")
    assert not hasattr(public,"message")
    result=bridge.read_saved_png(image_path,public,fake_pipeline(),UPSTREAM)
    assert result["status"]=="READ" and len(result["decoded_bits"])==32
    assert result["truth_used"] is False


@pytest.mark.parametrize("device",["cpu",None])
@pytest.mark.usefixtures("official_source")
def test_actual_official_writer_saved_png_read_bits_then_evaluate(tmp_path,device):
    def factory(config,record):
        record(dict(status="CPU_FAKE",revision=None));return fake_pipeline()
    result=run.run_fixed(tmp_path/"run",UPSTREAM,pipeline_factory=factory,device=device)
    assert result["runtime_device"]=="cpu"
    assert result["status"]=="FIXED_REFERENCE_COMPLETE", result["failures"]
    assert result["counts"]==dict(saved_images=2,readouts=4,evaluated=8)
    assert result["model_calls_executed"] and not result["model_loaded"] and not result["real_model_executed"]
    assert result["calls"]["GENERATE_OFF"]["unet"]==dict(attempted=50,completed=50)
    assert result["calls"]["GENERATE_GROW"]["unet"]==dict(attempted=50,completed=50)
    assert sum(row.get("vae_encoder",{}).get("completed",0) for row in result["calls"].values())==4
    assert sum(row.get("vae_decoder",{}).get("completed",0) for row in result["calls"].values())==2
    blind=tmp_path/"run/blind_readouts.json"
    assert bridge.sha256(blind)==result["blind_readouts_sha256_before_evaluation"]
    content=json.loads(blind.read_text())
    assert not content["evaluation_performed"] and content["no_truth_inputs"]
    for identity,row in result["reads"].items():
        arm=identity.split("/")[0]
        assert row["png_sha256"]==result["images"][arm]["sha256"]
    before=blind.read_bytes()
    config=run.load_config(UPSTREAM)
    _,_,codec=bridge.load_official(UPSTREAM)
    changed=copy.deepcopy(config);changed["wrong_message"]="ABCD"
    run.evaluate_readouts(blind,changed,codec)
    assert blind.read_bytes()==before
    assert all(type(row["exact_bits"]) is bool for row in result["evaluations"].values())


@pytest.mark.usefixtures("official_source")
def test_png_reopen_not_memory_object(tmp_path):
    path=tmp_path/"roundtrip.png"
    image=Image.new("RGB",(512,512),(10,20,30));image.save(path)
    image.paste((255,255,255),(0,0,512,512))
    public=bridge.PublicReaderConfig(device="cpu",dtype="float32",secret_key="watermark")
    first=bridge.read_saved_png(path,public,fake_pipeline(),UPSTREAM)
    image.save(path)
    second=bridge.read_saved_png(path,public,fake_pipeline(),UPSTREAM)
    assert first["png_sha256"]!=second["png_sha256"]


@pytest.mark.usefixtures("official_source")
def test_asset_missing_keeps_all_fixed_rows(tmp_path):
    def unavailable(config,record):
        record(dict(status="ASSET_ACCESS_REQUIRED",http_status=401,stage="MODEL_INFO"))
        raise RuntimeError("original checkpoint unavailable")
    result=run.run_fixed(tmp_path/"missing",UPSTREAM,pipeline_factory=unavailable,device="cpu")
    assert result["status"]=="ASSET_ACCESS_REQUIRED"
    assert result["counts"]==dict(saved_images=0,readouts=0,evaluated=0)
    assert [len(result[group]) for group in ("images","reads","evaluations")]==[2,4,8]
    assert all(row["status"]=="NOT_COMPLETED" for group in ("images","reads","evaluations") for row in result[group].values())
    assert not result["model_calls_executed"]


@pytest.mark.parametrize("failure,asset_status,top_status", [
    (401,"ASSET_ACCESS_REQUIRED","ASSET_ACCESS_REQUIRED"),
    (403,"ASSET_ACCESS_REQUIRED","ASSET_ACCESS_REQUIRED"),
    (404,"ASSET_ACCESS_REQUIRED","ASSET_ACCESS_REQUIRED"),
    (429,"LOAD_FAILED","ENGINEERING_FAILURE"),
    (503,"LOAD_FAILED","ENGINEERING_FAILURE"),
    ("timeout","LOAD_FAILED","ENGINEERING_FAILURE"),
])
@pytest.mark.usefixtures("official_source")
def test_component_load_failure_classification(monkeypatch,tmp_path,failure,asset_status,top_status):
    bridge.load_official(UPSTREAM)
    from diffusers import DDIMScheduler
    from huggingface_hub import HfApi
    def no_model_info(*args,**kwargs):
        raise AssertionError("live metadata must not block the pinned mirror")
    monkeypatch.setattr(HfApi,"model_info",no_model_info)
    def load(*args,**kwargs):
        if failure=="timeout": raise TimeoutError("component request timed out")
        exc=OSError(f"component HTTP {failure}")
        exc.response=SimpleNamespace(status_code=failure)
        raise exc
    monkeypatch.setattr(DDIMScheduler,"from_pretrained",load)
    result=run.run_fixed(tmp_path/str(failure),UPSTREAM)
    assert result["status"]==top_status
    assert result["assets"]["status"]==asset_status
    assert result["assets"]["stage"]=="SCHEDULER_LOAD"
    assert result["assets"]["http_status"]==(failure if isinstance(failure,int) else None)
    assert result["assets"]["error"]
    assert result["counts"]==dict(saved_images=0,readouts=0,evaluated=0)
    assert [len(result[group]) for group in ("images","reads","evaluations")]==[2,4,8]
    assert not result["model_loaded"] and not result["model_calls_executed"]
    assert all(row["status"]=="NOT_COMPLETED" for group in ("images","reads","evaluations") for row in result[group].values())


@pytest.mark.parametrize("cuda",[False,True])
@pytest.mark.usefixtures("official_source")
def test_mirror_loads_same_revision_without_metadata_or_cuda_gate(monkeypatch,cuda):
    bridge.load_official(UPSTREAM)
    import torch
    from diffusers import DDIMScheduler,StableDiffusionPipeline
    from huggingface_hub import HfApi
    monkeypatch.setattr(HfApi,"model_info",lambda *a,**k:pytest.fail("unnecessary metadata lookup"))
    monkeypatch.setattr(torch.cuda,"is_available",lambda:cuda)
    monkeypatch.setattr(torch.cuda,"get_device_name",lambda _:"mock GPU")
    calls=[]
    scheduler=SimpleNamespace(config={"prediction_type":"epsilon"})
    def scheduler_load(repo,**kwargs):
        calls.append((repo,kwargs));return scheduler
    class MockPipe:
        def __init__(self): self.scheduler=scheduler
        def to(self,device): self.device=device;return self
    def pipe_load(repo,**kwargs):
        calls.append((repo,kwargs));return MockPipe()
    monkeypatch.setattr(DDIMScheduler,"from_pretrained",scheduler_load)
    monkeypatch.setattr(StableDiffusionPipeline,"from_pretrained",pipe_load)
    cfg=run.load_config(UPSTREAM);receipts=[]
    pipe=run.load_real_pipeline(cfg,receipts.append)
    assert pipe.device==("cuda" if cuda else "cpu")
    assert all(repo=="sd2-community/stable-diffusion-2-1-base" for repo,_ in calls)
    assert all(k["revision"]==cfg["model_asset"]["revision"] for _,k in calls)
    assert calls[1][1]["use_safetensors"] is True
    assert calls[1][1]["torch_dtype"] is torch.float32
    assert receipts[-1]["status"]=="LOADED"
    assert receipts[-1]["asset_source_changed"] is True


@pytest.mark.parametrize("working_torch",[True,False])
def test_notebook_reuses_runtime_and_dependency_conflicts_do_not_stop(monkeypatch,tmp_path,working_torch):
    from scripts.build_image_trajectory_reference_notebook import build
    nb=json.loads(build(source_sha="a"*40,output=tmp_path/"test.ipynb").read_text())
    commands=[]
    def logged(command,stage,**kwargs):
        assert "venv" not in command and "ensurepip" not in command
        commands.append((command,stage,kwargs))
        if stage=="TORCH_PROBE": return 0 if working_torch else 1
        if stage=="DEPENDENCY_REPORT": return 1  # unrelated base-environment conflicts
        return 0
    namespace=dict(sys=SimpleNamespace(executable="/current/python",version_info=(3,13,0)),
        subprocess=SimpleNamespace(check_output=lambda *a,**k:"recorded versions"),
        PYTHON="/stale/failed-venv/bin/python",REPO=tmp_path,
        OUTPUT=tmp_path,setup={},logged=logged,
        failed=lambda *a:pytest.fail("environment unnecessarily blocked"))
    exec(compile("".join(nb["cells"][4]["source"]),"environment-cell","exec"),namespace)
    assert namespace["PYTHON"]=="/current/python"
    assert all(command[0]=="/current/python" for command,_,_ in commands)
    assert namespace["setup"]["python_executable"]=="/current/python"
    setup_source="".join(nb["cells"][2]["source"])
    assert "PYTHON = sys.executable" in setup_source and "VENV" not in setup_source
    assert any(stage=="TORCH_REPAIR" for _,stage,_ in commands)==(not working_torch)
    assert namespace["setup"]["dependency_check_returncode"]==1
    assert (tmp_path/"environment_freeze.txt").is_file()


def test_wrapped_resource_errors_are_not_confused_with_oom():
    class ResourceError(Exception):
        response=SimpleNamespace(status_code=403)
    wrapped=OSError("component unavailable");wrapped.__cause__=ResourceError()
    assert run.asset_access_error(wrapped)
    assert run.asset_access_error(FileNotFoundError("checkpoint"))
    assert not run.asset_access_error(RuntimeError("CUDA out of memory"))


@pytest.mark.usefixtures("official_source")
def test_bit_evaluation_retains_invalid_utf8_raw_bits(tmp_path):
    config=run.load_config(UPSTREAM);_,_,codec=bridge.load_official(UPSTREAM)
    rows={f"{arm}/{key}":dict(status="READ",image_id=arm,key_id=key,decoded_bits=[1]*32)
          for arm in run.ARMS for key in run.KEYS}
    path=tmp_path/"blind.json";run.atomic_json(path,dict(reads=rows))
    assert codec.bits_to_message([1]*32)==""
    scores=run.evaluate_readouts(path,config,codec)
    assert len(scores)==8 and all(row["exact_bits"] is False for row in scores.values())


def test_unpublished_notebook_retains_setup_denominator_before_failure(tmp_path):
    import ast
    from scripts.build_image_trajectory_reference_notebook import build
    notebook_path=build(output=tmp_path/"draft.ipynb")
    notebook=json.loads(notebook_path.read_text())
    assert "".join(notebook["cells"][0]["source"])=="from google.colab import drive\ndrive.mount('/content/drive')\n"
    for cell in notebook["cells"]:
        if cell["cell_type"]=="code":
            ast.parse("".join(cell["source"]))
            assert cell["outputs"]==[] and cell["execution_count"] is None
    setup="".join(notebook["cells"][2]["source"])
    setup=setup.replace("Path('/content/drive/MyDrive/Video-WM/Image-Trajectory-Reference-V1')",f"Path({str(tmp_path)!r})")
    with pytest.raises(RuntimeError,match="Unpublished draft"):
        exec(compile(setup,"notebook-setup","exec"),{})
    receipts=list(tmp_path.glob("*/setup_receipt.json"))
    assert len(receipts)==1
    receipt=json.loads(receipts[0].read_text())
    assert receipt["status"]=="SETUP_FAILED"
    assert [len(receipt[group]) for group in ("images","reads","evaluations")]==[2,4,8]
