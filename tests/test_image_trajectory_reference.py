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


@pytest.mark.usefixtures("official_source")
def test_actual_official_writer_saved_png_read_bits_then_evaluate(tmp_path):
    def factory(config,record):
        record(dict(status="CPU_FAKE",revision=None));return fake_pipeline()
    result=run.run_fixed(tmp_path/"run",UPSTREAM,pipeline_factory=factory,device="cpu")
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
    (429,"RESOLVE_FAILED","ENGINEERING_FAILURE"),
    (503,"RESOLVE_FAILED","ENGINEERING_FAILURE"),
    ("timeout","RESOLVE_FAILED","ENGINEERING_FAILURE"),
    ("invalid_revision","RESOLVE_FAILED","ENGINEERING_FAILURE"),
])
@pytest.mark.usefixtures("official_source")
def test_model_metadata_failure_classification(monkeypatch,tmp_path,failure,asset_status,top_status):
    # Diffusers may probe CUDA availability while importing dependencies. Load
    # them before intercepting the later, real pipeline CUDA admission check.
    bridge.load_official(UPSTREAM)
    from huggingface_hub import HfApi
    import torch
    def model_info(_self,_model_id):
        if failure=="invalid_revision":
            return SimpleNamespace(sha="not-an-immutable-revision")
        if failure=="timeout":
            raise TimeoutError("metadata request timed out")
        exc=OSError(f"metadata HTTP {failure}")
        exc.response=SimpleNamespace(status_code=failure)
        raise exc
    def no_cuda_probe():
        raise AssertionError("metadata failure must stop before the real pipeline CUDA check")
    monkeypatch.setattr(HfApi,"model_info",model_info)
    monkeypatch.setattr(torch.cuda,"is_available",no_cuda_probe)
    result=run.run_fixed(tmp_path/str(failure),UPSTREAM)
    assert result["status"]==top_status
    assert result["assets"]["status"]==asset_status
    assert result["assets"]["stage"]=="MODEL_INFO"
    assert result["assets"]["http_status"]==(failure if isinstance(failure,int) else None)
    assert result["assets"]["error"]
    assert result["counts"]==dict(saved_images=0,readouts=0,evaluated=0)
    assert [len(result[group]) for group in ("images","reads","evaluations")]==[2,4,8]
    assert not result["model_loaded"] and not result["model_calls_executed"]
    assert all(row["status"]=="NOT_COMPLETED" for group in ("images","reads","evaluations") for row in result[group].values())


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
