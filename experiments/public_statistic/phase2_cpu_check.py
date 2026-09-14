"""Explicit CPU-only engineering check; small networks, real installed UniPC."""
import argparse,json,sys,importlib.util,unittest,io,time
from pathlib import Path

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',required=True);args=p.parse_args();out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
 import torch,diffusers
 if torch.cuda.is_available():raise RuntimeError('run with CUDA_VISIBLE_DEVICES empty; CPU only')
 torch.set_num_threads(1);root=Path(__file__).resolve().parents[2];file=root/'tests/test_phase2_terminal_adapter.py';spec=importlib.util.spec_from_file_location('phase2_checks',file);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
 start=time.monotonic();log=io.StringIO();result=unittest.TextTestRunner(stream=log,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(module));(out/'tests.log').write_text(log.getvalue())
 from main.sc_sstw.terminal_feedback import clone_graph_state
 from runtime.public_statistic.phase2 import controlled_run
 a,z,state=module.setup()
 with torch.no_grad():off=a.rollout(z,clone_graph_state(state,torch))
 a.counts={k:0 for k in a.counts};run=controlled_run(a,z,state,module.config(),off1_terminal_rgb=off,enabled=True)
 report=dict(torch=torch.__version__,diffusers=diffusers.__version__,scheduler='real UniPCMultistepScheduler',networks='small differentiable substitutes, no Wan weights',tests_run=result.testsRun,tests_success=result.wasSuccessful(),errors=len(result.errors),failures=len(result.failures),counts=a.counts,feedback=run['feedback'],actual_configuration=run['configuration'],terminal_target_mse=float((run['q']-run['target']).square().mean()),fixed_reference_preserved=bool(torch.equal(off,run['reference_rgb'])),elapsed_seconds=time.monotonic()-start,media=0,model_downloads=0,gpu_runs=0,science_denominator=0)
 (out/'result.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
 for name in ('main/sc_sstw/terminal_feedback.py','main/sc_sstw/public_luma_statistic.py','runtime/public_statistic/phase2.py','tests/test_phase2_terminal_adapter.py','experiments/public_statistic/phase2_cpu_check.py'):
  dest=out/'source_snapshot'/name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes((root/name).read_bytes())
 print(json.dumps(report,indent=2));raise SystemExit(0 if result.wasSuccessful() else 1)
if __name__=='__main__':main()
