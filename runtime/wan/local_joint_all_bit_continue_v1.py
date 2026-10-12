"""Fixed continuation entry; shared All-Bit execution/checkpoint path, actual tensor updates."""
from main.tube_state import local_joint_all_bit_continue_v1 as method
from runtime.wan import local_joint_all_bit_v1 as engine

EXPECTED = engine.EXPECTED
CEILING = ("One seen saved ALLBIT_TWO terminal, two fresh .10 scalar steps in the original BASE ball. "
           "All 32 margin terms retained; lower loss does not guarantee bit coverage, state floor or quality. "
           "No old .5 steps, old HALF gradient reuse, scan, rollback, third step, cap increase, codec, "
           "blind decoding, FPR, new source or M1/M2. No scientific PASS.")


def initial_result(config):
    return engine.initial_result(config,method=method,ceiling=CEILING)


def refresh_reports(output,result):
    return engine.refresh_reports(output,result,method=method)


def finalize_interrupted(output,reason):
    return engine.finalize_interrupted(output,reason,method=method)


def model_work_remaining(output):
    return engine.model_work_remaining(output,method=method)


def reports_complete(output):
    return engine.reports_complete(output,method=method)


def run(config,output,**kwargs):
    return engine.run(config,output,method=method,continuation=True,ceiling=CEILING,**kwargs)
