import json
import random
from pathlib import Path
import numpy as np
import pytest
from runtime.wan.paper_checkpoint_v1 import DurableJournal,Boundary,BoundaryPause,UnrecoverableOperation
from runtime.wan.paper_suite_records_v1 import write,read


def setup(tmp_path):
    out=tmp_path/'out';state=out/'operations.json'
    write(state,{k:dict(status='NOT_STARTED',started=0,returned=0,kind='synthetic') for k in ('a','b')})
    return out,state,tmp_path/'cache'


def test_saved_return_pause_and_rng(tmp_path):
    out,state,cache=setup(tmp_path);calls=[];random.seed(19);np.random.seed(11)
    first=DurableJournal(out,state,cache,Boundary(max_items=1))
    a=first.call('a',lambda:(calls.append('a'),np.arange(12).reshape(3,4),random.random(),np.random.rand()))
    expected=(random.random(),np.random.rand())
    with pytest.raises(BoundaryPause):first.call('b',lambda:pytest.fail('started after boundary'))
    random.seed(999);np.random.seed(999)
    resumed=DurableJournal(out,state,cache)
    saved=resumed.call('a',lambda:pytest.fail('completed call repeated'))
    assert np.array_equal(a[1],saved[1])
    assert resumed.call('b',lambda:(random.random(),np.random.rand()))==expected
    assert calls==['a'] and all(r['started']==r['returned']==1 for r in read(state).values())


def test_consumed_without_return_never_retries(tmp_path):
    out,state,cache=setup(tmp_path);j=DurableJournal(out,state,cache)
    def interrupted():raise KeyboardInterrupt()
    with pytest.raises(KeyboardInterrupt):j.call('a',interrupted)
    with pytest.raises(UnrecoverableOperation):DurableJournal(out,state,cache).call('a',lambda:pytest.fail('repeat'))
    assert read(state)['a']['started']==1


def test_commit_before_ledger_write_is_recoverable(tmp_path):
    out,state,cache=setup(tmp_path);j=DurableJournal(out,state,cache)
    j.call('a',lambda:dict(data=np.arange(8)))
    rows=read(state);rows['a'].update(status='STARTED',returned=0);write(state,rows)
    resumed=DurableJournal(out,state,cache)
    assert resumed.call('a',lambda:pytest.fail('repeated actual completed output'))['data'].tolist()==list(range(8))
    assert read(state)['a']['recovered_saved_return']
    # Real loss is retained instead of silently recalculating.
    for file in (cache/'returns/a').glob('*.npy'):file.unlink()
    with pytest.raises(UnrecoverableOperation):resumed.call('a',lambda:pytest.fail('missing checkpoint repeated'))
