"""Offline CPU adapter. Only saved raw/old-graph reads; no media/model calls."""
from pathlib import Path
import numpy as np
from main.tube_state import b_soft_time_evidence_v1 as method
from .b_framewise_qim_time_recovery_v1 import Resources,ResourceExceeded


class Backend:
    def __init__(self):self.resources=Resources()
    def codebook(self):
        with self.resources.receiving(300):return method.codebook(self.resources.check)
    def view(self):return self.resources.receiving(600)
    def check(self):self.resources.check()
    def read_raw(self,path):
        self.check()
        with np.load(path,allow_pickle=False) as saved:
            result=dict(soft=saved['soft'],mask=saved['mask'])
        self.check();return result
    def read_old(self,path):self.check();return Path(path).read_bytes()
    def C(self,raw,old,table):
        graph=method.graph_input(old,len(raw['soft']))
        return method.costs(raw['soft'],raw['mask'],graph,table,self.check)
    def L(self,cost):self.check();return method.local_rank(cost)
    def G(self,cost,local):return method.global_rank(cost,local,self.check)
    def summary(self,cost,local,global_result):return method.row_summary(cost,local,global_result)
    def close(self):return self.resources.close()
