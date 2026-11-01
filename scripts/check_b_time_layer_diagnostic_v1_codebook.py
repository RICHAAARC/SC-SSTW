"""One delivery-time CPU check; not imported by the user diagnostic or its67 budget."""
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

def main():
    import numpy as np
    from main.tube_state.b_framewise_qim_v1 import encode
    from main.tube_state.b_framewise_qim_time_recovery_v1 import words,DOMAIN,BLOCK
    start=time.monotonic();mismatches=[];count=0
    for lo in range(0,DOMAIN,BLOCK):
        js=np.arange(lo,min(lo+BLOCK,DOMAIN),dtype=np.uint64);reader=words(js)
        for j,code in zip(js,reader):
            bits=encode(int(j));packed=sum(bit<<(62-k) for k,bit in enumerate(bits))
            if packed!=int(code):mismatches.append(int(j))
            count+=1
    result=dict(checked=count,domain=DOMAIN,bit_order='MSB first 62..0',mismatches=len(mismatches),examples=mismatches[:8],wall_seconds=time.monotonic()-start,scope='delivery engineering only; not user item68')
    print(result)
    if mismatches:raise SystemExit(1)
if __name__=='__main__':main()
