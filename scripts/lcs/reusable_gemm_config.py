#!/usr/bin/env python3
"""Choose shape-independent local GEMM tiles from a compute/storage budget.

Balanced output tiles reuse both operands. Reserve two slots for A/B/old-C/result
and one accumulator; banking's measured storage report remains the final gate.
"""
import argparse,json

def choose(lanes, budget):
    if lanes < 1 or lanes > 256 or budget <= 0:
        raise ValueError('positive storage and 1..256 compute lanes required')
    factors=[(r,lanes//r) for r in range(1,lanes+1) if lanes%r==0]
    mr,nr=min(factors,key=lambda rc:(sum(rc),abs(rc[0]-rc[1])))
    candidates=[]
    for m in (8,16,32,64,128):
        for n in (8,16,32,64,128):
            for k in (8,16,32,64,128):
                size=4*(2*m*k+2*k*n+5*m*n)
                if m%mr==0 and n%nr==0 and size+4096<=budget:
                    # Maximize ideal operand reuse, then reduction burst length.
                    candidates.append(((m*n/(m+n),k,-size),(m,n,k,size)))
    if not candidates:raise ValueError('storage budget cannot service requested lanes')
    m,n,k,size=max(candidates)[1]
    return dict(tile_m=m,tile_n=n,tile_k=k,logical_bytes=size,plm_bytes=budget,
                lanes=lanes,policy='balanced_operand_reuse_then_reduction_burst',
                domain='Runtime shapes and tails independent of these local extents')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('lanes',type=int);p.add_argument('budget',type=int)
    a=p.parse_args();print(json.dumps(choose(a.lanes,a.budget),indent=2))
