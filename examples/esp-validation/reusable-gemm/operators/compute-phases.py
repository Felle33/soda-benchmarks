#!/usr/bin/env python3
"""Compute node's time per invocation, split into: waiting for A/B tiles,
matmul (K steps), waiting for C, epilogue, waiting for the output buffer.
Tokens: node0->node1 depth-1 streams (A and B: one per K step; C: one per
output tile), node1->node2 depth-1 (the output buffer)."""
import sys,json,collections
log,probe=sys.argv[1],json.load(open(sys.argv[2]))
fif={f['id']:f for f in probe['fifos']}
ev=[l.split()[1:] for l in open(log) if l.startswith('EV ')]
st=[i for i,e in enumerate(ev) if e[1:]==['TOP','start']]
tok=[i for i,f in fif.items() if f.get('writer')==0 and f.get('reader')==1 and f['depth']==1]
out=[i for i,f in fif.items() if f.get('writer')==1 and f.get('reader')==2 and f['depth']==1]
for k,s in enumerate(st):
    e=st[k+1] if k+1<len(st) else len(ev);t0=int(ev[s][0])
    W=collections.defaultdict(list);R=collections.defaultdict(list);done=0;desc=[]
    for c,*x in ev[s:e]:
        c=int(c)-t0
        if x[0]=='FIFO' and x[1] in fif:
            (W if x[2]=='w' else R)[x[1]].append(c)
            f=fif[x[1]]
            if x[2]=='w' and f.get('writer')==0 and f.get('reader')==1 and f['depth']==16:desc.append(int(x[3]))
        if x==['TOP','done']:done=c
    ctok=min(tok,key=lambda i:len(W[i]));ab=[i for i in tok if i!=ctok]
    o=out[0]
    # walk the compute node: per K step, ready = max(A,B write), end = A/B read
    steps=list(zip(*[list(zip(W[i],R[i])) for i in ab]))
    tiles=len(W[o]);ksteps=len(steps)//tiles if tiles else 0
    t=0;acc=collections.Counter();si=0
    for tile in range(tiles):
        for q in range(ksteps):
            pair=steps[si];si+=1
            ready=max(p[0] for p in pair);end=min(p[1] for p in pair)
            acc['wait A/B']+=max(0,ready-t);acc['matmul']+=end-max(ready,t);t=end
        cready=W[ctok][tile];acc['wait C']+=max(0,cready-t);t=max(t,cready)
        # epilogue ends when the output token is written; it may first wait for the output buffer
        oend=W[o][tile];prev_free=R[o][tile-2] if tile>=2 else 0
        acc['epilogue']+=oend-max(t,prev_free);acc['wait out buffer']+=max(0,prev_free-t);t=oend
    acc['before first / after last (descriptor, fill, store)']=done-sum(acc.values())
    m,n,kk=desc[2],desc[3],desc[4]
    print(f"inv{k} M={m} N={n} K={kk} tiles={tiles} ksteps={ksteps} total={done}: "+', '.join(f"{a}={b}" for a,b in acc.items()))
