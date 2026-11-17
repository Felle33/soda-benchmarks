#!/usr/bin/env python3
"""Phases of single-output-tile invocations from a probe log: load (last
load->compute token written), matmul (A/B tokens freed), epilogue (compute->
store token), store (TOP done). Also M, N, K from the descriptor stream."""
import sys,json,collections
log,probe=sys.argv[1],json.load(open(sys.argv[2]))
fif={f['id']:f for f in probe['fifos']}
ev=[l.split()[1:] for l in open(log) if l.startswith('EV ')]
st=[i for i,e in enumerate(ev) if e[1:]==['TOP','start']]
# the load node feeds compute (node1) through depth-1 tokens; compute feeds the store node.
store=[f['id'] for f in probe['fifos'] if f.get('writer')==1 and f['depth']==1]
for k,s in enumerate(st):
    e=st[k+1] if k+1<len(st) else len(ev);t0=int(ev[s][0])
    w=collections.defaultdict(list);r=collections.defaultdict(list);desc=[];done=None
    for c,*x in ev[s:e]:
        c=int(c)-t0
        if x[0]=='FIFO':
            f=fif.get(x[1],{})
            if x[2]=='w':w[x[1]].append(c)
            else:r[x[1]].append(c)
            if x[2]=='w' and f.get('writer')==0 and f.get('reader')==1 and f['depth']==16:desc.append(int(x[3]))
        if x==['TOP','done']:done=c
    tok=[i for i,f in fif.items() if f.get('writer')==0 and f.get('reader')==1 and f['depth']==1]
    load=max(w[i][-1] for i in tok if w[i]);freed=sorted(r[i][0] for i in tok if r[i])
    out=min(w[i][0] for i in store if w[i] and fif[i].get('reader')==2)
    m,n,kk=desc[2],desc[3],desc[4]
    mm=freed[0] if len(freed)>1 else None
    print(f"inv{k} M={m} N={n} K={kk}: load={load} matmul_end={mm} (matmul {mm-load}) epilogue_end={out} (epilogue {out-mm}, {(out-mm)/(m*n):.2f}/valid elem, {(out-mm)/8192:.2f}/tile elem) store={done-out} total={done}")
