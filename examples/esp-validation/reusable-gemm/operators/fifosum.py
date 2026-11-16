#!/usr/bin/env python3
import sys,json,collections
log,inv=sys.argv[1],int(sys.argv[2]);probe=json.load(open(sys.argv[3]))
role={f['id']:(f.get('writer'),f.get('reader'),f['depth']) for f in probe['fifos']}
ev=[l.split()[1:] for l in open(log) if l.startswith('EV ')]
st=[i for i,e in enumerate(ev) if e[1:]==['TOP','start']];s=st[inv];e=st[inv+1] if inv+1<len(st) else len(ev);t0=int(ev[s][0])
d=collections.defaultdict(list)
for c,*r in ev[s:e]:
    c=int(c)-t0
    if r[0]=='FIFO':d[(r[1],r[2])].append(c)
    elif r[0] in('RD','WR'):d[(r[0],'')].append(c)
    else:d[(' '.join(r),'')].append(c)
for k,v in sorted(d.items(),key=lambda kv:kv[1][0]):
    w=role.get(k[0],('','',''))
    print(f"{k[0]:<16}{k[1]:<2} n={len(v):<5} first={v[0]:<7} last={v[-1]:<7} {'' if not w[0] and w[0]!=0 else f'node{w[0]}->node{w[1]} depth{w[2]}'}")
