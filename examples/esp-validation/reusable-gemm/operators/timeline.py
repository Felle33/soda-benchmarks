#!/usr/bin/env python3
"""Compact timeline of invocation <i> from a probe log: runs of DMA requests
are merged; times relative to TOP start."""
import re,sys
log,inv=sys.argv[1],int(sys.argv[2])
ev=[l.split()[1:] for l in open(log) if l.startswith('EV ')]
starts=[i for i,e in enumerate(ev) if e[1:]==['TOP','start']]
s=starts[inv];e=starts[inv+1] if inv+1<len(starts) else len(ev)
t0=int(ev[s][0]);run=None
def flush():
    global run
    if run: print(f"{run[1]-t0:>7}-{run[2]-t0:<7} {run[0]} x{run[3]} words={run[4]}");run=None
for c,*rest in ev[s:e]:
    c=int(c)
    if rest[0] in('RD','WR'):
        if run and run[0]==rest[0]: run[2]=c;run[3]+=1;run[4]+=int(rest[2])
        else: flush();run=[rest[0],c,c,1,int(rest[2])]
        continue
    flush();print(f"{c-t0:>7}         {' '.join(rest)}")
flush()
