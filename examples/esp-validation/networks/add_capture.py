#!/usr/bin/env python3
"""Independent packed-ABI RTL fixtures for reusable pointwise addition.

Tests odd rows, changing descriptors, a tile boundary, activation and IEEE
special values. This generates verification stimuli only, never hardware/host.
"""
import argparse
from pathlib import Path
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('output',type=Path)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    rng=np.random.default_rng(773)
    cursor=64;cases=[]
    for rows,cols,relu in ((3,7,0),(2,9,1),(1,4101,0),(1,7,1)):
        stride=(cols+1)//2*2
        x=rng.normal(size=(rows,cols)).astype(np.float32)
        y=rng.normal(size=(rows,cols)).astype(np.float32)
        if len(cases)==3:
            x[0]=[np.nan,np.inf,-np.inf,-0.,0.,-2.,3.]
            y[0]=[1.,2.,3.,-0.,0.,1.,-1.]
        with np.errstate(invalid='ignore'):
            z=x+y
            if relu:z=np.maximum(np.minimum(z,np.finfo(np.float32).max),np.float32(0))
        offsets=[cursor+i*rows*stride for i in range(3)]
        cursor+=3*rows*stride
        cases.append((rows,cols,stride,relu,x,y,z,offsets))
    memory=np.full(cursor,0x5a5a5a5a,dtype=np.uint32)
    for rows,cols,stride,relu,x,y,z,offsets in cases:
        for data,offset in zip((x,y),offsets):
            for r in range(rows):memory[offset+r*stride:offset+r*stride+cols]=data[r].view(np.uint32)
        # Full-row stores own the beat padding under the packed ABI. Keep
        # guards outside the tensor, and initialize owned padding to zero.
        for r in range(rows):memory[offsets[2]+r*stride+cols:offsets[2]+(r+1)*stride]=0
    write=lambda path,v:path.write_text(''.join(f'{int(x):08x}\n' for x in v))
    for i,(rows,cols,stride,relu,x,y,z,offsets) in enumerate(cases):
        header=np.zeros(64,dtype=np.uint32)
        fields={0:0x41444431,1:3,2:rows*cols,3:offsets[0]*4,4:offsets[1]*4,
                5:offsets[2]*4,6:relu,7:cols,8:rows,9:stride*4,
                10:0x3f800000 if relu else 0,31:64}
        for f,v in fields.items():header[2*f]=v
        memory[:64]=header
        write(a.output/f'initial-{i}.hex',memory)
        mask=np.zeros(cursor,dtype=np.uint32)
        for r in range(rows):
            begin=offsets[2]+r*stride
            memory[begin:begin+cols]=z[r].view(np.uint32);mask[begin:begin+cols]=2
        write(a.output/f'expected-{i}.hex',memory)
        write(a.output/f'mask-{i}.hex',mask)
    (a.output/'parameters.svh').write_text(
        f'localparam WORDS={cursor}, HEADER_WORDS=64, INVOCATIONS={len(cases)};\n')


if __name__=='__main__':main()
