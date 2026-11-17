#!/usr/bin/env python3
"""Direct-convolution RTL stimuli with independent PyTorch outputs.

One unchanged engine receives changing kernel, stride, padding, channel and
activation descriptors. No generated hardware or production host code here.
"""
import argparse
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output',type=Path)
    a=parser.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    torch.manual_seed(3301)
    cases=[];cursor=64
    specs=[(5,7,3,5,3,3,1,1,1,1,1,1,1,1,1),
           (4,6,2,3,1,1,2,2,0,0,0,0,1,1,0),
           (4,11,5,9,3,1,1,2,2,0,0,0,1,1,1),
           (3,5,65,3,1,1,1,1,0,0,0,0,1,1,0),
           (7,9,2,5,3,3,1,1,2,2,2,2,2,2,0)]
    for ih,iw,ic,oc,kh,kw,sh,sw,pt,pb,pl,pr,dh,dw,relu in specs:
        x=torch.randn(1,ic,ih,iw)*0.125
        w=torch.randn(oc,ic,kh,kw)*0.125
        bias=torch.randn(oc)*0.125
        y=F.conv2d(F.pad(x,(pl,pr,pt,pb)),w,bias,stride=(sh,sw),dilation=(dh,dw))
        if relu:y=torch.relu(y)
        tensors=[x.permute(0,2,3,1).contiguous().numpy(),
                 w.permute(0,2,3,1).contiguous().numpy(),bias.numpy(),
                 y.permute(0,2,3,1).contiguous().numpy()]
        info=[]
        for t in tensors:
            cols=t.shape[-1];rows=t.size//cols;stride=(cols+1)//2*2
            info.append((cursor,rows,cols,stride,t.reshape(rows,cols)))
            cursor+=rows*stride
        oh,ow=y.shape[2:]
        dims=[1,ih,iw,ic,oc,kh,kw,oh,ow,sh,sw,pt,pl,dh,dw,relu]
        cases.append((dims,info))
    memory=np.full(cursor,0x5a5a5a5a,dtype=np.uint32)
    for dims,info in cases:
        for offset,rows,cols,stride,data in info[:3]:
            for r in range(rows):memory[offset+r*stride:offset+r*stride+cols]=data[r].view(np.uint32)
    def write(name,values):
        (a.output/name).write_text(''.join(f'{int(v):08x}\n' for v in values))
    for i,(dims,info) in enumerate(cases):
        header=np.zeros(64,dtype=np.uint32);header[0]=0x434f4e56;header[2]=3
        for f,v in enumerate(dims,2):header[2*f]=v
        for f,item in enumerate(info,18):header[2*f]=item[0]*4
        header[50]=0x3f800000 if dims[-1] else 0;header[62]=64
        memory[:64]=header;write(f'initial-{i}.hex',memory)
        mask=np.zeros(cursor,dtype=np.uint32)
        offset,rows,cols,stride,data=info[-1]
        for r in range(rows):
            begin=offset+r*stride
            memory[begin:begin+cols]=data[r].view(np.uint32);mask[begin:begin+cols]=2
        write(f'expected-{i}.hex',memory);write(f'mask-{i}.hex',mask)
    (a.output/'parameters.svh').write_text(f'localparam WORDS={cursor}, HEADER_WORDS=64, INVOCATIONS={len(cases)};\n')


if __name__=='__main__':main()
