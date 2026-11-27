"""gfx938 LIT P/V pairing; mathematical dot unchanged, FP32 PV order differs."""
import numpy as np
import triton
import triton.language as tl

def physical_key(k):return (k//32)*32+((k%16)//4)*8+((k%32)//16)*4+k%4

def pack_v_paired(v):
 B,NK,H,D=v.shape;NB=(NK+127)//128
 dense=np.pad(v.transpose(0,2,3,1),((0,0),(0,0),(0,0),(0,NB*128-NK))).reshape(B,H,D,NB,128).transpose(0,1,3,2,4).copy()
 out=np.empty_like(dense);out[...,physical_key(np.arange(128))]=dense
 return out.reshape(-1)

@triton.jit
def gpu_pack_v_paired(inp,out,NK:tl.constexpr,H:tl.constexpr):
 bh=tl.program_id(2);head=bh%H;batch=bh//H
 row=tl.program_id(0)*64+tl.arange(0,64);dim=tl.program_id(1)*64+tl.arange(0,64)
 x=tl.load(inp+((batch*NK+row[:,None])*H+head)*128+dim[None,:],row[:,None]<NK,other=0)
 k=row%128;phys=(k//32)*32+((k%16)//4)*8+((k%32)//16)*4+k%4
 dst=((bh*tl.cdiv(NK,128)+row[:,None]//128)*128+dim[None,:])*128+phys[:,None]
 tl.store(out+dst,x)
