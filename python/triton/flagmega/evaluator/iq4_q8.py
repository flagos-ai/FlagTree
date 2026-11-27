# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Independent H003-N-v2 host semantics with explicit device-primitive oracle.

MUFU reciprocal and ex2.approx are hardware primitives, not RN division/exp.
The context must supply `iq4_q8_rcp_approx_ftz` and `iq4_q8_silu` for nonzero
quantization and GLU respectively. Missing oracle is an error, never a change
of numerical profile. These hooks consume/return binary32 NumPy values.
"""
import ctypes
from fractions import Fraction
import numpy as np
from triton.flagmega.errors import EvaluationError

_FMA=ctypes.CDLL(None).fmaf
_FMA.argtypes=(ctypes.c_float,ctypes.c_float,ctypes.c_float);_FMA.restype=ctypes.c_float
_LUT=np.array([-127,-104,-83,-65,-49,-35,-22,-10,1,13,25,38,53,69,89,113],np.int32)

def _ftz(x):
    x=np.float32(x)
    return np.copysign(np.float32(0),x) if abs(x)<np.finfo(np.float32).tiny else x

def _rz_add_half(x):
    exact=Fraction(float(x))+Fraction(1,2)
    rounded=np.float32(float(exact))
    if Fraction(float(rounded))>exact: rounded=np.nextafter(rounded,np.float32(0),dtype=np.float32)
    return _ftz(rounded)

def _array(x):
    return x.detach().cpu().float().numpy() if hasattr(x,'detach') and str(x.dtype)!='torch.uint8' else x.detach().cpu().numpy() if hasattr(x,'detach') else np.asarray(x)

def _result(a,context,dtype):
    return context.torch.from_numpy(np.asarray(a).copy()).to(getattr(context.torch,dtype))

def evaluate_prepare(node,value,context):
    a=node.attrs;x=_array(value).astype(np.float32).reshape(-1)
    x=np.pad(x,(0,a['padded_k']-a['logical_k'])).reshape(-1,32)
    out=np.zeros((1,a['group_count'],36),np.uint8)
    reciprocal=getattr(context,'iq4_q8_rcp_approx_ftz',None)
    c=np.array([0x3c010204],np.uint32).view(np.float32)[0]
    for g,raw in enumerate(x):
        z=np.array([_ftz(v) for v in raw],np.float32)
        if not np.isfinite(z).all(): raise EvaluationError('Q8 finite input domain required.')
        maximum=np.max(np.abs(z));d=_ftz(np.float32(maximum*c))
        if maximum==0: continue
        if reciprocal is None: raise EvaluationError('H003-N-v2 requires explicit pinned rcp.approx.ftz oracle; RN division is not its semantics.')
        r=np.float32(reciprocal(d))
        if d<=0 or not np.isfinite(r): raise EvaluationError('Unsupported Q8 reciprocal domain.')
        q=[]
        for v in z:
            quotient=_ftz(np.float32(v*r));rounded=int(_rz_add_half(abs(quotient)))
            q.append(-rounded if np.signbit(quotient) else rounded)
        if max(q)>127 or min(q)<-127: raise EvaluationError('Unsupported Q8 rounded payload domain.')
        high=_ftz(np.float32(d*np.float32(sum(q))))
        header=np.array([d,high],dtype='<f2')
        if not np.isfinite(header).all(): raise EvaluationError('Unsupported Q8 half-header domain.')
        out[0,g,:4]=header.view(np.uint8);out[0,g,4:]=np.array(q,np.int8).view(np.uint8)
    return _result(out,context,'uint8')

def _dot(q8,weight,k):
    q=np.asarray(q8,np.uint8).reshape(-1,36);w=np.asarray(weight,np.uint8).reshape(-1,k//256,136)
    values=q[:,4:].copy().view(np.int8).astype(np.int32)
    dq=q[:,:2].copy().view('<f2').reshape(-1).astype(np.float32)
    result=np.zeros(len(w),np.float32)
    for row,blocks in enumerate(w):
        p=np.zeros(128,np.float32)
        for tid in range(128):
            group=tid%8
            for b in range(tid//8,len(blocks),16):
                block=blocks[b];delta=block[:2].copy().view('<f2')[0]
                high=int(block[2])|(int(block[3])<<8)
                scale=((int(block[4+group//2])>>(4*(group%2)))&15)|(((high>>(2*group))&3)<<4);scale-=32
                packet=block[8+16*group:24+16*group]
                decoded=_LUT[np.concatenate((packet&15,packet>>4))]
                integer=int(np.dot(decoded.astype(np.int64),values[8*b+group].astype(np.int64)))*scale
                d=np.float32(np.float32(delta)*dq[8*b+group])
                p[tid]=_FMA(np.float32(integer),d,p[tid])
        p=p.reshape(4,32)
        for shift in (16,8,4,2,1): p=np.float32(p+p[:,np.arange(32)^shift])
        result[row]=np.float32(np.float32(np.float32(p[0,0]+p[1,0])+p[2,0])+p[3,0])
    return result

def evaluate_consumer(node,arguments,context):
    a=node.attrs;q=_array(arguments[0]);k=a['logical_k']
    gate=_dot(q,_array(arguments[1]),k)
    if node.op=='nn.iq4_xs_q8_1_glu':
        silu=getattr(context,'iq4_q8_silu',None)
        if silu is None: raise EvaluationError('Q8 GLU requires frozen ex2.approx/div.full SiLU oracle.')
        gate=np.float32(silu(gate)*_dot(q,_array(arguments[3]),k))
    if node.op=='nn.iq4_xs_q8_1_interleaved_projection':
        fields=gate.reshape(1,24,2,256)
        return tuple(_result(fields[:,:,i,:].reshape(1,24,32,8),context,'bfloat16') for i in range(2))
    lanes=tuple(a['output_lanes'])
    return _result(gate.reshape(1,a['logical_n']//int(np.prod(lanes)),*lanes),context,a['output_data_type'])
