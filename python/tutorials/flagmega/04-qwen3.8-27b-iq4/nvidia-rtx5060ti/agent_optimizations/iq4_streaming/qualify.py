"""Ordinary native self-check/resource qualification; no performance samples.

Run with the repository Python path and --attempt a fresh immutable directory.
All GPU calls are serialized by the same lock as the model/NCU harnesses.
"""
from __future__ import annotations
import argparse
import ctypes
import fcntl
import hashlib
import importlib.util
import json
import os
import re
from pathlib import Path
import subprocess
import sys
import traceback

import numpy as np
import torch
import triton
from triton.flagmega.codegen.triton.templates import TritonTemplateRegistry

WORK = Path(__file__).resolve().parents[2]
LUT = np.array([-127,-104,-83,-65,-49,-35,-22,-10,1,13,25,38,53,69,89,113], np.float32)
FMA = ctypes.CDLL('libm.so.6').fmaf
FMA.argtypes = [ctypes.c_float]*3
FMA.restype = ctypes.c_float


def oracle(weight, x):
    """CPU emitted-arithmetic oracle, with RN fmaf at only the five pins."""
    out = np.zeros(weight.shape[0], np.float32)
    for r, row in enumerate(weight):
        total = np.float32(0)
        for b in range(len(x)//256):
            raw = row[b*136:(b+1)*136]
            d = raw[:2].copy().view(np.float16)[0].astype(np.float32)
            high = int(raw[2]) | int(raw[3])<<8
            for half in (0,1):
                leaves=[]
                for s in range(8):
                    wv=[]; xv=[]
                    for j in range(16):
                        t=half*128+s+8*j; g=t//32
                        scale=((int(raw[4+g//2])>>(4*(g%2)))&15)|(((high>>(2*g))&3)<<4)
                        byte=int(raw[8+16*g+t%16]); code=(byte>>(4*((t%32)//16)))&15
                        wv.append(np.float32(np.float32(d*np.float32(scale-32))*LUT[code]))
                        xv.append(x[b*256+t])
                    leaf=np.float32(wv[1]*xv[1])
                    for j in (0,2,3,4,5):
                        leaf=np.float32(FMA(float(wv[j]),float(xv[j]),float(leaf)))
                    for j in range(6,16):
                        leaf=np.float32(np.float32(wv[j]*xv[j])+leaf)
                    leaves.append(leaf)
                p=[np.float32(leaves[i]+leaves[i+1]) for i in range(0,8,2)]
                part=np.float32(np.float32(p[0]+p[2])+np.float32(p[1]+p[3]))
                total=np.float32(total+part)
        out[r]=total
    return out


def load_native(attempt):
    source='import triton\nimport triton.language as tl\n'
    source+=TritonTemplateRegistry().render('kernels/_iq4_streaming.py.jinja',{})
    source+='''
@triton.jit
def qualify_kernel(x, w, uw, out, up_out, census, N: tl.constexpr, K: tl.constexpr, R: tl.constexpr, GLU: tl.constexpr, MODE: tl.constexpr, RECORD: tl.constexpr):
    rows=_iq4_erow(tl.program_id(0).to(tl.int64)*R+tl.arange(0,R))
    a,b=_iq4_stream_dot(x,w,uw,rows,rows<N,K,R,GLU)
    if MODE == 1 and GLU:
        a=_iq4_rn_mul(_iq4_e32(a*tl.sigmoid(a)),b)
    writer=_iq4_broadcast_row(rows<N)&(_iq4_lanes()==0)
    if MODE == 2:
        addr=(rows//512)*256+rows%256
        tl.store(out+_iq4_broadcast_row(addr),a.to(out.dtype.element_ty),mask=writer&_iq4_broadcast_row((rows%512)<256))
        tl.store(up_out+_iq4_broadcast_row(addr),a.to(up_out.dtype.element_ty),mask=writer&_iq4_broadcast_row((rows%512)>=256))
    else:
        tl.store(out+_iq4_broadcast_row(rows),a.to(out.dtype.element_ty),mask=writer)
        if GLU and MODE == 0:
            tl.store(up_out+_iq4_broadcast_row(rows),b,mask=writer)
    if RECORD:
        sm=tl.inline_asm_elementwise("mov.u32 $0, %smid;",constraints="=r",args=[],dtype=tl.uint32,is_pure=False,pack=1)
        tl.store(census+tl.program_id(0),sm)

@triton.jit
def q_census_kernel(x, w, first, second, compute, writer0, writer1, K: tl.constexpr, R: tl.constexpr):
    rows=_iq4_erow(tl.program_id(0).to(tl.int64)*R+tl.arange(0,R))
    active=_iq4_erow(rows<12288)
    value,_=_iq4_stream_dot(x,w,w,rows,active,K,R,False)
    addr=(rows//512)*256+rows%256
    writer=_iq4_broadcast_row(active)&(_iq4_lanes()==0)
    f0=writer&_iq4_broadcast_row((rows%512)<256)
    f1=writer&_iq4_broadcast_row((rows%512)>=256)
    tl.atomic_add(compute+_iq4_broadcast_row(rows),1,mask=writer,sem="relaxed")
    tl.atomic_add(writer0+_iq4_broadcast_row(addr),1,mask=f0,sem="relaxed")
    tl.atomic_add(writer1+_iq4_broadcast_row(addr),1,mask=f1,sem="relaxed")
    tl.store(first+_iq4_broadcast_row(addr),value.to(tl.bfloat16),mask=f0)
    tl.store(second+_iq4_broadcast_row(addr),value.to(tl.bfloat16),mask=f1)
'''
    path=attempt/'native.py'; path.write_text(source)
    spec=importlib.util.spec_from_file_location('iq4_qualification_native',path)
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    return module.qualify_kernel


def packed_fixture(n,k,seed=17):
    rng=np.random.default_rng(seed)
    w=rng.integers(0,256,(n,k//256,136),dtype=np.uint8)
    delta=rng.choice(np.array([0.,-0.,2**-24,0.06003,-0.03007,0.01503],np.float16),size=(n,k//256))
    w[:,:,:2]=delta.view(np.uint8).reshape(n,k//256,2)
    return w.reshape(n,-1)


def resource(kernel,attempt,tag):
    kernel._init_handles()
    files={}
    for kind in ('ttgir','ptx','cubin'):
        data=kernel.asm[kind];data=data if isinstance(data,bytes) else data.encode()
        p=attempt/(tag+'.'+kind);p.write_bytes(data)
        files[kind]={'path':str(p),'sha256':hashlib.sha256(data).hexdigest()}
    cuda=ctypes.CDLL('libcuda.so.1')
    def attr(index):
        v=ctypes.c_int();assert cuda.cuFuncGetAttribute(ctypes.byref(v),index,ctypes.c_void_p(kernel.function))==0
        return v.value
    resident=ctypes.c_int()
    assert cuda.cuOccupancyMaxActiveBlocksPerMultiprocessor(ctypes.byref(resident),ctypes.c_void_p(kernel.function),128,kernel.metadata.shared)==0
    sass=attempt/(tag+'.sass')
    with sass.open('w') as f:
        subprocess.run(['/usr/local/cuda/bin/cuobjdump','--dump-sass',str(attempt/(tag+'.cubin'))],stdout=f,check=True)
    files['sass']={'path':str(sass),'sha256':hashlib.sha256(sass.read_bytes()).hexdigest()}
    usage=subprocess.check_output(['/usr/local/cuda/bin/cuobjdump','--dump-resource-usage',str(attempt/(tag+'.cubin'))],text=True)
    usage_path=attempt/(tag+'.resources.txt');usage_path.write_text(usage)
    files['resource_usage']={'path':str(usage_path),'sha256':hashlib.sha256(usage_path.read_bytes()).hexdigest()}
    return {'registers_per_thread':attr(4),'static_shared_bytes':attr(1),'dynamic_shared_bytes':kernel.metadata.shared,
        'cubin_shared_bytes':int(re.search(r'SHARED:(\d+)',usage)[1]),'stack_bytes':int(re.search(r'STACK:(\d+)',usage)[1]),
        'local_bytes_per_thread':attr(3),'spill_words_reported':kernel.n_spills,'max_threads':attr(0),
        'queried_ctas_per_sm':resident.value,'num_warps':4,'compiled_hash':kernel.hash,'files':files}


def main():
    p=argparse.ArgumentParser();p.add_argument('--attempt',type=Path,required=True);p.add_argument('--row-tile',type=int,default=4,choices=(4,));a=p.parse_args()
    attempt=a.attempt.resolve();attempt.mkdir(parents=True,exist_ok=False)
    report={'status':'running','ordinary_timing':False,'row_tile':a.row_tile,'checks':[],'resources':[],
            'full_model_NUM':'pending','NCU':'pending','performance_accepted':False,'python':sys.executable}
    with open('/tmp/flagmega-gpu0.lock','a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],text=True)
        if apps.strip():raise RuntimeError('GPU not idle: '+apps)
        report['lease']={'path':'/tmp/flagmega-gpu0.lock','pid':os.getpid(),'pre_apps':apps}
        report['device']=subprocess.check_output(['nvidia-smi','--query-gpu=name,uuid,pci.bus_id','--format=csv,noheader'],text=True).strip()
        try:
            fn=load_native(attempt);R=a.row_tile
            rng=np.random.default_rng(19)
            # K512/tail with exact CPU oracle, BF16 and retained F32 inputs.
            for dtype in (torch.bfloat16,torch.float32):
                n,k=R+1,512;w=packed_fixture(n,k);uw=packed_fixture(n,k,21)
                x=torch.from_numpy(rng.standard_normal(k).astype(np.float32)).to('cuda',dtype=dtype)
                xc=x.float().cpu().numpy();ref=oracle(w,xc);uref=oracle(uw,xc)
                wg=torch.tensor(w,device='cuda');ug=torch.tensor(uw,device='cuda')
                out=torch.full((n+3,),float('nan'),device='cuda');up=out.clone();census=torch.empty(triton.cdiv(n,R),device='cuda',dtype=torch.int32)
                kern=fn[(triton.cdiv(n,R),)](x,wg,ug,out,up,census,n,k,R,True,0,False,num_warps=4)
                torch.cuda.synchronize()
                assert np.array_equal(out[:n].cpu().numpy().view(np.uint32),ref.view(np.uint32))
                assert np.array_equal(up[:n].cpu().numpy().view(np.uint32),uref.view(np.uint32))
                assert torch.isnan(out[n:]).all() and torch.isnan(up[n:]).all()
                report['checks'].append({'check':'exact-dots-tail','dtype':str(dtype),'rows':n,'K':k,'passed':True})
            # All 16 codes and all 64 signed scales, using actual kernel decode.
            n,k=1024,256;w=np.zeros((n,136),np.uint8);w[:,:2]=np.array([1.],np.float16).view(np.uint8)
            for scale in range(64):
                for code in range(16):
                    row=w[scale*16+code];row[2:4]=np.array([sum(((scale>>4)&3)<<(2*g) for g in range(8))],np.uint16).view(np.uint8)
                    row[4:8]=(scale&15)*17;row[8:]=code*17
            x=torch.ones(k,device='cuda');wg=torch.tensor(w,device='cuda');out=torch.empty(n,device='cuda');census=torch.empty(triton.cdiv(n,R),device='cuda',dtype=torch.int32)
            fn[(triton.cdiv(n,R),)](x,wg,wg,out,out,census,n,k,R,False,0,False,num_warps=4)
            torch.cuda.synchronize();expected=np.array([np.float32(0)+np.float32(256*(s-32)*v) for s in range(64) for v in LUT])
            assert np.array_equal(out.cpu().numpy().view(np.uint32),expected.view(np.uint32))
            report['checks'].append({'check':'all16-codes-all64-scales','passed':True})
            # KV R4 first, then all eight unique ordinary shape/output entries.
            shapes=[(1024,5120,0),(17408,5120,1),(12288,5120,2),(10240,5120,0),(6144,5120,0),(5120,6144,0),(5120,17408,0),(248320,5120,0)]
            for n,k,mode in shapes:
                tag=f'n{n}_k{k}_r{R}_mode{mode}'
                w=packed_fixture(n,k);wg=torch.tensor(w,device='cuda');ug=wg.clone() if mode==1 else wg
                x=torch.ones(k,device='cuda',dtype=torch.bfloat16)
                dtype=torch.float32 if n==248320 else torch.bfloat16
                size=n//2 if mode==2 else n
                out=torch.full((size+3,),float('nan'),device='cuda',dtype=dtype);up=out.clone()
                census=torch.full((triton.cdiv(n,R),),-1,device='cuda',dtype=torch.int32)
                kernel=fn[(triton.cdiv(n,R),)](x,wg,ug,out,up,census,n,k,R,mode==1,mode,False,num_warps=4)
                torch.cuda.synchronize()
                assert torch.isfinite(out[:size]).all() and torch.isnan(out[size:]).all()
                if mode==2:assert torch.isfinite(up[:size]).all() and torch.isnan(up[size:]).all()
                row={'N':n,'K':k,'mode':mode,'R':R,'grid':[triton.cdiv(n,R),1,1],**resource(kernel,attempt,tag)}
                # Separate instrumented placement cohort; do not call its
                # resources or execution an ordinary performance sample.
                placed=fn[(triton.cdiv(n,R),)](x,wg,ug,out,up,census,n,k,R,mode==1,mode,True,num_warps=4)
                torch.cuda.synchronize();ids=census.cpu().numpy();assert (ids>=0).all()
                row['placement']={'cohort':'smid-instrumented','all_ctas_completed':True,'sm_counts':dict(zip(*[v.tolist() for v in np.unique(ids,return_counts=True)])),
                                  'resources':resource(placed,attempt,tag+'_placement')}
                row['admission']='passed';report['resources'].append(row)
                (attempt/'qualification.json').write_text(json.dumps(report,indent=2)+'\n')
                print(tag,row['registers_per_thread'],row['queried_ctas_per_sm'],flush=True)
                del wg,ug,x,out,up,census
            report['status']='native-self-checks-passed'
        except BaseException as e:
            report.update(status='failed',error=repr(e),traceback=traceback.format_exc());raise
        finally:
            (attempt/'qualification.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
