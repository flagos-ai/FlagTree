"""Qualify the actual upstream-derived FP32 CUDA region; untimed only."""
import argparse
import ctypes
import fcntl
import json
from pathlib import Path
import subprocess
import traceback
import numpy as np
import torch
import triton
from ..qualify import packed_fixture,oracle,load_native


def main():
    p=argparse.ArgumentParser();p.add_argument('--library',type=Path,required=True);p.add_argument('--attempt',type=Path,required=True);a=p.parse_args()
    attempt=a.attempt.resolve();attempt.mkdir(parents=True,exist_ok=False)
    report={'status':'running','ordinary_timing':False,'K9':'OPEN: paired profiler measurements pending','checks':[]}
    with open('/tmp/flagmega-gpu0.lock','a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader'],text=True)
        if apps.strip():raise RuntimeError('GPU not idle')
        try:
            lib=ctypes.CDLL(str(a.library.resolve()));launch=lib.iq4_industry_launch
            launch.argtypes=[ctypes.c_void_p]*5+[ctypes.c_int]*5+[ctypes.c_void_p];launch.restype=ctypes.c_int
            fn=load_native(attempt)
            torch.set_num_threads(1)
            rng=np.random.default_rng(73)
            for dtype in (torch.bfloat16,torch.float32):
                for n,k,mode,glu in ((5,512,0,False),(17408,5120,1,True),(12288,5120,2,False),
                                    (10240,5120,1,False),(6144,5120,1,False),(5120,6144,1,False),
                                    (5120,17408,1,False),(1024,5120,1,False),(248320,5120,0,False)):
                    w=packed_fixture(n,k);upw=packed_fixture(n,k,59)
                    x=torch.tensor(rng.standard_normal(k).astype(np.float32),device='cuda',dtype=dtype)
                    wg=torch.tensor(w,device='cuda');ug=torch.tensor(upw,device='cuda') if glu else wg
                    outdtype=torch.float32 if mode==0 else torch.bfloat16;size=n//2 if mode==2 else n
                    out=torch.full((size+3,),float('nan'),device='cuda',dtype=outdtype);second=out.clone()
                    expected=out.clone();expected_second=second.clone();census=torch.empty(triton.cdiv(n,4),device='cuda',dtype=torch.int32)
                    assert launch(x.data_ptr(),wg.data_ptr(),ug.data_ptr(),out.data_ptr(),second.data_ptr(),n,k,int(dtype==torch.float32),mode,int(glu),torch.cuda.current_stream().cuda_stream)==0
                    fn[(triton.cdiv(n,4),)](x,wg,ug,expected,expected_second,census,n,k,4,glu,mode,False,num_warps=4)
                    torch.cuda.synchronize()
                    if n==5:
                        ref=oracle(w,x.float().cpu().numpy())
                        assert np.array_equal(out[:n].cpu().numpy().view(np.uint32),ref.view(np.uint32))
                    assert torch.equal(out[:size].view(torch.uint8),expected[:size].view(torch.uint8)),(n,k,mode,dtype)
                    if mode==2:assert torch.equal(second[:size].view(torch.uint8),expected_second[:size].view(torch.uint8))
                    assert torch.isnan(out[size:]).all() and torch.isnan(second[size:]).all()
                    report['checks'].append({'N':n,'K':k,'mode':mode,'source':str(dtype),'exact_candidate_equal':True,'sentinel_passed':True})
                    print(n,k,mode,dtype,'exact',flush=True)
                    del x,wg,ug,out,second,expected,expected_second,census
            report['status']='port-numerical-qualified'
        except BaseException as e:
            report.update(status='failed',error=repr(e),traceback=traceback.format_exc());raise
        finally:(attempt/'qualification.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
