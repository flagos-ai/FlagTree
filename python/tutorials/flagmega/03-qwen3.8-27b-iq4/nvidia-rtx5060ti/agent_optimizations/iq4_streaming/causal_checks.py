"""Exact adversarial dots and GPU compute/writer census; untimed diagnostics."""
import argparse
import fcntl
import json
from pathlib import Path
import subprocess
import sys
import traceback
import numpy as np
import torch
import triton
from .qualify import load_native,packed_fixture,oracle,resource


def main():
    p=argparse.ArgumentParser();p.add_argument('--attempt',type=Path,required=True);a=p.parse_args()
    attempt=a.attempt.resolve();attempt.mkdir(parents=True,exist_ok=False)
    report={'status':'running','ordinary_timing':False,'checks':[],'resources':[]}
    with open('/tmp/flagmega-gpu0.lock','a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader'],text=True)
        if apps.strip():raise RuntimeError('GPU not idle')
        try:
            fn=load_native(attempt);cfn=sys.modules['iq4_qualification_native'].q_census_kernel
            rng=np.random.default_rng(821)
            for r in (4,8,16):
                for dtype in (torch.bfloat16,torch.float32):
                    for k in (512,5120,6144,17408):
                        n=r+1;w=packed_fixture(n,k);uw=packed_fixture(n,k,71)
                        raw=rng.standard_normal(k).astype(np.float32)*np.exp2(rng.integers(-12,12,k)).astype(np.float32)
                        x=torch.tensor(raw,device='cuda',dtype=dtype);xc=x.float().cpu().numpy()
                        # Nonzero 4-byte aligned bank interior and source/output
                        # offsets. The real row stride remains K/256*136.
                        wb=torch.zeros(w.size+12,device='cuda',dtype=torch.uint8);wb[12:].copy_(torch.tensor(w.ravel(),device='cuda'))
                        ub=torch.zeros(uw.size+20,device='cuda',dtype=torch.uint8);ub[20:].copy_(torch.tensor(uw.ravel(),device='cuda'))
                        out=torch.full((n+8,),float('nan'),device='cuda');up=out.clone();census=torch.empty(triton.cdiv(n,r),device='cuda',dtype=torch.int32)
                        fn[(triton.cdiv(n,r),)](x,wb[12:],ub[20:],out[4:],up[4:],census,n,k,r,True,0,False,num_warps=4)
                        torch.cuda.synchronize()
                        for result,weight in ((out,w),(up,uw)):
                            ref=oracle(weight,xc)
                            assert np.array_equal(result[4:4+n].cpu().numpy().view(np.uint32),ref.view(np.uint32)),(r,dtype,k)
                            assert torch.isnan(result[:4]).all() and torch.isnan(result[4+n:]).all()
                        report['checks'].append({'check':'bitwise-cancellation-delta-edges-bank-offset-tail','R':r,'source':str(dtype),'K':k,'passed':True})
                n,k=12288,5120;w=packed_fixture(n,k);wg=torch.tensor(w,device='cuda');x=torch.ones(k,device='cuda',dtype=torch.bfloat16)
                outputs=[torch.full((6144+3,),float('nan'),device='cuda',dtype=torch.bfloat16) for _ in range(2)]
                compute=torch.zeros(n,device='cuda',dtype=torch.int32);writers=[torch.zeros(6144,device='cuda',dtype=torch.int32) for _ in range(2)]
                kern=cfn[(triton.cdiv(n,r),)](x,wg,*outputs,compute,*writers,k,r,num_warps=4)
                torch.cuda.synchronize()
                assert (compute==1).all() and all((v==1).all() for v in writers)
                assert all(torch.isfinite(v[:6144]).all() and torch.isnan(v[6144:]).all() for v in outputs)
                report['checks'].append({'check':'GPU-Q-compute-and-both-field-writer-census','R':r,'computed_rows':12288,'writers_per_field':6144,'all_counts_one':True})
                report['resources'].append({'cohort':'atomic-census-instrumented','R':r,**resource(kern,attempt,f'q_census_r{r}')})
                print('R',r,'all causal checks passed',flush=True)
            report['status']='causal-checks-passed'
        except BaseException as e:
            report.update(status='failed',error=repr(e),traceback=traceback.format_exc());raise
        finally:(attempt/'checks.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
