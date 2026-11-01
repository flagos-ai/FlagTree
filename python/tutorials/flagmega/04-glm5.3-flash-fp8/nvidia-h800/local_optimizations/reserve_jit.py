"""Reserve one idle GPU for small correctness tests with guarded ownership."""
import argparse,csv,ctypes,json,os,signal,subprocess,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--device',type=int,default=5);p.add_argument('--output',type=Path,required=True);p.add_argument('--budget-bytes',type=int,default=1_073_741_824);p.add_argument('command',nargs=argparse.REMAINDER);a=p.parse_args()
root=a.output.resolve();root.mkdir(exist_ok=False)
command=a.command[1:] if a.command[:1]==['--'] else a.command;assert command

def inventory():
    owners={}
    for uuid,pid in csv.reader(subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True).splitlines()):owners.setdefault(uuid.strip(),set()).add(int(pid))
    data=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,utilization.gpu,temperature.gpu,clocks_event_reasons.sw_thermal_slowdown','--format=csv,noheader,nounits'],text=True)
    return [dict(index=int(i),uuid=u.strip(),memory_mib=int(m),util=int(v),temperature=int(t),thermal=s.strip(),pids=sorted(owners.get(u.strip(),set()))) for i,u,m,v,t,s in csv.reader(data.splitlines())]
while True:
    free=[g for g in inventory() if g['index']==a.device and g['index']!=6 and not g['pids'] and g['memory_mib']<128 and g['util']==0 and g['temperature']<65 and g['thermal']=='Not Active']
    if free:break
    print(json.dumps(dict(event='waiting_for_one_idle_gpu_for_jit')),flush=True);time.sleep(10)
card=free[0];os.environ['CUDA_VISIBLE_DEVICES']=card['uuid']
driver=ctypes.CDLL('libcuda.so.1');assert driver.cuInit(0)==0
driver.cuMemAlloc_v2.argtypes=[ctypes.POINTER(ctypes.c_uint64),ctypes.c_size_t];driver.cuMemFree_v2.argtypes=[ctypes.c_uint64]
context=ctypes.c_void_p();pointer=ctypes.c_uint64();retained=allocated=False;child=None
try:
    assert driver.cuDevicePrimaryCtxRetain(ctypes.byref(context),0)==0;retained=True
    assert driver.cuCtxSetCurrent(context)==0
    assert driver.cuMemAlloc_v2(ctypes.byref(pointer),a.budget_bytes)==0,('JIT reservation raced with another owner',card)
    allocated=True;assert driver.cuCtxSetCurrent(None)==0
    observed=next(g for g in inventory() if g['uuid']==card['uuid'])
    # NVML reports host PIDs while this workspace uses container PIDs.
    # The successfulexplicit diagnostic reservation must have exactly one visible context owner.
    assert len(observed['pids'])==1 and observed['memory_mib']>=a.budget_bytes//(1024**2),observed
    keeper=observed['pids'][0]
    (root/'launch.json').write_text(json.dumps(dict(card=card,keeper=keeper,container_pid=os.getpid(),reserved_bytes=a.budget_bytes,command=command,device_correctness_tests=True),indent=2)+'\n')
    print(json.dumps(dict(event='launch_reserved_device_tests',card=card,command=command)),flush=True)
    with (root/'run.log').open('w') as log:
        child=subprocess.Popen(command,env=dict(os.environ,TMPDIR='/dev/shm',PYTHONDONTWRITEBYTECODE='1'),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        begin=time.monotonic()
        while child.poll() is None:
            observed=next(g for g in inventory() if g['uuid']==card['uuid'])
            assert keeper in observed['pids'] and len(observed['pids'])<=2,('unexpected owner; stopping only this owned device-test job',observed)
            assert time.monotonic()-begin<1200,'Device-test child exceeded20min'
            time.sleep(1)
        assert child.returncode==0,child.returncode
    (root/'DONE').write_text('Device correctness tests completed.\n')
finally:
    if child is not None and child.poll() is None:
        os.killpg(child.pid,signal.SIGTERM)
        try:child.wait(timeout=15)
        except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait(timeout=15)
    if allocated:
        assert driver.cuCtxSetCurrent(context)==0;assert driver.cuMemFree_v2(pointer)==0
    assert driver.cuCtxSetCurrent(None)==0
    if retained:assert driver.cuDevicePrimaryCtxRelease(0)==0
