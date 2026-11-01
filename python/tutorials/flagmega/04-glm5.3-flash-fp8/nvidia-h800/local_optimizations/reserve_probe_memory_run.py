"""Reserve a full four-card budget, hand it to workers, never submit GPU work."""
import argparse
import csv
import ctypes
import json
import os
from pathlib import Path
import signal
import subprocess
import time

parser = argparse.ArgumentParser()
parser.add_argument('--output',type=Path,required=True)
parser.add_argument('--num-devices',type=int,choices=(2,4),default=4)
parser.add_argument('--exclude-device',type=int,action='append',default=[])
parser.add_argument('--ready-report',type=Path)
parser.add_argument('--budget-bytes',type=int,default=80_875_032_929)
parser.add_argument('command',nargs=argparse.REMAINDER)
args=parser.parse_args()
root=args.output.resolve();root.mkdir(exist_ok=False)
reservation=root/'reservation';reservation.mkdir()
command=args.command[1:] if args.command[:1]==['--'] else args.command
assert command
if args.ready_report:
    while not args.ready_report.is_file():
        time.sleep(3)
    assert json.loads(args.ready_report.read_text())['jit_ok'],args.ready_report.read_text()


def inventory():
    apps={}
    for uuid,pid in csv.reader(subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True).splitlines()):
        apps.setdefault(uuid.strip(),set()).add(int(pid))
    values=csv.reader(subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,utilization.gpu,temperature.gpu,clocks_event_reasons.sw_thermal_slowdown,clocks.current.sm','--format=csv,noheader,nounits'],text=True).splitlines())
    return [dict(index=int(i),uuid=u.strip(),memory_mib=int(m),util=int(util),temperature=int(t),thermal=s.strip(),sm_mhz=int(c),pids=sorted(apps.get(u.strip(),set()))) for i,u,m,util,t,s,c in values]


while True:
    cards=[g for g in inventory() if g['index'] not in {6,*args.exclude_device} and not g['pids'] and g['memory_mib']<128 and g['util']==0 and g['temperature']<65 and g['thermal']=='Not Active'][:args.num_devices]
    if len(cards)==args.num_devices:
        break
    print(json.dumps(dict(event='waiting_for_four_idle_cards',cards=cards)),flush=True)
    time.sleep(10)
os.environ['CUDA_VISIBLE_DEVICES']=','.join(g['uuid'] for g in cards)
driver=ctypes.CDLL('libcuda.so.1')
assert driver.cuInit(0)==0
driver.cuMemAlloc_v2.argtypes=[ctypes.POINTER(ctypes.c_uint64),ctypes.c_size_t]
driver.cuMemFree_v2.argtypes=[ctypes.c_uint64]
contexts={};allocations={};child=None
budget=args.budget_bytes
assert budget>0
try:
    for ordinal in range(args.num_devices):
        context=ctypes.c_void_p()
        assert driver.cuDevicePrimaryCtxRetain(ctypes.byref(context),ordinal)==0
        contexts[ordinal]=context
        assert driver.cuCtxSetCurrent(context)==0
        pointer=ctypes.c_uint64()
        assert driver.cuMemAlloc_v2(ctypes.byref(pointer),budget)==0,('reservation failed',ordinal)
        allocations[ordinal]=pointer
    assert driver.cuCtxSetCurrent(None)==0
    current=[g for g in inventory() if g['uuid'] in {c['uuid'] for c in cards}]
    common=set.intersection(*(set(g['pids']) for g in current))
    assert len(common)==1,current
    keeper=next(iter(common))
    assert all(set(g['pids'])=={keeper} for g in current),current
    (root/'launch.json').write_text(json.dumps(dict(cards=cards,keeper=keeper,command=command,reserved_bytes_per_card=budget),indent=2)+'\n')
    print(json.dumps(dict(event='reserved_full_budget',cards=cards,nbytes=budget)),flush=True)
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',TMPDIR='/dev/shm',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',
             FLAGMEGA_GPU_RESERVATION_DIR=str(reservation),FLAGMEGA_RUN_OUTPUT=str(root))
    start=time.monotonic();released=set()
    with (root/'run.log').open('w') as log,(root/'telemetry.jsonl').open('w') as monitor:
        child=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        while child.poll() is None:
            for path in reservation.glob('ready-*.json'):
                if path.name in released:
                    continue
                ready=json.loads(path.read_text());ordinal=ready['rank']
                os.kill(ready['pid'],0)  # Ready file alone is not live-worker evidence.
                assert ready['uuid']==cards[ordinal]['uuid'],ready
                if ordinal in allocations:
                    assert driver.cuCtxSetCurrent(contexts[ordinal])==0
                    assert driver.cuMemFree_v2(allocations.pop(ordinal))==0
                    assert driver.cuCtxSetCurrent(None)==0
                (reservation/f"released-{ordinal}-{ready['pid']}").write_text('GPU memory handed to this live worker\n')
                released.add(path.name)
                print(json.dumps(dict(event='handoff',rank=ordinal,pid=ready['pid'])),flush=True)
            monitor.write(json.dumps(dict(timestamp=time.time(),gpus=inventory()))+'\n');monitor.flush()
            assert time.monotonic()-start<1200,'Owned task exceeded 20-minute timeout'
            time.sleep(.5)
        print(json.dumps(dict(event='finished',returncode=child.returncode,seconds=time.monotonic()-start)),flush=True)
        assert child.returncode==0,child.returncode
finally:
    if child is not None and child.poll() is None:
        os.killpg(child.pid,signal.SIGTERM)
        try:
            child.wait(timeout=15)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid,signal.SIGKILL);child.wait(timeout=15)
    for ordinal,pointer in allocations.items():
        assert driver.cuCtxSetCurrent(contexts[ordinal])==0
        assert driver.cuMemFree_v2(pointer)==0
    assert driver.cuCtxSetCurrent(None)==0
    for ordinal in contexts:
        assert driver.cuDevicePrimaryCtxRelease(ordinal)==0
