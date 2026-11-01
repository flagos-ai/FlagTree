"""Fresh same-card native/FlagMega/FlagMega/native full-model graph comparison."""
import argparse
import csv
import ctypes
import hashlib
import json
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from local_optimizations.scenarios import report_scenarios, reference_scenarios, validate_lengths
import pynvml as nvml
nvml.nvmlInit()

parser=argparse.ArgumentParser()
parser.add_argument('--output',type=Path,required=True)
parser.add_argument('--trial',type=Path,required=True)
parser.add_argument('--validation',type=Path,required=True)
parser.add_argument('--checkpoint',type=Path,required=True)
parser.add_argument('--native-python',type=Path,required=True)
parser.add_argument('--sdk',type=Path,required=True)
parser.add_argument('--native-triton-cache',type=Path,required=True)
parser.add_argument('--native-inductor-cache',type=Path,required=True)
parser.add_argument('--prompt-lengths',type=int,nargs='+',default=[1])
parser.add_argument('--validate-only',action='store_true')
parser.add_argument('--scenario-validation',type=Path)
parser.add_argument('--reverse',action='store_true')
args=parser.parse_args()
validate_lengths(args.prompt_lengths)
if args.prompt_lengths!=[1] and not args.validate_only and not args.scenario_validation:
    parser.error('Long-prompt measurements require prior scenario validation')
root=args.output.resolve();root.mkdir(exist_ok=False)
trial=args.trial.resolve()
here=trial/'local_optimizations'
selection=dict(trial=str(trial),numerics=str(args.validation.resolve()),ptxas_path=None)
os.environ['FLAGCX_MODULE_PATH']=str(args.sdk.resolve())
(root/'selection.json').write_text(json.dumps(selection,indent=2)+'\n')
expected=[3847,284,1373,8282,4218,3755,947,906]


def inventory():
    values=[]
    thermal_mask=nvml.nvmlClocksEventReasonSwThermalSlowdown | nvml.nvmlClocksEventReasonHwThermalSlowdown
    for index in range(nvml.nvmlDeviceGetCount()):
        handle=nvml.nvmlDeviceGetHandleByIndex(index)
        uuid=nvml.nvmlDeviceGetUUID(handle)
        if isinstance(uuid,bytes):uuid=uuid.decode()
        values.append(dict(index=index,uuid=uuid,
            memory_mib=nvml.nvmlDeviceGetMemoryInfo(handle,version=nvml.nvmlMemory_v2).used//(1024*1024),
            util=nvml.nvmlDeviceGetUtilizationRates(handle).gpu,
            temperature=nvml.nvmlDeviceGetTemperature(handle,nvml.NVML_TEMPERATURE_GPU),
            thermal='Active' if nvml.nvmlDeviceGetCurrentClocksEventReasons(handle)&thermal_mask else 'Not Active',
            sm_mhz=nvml.nvmlDeviceGetClockInfo(handle,nvml.NVML_CLOCK_SM),
            pids=sorted(p.pid for p in nvml.nvmlDeviceGetComputeRunningProcesses(handle))))
    return values


def check_validation():
    output=Path(selection['numerics'])
    receipt=json.loads((output/'supervisor-exit.json').read_text())
    verified=json.loads((output/'verified.json').read_text())
    assert receipt['exit_code']==0 and receipt['real_child_wait'] and receipt['source_receipts_unchanged']
    assert verified['passed'] and verified['state_byte_digests_verified'] and verified['all_official8_passed'] and verified['all_zero_spill']
    warmup=json.loads((trial/'warmup.json').read_text())
    assert verified['source_sha256']==warmup['source_sha256']
    assert receipt['sdk_host_sha256']==hashlib.sha256((args.sdk/'libflagcx.so').read_bytes()).hexdigest()
    entries={e['symbol']:e['hash'] for e in warmup['entries']}
    assert len(entries)==26
    reports=[json.loads((output/f'rank-{i}.json').read_text()) for i in range(4)]
    assert all(q['passed'] and q['mode']=='check-owned' and q['compiled_entries']==entries and len(q['resources'])==228 and all(v['spill_bytes']==0 for v in q['resources']) for q in reports)
    assert all(len(q['requests'])==2 and all(v['tokens']==expected and v['all_state_bytes_equal'] for v in q['requests']) for q in reports)
    return verified


def main():
    validation=check_validation()
    if args.scenario_validation:
        scenario_dir=args.scenario_validation.resolve()
        scenario_receipt=json.loads((scenario_dir/'scenario-validation.json').read_text())
        assert scenario_receipt['passed'] and scenario_receipt['prompt_lengths']==args.prompt_lengths
        assert scenario_receipt['source_sha256']==validation['source_sha256']
        reference=reference_scenarios(json.loads((scenario_dir/'N0/report.json').read_text()),args.prompt_lengths)
        for rank in range(4):
            q=json.loads((scenario_dir/f'F0/rank-{rank}.json').read_text())
            assert q['passed'] and q['validation_only']
            for scenario in report_scenarios(q):
                assert len(scenario['requests'])==2 and scenario['validation_requests']
                assert all(r['tokens']==reference[scenario['prompt_length']]['requests'][0]['tokens'] for r in scenario['requests'])
    identities=json.loads((trial/'source-identities-v3.json').read_text())['files']
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in identities.items()),'Candidate numerical source inputs changed'
    while True:
        available=[g for g in inventory() if g['index'] not in {0,6} and not g['pids'] and g['memory_mib']<128 and g['util']==0 and g['temperature']<65 and g['thermal']=='Not Active']
        if len(available)>=4:break
        print(json.dumps(dict(event='waiting_for_four_idle_cards',available=available)),flush=True)
        time.sleep(10)
    cards=available[:4];uuids={g['uuid'] for g in cards}
    os.environ['CUDA_VISIBLE_DEVICES']=','.join(g['uuid'] for g in cards)
    driver=ctypes.CDLL('libcuda.so.1');assert driver.cuInit(0)==0
    driver.cuMemAlloc_v2.argtypes=[ctypes.POINTER(ctypes.c_uint64),ctypes.c_size_t]
    driver.cuMemFree_v2.argtypes=[ctypes.c_uint64]
    contexts={};allocations={};child=None
    try:
        for ordinal in range(4):
            context=ctypes.c_void_p();assert driver.cuDevicePrimaryCtxRetain(ctypes.byref(context),ordinal)==0
            contexts[ordinal]=context
        assert driver.cuCtxSetCurrent(None)==0
        current=[g for g in inventory() if g['uuid'] in uuids]
        common=set.intersection(*(set(g['pids']) for g in current));assert len(common)==1,current
        keeper=next(iter(common));assert all(set(g['pids'])=={keeper} for g in current),current
        files=json.loads((trial/'source-identities-v3.json').read_text())['files']
        assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in files.items())
        scripts=[Path(p) for p in files]
        scripts.extend([Path(__file__),trial/'benchmark_vllm.py',here/'analyse_native.py',trial/'artifact/generated_kernels.py',trial/'warmup.json',trial/'dag-audit.json',args.sdk/'libflagcx.so',args.sdk/'libflagcx_device.bc'])
        native_root=args.native_python.absolute().parent.parent/'lib/python3.10/site-packages/vllm'
        assert native_root.is_dir(),native_root
        scripts.extend(p for p in native_root.rglob('*.py') if p.is_file())
        order=['N0','F0'] if args.validate_only else (['F0','N0','N1','F1'] if args.reverse else ['N0','F0','F1','N1'])
        plan=dict(cards=cards,keeper=keeper,trial=str(trial),order=order,validation=validation,checkpoint=str(args.checkpoint),sdk=str(args.sdk),native_python=str(args.native_python),
            hardware_sampler='NVML 20ms, SW/HW thermal flags and all compute process IDs',
            prompt_lengths=args.prompt_lengths,validation_only=args.validate_only,
            scope=f'Full45/E288/top8/H4096/cache4096/TP4/prompts{args.prompt_lengths}/output8; graph-on; FlagMega compiled token-scan prefill, native batched prefill; independent requests including state carry, greedy sample and token handoff.',
            source_receipts={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in scripts},
            diagnostic_only=True,performance_acceptance=False,public_promoted=False)
        (root/'launch.json').write_text(json.dumps(plan,indent=2)+'\n')
        for label in order:
            engine='native' if label.startswith('N') else 'flagmega'
            current=[g for g in inventory() if g['uuid'] in uuids]
            assert all(set(g['pids'])=={keeper} and g['thermal']=='Not Active' for g in current),current
            output=root/label;output.mkdir(exist_ok=False)
            env=dict(os.environ,TMPDIR='/dev/shm',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',GLOO_SOCKET_IFNAME='lo',VLLM_HOST_IP='127.0.0.1')
            env.pop('TRITON_PTXAS_PATH',None)
            if engine=='flagmega' and selection.get('ptxas_path'):
                env['TRITON_PTXAS_PATH']=selection['ptxas_path']
            if engine=='native':
                env.update(TRITON_CACHE_DIR=str(args.native_triton_cache),TORCHINDUCTOR_CACHE_DIR=str(args.native_inductor_cache),VLLM_ALLOW_INSECURE_SERIALIZATION='1')
                command=[str(args.native_python),str(trial/'benchmark_vllm.py'),'--output',str(output),'--checkpoint',str(args.checkpoint)]
            else:
                reservation=output/'reservation';reservation.mkdir()
                env.update(FLAGMEGA_GPU_RESERVATION_DIR=str(reservation),FLAGMEGA_RUN_OUTPUT=str(output))
                for ordinal in range(4):
                    assert driver.cuCtxSetCurrent(contexts[ordinal])==0
                    pointer=ctypes.c_uint64()
                    assert driver.cuMemAlloc_v2(ctypes.byref(pointer),80_000_017_920)==0,('full FlagMega budget unavailable',label,ordinal)
                    allocations[ordinal]=pointer
                assert driver.cuCtxSetCurrent(None)==0
                command=[sys.executable,'-m','torch.distributed.run','--nnodes=1','--nproc-per-node=4','--rdzv-backend=c10d','--rdzv-endpoint=127.0.0.1:0','--local-addr=127.0.0.1',str(trial/'benchmark_gpu.py'),'--output',str(output),'--trial',str(trial)]

            command.extend(['--prompt-lengths',*[str(v) for v in args.prompt_lengths]])
            if args.validate_only:command.append('--validate-only')
            if engine=='flagmega' and args.prompt_lengths!=[1]:
                reference_path=(args.scenario_validation/'N0/report.json' if args.reverse else root/'N0/report.json')
                command.extend(['--native-reference',str(reference_path.resolve())])

            (output/'launch.json').write_text(json.dumps(dict(command=command,cards=cards,keeper=keeper,environment={k:env.get(k) for k in ('CUDA_VISIBLE_DEVICES','TRITON_CACHE_DIR','TORCHINDUCTOR_CACHE_DIR')}),indent=2)+'\n')
            print(json.dumps(dict(event='launch',label=label,command=command)),flush=True)
            started=time.monotonic();released=set()
            with (output/'run.log').open('w') as log,(output/'gpu-monitor.jsonl').open('w') as monitor:
                child=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                while child.poll() is None:
                    if engine=='flagmega':
                        for path in reservation.glob('ready-*.json'):
                            if path.name in released:continue
                            ready=json.loads(path.read_text());ordinal=ready['rank']
                            os.kill(ready['pid'],0)
                            assert ready['uuid']==cards[ordinal]['uuid']
                            assert ordinal in allocations
                            assert driver.cuCtxSetCurrent(contexts[ordinal])==0
                            assert driver.cuMemFree_v2(allocations.pop(ordinal))==0
                            assert driver.cuCtxSetCurrent(None)==0
                            (reservation/f"released-{ordinal}-{ready['pid']}").write_text('Full budget handed to this live worker\n')
                            released.add(path.name)
                    sample=inventory()
                    active=[g for g in sample if g['uuid'] in uuids]
                    assert all(keeper in g['pids'] and len(g['pids'])<=2 for g in active),('unexpected additional CUDA process; abort only owned job',label,active)
                    monitor.write(json.dumps(dict(timestamp=time.time(),gpus=sample))+'\n');monitor.flush()
                    assert time.monotonic()-started<3600,('owned job exceeded60min',label)
                    time.sleep(.02)
                assert child.returncode==0,(label,child.returncode)
                assert not allocations,('Unreleased FlagMega budget after phase',label)
            print(json.dumps(dict(event='finished',label=label,returncode=child.returncode)),flush=True)
            for _ in range(60):
                current=[g for g in inventory() if g['uuid'] in uuids]
                if all(set(g['pids'])=={keeper} for g in current):break
                time.sleep(1)
            else:raise RuntimeError('Owned workers did not exit; no next engine launched')
        if args.validate_only:
            refs=reference_scenarios(json.loads((root/'N0/report.json').read_text()),args.prompt_lengths)
            for rank in range(4):
                q=json.loads((root/f'F0/rank-{rank}.json').read_text())
                assert q['passed'] and q['validation_only'] and q['source_sha256']==validation['source_sha256']
                for scenario in report_scenarios(q):
                    assert scenario['prompt']==refs[scenario['prompt_length']]['prompt']
                    assert len(scenario['requests'])==2 and scenario['validation_requests']
                    assert all(r['tokens']==refs[scenario['prompt_length']]['requests'][0]['tokens'] for r in scenario['requests'])
            (root/'scenario-validation.json').write_text(json.dumps(dict(passed=True,prompt_lengths=args.prompt_lengths,source_sha256=validation['source_sha256'],ranks=4,independent_requests_per_scenario=2,all_native_tokens_equal=True,all_state_finite_and_cache_tails_untouched=True,state_byte_equality_evaluated=False),indent=2)+'\n')
        (root/'DONE').write_text('All four fresh same-card rounds completed. Analysis and telemetry acceptance required.\n')
    finally:
        if child is not None and child.poll() is None:
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=15)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait(timeout=15)
        if child is not None:
            # A dead engine leader can leave workers holding inherited pipes.
            # This group was created by our own Popen(start_new_session=True).
            try:
                os.killpg(child.pid,signal.SIGKILL)
            except ProcessLookupError:
                pass
        for ordinal,pointer in allocations.items():
            assert driver.cuCtxSetCurrent(contexts[ordinal])==0
            assert driver.cuMemFree_v2(pointer)==0
        assert driver.cuCtxSetCurrent(None)==0
        for ordinal in contexts:assert driver.cuDevicePrimaryCtxRelease(ordinal)==0


if __name__=='__main__':
    main()
    if not args.validate_only:
        subprocess.run([sys.executable,str(here/'analyse_native.py'),'--output',str(root)],check=True)
