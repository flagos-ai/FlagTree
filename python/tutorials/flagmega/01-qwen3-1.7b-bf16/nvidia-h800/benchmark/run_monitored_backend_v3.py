#!/usr/bin/env python3
"""Versioned single-backend monitor. --preflight-only performs no GPU query.

The declared CUDA selector is never replaced with UUID text. NVML monitors the
leased UUID; a fresh child separately checks CUDA0 UUID/PCI before model load.
Local process identity and the lone NVML host PID epoch remain distinct.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parent/'backends'))
from device_identity import selector_policy, normalize_uuid, normalize_pci, validate_visibility


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def utc(): return datetime.now(timezone.utc).isoformat()
def write_new(path, data):
    with Path(path).open('x') as stream: json.dump(data, stream, indent=2); stream.write('\n')


def validate_config(config, work):
    policy = selector_policy(config['backend'], config.get('GPU_UUID'), config['gpu_selector_kind'],
                             config.get('cuda_visible_device'))
    environment = config['environment']
    if environment.get('CUDA_VISIBLE_DEVICES') != policy['cuda_visible_device'] or environment.get('CUDA_DEVICE_ORDER') != 'PCI_BUS_ID':
        raise ValueError('Configured environment must preserve the explicitly declared selector/order')
    argv = config['argv']
    if not isinstance(argv, list) or not argv or any(not isinstance(a, str) for a in argv):
        raise ValueError('Use a concrete argv array, not shell text')
    def option(name):
        indices = [i for i, value in enumerate(argv) if value == name]
        if len(indices) != 1 or indices[0]+1 >= len(argv):raise ValueError('Missing/duplicate child CLI identity: '+name)
        return argv[indices[0]+1]
    for name, value in (('--gpu-selector-kind',policy['selector_kind']),
                        ('--physical-gpu-uuid',policy['physical_gpu_uuid']),
                        ('--cuda-visible-device',policy['cuda_visible_device'])):
        if value is not None and option(name) != value:raise ValueError('Child CLI and declared selector differ: '+name)
    proof_path = Path(option('--device-identity-proof')).resolve()
    job = (work/'.local/jobs'/config['job_id']).resolve()
    if proof_path != job/'device-identity.json':raise ValueError('Identity proof must belong to this new job')
    pins = config['source_pins']
    if not pins:raise ValueError('Frozen source/dependency pins required')
    for pin in pins:
        path = Path(pin['path']);path = path if path.is_absolute() else work/path
        if sha(path) != pin['sha256']:raise ValueError('Frozen source/dependency changed: '+str(path))
    return {'selector_policy':policy,'identity_proof_path':str(proof_path),
            'source_pins_checked':len(pins),'GPU_initialized':False,
            'runtime_gates':'fresh root lease, NVML UUID/index/PCI, child CUDA0/native proof and isolation pending'}


def apply_selector(config, nvml_identity, inherited=None):
    """Pure validation after the launcher obtains actual NVML identity."""
    identity={'schema':'flagmega.device-identity/v3','evidence_kind':'actual-NVML-query',
        'backend':config['backend'],'physical_gpu_uuid':normalize_uuid(config['GPU_UUID']),
        'selector_kind':config['gpu_selector_kind'],'cuda_visible_device':config['cuda_visible_device'],
        'CUDA_DEVICE_ORDER':'PCI_BUS_ID','nvml_index':nvml_identity['nvml_index'],
        'pci_bus_id':normalize_pci(nvml_identity['pci_bus_id'])}
    if normalize_uuid(nvml_identity['uuid']) != identity['physical_gpu_uuid']:
        raise ValueError('Actual NVML handle UUID differs from leased GPU')
    env=dict(os.environ if inherited is None else inherited);env.update(config['environment'])
    validate_visibility(identity,env)
    # No fallback/overwrite of visibility. Only benchmark-owned proof and Python
    # bytecode settings are added after the exact configured selector is checked.
    return identity,env


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--work-dir',type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument('--preflight-only',action='store_true')
    parser.add_argument('--output',type=Path)
    args=parser.parse_args(argv);work=args.work_dir.resolve();config=json.loads(args.config.read_text())
    preflight=validate_config(config,work)
    if args.preflight_only:
        if not args.output:parser.error('--preflight-only requires immutable --output')
        write_new(args.output,{'schema':'flagmega.monitor-preflight/v3','status':'CPU-preflight',
            'config':str(args.config),'config_sha256':sha(args.config),'monitor_sha256':sha(__file__),**preflight})
        return 0
    if config.get('GPU_access') is not True or not config.get('root_lease_evidence'):
        raise ValueError('This prepared CPU config has no root-recorded GPU lease')
    if not config.get('GPU_UUID') or config.get('cuda_visible_device') is None:
        raise ValueError('Runtime requires fresh concrete lease UUID and selector; no index inference')
    job=work/'.local/jobs'/config['job_id'];job.mkdir(parents=True,exist_ok=False)
    # First and only NVML import/query is in this separately authorized GPU branch.
    records=[];violations=[];host_pid=None;child=None;code=None;error=None;launch=None;identity=None
    stream=(job/'monitor.jsonl').open('x');nv=None;initialized=False;handle=None
    def snapshot(stage):
        apps=nv.nvmlDeviceGetComputeRunningProcesses(handle)
        memory=nv.nvmlDeviceGetMemoryInfo(handle,version=nv.nvmlMemory_v2)
        util=nv.nvmlDeviceGetUtilizationRates(handle)
        row={'utc':utc(),'monotonic_ns':time.monotonic_ns(),'stage':stage,
            'GPU_UUID':nv.nvmlDeviceGetUUID(handle),'physical_index':nv.nvmlDeviceGetIndex(handle),
            'memory_used_bytes':memory.used,'memory_reserved_bytes':memory.reserved,
            'GPU_utilization_pct':util.gpu,'memory_utilization_pct':util.memory,
            'SM_clock_MHz':nv.nvmlDeviceGetClockInfo(handle,nv.NVML_CLOCK_SM),
            'memory_clock_MHz':nv.nvmlDeviceGetClockInfo(handle,nv.NVML_CLOCK_MEM),
            'power_mW':nv.nvmlDeviceGetPowerUsage(handle),
            'compute_processes':[{'host_pid':p.pid,'memory_used_bytes':p.usedGpuMemory} for p in apps]}
        stream.write(json.dumps(row)+'\n');stream.flush();records.append(row);return row
    def idle(row):return not row['compute_processes'] and row['memory_used_bytes']==0 and row['GPU_utilization_pct']==0
    def stop_owned():
        if child and child.poll() is None:
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=15)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait(timeout=15)
    try:
        import pynvml as nv
        nv.nvmlInit();initialized=True
        handle=nv.nvmlDeviceGetHandleByUUID(config['GPU_UUID'])
        before=[snapshot('prelaunch_1')];time.sleep(.25);before.append(snapshot('prelaunch_2'))
        if not all(idle(row) for row in before):raise RuntimeError('Strict idle prelaunch failed; no child launched')
        pci=nv.nvmlDeviceGetPciInfo(handle)
        observed={'uuid':nv.nvmlDeviceGetUUID(handle),'nvml_index':nv.nvmlDeviceGetIndex(handle),'pci_bus_id':pci.busId}
        identity,env=apply_selector(config,observed)
        identity.update(query_api='NVML GetHandleByUUID/GetUUID/GetIndex/GetPciInfo',timestamp_utc=utc(),
            NVML_version=nv.nvmlSystemGetNVMLVersion(),driver_version=nv.nvmlSystemGetDriverVersion(),
            python_binding={'path':str(Path(nv.__file__).resolve()),'sha256':sha(nv.__file__)},
            loaded_library_paths=sorted({line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines()
                if 'libnvidia-ml.so' in line and '/' in line}),
            monitor_source={'path':str(Path(__file__).resolve()),'sha256':sha(__file__)},config_sha256=sha(args.config))
        write_new(job/'device-identity.json',identity)
        env['FLAGMEGA_DEVICE_IDENTITY_PROOF']=str(job/'device-identity.json');env['PYTHONDONTWRITEBYTECODE']='1'
        with (job/'stdout.log').open('wb') as out,(job/'stderr.log').open('wb') as err:
            child=subprocess.Popen(config['argv'],cwd=work,env=env,stdout=out,stderr=err,start_new_session=True)
            ticks=int(Path('/proc',str(child.pid),'stat').read_text().rsplit(')',1)[1].split()[19])
            launch={'job_id':config['job_id'],'backend':config['backend'],'local_child_PID':child.pid,
                'local_child_start_ticks':ticks,'owned_process_group':child.pid,'argv':config['argv'],
                'configured_environment':config['environment'],
                'launched_environment':{k:env[k] for k in (*config['environment'],'FLAGMEGA_DEVICE_IDENTITY_PROOF','PYTHONDONTWRITEBYTECODE')},
                'physical_gpu_uuid':config['GPU_UUID'],'identity':identity,'started_utc':utc(),
                'source_pins':config['source_pins'],'runner_sha256':sha(__file__)}
            write_new(job/'launch.json',launch);print(json.dumps(launch),flush=True)
            start=progress=time.monotonic()
            while child.poll() is None:
                row=snapshot('workload_live');pids=[p['host_pid'] for p in row['compute_processes']]
                wrong=len(pids)>1
                if len(pids)==1:
                    if host_pid is None:host_pid=pids[0]
                    elif host_pid!=pids[0]:wrong=True
                if wrong:
                    violations.append(row);error='Multiple/changed NVML host PID epoch; stopped own child';stop_owned();break
                if time.monotonic()-start>config.get('timeout_seconds',1200):
                    error='Bounded native runtime timeout';stop_owned();break
                if time.monotonic()-progress>30:
                    print(json.dumps({'job_id':config['job_id'],'elapsed_seconds':time.monotonic()-start,
                        'host_PID_epoch':host_pid,'used_bytes':row['memory_used_bytes'],'reserved_bytes':row['memory_reserved_bytes']}),flush=True)
                    progress=time.monotonic()
                time.sleep(.25)
            code=child.wait()
        for index in range(24):
            row=snapshot('post_terminal')
            if index>=1 and idle(row) and idle(records[-2]):break
            time.sleep(.25)
        if not all(idle(row) for row in records[-2:]):error=error or 'Post-terminal idle gate failed'
    except BaseException as caught:
        error=error or str(caught);stop_owned();code=child.poll() if child else None
        (job/'runner-error.txt').write_text(traceback.format_exc())
        try:snapshot('error_terminal')
        except BaseException:pass
    finally:
        stream.close()
        if initialized:nv.nvmlShutdown()
    post=records[-1] if records else {}
    receipt={'schema':'flagmega.actual-owned-GPU-job/v3','agent_name':'graph profiler','agent_id':config['agent_id'],
        'assignment_id':config['assignment_id'],'job_id':config['job_id'],'backend':config['backend'],
        'config':str(args.config),'config_sha256':sha(args.config),'launch':launch,'identity':identity,
        'exit_code':code,'terminal':True,'status':'completed' if code==0 and not error else 'failed','error':error,
        'GPU_UUID':config['GPU_UUID'],'GPU_before':records[:2],'GPU_after':post,
        'prelaunch_idle':len(records)>=2 and all(idle(row) for row in records[:2]),
        'post_terminal_idle':len(records)>=2 and all(idle(row) for row in records[-2:]),
        'NVML_poll_seconds':.25,'NVML_samples':len(records),'observed_host_PID_epoch':host_pid,
        'observed_host_PIDs':sorted({p['host_pid'] for row in records for p in row['compute_processes']}),
        'contention_samples':violations,
        'PID_attribution':'Local child PID/start/process group separate from stable lone NVML host PID epoch; no literal equality',
        'sampling_limit':'0.25s samples cannot exclude sub-tick/graphics/CPU activity',
        'formal_three_rotated_performance_comparison':False,'monitor_sha256':sha(job/'monitor.jsonl'),
        'stdout_sha256':sha(job/'stdout.log') if child else None,'stderr_sha256':sha(job/'stderr.log') if child else None,
        'ended_utc':utc()}
    write_new(job/'receipt.json',receipt);print(json.dumps({k:receipt[k] for k in ('job_id','status','exit_code','error','post_terminal_idle')}),flush=True)
    return 0 if receipt['status']=='completed' else 1


if __name__=='__main__':raise SystemExit(main())
