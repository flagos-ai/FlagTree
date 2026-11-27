"""Serialize matched ordinary measurements; preserve all raw runs and failures."""
import argparse, datetime, json, os, statistics, subprocess, sys, traceback
from pathlib import Path
from reference_identity import file_pin, save_json, validate_build
from timing_isolation import POLICY_SCHEMA, POLICY_ID, LIMITS, clock_identity, monotonic_ns, validate_round_result, classify_rounds
B=Path(__file__).resolve().parent; W=B.parent; REPO=W.parents[4]
SOURCE=Path('/home/sunnycase/.cache/flagtree-cpp-qwen38/reference-src/llama.cpp-5ce06fc7')
BUILD=Path('/home/sunnycase/.cache/flagtree-cpp-qwen38/reference-builds/20261105T031000Z-cpp-decode/rebuild-r1')
def execute(command,out,policy):
    out.parent.mkdir(parents=True,exist_ok=True)
    cmd=[sys.executable,str(B/'monitor.py'),'--output',str(out),'--policy',str(policy),'--',*map(str,command)]
    with out.with_suffix('.launcher.log').open('x') as f: result=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT)
    print(f'{out.name}: exit={result.returncode}',flush=True);return result.returncode

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--artifact',type=Path,required=True);p.add_argument('--repair-synthetic',action='store_true');a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=True); task=json.loads((W/'.local/task.yaml').read_text())
    env=dict(PYTHONPATH=str(REPO/'python'),PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='0',TRITON_CACHE_DIR=str(W/'.local/cache/kernels'))
    os.environ.update(env)
    uuid=subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
    policy=a.output/'policy.json'
    if policy.exists() and not a.repair_synthetic: raise ValueError('Use fresh run directory')
    if not a.repair_synthetic: save_json(policy,dict(schema=POLICY_SCHEMA,id=POLICY_ID,limits=LIMITS,clock=clock_identity(),created_ns=str(monotonic_ns()),target_gpu_uuid=uuid))
    receipt=validate_build(SOURCE,BUILD)
    save_json(a.output/('reference-repair-admission.json' if a.repair_synthetic else 'reference-admission.json'),dict(admission=receipt,old_source_head=subprocess.check_output(['git','-C',str(Path('/home/sunnycase/.cache/flagtree-cpp-qwen38/reference/llama.cpp')),'rev-parse','HEAD'],text=True).strip(),restoration='4070 tracked byte identities match pinned tree; historical source path preserved in build receipt'))
    contract=dict(schema='flagmega.matched-protocol/v1',task=file_pin(W/'.local/task.yaml'),artifact=file_pin(a.artifact/'artifact.json'),device_uuid=uuid,
                  processes=2,rounds=3,warmup_rounds=1,order=['triton-p0','llama-p0','llama-p1','triton-p1'],
                  scenario_interpretation='32/64 decode-1 calls; prompt tokens are injected before clock starts, continuation follows each backend own tokens; first 10 generated tokens start at max(prompt_count-1,0)',
                  uncertainty='Per-process means, sample SD across process means, min/max range and t(df=1) 95% CI; paired process differences retained',
                  ordinary_instrumented_separate=True,clock=clock_identity(),env=env)
    save_json(a.output/('repair-protocol.json' if a.repair_synthetic else 'protocol.json'),dict(contract,actual_repair_order=['triton-p2','llama-p2'],selection='synthetic paired p1(L,T) and p2(T,L); failed p0 and llama p0 retained as excluded/control') if a.repair_synthetic else contract)
    checkpoint=task['workload']['checkpoint']; results=json.loads((a.output/'run-index.json').read_text()) if a.repair_synthetic else []
    if a.repair_synthetic:
        save_json(a.output/'run-index-before-repair.json',results)
        save_json(a.output/'summary-before-repair.json',json.loads((a.output/'summary.json').read_text()))
    for scenario,spec in task['workload']['scenarios'].items():
        if a.repair_synthetic: continue
        golden=a.output/'golden'/scenario
        cmd=[sys.executable,B/'llama_golden.py','--llama-source',SOURCE,'--llama-build',BUILD,'--checkpoint',checkpoint,'--output',golden,'--steps',str(spec['steps']),'--repeats','1']
        if 'prompt' in spec: cmd+=['--prompt',spec['prompt']]
        rc=execute(cmd,a.output/'monitors'/('golden-'+scenario),policy)
        if rc: raise RuntimeError(f'Golden failed {scenario}; evidence retained')
    for scenario,spec in task['workload']['scenarios'].items():
        if a.repair_synthetic and scenario!='synthetic32': continue
        for backend,process in ([('triton',2),('llama',2)] if a.repair_synthetic else [('triton',0),('llama',0),('llama',1),('triton',1)]):
            run=a.output/'ordinary'/f'{scenario}-{backend}-p{process}'; run.mkdir(parents=True)
            if backend=='triton':
                result=run/'result.json';cmd=[sys.executable,B/'benchmark_model.py','--artifact',a.artifact,'--golden',a.output/'golden'/scenario,'--scenario',scenario,'--steps',str(spec['steps']),'--output',result]
            else:
                result=run/'native/metadata.json';cmd=[sys.executable,B/'llama_performance.py','--llama-source',SOURCE,'--llama-build',BUILD,'--checkpoint',checkpoint,'--output',run/'native','--steps',str(spec['steps']),'--repeats','3']
                if 'prompt' in spec:cmd+=['--prompt',spec['prompt']]
            monitor=a.output/'monitors'/f'{scenario}-{backend}-p{process}'
            rc=execute(cmd,monitor,policy)
            row=dict(scenario=scenario,backend=backend,process=process,exit=rc,result=str(result),monitor=str(monitor))
            if result.exists():
                r=json.loads(result.read_text())
                try:
                    raw=(result.with_suffix('.rounds.jsonl')).read_bytes() if backend=='triton' else None
                    validate_round_result(r,rounds_raw=raw)
                    mr=json.loads((monitor/'monitor.json').read_text()); telemetry=(monitor/'telemetry.jsonl').read_bytes()
                    isolation=classify_rounds(r,mr,[json.loads(x) for x in telemetry.splitlines()],telemetry_raw=telemetry,policy_raw=(monitor/'isolation-policy.json').read_bytes(),rounds_raw=raw)
                    save_json(run/'isolation.json',isolation)
                    row.update(mean_ms=r['mean_ms'],isolation_clear=isolation['all_measured_rounds_sampled_clear'],result_pin=file_pin(result),numerical_status=r.get('numerical_status'))
                    if backend=='llama':
                        g=json.loads((a.output/'golden'/scenario/'metadata.json').read_text());assert r['tokens']==g['tokens'], 'GPU sampled reference differs from golden'
                        row['golden_trajectory_equal']=True
                except Exception as e: row.update(audit_error=repr(e),traceback=traceback.format_exc())
            results.append(row);save_json(a.output/'run-index.json',results)
            print(json.dumps(row,ensure_ascii=False),flush=True)
    groups=[]
    for scenario in task['workload']['scenarios']:
        for backend in ['triton','llama']:
            rows=[r for r in results if r['scenario']==scenario and r['backend']==backend and 'mean_ms' in r and (not a.repair_synthetic or scenario!='synthetic32' or r['process'] in (1,2))]
            means=[r['mean_ms'] for r in rows]
            group=dict(scenario=scenario,backend=backend,process_mean_ms=means,selected_processes=[r['process'] for r in rows],measurement_ids=[f"{scenario}-{backend}-p{r['process']}" for r in rows],status='measured-unreviewed' if len(means)==2 else 'incomplete')
            if len(means)==2:
                mean=statistics.mean(means);sd=statistics.stdev(means); ci=12.706204736*sd/(2**.5)
                group.update(mean_ms=mean,process_min_ms=min(means),process_max_ms=max(means),process_spread_ms=max(means)-min(means),process_sd_ms=sd,ci95_half_ms=ci,
                             bandwidth_utilization=task['secondary_metric']['bytes_per_token']/(mean*1e-3*task['secondary_metric']['device_peak_bytes_per_second']),
                             sampled_isolation_clear=all(r.get('isolation_clear') for r in rows))
            groups.append(group)
    save_json(a.output/'summary.json',dict(schema='flagmega.matched-summary/v1',cohort='ordinary',groups=groups,runs=results,performance_accepted=False,reviewer_pending=True))
    print(json.dumps(groups,indent=2),flush=True)
if __name__=='__main__':main()
