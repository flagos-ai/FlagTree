"""Validate every prompt, reproduce both engine orders, retain every formal sample."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

from local_optimizations.scenarios import DEFAULT_LENGTHS, validate_lengths


def main():
    p=argparse.ArgumentParser()
    for name in ('trial','validation','checkpoint','sdk','native-python','native-triton-cache','native-inductor-cache','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--prompt-lengths',type=int,nargs='+',default=DEFAULT_LENGTHS)
    a=p.parse_args();validate_lengths(a.prompt_lengths)
    trial=a.trial.resolve();out=a.output.resolve();out.mkdir(exist_ok=False)
    here=trial/'local_optimizations'
    files=json.loads((trial/'source-identities-v3.json').read_text())['files']
    def unchanged():return all(hashlib.sha256(Path(k).read_bytes()).hexdigest()==v for k,v in files.items())
    assert unchanged(),'Validated candidate sources changed'
    criteria=dict(prompt_lengths=a.prompt_lengths,two_complete_orders_required=True,
        each_flag_leg_median_must_beat_each_native_leg_median=True,
        all40_formal_samples_per_scenario_and_runtime_retained=True,
        criterion_recorded_before_timing=True)
    (out/'acceptance-criteria.json').write_text(json.dumps(criteria,indent=2)+'\n')
    env=dict(os.environ,BASH_ENV='/dev/null',ENV='/dev/null',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',GLOO_SOCKET_IFNAME='lo',FLAGCX_MODULE_PATH=str(a.sdk.resolve()))
    env.pop('TRITON_PTXAS_PATH',None);env.pop('TRITON_CACHE_DIR',None)
    receipts=[];comparisons=[]
    for name,controller in [('validation','compare_native.py'),('NFFN','compare_native.py'),('FNNF','compare_reverse.py')]:
        cmd=[sys.executable,str(here/controller),'--output',str(out/name),
             '--prompt-lengths',*[str(v) for v in a.prompt_lengths]]
        if name=='validation':cmd.append('--validate-only')
        else:cmd.extend(['--scenario-validation',str(out/'validation')])
        for option in ('trial','validation','checkpoint','sdk','native-python','native-triton-cache','native-inductor-cache'):
            value=getattr(a,option.replace('-','_'))
            cmd.extend(['--'+option,str(value.absolute() if option=='native-python' else value.resolve())])
        start=time.monotonic()
        print(json.dumps(dict(event='starting',order=name)),flush=True)
        with (out/(name+'.log')).open('w') as log:
            child=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT);code=child.wait()
        receipt=dict(order=name,pid=child.pid,actual_exit_code=code,real_child_wait=True,seconds=time.monotonic()-start,timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_inputs_unchanged=unchanged())
        receipts.append(receipt);(out/'receipts.json').write_text(json.dumps(receipts,indent=2)+'\n')
        print(json.dumps(receipt),flush=True)
        if code:raise SystemExit(code)
        assert receipt['source_inputs_unchanged']
        if name=='validation':continue
        c=json.loads((out/name/'comparison.json').read_text())
        assert c['hardware_acceptance'] and c['all_scenario_tokens_passed'] and c['compute_body_five_op_gate_passed']
        assert len(c['rows'])==4*len(a.prompt_lengths)
        comparisons.append(c)
    launches=[json.loads((out/name/'launch.json').read_text()) for name in ('NFFN','FNNF')]
    assert {g['uuid'] for g in launches[0]['cards']}=={g['uuid'] for g in launches[1]['cards']},'Both orders must use the same four cards'
    assert all(v['trial']==str(trial) for v in launches)
    assert launches[0]['validation']['source_sha256']==launches[1]['validation']['source_sha256']
    scenarios=[]
    for length in a.prompt_lengths:
        series={'vllm':[],'flagmega':[]};legs={'vllm':[],'flagmega':[]};rounds=[]
        for c in comparisons:
            local={'vllm':[],'flagmega':[]}
            for row in c['rows']:
                if row['prompt_length']!=length:continue
                assert row['guards']['passed'] and row['correctness'] and row['cuda_graph']
                samples=row['samples_ms'];assert len(samples)==10 and statistics.median(samples)==row['median_decode_tpot_ms']
                runtime=row['runtime']
                series[runtime].extend(samples);legs[runtime].append(row['median_decode_tpot_ms']);local[runtime].append(row['median_decode_tpot_ms'])
            point={k:statistics.median(v for r in c['rows'] if r['prompt_length']==length and r['runtime']==k for v in r['samples_ms']) for k in series}
            rounds.append(dict(phase_order=c['phase_order'],point_estimate_faster=point['flagmega']<point['vllm'],all_flag_legs_below_all_native_legs=max(local['flagmega'])<min(local['vllm']),leg_medians=local))
        assert all(len(v)==40 for v in series.values())
        medians={k:statistics.median(v) for k,v in series.items()}
        checks=dict(both_independent_rounds_faster=all(r['point_estimate_faster'] for r in rounds),both_orders_faster_without_bracket_overlap=all(r['all_flag_legs_below_all_native_legs'] for r in rounds),all_four_flag_leg_medians_below_all_four_native_leg_medians=max(legs['flagmega'])<min(legs['vllm']),pooled_median_faster=medians['flagmega']<medians['vllm'])
        scenarios.append(dict(prompt_length=length,rounds=rounds,all_formal_samples_ms=series,
            total_formal_requests_per_runtime=40,all_leg_medians=legs,median_decode_tpot_ms=medians,
            pooled_latency_reduction_percent=100*(1-medians['flagmega']/medians['vllm']),
            stable_checks=checks,stable_native_win_established=all(checks.values())))
    result=dict(scenarios=scenarios,no_formal_samples_removed=True,receipts=receipts,
        source_inputs_unchanged=unchanged(),scope=comparisons[0]['scope'],
        performance_acceptance=all(s['stable_native_win_established'] for s in scenarios))
    (out/'joint-acceptance.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='scenarios'}),flush=True)


if __name__=='__main__':main()
