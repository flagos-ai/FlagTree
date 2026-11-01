"""Summarize complete request histories for every prompt length and engine leg."""
import argparse
import json
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from local_optimizations.scenarios import report_scenarios, reference_scenarios


def analyse(root):
    assert (root/'DONE').is_file(), 'Only completed measurements can be summarized'
    launch=json.loads((root/'launch.json').read_text())
    phases=launch['order']
    uuids={c['uuid'] for c in launch['cards']};keeper=launch['keeper']
    trial=Path(launch['trial'])
    warmup=json.loads((trial/'warmup.json').read_text())
    entries={e['symbol']:e['hash'] for e in warmup['entries']}
    native_reference=json.loads((root/'N0/report.json').read_text())
    lengths=launch.get('prompt_lengths',[1])
    expected=reference_scenarios(native_reference,lengths)
    rows=[]
    for label in phases:
        output=root/label;native=label.startswith('N')
        reports=([json.loads((output/'report.json').read_text())] if native else
                 [json.loads((output/f'rank-{rank}.json').read_text()) for rank in range(4)])
        assert all(r['passed'] for r in reports)
        if native:
            assert reports[0]['enforce_eager'] is False and reports[0]['prefix_caching'] is False
            assert 'NONE' not in reports[0]['configuration']['cudagraph_mode'].upper()
        else:
            assert all(r['cuda_graph'] and len(r['resources'])==228 and all(k['spill_bytes']==0 for k in r['resources']) for r in reports)
            assert all(r['source_sha256']==warmup['source_sha256'] and r['compiled_entries']==entries for r in reports)
        grouped=[{s['prompt_length']:s for s in report_scenarios(r)} for r in reports]
        assert all(set(s)==set(lengths) for s in grouped)
        telemetry=[json.loads(line) for line in (output/'gpu-monitor.jsonl').read_text().splitlines()]
        for length in lengths:
            scenarios=[s[length] for s in grouped]
            assert all(s['prompt']==expected[length]['prompt'] and s['output_tokens']==8 and len(s['requests'])==12 for s in scenarios)
            assert all(all(q['tokens']==expected[length]['requests'][0]['tokens'] and q['request']==i and q['warmup']==(i<2) for i,q in enumerate(s['requests'])) for s in scenarios)
            samples=[max(s['requests'][i]['decode_tpot_ms'] for s in scenarios) for i in range(2,12)]
            begin=min(s['requests'][2]['wall_start_utc'] for s in scenarios)
            end=max(s['requests'][11]['wall_end_utc'] for s in scenarios)
            actual=[record for record in telemetry if begin-.1<=record['timestamp']<=end+.1]
            cards=[g for record in actual for g in record['gpus'] if g['uuid'] in uuids]
            guards=dict(samples=len(actual),observed_all_cards=bool(actual) and len(cards)==4*len(actual),
                process_isolation=bool(cards) and all(keeper in g['pids'] and len(g['pids'])==2 for g in cards),
                no_thermal_slowdown=bool(cards) and all(g['thermal']=='Not Active' for g in cards),
                observed_sm_mhz=sorted({g['sm_mhz'] for g in cards}),
                expected_sm_clock=bool(cards) and all(g['sm_mhz']==1980 for g in cards),
                max_temperature=max((g['temperature'] for g in cards),default=None))
            guards['passed']=all(guards[k] for k in ('observed_all_cards','process_isolation','no_thermal_slowdown','expected_sm_clock'))
            row=dict(label=label,prompt_length=length,runtime='vllm' if native else 'flagmega',median_decode_tpot_ms=statistics.median(samples),
                samples_ms=samples,correctness=True,cuda_graph=True,guards=guards,
                initialization_seconds=[r.get('engine_initialization_seconds',r.get('initialization_seconds')) for r in reports])
            if not native:
                assert all(s['validation_requests'] for s in scenarios)
                device=[max(statistics.mean(s['requests'][i]['device_step_ms_with_state_carry'][1:]) for s in scenarios) for i in range(2,12)]
                row['median_device_ms_with_state_carry']=statistics.median(device)
            rows.append(row)
    compile_report=json.loads((trial/'dag-audit.json').read_text())
    assert compile_report['semantic_hash']==warmup['semantic_hash']
    structural=not any(compile_report['compute_body_five_op_counts'].values());assert structural
    summaries=[]
    for length in lengths:
        selected=[r for r in rows if r['prompt_length']==length]
        native_rows=[r for r in selected if r['runtime']=='vllm']
        flag_rows=[r for r in selected if r['runtime']=='flagmega']
        native=statistics.median(v for r in native_rows for v in r['samples_ms'])
        flagmega=statistics.median(v for r in flag_rows for v in r['samples_ms'])
        summaries.append(dict(prompt_length=length,native_decode_tpot_ms=native,flagmega_decode_tpot_ms=flagmega,
            native_leg_drift_percent=100*(native_rows[-1]['median_decode_tpot_ms']/native_rows[0]['median_decode_tpot_ms']-1),
            flagmega_latency_lower=flagmega<native))
    report=dict(scope=launch['scope'],phase_order=phases,cards=launch['cards'],all_official8_passed=True,
        all_scenario_tokens_passed=True,hardware_acceptance=all(r['guards']['passed'] for r in rows),
        compute_body_five_op_gate_passed=structural,scenarios=summaries,rows=rows,
        no_formal_samples_removed=True,performance_acceptance=all(s['flagmega_latency_lower'] for s in summaries))
    if lengths==[1]:report.update(summaries[0])
    (root/'comparison.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('rows','cards')}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    analyse(parser.parse_args().output.resolve())
