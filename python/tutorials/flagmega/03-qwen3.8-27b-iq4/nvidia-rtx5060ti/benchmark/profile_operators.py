"""Separate instrumented cohort: source-bound TIR caller envelopes for every CTA."""
import argparse, ast, importlib.util, json, math, statistics, time, traceback
from pathlib import Path
from reference_identity import file_pin, save_json
from timing_isolation import clock_identity, monotonic_ns
from benchmark_model import DecodeContext
B=Path(__file__).resolve().parent

def instrument(artifact,out):
    source=artifact/'generated_kernels.py'; text=source.read_text(); tree=ast.parse(text);lines=text.splitlines();sites=[];edits=[]
    tir=artifact/'ir/final.py';tir_lines=tir.read_text().splitlines()
    import re
    tir_calls={}
    for i,line in enumerate(tir_lines,1):
        match=re.search(r"call_id='([^']+)'",line)
        if match:tir_calls.setdefault(match[1],[]).append(i)
    functions=[n for n in tree.body if isinstance(n,ast.FunctionDef) and (n.name=='flagmega_main' or n.name.startswith('_flagmega_function_'))]
    assert len(functions)==3, 'Only actual main and two consumer functions expected'
    for fn in functions:
        # The frozen main calls exactly 48 GDN + 16 attention reusable functions.
        end_sig=next(i for i in range(fn.lineno-1,fn.body[0].lineno-1) if lines[i].strip()=='):')
        edits.append((end_sig,0,'    , _gp_trace'))
        for node in ast.walk(fn):
            if not isinstance(node,ast.Expr) or not isinstance(node.value,ast.Call):continue
            call=node.value
            if not isinstance(call.func,ast.Name):continue
            name=call.func.id
            if name.startswith('_flagmega_function_'):
                i=node.lineno-1;assert node.lineno==node.end_lineno
                old=lines[i];edits.append((i,1,old.rstrip()[:-1]+', _gp_trace)'));continue
            if not name.startswith('_flagmega_') or '_call_' not in name:continue
            i=node.lineno-1;assert node.lineno==node.end_lineno
            comment=next((lines[j].strip() for j in range(i-1,max(fn.lineno,i-6)-1,-1) if '# tir.call ' in lines[j]),None)
            assert comment, (name,node.lineno)
            role='consumer';function=fn.name;idx=len(sites)
            if 'linear' in function:layer='(decode_linear_0_layer_id + decode_linear_0_layer_id // 3)';instances=[j+j//3 for j in range(48)]
            elif 'attention' in function:layer='(decode_attention_1_layer_id * 4 + 3)';instances=[j*4+3 for j in range(16)]
            else:layer='64';instances=[64]
            # Retain exact ancestry predicates for expected executed CTA coverage.
            parents=[]
            def find(root,path):
                if root is node:return path
                for child in ast.iter_child_nodes(root):
                    got=find(child,path+[root])
                    if got is not None:return got
            parents=find(fn,[])
            predicates=[ast.unparse(n.test) for n in parents if isinstance(n,ast.If)]
            call_id=comment.split(' -> ')[0].replace('# tir.call ','')
            assert len(tir_calls.get(call_id,[]))==1, f'TIR call identity ambiguous/missing {call_id}'
            sites.append(dict(site=idx,symbol=name,function=function,call_id=call_id,callsite=comment,source_link=dict(path=str(tir),line=tir_calls[call_id][0]),generated_source_link=dict(path=str(source),line=node.lineno),role=role,instances=instances,predicates=predicates))
            indent=lines[i][:len(lines[i])-len(lines[i].lstrip())]
            mark=indent+f'_gp_mark(_gp_trace, ({layer}) * _GP_SITES + {idx}, '
            edits.append((i,0,indent+'tl.debug_barrier()\n'+mark+'0)'))
            edits.append((i+1,0,indent+'tl.debug_barrier()\n'+mark+'1)'))
    main=next(f for f in functions if f.name=='flagmega_main')
    edits.append((main.body[0].lineno-1,0,'    _gp_mark(_gp_trace, 64 * _GP_SITES + '+str(len(sites))+', 0)'))
    edits.append((main.end_lineno,0,'    tl.debug_barrier()\n    _gp_mark(_gp_trace, 64 * _GP_SITES + '+str(len(sites))+', 1)'))
    for i,delete,value in sorted(edits,key=lambda x:(x[0],x[1]),reverse=True):lines[i:i+delete]=value.splitlines()
    helper='''
@triton.jit
def _gp_mark(ptr, occurrence, point: tl.constexpr):
    stamp = tl.inline_asm_elementwise("mov.u64 $0, %globaltimer;", constraints="=l", args=[], dtype=tl.uint64, is_pure=False, pack=1)
    sm = tl.inline_asm_elementwise("mov.u32 $0, %smid;", constraints="=r", args=[], dtype=tl.uint32, is_pure=False, pack=1)
    slot = (occurrence.to(tl.int64) * 32 + tl.program_id(0)) * 4
    tl.store(ptr + slot + point, stamp)
    tl.store(ptr + slot + 2 + point, sm.to(tl.uint64))
'''
    # main root layer is a Python literal; normalize to scalar i32 for helper.
    helper=helper.replace('occurrence.to(tl.int64)','tl.full((), occurrence, tl.int32).to(tl.int64)' ) if False else helper
    generated='\n'.join(lines)+'\n';generated=generated.replace('occurrence.to(tl.int64)','tl.cast(occurrence, tl.int64)')
    helper=helper.replace('occurrence.to(tl.int64)','tl.cast(occurrence, tl.int64)')
    insert=generated.index('@triton.jit');generated=generated[:insert]+f'_GP_SITES = tl.constexpr({len(sites)+1})\n'+helper+'\n'+generated[insert:]
    path=out/'instrumented_kernels.py';path.write_text(generated);ast.parse(generated)
    save_json(out/'identity-map.json',dict(base_source=file_pin(source),instrumented_source=file_pin(path),sites=sites,record_layout='uint64[65,site,32,4]: start,end,SM_at_start,SM_at_end; root=layer64',roles='Only real consumer execution; no producer/warp-specialized schedule in frozen generated source'))
    with path.open('a') as stream: stream.write('''
@triton.jit
def _gp_clock_probe(ptr):
    cta = tl.program_id(0)
    sm = tl.inline_asm_elementwise("mov.u32 $0, %smid;", constraints="=r", args=[], dtype=tl.uint32, is_pure=False, pack=1)
    tl.store(ptr + cta * 132, sm.to(tl.uint64))
    tle.distributed_barrier(FLAGMEGA_GRID_MESH)
    for i in tl.range(0, 128, loop_unroll_factor=1):
        now = tl.inline_asm_elementwise("mov.u64 $0, %globaltimer;", constraints="=l", args=[], dtype=tl.uint64, is_pure=False, pack=1)
        tl.store(ptr + cta * 132 + i + 1, now)
    start = tl.inline_asm_elementwise("mov.u64 $0, %globaltimer;", constraints="=l", args=[], dtype=tl.uint64, is_pure=False, pack=1)
    now = start
    while now - start < 20000000:
        now = tl.inline_asm_elementwise("mov.u64 $0, %globaltimer;", constraints="=l", args=[], dtype=tl.uint64, is_pure=False, pack=1)
    tl.store(ptr + cta * 132 + 129, start)
    tl.store(ptr + cta * 132 + 130, now)
    tle.distributed_barrier(FLAGMEGA_GRID_MESH)
    stop = tl.inline_asm_elementwise("mov.u64 $0, %globaltimer;", constraints="=l", args=[], dtype=tl.uint64, is_pure=False, pack=1)
    tl.store(ptr + cta * 132 + 131, stop)
''')
    identity=json.loads((out/'identity-map.json').read_text());identity['instrumented_source']=file_pin(path);identity['TIR_source']=file_pin(artifact/'ir/final.py');save_json(out/'identity-map.json',identity)
    return path,sites

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--artifact',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--golden',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    import shutil
    (a.output/'harness').mkdir()
    for name in ['profile_operators.py','benchmark_model.py','timing_isolation.py','reference_identity.py','clock_correlation.py']:
        shutil.copy2(B/name,a.output/'harness'/name)
    report=dict(schema='flagmega.graph-profile/v1',cohort='instrumented-TIR-envelopes',status='running',artifact=file_pin(a.artifact/'artifact.json'),host_clock=clock_identity(),harness_pins=[file_pin(p) for p in sorted((a.output/'harness').glob('*.py'))])
    c=None
    try:
        path,sites=instrument(a.artifact,a.output)
        import torch, numpy as np
        from triton.flagmega.runtime.prepared import PreparedKernel
        from triton.flagmega.runtime.module import compilation_knobs, compilation_options
        torch.set_num_threads(1);c=DecodeContext(a.artifact)
        original=c.runtime._prepared[0];spec=importlib.util.spec_from_file_location('gp_capture',path);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
        trace=torch.zeros((65,len(sites)+1,32,4),dtype=torch.uint64,device='cuda')
        arguments=(*original._arguments,trace)
        with compilation_knobs('nvidia-sm120'):
            compiled=mod.flagmega_main.run(*arguments,grid=original.grid,warmup=True,**compilation_options(original.contract,'nvidia-sm120',floating_point_contract=c.runtime.ir_module.metadata.get('floating_point_contract')))
        compiled._init_handles()
        profiled=PreparedKernel(compiled,arguments,(),grid=original.grid,contract=original.contract,allocation_device='cuda:0')
        report.update(ordinary_resources=original.resource_report,instrumented_resources=profiled.resource_report,trace_bytes=trace.numel()*8)
        save_json(a.output/'manifest.json',report)
        def call(instrumented):
            c.values['paged_state.slot_mapping'].copy_(c.values['paged_state.seq_lens'])
            if instrumented:profiled.launch(stream=c.stream.cuda_stream)
            else:c.runtime.run_into(*c.arguments)
            c.values['input_ids'].copy_(c.token.reshape_as(c.values['input_ids']))
        # Control/trace first 10 independently generated synthetic tokens; resets remain outside.
        outputs={};timings={}
        for label,enabled in [('ordinary-control',False),('instrumented',True)]:
            c.reset();c.stream.synchronize();seq=[];lat=[]
            for step in range(10):
                c.stream.synchronize();before=time.perf_counter();call(enabled);seq.append(int(c.token.item()));lat.append((time.perf_counter()-before)*1000)
                assert torch.isfinite(c.logits).all().item()
            outputs[label]=seq;timings[label]=lat;c.check_state(10)
        golden=json.loads((a.golden/'metadata.json').read_text())
        assert outputs['ordinary-control']==outputs['instrumented']==golden['tokens'][:10], 'Instrumented/control/golden prefix mismatch'
        report.update(control_tokens=outputs,control_ms=timings,instrumentation_overhead_ratio=statistics.mean(timings['instrumented'][1:])/statistics.mean(timings['ordinary-control'][1:]),finite=True,trajectory_equal=True)
        # One full real decode-1 at synthetic step 0. All CTAs are captured.
        from clock_correlation import ClockCorrelation
        correlation=ClockCorrelation(c)
        c.reset();trace.zero_();c.stream.synchronize();correlation.bracket('before-trace')
        begin=monotonic_ns();call(True);token=int(c.token.item());host_end=monotonic_ns();c.stream.synchronize()
        correlation.bracket('after-trace')
        raw=trace.cpu().numpy();np.save(a.output/'trace.npy',raw)
        spans=[];expected=0;actual=0
        for site in sites:
            for layer in site['instances']:
                for cta in range(32):
                    env=dict(shard_y=cta//8,shard_x=cta%8,shard_coord0=cta//8,shard_coord1=cta%8)
                    eligible=all(bool(eval(pred,{'__builtins__':{}},env)) for pred in site['predicates']);expected+=eligible
                    start,stop,sm0,sm1=map(int,raw[layer,site['site'],cta]);actual+=bool(start)
                    assert bool(start)==eligible, f'Missing/unexpected invocation {site} layer={layer} cta={cta}'
                    if not eligible:assert stop==0;continue
                    assert stop>=start and sm0==sm1
                    spans.append(dict(id=f"L{layer}-S{site['site']}-CTA{cta}",start_ns=start,end_ns=stop,cta=cta,SM=sm0,role='consumer',layer=layer if layer<64 else None,token=0,rank=0,
                                      site=site['site'],symbol=site['symbol'],callsite=site['callsite'],source_link=site['source_link'],kind='simt',evidence='inflight_envelope'))
        assert expected==actual
        save_json(a.output/'spans.json',spans)
        # Concrete TIR identity; union across CTAs and repeated layers, no overlap double-count.
        operators=[]
        for site in sites:
            rows=[s for s in spans if s['site']==site['site']];intervals=sorted((s['start_ns'],s['end_ns']) for s in rows);total=0;right=0
            for left,end in intervals:
                total+=max(0,end-max(left,right));right=max(right,end)
            operators.append(dict(label=site['callsite'].split(' -> ')[0].replace('# tir.call ',''),value=total/1e6,status='measured',measurement_ids=[a.output.name],span_ids=[s['id'] for s in rows],source_link=site['source_link'],symbol=site['symbol']))
        operators.sort(key=lambda r:r['value'],reverse=True);save_json(a.output/'operator-ranking.json',operators)
        # Validate nominal ns scale against CUDA events, repeated-grid timestamps,
        # actual timer granularity and observed CTA-to-SM mapping in this execution.
        probe=torch.zeros((32,132),dtype=torch.uint64,device='cuda')
        with compilation_knobs('nvidia-sm120'):
            cp=mod._gp_clock_probe.run(probe,grid=original.grid,warmup=True,**compilation_options(original.contract,'nvidia-sm120'))
        cp._init_handles()
        prep=PreparedKernel(cp,(probe,),(),grid=original.grid,contract=original.contract,allocation_device='cuda:0')
        calibration=[]
        for repeat in range(4):
            c.stream.synchronize();ev0=torch.cuda.Event(enable_timing=True);ev1=torch.cuda.Event(enable_timing=True)
            h0=monotonic_ns();ev0.record(c.stream);prep.launch(stream=c.stream.cuda_stream);ev1.record(c.stream);ev1.synchronize();h1=monotonic_ns()
            arr=probe.cpu().numpy();np.save(a.output/f'clock-probe-{repeat}.npy',arr)
            device_ns=int(arr[:,131].max())-int(arr[:,1].min());event_ns=ev0.elapsed_time(ev1)*1e6
            delta=[int(v) for v in np.diff(arr[:,1:129].astype(np.int64)).flatten() if v>0]
            granularity=math.gcd(*delta)
            assert .98 < device_ns/event_ns < 1.02, 'Globaltimer scale differs from CUDA-event ns'
            assert len(set(map(int,arr[:,0])))==32, 'Unexpected CTA-to-SM mapping in clock probe'
            calibration.append(dict(repeat=repeat,cuda_event_ns=event_ns,globaltimer_window_ns=device_ns,ratio=device_ns/event_ns,
                                    observed_granularity_ns=granularity,post_grid_barrier_start_skew_ns=int(arr[:,1].max())-int(arr[:,1].min()),
                                    host_bracket_ns=[h0,h1],raw=file_pin(a.output/f'clock-probe-{repeat}.npy')))
        save_json(a.output/'clock-calibration.json',dict(status='measured',common_clock='same-device PTX globaltimer',calibration=calibration,
                 alignment='validated_common device-only; host brackets recorded, no pointwise host correlation claim',
                 limitation='Caller envelopes observe CTA boundaries; no engine busy or warp activity claim. Granularity is observed, not a hardware resolution guarantee.'))
        roots=raw[64,len(sites),:,:];root_window=[int(roots[:,0].min()),int(roots[:,1].max())]
        window=[min(root_window[0],min(x['start_ns'] for x in spans)),max(root_window[1],max(x['end_ns'] for x in spans))]
        assert all(window[0]<=x['start_ns']<=x['end_ns']<=window[1] for x in spans)
        correlation_report=correlation.save(a.output,spans,window)
        
        report.update(status='captured-unreviewed',token=token,host_window_ns=[begin,host_end],expected_invocations=expected,captured_invocations=actual,coverage='every eligible TIR caller × layer × CTA for one full synthetic step0; consumer-only actual source',
                      raw_trace=file_pin(a.output/'trace.npy'),spans=file_pin(a.output/'spans.json'),clock=file_pin(a.output/'clock-calibration.json'),TIR_window_ns=window,root_markers_window_ns=root_window,window_rule='min/max of actual root and TIR boundary observations; no padding or fabricated endpoints',
                      host_device_correlation=file_pin(a.output/'host-device-correlation.json'),
                      open_gates=['reviewer independent raw-to-chart/resources/clock calibration','full ordinary GPU-graph profile overhead control'])
        save_json(a.output/'manifest.json',report)
        # Until calibration is measured, canonical full-model graph remains pending.
        provenance=[file_pin(a.output/'manifest.json'),file_pin(a.output/'spans.json'),file_pin(a.output/'identity-map.json'),file_pin(a.output/'clock-calibration.json'),file_pin(a.output/'host-device-correlation.json')]
        save_json(a.output/'figure-packet.json',dict(provenance=provenance,
            graph=dict(pending_only=False,scope='whole_model',invocation=a.output.name+'/synthetic32/step0',window_ns=window,timebase=dict(source='same-device PTX globaltimer; per-point host CLOCK_MONOTONIC correlation measured with bracket bounds',resolution_ns=min(r['observed_granularity_ns'] for r in calibration),alignment='validated_common',host_alignment='affine-bracketed',host_endpoint_bound_max_ns=correlation_report['max_endpoint_bound_width_ns']),rows=[dict(label=f'consumer CTA{cta} / SM{next(s["SM"] for s in spans if s["cta"]==cta)}',status='captured',intervals=[s for s in spans if s['cta']==cta]) for cta in range(32)]+[dict(label='producer: not used in actual source',status='not_used',intervals=[])]),
            operators=dict(pending_only=False,metric='Union of TIR caller envelopes',unit='ms',sort_by='value_desc',aggregation='Union of concrete callsite envelopes over 32 CTAs and all layers, same trace.npy; sums across operators are not model elapsed time.',rows=operators)))
        print(json.dumps(report,indent=2),flush=True)
    except BaseException as e:
        report.update(status='failed',error=repr(e),traceback=traceback.format_exc());save_json(a.output/'manifest.json',report);raise
    finally:
        if c:c.close()
if __name__=='__main__':main()
