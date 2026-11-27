"""Native, versioned GLU/Q8/down source scheduling on a single cyclic pipe."""
from copy import deepcopy
import os
from triton.flagmega.errors import CodegenError
ABI='H003ConsumerPartial-f32-3x32/bar3-count128/v9-continuous-stream'
def fuse_single(module, result):
    ps=result['programs'];ks=result['kernels']
    if len(ps)!=4 or [p['render_calls'][0]['family'] for p in ps]!=['q8_1_prepare','iq4_q8_glu','q8_1_prepare','iq4_q8_projection']:
        raise CodegenError('continuous stream requires exactly XQ, GLU, YQ, down')
    g=deepcopy(ps[1]);gc=g['render_calls'][0];dc=ps[3]['render_calls'][0];qc=ps[2]['render_calls'][0]
    if (gc['transport_n'],gc['transport_k'],dc['constexprs']['N'],dc['constexprs']['K'])!=(17408,5120,5120,17408):
        raise CodegenError('continuous stream fixed numerical shape mismatch')
    if not gc.get('no_shared_staging') and (gc['num_stages']!=2 or not gc['single_epoch'] or gc['barrier_batch_g']!=4):
        raise CodegenError('continuous stream requires qualified paired cap2/G4')
    if any(b['abi'].get('pool_scope_stride_bytes',0) for c in (gc,dc,qc) for side in ('inputs','outputs') for binding in c.get(side,()) for b in binding.get('buffers',())):
        raise CodegenError('continuous stream activation and weights must have CHIP backing')
    names=[n.strip() for n in gc['signature'].split(',')]
    extra=[n.strip() for n in dc['signature'].split(',') if n.strip() not in names]
    gc['signature']+=', '+', '.join(extra)
    gc.update(down_pair_pack=os.environ.get("DOWN_PAIR_PACK","0")=="1",continuous_stream=True,continuous_down_pointers=dc['pointers'],continuous_q8_pointers=qc['pointers'])
    aligned=os.environ.get('Q8_ALIGNED','0')=='1'
    if aligned:
        grid=gc['grid'][0]
        if os.environ.get('Q8_WORK_QUEUE')!='1' or grid not in (216,252,288,324): raise CodegenError('R22 requires resident grid216 work queue')
        gc.update(q8_boundary_trace=os.environ.get("Q8_BOUNDARY_TRACE","0")=="1",q8_local_end=os.environ.get("Q8_PREPARE_STAGE","incremental")=="local_end",q8_cyclic=os.environ.get("Q8_GROUP_ORDER","contiguous")=="cyclic",q8_work_queue=True,q8_aligned=True,q8_rows_per_cta=32,q8_groups_per_cta=1)
    abi='H003ConsumerPartial-f32-3x32/bar3-count128/v11-bounded-continuous' if gc.get('bounded_prefetch') else ABI
    if aligned: abi='H003ConsumerPartial-f32-3x32/bar3-count128/v13-q8-work-queue'
    if gc.get('no_shared_staging'): abi='H003ConsumerPartial-f32-3x32/bar3-count128/v26-global-pf2'
    gc['compile_options']['consumer_barrier_abi']=abi
    schedule=g['pipeline_schedule']
    # Native cyclic schedule replaces generic inter-stage arena drains.
    for role in ('producer','consumer'):
        schedule[role+'_events']=[e for e in schedule[role+'_events'] if e.get('kind')!='drain']
    for role in ('producer','consumer'):
        for e in schedule[role+'_events']:
            if e.get('call')==ps[1]['entry_events'][0]['call']:e['arguments']+=', '+', '.join(extra)
    # The original TIR GLU stage owns every physical slot. Down reuses these
    # exact endpoints at epoch GLU_count + down_row*5 + k_chunk.
    schedule['continuous_stream_contract']=dict(schema='flagmega.continuous-iq4-stream/v1',origin_calls=[gc['call_id'] if 'call_id' in gc else ps[1]['entry_events'][0]['call'],ps[2]['entry_events'][0]['call'],ps[3]['entry_events'][0]['call']],slot_sets=1,epoch_reset=False,drain_at_boundary=False,down_k256_per_epoch=32 if gc["down_pair_pack"] else 16,down_epochs_per_row=3 if gc["down_pair_pack"] else 5,down_banks_per_epoch=2 if gc["down_pair_pack"] else 1)
    g['entry_events']=schedule['consumer_events'];g['render_calls']=[gc]
    g['requires_grid_wide_sync']=not aligned
    launch=deepcopy(ks[1]);launch.pop('launch_recipe',None);launch['calls']=sum([k['calls'] for k in ks[1:]],[]);launch['requires_grid_wide_sync']=not aligned;launch['compile_options']['consumer_barrier_abi']=abi
    if aligned:
        # Native storage extension keeps XQ and YQ disjoint throughout the fused invocation.
        gc['continuous_q8_pointers']=list(qc['pointers'])
        gc['continuous_q8_pointers'][1]='('+qc['pointers'][1]+' + 32768)'
        gc['continuous_down_pointers']=list(dc['pointers'])
        gc['continuous_down_pointers'][0]=gc['continuous_q8_pointers'][1]
        schedule['continuous_stream_contract'].update(q8_aligned=True,q8_storage_base=32768,q8_storage_bytes=36*544+4*544+8,work_queue=True,queue_offset=54528,queue_claims_per_invocation=544+grid,queue_ticket_dtype='int32',queue_wraparound_supported=False,queue_max_invocations=2147483647//(544+grid),groups_per_invocation=544,resident_ctas_per_sm=min(9,65536//(32*5*gc["compile_options"]["maxnreg"])) if gc.get("no_shared_staging") else 6,rows_per_group=32,groups_per_cta=None,rows_per_cta=None,ready_protocol='per-group-release-acquire-generation',grid_barriers=0,requires_resident_grid=True)
    if aligned and gc['q8_cyclic']:schedule['continuous_stream_contract']['row_order']='cyclic-q8-groups'
    if aligned and gc['q8_local_end']:schedule['continuous_stream_contract']['q8_prepare_stage']='local_end'
    if aligned and gc['q8_boundary_trace']:
        schedule['continuous_stream_contract']['boundary_trace']=True
        schedule['continuous_stream_contract']['q8_storage_bytes']+=grid*48
    if gc.get('no_shared_staging'):
        schedule['continuous_stream_contract'].update(slot_sets=0,no_shared_staging=True,weight_copy_warps=0,independent_prefetch_warps=1,warpgroup_padding_warps=0,global_grid_arrival_counters=0,fixed_registers=gc['compile_options']['maxnreg'],prefetch_window=gc['prefetch_window'])
    launch['continuous_stream_contract']=schedule['continuous_stream_contract']
    result.update(programs=[ps[0],g],kernels=[ks[0],launch],continuous_stream_contract=schedule['continuous_stream_contract'])
    return result


def fuse_programs(module, result):
    n=int(os.environ.get('CROSS_LAYERS','1'))
    if n==1:return fuse_single(module,result)
    ps=result['programs'];ks=result['kernels']
    if n not in (2,4) or len(ps)!=4*n:
        raise CodegenError('cross-layer native stream requires 2 or 4 full dependent chains')
    if os.environ.get('CROSS_LAYERWISE')=='1':
        programs=[];kernels=[];contract=None
        for i in range(n):
            part=fuse_single(module,{'programs':deepcopy(ps[4*i:4*i+4]),'kernels':deepcopy(ks[4*i:4*i+4])})
            programs.extend(part['programs']);kernels.extend(part['kernels']);contract=part['continuous_stream_contract']
        result.update(programs=programs,kernels=kernels,continuous_stream_contract=contract)
        return result
    layers=[]
    for i in range(n):
        part={'programs':deepcopy(ps[4*i:4*i+4]),'kernels':deepcopy(ks[4*i:4*i+4])}
        part=fuse_single(module,part)
        c=part['programs'][1]['render_calls'][0]
        layers.append(dict(glu_pointers=list(c['pointers']),down_pointers=list(c['continuous_down_pointers']),q8_pointers=list(c['continuous_q8_pointers'])))
    base=fuse_single(module,{'programs':deepcopy(ps[:4]),'kernels':deepcopy(ks[:4])})
    g=base['programs'][1];gc=g['render_calls'][0];launch=base['kernels'][1]
    names=[x.strip() for x in gc['signature'].split(',')]
    extra=list(dict.fromkeys(x.strip() for p in ps[4:] for x in p['render_calls'][0]['signature'].split(',') if x.strip() not in names))
    gc['signature']+=', '+', '.join(extra)
    for role in ('producer','consumer'):
        for e in g['pipeline_schedule'][role+'_events']:
            if e.get('symbol','').startswith(gc['symbol']):e['arguments']+=', '+', '.join(extra)
    # CHIP backing follows each typed binding; native extension reserves a
    # disjoint YQ/ready/queue/XQ/row-counter bank per layer.
    for i,l in enumerate(layers):
        yq=f'(workspace + {32768+i*32768})'
        l['q8_pointers'][1]=yq;l['down_pointers'][0]=yq
        if i:l['glu_pointers'][0]=f'(workspace + {32768+(i-1)*32768+21776})'
    g['pipeline_schedule']['producer_registers']=32 if gc.get('no_shared_staging') else 24
    g['pipeline_schedule']['producer_warps']=1 if gc.get('no_shared_staging') else 4
    gc['padding_prefetch']=False
    gc['cross_worker_warps']=1 if gc.get('no_shared_staging') else 4
    gc['transport_copy_tiles']=((0,128),(128,32),(160,8),(168,8))
    gc.update(cross_trace_diagnostics=os.environ.get('CROSS_TRACE_DIAGNOSTICS')=='1',cross_lookahead_rows=int(os.environ.get('CROSS_LOOKAHEAD_ROWS','0')),producer_registers=g['pipeline_schedule']['producer_registers'],cross_prefetch_registers=int(os.environ.get('CROSS_PREFETCH_REGISTERS','24')),cross_layers=n,cross_layer_pointers=layers,cross_trace=os.environ.get('CROSS_TRACE')=='1',cross_prefetch_rows=int(os.environ.get('CROSS_PREFETCH_ROWS','4')),q8_boundary_trace=False)
    # workspace is the typed pool root in this fixed ABI, verified below.
    if 'workspace' not in gc['signature'].split(', '):raise CodegenError('cross stream needs typed workspace pool')
    table_base=32768+n*32768
    trace_base=table_base+256
    if not gc.get('no_shared_staging'):
        gc['compile_options']['cross_layer_registers']=True
        launch['compile_options']['cross_layer_registers']=True
    gc['cross_trace_base']=trace_base
    trace_grid=gc['grid'][0] if gc.get('no_shared_staging') else 216
    gc['cross_pftrace_base']=trace_base+n*trace_grid*24*64
    gc['cross_table_base']=table_base
    import re
    table_arguments=[]
    for l in layers:
        for section,index in (('glu_pointers',1),('glu_pointers',2),('glu_pointers',3),('down_pointers',1),('down_pointers',3)):
            expr=l[section][index];root=expr.split('.to(',1)[0]
            if not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*',root):raise CodegenError('cross table requires direct typed CHIP pointer')
            table_arguments.append(root)
    contract=g['pipeline_schedule']['continuous_stream_contract']
    contract.update(schema='flagmega.cross-layer-iq4-stream/v1',layers=n,producer_registers=gc['producer_registers'],prefetch_registers=gc['cross_prefetch_registers'],slot_sets=1,epoch_reset=False,drain_at_boundary=False,pipe_rebuild=False,grid_barriers=0,global_grid_arrival_counters=0,output_publication='160 independent release EXCH flags; one CTA owns all 32 rows in each output Q8 group; no completion counters',down_row_mapping='CTA i owns rows [32*i,32*i+32), i<160',layer_stride=32768,trace_base=trace_base,trace_records_per_cta_layer=24,trace_record_bytes=64,cross_trace=gc['cross_trace'],pointer_table_base=table_base,pointer_table_arguments=table_arguments,pointer_table_entry_bytes=8,q8_storage_bytes=n*32768+256+(n*trace_grid*24*64 if gc['cross_trace'] else 0),queue_max_invocations=2147483647//(544+gc['grid'][0]),cross_prefetch_rows=gc['cross_prefetch_rows'],register_phase_protocol='whole warpgroups: consumer wait32/compute56, 4-weight-worker wait24/copy48; peak320',register_lease='one CTA-local shared CAS owner; whole warpgroup claims before increasing, releases after decreasing and 128-thread sync; no global arrival counter',weight_copy_warps=1,warpgroup_padding_warps=3,physical_weight_worker_warps=4,independent_prefetch_warps=0)
    if gc.get('no_shared_staging'):
        contract.update(slot_sets=0,weight_copy_warps=0,warpgroup_padding_warps=0,physical_weight_worker_warps=1,independent_prefetch_warps=1,fixed_registers=gc['compile_options']['maxnreg'],register_phase_protocol='128-thread consumer warpgroup keeps fixed maxnreg; 32-thread producer performs no dynamic register redistribution',register_lease='CTA-local CAS owner for the 128-thread consumer cohort; no producer lease and no grid arrivals',prefetch_experiment='one bounded PF2 batch per weight row; CTA mailbox acknowledgement before reuse; no refresh',trace_grid=trace_grid)
    if gc['cross_lookahead_rows']:
        if gc['cross_lookahead_rows'] not in (1,2,4,8):raise CodegenError('boundary PF2 depth requires 1,2,4,8')
        contract.update(prefetch_experiment='cyclic gate and recycle refresh; int32 CTA gate publication',lookahead_rows=gc['cross_lookahead_rows'],lookahead_transport='three padding warps issue cyclic PF2 for first N rows while own input gate or first slot recycling is pending; no shared weight writes before gate; cap2 unchanged')
        if gc['cross_trace']:
            contract['q8_storage_bytes']+=n*trace_grid*8192*32
            contract['pftrace_base']=gc['cross_pftrace_base']
            contract['pftrace_batches_per_phase']=4096
    launch['calls']=sum((k['calls'] for k in ks[1:]),[]);launch['continuous_stream_contract']=contract
    result.update(programs=[ps[0],g],kernels=[ks[0],launch],continuous_stream_contract=contract)
    return result
