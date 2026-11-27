from dataclasses import replace
import os
from . import iq4_q8_pipeline_renderer as prior
from . import iq4_q8_persistent_renderer as persistent
from . import iq4_q8_renderer as accepted
from triton.flagmega.ir.launch_recipe import TransportRowLaunchRecipe
from triton.flagmega.ir.tir import TIRSharedWorkspaceDescriptor,TIRTransferPipelineContract,TIRTransferPipelineChannel
from triton.flagmega.ir import tensor_type
from triton.flagmega.errors import CodegenError
NO_SHARED=os.environ.get('NO_SHARED_STAGING')=='1'
BOUNDED_PREFETCH=int(os.environ.get('BOUNDED_PREFETCH','0'))
PREFETCH_WINDOW=int(os.environ.get('PREFETCH_WINDOW','4'))
assert PREFETCH_WINDOW in (0,1,2,4,8,16)
PADDING_PREFETCH=int(os.environ.get('PADDING_PREFETCH','0'))
DOWN=os.environ.get('CROSS_SHAPE_DOWN','0')=='1'
N,K=(5120,17408) if DOWN else (17408,5120)
FAMILY='iq4_q8_projection' if DOWN else 'iq4_q8_glu'
ROW_BYTES=136*(K//256)
ROW_VECTORS=ROW_BYTES//16
CARRIER_VECTORS=((ROW_VECTORS+6+15)//16)*16
PRODUCER_PIPELINE=os.environ.get("PRODUCER_PIPELINE","serial")
assert PRODUCER_PIPELINE in ("serial","prime","batch")
RESERVE_DEPTH=int(os.environ.get("RESERVE_DEPTH","0"))
assert RESERVE_DEPTH in (0,1,2,3)
CP_GROUP_PER_STAGE=int(os.environ.get("CP_GROUP_PER_STAGE","0"))
assert CP_GROUP_PER_STAGE in (0,1)
PAYLOAD_OBSERVE=int(os.environ.get("PAYLOAD_OBSERVE","0"))
ISSUE_CREDITS=int(os.environ.get("ISSUE_CREDITS","0"))
assert ISSUE_CREDITS in (0,1,2,4)
assert PAYLOAD_OBSERVE in (0,1)
SINGLE_EPOCH=int(os.environ.get("SINGLE_EPOCH","0"))
assert SINGLE_EPOCH in (0,1)
CP_COMMIT_GROUP=int(os.environ.get("CP_COMMIT_GROUP","1"))
assert CP_COMMIT_GROUP in (0,1)
CP_MERGED_TAIL=int(os.environ.get("CP_MERGED_TAIL","0"))
assert CP_MERGED_TAIL in (0,1)
CP_ALIGNED_CARRIER=int(os.environ.get("CP_ALIGNED_CARRIER","0"))
assert CP_ALIGNED_CARRIER in (0,1)
CP_CACHE=os.environ.get("CP_CACHE",".cg")
assert CP_CACHE in (".cg",".ca")
CP_PHASE_ALIGN=int(os.environ.get("CP_PHASE_ALIGN","0"))
assert CP_PHASE_ALIGN in (0,1)
PRODUCER_REGISTERS=int(os.environ.get("PRODUCER_REGISTERS","32"))
assert PRODUCER_REGISTERS in (32,64,128,192,256)
PREFETCH_DEPTH=int(os.environ.get("PREFETCH_DEPTH","0"))
assert PREFETCH_DEPTH in (0,1,2,4,8)
REG_BUDGET=int(os.environ.get("TRANSPORT_MAXNREG","104"))
assert REG_BUDGET in (40,48,56,64,104,128,160,192,256)
STAGE=int(os.environ.get('TRANSPORT_STAGE','1'))
assert STAGE in (1,2,3)
GRID=int(os.environ.get('WARP_GRID','36'))
assert GRID in (544,272,136,68,36,72,108,144,180,216,252,288,324,432,864,1728,2176,3456,4352,8640,8704,17408)
CAPACITY=int(os.environ.get('WARP_CAPACITY',str({36:17,72:8,144:4,216:2,288:1}.get(GRID,2))))
assert CAPACITY in (1,2,3,4,8,16,17)
BATCH_G=int(os.environ.get('BARRIER_BATCH_G','1'))
assert BATCH_G in (1,2,4,8)
VARIANT='r006_glu_transport_t'+str(STAGE)
SCHEMA='flagmega.iq4-q8-transport-fix/t'+str(STAGE)
RECIPE=TransportRowLaunchRecipe(N,1,'persistent',GRID,(GRID+35)//36,capacity=CAPACITY,whole_row=STAGE>=2)
SINGLE=int(os.environ.get('ADDRESS_SINGLE','0'))
PACKET_MODE=os.environ.get('PACKET_MODE','original')
CONSUMER_MODE=os.environ.get('CONSUMER_MODE','normal')
assert CONSUMER_MODE in ('normal','load_only','early_release')
NO_OUTPUT=int(os.environ.get('DIAGNOSTIC_NO_OUTPUT','0'))
assert not NO_OUTPUT or CONSUMER_MODE=='load_only'
PRODUCER_WARPS=int(os.environ.get('PRODUCER_WARPS','1'))
ROWS_PER_TX=int(os.environ.get('ROWS_PER_TX','1'))
assert PRODUCER_WARPS in (1,2) and ROWS_PER_TX in (1,2,4,8)
ABI=persistent.ABI if PRODUCER_WARPS==1 else "H003ConsumerPartial-f32-3x32/bar3-count128/v6-dual-independent"
if PADDING_PREFETCH: ABI='H003ConsumerPartial-f32-3x32/bar3-count128/v8-padding-prefetch'
if PADDING_PREFETCH and BOUNDED_PREFETCH: ABI='H003ConsumerPartial-f32-3x32/bar3-count128/v10-bounded-prefetch'
TRANSPORT_KIND=os.environ.get("TRANSPORT_KIND","tma")
if TRANSPORT_KIND in ("vector_named","cp_named","global_named"):
 ABI="H003ConsumerPartial-f32-3x32/bar5-count128/v7-named-transport160"
if NO_SHARED: ABI='H003ConsumerPartial-f32-3x32/bar3-count128/v26-global-pf2'
G4_RECYCLE=int(os.environ.get("G4_RECYCLE","0"))
assert TRANSPORT_KIND in ("tma","cp_async","cp_mbarrier","vector","vector_direct","vector_chunk","vector_named","cp_named","global_named","global_idle")
SEGMENTS=(256,) if STAGE==1 else ((CARRIER_VECTORS,) if SINGLE else (128,32,8,8))
def parameters(family):
 if family!=FAMILY:raise CodegenError('transport fix qualifies only GLU')
 return {**prior.parameters(family),'num_stages':CAPACITY,'producer_registers':PRODUCER_REGISTERS,'maxnreg':REG_BUDGET,'consumer_barrier_abi':ABI,'transfer_schema':SCHEMA,
 'no_shared_staging':NO_SHARED,'transfer_block_shape':[1,256,16] if STAGE==1 else [1,SEGMENTS[0],16],
 **({'diagnostic_no_output':True} if NO_OUTPUT else {}),**({'producer_warps':PRODUCER_WARPS} if PRODUCER_WARPS!=1 else {}),**({'rows_per_transaction':ROWS_PER_TX} if ROWS_PER_TX!=1 else {}),**({'consumer_mode':CONSUMER_MODE} if CONSUMER_MODE!='normal' else {}),**({'transport_kind':TRANSPORT_KIND} if TRANSPORT_KIND!='tma' else {}),**({'g4_recycle':G4_RECYCLE} if G4_RECYCLE else {}),'packet_mode':PACKET_MODE,'transport_segments':list(SEGMENTS),'transport_whole_row':STAGE>=2,'cp_phase_alignment':bool(CP_PHASE_ALIGN),'cp_cache_modifier':CP_CACHE,'cp_aligned_carrier':bool(CP_ALIGNED_CARRIER),'cp_merged_tail':bool(CP_MERGED_TAIL),'cp_commit_group':bool(CP_COMMIT_GROUP),'single_epoch':bool(SINGLE_EPOCH),'issue_credits':ISSUE_CREDITS,'payload_observe':bool(PAYLOAD_OBSERVE),'bounded_prefetch':bool(BOUNDED_PREFETCH),'prefetch_window':PREFETCH_WINDOW,'padding_prefetch':bool(PADDING_PREFETCH),'producer_pipeline':PRODUCER_PIPELINE,'reserve_depth':RESERVE_DEPTH,'cp_group_per_stage':bool(CP_GROUP_PER_STAGE),'persistent_row_protocol':RECIPE.to_data(),
 **({'barrier_batch_g':BATCH_G,'batch_partial_floats':256*BATCH_G,'batch_partial_storage':'u64-p-u-bitpack'} if BATCH_G>1 else {})}
def implementations():
 if PRODUCER_PIPELINE != "serial" and (not SINGLE_EPOCH or not CP_ALIGNED_CARRIER or ISSUE_CREDITS):
  raise CodegenError("producer reservation window requires the qualified paired epoch")
 if RESERVE_DEPTH > CAPACITY:
  raise CodegenError("reservation depth exceeds capacity")
 if SINGLE_EPOCH and (TRANSPORT_KIND != "cp_mbarrier" or not CP_ALIGNED_CARRIER or PRODUCER_WARPS != 1):
  raise CodegenError("single epoch requires the qualified one-producer aligned gate/up carrier")
 if ISSUE_CREDITS and (not SINGLE_EPOCH or ISSUE_CREDITS > CAPACITY):
  raise CodegenError("issuance credits require a single epoch and cannot exceed capacity")
 if PAYLOAD_OBSERVE and not SINGLE_EPOCH:
  raise CodegenError("stage completion observations require the paired epoch")
 if CP_ALIGNED_CARRIER and not CP_PHASE_ALIGN:
  raise CodegenError("aligned carrier requires the phase-preserving reader view")
 if CP_PHASE_ALIGN and (TRANSPORT_KIND != "cp_mbarrier" or GRID % 4):
  raise CodegenError("cp.async phase alignment requires whole single rows and a grid divisible by four")
 spaces=tuple(TIRSharedWorkspaceDescriptor(bank+'_s'+str(j),tensor_type('uint8',(CAPACITY,ROWS_PER_TX,32 if DOWN and bank=='up' else n,16)),128,matrix_compatible=True)
  for bank in ('gate','up') for j,n in enumerate(SEGMENTS))
 channels=tuple(TIRTransferPipelineChannel(bank,(arg,),tuple(range(i*len(SEGMENTS),(i+1)*len(SEGMENTS))),16)
  for i,(bank,arg) in enumerate((('gate',1),('up',1 if DOWN else 3))))
 return tuple(replace(i,id='tir.'+FAMILY+'.'+VARIANT,variant=VARIANT,parameters=parameters(i.family),
  shared_workspaces=spaces,transfer_pipeline=TIRTransferPipelineContract(channels,capacity=CAPACITY),
  contract={**i.contract,'mapping_identity':VARIANT,'row_epoch_recipe':RECIPE.to_data()})
  for i in prior.implementations() if i.family==FAMILY)
def validate_raw(raw):
 if raw['variant']!=VARIANT or raw['family']!=FAMILY:raise CodegenError('wrong transport subject')
 if (raw['semantic_attrs']['logical_n'],raw['semantic_attrs']['logical_k'])!=(N,K):raise CodegenError('wrong transport shape')
 for k,v in parameters(raw['family']).items():
  if accepted._canonical(raw['parameters'].get(k, False if k=='no_shared_staging' else 'original' if k=='packet_mode' else False if k in ('cp_phase_alignment','cp_aligned_carrier','cp_merged_tail','single_epoch','payload_observe') else 'serial' if k=='producer_pipeline' else False if k=='cp_group_per_stage' else 0 if k in ('issue_credits','reserve_depth') else True if k=='cp_commit_group' else '.cg' if k=='cp_cache_modifier' else None))!=accepted._canonical(v):raise CodegenError('transport parameter mismatch '+k)
 if accepted._canonical(raw['parameters'].get('task_mapping'))!=accepted._canonical(RECIPE.to_data()):raise CodegenError('transport mapping mismatch')
 accepted.validate_raw({**raw,'variant':accepted.VARIANT,'parameters':accepted.implementation_parameters(raw['family'])})
 e=implementations()[0];t=e.transfer_pipeline
 exported=dict(capacity=t.capacity,producer_read_argument_indices=t.producer_read_argument_indices,
 consumer_shared_workspace_indices=t.consumer_shared_workspace_indices,auxiliary_consumer=None,
 channels=tuple(dict(name=c.name,source_argument_indices=c.source_argument_indices,shared_workspace_indices=c.shared_workspace_indices,source_alignment_bytes=c.source_alignment_bytes) for c in t.channels))
 if accepted._canonical(raw['transfer_pipeline'])!=accepted._canonical(exported):raise CodegenError('transport contract mismatch')
 if len(raw['shared_workspaces'])!=len(e.shared_workspaces):raise CodegenError('transport workspace count')
 for a,b in zip(raw['shared_workspaces'],e.shared_workspaces):
  if tuple(a['shape'])!=tuple(x.fixed_value for x in b.type.shape) or a['dtype']!='uint8' or not a['matrix_compatible']:raise CodegenError('transport staging ABI '+repr((a,b.type.shape)))
def materialize(raw):
 validate_raw(raw)
 r=accepted.materialize({**raw,'variant':accepted.VARIANT,'parameters':accepted.implementation_parameters(raw['family'])})
 ds=[]
 for bank,name in ((('gate','weight'),) if DOWN else (('gate','gate_weight'),('up','up_weight'))):
  b=accepted._bindings(raw,'inputs',name)[0]
  for j,n in enumerate(SEGMENTS):
   ds.append(dict(parameter=bank+'_map'+str(j),source=b['runtime_argument'],kind='single',offset_bytes=b['abi']['pool_byte_offset'],dtype='uint8',
    shape=(N,ROW_VECTORS,16),strides=(ROW_BYTES,16,1),block_shape=(ROWS_PER_TX,ROW_VECTORS if SINGLE else n,16),source_shape_axes=((),(),()),padding='zero'))
 if NO_SHARED: ds=[]
 if DOWN: ds=[]  # cp.async uses ordinary pointers; no tensor map is consumed.
 return {**r,'schema':SCHEMA,'jit_module':'','jit_entry':'','grid':[GRID,1,1],'program_kind':'persistent',
 'pipeline_contract':dict(raw['transfer_pipeline']),'shared_workspaces':tuple(raw['shared_workspaces']),
 'compile_options':{**r['compile_options'],'maxnreg':REG_BUDGET,'consumer_barrier_abi':ABI,'ptx_options':os.environ.get('OCCUPANCY_PTX_OPTIONS','--minnctapersm=1')},'num_stages':CAPACITY,'producer_warps':PRODUCER_WARPS,'producer_registers':PRODUCER_REGISTERS,
 'host_tensor_descriptor_requests':tuple(ds),'transport_descriptors':tuple(d['parameter'] for d in ds),
 'transport_segments':SEGMENTS,'transport_whole_row':STAGE>=2,'cp_phase_alignment':bool(CP_PHASE_ALIGN),'cp_cache_modifier':CP_CACHE,'cp_aligned_carrier':bool(CP_ALIGNED_CARRIER),'cp_merged_tail':bool(CP_MERGED_TAIL),'cp_commit_group':bool(CP_COMMIT_GROUP),'single_epoch':bool(SINGLE_EPOCH),'issue_credits':ISSUE_CREDITS,'payload_observe':bool(PAYLOAD_OBSERVE),'bounded_prefetch':bool(BOUNDED_PREFETCH),'prefetch_window':PREFETCH_WINDOW,'padding_prefetch':bool(PADDING_PREFETCH),'producer_pipeline':PRODUCER_PIPELINE,'reserve_depth':RESERVE_DEPTH,'cp_group_per_stage':bool(CP_GROUP_PER_STAGE),'transport_n':N,'transport_k':K,'no_shared_trace':os.environ.get('NO_SHARED_TRACE')=='1','cross_table_base':0,'cross_trace':False,'cross_trace_diagnostics':False,'cross_lookahead_rows':0,'no_shared_staging':NO_SHARED,'transport_mode':0 if DOWN else 1,'cross_shape_down':DOWN,'transport_row_bytes':ROW_BYTES,'transport_row_vectors':ROW_VECTORS,'transport_carrier_vectors':CARRIER_VECTORS,'transport_copy_tiles':((0,256),(256,256),(512,64),(576,8)) if DOWN else ((0,128),(128,32),(160,8),(168,8)),
 'pipeline_commits_per_invocation':1 if STAGE>=2 else 2,'persistent_row_recipe':RECIPE.to_data(),
 'barrier_batch_g':BATCH_G,'address_single':SINGLE,'prefetch_depth':PREFETCH_DEPTH,'packet_mode':PACKET_MODE,'consumer_mode':CONSUMER_MODE,'rows_per_transaction':ROWS_PER_TX,'diagnostic_no_output':bool(NO_OUTPUT),'transport_kind':TRANSPORT_KIND,'g4_recycle':G4_RECYCLE,'cp_phase_alignment':bool(CP_PHASE_ALIGN),'cp_cache_modifier':CP_CACHE,'cp_aligned_carrier':bool(CP_ALIGNED_CARRIER),'cp_merged_tail':bool(CP_MERGED_TAIL),'cp_commit_group':bool(CP_COMMIT_GROUP),'single_epoch':bool(SINGLE_EPOCH),'issue_credits':ISSUE_CREDITS,'payload_observe':bool(PAYLOAD_OBSERVE),'bounded_prefetch':bool(BOUNDED_PREFETCH),'prefetch_window':PREFETCH_WINDOW,'padding_prefetch':bool(PADDING_PREFETCH),'producer_pipeline':PRODUCER_PIPELINE,'reserve_depth':RESERVE_DEPTH,'cp_group_per_stage':bool(CP_GROUP_PER_STAGE)}
