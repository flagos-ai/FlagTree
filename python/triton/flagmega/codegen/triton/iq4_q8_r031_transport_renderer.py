# Copyright 2026 FlagOS Contributors; SPDX-License-Identifier: MIT
"""Literal signed R031 t3 recipe; one checked token row, no environment reads."""
from dataclasses import replace
from . import iq4_q8_pipeline_renderer as prior
from . import iq4_q8_renderer as accepted
from triton.flagmega.ir.launch_recipe import TransportRowLaunchRecipe
from triton.flagmega.ir.tir import TIRSharedWorkspaceDescriptor,TIRTransferPipelineContract,TIRTransferPipelineChannel
from triton.flagmega.ir import tensor_type
from triton.flagmega.errors import CodegenError
# Literal extraction of the signed R031 transport recipe. No environment reads.
NO_SHARED=True
BOUNDED_PREFETCH=1
PREFETCH_WINDOW=2
PADDING_PREFETCH=0
DOWN=False
N,K=17408,5120
FAMILY='iq4_q8_glu'
ROW_BYTES=2720
ROW_VECTORS=170
CARRIER_VECTORS=176
PRODUCER_PIPELINE='serial'
RESERVE_DEPTH=0
CP_GROUP_PER_STAGE=0
PAYLOAD_OBSERVE=0
ISSUE_CREDITS=0
SINGLE_EPOCH=0
CP_COMMIT_GROUP=1
CP_MERGED_TAIL=0
CP_ALIGNED_CARRIER=0
CP_CACHE='.cg'
CP_PHASE_ALIGN=0
PRODUCER_REGISTERS=32
PREFETCH_DEPTH=0
REG_BUDGET=40
STAGE=3
GRID=17408
CAPACITY=2
BATCH_G=1
VARIANT='r006_glu_transport_t3'
SCHEMA='flagmega.iq4-q8-transport-fix/t3'
RECIPE=TransportRowLaunchRecipe(N,1,'persistent',GRID,484,capacity=2,whole_row=True)
SINGLE=1
PACKET_MODE='cached_tables'
CONSUMER_MODE='normal'
NO_OUTPUT=0
PRODUCER_WARPS=1
ROWS_PER_TX=1
ABI='H003ConsumerPartial-f32-3x32/bar3-count128/v26-global-pf2'
TRANSPORT_KIND='global_idle'
G4_RECYCLE=0
SEGMENTS=(176,)
def parameters(family):
 if family!=FAMILY:raise CodegenError('transport fix qualifies only GLU')
 return {**prior.parameters(family),'num_warps':4,'num_ctas':1,'num_stages':CAPACITY,'producer_registers':PRODUCER_REGISTERS,'maxnreg':REG_BUDGET,'consumer_barrier_abi':ABI,'transfer_schema':SCHEMA,
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
 if accepted._shape(accepted._bindings(raw,'inputs','q8')[0]['abi'],'logical_shape')[0]!=1:
  raise CodegenError('signed transport t3 qualifies one token row; multirow requires the versioned r041 implementation')
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
 'compile_options':{**r['compile_options'],'maxnreg':REG_BUDGET,'consumer_barrier_abi':ABI,'ptx_options':'--minnctapersm=9 --opt-level=2'},'num_stages':CAPACITY,'producer_warps':PRODUCER_WARPS,'producer_registers':PRODUCER_REGISTERS,
 'host_tensor_descriptor_requests':tuple(ds),'transport_descriptors':tuple(d['parameter'] for d in ds),
 'transport_segments':SEGMENTS,'transport_whole_row':STAGE>=2,'cp_phase_alignment':bool(CP_PHASE_ALIGN),'cp_cache_modifier':CP_CACHE,'cp_aligned_carrier':bool(CP_ALIGNED_CARRIER),'cp_merged_tail':bool(CP_MERGED_TAIL),'cp_commit_group':bool(CP_COMMIT_GROUP),'single_epoch':bool(SINGLE_EPOCH),'issue_credits':ISSUE_CREDITS,'payload_observe':bool(PAYLOAD_OBSERVE),'bounded_prefetch':bool(BOUNDED_PREFETCH),'prefetch_window':PREFETCH_WINDOW,'padding_prefetch':bool(PADDING_PREFETCH),'producer_pipeline':PRODUCER_PIPELINE,'reserve_depth':RESERVE_DEPTH,'cp_group_per_stage':bool(CP_GROUP_PER_STAGE),'transport_n':N,'transport_k':K,'no_shared_trace':False,'cross_table_base':0,'cross_trace':False,'cross_trace_diagnostics':False,'cross_lookahead_rows':0,'no_shared_staging':NO_SHARED,'transport_mode':0 if DOWN else 1,'cross_shape_down':DOWN,'transport_row_bytes':ROW_BYTES,'transport_row_vectors':ROW_VECTORS,'transport_carrier_vectors':CARRIER_VECTORS,'transport_copy_tiles':((0,256),(256,256),(512,64),(576,8)) if DOWN else ((0,128),(128,32),(160,8),(168,8)),
 'pipeline_commits_per_invocation':1 if STAGE>=2 else 2,'persistent_row_recipe':RECIPE.to_data(),
 'barrier_batch_g':BATCH_G,'address_single':SINGLE,'prefetch_depth':PREFETCH_DEPTH,'packet_mode':PACKET_MODE,'consumer_mode':CONSUMER_MODE,'rows_per_transaction':ROWS_PER_TX,'diagnostic_no_output':bool(NO_OUTPUT),'transport_kind':TRANSPORT_KIND,'g4_recycle':G4_RECYCLE,'cp_phase_alignment':bool(CP_PHASE_ALIGN),'cp_cache_modifier':CP_CACHE,'cp_aligned_carrier':bool(CP_ALIGNED_CARRIER),'cp_merged_tail':bool(CP_MERGED_TAIL),'cp_commit_group':bool(CP_COMMIT_GROUP),'single_epoch':bool(SINGLE_EPOCH),'issue_credits':ISSUE_CREDITS,'payload_observe':bool(PAYLOAD_OBSERVE),'bounded_prefetch':bool(BOUNDED_PREFETCH),'prefetch_window':PREFETCH_WINDOW,'padding_prefetch':bool(PADDING_PREFETCH),'producer_pipeline':PRODUCER_PIPELINE,'reserve_depth':RESERVE_DEPTH,'cp_group_per_stage':bool(CP_GROUP_PER_STAGE)}
