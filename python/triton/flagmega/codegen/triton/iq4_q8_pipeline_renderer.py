# Copyright 2026 FlagOS Contributors; SPDX-License-Identifier: MIT
"""Versioned IQ4 byte transport; accepted numerical helper is unchanged."""
from . import iq4_q8_renderer as accepted
from triton.flagmega.errors import CodegenError
from triton.flagmega.codegen.triton.implementation import TritonImplementation
from triton.flagmega.ir import tensor_type
from triton.flagmega.ir.tir import TIRSharedWorkspaceDescriptor,TIRTransferPipelineContract,TIRTransferPipelineChannel
VARIANT='q8_r006_consumer_barrier_v3'
SCHEMA='flagmega.iq4-q8-consumer-transfer/r006-v3'
CONSUMER_BARRIER_ABI='H003ConsumerPartial-f32-3x32/bar5-count128/v2'
FAMILIES=accepted.FAMILIES[1:]
BLOCK_SHAPE=(1,256,16)
def parameters(family):
 p=accepted.implementation_parameters(family)
 p.update(num_stages=2,producer_warps=1,producer_registers=32,transfer_schema=SCHEMA,
          transfer_block_shape=list(BLOCK_SHAPE),transfer_k256_blocks=16,consumer_barrier_abi=CONSUMER_BARRIER_ABI)
 return p
def implementations():
 result=[]
 for f in FAMILIES:
  names=('gate','up') if f=='iq4_q8_glu' else ('weight',)
  indices=(1,3) if f=='iq4_q8_glu' else (1,)
  spaces=tuple(TIRSharedWorkspaceDescriptor(n+'_stage',tensor_type('uint8',(2,*BLOCK_SHAPE)),128,matrix_compatible=True) for n in names)
  channels=tuple(TIRTransferPipelineChannel(n,(arg,),(i,),16) for i,(n,arg) in enumerate(zip(names,indices)))
  result.append(TritonImplementation('tir.'+f+'.'+VARIANT,f,VARIANT,parameters(f),
    contract={'weight_encoding':'IQ4_XS','q8_abi':accepted.Q8_ABI,'row_tile':1},
    requires=('tma','warp_specialize'),facts={'required_codegen_backend':'triton','internal_grid_barriers':0,'transfer_pipeline':True,'host_tensor_descriptor':True},
    shared_workspaces=spaces,transfer_pipeline=TIRTransferPipelineContract(channels,capacity=2)))
 return tuple(result)
def validate_raw(raw):
 f=raw['family']
 if f not in FAMILIES or raw['variant']!=VARIANT:raise CodegenError('r006: wrong family/variant')
 p=raw['parameters']
 for k,v in parameters(f).items():
  if k not in p or type(p[k]) is not type(v) or accepted._canonical(p[k])!=accepted._canonical(v):raise CodegenError('r006: wrong parameter '+k)
 # The old verifier is a read-only semantic/ABI sub-proof. The new transport
 # options and physical alignment are independently checked here.
 accepted.validate_raw({**raw,'variant':accepted.VARIANT,'parameters':accepted.implementation_parameters(f)})
 names=('gate_weight','up_weight') if f=='iq4_q8_glu' else ('weight',)
 for name in names:
  b=accepted._bindings(raw,'inputs',name)[0]
  accepted._abi(b,dtype='uint8',shape=(raw['semantic_attrs']['logical_n'],136*(raw['semantic_attrs']['logical_k']//256)),alignment=16)
 expected=next(i for i in implementations() if i.family==f)
 validate_transfer_contract(raw,expected)

def validate_transfer_contract(raw,expected):
 contract=expected.transfer_pipeline
 exported=dict(capacity=contract.capacity,producer_read_argument_indices=contract.producer_read_argument_indices,
  consumer_shared_workspace_indices=contract.consumer_shared_workspace_indices,auxiliary_consumer=None,
  channels=tuple(dict(name=c.name,source_argument_indices=c.source_argument_indices,
   shared_workspace_indices=c.shared_workspace_indices,source_alignment_bytes=c.source_alignment_bytes) for c in contract.channels))
 if accepted._canonical(raw.get('transfer_pipeline'))!=accepted._canonical(exported):raise CodegenError('r006: transfer contract mismatch')
 actual=raw.get('shared_workspaces',())
 if len(actual)!=len(expected.shared_workspaces) or any(tuple(a['shape'])!=tuple(d.fixed_value for d in b.type.shape) or a['dtype']!=b.type.dtype.value or a['matrix_compatible']!=b.matrix_compatible for a,b in zip(actual,expected.shared_workspaces)):raise CodegenError('r006: staging ABI mismatch')
def materialize(raw):
 validate_raw(raw);f=raw['family'];a=raw['semantic_attrs'];k=a['logical_k'];n=a['logical_n']
 prior=accepted.materialize({**raw,'variant':accepted.VARIANT,'parameters':accepted.implementation_parameters(f)})
 names=('gate_weight','up_weight') if f=='iq4_q8_glu' else ('weight',)
 descriptors=[]
 for name in names:
  b=accepted._bindings(raw,'inputs',name)[0];abi=b['abi']
  descriptors.append(dict(parameter=name+'_descriptor',source=b['runtime_argument'],kind='single',
    offset_bytes=abi['pool_byte_offset'],dtype='uint8',shape=(n,136*(k//256)//16,16),
    strides=(136*(k//256),16,1),block_shape=BLOCK_SHAPE,source_shape_axes=((),(),()),padding='zero'))
 return {**prior,'schema':SCHEMA,'jit_module':'','jit_entry':'','jit_source_sha256':accepted.JIT_SOURCE_SHA256,
   'pipeline_contract':dict(raw['transfer_pipeline']),'shared_workspaces':tuple(raw['shared_workspaces']),
   'compile_options':{**prior['compile_options'],'consumer_barrier_abi':CONSUMER_BARRIER_ABI},
   'num_stages':2,'producer_warps':1,'producer_registers':32,'host_tensor_descriptor_requests':tuple(descriptors),
   'pipeline_commits_per_invocation':(k//256+15)//16,'transport_n':n,'transport_k':k,
   'transport_mode':1 if f=='iq4_q8_glu' else 2 if f=='iq4_q8_interleaved' else 0,
   'transport_descriptors':tuple(d['parameter'] for d in descriptors)}
