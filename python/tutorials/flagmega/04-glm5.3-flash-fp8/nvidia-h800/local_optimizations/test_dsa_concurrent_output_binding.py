"""Fail-before reproducer: concurrent Q must publish to its normally allocated output."""
import copy,importlib.util
from pathlib import Path
import pytest
path=Path(__file__).with_name('dsa_concurrent.py')
spec=importlib.util.spec_from_file_location('dsa_concurrent_root_reproducer',path)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)

@pytest.mark.parametrize('base',[0,5632,5760,8192])
def test_concurrent_Q_preserves_normal_output_binding_and_complete_owner_coverage(base):
 typed_pointer=f'workspace.to(tl.pointer_type(tl.bfloat16)) + {base} + shard_index * 8'
 def call(symbol,capacity):
  return dict(symbol=symbol,variant='packed_k_major_mma_prefetch',block_k=128,prefetch_k=2048,prefetch_n=8,local_n_capacity=8,tile_n=16,local_k_capacity=capacity,signature='workspace, weight, device_dptr',result=typed_pointer,result_offset='original_owner',result_active='dense_local_n_offsets // 8 < 1',prefetch_result_active='dense_bulk_n_offsets // 8 < 1',prefetch_weight_offset='dense_bulk_n_offsets[:, None]')
 calls=[call('dsa_indexer_q',1536),call('dsa_indexer_key',4096),call('dsa_indexer_gate',4096),call('dsa_indexer_weights',4096)]
 events=[dict(kind='tir.kernel_call',symbol=c['symbol'],arguments=f'workspace, weight_{i}, device_dptr') for i,c in enumerate(calls)]
 descriptor=dict(render_calls=calls,persistent_window_resources=dict(resources=[dict(name='dsa_deferred_results',nbytes=768,offset=3000000)]),device_functions=[dict(function='decode_mla_attention_sparse.prefix',schedule=dict(consumer_events=events))],programs=[],implementation_override=dict(inputs={}))
 result=module.describe(None,lambda _:copy.deepcopy(descriptor));q=result['render_calls'][0]
 assert q['result']==typed_pointer, 'Concurrent override discarded the normal typed allocation base'
 # Evaluate the actual emitted pointer plus relative index. The typed pointer
 # already owns its shard displacement; counting it twice must fail coverage.
 expression=f"({q['result'].replace('workspace.to(tl.pointer_type(tl.bfloat16))','0')}) + ({q['result_offset']})"
 # Independent coverage of every element in the unchanged 4-chip Q geometry.
 offsets=[]
 for chip in range(4):
  for pid in range(64):
   row,column=divmod(pid,8);shard_index=(chip*8+row)*16+column*2
   offsets.extend(eval(expression,{'__builtins__':{}},{'shard_index':shard_index,'dense_local_n_offsets':i}) for i in range(16))
 assert sorted(offsets)==list(range(base,base+4096))
