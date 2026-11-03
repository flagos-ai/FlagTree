"""The full vocabulary pipeline preserves unrelated calls and exact K math."""
import copy
from dataclasses import dataclass
from types import SimpleNamespace
import pytest
from triton.flagmega.codegen.triton.templates import TritonTemplateRegistry
from triton.flagmega.errors import CodegenError
from triton.flagmega.ir import Node, NoneType
from local_optimizations import logits_load, lm_head_pipeline as pipeline


@dataclass(frozen=True)
class Selection:
    family: str
    variant: str
    parameters: dict


@dataclass(frozen=True)
class Dispatch:
    microkernel: Selection


@dataclass(frozen=True)
class Definition:
    name: str
    dispatch: Dispatch


def test_typed_callee_requirement_changes_with_separate_program():
    node=Node(pipeline.CALL,'tir.call',(),NoneType(),attrs={'callee':'full_logits'})
    module=SimpleNamespace(node_map={pipeline.CALL:node},nodes=[node])
    definition=Definition('full_logits',Dispatch(Selection('dense_matmul','packed_k_major_mma',
        dict(compute_num_warps=8,block_k=128,tile_n=64,packed_layout='k_major_n8_k16'))))
    revised,name=pipeline.select_definition(module,[definition])
    assert name=='full_logits' and revised[0].dispatch.microkernel.parameters['compute_num_warps']==4
    assert revised[0].dispatch.microkernel.parameters == {**definition.dispatch.microkernel.parameters,'compute_num_warps':4}
    assert definition.dispatch.microkernel.parameters['compute_num_warps']==8
    module.nodes.append(Node('neighbor','tir.call',(),NoneType(),attrs={'callee':'full_logits'}))
    with pytest.raises(CodegenError,match='shared'):
        pipeline.select_definition(module,[definition])


def test_single_full_logits_program_changes_only_warps():
    before = [dict(symbol='prefix', calls=['prefix'], num_warps=8, grid=[128,1,1]),
              dict(symbol='logits', calls=[pipeline.CALL], num_warps=8, grid=[128,1,1], depends_on=['prefix'])]
    after, selected = pipeline.select_programs(before)
    assert selected == ['logits'] and after[0] is before[0]
    assert {**after[1], 'num_warps':8} == before[1]
    assert before[1]['num_warps'] == 8


@pytest.mark.parametrize('calls,warps', [([pipeline.CALL,'other'],8),([pipeline.CALL],4)])
def test_fused_or_changed_logits_invocation_rejected(calls, warps):
    with pytest.raises(CodegenError):
        pipeline.select_programs([dict(symbol='logits',calls=calls,num_warps=warps)])


def test_original_k_math_masks_and_output_stores_preserved():
    native = (TritonTemplateRegistry().root / logits_load.KEY).read_text()
    # Recover the exact prior coalesced-load overlay for a source comparison.
    anchor='{{ entry.kernel_decorator(call) }}'
    previous=native.replace(anchor,'{% if call.logits_coalesced_load | default(false) %}\n{{ call.symbol }}__physical_load_layout=tl.constexpr(tle.gpu.BlockEncoding([1,8],[8,4],[4,2],[1,0]))\n{% endif %}\n'+anchor)
    start=previous.index('                dense_weight_value = tl.load(')
    end=previous.index('                dense_activation_matrix =',start)
    load='                dense_weight_value = tle.gpu.set_layout(tl.load(tle.gpu.set_layout({{ call.weight }} + ({{ call.weight_offset }}), {{ call.symbol }}__physical_load_layout), mask=dense_n_mask[:,None] & dense_k_mask[None,:], other=0.), {{ call.symbol }}__physical_load_layout)\n'
    previous=previous[:start]+'{% if call.logits_coalesced_load | default(false) %}\n'+load+'{% else %}\n'+previous[start:end]+'{% endif %}\n'+previous[end:]
    changed=pipeline.transform(previous)
    assert changed.replace('[4,{% if call.logits_four_warp_pipeline | default(false) %}1{% else %}2{% endif %}]','[4,2]').replace('{% if call.logits_four_warp_pipeline | default(false) %}, loop_unroll_factor=2{% endif %}','') == previous
    assert logits_load.transform(native) == changed


def test_descriptor_keeps_all_other_kernel_calls():
    call=dict(call=pipeline.CALL,family='dense_matmul',variant='packed_k_major_mma',local_m_capacity=1,
              local_n_capacity=304,local_k_capacity=4096,tile_n=64,block_k=128,
              signature='workspace, rdata, device_dptr',logits_coalesced_load=True)
    before=dict(render_calls=[call,dict(call='other')],implementation_override=dict(inputs={}))
    after=pipeline.describe(None,lambda _:copy.deepcopy(before))
    assert after['render_calls'][1] == before['render_calls'][1]
    assert after['render_calls'][0].pop('logits_four_warp_pipeline') is True
    assert after['render_calls'][0] == call


@pytest.mark.parametrize('field,value',[('local_n_capacity',256),('local_k_capacity',2048),('block_k',256),('signature','workspace, rdata')])
def test_changed_full_vocabulary_contract_rejected(field,value):
    call=dict(call=pipeline.CALL,family='dense_matmul',variant='packed_k_major_mma',local_m_capacity=1,
              local_n_capacity=304,local_k_capacity=4096,tile_n=64,block_k=128,
              signature='workspace, rdata, device_dptr',logits_coalesced_load=True)
    call[field]=value
    with pytest.raises(CodegenError):pipeline.describe(None,lambda _:dict(render_calls=[call]))
