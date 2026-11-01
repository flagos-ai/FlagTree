"""Scope and synchronization regressions for the native DSA pool schedule."""
from pathlib import Path
import ast
import re
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from local_optimizations.native_dsa_pool_scores import KEY, _template
from triton.flagmega.codegen.triton.templates import TritonTemplateRegistry


def wrapper_source(parallel):
    registry = TritonTemplateRegistry()
    original = registry.environment.loader.get_source(registry.environment, KEY)[0]
    source = _template(original) if parallel else original
    start = source.rindex('@triton.jit\ndef {{ call.symbol }}(')
    # Render the actual wrapper with the ordinary mesh/barrier macros. Algorithm
    # helpers do not need to execute for lexical scope and phase-order checks.
    tail = source[start:].removesuffix('{% endif %}\n{% endfor %}\n')
    assert not tail.endswith('{% endfor %}\n')
    call = dict(symbol='dsa', signature='workspace, state, sequence_lengths, result, device_dptr',
                advance_sequence='False', pool_score_group=dict(index='((0 * 8 + shard_coord1) * 16 + shard_coord2)', count=128) if parallel else None)
    context = dict(call=call, distributed_entry=True, grid_mesh=True, mesh_rank=3,
                   mesh_hierarchy=(4, 8, 16), mesh_levels='cbb', mesh_axis_names=('device','block_y','block_x'),
                   device_context_abi='runtime-handles/v2')
    return registry.environment.from_string('{% import "entrypoints/_common.py.jinja" as entry with context %}\n'+tail).render(context)


def test_replica_selector_coordinates_belong_to_wrapper_scope():
    source = wrapper_source(True)
    function = ast.parse(source).body[0]
    assigned = {node.id for node in ast.walk(function) if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)}
    used = {node.id for node in ast.walk(function) if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load) and re.fullmatch('shard_coord[0-9]+', node.id)}
    assert used and used <= assigned, (used - assigned, source)


def test_parallel_scores_are_published_before_single_replica_selection():
    function = ast.parse(wrapper_source(True)).body[0]
    calls = [ast.unparse(node.value.func) for node in function.body if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)]
    assert calls == ['dsa__write', 'tle.distributed_barrier', 'dsa__pool_scores', 'tle.distributed_barrier', 'tle.distributed_barrier']
    selectors = [node for node in function.body if isinstance(node, ast.If) and any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == 'dsa__select' for n in ast.walk(node))]
    assert len(selectors) == 1 and 'shard_coord1' in ast.unparse(selectors[0].test)


def test_unselected_wrapper_retains_original_two_barriers():
    source = wrapper_source(False)
    assert '__pool_scores' not in source
    assert source.count('tle.distributed_barrier(') == 2


def render_score_only_selectors(destination):
    """Render the real template at the score/selection boundary for GPU checks."""
    registry = TritonTemplateRegistry()
    original = registry.environment.loader.get_source(registry.environment, KEY)[0]
    call = dict(symbol='dsa', family='dsa_indexer', variant='decode',
        signature='workspace, weights, state, sequence, logical_layer, advance, key, gate, norm, result, device_dptr',
        state_layer_ids=(3,), layer_id='logical_layer', state='state', sequence_lengths='sequence', result='result',
        state_layer_stride=1052672, state_token_stride=257, state_value_stride=1,
        index_dim=128,index_heads=32,q_width=4096,q='workspace + 26240',q_stride=1,
        weights='weights',weights_stride=1,compress_ape='norm',ape_row_stride=128,ape_dim_stride=1,
        key='key',key_stride=1,gate='gate',gate_stride=1,k_norm_weight='key',k_norm_bias='gate',norm_stride=1,
        writer_active='True',advance_sequence='advance',epsilon=1e-5,scale=0.08838834764831845,
        state_capacity=4096,state_tile=4096,pool=4,pool_tile=1024,index_topk=2048,
        output_width=2051,output_tile=4096,value_type='tl.bfloat16')
    outputs=[]
    for parallel in (False,True):
        local={**call,'pool_score_group':dict(index='tl.program_id(0)',count=128) if parallel else None}
        context=dict(render_calls=[local],distributed_entry=False,grid_mesh=False,mesh_rank=3,mesh_hierarchy=(1,1,1),mesh_levels="cbb")
        source=registry.environment.from_string('{% import "entrypoints/_common.py.jinja" as entry with context %}\n'+(_template(original) if parallel else original)).render(context)
        function,=[n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='dsa__select']
        if not parallel:
            function.body=[ast.parse('pool_scores = tl.load(result.to(tl.pointer_type(tl.float32)) + pool_ids)').body[0]
                           if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=='pool_index' else n
                           for n in function.body]
        function.name='dsa_selector_parallel' if parallel else 'dsa_selector_original'
        outputs.append(ast.unparse(function))
    path=Path(destination)/'native_dsa_score_selectors.py'
    path.write_text('import triton\nimport triton.language as tl\n\n'+'\n\n'.join(outputs)+'\n')
    return path


def test_device_selector_preserves_maxnum_and_tie_contract(tmp_path):
    import importlib.util
    import math
    torch=pytest.importorskip('torch')
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()!=(9,0):
        pytest.skip('requires SM90')
    from triton.flagmega.runtime.prepared import ResourceContract,prepare_jit_kernel
    from triton.flagmega.runtime.nvidia_launch import compilation_options
    path=render_score_only_selectors(tmp_path)
    spec=importlib.util.spec_from_file_location('native_dsa_selector_regression',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    workspace=torch.zeros(30336,dtype=torch.bfloat16,device='cuda')
    weights=torch.zeros(32,dtype=torch.bfloat16,device='cuda');ape=torch.zeros((4,128),dtype=torch.bfloat16,device='cuda')
    packed=torch.zeros((1,4096,257),dtype=torch.bfloat16,device='cuda');sequence=torch.zeros(1,dtype=torch.int32,device='cuda')
    outputs=[torch.full((2083,),-991,dtype=torch.int32,device='cuda') for _ in range(2)]
    programs=[];graphs=[]
    for kernel,result in zip((module.dsa_selector_original,module.dsa_selector_parallel),outputs,strict=True):
        contract=ResourceContract(compute_num_warps=8,resident_blocks_per_sm=1)
        program=prepare_jit_kernel(kernel,(workspace,weights,packed,sequence,3,False,workspace,workspace,ape,result,workspace),(),grid=(1,1,1),contract=contract,allocation_device='cuda:0',**compilation_options(contract))
        assert program.resource_report['spill_bytes']==0
        with torch.cuda.graph(graph:=torch.cuda.CUDAGraph()):program.launch()
        programs.append(program);graphs.append(graph)
    for values in ([float('nan'),1.,2.,0.,-0.,3.,-float('inf'),float('inf')],
                   [-float('nan'),1.,2.,0.,-0.,3.,-float('inf'),float('inf')],
                   [float('nan')]*1024,[0.,-0.,0.,-0.,1.,-1.,2.,-2.]):
        count=len(values);scores=torch.full((1024,),-float('inf'),dtype=torch.float32);scores[:count]=torch.tensor(values,dtype=torch.float32)
        packed.zero_();packed[0,:count*4,256]=1;sequence.fill_(count*4-1)
        expected=torch.full((2051,),-1,dtype=torch.int32)
        # maxnum excludes NaNs; selecting either NaN or -inf produces -1.
        order=sorted(range(count),key=lambda i:(-values[i] if not math.isnan(values[i]) else float('inf'),i))[:512]
        for index,pool in enumerate(order):
            if not math.isnan(values[pool]) and values[pool]>-float('inf'):
                expected[index*4:index*4+4]=pool*4+torch.arange(4,dtype=torch.int32)
        for replay in range(3):
            for result,program,graph in zip(outputs,programs,graphs,strict=True):
                result.fill_(-991);result[:1024].copy_(scores.view(torch.int32))
                if replay<2:graph.replay()
                else:program.launch()
            torch.cuda.synchronize()
            for result in outputs:
                assert torch.equal(result[:2051].cpu(),expected)
                assert bool(result[2051:].eq(-991).all())
