"""Long query geometry must follow the complete score batch, keeping all columns."""
import ast
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from triton.flagmega.codegen.triton.templates import TritonTemplateRegistry
from triton.flagmega.targets import NvidiaSm90Target
from local_optimizations import dsa_absorbed_overlap
from local_optimizations.joint_dsa_absorbed_pool_scores import implementation_model, wrapper


def test_named_query_tile_follows_complete_score_batch():
    model = implementation_model(NvidiaSm90Target().triton_implementation_model)
    selected = [v for v in model.implementations if v.id.endswith('.joint_pool_query')]
    assert selected
    for impl in selected:
        assert impl.parameters['long_absorbed_query_tile_n'] == 256
        assert impl.parameters['long_absorbed_query_single_batch_pools'] == 96 * 8
        assert impl.parameters['long_absorbed_query_ctas'] == 32


def test_wrapper_dispatch_uses_real_visible_pool_count_and_keeps_all_barriers():
    registry = TritonTemplateRegistry()
    call = dict(symbol='dsa', signature='workspace, result, device_dptr',
                sequence_lengths='Sequence', result='Result', advance_sequence='Advance',
                joint_dsa_pool_scores=dict(count=64,long_count=96,long_min_base=1024),
                joint_dsa_sort_tiles=(16,64,256,512,1024), parallel_dsa_sort=True,index_topk=2048,
                adaptive_absorbed_query=dict(tile_n=256,single_batch_pools=768),
                dsa_absorbed_overlap=dict(symbol='absorbed',arguments='workspace,weight,device_dptr'))
    context = dict(call=call,distributed_entry=True,grid_mesh=True,mesh_rank=3,
                   mesh_hierarchy=(4,8,16),mesh_levels='cbb',
                   mesh_axis_names=('device','block_y','block_x'),device_context_abi='runtime-handles/v2')
    source = registry.environment.from_string('{% import "entrypoints/_common.py.jinja" as entry with context %}\n'+wrapper('unused')).render(context)
    fn = ast.parse(source).body[0]
    text = ast.unparse(fn)
    assert 'tl.load(Result + 2048)' in text
    assert 'absorbed__long96__wide(' in text
    assert 'absorbed__long96(' in text
    assert text.count('distributed_barrier') == 3
    assert 'tl.program_id(0) < 96' in text and 'tl.program_id(0) < 64' in text


def test_wide_helper_covers_every_column_without_changing_K_reduction():
    # These are the same task/column/K expressions in the normally rendered
    # cooperative MLA template. Only N task partitioning is changed.
    source = '''{% for call in render_calls %}
{% if call.family == "mla_projection" and call.variant == "mma" %}
@triton.jit(noinline=True)
def {{ call.symbol }}(workspace, weight, device_dptr):
    for _fm_mla_task in tl.range(tl.program_id(0) - {{ call.dsa_absorbed_score_ctas | default(1) }}, {{ call.heads * ((call.output + call.tile_n - 1) // call.tile_n) }}, {{ 128 - (call.dsa_absorbed_score_ctas | default(1)) }}):
        head = _fm_mla_task // {{ (call.output + call.tile_n - 1) // call.tile_n }}
        columns = (_fm_mla_task % {{ (call.output + call.tile_n - 1) // call.tile_n }}) * {{ call.tile_n }} + tl.arange(0, {{ call.tile_n }})
        for k in tl.range(0, {{ call.reduction }}, {{ call.reduction_tile }}):
            consume_full_K(head, columns, k)
{% endif %}
{% endfor %}
'''
    registry = TritonTemplateRegistry()
    call = dict(family='mla_projection',variant='mma',symbol='absorbed',heads=16,
                output=512,tile_n=64,reduction=256,reduction_tile=128,
                dsa_absorbed_score_ctas=64,dsa_adaptive_long_score_ctas=96,
                dsa_adaptive_query_tile_n=256)
    tree = ast.parse(registry.environment.from_string(dsa_absorbed_overlap.adaptive_helpers(source)).render(render_calls=[call]))
    wide = next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='absorbed__long96__wide')
    loop = wide.body[0]
    assert ast.unparse(loop.iter)=='tl.range(tl.program_id(0) - 96, 32, 32)'
    assert ast.unparse(loop.body[-1].iter)=='tl.range(0, 256, 128)'
    owners=[(task//2,col) for cta in range(32) for task in range(cta,32,32) for col in range(task%2*256,(task%2+1)*256)]
    assert sorted(owners)==[(head,col) for head in range(16) for col in range(512)]
