"""Publication, complete work ownership and named implementation regressions."""
import ast,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from triton.flagmega.codegen.triton.templates import TritonTemplateRegistry
from triton.flagmega.targets import NvidiaSm90Target
from local_optimizations import dsa_absorbed_overlap
from local_optimizations.joint_dsa_absorbed_pool_scores import implementation_model,wrapper,helpers,SCHEDULE
from local_optimizations.native_dsa_pool_scores import KEY

def context():
    call=dict(symbol='dsa',signature='workspace, result, device_dptr',advance_sequence='Advance',
        sequence_lengths='Sequence',joint_dsa_pool_scores=dict(index='tl.program_id(0)',count=64,long_count=96,long_min_base=1024),
        dsa_absorbed_overlap=dict(symbol='absorbed',arguments='workspace, device_dptr'))
    return dict(call=call,distributed_entry=True,grid_mesh=True,mesh_rank=3,mesh_hierarchy=(4,8,16),mesh_levels='cbb',
        mesh_axis_names=('device','block_y','block_x'),device_context_abi='runtime-handles/v2')

def test_joint_publish_before_selection_and_advance():
    registry=TritonTemplateRegistry()
    source=registry.environment.from_string('{% import "entrypoints/_common.py.jinja" as entry with context %}\n'+wrapper('unused')).render(context())
    fn=ast.parse(source).body[0];body=[ast.unparse(n) for n in fn.body]
    assert '__write' in body[0] and 'distributed_barrier' in body[1]
    assert '__pool_scores' in body[2] and 'absorbed(' in body[2] and 'distributed_barrier' in body[3]
    assert '__select' in body[4] and 'distributed_barrier' in body[5] and '__advance' in body[6]
    assert 'tl.program_id(0) < 64' in body[2]
    assert 'tl.load(Sequence).to(tl.int32) >= 1024' in body[2] and 'tl.program_id(0) < 96' in body[2]
    raw=registry.environment.loader.get_source(registry.environment,KEY)[0]
    changed=helpers(raw)
    assert 'call.joint_dsa_pool_scores' in changed and 'call.joint_dsa_sort_tiles' in changed

def test_named_joint_implementation_keeps_original_and_semantics():
    original=NvidiaSm90Target().triton_implementation_model
    selected=implementation_model(original)
    old=[v for v in original.implementations if v.family=='dsa_indexer' and v.variant=='decode']
    for v in old:
        assert selected.implementation(v.id)==v
        new=selected.implementation(v.id+'.joint_pool_query')
        assert new.family==v.family and new.variant==v.variant and new.requires==v.requires and new.shared_workspaces==v.shared_workspaces
        assert new.parameters['pool_score_schedule']==SCHEDULE and new.parameters['pool_score_ctas']==64 and new.parameters['absorbed_query_ctas']==64
        assert new.facts=={**v.facts,'internal_grid_barriers':3}
        assert (new.parameters['long_pool_score_ctas'],new.parameters['long_absorbed_query_ctas'],new.parameters['long_min_base'])==(96,32,1024)
        assert selected.preferences['dsa_indexer'][0]==new.id

def test_query_loop_covers_every_original_tile_and_legacy_default():
    registry=TritonTemplateRegistry();tree=ast.parse((Path(__file__).parent/'mla.py').read_text())
    cooperative=next(n.value for n in ast.walk(tree) if isinstance(n,ast.Constant) and isinstance(n.value,str) and 'for _fm_mla_task' in n.value)
    source=dsa_absorbed_overlap.transform(cooperative)
    line=next(v for v in source.splitlines() if 'for _fm_mla_task' in v)
    for score_ctas in (1,32,64):
        call=dict(dsa_absorbed_overlap=True,heads=16,output=512,tile_n=64)
        if score_ctas!=1:call['dsa_absorbed_score_ctas']=score_ctas
        rendered=registry.environment.from_string(line).render(call=call)
        loop=ast.parse(rendered.strip()+'\n    pass').body[0]
        assert ast.unparse(loop.iter.args[0])==f'tl.program_id(0) - {score_ctas}'
        count=ast.literal_eval(loop.iter.args[2]);assert count==128-score_ctas
        tasks=[i for cta in range(score_ctas,128) for i in range(cta-score_ctas,128,count)]
        assert sorted(tasks)==list(range(128)) and len(tasks)==len(set(tasks))
    scores=[i for cta in range(64) for i in range(cta,1024,64)]
    assert sorted(scores)==list(range(1024))
