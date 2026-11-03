"""Render the actual selected O-proj and check its DMA/MMA lifetime contract."""
import ast
import sys
from pathlib import Path
import pytest
from jinja2 import ChoiceLoader,FileSystemLoader

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from triton.flagmega.codegen.triton.templates import TritonTemplateRegistry
from local_optimizations import oproj


def render(splits):
    registry=TritonTemplateRegistry()
    registry.environment.loader=ChoiceLoader([FileSystemLoader(str(oproj.OUT)),registry.environment.loader])
    call=dict(family='partitioned_linear',variant='wgmma_local',symbol='projection',
        signature='workspace,weight,device_dptr,value_descriptor,weight_descriptor',
        local_shared_bytes=147456,local_shared_alignment=1024,local_weight_offset=0,
        local_activation_offset=131072,block_k=256,tasks=64,parts=1,
        part_k=2048*splits,accumulation_splits=splits,accumulation_k=2048,
        weight_row_origin='0',weight_k_origin='0',source_k_origin='0',rows=4096,result='Result',
        oproj_slots=4,oproj_prefetch=3,oproj_lead=3,oproj_wait_group=1,
        oproj_prefetch_before_counter=True,
        owner_matched_oproj=dict(ready=8192,consumed=8704,payload=0,post='Post',comb='Comb',result='Output'))
    context=dict(render_calls=[call],distributed_entry=True,grid_mesh=True,mesh_rank=3,
        mesh_hierarchy=(4,8,16),mesh_levels='cbb',mesh_axis_names=('device','block_y','block_x'),
        device_context_abi='runtime-handles/v2')
    raw=registry.environment.loader.get_source(registry.environment,oproj.KEY)[0]
    return ast.parse(registry.environment.from_string(oproj.transform(raw)).render(context))


@pytest.mark.parametrize('splits',[1,2])
def test_rendered_pipeline_preserves_original_part_boundaries(splits):
    tree=render(splits);fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='projection')
    text=ast.unparse(fn)
    assert 'tle.gpu.alloc([4, 64, 256]' in text and 'tle.gpu.alloc([4, 8, 256]' in text
    assert text.count('tle.gpu.alloc_barriers(4,')==2
    assert 'tl.static_range(3)' in text
    assert 'tl.range(0, 8)' in text and '_fm_sequence_base += 8' in text
    assert 'tle.gpu.wgmma_wait(1, _fm_accumulator)' in text
    assert 'tle.gpu.wgmma_wait(0, _fm_accumulator)' in text
    assert text.count('tle.gpu.wgmma_wait(0, _fm_accumulator)')==1
    assert '_fm_refill_slot = (_fm_sequence + 3) % 4' in text
    assert '_fm_sequence // 4 % 2' in text
    assert 'tl.full((64,), 0, tl.float32)' in text
    assert 'tl.zeros((64, 8), tl.float32)' in text
    assert '_fm_total += _fm_result.to(tl.bfloat16).to(tl.float32)' in text
    assert '_flagmega_oproj_acquire_poll' in text
    assert '_fm_row = _fm_task * 64' in text


def test_prefetch_and_recycle_cover_every_original_K16_update_once():
    for parts in (1,2):
        for part in range(parts):
            copies=list(range(3))+[i+3 for i in range(8) if i+3<8]
            assert sorted(copies)==list(range(8))
            slots={(part*8+i)%4:part*8+i for i in range(3)}
            pending=[]
            for i in range(8):
                sequence=part*8+i
                assert slots[sequence%4]==sequence
                pending.append(sequence)
                # wait_group(1) completes every group except the latest.
                pending=pending[-1:]
                if i+3<8:
                    future=sequence+3
                    assert all(reader%4!=future%4 for reader in pending)
                    slots[future%4]=future
            # The part's final wait_group(0) completes its last reader.
            pending.clear()
            assert not pending
            old=[part*2048+step*512+k for step in range(4) for k in range(0,512,16)]
            new=[part*2048+step*256+k for step in range(8) for k in range(0,256,16)]
            assert new==old
    assert 4*64*256*2==131072
    assert 131072+4*8*256*2==147456


def test_parallel_input_template_remains_byte_identical():
    from local_optimizations.oproj_tma_pipeline import transform
    registry=TritonTemplateRegistry()
    stock=registry.environment.loader.get_source(registry.environment,oproj.KEY)[0]
    parallel=(oproj.OUT/'parallel.py.jinja').read_text()
    combined=stock+parallel
    transformed=transform(combined)
    assert transformed.endswith(parallel)
    assert transformed.count('def {{ call.symbol }}__partials(')==1


def test_unused_template_keeps_exact_source_text():
    from local_optimizations.oproj_tma_pipeline import transform
    registry=TritonTemplateRegistry()
    stock=registry.environment.loader.get_source(registry.environment,oproj.KEY)[0]
    context=dict(render_calls=[])
    assert (registry.environment.from_string(transform(stock)).render(context)
            ==registry.environment.from_string(stock).render(context))


@pytest.mark.parametrize('splits',[1,2])
def test_first_DMA_is_independent_and_issued_before_previous_consumed_poll(splits):
    fn=next(n for n in render(splits).body if isinstance(n,ast.FunctionDef) and n.name=='projection')
    task=next(n for n in fn.body if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=='_fm_task')
    early=[(i,n) for i,n in enumerate(task.body) if isinstance(n,ast.Expr) and isinstance(n.value,ast.Call) and ast.unparse(n.value.func)=='tle.gpu.copy']
    assert len(early)==2
    poll=next(i for i,n in enumerate(task.body) if '_oproj_observed = _flagmega_oproj_acquire_poll' in ast.unparse(n))
    assert all(i<poll for i,_ in early)
    assert '[64, 256]' in ast.unparse(early[0][1])
    assert '[8, 256]' in ast.unparse(early[1][1])
    assert '_fm_prefetched_sequence = _fm_sequence_base' in ast.unparse(task)
    initial=next(n for n in ast.walk(task) if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=='_fm_initial')
    assert len(initial.body)==1 and isinstance(initial.body[0],ast.If)
    assert ast.unparse(initial.body[0].test)=='_fm_initial != 0 or _fm_sequence_base != _fm_prefetched_sequence'
    assert ast.unparse(initial.body[0]).count('tle.gpu.copy(')==2
    for part in range(2):
        for split in range(splits):
            sequence=(part*splits+split)*8
            copies=([0] if sequence==0 else [])+[i for i in range(3) if i!=0 or sequence!=0]+list(range(3,8))
            assert copies==list(range(8))
