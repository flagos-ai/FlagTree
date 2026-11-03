"""Verify complete tensor-body sharing and fail closed on unequal math."""
import ast
import pytest
from local_optimizations.shared_mhc_mix import share_source,SYMBOL
from triton.flagmega.errors import CodegenError


def fixture():
    fields=('partials','bias','scales','post','comb')
    calls=[]
    functions=[]
    for index in range(2):
        name='mix'+str(index)
        inputs={k:k+str(index) for k in fields}
        calls.append(dict(symbol=name,family='mhc_pre',variant='cooperative_stream_fma',projection_partials=inputs['partials'],base=inputs['bias'],scale=inputs['scales'],post=inputs['post'],comb=inputs['comb']))
        functions.append(f"""    @triton.jit(noinline=True)
    def {name}__mix({', '.join(inputs.values())}, device_dptr):
        _fm_cta = tl.program_id(0) - {index}
        _fm_ctas = tl.num_programs(0)
        _fm_i = tl.arange(0,4)
        work = tl.load({inputs['partials']} + _fm_i)
        value = work + tl.load({inputs['bias']} + _fm_i)
        value = value * tl.load({inputs['scales']})
        for iteration in tl.range(0,19,loop_unroll_factor=19):
            value = libdevice.div_rn(value,tl.sum(value,0)+1e-6)
        tl.store({inputs['post']} + _fm_i,value)
        tl.store({inputs['comb']} + _fm_i,value)
""")
    return 'def build():\n'+''.join(functions)+'    return mix0__mix, mix1__mix\n',calls


def test_two_full_bodies_become_one_with_only_scalar_pointer_abi():
    source,calls=fixture()
    output,receipt=share_source(source,calls)
    tree=ast.parse(output)
    helper=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name==SYMBOL)
    assert len(receipt)==2
    assert [a.arg for a in helper.args.args]==['Partials','Bias','Scales','Post','Comb','_fm_cta','_fm_ctas']
    assert source.count('loop_unroll_factor=19')==2
    assert output.count('loop_unroll_factor=19')==1
    for index in range(2):
        wrapper=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='mix'+str(index)+'__mix')
        assert ast.unparse(wrapper.body[0])==f'_fm_cta = tl.program_id(0) - {index}'
        assert ast.unparse(wrapper.decorator_list[0])=='triton.jit'
        call=wrapper.body[-1].value
        assert ast.unparse(call.func)==SYMBOL
        assert [ast.unparse(n) for n in call.args]==[k+str(index) for k in ('partials','bias','scales','post','comb')]+['_fm_cta','_fm_ctas']
        assert all(r['normalized_full_body_AST_equal'] for r in receipt)


@pytest.mark.parametrize('change',[('tl.range(0,19,loop_unroll_factor=19)','tl.range(0,18,loop_unroll_factor=18)'),('value * tl.load(scales1)','value + tl.load(scales1)'),('post1 + _fm_i','post1 + _fm_i + 1')])
def test_different_recurrence_math_or_output_geometry_is_rejected(change):
    source,calls=fixture()
    if 'scales1' in change[0] or 'post1' in change[0]:
        changed=source.replace(*change)
    else:
        before,after=source.rsplit(change[0],1);changed=before+change[1]+after
    assert changed!=source
    with pytest.raises(CodegenError,match='math/geometry differs'):
        share_source(changed,calls)


def test_single_body_is_preserved_without_new_function_or_workspace():
    source,calls=fixture()
    output,receipt=share_source(source,calls[:1])
    assert output==source and receipt==[]


def test_declared_scalar_reciprocal_helper_is_bound_during_full_body_sharing():
    from pathlib import Path
    source,calls=fixture()
    helper=(Path(__file__).parent/'sigmoid-reciprocal.py.jinja').read_text()
    source=source.replace('def build():\n','def build():\n'+''.join('    '+line+'\n' for line in helper.splitlines()))
    for index in range(2):
        before=f'value = value * tl.load(scales{index})'
        after=f'value = _flagmega_sigmoid_unit_reciprocal(1. + libdevice.exp(-value)) * tl.load(scales{index})'
        source=source.replace(before,after)
    output,receipt=share_source(source,calls)
    assert len(receipt)==2
    assert output.count('def _flagmega_sigmoid_unit_reciprocal(')==1
    canonical=next(n for n in ast.walk(ast.parse(output)) if isinstance(n,ast.FunctionDef) and n.name==SYMBOL)
    assert '_flagmega_sigmoid_unit_reciprocal(1.0 + libdevice.exp(-value))' in ast.unparse(canonical)


def test_undeclared_scalar_reciprocal_helper_is_rejected():
    source,calls=fixture()
    for index in range(2):
        source=source.replace(f'value = value * tl.load(scales{index})',
                              f'value = _flagmega_sigmoid_unit_reciprocal(value) * tl.load(scales{index})')
    with pytest.raises(CodegenError,match='must be declared'):
        share_source(source,calls)
