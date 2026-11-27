"""R039 scheduling qualification must preserve the existing rejection gates."""
import ast
from pathlib import Path
import jinja2
import pytest
from triton.flagmega.codegen.triton.kernel_programs import _q8_program_options
from triton.flagmega.errors import CodegenError


def subject():
    call = dict(family='iq4_q8_projection', variant='q8_r003_v2',
                jit_module='triton.flagmega.codegen.triton.iq4_q8_multi_column',
                jit_entry='dp4a_two_columns',
                constexprs=dict(N=5120, K=6144, ROWS=5, MODE=0, COLS=2),
                compile_options=dict(num_warps=4, num_ctas=1, num_stages=3,
                                     maxnreg=56, enable_fp_fusion=False,
                                     enable_static_shared=True, ptx_options='--minnctapersm=1'))
    launch = dict(program_kind='ordinary', num_warps=4, maxnreg=80, enable_static_shared=True)
    return call, launch


def test_two_columns_and_original_four_columns_are_exact_contracts():
    call, launch = subject()
    _q8_program_options([call], launch)
    assert launch['maxnreg'] == 56
    call, launch = subject()
    call['constexprs']['COLS'] = 4
    call['jit_entry'] = 'dp4a_columns'
    call['compile_options']['maxnreg'] = 80
    _q8_program_options([call], launch)
    assert launch['maxnreg'] == 80


@pytest.mark.parametrize('field,value', [('COLS',1), ('COLS',4), ('MODE',1), ('K',17408), ('N',2147483648)])
def test_two_column_schedule_rejects_unqualified_constants(field, value):
    call, launch = subject()
    call['constexprs'][field] = value
    with pytest.raises(CodegenError):
        _q8_program_options([call], launch)


@pytest.mark.parametrize('mutation', ['entry', 'module', 'budget', 'upper_bound'])
def test_two_column_schedule_keeps_entry_and_register_gates(mutation):
    call, launch = subject()
    if mutation == 'entry': call['jit_entry'] = 'dp4a_columns'
    if mutation == 'module': call['jit_module'] = 'wrong'
    if mutation == 'budget': call['compile_options']['maxnreg'] = 80
    if mutation == 'upper_bound': launch['maxnreg'] = 40
    with pytest.raises(CodegenError):
        _q8_program_options([call], launch)


@pytest.mark.parametrize('family', ['iq4_q8_projection', 'iq4_q8_interleaved'])
@pytest.mark.parametrize('columns,entry', [(None,'dp4a_m1'), (2,'dp4a_two_columns'), (4,'dp4a_columns')])
def test_rendered_wrapper_calls_the_declared_entry(family, columns, entry):
    from triton.flagmega.codegen.triton import iq4_q8_renderer
    template=Path(iq4_q8_renderer.__file__).parent/'kernels'/family/'q8_r003_v2.py.jinja'
    constants=dict(N=5120,K=6144,ROWS=1 if columns is None else 5,MODE=0)
    if columns is not None:constants['COLS']=columns
    source=jinja2.Template(template.read_text()).render(render_calls=[dict(
        family=family,symbol='project',signature='Q,W,U,Y,Z',
        constexprs=constants,pointers=['Q','W','U','Y','Z'])])
    tree=ast.parse(source)
    imports={n.asname or n.name:n.name for s in tree.body if isinstance(s,ast.ImportFrom) for n in s.names}
    function=next(n for n in tree.body if isinstance(n,ast.FunctionDef))
    call=next(n for n in ast.walk(function) if isinstance(n,ast.Call))
    assert imports[call.func.id]==entry
