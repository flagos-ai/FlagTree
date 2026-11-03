"""Full output ownership and ordered RMS remain valid with eight consumers."""
import ast
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent


def _body():
    module = ast.parse((ROOT/'fused-body.py.jinja').read_text())
    return next(n for n in module.body if isinstance(n,ast.FunctionDef))


def test_all_hidden_outputs_have_exactly_one_active_owner():
    expression = next(n.value for n in _body().body if isinstance(n,ast.Assign)
                      and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='own_hidden')
    code=compile(ast.Expression(expression),str(ROOT/'fused-body.py.jinja'),'eval')
    class TL:
        @staticmethod
        def arange(a,b):return np.arange(a,b)
        @staticmethod
        def program_id(axis):return cta
    output=[]
    for cta in range(120,128):output.extend(eval(code,{'tl':TL}).tolist())
    assert Counter(output) == Counter(range(4096))
    assert not any(isinstance(n,ast.If) for n in _body().body)


def test_full_RMS_and_native_bfloat16_conversion_order_preserved():
    body=_body().body
    prefix=body[:next(i for i,n in enumerate(body) if isinstance(n,ast.Assign)
        and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='own_hidden')]
    total_terms=[ast.unparse(n.value.right) for n in prefix if isinstance(n,ast.Assign)
        and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='total' and isinstance(n.value,ast.BinOp)]
    assert total_terms == [f'tl.reshape(v{i}, (64,))' for i in (0,8,1,9,2,10,3,11,4,12,5,13,6,14,7,15)]
    loops=[n for n in prefix if isinstance(n,ast.For)]
    assert [ast.unparse(n.iter) for n in loops] == ['tl.static_range(8)','tl.range(0, 4096, 1024)']
    tail=body[len(prefix):]
    own_fma=[ast.unparse(n.value) for n in tail if isinstance(n,ast.Assign)
        and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='own_value' and isinstance(n.value,ast.Call)
        and ast.unparse(n.value.func)=='tl.fma']
    assert len(own_fma)==4
    for i,expression in enumerate(own_fma):
        assert expression.startswith(f'tl.fma(p{i}, tl.load(')
        assert '.to(tl.bfloat16).to(tl.float32), own_value)' in expression
