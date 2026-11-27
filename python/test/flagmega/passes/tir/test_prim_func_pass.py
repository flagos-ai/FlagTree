# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

from dataclasses import replace

from triton.flagmega import ir as fm
from triton.flagmega.passes import PassManager, PrimFuncPass
from triton.flagmega.passes.context import current_pass_context

from .helpers import make_prim_module


class _UnrollSerialLoops(fm.TIRRewriter):
    def visit_for(self, node):
        rewritten = self.generic_visit(node)
        if rewritten.mode is fm.LoopMode.SERIAL:
            self.is_mutated = True
            return replace(rewritten, mode=fm.LoopMode.UNROLLED)
        return rewritten


def test_prim_func_pass_runs_fresh_mutators_to_a_real_fixed_point():
    created = []

    def factory():
        created.append(object())
        return _UnrollSerialLoops()

    result = PassManager("tir").add(
        PrimFuncPass("UnrollSerial").add(factory)
    ).run(make_prim_module())
    function = result.module.prim_function_map["copy_4"]
    loop = function.body.fields[0]

    assert result.executed == ("UnrollSerial",)
    assert loop.mode is fm.LoopMode.UNROLLED
    # First instance mutates, a fresh second instance proves the fixed point.
    assert len(created) == 2


def test_prim_func_pass_context_is_scoped_to_prim_function_name():
    observed = []

    class Observe(fm.TIRRewriter):
        def visit_prim_function(self, node):
            observed.append(current_pass_context().function)
            return self.generic_visit(node)

    PassManager("tir").add(PrimFuncPass("Observe").add(Observe)).run(make_prim_module())

    assert observed == ["copy_4"]
