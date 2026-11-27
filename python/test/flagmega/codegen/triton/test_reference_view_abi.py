# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

from triton.flagmega.codegen.triton.tir_package import describe_tir_package, render_tir_package
from triton.flagmega.compiler import Compiler
from python.test.flagmega.passes.tir.bufferize.test_ref_slice import state_slice_graph


def test_reusable_reference_slice_uses_one_body_runtime_scalar_and_wide_offsets():
    module = Compiler().compile(state_slice_graph(reusable=True)).module
    source = render_tir_package(describe_tir_package(module), "test")
    compile(source, "generated.py", "exec")
    assert source.count("def _flagmega_function_worker__consumer(") == 1
    assert source.count("    _flagmega_function_worker__consumer(") == 2
    assert "entry_layer: tl.int32" in source
    assert "@triton.jit(do_not_specialize=['entry_layer'])" in source
    assert "layer: tl.constexpr" not in source
    assert "(layer).to(tl.int64) * 96" in source
