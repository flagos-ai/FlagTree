# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT

from dataclasses import replace

import pytest

from triton.flagmega import ir as fm
from triton.flagmega.errors import IRVerificationError
from triton.flagmega.passes.tir import materialize_kernel_prim_functions
from triton.flagmega.passes.tir.bufferize import BufferizationOptions


def test_private_callee_pool_reuses_after_return_while_result_stays_live():
    b = fm.IRBuilder(dialect="semantic_tir", stage="selected_tir")
    t = fm.tensor_type("float32", (16,))
    parameter = b.var("parameter", t, id="parameter")
    result = b.call("tir.kernel", (parameter,), t, id="result", attrs={
        "semantic_op": "math.silu", "candidate": "tir.unary.local",
        "parameters": {"family": "unary", "variant": "local", "workspaces": ({
            "name": "scratch", "type": fm.tensor_type("uint8", (256,)),
            "memory_space": "scratch", "alignment": 64},)}, "facts": {}, "semantic_attrs": {}})
    b.function("worker", (parameter,), (result,))
    source = b.var("source", t, id="source")
    first = b.call("tir.call", (source,), t, id="first", attrs={"callee": "worker"})
    second = b.call("tir.call", (first,), t, id="second", attrs={"callee": "worker"})
    output = b.call("math.add", (first, second), t, id="output")
    b.function("main", (source,), (output,))
    module = materialize_kernel_prim_functions(b.build(entry="main"))
    generic = BufferizationOptions.generic()
    scratch = fm.MemorySpace("scratch", "device", 64, 256, fm.AllocationStrategy.REUSE)
    options = replace(generic, optimization_level="fast", memory_spaces=(*generic.memory_spaces, scratch))
    plan = fm.make_buffer_plan(module, options=options)
    first_pool = plan.call_map["first"].memory_pool_map["scratch"]
    second_pool = plan.call_map["second"].memory_pool_map["scratch"]
    assert first_pool.offset == second_pool.offset == 0
    assert plan.function_memory_space_bytes("main", "scratch") == 256
    first_frame = plan.physical_buffer_map[first_pool.allocation]
    second_frame = plan.physical_buffer_map[second_pool.allocation]
    assert first_frame.live_end < second_frame.live_start
    assert plan.buffer_map["first"].live_end >= second_frame.live_start


def test_external_storage_requires_explicit_source_annotation():
    b = fm.IRBuilder(dialect="tir", stage="selected_tir")
    t = fm.tensor_type("float32", (16,))
    source = b.var("source", t, id="source")
    result = b.call("math.silu", (source,), t, id="result")
    b.function("main", (source,), (result,))
    module = b.build(entry="main")
    module = replace(module, nodes=tuple(replace(n, metadata={"bufferization.external_memory_space": "packed"})
                                        if n.id == "source" else n for n in module.nodes))
    generic = BufferizationOptions.generic()
    packed = fm.MemorySpace("packed", "device", 64, 4096, fm.AllocationStrategy.EXTERNAL,
                            allocation_scope=fm.MemoryAllocationScope.EXTERNAL)
    plan = fm.make_buffer_plan(module, options=replace(generic, memory_spaces=(*generic.memory_spaces, packed)))
    buffered = replace(module, dialect="bufferized_tir", stage="allocated_tir", metadata={"buffer_plan": plan.to_data()})
    assert fm.verify_buffer_plan(buffered).buffer_map["source"].mem_span.buffer.memory_space == "packed"
    changed = replace(buffered, nodes=tuple(replace(n, metadata={}) if n.id == "source" else n for n in buffered.nodes))
    with pytest.raises(IRVerificationError, match="external"):
        fm.verify_buffer_plan(changed)
