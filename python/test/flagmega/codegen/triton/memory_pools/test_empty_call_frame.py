# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""A used, zero-byte pool still belongs to the reusable function's pointer ABI."""

import pytest
from dataclasses import replace
from triton.flagmega import ir as fm
from triton.flagmega.codegen.triton import describe_tir_package
from triton.flagmega.codegen.triton.runtime_binding import describe_function_runtime_binding
from triton.flagmega.compiler import Compiler
from triton.flagmega.options import CompileOptions
from triton.flagmega.errors import IRVerificationError


def empty_chain(*, nested, distributed=False):
    ty = fm.tensor_type("bfloat16", (0, 7))
    scalar_ty = fm.tensor_type("bfloat16", (1, 1))
    output_ty = fm.tensor_type("float32", (0, 7))
    metadata = {}
    if distributed:
        placement = fm.Placement((2, 2), "yx", "bb")
        ty, scalar_ty, output_ty = (fm.DistributedType(t, (fm.SBP.broadcast(),) * 2, placement)
                                  for t in (ty, scalar_ty, output_ty))
        metadata = {"auto_distribution": {"placement": placement.to_data()}}

    class Graph(fm.Module):
        def forward(self):
            x = self.input("x", ty, id="x")
            scale = self.input("scale", scalar_ty, id="scale")
            prepared = fm.F.tensors.broadcast_to(scale, shape=(0, 7), name="prepared")
            product = fm.F.math.mul(prepared, x, name="product")
            wide = fm.F.tensors.cast(product, "float32", name="wide")
            result = fm.F.math.add(wide, wide, name="result")
            self.function("worker", (x, scale), (result,), attrs={"noinline": True, "reusable": True})
            callee = "worker"
            if nested:
                outer = self.input("outer", ty, id="outer")
                outer_scale = self.input("outer_scale", scalar_ty, id="outer_scale")
                output = fm.F.builtin.call(outer, outer_scale, result_type=output_ty, callee=callee, name="outer_call")
                self.function("wrapper", (outer, outer_scale), (output,), attrs={"noinline": True, "reusable": True})
                callee = "wrapper"
            inputs, outputs = [], []
            for index in range(2):
                value = self.input(f"value_{index}", ty, id=f"value_{index}")
                scalar = self.input(f"scale_{index}", scalar_ty, id=f"scale_{index}")
                output = fm.F.builtin.call(value, scalar, result_type=output_ty, callee=callee, name=f"call_{index}")
                inputs.extend((value, scalar))
                outputs.append(output)
            self.function("main", inputs, outputs)

    return Graph(dialect="nn", stage="frozen_constants", entry="main", metadata=metadata).build()


@pytest.mark.parametrize("level", ["fast", "optimized"])
@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("distributed", [False, True])
def test_empty_pool_is_bound_through_reusable_calls(tmp_path, level, nested, distributed):
    module = Compiler(CompileOptions(bufferize_opt_level=level)).compile(
        empty_chain(nested=nested, distributed=distributed)).module
    package = describe_tir_package(module)
    assert [f["function"] for f in package["device_functions"]] == (["worker", "wrapper"] if nested else ["worker"])
    plan = fm.verify_buffer_plan(module)
    storage = "block_local_data" if distributed else "workspace"
    for name in ("main", "worker", *(("wrapper",) if nested else ())):
        binding = describe_function_runtime_binding(module, function_name=name)
        pool, = (p for p in binding["pools"] if p["storage"] == storage)
        assert pool["nbytes"] == 0
        fp = plan.function_map[name]
        assert fp.memory_pool_map[storage].allocations
        for call in fp.calls:
            frame = call.memory_pool_map[storage]
            assert frame.scope_bytes == 0 and frame.offset == 0 and frame.allocation is not None
    assert fm.load_module(fm.emit_module(module, tmp_path / "empty.py")) == module
    # The existence rule is enforced by the verifier, not just codegen.
    entry = plan.function_map["main"]
    damaged_entry = replace(entry, calls=tuple(replace(c, memory_pools=()) for c in entry.calls))
    damaged_plan = replace(plan, functions=tuple(
        damaged_entry if f.name == "main" else f for f in plan.functions))
    with pytest.raises(IRVerificationError, match="memory-pool frame set"):
        fm.verify_buffer_plan(replace(module, metadata={**module.metadata, "buffer_plan": damaged_plan.to_data()}))
