# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

from triton.flagmega import ir as fm
from triton.flagmega.codegen.triton.call_abi import describe_local_buffer_abi
from triton.flagmega.codegen.triton import render_triton_package
from triton.flagmega.codegen.triton.kernel_call_renderers import prepare_kernel_calls
from triton.flagmega.compiler import Compiler


def _descriptor(name, dtype, shape):
    nbytes = dtype.itemsize
    for extent in shape:
        nbytes *= extent
    physical = fm.PhysicalBuffer(
        f"physical:{name}", "workspace", fm.dim(nbytes), 16
    )
    strides = []
    stride = 1
    for extent in reversed(shape):
        strides.append(stride)
        stride *= extent
    return fm.BufferDescriptor(
        name,
        dtype,
        shape,
        tuple(reversed(strides)),
        "workspace",
        16,
        fm.MemSpan(physical),
    )


def _parameter(formal, descriptor):
    return {
        "formal": formal,
        "buffers": ({
            "formal": formal,
            "actual": descriptor.id,
            "runtime_argument": descriptor.id,
            "runtime_value_kind": "pointer",
            "abi": describe_local_buffer_abi(descriptor),
        },),
    }


def test_rope_renderer_scalarizes_input_and_double_lane_rotary_tables():
    value = _descriptor(
        "value", fm.vector_type("bfloat16", (8,)), (1, 8, 2)
    )
    table_type = fm.vector_type("float32", (2, 8))
    cos = _descriptor("cos", table_type, (1, 1, 1))
    sin = _descriptor("sin", table_type, (1, 1, 1))
    result = _descriptor(
        "result", fm.vector_type("bfloat16", (8,)), (1, 8, 2)
    )

    call = prepare_kernel_calls(({
        "call": "rope",
        "family": "rope",
        "variant": "decode",
        "execution_kind": "local_shard",
        "parameters": {"elements_per_program": 64},
        "semantic_attrs": {},
        "inputs": (
            _parameter("input", value),
            _parameter("cos", cos),
            _parameter("sin", sin),
        ),
        "outputs": (_parameter("result", result),),
        "workspaces": (),
    },), function_name="main")[0]

    assert call["local_capacity"] == 128
    assert call["head_dim"] == 16
    assert "% 8" in call["dimension"]
    assert "% 8" in call["source_offset"]
    assert "% 8" in call["partner_offset"]
    assert "% 16" in call["cosine_offset"]
    assert "% 16" in call["sine_offset"]
    assert "% 8" in call["result_offset"]


def test_scalar_rope_keeps_the_same_renderer_path_without_lane_coordinates():
    value = _descriptor("value", fm.DType.BFLOAT16, (1, 8, 16))
    cos = _descriptor("cos", fm.DType.FLOAT32, (1, 1, 16))
    sin = _descriptor("sin", fm.DType.FLOAT32, (1, 1, 16))
    result = _descriptor("result", fm.DType.BFLOAT16, (1, 8, 16))

    call = prepare_kernel_calls(({
        "call": "rope",
        "family": "rope",
        "variant": "decode",
        "execution_kind": "local_shard",
        "parameters": {"elements_per_program": 64},
        "semantic_attrs": {},
        "inputs": (
            _parameter("input", value),
            _parameter("cos", cos),
            _parameter("sin", sin),
        ),
        "outputs": (_parameter("result", result),),
        "workspaces": (),
    },), function_name="main")[0]

    assert call["local_capacity"] == 128
    assert call["head_dim"] == 16
    assert "% 8" not in call["dimension"]
    assert call["result_offset"]


def test_explicit_vectorized_rope_compiles_through_tir_and_package_rendering(tmp_path):
    class Graph(fm.Module):
        def __init__(self):
            super().__init__(
                dialect="high_level", stage="vectorized", entry="main"
            )

        def forward(self):
            value = self.input(
                "value",
                fm.tensor_type(fm.vector_type("bfloat16", (8,)), (1, 8, 2)),
                id="value",
            )
            table_type = fm.tensor_type(
                fm.vector_type("float32", (2, 8)), (1, 1, 1)
            )
            cos = self.input("cos", table_type, id="cos")
            sin = self.input("sin", table_type, id="sin")
            output = fm.F.ntt.vectorized_rope(
                value, cos, sin, name="output"
            )
            self.function("main", (value, cos, sin), (output,))

    compiled = Compiler().compile(Graph().build()).module
    # A logical entry argument may now select a local shard. Trace the public
    # result through its explicit materialization/view rather than assuming
    # the semantic kernel itself owns the final external buffer.
    pending = list(compiled.function_map["main"].outputs)
    seen, dispatches = set(), []
    while pending:
        node_id = pending.pop()
        if node_id in seen:
            continue
        seen.add(node_id)
        node = compiled.node_map[node_id]
        dispatch = fm.kernel_dispatch_for_call(compiled, node)
        if dispatch is not None and dispatch.semantic_op == "ntt.vectorized_rope":
            dispatches.append(dispatch)
        pending.extend(node.inputs)
    assert len(dispatches) == 1
    dispatch = dispatches[0]
    assert dispatch.semantic_op == "ntt.vectorized_rope"
    assert dispatch.microkernel is not None
    assert dispatch.microkernel.family == "rope"

    package = render_triton_package(compiled, tmp_path)
    source = (tmp_path / "generated_kernels.py").read_text(encoding="utf-8")
    assert package["kind"] == "tir_call_graph/v1"
    assert "rope_dimension" in source
    assert "tl.store" in source
    compile(source, "generated_kernels.py", "exec")
