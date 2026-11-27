# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

import pytest

from triton.flagmega import ir as fm
from triton.flagmega.errors import IRSchemaError, IRVerificationError


def _buffer(name: str, *, role: str = "workspace") -> fm.Buffer:
    physical = fm.T.physical_buffer(f"{name}.storage", "global", 16, 16, role=role)
    return fm.T.buffer(name, "float32", fm.T.mem_span(physical), (4,), (1,))


def test_physical_buffer_memspan_and_strided_buffer_prove_byte_bounds():
    with pytest.raises(IRSchemaError, match="power of two"):
        fm.T.physical_buffer("bad", "global", 16, 3)
    physical = fm.T.physical_buffer("small", "global", 4, 4)
    with pytest.raises(IRSchemaError, match="exceeds PhysicalBuffer"):
        fm.T.mem_span(physical, 0, 8)
    padded = fm.T.physical_buffer("padded", "global", 16, 16)
    with pytest.raises(IRSchemaError, match="needs 24 bytes"):
        fm.T.buffer("view", "float32", fm.T.mem_span(padded), (2, 2), (4, 1))


def test_prim_function_rejects_abi_phase_regression_and_workspace_result():
    tensor = fm.tensor_type("float32", (4,))
    with pytest.raises(IRSchemaError, match="ordered inputs, outputs, workspaces"):
        fm.T.prim_function(
            "bad_order", "triton",
            (
                fm.T.prim_parameter("out", tensor, fm.T.PrimParameterRole.OUTPUT),
                fm.T.prim_parameter("late_input", tensor, fm.T.PrimParameterRole.INPUT),
            ),
            fm.T.sequential(),
        )

    workspace = _buffer("workspace")
    with pytest.raises(IRSchemaError, match="not an input/output ABI parameter"):
        fm.T.prim_function(
            "bad_result", "triton",
            (fm.T.prim_parameter(
                "workspace",
                tensor,
                fm.T.PrimParameterRole.WORKSPACE,
                memory_space="workspace",
            ),),
            fm.T.sequential(),
            fm.T.return_((fm.T.return_binding(workspace, "workspace"),)),
        )


def test_tir_verifier_rejects_undeclared_block_effects_and_input_writes():
    tensor = fm.tensor_type("float32", (4,))
    source = _buffer("source", role="input")
    output = _buffer("output", role="output")
    index = fm.dim("i", minimum=0, maximum=3)
    store = fm.T.buffer_store(output, (index,), fm.T.buffer_load(source, (index,)))
    bad_block = fm.T.block(
        "copy",
        fm.T.sequential((store,)),
        writes=(fm.T.buffer_region(output, (fm.T.range(0, 4),)),),
    )
    function = fm.T.prim_function(
        "missing_read", "triton",
        (
            fm.T.prim_parameter("source", tensor, fm.T.PrimParameterRole.INPUT),
            fm.T.prim_parameter("output", tensor, fm.T.PrimParameterRole.OUTPUT),
        ),
        fm.T.sequential((bad_block,)),
        fm.T.return_((fm.T.return_binding(output, "output"),)),
    )
    with pytest.raises(IRVerificationError, match="missing reads=.*source"):
        fm.verify_prim_function(function)

    write_input = fm.T.prim_function(
        "write_input", "triton",
        (fm.T.prim_parameter("source", tensor, fm.T.PrimParameterRole.INPUT),),
        fm.T.sequential((fm.T.buffer_store(
            source, (index,), fm.T.buffer_load(source, (index,))
        ),)),
    )
    with pytest.raises(IRVerificationError, match="writes read-only input"):
        fm.verify_prim_function(write_input)
