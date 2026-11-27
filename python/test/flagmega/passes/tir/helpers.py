# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

from triton.flagmega import ir as fm


def make_prim_module() -> fm.IRModule:
    tensor = fm.tensor_type("float32", (4,))
    input_buffer = fm.T.buffer(
        "input",
        "float32",
        fm.T.mem_span(fm.T.physical_buffer("input.storage", "global", 16, 16, role="input")),
        (4,),
        (1,),
    )
    output_buffer = fm.T.buffer(
        "output",
        "float32",
        fm.T.mem_span(fm.T.physical_buffer("output.storage", "global", 16, 16, role="output")),
        (4,),
        (1,),
    )
    index = fm.dim("i", minimum=0, maximum=3)
    loop = fm.T.for_loop(
        index,
        fm.T.range(0, 4),
        fm.T.LoopMode.SERIAL,
        fm.T.sequential((fm.T.buffer_store(
            output_buffer, (index,), fm.T.buffer_load(input_buffer, (index,))
        ),)),
    )
    prim = fm.T.prim_function(
        "copy_4",
        "triton",
        (
            fm.T.prim_parameter("input", tensor, fm.T.PrimParameterRole.INPUT),
            fm.T.prim_parameter("output", tensor, fm.T.PrimParameterRole.OUTPUT),
        ),
        fm.T.sequential((loop,)),
        fm.T.return_((fm.T.return_binding(output_buffer, "output"),)),
    )
    builder = fm.IRBuilder(dialect="semantic_tir", stage="tir_selected")
    source = builder.var("source", tensor, id="source")
    builder.prim_function(prim)
    result = builder.call("tir.call", (source,), tensor, id="result", attrs={"callee": "copy_4"})
    builder.function("main", (source,), (result,))
    return fm.verify_module(builder.build(entry="main"))
