# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

from triton.flagmega import ir as fm
from triton.flagmega.compiler import Compiler


def _packed_matmul_module():
    builder = fm.IRBuilder(dialect="high_level", stage="packed")
    value_type = fm.tensor_type("bfloat16", (1, 1024))
    weight_type = fm.tensor_type(fm.vector_type("float8_e4m3fn", (2, 16)), (1024, 32))
    scale_type = fm.tensor_type("float32", (8, 8))
    value = builder.var("value", value_type, id="value")
    weight = builder.weight(
        "weight", weight_type, source="weights", key="weight", id="weight"
    )
    scale = builder.weight(
        "scale", scale_type, source="weights", key="scale", id="scale"
    )
    output = builder.call(
        "math.packed_block_scaled_matmul",
        (value, weight, scale),
        value_type,
        id="output",
        attrs={
            "weight_block_n": 128,
            "weight_block_k": 128,
            "k_pack": 2,
            "k_vector": 16,
            "packed_layout": "n_major_k_packed",
        },
    )
    builder.function("main", (value,), (output,))
    return builder.build(entry="main")


def test_buffer_plan_promotes_producer_abi_for_program_output_sharded_view():
    module = Compiler().compile(_packed_matmul_module()).module
    plan = fm.verify_buffer_plan(module)

    weight = plan.buffer_map["weight.reshard2"]
    local_output = plan.buffer_map["output"]
    gathered_output = plan.buffer_map["output.reshard4"]
    assert weight.distributed_storage_kind is fm.DistributedBufferStorageKind.CANONICAL_GLOBAL
    assert local_output.distributed_storage_kind is fm.DistributedBufferStorageKind.CANONICAL_GLOBAL
    assert local_output.local_shape == (1, 8)
    assert local_output.nbytes == 2048
    assert local_output.mem_span.buffer.nbytes == 2048
    assert local_output.strides == (1024, 1)
    assert gathered_output.distributed_storage_kind is fm.DistributedBufferStorageKind.CANONICAL_GLOBAL
    assert local_output.mem_span.must_alias(gathered_output.mem_span)

    matmul = next(
        function
        for function in module.kernel_definitions
        if fm.kernel_dispatch_of(function).semantic_op == "math.packed_block_scaled_matmul"
    )
    result = matmul.output_parameters[0].buffers[0]
    assert result.distributed_type == local_output.distributed_type
    assert result.distributed_storage_kind is fm.DistributedBufferStorageKind.CANONICAL_GLOBAL
    assert result.mem_span.size.fixed_value == 2048
    assert result.mem_span.buffer.size.fixed_value == 2048

    boxing = next(
        function
        for function in module.kernel_definitions
        if fm.kernel_dispatch_of(function).semantic_op == "distributed.boxing"
    )
    boxing_input = boxing.runtime_parameters[0].buffers[0]
    boxing_output = boxing.output_parameters[0].buffers[0]
    assert boxing_input.distributed_storage_kind is fm.DistributedBufferStorageKind.COMPACT_LOCAL
    # This is the entry tensor-load boxing, not the output sharded view above.
    # Its broadcast result is consumed owner-locally and therefore lives in
    # the replicated block pool; the program output remains canonical-global.
    assert boxing_output.distributed_storage_kind is fm.DistributedBufferStorageKind.COMPACT_LOCAL
    assert plan.buffer_map["value.reshard1"].storage == "block_local_data"
