# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

from collections import Counter
from pathlib import Path
import re

from triton.flagmega.compiler import Compiler
from triton.flagmega.codegen.triton.call_abi import describe_function_call_abi
from triton.flagmega.codegen.triton import describe_tir_package
from triton.flagmega.codegen.triton import render_tir_package
from triton.flagmega.codegen.triton.function_schedule import (
    describe_function_schedule,
)
from triton.flagmega.codegen.triton.runtime_binding import (
    describe_function_runtime_binding,
)
from triton.flagmega.codegen.triton.function_call import (
    emit_function_call_arguments,
)
from triton.flagmega.importer import import_qwen3_model
from triton.flagmega import ir as fm
from triton.flagmega.runtime import (
    GeneratedTirPagedAttentionModelModule,
    create_tir_runtime,
)

from .helpers import full_checkpoint


def test_two_layer_model_compiles_every_compute_op_on_rank_two_mesh():
    module = import_qwen3_model(full_checkpoint(num_hidden_layers=2))

    result = Compiler().compile(module).module

    assert result.stage == "bufferized_tir"
    assert result.metadata["auto_distribution"]["placement"] == {
        "hierarchy": (8, 16),
        "hierarchy_levels": "bb",
        "name": "yx",
    }
    kernels = tuple(
        fm.kernel_dispatch_for_call(result, node)
        for node in result.nodes
        if fm.kernel_dispatch_for_call(result, node) is not None
    )
    semantics = Counter(node.semantic_op for node in kernels)
    assert semantics == Counter({
        "nn.embedding": 1,
        "nn.norm_apply": 3,
        "nn.norm_stats": 1,
        "nn.rotary_embedding": 1,
        "ntt.packed_qkv_parallel_linear_fused_rhs": 1,
        "nn.qkv_rope_with_cache": 1,
        # Function-boundary layout propagation preserves the decode layer's
        # packed local ABI, so only the public input requires Boxing.
        "distributed.boxing": 1,
        "ntt.paged_attention_partial": 1,
        "ntt.paged_attention_combine": 1,
        "ntt.matmul_norm_stats": 2,
        # AutoPacking is shape/layout based like nncase.  The target supplies
        # a packed K-major GEMV consumer when this tiny reduction cannot use
        # its full-tile producer/consumer pipeline.
        "nn.packed_dense_matmul_glu": 1,
        "ntt.packed_matmul": 1,
        "ntt.vectorized_cast": 1,
        "nn.greedy_sample": 1,
    })
    assert all(
        node.parameters["distribution"]["placement"]["hierarchy"]
        == (8, 16)
        for node in kernels
    )
    package = result.metadata["codegen_package_plan"]
    assert package["kind"] == "tir_call_graph"
    assert set(package) == {
        "schema", "kind", "profile", "calls", "kernel_templates", "launch"
    }
    assert any(
        call["family"] == "qkv_parallel_linear"
        and call["variant"] == "packed_fused_gemv"
        for call in package["calls"]
    )
    # The selected LM-head/cast contract is sharded.  The program boundary is
    # broadcast, so AutoDistributed must export the realized boxing view rather
    # than leaking the producer's local shard as the public logits value.  SSA
    # ids are deliberately not part of the function ABI contract.
    main_outputs = result.function_map["main"].outputs
    assert main_outputs[1:] == ("next_token", "updated_state")
    logits_output = result.node_map[main_outputs[0]]
    assert logits_output.op == "tir.buffer_view"
    assert len(logits_output.inputs) == 1
    logits_source = result.node_map[logits_output.inputs[0]]
    logits_dispatch = fm.kernel_dispatch_for_call(result, logits_source)
    assert logits_dispatch is not None
    assert logits_dispatch.semantic_op == "ntt.vectorized_cast"
    assert all(
        isinstance(policy, fm.SBPBroadCast)
        for policy in logits_output.type.axis_policies
    )
    assert result.function_map["decode_layer"].attrs["noinline"] is True
    assert sum(
        node.op == "tir.call" and node.attrs["callee"] == "decode_layer"
        for node in result.nodes
    ) == 2
    assert set(result.execution_function_map) == {"main", "decode_layer"}
    assert [
        call.callee
        for call in fm.execution_calls_of(result.execution_function_map["main"])
        if call.callee == "decode_layer"
    ] == ["decode_layer", "decode_layer"]
    assert len(fm.execution_calls_of(
        result.execution_function_map["decode_layer"]
    )) == 9
    assert all(
        result.kernel_callable_map[call.kernel].dispatch.semantic_op != "nn.rotary_embedding"
        for call in fm.execution_calls_of(result.execution_function_map["decode_layer"])
        if isinstance(call, fm.KernelInvoke)
    )
    assert sum(
        result.kernel_callable_map[call.kernel].dispatch.semantic_op == "nn.rotary_embedding"
        for call in fm.execution_calls_of(result.execution_function_map["main"])
        if isinstance(call, fm.KernelInvoke)
    ) == 1
    assert any(
        isinstance(value, fm.Barrier)
        for value in result.execution_function_map["decode_layer"].body.fields
    )

    main_abi = describe_function_call_abi(result, function_name="main")
    decode_abi = describe_function_call_abi(result, function_name="decode_layer")
    assert [
        call["semantic_op"] for call in main_abi["kernel_calls"]
    ] == [
        "distributed.boxing",
        "nn.embedding",
        "nn.rotary_embedding",
        "nn.norm_stats",
        "nn.norm_apply",
        "ntt.packed_matmul",
        "ntt.vectorized_cast",
        "nn.greedy_sample",
    ]
    assert sum(
        event["kind"] == "function_call" for event in main_abi["events"]
    ) == 2
    assert [
        event["callee"]
        for event in main_abi["events"]
        if event["kind"] == "function_call"
    ] == ["decode_layer", "decode_layer"]
    assert [
        call["semantic_op"] for call in decode_abi["kernel_calls"]
    ] == [
        "nn.norm_apply",
        "ntt.packed_qkv_parallel_linear_fused_rhs",
        "nn.qkv_rope_with_cache",
        "ntt.paged_attention_partial",
        "ntt.paged_attention_combine",
        "ntt.matmul_norm_stats",
        "nn.norm_apply",
        "nn.packed_dense_matmul_glu",
        "ntt.matmul_norm_stats",
    ]
    assert decode_abi["reusable"] is True
    assert decode_abi["noinline"] is True

    main_schedule = describe_function_schedule(result)
    decode_schedule = main_schedule["reachable_functions"]["decode_layer"]
    assert main_schedule["physical_strategy"] == "entry_schedule"
    assert decode_schedule["physical_strategy"] == "direct_noinline"
    assert [
        region["lowering"]
        for region in main_schedule["regions"]
        if region["kind"] == "function_schedule_call"
    ] == ["direct_device_call", "direct_device_call"]
    assert all(
        region["owner_participation"] == "all"
        for region in decode_schedule["regions"]
        if region["kind"] == "local_segment"
    )

    main_binding = describe_function_runtime_binding(result)
    decode_binding = describe_function_runtime_binding(
        result,
        function_name="decode_layer",
    )
    assert main_binding["signature"][-3:] == [
        "rdata", "workspace", "block_local_data",
    ]
    nested_events = [
        event
        for event in main_binding["call_abi"]["events"]
        if event["kind"] == "function_call"
    ]
    assert [
        [
            edge["actual_runtime_argument"]
            for edge in event["arguments"]
            if edge["actual_runtime_value_kind"] == "immediate"
        ]
        for event in nested_events
    ] == [["0", "False"], ["1", "True"]]
    assert all(
        {
            frame["memory_space"]: frame["runtime_argument"]
            for frame in event["memory_pools"]
        } == {
            "workspace": "workspace",
            "block_local_data": "block_local_data",
        }
        for event in nested_events
    )
    assert decode_binding["signature"][-3:] == [
        "rdata", "workspace", "block_local_data",
    ]
    assert all(
        "runtime_argument" in buffer
        for call in decode_binding["call_abi"]["kernel_calls"]
        for parameter in (*call["inputs"], *call["outputs"], *call["workspaces"])
        for buffer in parameter["buffers"]
    )
    nested_arguments = [
        emit_function_call_arguments(result, main_binding, event)
        for event in nested_events
    ]
    assert all(
        len(arguments) == len(decode_binding["signature"])
        for arguments in nested_arguments
    )
    rdata_index = decode_binding["signature"].index("rdata")
    workspace_index = decode_binding["signature"].index("workspace")
    block_local_index = decode_binding["signature"].index("block_local_data")
    assert [arguments[rdata_index] for arguments in nested_arguments] == [
        "rdata",
        "rdata",
    ]
    workspace_frames = [
        arguments[workspace_index] for arguments in nested_arguments
    ]
    assert workspace_frames[0] == workspace_frames[1]
    assert re.fullmatch(
        r"\(workspace \+ [1-9][0-9]*\)", workspace_frames[0]
    )
    block_local_frames = [
        arguments[block_local_index] for arguments in nested_arguments
    ]
    assert block_local_frames[0] == block_local_frames[1]
    assert re.fullmatch(
        r"_flagmega_block_scope_base\(block_local_data, [1-9][0-9]*\)",
        block_local_frames[0],
    )

    # Codegen expands the reusable function from its generic call ABI.  The
    # wrapper is emitted once, while call-site substitution keeps each layer's
    # independently packed readonly-data span.
    descriptor = describe_tir_package(result)
    rotary = next(
        call for call in descriptor["render_calls"]
        if call["family"] == "rotary_embedding"
    )
    assert rotary["local_capacity"] == 8
    assert rotary["tile"] == 8
    partial = next(
        call for call in descriptor["render_calls"]
        if call["family"] == "paged_attention_partial"
    )
    combine = next(
        call for call in descriptor["render_calls"]
        if call["family"] == "paged_attention_combine"
    )
    qkv_cache = next(
        call for call in descriptor["render_calls"]
        if call["family"] == "qkv_rope_with_cache"
    )
    norm_stats = next(
        call for call in descriptor["render_calls"]
        if call["family"] == "norm_stats"
    )
    norm_applies = [
        call for call in descriptor["render_calls"]
        if call["family"] == "norm_apply"
    ]
    assert not any(
        call["family"] == "rms_norm" for call in descriptor["render_calls"]
    )
    assert qkv_cache["variant"] == "decode"
    assert qkv_cache["q"]["outer_capacity"] == 2
    assert qkv_cache["k"]["outer_capacity"] == 1
    assert qkv_cache["q"]["normalization_size"] == 8
    assert qkv_cache["k"]["normalization_size"] == 8
    assert qkv_cache["v"]["capacity"] == 8
    assert qkv_cache["layer_id"] == "decode_layer_layer_id"
    assert qkv_cache["advance_sequence"] == "decode_layer_advance_sequence"
    assert norm_stats["variant"] == "local"
    assert norm_stats["partial_stats"] is None
    assert len(norm_applies) == 3
    assert all(call["variant"] == "local" for call in norm_applies)
    assert all(call["bias"] for call in norm_applies)
    assert {
        call["call"]: call["writer_active"] for call in norm_applies
    } == {
        # Each reusable-function output has one block-local backing per
        # physical owner, including the loop-carried decode input norm.
        "final_norm.vectorized.compute": "True",
        "decode_layer_input_norm.vectorized.compute": "True",
        "decode_layer_post_attention_norm.vectorized.compute": "True",
    }
    assert partial["context_shard"] == "shard_y"
    assert combine["partial_owner"] == (
        "((attention_part) * 16 + (shard_x))"
    )
    assert "shard_index" not in combine["partial_max"]
    assert combine["writer_active"] == "True"
    source = render_tir_package(descriptor, "flagmega-test")
    compile(source, "generated_kernels.py", "exec")
    assert "libdevice.fast_cosf(rotary_angle)" in source
    assert "libdevice.fast_sinf(rotary_angle)" in source
    assert "# flagmega-kernel: norm_stats/local platform=generic" in source
    assert "# flagmega-kernel: norm_apply/local platform=generic" in source
    assert "# flagmega-kernel: qkv_rope_with_cache/decode platform=generic" in source
    assert "norm_apply_bias" in source
    assert "norm_apply/distributed_rms" not in source
    assert "norm_apply/persistent_rms" not in source
    assert "for norm_outer_index in tl.range" in source
    assert "norm_reduction_offsets" in source
    assert "for qkv_q_outer_index in tl.range" in source
    assert "for qkv_k_outer_index in tl.range" in source
    assert "for qkv_v_start in tl.range" in source
    assert "partial_index = 0" in source
    assert "context_shard * num_query_heads" not in source
    assert (
        "dense_accumulator += "
        "_flagmega_dense_matmul_packed_k_major_gemv_accumulate("
        in source
    )
    assert (
        "_flagmega_dense_matmul_glu_packed_k_major_gemv_accumulate("
        in source
    )
    assert "def _flagmega_dense_gemv(" not in source
    assert "def _flagmega_dense_gemv_glu(" not in source
    assert "qwen" not in source.lower()
    runtime = create_tir_runtime(
        Path("."),
        {"codegen": {**descriptor, "kind": "tir_call_graph/v1"}},
        result,
        object(),
    )
    assert isinstance(runtime, GeneratedTirPagedAttentionModelModule)
    assert runtime.state_config.num_layers == 2
    # The returned RefType must alias its input state, so the five state
    # fields occur once in the external signature rather than as duplicate
    # output buffers.
    assert len(runtime.external_arguments) == 8
    decode_definition, = descriptor["device_functions"]
    qkv_events = [event for event in decode_definition["schedule"]["consumer_events"]
                  if event.get("family") == "qkv_parallel_linear"]
    assert len(qkv_events) == 1
    decode_calls = [event for event in descriptor["entry_events"]
                    if event["kind"] == "function_call"]
    assert [event["call"] for event in decode_calls] == [
        "layer_0_decode_layer_call",
        "layer_1_decode_layer_call",
    ]
    assert decode_calls[0]["symbol"] == decode_calls[1]["symbol"]
    assert decode_calls[0]["arguments"] != decode_calls[1]["arguments"]
    rdata_offsets = [
        tuple(re.findall(r"rdata.*?\+ (\d+)", event["arguments"]))
        for event in decode_calls
    ]
    assert all(rdata_offsets)
    assert rdata_offsets[0] != rdata_offsets[1]
    second_layer_barrier = next(
        event for event in descriptor["entry_events"]
        if event["kind"] == "barrier"
        and event["before"] == "layer_1_decode_layer_call"
    )
    assert second_layer_barrier["scope"] == "grid"
    assert second_layer_barrier["hazards"]


def test_vectorization_optimizes_the_reusable_decode_body_once():
    module = import_qwen3_model(full_checkpoint(num_hidden_layers=3))

    compiler = Compiler()
    decomposed = compiler.run_stage(module, "decompose-gdn").module
    candidates = compiler.run_stage(decomposed, "propose-vectorization").module
    vectorized = compiler.run_stage(candidates, "apply-vectorization").module

    compute = vectorized.node_map["decode_layer_after_attention.vectorized.compute"]
    assert compute.metadata["vectorization_inputs"] == (
        "decode_layer_hidden",
        "decode_layer_attention_output",
    )
    assert sum(node.op == "builtin.call" for node in vectorized.nodes) == 3


def test_qkv_offline_transpose_and_pack_are_lifted_to_each_call_actual():
    compiler = Compiler()
    current = import_qwen3_model(full_checkpoint(num_hidden_layers=2))
    for stage in (
        "decompose-gdn",
        "propose-vectorization",
        "apply-vectorization",
        "propose-packing",
        "apply-packing",
    ):
        current = compiler.run_stage(current, stage).module

    decode_layer = current.function_map["decode_layer"]
    packed_formals = tuple(
        value for value in decode_layer.parameters
        if "qkv_projection" in value and value.endswith(".parameter")
    )
    assert len(packed_formals) == 3
    for layer in range(2):
        call = current.node_map[f"layer_{layer}_decode_layer_call"]
        call_inputs = set(call.inputs)
        lifted_roots = tuple(
            node
            for node in current.nodes
            if node.id.startswith(f"layer_{layer}_decode_layer_call.")
            and "qkv_projection" in node.id
            and node.metadata.get("lifted_from_function") == "decode_layer"
            and node.metadata.get("packed_layout") == "k_major"
        )
        assert len(lifted_roots) == 3
        assert {node.id for node in lifted_roots} <= call_inputs
