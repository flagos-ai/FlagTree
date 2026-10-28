# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Flat SME selection requires the complete packed-row and owner ABI."""

import importlib.util
from math import prod

import pytest

from triton.flagmega import ir as fm
from triton.flagmega.codegen.triton.call_abi import describe_local_buffer_abi
from triton.flagmega.codegen.triton.corex_sme_abi import attach_flat_sme_calls
from triton.flagmega.codegen.triton.physical_access import emit_buffer_pointer, emit_local_scalar_offset
from triton.flagmega.codegen.triton.templates import KernelTemplateSpec, TritonTemplateRegistry


def _call(*, typed=False, compact=False, n=128, k=160, owners=(2, 2), tail=0):
    mesh = fm.Placement(owners, "yx", "bb")
    total_n = n * prod(owners) - tail
    broadcast = fm.SBP.broadcast()
    axes = tuple(range(len(owners)))
    buffers = {}
    for name in ("value", "gate_weight", "up_weight", "result"):
        packed = name in ("gate_weight", "up_weight")
        dtype = fm.VectorType(fm.DType.BFLOAT16, (8, 2, 8)
                              ) if typed and packed else fm.DType.BFLOAT16
        if packed:
            shape = (k // 16, total_n // 8) if typed else (k // 16, total_n // 8, 2, 64)
            sbp = (broadcast, fm.SBP.split_contiguous(axes, n // 8)) + \
                (broadcast,) * (len(shape) - 2)
        else:
            shape = (1, k if name == "value" else total_n)
            sbp = (broadcast, broadcast if name == "value" else fm.SBP.split_contiguous(axes, n))
        local_shape = shape if name == "value" else (
            (shape[0], n // 8, *shape[2:]) if packed else (1, n)
        )
        storage_shape = local_shape if compact else shape
        strides = tuple(prod(storage_shape[index + 1:]) for index in range(len(shape)))
        kind = (fm.DistributedBufferStorageKind.COMPACT_PER_OWNER if compact
                else fm.DistributedBufferStorageKind.CANONICAL_GLOBAL)
        physical = fm.PhysicalBuffer(name, "input", prod(
            shape) * dtype.itemsize * prod(owners), 256)
        descriptor = fm.BufferDescriptor(
            name, dtype, shape, strides, "input", 256,
            fm.MemSpan(physical, 0, prod(storage_shape) * dtype.itemsize),
            distributed_type=fm.DistributedType(fm.tensor_type(dtype, shape), sbp, mesh),
            distributed_storage_kind=kind,
        )
        buffers[name] = dict(formal=name, runtime_argument=name.upper(),
                             abi=describe_local_buffer_abi(descriptor))
    return dict(family="dense_matmul_glu", variant="packed_k_major_gemv",
                packed_layout="k_major_n8_k16", tile_n=128, block_k=32,
                local_n_capacity=n, local_k_capacity=k,
                inputs=[dict(buffers=[buffers[key]])
                        for key in ("value", "gate_weight", "up_weight")],
                outputs=[dict(buffers=[buffers["result"]])])


def _select(call, schedule=None):
    attach_flat_sme_calls({"glu": [call]}, schedule)
    return call.get("corex_flat_sme_glu")


@pytest.mark.parametrize("typed", (False, True))
@pytest.mark.parametrize("compact", (False, True))
def test_packed_owner_rows_support_both_storage_representations(typed, compact):
    call = _call(typed=typed, compact=compact)
    selected = _select(call)
    assert selected is not None
    assert selected["weight_k_group_stride"] == (128 if compact else 512) * 16
    assert ("shard_index" in selected["gate_weight"]) == compact
    assert ("shard_x" in selected["result"]) != compact


@pytest.mark.parametrize("change", (
    "alignment", "offset", "runtime_view", "k_tail", "n_tail", "ragged_owner",
    "stride", "component_stride", "pool_scope_stride", "dtype", "origin", "row_count", "mixed_storage", "source_capacity",
))
def test_incomplete_transport_proofs_keep_the_general_kernel(change):
    call = _call(tail=32 if change == "ragged_owner" else 0)
    source, gate, up = [parameter["buffers"][0]["abi"] for parameter in call["inputs"]]
    if change == "alignment":
        gate["alignment_bytes"] = 32
    elif change == "offset":
        gate["pool_byte_offset"] = 2
    elif change == "runtime_view":
        gate["view_byte_offset"] = {}
    elif change == "k_tail":
        call["local_k_capacity"] += 4
    elif change == "n_tail":
        call["local_n_capacity"] -= 1
    elif change == "stride":
        gate["scalar_storage_strides"] = (8192, 192, 64, 1)
    elif change == "component_stride":
        gate["component_stride_scalar_elements"] = 1
    elif change == "pool_scope_stride":
        gate["pool_scope_stride_bytes"] = 2
    elif change == "dtype":
        up["scalar_dtype"] = "float16"
    elif change == "origin":
        different = _call(n=64)
        up["distributed_type"] = different["inputs"][2]["buffers"][0]["abi"]["distributed_type"]
    elif change == "row_count":
        source["local_capacity_shape"] = (2, 160)
    elif change == "source_capacity":
        call["local_k_capacity"] -= 32
    elif change == "mixed_storage":
        call["inputs"][2] = _call(compact=True)["inputs"][2]
    assert _select(call) is None


def test_closed_function_pool_alignment_is_specific_to_each_formal():
    call = _call()
    gate, up = [parameter["buffers"][0] for parameter in call["inputs"][1:]]
    gate["abi"]["alignment_bytes"] = up["abi"]["alignment_bytes"] = 1
    schedule = dict(device_functions=[dict(function="glu", function_pool_abi=dict(bindings=[
        dict(formal="GATE_WEIGHT", alignment_bytes=256),
    ]))])
    assert _select(call, schedule) is None
    schedule["device_functions"][0]["function_pool_abi"]["bindings"].append(
        dict(formal="UP_WEIGHT", alignment_bytes=256)
    )
    assert _select(call, schedule) is not None


@pytest.mark.parametrize("typed,compact,n", (
    (False, False, 264), (True, False, 280), (True, True, 512),
))
@pytest.mark.parametrize("change", (None, "alignment", "runtime_view", "source_capacity", "k_tail"))
def test_gemv_transport_proves_owner_origins_and_partial_n32_rows(typed, compact, n, change):
    call = _call(typed=typed, compact=compact, n=n, k=192)
    call.update(family="dense_matmul", variant="packed_k_major_gemv",
                block_k=64, local_m_capacity=1)
    source, weight = [parameter["buffers"][0] for parameter in call["inputs"][:2]]
    source["formal"], weight["formal"] = "lhs", "rhs"
    call["inputs"] = call["inputs"][:2]
    if change == "alignment":
        weight["abi"]["alignment_bytes"] = 32
    elif change == "runtime_view":
        weight["abi"]["view_byte_offset"] = {}
    elif change == "source_capacity":
        source["abi"]["local_capacity_shape"] = (1, 128)
    elif change == "k_tail":
        call["local_k_capacity"] -= 8
    attach_flat_sme_calls({"gemv": [call]}, None)
    assert ("corex_flat_sme_gemv" in call) == (change is None)
    if change is None:
        selected = call["corex_flat_sme_gemv"]
        assert selected["weight_k_group_stride"] == n * (1 if compact else 4) * 16
        assert ("shard_index" in selected["weight"]) == compact


@pytest.mark.parametrize("compact", (False, True))
@pytest.mark.parametrize("change", (None, "alignment", "source_tail", "row_count", "weight_stride"))
def test_down_transport_requires_packed_row_and_scalar_source(compact, change):
    glu = _call(typed=True, compact=compact, n=128, k=512)
    source, weight = [parameter["buffers"][0] for parameter in glu["inputs"][:2]]
    result = glu["outputs"][0]["buffers"][0]
    residual = dict(result, formal="addend")
    source["formal"], weight["formal"], result["formal"] = "lhs", "rhs", "result_0"
    if change == "alignment":
        weight["abi"]["alignment_bytes"] = 32
    elif change == "source_tail":
        source["abi"]["local_capacity_shape"] = (1, 516)
    elif change == "row_count":
        source["abi"]["local_capacity_shape"] = (2, 512)
    elif change == "weight_stride":
        weight["abi"]["scalar_storage_strides"] = (8192, 192)
    stats = dict(formal="result_1", runtime_argument="S", abi=dict(
        local_capacity_shape=(1,), scalar_lane_count=1, scalar_dtype="float32"))
    call = dict(family="dense_matmul", variant="packed_k_major_gemv_norm_stats",
                packed_layout="k_major_n8_k16", block_k=128, tile_n=64,
                local_n_capacity=128, local_k_capacity=512,
                inputs=[dict(buffers=[item]) for item in (source, weight, residual)],
                outputs=[dict(buffers=[item]) for item in (result, stats)])
    attach_flat_sme_calls({"down": [call]}, None)
    assert ("corex_flat_sme_down" in call) == (change is None)


@pytest.mark.parametrize("n,k,round_activation,limits,cancellation", (
    (32, 64, False, (None, None), False),
    (64, 160, True, (0.75, 0.5), False),
    (128, 512, False, (0.75, 0.5), False),
    (256, 160, True, (None, None), False),
    (384, 512, False, (None, None), False),
    (512, 64, True, (0.75, 0.5), False),
    (32, 64, True, (None, None), True),
))
def test_warp_owned_sme_preserves_original_glu_arithmetic(
    tmp_path, n, k, round_activation, limits, cancellation,
):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available() or not getattr(torch, "corex", False):
        pytest.skip("requires an Iluvatar COREX GPU")
    call = dict(
        family="dense_matmul_glu", variant="packed_k_major_gemv", symbol="run",
        signature="X, W, U, Y", local_m_capacity=1, local_n_capacity=n, local_k_capacity=k,
        source="X", gate_weight="W", up_weight="U", result="Y",
        source_offset="dense_glu_local_k_offsets", result_offset="dense_glu_local_n_offsets",
        global_n="dense_glu_local_n_offsets", global_k="dense_glu_local_k_offsets",
        source_active="True", result_active="True", tile_n=128, block_k=32,
        output_type="tl.bfloat16", round_activation=round_activation,
        gate_clamp_max=limits[0], up_clamp=limits[1],
    )
    offset = (f"(dense_glu_global_k[None, :] // 16) * {n * 16} + "
              "(dense_glu_global_n[:, None] // 8) * 128 + "
              "(dense_glu_global_n[:, None] % 8) * 16 + dense_glu_global_k[None, :] % 16")
    call.update(gate_weight_offset=offset, up_weight_offset=offset)
    generator = torch.Generator().manual_seed(n + k)
    x = torch.randn(k, generator=generator).bfloat16().cuda()

    def weight():
        dense = (torch.randn((n, k), generator=generator) / 32).bfloat16()
        return dense.reshape(n // 8, 8, k // 16, 16).permute(2, 0, 1, 3).contiguous().cuda()

    gate, up = weight(), weight()
    if cancellation:
        # The original K32 tree adds adjacent register pairs, then distant
        # halves. Repeated adjacent pairs produce -1 instead of -2 here.
        streams = torch.tensor([
            1, 0, 1, 33554432, 1, -1, 0, -1,
            -33554432, -33554432, -1, -1, 33554432, -1, -33554432, 33554432,
            0, 33554432, -1, 1, -33554432, 1, 1, 0,
            0, -1, 0, 0, 1, -1, -1, 1,
        ], dtype=torch.bfloat16)
        x.fill_(1)
        gate_matrix = torch.zeros((n, k), dtype=torch.bfloat16)
        gate_matrix[:, 0] = 1
        up_matrix = torch.zeros_like(gate_matrix)
        up_matrix[:, :32] = streams
        gate = gate_matrix.reshape(n // 8, 8, k // 16, 16).permute(2, 0, 1, 3).contiguous().cuda()
        up = up_matrix.reshape(n // 8, 8, k // 16, 16).permute(2, 0, 1, 3).contiguous().cuda()

    results = []
    for corex, warps in ((False, 32), (True, 16)):
        if corex:
            call["corex_flat_sme_glu"] = dict(
                source="X", gate_weight="W", up_weight="U", result="Y", weight_k_group_stride=n * 16,
            )
        source = TritonTemplateRegistry().render_kernel(
            KernelTemplateSpec("dense_matmul_glu", "packed_k_major_gemv",
                               "corex" if corex else None, "bi_v150" if corex else None),
            dict(render_calls=[call], num_warps=warps,
                 distributed_entry=False, mesh_hierarchy=(1, 1)),
        ).source
        path = tmp_path / f"glu_{warps}.py"
        path.write_text("import triton\nimport triton.language as tl\n"
                        "import triton.experimental.tle.language as tle\n" + source)
        spec = importlib.util.spec_from_file_location(f"glu_{warps}", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        output = torch.empty(n, device="cuda", dtype=torch.bfloat16)
        module.run[(1,)](x, gate, up, output, num_warps=warps, enable_fp_fusion=False)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            module.run[(1,)](x, gate, up, output, num_warps=warps, enable_fp_fusion=False)
        for _ in range(10):
            graph.replay()
        torch.cuda.synchronize()
        results.append(output)
    assert torch.equal(*results)


@pytest.mark.parametrize("typed,compact_gate,compact_up", (
    (False, False, False), (True, True, True),
))
@pytest.mark.parametrize("family", ("dense_matmul_glu", "dense_matmul"))
def test_selected_transport_addresses_every_owner(tmp_path, typed, compact_gate, compact_up, family):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available() or not getattr(torch, "corex", False):
        pytest.skip("requires an Iluvatar COREX GPU")
    n, k, owners = (128, 160, (2, 2)) if family == "dense_matmul_glu" else (280, 192, (2, 2))
    call = _call(typed=typed, compact=compact_gate, n=n, k=k, owners=owners)
    other = _call(typed=typed, compact=compact_up, n=n, k=k, owners=owners)
    call["inputs"][2] = other["inputs"][2]
    if family == "dense_matmul_glu":
        assert _select(call) is not None
    x, gate, up, result = [parameter["buffers"][0] for group in ("inputs", "outputs")
                           for parameter in call[group]]
    call.update(symbol="run", signature="VALUE, GATE_WEIGHT, UP_WEIGHT, RESULT",
                local_m_capacity=1, source_active="True", result_active="True",
                global_n="dense_glu_local_n_offsets", global_k="dense_glu_local_k_offsets",
                output_type="tl.bfloat16", round_activation=False, gate_clamp_max=None, up_clamp=None)
    for field, binding in (("source", x), ("gate_weight", gate), ("up_weight", up), ("result", result)):
        call[field] = emit_buffer_pointer(binding["abi"], binding["runtime_argument"])
    call["source_offset"] = emit_local_scalar_offset(x["abi"], ("0", "dense_glu_local_k_offsets"))
    call["result_offset"] = emit_local_scalar_offset(
        result["abi"], ("0", "dense_glu_local_n_offsets"))
    lane = "((dense_glu_global_n[:, None] % 8) * 16 + dense_glu_global_k[None, :] % 16)"
    for field, binding in (("gate_weight_offset", gate), ("up_weight_offset", up)):
        coordinates = ("dense_glu_global_k[None, :] // 16", "dense_glu_global_n[:, None] // 8")
        call[field] = emit_local_scalar_offset(
            binding["abi"], coordinates if typed else (
                *coordinates, f"{lane} // 64", f"{lane} % 64"),
            lane_coordinate=lane if typed else None,
        )
    transport = "corex_flat_sme_glu"
    if family == "dense_matmul":
        call.pop(transport, None)
        call.update(family=family, variant="packed_k_major_gemv", block_k=64,
                    weight=call["gate_weight"], weight_offset=call["gate_weight_offset"])
        x["formal"], gate["formal"] = "lhs", "rhs"
        call["inputs"] = call["inputs"][:2]
        for field in ("global_n", "global_k", "source_offset", "result_offset", "weight_offset"):
            call[field] = call[field].replace("dense_glu_", "dense_")
        attach_flat_sme_calls({"gemv": [call]}, None)
        transport = "corex_flat_sme_gemv"
        assert transport in call
    generator = torch.Generator().manual_seed(739)
    value = torch.randn(k, generator=generator).bfloat16().cuda()
    if compact_gate:
        value = value.repeat(prod(owners))

    def weight(compact):
        dense = (torch.randn((n * prod(owners), k), generator=generator) / 32).bfloat16()
        packed = dense.reshape(prod(owners), n // 8, 8, k // 16, 16)
        return packed.permute(0, 3, 1, 2, 4).contiguous().cuda() if compact else (
            packed.permute(3, 0, 1, 2, 4).contiguous().cuda()
        )

    weights = weight(compact_gate), weight(compact_up)
    results = []
    for corex, warps in ((False, 32), (True, 16)):
        render_call = dict(call)
        if not corex:
            render_call.pop(transport)
        source = TritonTemplateRegistry().render_kernel(
            KernelTemplateSpec(family, "packed_k_major_gemv",
                               "corex" if corex else None, "bi_v150" if corex else None),
            dict(render_calls=[render_call], num_warps=warps, distributed_entry=True,
                 mesh_hierarchy=owners, mesh_levels="bb", mesh_axis_names=["block_y", "block_x"]),
        ).source
        path = tmp_path / f"owner_glu_{warps}.py"
        path.write_text("import triton\nimport triton.language as tl\n"
                        "import triton.experimental.tle.language as tle\n"
                        "FLAGMEGA_GRID_MESH = tl.constexpr(tle.device_mesh("
                        "{'block': [('block_y', 2), ('block_x', 2)]}))\n" + source)
        spec = importlib.util.spec_from_file_location(f"owner_glu_{warps}", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        output = torch.empty(n * prod(owners), device="cuda", dtype=torch.bfloat16)
        module.run[(prod(owners),)](value, *weights, output,
                                    num_warps=warps, enable_fp_fusion=False)
        torch.cuda.synchronize()
        results.append(output)
    assert torch.equal(*results)
