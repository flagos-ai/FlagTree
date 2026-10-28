# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""SME projections retain independent K-owner partials and packed origins."""

from copy import deepcopy
import importlib.util
from math import prod

import pytest

from triton.flagmega import ir as fm
from triton.flagmega.codegen.triton.call_abi import describe_local_buffer_abi
from triton.flagmega.codegen.triton.corex_sme_abi import attach_flat_sme_calls
from triton.flagmega.codegen.triton.physical_access import (
    emit_buffer_pointer, emit_local_scalar_offset,
)
from triton.flagmega.codegen.triton.templates import KernelTemplateSpec, TritonTemplateRegistry


def _call(n=512, k=512, owners=(4, 4), typed=True, compact_weight=False):
    mesh = fm.Placement(owners, "yx", "bb")
    broadcast = fm.SBP.broadcast()
    buffers = {}
    for name in ("lhs", "rhs", "result"):
        partial = None
        if name == "lhs":
            dtype = fm.DType.BFLOAT16
            shape, local = (1, k*owners[1]), (1, k)
            policies = (broadcast, fm.SBP.split_contiguous((1,), k))
        elif name == "rhs":
            dtype = fm.VectorType(fm.DType.BFLOAT16, (8, 2, 8)) if typed else fm.DType.BFLOAT16
            shape = (k*owners[1]//16, n*owners[0]//8) + (() if typed else (2, 64))
            local = (k//16, n//8) + (() if typed else (2, 64))
            policies = (fm.SBP.split_contiguous((1,), k//16),
                        fm.SBP.split_contiguous((0,), n//8)) + (broadcast,)*(len(shape)-2)
        else:
            dtype = fm.VectorType(fm.DType.FLOAT32, (2, 4))
            shape, local = (1, n*owners[0]//8), (1, n//8)
            policies = (broadcast, fm.SBP.split_contiguous((0,), n//8))
            partial = fm.SBP.partial((1,), fm.ReduceOp.SUM)
        compact = name != "rhs" or compact_weight
        storage_shape = local if compact else shape
        strides = tuple(prod(storage_shape[i+1:]) for i in range(len(shape)))
        size = prod(storage_shape)*dtype.itemsize
        physical = fm.PhysicalBuffer(name, "input", size*prod(owners), 256)
        descriptor = fm.BufferDescriptor(
            name, dtype, shape, strides, "input", 256,
            fm.MemSpan(physical, 0, size),
            distributed_type=fm.DistributedType(fm.tensor_type(dtype, shape), policies, mesh, partial=partial),
            distributed_storage_kind=(fm.DistributedBufferStorageKind.COMPACT_PER_OWNER if compact
                                      else fm.DistributedBufferStorageKind.CANONICAL_GLOBAL),
        )
        buffers[name] = dict(formal=name, runtime_argument=name.upper(),
                             abi=describe_local_buffer_abi(descriptor))
    return dict(family="dense_matmul", variant="split_k_n_packed_k_major_gemv",
                packed_layout="k_major_n8_k16", local_m_capacity=1,
                local_n_capacity=n, local_k_capacity=k, tile_n=128, block_k=64,
                inputs=[dict(buffers=[buffers[name]]) for name in ("lhs", "rhs")],
                outputs=[dict(buffers=[buffers["result"]])])


@pytest.mark.parametrize("typed,compact_weight", [(True, False), (False, False), (True, True)])
def test_split_projection_proves_separate_sum_owner_storage(typed, compact_weight):
    call = _call(typed=typed, compact_weight=compact_weight)
    attach_flat_sme_calls({"projection": [call]}, None)
    selected = call["corex_flat_sme_gemv"]
    assert selected["weight_k_group_stride"] == (512 if compact_weight else 2048)*16
    assert "shard_index" in selected["source"]
    assert "shard_index" in selected["result"]
    assert ("shard_index" in selected["weight"]) == compact_weight


@pytest.mark.parametrize("change", [
    "partial_axes", "partial_op", "no_partial", "owner_stride", "owner_count",
    "canonical_result", "alignment", "origin", "k_tail", "ragged_owner",
])
def test_incomplete_split_projection_proofs_do_not_select_sme(change):
    call = _call()
    x, w, y = [p["buffers"][0]["abi"] for group in ("inputs", "outputs") for p in call[group]]
    if change == "partial_axes":
        y["distributed_type"]["partial"]["axes"] = [0]
    elif change == "partial_op":
        y["distributed_type"]["partial"]["reduce_op"] = "max"
    elif change == "no_partial":
        y["distributed_type"]["partial"] = None
    elif change == "owner_stride":
        y["component_stride_scalar_elements"] = 511
    elif change == "owner_count":
        y["owner_count"] = 4
    elif change == "canonical_result":
        y["storage_kind"] = "canonical_global"
    elif change == "alignment":
        w["alignment_bytes"] = 1
    elif change == "origin":
        w["distributed_type"]["axis_policies"][0]["stages"][0]["hierarchy_axes"] = [0]
        w["distributed_type"]["axis_policies"][1]["stages"][0]["hierarchy_axes"] = [1]
    elif change == "k_tail":
        call["local_k_capacity"] = 516
    elif change == "ragged_owner":
        w["distributed_type"]["tensor"]["shape"][1]["value"] = 252
        w["distributed_type"]["axis_policies"][1]["stages"][0]["distribution"]["granularity"]["value"] = 63
    attach_flat_sme_calls({"projection": [call]}, None)
    assert "corex_flat_sme_gemv" not in call


def _render(call, warps, owners):
    return TritonTemplateRegistry().render_kernel(
        KernelTemplateSpec("dense_matmul", "split_k_n_packed_k_major_gemv",
                           "corex" if warps == 16 else None, "bi_v150" if warps == 16 else None),
        dict(render_calls=[call], num_warps=warps, distributed_entry=True,
             mesh_hierarchy=owners, mesh_axis_names=("block_y", "block_x")),
    ).source


@pytest.mark.parametrize("n,k,owners,typed,compact_weight,cancellation", [
    (512, 512, (4, 4), True, False, False),
    (280, 192, (2, 2), False, False, False),
    (256, 128, (2, 4), True, True, False),
    (264, 192, (2, 2), True, False, True),
    (264, 192, (2, 2), False, False, True),
    (512, 512, (4, 4), False, False, False),
    (256, 128, (2, 4), False, True, True),
    (264, 192, (2, 2), False, False, False),
    (512, 512, (4, 4), False, False, True),
    (512, 128, (2, 4), True, True, True),
    (520, 192, (4, 4), True, False, False),
    (520, 192, (4, 4), False, False, True),
    (768, 256, (2, 2), False, True, False),
])
def test_split_projection_matches_every_original_owner_partial(
    tmp_path, n, k, owners, typed, compact_weight, cancellation,
):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available() or not getattr(torch, "corex", False):
        pytest.skip("requires an Iluvatar COREX GPU")
    call = _call(n, k, owners, typed, compact_weight)
    x, w, y = [p["buffers"][0] for group in ("inputs", "outputs") for p in call[group]]
    call.update(symbol="run", signature="LHS, RHS, RESULT", source_active="True",
                result_active="True", output_type="tl.float32")
    for field, binding in (("source", x), ("weight", w), ("result", y)):
        call[field] = emit_buffer_pointer(binding["abi"], binding["runtime_argument"])
    call["global_n"], call["global_k"] = "dense_local_n_offsets", "dense_local_k_offsets"
    call["source_offset"] = emit_local_scalar_offset(x["abi"], ("0", "dense_local_k_offsets"))
    call["result_offset"] = emit_local_scalar_offset(
        y["abi"], ("0", "dense_local_n_offsets//8"), lane_coordinate="dense_local_n_offsets%8")
    lane = "dense_global_n[:,None]%8*16+dense_global_k[None,:]%16"
    coordinates = ("dense_global_k[None,:]//16", "dense_global_n[:,None]//8")
    call["weight_offset"] = emit_local_scalar_offset(
        w["abi"], coordinates if typed else (*coordinates, f"({lane})//64", f"({lane})%64"),
        lane_coordinate=lane if typed else None,
    )
    attach_flat_sme_calls({"projection": [call]}, None)
    assert "corex_flat_sme_gemv" in call
    generator = torch.Generator().manual_seed(749)
    source = torch.randn((prod(owners), k), generator=generator).bfloat16().cuda()
    dense = (torch.randn((owners[0]*n, owners[1]*k), generator=generator)/32).bfloat16()
    if cancellation:
        source.fill_(1)
        dense.zero_()
        for owner in range(owners[1]):
            offsets = torch.tensor([0, 1, 2, 3] if typed else [0, 16, 32, 48]) + owner*k
            dense[:, offsets] = torch.tensor([33554432., 1., -33554432., 1.]).bfloat16()
    packed = dense.reshape(owners[0], n//8, 8, owners[1], k//16, 16)
    weight = (packed.permute(0, 3, 4, 1, 2, 5) if compact_weight else
              packed.permute(3, 4, 0, 1, 2, 5)).contiguous().cuda()
    outputs = []
    for warps in (32, 16):
        rendered_call = deepcopy(call)
        if warps == 32:
            rendered_call.pop("corex_flat_sme_gemv")
        path = tmp_path/f"split_{warps}.py"
        path.write_text("import triton\nimport triton.language as tl\n"
                        "import triton.experimental.tle.language as tle\n"
                        "FLAGMEGA_GRID_MESH = tl.constexpr(tle.device_mesh("
                        f"{{'block': [('block_y',{owners[0]}),('block_x',{owners[1]})]}}))\n"
                        + _render(rendered_call, warps, owners))
        spec = importlib.util.spec_from_file_location(f"split_{warps}", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        output = torch.full((prod(owners)*n+17,), -7, device="cuda", dtype=torch.float32)
        module.run[(prod(owners),)](source, weight, output, num_warps=warps, enable_fp_fusion=False)
        with torch.cuda.graph(graph := torch.cuda.CUDAGraph()):
            module.run[(prod(owners),)](source, weight, output, num_warps=warps, enable_fp_fusion=False)
        for _ in range(10):
            graph.replay()
        torch.cuda.synchronize()
        outputs.append(output)
    assert torch.equal(*outputs)
    assert torch.equal(outputs[1][-17:], torch.full((17,), -7, device="cuda", dtype=torch.float32))
    if cancellation:
        assert torch.equal(outputs[1][:-17], torch.ones_like(outputs[1][:-17]))
