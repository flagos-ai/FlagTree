# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Owner-local N tiles preserve Q/K/V slices and explicit BF16 partials."""

from dataclasses import replace
from functools import lru_cache
from itertools import product

import pytest

from triton.flagmega.codegen.triton.tir_package import describe_tir_package
from triton.flagmega.targets import NvidiaSm90Target

from .conftest import _packed_qkv_mma_pipeline_module


def implementation_id(kind, descriptor):
    return f"tir.qkv_parallel_linear.packed_partial_{kind}_n_tiled_{descriptor}_pipeline"


@lru_cache(maxsize=None)
def tiled_module(kind, descriptor, widths, kv_heads):
    return _packed_qkv_mma_pipeline_module(implementation_id(kind, descriptor),
                                         projection_widths=widths, num_kv_heads=kv_heads)


@pytest.mark.parametrize("kind", ("gemv", "mma"))
@pytest.mark.parametrize("descriptor", ("single", "table"))
def test_n_tiled_candidate_is_declared_without_changing_default(kind, descriptor):
    model = NvidiaSm90Target().triton_implementation_model
    identity = implementation_id(kind, descriptor)
    assert identity in {value.id for value in model.implementations}
    candidate = model.implementation(identity)
    assert candidate.parameters["n_tiling"] is True
    assert "required_local_output_extent" not in candidate.contract
    assert candidate.contract["required_output_partial_reduce_op"] == "sum"
    assert candidate.contract["requires_partial_axes_match_input_reduction_ownership"] is True
    assert identity not in model.preferences["qkv_parallel_linear"]


@pytest.mark.parametrize("kind,descriptor", (("gemv", "single"), ("mma", "table")))
@pytest.mark.parametrize("widths,kv_heads,capacities", (((4096, 512, 512), 2, (256, 32, 32)),
                                                      ((2048, 256, 256), 2, (128, 16, 16)),
                                                      ((2048, 384, 384), 3, (128, 32, 32))))
def test_n_tiles_describe_real_weight_rows_and_projection_slices(kind, descriptor, widths, kv_heads, capacities):
    module = tiled_module(kind, descriptor, widths, kv_heads)
    package = describe_tir_package(module)
    call, = (value for value in package["render_calls"] if value["implementation"] == implementation_id(kind, descriptor))
    local_n = sum(capacities)
    assert call["num_n_tiles"] == (local_n + call["block_n"] - 1) // call["block_n"]
    assert tuple(value["local_n_capacity"] for value in call["outputs"]) == capacities
    assert [value["projection_start"] for value in call["outputs"]] == [0, capacities[0], sum(capacities[:2])]
    request, = call["host_tensor_descriptor_requests"]
    shape = request["shape"] if descriptor == "single" else request["entries"][0]["shape"]
    assert shape[1 if descriptor == "single" else 0] == local_n // 8
    assert any("qkv_n_tile" in offset for offset in call["descriptor_offsets"])


@pytest.mark.parametrize("violation", ("rows", "reduction", "output_dtype", "partial_axes", "missing_partial"))
def test_n_tiling_does_not_weaken_input_or_partial_contract(violation):
    from triton.flagmega import ir as fm
    from .test_packed_qkv_mma_candidate import _candidate_ids, _context

    module = tiled_module("mma", "table", (4096, 512, 512), 2)
    context = _context(module)
    name = context.dispatch.arguments[0] if violation in {"rows", "reduction"} else context.dispatch.outputs[1]
    original = context.function.parameter_map[name].type
    if violation in {"rows", "reduction"}:
        changed = replace(original, tensor=fm.tensor_type("bfloat16", (2, 2048) if violation == "rows" else (1, 1536)))
    elif violation == "output_dtype":
        changed = replace(original, tensor=replace(original.tensor, dtype=fm.VectorType(fm.DType.FLOAT32, (8,))))
    else:
        changed = replace(original, partial=None if violation == "missing_partial" else fm.SBP.partial((1,), fm.ReduceOp.SUM))
    ids = _candidate_ids(_context(module, parameter_types={name: changed}))
    assert not {implementation_id(kind, descriptor) for kind in ("gemv", "mma")
                for descriptor in ("single", "table")}.intersection(ids)


@pytest.mark.parametrize("kind", ("gemv", "mma"))
@pytest.mark.parametrize("descriptor", ("single", "table"))
@pytest.mark.parametrize("widths,kv_heads", (((4096, 512, 512), 2), ((2048, 256, 256), 2),
                                           ((2048, 384, 384), 3)))
def test_n_tiles_exact_coordinates_and_partial_boundary(kind, descriptor, widths, kv_heads, tmp_path):
    module = tiled_module(kind, descriptor, widths, kv_heads)
    check_exact_coordinates_and_partials(module, implementation_id(kind, descriptor), widths, 2048, tmp_path)


def check_exact_coordinates_and_partials(module, identity, widths, input_extent, tmp_path):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("requires SM90")
    from triton.flagmega.artifacts import write_artifact
    from triton.flagmega.importer import MemoryCheckpoint, TensorInfo
    from triton.flagmega.ir import DType
    from triton.flagmega.ir.local_shard import local_shard_descriptor
    from triton.flagmega.runtime import load

    names = ("q_weight", "k_weight", "v_weight")
    weights = {name: ((torch.arange(input_extent)[:, None] // 16 * 3 + torch.arange(input_extent)[:, None] % 16
                      + torch.arange(n)[None, :] * 5 + index * 7) % 61 - 30).to(torch.bfloat16)
               for index, (name, n) in enumerate(zip(names, widths))}
    checkpoint = MemoryCheckpoint({}, {
        name: TensorInfo(name, DType.BFLOAT16, tuple(value.shape), "memory") for name, value in weights.items()
    }, weights)
    artifact = write_artifact(module, tmp_path / "artifact", target="nvidia-sm90",
                              checkpoint=checkpoint, emit_executable=True)
    if identity.endswith("_direct_pipeline"):
        source = (tmp_path / "artifact" / "generated_kernels.py").read_text()
        assert source.count("for qkv_k_tile in tl.range(") == 2
        assert "for qkv_k_tile in tl.static_range(" not in source
    runtime = load(artifact, device="cuda:0")
    value = torch.zeros((1, input_extent), device="cuda", dtype=torch.bfloat16)
    outputs = tuple(torch.empty((1, n), device="cuda", dtype=torch.bfloat16) for n in widths)
    runtime.prepare(value, *outputs)
    for k in sorted({k for k in (0, 15, 16, 63, 64, 127, 128, 191, 192, 255, 256, input_extent - 1)
                     if k < input_extent}):
        value.zero_()
        value[0, k] = 1
        for output in outputs:
            output.fill_(float("nan"))
        runtime.run_into(value, *outputs)
        torch.cuda.synchronize()
        for name, output in zip(names, outputs):
            torch.testing.assert_close(output.cpu(), weights[name][k:k + 1], rtol=0, atol=0)

    kernel, = (definition for definition in module.kernel_definitions
               if definition.dispatch.microkernel.implementation == identity)
    source_type = kernel.parameter_map[kernel.dispatch.arguments[0]].type
    reductions = set()
    for owner in product(*(range(extent) for extent in source_type.placement.hierarchy)):
        axis = local_shard_descriptor(source_type, owner).axes[-1]
        reductions.add(tuple(axis.map_local_to_global(k).fixed_value for k in range(axis.active_extent.fixed_value)))
    dyadic = (((torch.arange(input_extent) * 7) % 17 - 8).double() / 128).reshape(1, -1)
    value.copy_(dyadic.to(torch.bfloat16))
    runtime.run_into(value, *outputs)
    torch.cuda.synchronize()
    for name, output in zip(names, outputs):
        partials = [(dyadic[:, list(indices)] @ weights[name][list(indices)].double()).to(torch.bfloat16).double()
                    for indices in sorted(reductions)]
        expected = torch.stack(partials).sum(0).to(torch.bfloat16)
        torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0)
    assert runtime.resource_report["spill_bytes"] == 0
