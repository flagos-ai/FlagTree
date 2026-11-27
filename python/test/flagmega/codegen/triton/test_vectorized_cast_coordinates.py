from itertools import product

import pytest
import torch

from triton.flagmega.codegen.triton.kernel_call_renderers import _repack_vector_coordinates
from triton.flagmega.ir.ops.tensors.pack import pack_physical, axis_lane_products


@pytest.mark.parametrize("axes,input_lanes,output_lanes", [
    ((0,), (4,), (8,)),
    ((0, 0), (2, 4), (4, 8)),
    ((1, 0, 1), (2, 4, 2), (4, 2, 4)),
])
def test_repack_coordinates_agree_with_physical_pack(axes, input_lanes, output_lanes):
    rank = max(axes) + 1
    shape = (64,) * rank
    scalar = torch.arange(64 ** rank).reshape(shape)
    input_value = pack_physical(scalar, rank, input_lanes, axes)
    output_value = pack_physical(scalar, rank, output_lanes, axes)
    extents = axis_lane_products(output_lanes, axes)
    output_shape = tuple(shape[axis] // extents.get(axis, 1) for axis in range(rank))
    # Representative outer coordinates and every lane, including interleaved
    # repeated axes. Expressions are compiler-produced integer arithmetic.
    for base in product(*((0, size-1) for size in output_shape)):
        for lane in product(*(range(size) for size in output_lanes)):
            coords, components = _repack_vector_coordinates(
                tuple(map(str, base)), axes, input_lanes, output_lanes, tuple(map(str, lane)))
            input_index = tuple(eval(value, {"__builtins__": {}}, {}) for value in (*coords, *components))
            assert input_value[input_index].item() == output_value[(*base, *lane)].item()


@pytest.mark.parametrize("widen", [False, True])
def test_repeated_axis_cast_runs_through_selected_tir(tmp_path, widen):
    from triton.flagmega import ir as fm
    from triton.flagmega.artifacts import write_artifact
    from triton.flagmega.compiler import Compiler
    from triton.flagmega.runtime import load

    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("SM90 CUDA is required")
    source_dtype, target_dtype = ("bfloat16", "float32") if widen else ("float32", "bfloat16")
    source_lanes, target_lanes = ((4, 8), (2, 4)) if widen else ((2, 4), (4, 8))
    source_extent, target_extent = (2, 8) if widen else (8, 2)
    class CastGraph(fm.Module):
        def __init__(self):
            super().__init__(dialect="high_level", stage="imported", entry="main")

        def forward(self):
            value = self.input("value", fm.tensor_type(fm.vector_type(source_dtype, source_lanes), (source_extent,)))
            result = fm.F.ntt.vectorized_cast(value, fm.vector_type(target_dtype, target_lanes), (0, 0))
            self.function("main", (value,), (result,))

    module = Compiler().compile(CastGraph().build()).module
    artifact = write_artifact(module, tmp_path / "cast", target="nvidia-sm90", emit_executable=True)
    runtime = load(artifact, device="cuda:0")
    source = torch.randn((source_extent, *source_lanes), dtype=getattr(torch, source_dtype), device="cuda")
    runtime.prepare(source)
    output = runtime.run(source)
    torch.cuda.synchronize()
    torch.testing.assert_close(output, source.to(getattr(torch, target_dtype)).reshape(target_extent, *target_lanes),
                               rtol=0, atol=0)
