# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT

import pytest

from triton.flagmega.artifacts import write_artifact
from triton.flagmega.importer import MemoryCheckpoint, TensorInfo
from triton.flagmega.ir import DType
from triton.flagmega.runtime import load


@pytest.mark.parametrize("n,k", [(2048, 2048), (80, 320), (2056, 528)])
@pytest.mark.parametrize("tile_n,block_k", [(16, 128), (32, 128), (64, 256)])
def test_packed_compute_tile_matches_reference_with_local_and_reduction_tails(
    tmp_path, compile_packed_projection, n, k, tile_n, block_k,
):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 9:
        pytest.skip("requires Hopper CUDA")
    implementation = f"tir.dense_matmul.packed_k_major_gemv_tn{tile_n}_bk{block_k}"
    module = compile_packed_projection(implementation, k=k, n=n)
    generator = torch.Generator().manual_seed(n + k)
    weight = torch.randn((n, k), generator=generator, dtype=torch.bfloat16) / k ** .5
    checkpoint = MemoryCheckpoint({}, {
        "weight": TensorInfo("weight", DType.BFLOAT16, (n, k), "memory"),
    }, {"weight": weight})
    artifact = write_artifact(module, tmp_path / "packed-tile", target="nvidia-sm90",
                              checkpoint=checkpoint, emit_executable=True)
    runtime = load(artifact, device="cuda:0")
    source = torch.randn((1, k), generator=generator, dtype=torch.bfloat16).cuda()
    output = torch.full((1, n), float("nan"), dtype=torch.bfloat16, device="cuda")
    runtime.prepare(source, output=output)
    runtime.run_into(output, source)
    expected = (source.float() @ weight.cuda().float().T).to(torch.bfloat16)
    torch.testing.assert_close(output, expected, rtol=.03, atol=.03)
    assert runtime.resource_report["spill_bytes"] == 0
