"""A wide activation never authorizes removing the projection casts."""

import pytest

from triton.flagmega.artifacts import write_artifact
from triton.flagmega.importer import MemoryCheckpoint, TensorInfo
from triton.flagmega.ir import DType
from triton.flagmega.runtime import load
from .conftest import _packed_glu_descriptor_pipeline_module


@pytest.mark.parametrize("implementation", [
    "tir.dense_matmul_glu.packed_k_major_gemv_tn16",
    "tir.dense_matmul_glu.packed_tensor_descriptor_smem_pipeline_full_lhs_gemv",
    "tir.dense_matmul_glu.packed_tensor_descriptor_table_paired_smem_pipeline_inline_gemv",
])
def test_wide_activation_preserves_both_bf16_projection_boundaries(tmp_path, implementation):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("SM90 CUDA is required")
    module = _packed_glu_descriptor_pipeline_module(implementation, round_activation=False)
    gate = torch.zeros((2048, 2048), dtype=torch.bfloat16)
    up = torch.zeros_like(gate)
    gate[:, 0] = 1
    up[:, 0] = 1
    gate[0::3, 1] = 1 / 256
    up[1::3, 1] = 1 / 256
    up[2::3, 0] = 129 / 128
    weights = {"gate": gate, "up": up}
    checkpoint = MemoryCheckpoint({}, {name: TensorInfo(name, DType.BFLOAT16, tuple(value.shape), "memory")
                                       for name, value in weights.items()}, weights)
    artifact = write_artifact(module, tmp_path / "wide_glu", target="nvidia-sm90", checkpoint=checkpoint,
                              emit_executable=True)
    runtime = load(artifact, device="cuda:0")
    value = torch.zeros((1, 2048), dtype=torch.bfloat16, device="cuda")
    value[:, :2] = 1
    g = gate[:, :2].float().sum(-1).cuda().bfloat16()
    u = up[:, :2].float().sum(-1).cuda().bfloat16()
    expected = (torch.nn.functional.silu(g.float()) * u.float()).bfloat16()
    assert torch.count_nonzero(expected != torch.nn.functional.silu(g) * u)
    output = torch.empty_like(value)
    runtime.prepare(value, output=output)
    runtime.run_into(output, value)
    torch.cuda.synchronize()
    torch.testing.assert_close(output.flatten(), expected, rtol=0, atol=0)
