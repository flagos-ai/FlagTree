import math

import pytest
import triton
import triton.language as tl
import triton.experimental.tle.language as tle


@triton.jit
def _coordinates(output, mesh: tl.constexpr, rank: tl.constexpr):
    pid = tl.program_id(0)
    for axis in tl.static_range(rank):
        tl.store(output + pid * rank + axis, tle.shard_id(mesh, axis))


@pytest.mark.parametrize("shape", [(2, 1), (2, 1, 2), (1, 4), (2, 3)])
def test_shard_id_unit_axes_are_zero(shape):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("requires a GPU")
    mesh = tle.device_mesh({"block": [(f"axis{i}", size) for i, size in enumerate(shape)]})
    count = math.prod(shape)
    output = torch.full((count, len(shape)), -1, device="cuda", dtype=torch.int32)
    _coordinates[(count,)](output, mesh, len(shape))
    expected = torch.tensor([[pid // math.prod(shape[i + 1:]) % size
                              for i, size in enumerate(shape)] for pid in range(count)], dtype=torch.int32)
    torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0)
