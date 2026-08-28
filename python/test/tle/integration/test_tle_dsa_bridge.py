import pytest
import torch

import triton
import triton.language as tl
import triton.experimental.tle.language as tle


def _is_txda():
    try:
        import torch_txda  # noqa: F401
    except ImportError:
        return False
    target = triton.runtime.driver.active.get_current_target()
    return getattr(target, "backend", None) == "txda"


pytestmark = pytest.mark.skipif(
    not _is_txda(), reason="TLE DSA tests require TsingMicro (txda) backend"
)


@triton.jit
def to_buffer_to_tensor_kernel(x_ptr, y_ptr, out_ptr, M, N, P, Q,
                               BM: tl.constexpr, BN: tl.constexpr, BP: tl.constexpr, BQ: tl.constexpr):
    """4D round-trip: to_buffer -> to_tensor -> compute -> to_buffer -> store."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_p = tl.program_id(2)
    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_p = pid_p * BP + tl.arange(0, BP)
    offs_q = tl.arange(0, BQ)
    idx = (offs_m[:, None, None, None] * N * P * Q
           + offs_n[None, :, None, None] * P * Q
           + offs_p[None, None, :, None] * Q
           + offs_q[None, None, None, :])
    mask = (offs_m[:, None, None, None] < M) & (offs_n[None, :, None, None] < N) & \
           (offs_p[None, None, :, None] < P) & (offs_q[None, None, None, :] < Q)

    x = tl.load(x_ptr + idx, mask=mask)
    y = tl.load(y_ptr + idx, mask=mask)

    # tl.tensor -> fresh SPM buffer.
    buf_x = tle.dsa.to_buffer(x, tle.dsa.tsingmicro.SPM)
    buf_y = tle.dsa.to_buffer(y, tle.dsa.tsingmicro.SPM)

    # SPM buffer -> zero-copy tl.tensor view, then compute.
    tx = tle.dsa.to_tensor(buf_x)
    ty = tle.dsa.to_tensor(buf_y)
    z = tx * ty

    # Result into a fresh SPM buffer, then read back out.
    buf_z = tle.dsa.to_buffer(z, tle.dsa.tsingmicro.SPM)
    zz = tle.dsa.to_tensor(buf_z)
    tl.store(out_ptr + idx, zz, mask=mask)


class TestTLEDsaBridge:
    """to_tensor / to_buffer bridge between tl.tensor and SPM buffers."""

    @pytest.mark.parametrize(
        "shape,block",
        [
            ((17, 13, 9, 8), (16, 8, 8, 8)),  # tails on m/n/p; q fully covered (3-axis grid)
        ],
    )
    def test_bridge(self, shape, block):
        torch.manual_seed(42)
        m, n, p, q = shape
        bm, bn, bp, bq = block
        a = torch.randn(*shape, device="txda", dtype=torch.float32)
        b = torch.randn(*shape, device="txda", dtype=torch.float32)
        out = torch.empty_like(a)

        grid = (triton.cdiv(m, bm), triton.cdiv(n, bn), triton.cdiv(p, bp))
        to_buffer_to_tensor_kernel[grid](a, b, out, m, n, p, q,
                                         BM=bm, BN=bn, BP=bp, BQ=bq, num_ctas=1)

        torch.testing.assert_close(out, a * b, atol=1e-4, rtol=1e-4)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
