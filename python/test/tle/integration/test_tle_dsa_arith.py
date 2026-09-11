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


pytestmark = pytest.mark.skipif(not _is_txda(), reason="TLE DSA tests require TsingMicro (txda) backend")


@triton.jit
def dsa_arith_kernel(x_ptr, y_ptr, out_ptr, M, N, P, Q, BM: tl.constexpr, BN: tl.constexpr, BP: tl.constexpr,
                     BQ: tl.constexpr, OP: tl.constexpr):
    """Four-dimensional three-operand buffer arithmetic (tiled)."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_p = tl.program_id(2)
    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_p = pid_p * BP + tl.arange(0, BP)
    offs_q = tl.arange(0, BQ)
    idx = (offs_m[:, None, None, None] * N * P * Q + offs_n[None, :, None, None] * P * Q +
           offs_p[None, None, :, None] * Q + offs_q[None, None, None, :])
    mask = (offs_m[:, None, None, None] < M) & (offs_n[None, :, None, None] < N) & \
           (offs_p[None, None, :, None] < P) & (offs_q[None, None, None, :] < Q)

    lhs = tl.load(x_ptr + idx, mask=mask)
    rhs = tl.load(y_ptr + idx, mask=mask)

    lhs_buf = tle.dsa.to_buffer(lhs, tle.dsa.tsingmicro.SPM)
    rhs_buf = tle.dsa.to_buffer(rhs, tle.dsa.tsingmicro.SPM)
    out_buf = tle.dsa.alloc((BM, BN, BP, BQ), tl.float32)

    if OP == "add":
        tle.dsa.add(lhs_buf, rhs_buf, out_buf)
    elif OP == "sub":
        tle.dsa.sub(lhs_buf, rhs_buf, out_buf)
    elif OP == "mul":
        tle.dsa.mul(lhs_buf, rhs_buf, out_buf)
    elif OP == "max":
        tle.dsa.max(lhs_buf, rhs_buf, out_buf)
    elif OP == "min":
        tle.dsa.min(lhs_buf, rhs_buf, out_buf)
    elif OP == "div":
        tle.dsa.div(lhs_buf, rhs_buf, out_buf)

    result = tle.dsa.to_tensor(out_buf)
    tl.store(out_ptr + idx, result, mask=mask)


class TestTLEDsaArith:
    """Three-operand DSA buffer arithmetic (add/sub/mul/max/min/div)."""

    @pytest.mark.parametrize(
        "op,ref",
        [
            ("add", lambda a, b: a + b),
            ("sub", lambda a, b: a - b),
            ("mul", lambda a, b: a * b),
            ("max", lambda a, b: torch.maximum(a, b)),
            ("min", lambda a, b: torch.minimum(a, b)),
            ("div", lambda a, b: a / b),
        ],
    )
    @pytest.mark.parametrize(
        "shape,block",
        [((17, 13, 9, 8), (16, 8, 8, 8)),  # tails on m/n/p; q fully covered (3-axis grid)
         ],
    )
    def test_arith(self, op, ref, shape, block):
        torch.manual_seed(42)
        m, n, p, q = shape
        bm, bn, bp, bq = block
        a = torch.randn(*shape, device="txda", dtype=torch.float32)
        b = torch.randn(*shape, device="txda", dtype=torch.float32)
        out = torch.empty_like(a)

        grid = (triton.cdiv(m, bm), triton.cdiv(n, bn), triton.cdiv(p, bp))
        dsa_arith_kernel[grid](a, b, out, m, n, p, q, BM=bm, BN=bn, BP=bp, BQ=bq, num_ctas=1, OP=op)

        expected = ref(a, b)
        torch.testing.assert_close(out, expected, atol=1e-4, rtol=1e-4)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
