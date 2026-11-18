"""Matrix-vector product for one row of BF16 or FP16 activations."""
import torch
import triton
import triton.language as tl


@triton.jit
def gemv(A, B, C, N: tl.constexpr, K: tl.constexpr,
         ROWS: tl.constexpr, WIDTH: tl.constexpr):
    row = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    k = tl.arange(0, WIDTH)
    a = tl.load(A + k, k < K, 0).to(tl.float32)
    b = tl.load(B + row[:, None] * K + k[None, :],
                (row[:, None] < N) & (k[None, :] < K), 0).to(tl.float32)
    tl.store(C + row, tl.sum(b * a[None, :], 1), row < N)


class SingleRowGemmPlan:
    """A[1,K] @ B[N,K].T with FP32 reduction and plan-owned output."""
    def __init__(self, a, b):
        if (a.ndim != 2 or b.ndim != 2 or a.shape[0] != 1 or a.shape[1] != b.shape[1]
                or a.device.type != "cuda" or a.device != b.device
                or a.dtype not in (torch.float16, torch.bfloat16) or b.dtype != a.dtype
                or not a.is_contiguous() or not b.is_contiguous()):
            raise ValueError("expected contiguous matching GPU FP16/BF16 A[1,K] and B[N,K]")
        self.n, self.k = b.shape
        if self.n < 1 or not 1 <= self.k <= 8192:
            raise ValueError("expected positive N and K in [1,8192]")
        if b.numel() * b.element_size() > 0x7ffffffe:
            raise ValueError("weight span exceeds the validated buffer extent")
        self.a, self.b = a, b
        self.output = torch.empty((1, self.n), dtype=a.dtype, device=a.device)
        self.compiled = None

    def __call__(self):
        self.compiled = gemv[(triton.cdiv(self.n, 2),)](
            self.a, self.b, self.output, self.n, self.k, 2, triton.next_power_of_2(self.k),
            num_warps=4, num_stages=1)
        return self.output
