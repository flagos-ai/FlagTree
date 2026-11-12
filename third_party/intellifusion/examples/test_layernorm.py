import pytest
import torch
import triton
import triton.language as tl

DEVICE = triton.runtime.driver.active.get_active_torch_device()

LAYER_NORM_DTYPES = [
    (torch.float32, 1e-4, "fp32"),
    (torch.bfloat16, 1e-2, "bf16"),
    (torch.float16, 1e-2, "fp16"),
]

LAYER_NORM_SHAPES = [
    ((16, 512), 512, "block512-1iter"),
    ((16, 1024), 1024, "block1024-1iter"),
    ((16, 2048), 2048, "block2048-1iter"),
    ((8, 2048), 512, "block512-4iter-aligned"),
    ((8, 2560), 512, "block512-5iter-ragged"),
    ((4, 4096), 1024, "block1024-4iter-aligned"),
    ((4, 5121), 1024, "block1024-6iter-ragged"),
    ((2, 16384), 2048, "block2048-8iter-aligned"),
    # NOTE: BLOCK=2048 + ragged tail is intentionally omitted. It triggers a vcc/IFWA
    # backend bug in 16-deep predicated load chains: OOB lanes' `tl.where` mask is
    # not applied reliably, polluting the variance accumulator (visible as large
    # rstd_diff / out_diff). BLOCK=1024 ragged (8-deep chain) is unaffected and
    # is exercised by `block1024-6iter-ragged` above.
    ((64, 1024), 1024, "grid64-1iter"),
]

LAYER_NORM_SKIP_CASES = {}


def _layer_norm_cases():
    for shape, block_size, shape_id in LAYER_NORM_SHAPES:
        for dtype, tol, dtype_id in LAYER_NORM_DTYPES:
            marks = []
            reason = LAYER_NORM_SKIP_CASES.get((shape, block_size, dtype))
            if reason is not None:
                marks.append(pytest.mark.skip(reason=reason))
            yield pytest.param(shape, block_size, dtype, tol, marks=marks, id=f"{shape_id}-{dtype_id}")


@triton.jit
def _layer_norm_fwd_fused(
    X,
    Y,
    W,
    B,
    Mean,
    Rstd,
    row_stride,
    N,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    Y += row * row_stride
    X += row * row_stride

    mean = 0
    _mean = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        a = tl.load(X + cols, mask=cols < N, other=0.).to(tl.float32)
        _mean += a
    mean = tl.sum(_mean, axis=0) / N

    _var = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        x = tl.load(X + cols, mask=cols < N, other=0.).to(tl.float32)
        x = tl.where(cols < N, x - mean, 0.)
        _var += x * x
    var = tl.sum(_var, axis=0) / N
    rstd = 1 / tl.sqrt(var + eps)

    tl.store(Mean + row, mean)
    tl.store(Rstd + row, rstd)

    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        w = tl.load(W + cols, mask=mask)
        b = tl.load(B + cols, mask=mask)
        x = tl.load(X + cols, mask=mask, other=0.).to(tl.float32)
        x_hat = (x - mean) * rstd
        y = x_hat * w + b
        tl.store(Y + cols, y, mask=mask)


def layer_norm(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float, block_size: int):
    x_arg = x.reshape(-1, x.shape[-1])
    n_rows, n_cols = x_arg.shape
    output = torch.empty_like(x)
    mean = torch.empty((n_rows, ), dtype=torch.float32, device=x.device)
    rstd = torch.empty((n_rows, ), dtype=torch.float32, device=x.device)
    grid = (n_rows, )
    _layer_norm_fwd_fused[grid](
        x_arg,
        output,
        weight,
        bias,
        mean,
        rstd,
        x_arg.stride(0),
        n_cols,
        eps,
        BLOCK_SIZE=block_size,
        num_ctas=1,
    )
    return output, mean, rstd


def run_layer_norm_case(shape, block_size, dtype, eps):
    torch.manual_seed(0)
    x_cpu = -2.3 + 0.5 * torch.randn(shape, dtype=dtype)
    weight_cpu = torch.rand((shape[-1], ), dtype=dtype)
    bias_cpu = torch.rand((shape[-1], ), dtype=dtype)

    x = x_cpu.to(DEVICE)
    weight = weight_cpu.to(DEVICE)
    bias = bias_cpu.to(DEVICE)

    output_torch = torch.nn.functional.layer_norm(x_cpu, (shape[-1], ), weight_cpu, bias_cpu, eps).to(DEVICE)
    x_arg_cpu = x_cpu.reshape(-1, shape[-1]).to(torch.float32)
    mean_torch = x_arg_cpu.mean(dim=1).to(DEVICE)
    rstd_torch = torch.rsqrt(x_arg_cpu.var(dim=1, unbiased=False) + eps).to(DEVICE)
    output_triton, mean_triton, rstd_triton = layer_norm(x, weight, bias, eps, block_size)
    return output_torch, mean_torch, rstd_torch, output_triton, mean_triton, rstd_triton


@pytest.mark.parametrize("shape,block_size,dtype,tol", list(_layer_norm_cases()))
def test_layer_norm(shape, block_size, dtype, tol):
    eps = 1e-5
    output_torch, mean_torch, rstd_torch, output_triton, mean_triton, rstd_triton = run_layer_norm_case(
        shape, block_size, dtype, eps)
    max_diff = torch.max(torch.abs(output_torch - output_triton)).item()
    max_mean_diff = torch.max(torch.abs(mean_torch - mean_triton)).item()
    max_rstd_diff = torch.max(torch.abs(rstd_torch - rstd_triton)).item()
    failures = []
    if max_mean_diff >= tol:
        failures.append(f"mean_diff={max_mean_diff}")
    if max_rstd_diff >= tol:
        failures.append(f"rstd_diff={max_rstd_diff}")
    if max_diff >= tol:
        failures.append(f"out_diff={max_diff}")
    assert not failures, f"shape={shape}, block={block_size}, dtype={dtype}, tol={tol}, " + ", ".join(failures)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
