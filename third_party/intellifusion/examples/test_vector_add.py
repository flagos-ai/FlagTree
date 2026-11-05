import pytest
import torch
import triton
import triton.language as tl

DEVICE = triton.runtime.driver.active.get_active_torch_device()


@triton.jit
def add_kernel(
    x_ptr,
    y_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)


def add(x: torch.Tensor, y: torch.Tensor, block_size: int, num_warps: int):
    output = torch.empty_like(x)
    n_elements = output.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=block_size, num_warps=num_warps)
    return output


@pytest.mark.parametrize("size", [127, 95532])
@pytest.mark.parametrize("block_size", [128, 1024])
@pytest.mark.parametrize("num_warps", [1, 2, 4])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_vector_add(size, block_size, num_warps, dtype):
    torch.manual_seed(0)
    x = torch.rand(size, device=DEVICE, dtype=dtype)
    y = torch.rand(size, device=DEVICE, dtype=dtype)

    output_torch = x + y
    output_triton = add(x, y, block_size, num_warps)
    max_diff = torch.max(torch.abs(output_torch - output_triton)).item()
    assert max_diff == 0, (f"size={size}, block_size={block_size}, num_warps={num_warps}, max_diff={max_diff}")


@triton.jit
def attributes_kernel(src, dst, n, BLOCK: tl.constexpr, MODE: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    mask = tl.max_constancy(offsets < n, 4)
    if MODE == "volatile":
        value = tl.load(src + offsets, mask, other=-7.0, volatile=True)
        tl.store(dst + offsets, value, mask)
    elif MODE == "cache":
        value = tl.load(src + offsets, mask, other=-7.0, cache_modifier=".cg")
        tl.store(dst + offsets, value, mask, cache_modifier=".wt")
    else:
        value = tl.load(src + offsets, mask, other=-7.0, eviction_policy=MODE)
        tl.store(dst + offsets, value, mask, eviction_policy=MODE)


@pytest.mark.parametrize("shape", [96])
@pytest.mark.parametrize("num_warps", [4])
@pytest.mark.parametrize("block_size", [128])
@pytest.mark.parametrize("mode", ["volatile", "cache", "evict_first", "evict_last"])
def test_load_store_attributes(shape, num_warps, block_size, mode):
    # The last valid load is at the allocation boundary; stores have canaries.
    expected = torch.arange(shape, dtype=torch.float32)
    src = expected.to(DEVICE)
    dst = torch.full((block_size + 2, ), -123.0, dtype=torch.float32).to(DEVICE)
    attributes_kernel[(1, )](src, dst[1:], shape, BLOCK=block_size, MODE=mode, num_warps=num_warps)
    actual = dst.cpu()
    torch.testing.assert_close(actual[1:shape + 1], expected, rtol=0, atol=0)
    assert actual[0].item() == -123.0
    torch.testing.assert_close(actual[shape + 1:], torch.full_like(actual[shape + 1:], -123.0), rtol=0, atol=0)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
