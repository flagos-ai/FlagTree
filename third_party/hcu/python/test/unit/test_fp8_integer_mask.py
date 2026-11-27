"""Integer FP8 conversion and bit-exact masked-load fallback regressions."""
import pytest
import torch
import triton
import triton.language as tl


@triton.jit
def _masked_fp8(x, y, n: tl.constexpr, other: tl.constexpr):
    offsets = tl.arange(0, 256)
    values = tl.load(x + offsets, offsets < n, other=other)
    tl.store(y + offsets, values.to(tl.float32))


@triton.jit
def _integer_fp8(x, y, dst: tl.constexpr):
    offsets = tl.arange(0, 128)
    values = tl.load(x + offsets)
    tl.store(y + offsets, values.to(dst).to(tl.float32))


@pytest.mark.parametrize("dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
@pytest.mark.parametrize("n", [0, 7, 129])
@pytest.mark.parametrize("other", [0, 1, -3, 129])
def test_fp8_integer_mask_other(dtype, n, other):
    cpu = (torch.arange(n).float() % 17 - 8).to(dtype)
    x = cpu.cuda()
    storage = torch.full((258,), 12345.25, device="cuda")
    y = storage[1:257]
    _masked_fp8[(1,)](x, y, n, other, num_warps=4)
    expected = torch.full((256,), torch.tensor(float(other)).to(dtype).float().item())
    expected[:n] = cpu.float()
    torch.testing.assert_close(y.cpu(), expected, atol=0, rtol=0)
    assert storage[0].item() == 12345.25 and storage[-1].item() == 12345.25


@pytest.mark.parametrize("dtype", [torch.float8_e4m3fn, torch.float8_e5m2, torch.float32])
@pytest.mark.parametrize("n", [0, 7])
@pytest.mark.parametrize("other", [0.0, -0.0])
def test_masked_load_signed_zero_bits(dtype, n, other, monkeypatch):
    # The buffer path must retain negative-zero `other` instead of relying on
    # the instruction's all-zero-bit masked lanes. Positive zero stays eligible.
    monkeypatch.setenv("AMDGCN_USE_BUFFER_OPS", "1")
    cpu = torch.tensor([0.0, -0.0, 1.0, -1.0, 2.0, -2.0, -0.0][:n]).to(dtype)
    y = torch.empty(256, device="cuda")
    _masked_fp8[(1,)](cpu.cuda(), y, n, other, num_warps=4)
    expected = torch.full((256,), other)
    expected[:n] = cpu.float()
    assert torch.equal(y.cpu().view(torch.int32), expected.view(torch.int32))


@pytest.mark.parametrize("dtype,dst", [(torch.float8_e4m3fn, tl.float8e4nv),
                                       (torch.float8_e5m2, tl.float8e5)])
@pytest.mark.parametrize("integer", [torch.bool, torch.int8, torch.uint8, torch.int16,
                                    torch.int32, torch.uint32, torch.int64, torch.uint64])
def test_integer_to_fp8(dtype, dst, integer):
    cpu = (torch.arange(128) % 2 if integer == torch.bool else torch.arange(128)).to(integer)
    y = torch.empty(128, device="cuda")
    _integer_fp8[(1,)](cpu.cuda(), y, dst, num_warps=4)
    torch.testing.assert_close(y.cpu(), cpu.float().to(dtype).float(), atol=0, rtol=0)


@pytest.mark.parametrize("dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
def test_fp8_integer_other_graph_changed_input(dtype):
    x = torch.arange(7).to(dtype).cuda()
    y = torch.empty(256, device="cuda")
    _masked_fp8[(1,)](x, y, 7, 0, num_warps=4)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        _masked_fp8[(1,)](x, y, 7, 0, num_warps=4)
    for epoch in (1, 2, 0):
        cpu = (torch.arange(7).float() - epoch).to(dtype)
        x.copy_(cpu)
        y.fill_(float("nan"))
        graph.replay()
        expected = torch.zeros(256)
        expected[:7] = cpu.float()
        torch.testing.assert_close(y.cpu(), expected, atol=0, rtol=0)
