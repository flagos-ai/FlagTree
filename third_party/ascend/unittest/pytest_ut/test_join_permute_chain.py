"""Join/permute chains preserve input bits through compact/padded UB views."""
import pytest
import torch
import torch_npu  # noqa: F401
import triton
import triton.language as tl


@triton.jit
def _merge(A, B, Out, VALID, WIDTH: tl.constexpr, INNER: tl.constexpr):
    index = tl.arange(0, WIDTH * INNER)
    a = tl.load(A + index, index < VALID, 0)
    b = tl.load(B + index, index < VALID, 0)
    joined = tl.join(tl.reshape(a, (WIDTH, INNER)), tl.reshape(b, (WIDTH, INNER)))
    merged = tl.reshape(tl.permute(joined, 0, 2, 1), (2 * WIDTH * INNER, ))
    out = tl.arange(0, 2 * WIDTH * INNER)
    tl.store(Out + out, merged)


@pytest.mark.parametrize('mode', ['auto', 'native', 'packed'])
@pytest.mark.parametrize('dtype', [torch.float32, torch.float16, torch.bfloat16, torch.int32])
@pytest.mark.parametrize('shape', [(512, 2), (32, 4)])
def test_join_permute_chain(shape, dtype, mode):
    width, inner = shape
    count, guard = width * inner, 128
    bits_type = torch.int16 if dtype in (torch.float16, torch.bfloat16) else torch.int32
    torch.manual_seed(101102)
    a_bits = torch.randint(-16000, 16000, (count, ), dtype=bits_type)
    b_bits = torch.randint(-16000, 16000, (count, ), dtype=bits_type)
    # Include signed zero, infinities and NaN payloads without arithmetic.
    if bits_type == torch.int32:
        a_bits[:4] = torch.tensor([0, -2147483648, 2139095040, 2143363909], dtype=bits_type)
    a, b = a_bits.view(dtype), b_bits.view(dtype)
    da, db = a.npu(), b.npu()
    guarded = torch.full((2 * count + 2 * guard, ), 123, dtype=bits_type, device='npu')
    out = guarded[guard:-guard].view(dtype)
    for valid in (count, count - 3):
        aa, bb = a_bits.clone(), b_bits.clone()
        aa[valid:] = 0
        bb[valid:] = 0
        expected = torch.stack((aa.reshape(width, inner), bb.reshape(width, inner)), dim=1).flatten()

        def launch():
            _merge[(1, )](da, db, out, valid, width, inner, num_warps=4, num_stages=1, multibuffer=False,
                          enable_auto_blockify=True, structured_join_lowering=mode)

        launch()
        torch.npu.synchronize()
        assert torch.equal(out.cpu().view(bits_type), expected)
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            launch()
        out.view(bits_type).fill_(321)
        graph.replay()
        torch.npu.synchronize()
        assert torch.equal(out.cpu().view(bits_type), expected)
        assert (guarded[:guard].cpu() == 123).all()
        assert (guarded[-guard:].cpu() == 123).all()
        assert torch.equal(da.cpu().view(bits_type), a_bits)
        assert torch.equal(db.cpu().view(bits_type), b_bits)
