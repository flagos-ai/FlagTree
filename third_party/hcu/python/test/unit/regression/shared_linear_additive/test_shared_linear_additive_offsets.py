"""gfx938 regression: shared offsets 16 and 48 require XOR across register groups."""
from pathlib import Path
import pytest
import torch
import triton
from triton.backends.compiler import GPUTarget

@pytest.mark.parametrize('dtype', ['int8', 'fp8'])
def test_shared_linear_repeated_mmac(dtype):
    if not torch.cuda.is_available():
        pytest.skip('HCU device required')
    target = triton.runtime.driver.active.get_current_target()
    if target.backend != 'hip' or target.arch != 'gfx938':
        pytest.skip('Regression validated for gfx938')
    source = Path(__file__).parent / 'fixtures' / f'{dtype}.ttgir'
    kernel = triton.compile(str(source), target=GPUTarget('hip', 'gfx938', 64),
                            options={'num_warps': 4, 'num_stages': 1})
    for seed in (11, 29):
        gen = torch.Generator().manual_seed(seed)
        a = torch.randint(-3, 4, (64, 128), generator=gen).float()
        b = torch.randint(-3, 4, (128, 128), generator=gen).float()
        ref = a @ b
        dt = torch.int8 if dtype == 'int8' else torch.float8_e4m3fn
        ga, gb = a.to(dt).cuda(), b.to(dt).cuda()
        out = torch.full((64, 128), 123456, device='cuda',
                         dtype=torch.int32 if dtype == 'int8' else torch.float32)
        kernel[(1, 1, 1)](ga, gb, out)
        torch.cuda.synchronize()
        torch.testing.assert_close(out.cpu().float(), ref, atol=0, rtol=0)
