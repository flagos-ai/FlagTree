import torch
import triton
import triton.language as tl
import triton.language.extra.deeplink as dl
from triton.backends.spine_triton.driver import CPUDriver

triton.runtime.driver.set_active(CPUDriver())

# eg: pytest -v test_compile_hint.py::test_compile_hint
#############################


@triton.jit
def triton_compile_hint(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr, XBLOCK_SUB: tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    for xoffset_sub in range(0, XBLOCK, XBLOCK_SUB):
        xindex = xoffset + xoffset_sub + tl.arange(0, XBLOCK_SUB)[:]
        xmask = xindex < xnumel
        x0 = xindex
        tmp0 = tl.load(in_ptr0 + (x0), xmask)
        dl.compile_hint(tmp0, "hint_a")
        tmp2 = tmp0
        dl.compile_hint(tmp2, "hint_b", 42)
        dl.compile_hint(tmp2, "hint_c", True)
        tl.store(out_ptr0 + (xindex), tmp2, xmask)


def test_compile_hint(param_list):
    dtype_str, shape, ncore, xblock, xblock_sub = param_list
    dtype = getattr(torch, dtype_str)
    x0 = torch.rand(shape, dtype=dtype).cpu()
    y_ref = x0
    y_cal = torch.rand(shape, dtype=dtype).cpu()
    triton_compile_hint[(ncore, )](x0, y_cal, x0.numel(), xblock, xblock_sub)
    assert torch.allclose(y_cal, y_ref)
    print(f'The maximum difference between torch and triton is '
          f'{torch.max(torch.abs(y_cal - y_ref))}')
    assert y_cal.dtype == y_ref.dtype
    print("dtype is same.")


if __name__ == "__main__":
    param_list = ['float32', (2, 4096, 8), 2, 32768, 1024]
    test_compile_hint(param_list)
