# Copyright 2026- Xcoresigma Technology Co., Ltd
import triton
import triton.runtime as runtime
from triton._C.clear_l2 import do_bench_clear


def test_gm_to_ub():
    device = triton.runtime.driver.active.get_current_device()
    stream = triton.runtime.driver.active.get_current_stream(device)
    # size_bytes = 192 * 1024 * 1024
    # src = torch.randint(0, 255, (size_bytes // 2,), dtype=torch.float16, device='cpu')
    buffer = runtime.driver.active.get_empty_cache_for_benchmark()
    print(buffer.numel())
    do_bench_clear(buffer.data_ptr(), buffer.numel(), stream)

    print(f"Test passed: {buffer.numel()*buffer.element_size()/1024/1024:.1f}MB GM→UB executed")
    print(buffer)


if __name__ == "__main__":
    test_gm_to_ub()
