import time
import torch
import triton
from triton.backends.spine_triton.driver import CPUDriver

triton.runtime.driver.set_active(CPUDriver())

if __name__ == "__main__":
    test_warm_up = 5
    test_iterations = 100
    test_shape_list = [(1024, 1024), (512, 512), (256, 256), (128, 128), (32, 32)]
    test_dtype_list = [torch.float32, torch.float16]

    test_op_list = {
        "conv1d": lambda x: torch.nn.functional.conv1d(x.unsqueeze(0), torch.randn(1, x.shape[1], 3)).squeeze(0),
        "conv2d": lambda x: torch.nn.functional.conv2d(x.unsqueeze(0).unsqueeze(0), torch.randn(1, 1, 3, 3)).squeeze(0),
    }

    print("Convolution Operations Performance Test")
    print("=======================================")

    for op_name, op_func in test_op_list.items():
        print(f"\n--- {op_name} ---")

        for test_dtype in test_dtype_list:
            for test_shape in test_shape_list:
                try:
                    if op_name == "conv1d":
                        x = torch.randn(test_shape[0], test_shape[1], device="cpu", dtype=test_dtype)
                    elif op_name == "conv2d":
                        x = torch.randn(test_shape[0], test_shape[1], device="cpu", dtype=test_dtype)

                    for _ in range(test_warm_up):
                        ref = op_func(x)

                    start = time.time()
                    for _ in range(test_iterations):
                        ref = op_func(x)
                    end = time.time()

                    print(
                        f"dtype {test_dtype} shape {test_shape}, cost {1000 * (end - start) / test_iterations:.3f} ms")
                except Exception as e:
                    print(f"dtype {test_dtype} shape {test_shape}, Failed: {str(e)}")
