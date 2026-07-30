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
        "silu": torch.nn.functional.silu,
        "relu": torch.nn.functional.relu,
        "gelu_none": lambda x: torch.nn.functional.gelu(x, approximate="none"),
        "gelu_tanh": lambda x: torch.nn.functional.gelu(x, approximate="tanh"),
    }

    for op_name, op_func in test_op_list.items():
        print(f"############ {op_name} ############")

        for test_dtype in test_dtype_list:
            for test_shape in test_shape_list:
                x = torch.randn(test_shape, dtype=test_dtype, device="cpu")

                for _ in range(test_warm_up):
                    ref = op_func(x)

                start = time.time()
                for _ in range(test_iterations):
                    ref = op_func(x)
                end = time.time()

                print(
                    f"torch: dtype {test_dtype} shape {test_shape}, cost {1000 * (end - start) / test_iterations:.3f} ms"
                )
