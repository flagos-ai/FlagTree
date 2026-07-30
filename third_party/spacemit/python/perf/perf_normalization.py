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
        "layernorm": lambda x: torch.nn.functional.layer_norm(x, x.shape[-1:]),
        "groupnorm": lambda x: torch.nn.functional.group_norm(x, num_groups=1),
        "batch_norm": lambda x: torch.nn.functional.batch_norm(x, torch.zeros(x.shape[1]), torch.ones(x.shape[1])),
    }

    print("Normalization Layers Performance Test")
    print("======================================")

    for op_name, op_func in test_op_list.items():
        print(f"\n--- {op_name} ---")

        for test_dtype in test_dtype_list:
            for test_shape in test_shape_list:
                try:
                    if op_name == "batch_norm":
                        x = torch.randn(test_shape[0], test_shape[1], device="cpu", dtype=test_dtype)
                    else:
                        x = torch.randn(test_shape, dtype=test_dtype, device="cpu")

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
