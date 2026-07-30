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

    test_op_groups = {
        "activate": {
            "silu": torch.nn.functional.silu,
            "relu": torch.nn.functional.relu,
            "gelu_none": lambda x: torch.nn.functional.gelu(x, approximate="none"),
            "gelu_tanh": lambda x: torch.nn.functional.gelu(x, approximate="tanh"),
        },
        "matrix": {
            "mm": torch.mm,
            "addmm": lambda x: torch.addmm(x, x, x),
            "bmm": lambda x: torch.bmm(x.unsqueeze(0), x.unsqueeze(0)).squeeze(0),
            "mv": lambda x: torch.mv(x, x[0]),
            "outer": torch.outer,
        },
        "normalization": {
            "layernorm": lambda x: torch.nn.functional.layer_norm(x, x.shape[-1:]),
            "groupnorm": lambda x: torch.nn.functional.group_norm(x, num_groups=1),
            "batch_norm": lambda x: torch.nn.functional.batch_norm(x, torch.zeros(x.shape[1]), torch.ones(x.shape[1])),
        },
        "convolution": {
            "conv1d": lambda x: torch.nn.functional.conv1d(x.unsqueeze(0), torch.randn(1, x.shape[1], 3)).squeeze(0),
            "conv2d":
            lambda x: torch.nn.functional.conv2d(x.unsqueeze(0).unsqueeze(0), torch.randn(1, 1, 3, 3)).squeeze(0),
        },
        "basic_arithmetic": {
            "div": lambda x: torch.div(x, x + 1.0),
            "sub": lambda x: torch.sub(x, x),
        },
    }

    for group_name, op_group in test_op_groups.items():
        print(f"\n{'='*60}")
        print(f"############ {group_name} ############")
        print(f"{'='*60}")

        for test_dtype in test_dtype_list:
            for test_shape in test_shape_list:
                print(f"\n--- dtype: {test_dtype}, shape: {test_shape} ---")

                for op_name, op_func in op_group.items():
                    try:
                        if op_name in ["conv1d", "conv2d", "batch_norm"]:
                            x = torch.randn(
                                test_shape[0],
                                test_shape[1],
                                device="cpu",
                                dtype=test_dtype,
                            )
                        else:
                            x = torch.randn(test_shape, dtype=test_dtype, device="cpu")

                        for _ in range(test_warm_up):
                            ref = op_func(x)
                        start = time.time()
                        for _ in range(test_iterations):
                            ref = op_func(x)
                        end = time.time()

                        cost_ms = 1000 * (end - start) / test_iterations
                        print(f"  {op_name:15}: {cost_ms:8.3f} ms")

                    except Exception as e:
                        print(f"  {op_name:15}: Failed - {str(e)}")
