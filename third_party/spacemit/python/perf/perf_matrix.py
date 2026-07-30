import time
import torch

triton_init = False
try:
    import triton
    from triton.backends.spine_triton.driver import CPUDriver
    triton.runtime.driver.set_active(CPUDriver())
    triton_init = True
except Exception:
    triton_init = False

if __name__ == "__main__":
    test_warm_up = 5
    test_iterations = 100
    # M, N, K
    test_shape_list = [(1024, 1024, 2048), (512, 512, 1024), (256, 256, 512)]
    test_dtype_list = [torch.float32]

    test_op_list = {
        "mm": lambda x: torch.mm(x[0], x[1]),
        "addmm": lambda x: torch.addmm(x[2], x[0], x[1]),
        # [1, m, k] @ [k, n]
        "bmm": lambda x: torch.bmm(x[0].unsqueeze(0), x[1].unsqueeze(0)),
        # [n, k] @ [k]
        "mv": lambda x: torch.mv(x[1].permute(1, 0), x[0][0, :]),
        # [m] @ [n]
        "outer": lambda x: torch.outer(x[0][:, 0], x[1][0, :]),
    }

    test_op_ops = {
        "mm": lambda m, n, k: 2 * m * n * k,
        "addmm": lambda m, n, k: 2 * m * n * k + m * n,
        # [1, m, k] @ [k, n]
        "bmm": lambda m, n, k: 2 * m * n * k,
        # [n, k] @ [k]
        "mv": lambda m, n, k: 2 * n * k,
        # [m] @ [n]
        "outer": lambda m, n, k: m * n,
    }

    print("Matrix Operations Performance Test")
    print("===================================")

    for op_name, op_func in test_op_list.items():
        print(f"\n--- {op_name} ---")

        for test_dtype in test_dtype_list:
            for test_shape in test_shape_list:
                try:
                    m, n, k = test_shape
                    A = torch.randn([m, k], dtype=test_dtype, device="cpu", requires_grad=False)
                    B = torch.randn([k, n], dtype=test_dtype, device="cpu", requires_grad=False)
                    Bias = torch.zeros((n, ), dtype=test_dtype, device="cpu", requires_grad=False)

                    for _ in range(test_warm_up):
                        ref = op_func((A, B, Bias))

                    start = time.time()
                    for _ in range(test_iterations):
                        ref = op_func((A, B, Bias))
                    end = time.time()

                    gops = test_iterations * test_op_ops[op_name](m, n, k) / 1024 / 1024 / 1024 / (end - start)
                    print(
                        f"dtype {test_dtype} shape {test_shape}, cost {1000 * (end - start) / test_iterations:.3f} ms. "
                        f"gops {gops:.2f}")
                except Exception as e:
                    print(f"dtype {test_dtype} shape {test_shape}, Failed: {str(e)}")
