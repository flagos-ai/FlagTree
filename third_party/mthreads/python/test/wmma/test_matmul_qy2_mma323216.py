import itertools

import pytest
import torch

import triton
import triton.language as tl
import triton.ops
import torch_musa


def get_backend():
    return triton.runtime.driver.active.get_current_target().backend


@pytest.mark.parametrize(
    "BLOCK_M, BLOCK_N, BLOCK_K, SPLIT_K, NWARP, NSTAGE, M, N, K, AT, BT, ADTYPE, BDTYPE, INPUT_PRECISION, F8_FASTACCUM, ACC_DTYPE, OUTPUT_DTYPE",
    itertools.chain(
        *[[
            # 1 warp
            (32, 32, 16, 1, 1, 2, None, None, None, AT, BT, DTYPE, DTYPE, None, True, "float32", "float32"),
            (64, 32, 16, 1, 2, 2, None, None, None, AT, BT, DTYPE, DTYPE, None, True, "float32", "float32"),
            (32, 64, 16, 1, 2, 2, None, None, None, AT, BT, DTYPE, DTYPE, None, True, "float32", "float32"),
            (64, 64, 16, 1, 4, 2, None, None, None, AT, BT, DTYPE, DTYPE, None, True, "float32", "float32"),
            # 2 warp
            (64, 32, 16, 1, 2, 2, None, None, None, AT, BT, DTYPE, DTYPE, None, True, "float32", "float32"),
            (32, 64, 16, 1, 2, 2, None, None, None, AT, BT, DTYPE, DTYPE, None, True, "float32", "float32"),
            (128, 32, 16, 1, 4, 2, None, None, None, AT, BT, DTYPE, DTYPE, None, True, "float32", "float32"),
            (32, 128, 16, 1, 4, 2, None, None, None, AT, BT, DTYPE, DTYPE, None, True, "float32", "float32"),
            # 4 warp
            (128, 64, 16, 1, 8, 2, None, None, None, AT, BT, DTYPE, DTYPE, None, True, "float32", "float32"),
            (64, 128, 16, 1, 8, 2, None, None, None, AT, BT, DTYPE, DTYPE, None, True, "float32", "float32"),
            # variable input
            (64, 64, 16, 1, 2, 2, 128, 128, 64, AT, BT, DTYPE, DTYPE, None, True, "float32", "float32"),
            (128, 128, 16, 1, 8, 2, 256, 256, 128, AT, BT, DTYPE, DTYPE, None, True, "float32", "float32"),
            # Test input precision and fast accumulation flags (even though they won't affect FP16/BF16 MMA)
            (64, 64, 16, 1, 2, 2, 128, 128, 64, AT, BT, DTYPE, DTYPE, "tf32x3", False, "float32", "float32"),
            (64, 64, 16, 1, 2, 2, 128, 128, 64, AT, BT, DTYPE, DTYPE, None, False, "float32", "float32"),
        ] for DTYPE in ["float16", "bfloat16"] for AT in [False, True] for BT in [False, True]],
        # n-stage
        *[[
            (32, 32, 16, 1, 1, STAGES, 64, 64, 32, AT, BT, DTYPE, DTYPE, None, True, "float32", "float32"),
            (64, 64, 16, 1, 4, STAGES, 128, 128, 64, AT, BT, DTYPE, DTYPE, None, True, "float32", "float32"),
            (128, 128, 16, 1, 8, STAGES, 256, 256, 128, AT, BT, DTYPE, DTYPE, None, True, "float32", "float32"),
        ] for DTYPE in ["float16", "bfloat16"] for AT in [False, True] for BT in [False, True] for STAGES in [4]],
    ),
)
def test_op(BLOCK_M, BLOCK_N, BLOCK_K, SPLIT_K, NWARP, NSTAGE, M, N, K, AT, BT, ADTYPE, BDTYPE, INPUT_PRECISION,
            F8_FASTACCUM, ACC_DTYPE, OUTPUT_DTYPE):
    capability = torch_musa.get_device_capability()
    assert (capability[0] == 2 and capability[1] == 2) and "Only used for QY2."
    torch.manual_seed(0)

    def init_input(m, n, dtype, acc_dtype):
        if dtype == "int8":
            return torch.randint(-128, 127, (m, n), device="musa", dtype=torch.int8)
        # Use small range of values to prevent numerical issues
        min_exp = -4 if acc_dtype == "float16" else -10
        exponents = torch.randint(min_exp, 0, size=(m, n), device="musa")
        ret = (2.**exponents).to(getattr(torch, dtype))
        # Randomly flip signs
        signs = torch.randint(0, 2, (m, n), device="musa") * 2 - 1
        return ret * signs

    # nuke kernel decorators -- will set meta-parameters manually
    kwargs = {'BLOCK_M': BLOCK_M, 'BLOCK_N': BLOCK_N, 'BLOCK_K': BLOCK_K, 'SPLIT_K': SPLIT_K}
    pre_hook = None if SPLIT_K == 1 else lambda nargs: nargs['C'].zero_()
    configs = [triton.Config(kwargs=kwargs, num_warps=NWARP, num_stages=NSTAGE, pre_hook=pre_hook)]
    kernel = triton.ops._matmul.kernel
    kernel.configs = configs

    # get matrix shape
    M = BLOCK_M if M is None else M
    N = BLOCK_N if N is None else N
    K = BLOCK_K * SPLIT_K if K is None else K

    # allocate/transpose inputs
    a = init_input(M, K, ADTYPE, ACC_DTYPE)
    b = init_input(K, N, BDTYPE, ACC_DTYPE)
    a = a if not AT else a.T.contiguous().T
    b = b if not BT else b.T.contiguous().T

    # run test
    th_a = a.float() if a.dtype in [torch.float16, torch.bfloat16] else a
    th_b = b.float() if b.dtype in [torch.float16, torch.bfloat16] else b
    th_c = torch.matmul(th_a, th_b)  # FP32 output
    acc_dtype = torch.float32 if ACC_DTYPE in ["float32"] else torch.float32
    output_dtype = acc_dtype

    try:
        # Pass through input precision and fast accum flags even though they may not be used
        tt_c = triton.ops.matmul(a, b, acc_dtype, INPUT_PRECISION, F8_FASTACCUM, output_dtype)

        # Adjust tolerances based on input types
        rtol = 1e-2 if ADTYPE == "float16" or BDTYPE == "float16" else 5e-2
        atol = 1e-2 if ADTYPE == "float16" or BDTYPE == "float16" else 5e-2
        torch.testing.assert_close(th_c, tt_c, rtol=rtol, atol=atol)
    except triton.OutOfResources as e:
        pytest.skip(str(e))
