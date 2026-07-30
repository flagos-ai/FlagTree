# python3 -m pytest tests/test_smt.py::test_descriptor_load -v -s
import torch
import triton

from triton.backends.spine_triton.driver import CPUDriver

triton.runtime.driver.set_active(CPUDriver())
import triton.language as tl
import pytest
import triton.language.extra.smt as smt
import os


@pytest.mark.parametrize("M, SUB_BLK_M, BLOCK_SIZE_M, BLOCK_SIZE_K, MICRO_M, MICRO_K", [
    (512, 64, 32, 256, 16, 8),
])
def test_descriptor_load(M, SUB_BLK_M, BLOCK_SIZE_M, BLOCK_SIZE_K, MICRO_M, MICRO_K):

    def run_descriptor_load():

        @triton.jit
        def descriptor_load(
            a_ptr,
            output_ptr,
            M,
            K,
            stride_am,
            stride_ak,
            stride_om0,
            stride_om1,
            stride_om2,
            stride_om3,
            num_blocks_m: tl.constexpr,
            num_blocks_k: tl.constexpr,
            SUB_BLK_M: tl.constexpr,
            BLOCK_SIZE_M: tl.constexpr,
            BLOCK_SIZE_K: tl.constexpr,
            MICRO_M: tl.constexpr,
            MICRO_K: tl.constexpr,
        ):
            pid = tl.program_id(0)

            a_block_ptr = tl.make_block_ptr(
                base=a_ptr,
                shape=[M, K],
                strides=[stride_am, stride_ak],
                offsets=[pid * BLOCK_SIZE_M, 0],
                block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_K],
                order=[1, 0],
            )

            a_descriptor_load = smt.descriptor_load(a_block_ptr, (0, 0))
            a = smt.view(a_descriptor_load, (0, 0), (SUB_BLK_M, BLOCK_SIZE_K), (MICRO_M, MICRO_K))

            output_block_ptr = tl.make_block_ptr(
                base=output_ptr,
                shape=[num_blocks_m, num_blocks_k, MICRO_M, MICRO_K],
                strides=[stride_om0, stride_om1, stride_om2, stride_om3],
                offsets=[0, 0, 0, 0],
                block_shape=[num_blocks_m, num_blocks_k, MICRO_M, MICRO_K],
                order=[3, 2, 1, 0],
            )

            tl.store(output_block_ptr, a)

        torch.manual_seed(42)
        K = BLOCK_SIZE_K
        device = "cpu"
        a = torch.randn((M, K), dtype=torch.float16, device=device)
        print("a", a)

        num_blocks_m = (SUB_BLK_M + MICRO_M - 1) // MICRO_M
        num_blocks_k = (BLOCK_SIZE_K + MICRO_K - 1) // MICRO_K
        output_shape = (num_blocks_m, num_blocks_k, MICRO_M, MICRO_K)
        output = torch.zeros(output_shape, dtype=torch.float16, device=device)

        grid = (triton.cdiv(M, BLOCK_SIZE_M), )

        descriptor_load[grid](a, output, M, K, a.stride(0), a.stride(1), output.stride(0), output.stride(1),
                              output.stride(2), output.stride(3), num_blocks_m=num_blocks_m, num_blocks_k=num_blocks_k,
                              SUB_BLK_M=SUB_BLK_M, BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_K=BLOCK_SIZE_K,
                              MICRO_M=MICRO_M, MICRO_K=MICRO_K)
        return output

    os.environ['SPINE_TRITON_USE_REF_PIPELINE'] = '1'
    os.environ['TRITON_ALWAYS_COMPILE'] = '1'
    output_ref = run_descriptor_load()
    print("=== set SPINE_TRITON_USE_REF_PIPELINE ===")
    print("output_ref shape:", output_ref.shape)
    print("output_ref sum:", output_ref.sum().item())
    print("output_ref mean:", output_ref.mean().item())
    print("output_ref:", output_ref)

    del os.environ['SPINE_TRITON_USE_REF_PIPELINE']
    output = run_descriptor_load()
    print("\n=== unset SPINE_TRITON_USE_REF_PIPELINE ===")
    print("Output shape:", output.shape)
    print("Output sum:", output.sum().item())
    print("Output mean:", output.mean().item())
    print("Output:", output)

    print("\n=== Result Comparison ===")
    print("Shapes are identical:", output_ref.shape == output.shape)
    print("Values are exactly the same:", torch.allclose(output_ref, output))
    print("Maximum absolute difference:", torch.max(torch.abs(output_ref - output)).item())
    print("Mean absolute difference:", torch.mean(torch.abs(output_ref - output)).item())

    assert output_ref.shape == output.shape, "Shapes are not identical"
    assert torch.allclose(output_ref, output, rtol=1e-5, atol=1e-8), "Values are not exactly the same"

    max_diff = torch.max(torch.abs(output_ref - output)).item()
    mean_diff = torch.mean(torch.abs(output_ref - output)).item()

    print(f"✅ All checks passed! Max diff: {max_diff:.2e}, Mean diff: {mean_diff:.2e}")
    print("✅ Continuing execution...")


@pytest.mark.parametrize("M, SUB_BLK_M, BLOCK_SIZE_M, BLOCK_SIZE_K, MICRO_M, MICRO_K", [
    (512, 64, 32, 256, 16, 8),
])
def test_descriptor_load_transpose(M, SUB_BLK_M, BLOCK_SIZE_M, BLOCK_SIZE_K, MICRO_M, MICRO_K):

    def run_descriptor_load_transpose():

        @triton.jit
        def descriptor_load_transpose(
            a_ptr,
            output_ptr,
            M,
            K,
            stride_am,
            stride_ak,
            stride_om0,
            stride_om1,
            stride_om2,
            stride_om3,
            num_blocks_m: tl.constexpr,
            num_blocks_k: tl.constexpr,
            SUB_BLK_M: tl.constexpr,
            BLOCK_SIZE_M: tl.constexpr,
            BLOCK_SIZE_K: tl.constexpr,
            MICRO_M: tl.constexpr,
            MICRO_K: tl.constexpr,
        ):
            pid = tl.program_id(0)

            a_block_ptr = tl.make_block_ptr(
                base=a_ptr,
                shape=[M, K],
                strides=[stride_am, stride_ak],
                offsets=[pid * BLOCK_SIZE_M, 0],
                block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_K],
                order=[1, 0],
            )

            a_descriptor_load = smt.descriptor_load(a_block_ptr, (0, 0))
            a = smt.view(a_descriptor_load, (0, 0), (SUB_BLK_M, BLOCK_SIZE_K), (MICRO_M, MICRO_K))
            a = tl.permute(a, (1, 0, 3, 2))

            output_block_ptr = tl.make_block_ptr(
                base=output_ptr,
                shape=[num_blocks_k, num_blocks_m, MICRO_K, MICRO_M],
                strides=[stride_om0, stride_om1, stride_om2, stride_om3],
                offsets=[0, 0, 0, 0],
                block_shape=[num_blocks_k, num_blocks_m, MICRO_K, MICRO_M],
                order=[3, 2, 1, 0],
            )

            tl.store(output_block_ptr, a)

        torch.manual_seed(42)
        K = BLOCK_SIZE_K
        device = "cpu"
        a = torch.randn((M, K), dtype=torch.float16, device=device)
        print("a", a)

        num_blocks_m = (SUB_BLK_M + MICRO_M - 1) // MICRO_M
        num_blocks_k = (BLOCK_SIZE_K + MICRO_K - 1) // MICRO_K
        output_shape = (num_blocks_k, num_blocks_m, MICRO_K, MICRO_M)
        output = torch.zeros(output_shape, dtype=torch.float16, device=device)

        grid = (triton.cdiv(M, BLOCK_SIZE_M), )

        descriptor_load_transpose[grid](a, output, M, K, a.stride(0), a.stride(1), output.stride(0), output.stride(1),
                                        output.stride(2), output.stride(3), num_blocks_m=num_blocks_m,
                                        num_blocks_k=num_blocks_k, SUB_BLK_M=SUB_BLK_M, BLOCK_SIZE_M=BLOCK_SIZE_M,
                                        BLOCK_SIZE_K=BLOCK_SIZE_K, MICRO_M=MICRO_M, MICRO_K=MICRO_K)
        return output

    os.environ['SPINE_TRITON_USE_REF_PIPELINE'] = '1'
    os.environ['TRITON_ALWAYS_COMPILE'] = '1'
    output_ref = run_descriptor_load_transpose()
    print("=== set SPINE_TRITON_USE_REF_PIPELINE ===")
    print("output_ref shape:", output_ref.shape)
    print("output_ref sum:", output_ref.sum().item())
    print("output_ref mean:", output_ref.mean().item())
    print("output_ref:", output_ref)

    del os.environ['SPINE_TRITON_USE_REF_PIPELINE']
    output = run_descriptor_load_transpose()
    print("\n=== unset SPINE_TRITON_USE_REF_PIPELINE ===")
    print("Output shape:", output.shape)
    print("Output sum:", output.sum().item())
    print("Output mean:", output.mean().item())
    print("Output:", output)

    print("\n=== Result Comparison ===")
    print("Shapes are identical:", output_ref.shape == output.shape)
    print("Values are exactly the same:", torch.allclose(output_ref, output))
    print("Maximum absolute difference:", torch.max(torch.abs(output_ref - output)).item())
    print("Mean absolute difference:", torch.mean(torch.abs(output_ref - output)).item())

    assert output_ref.shape == output.shape, "Shapes are not identical"
    assert torch.allclose(output_ref, output, rtol=1e-5, atol=1e-8), "Values are not exactly the same"

    max_diff = torch.max(torch.abs(output_ref - output)).item()
    mean_diff = torch.mean(torch.abs(output_ref - output)).item()

    print(f"✅ All checks passed! Max diff: {max_diff:.2e}, Mean diff: {mean_diff:.2e}")
    print("✅ Continuing execution...")


@pytest.mark.parametrize("BLOCK_SIZE", [
    (8),
    (16),
])
def test_mbarrier(BLOCK_SIZE):

    def run_simple_test():

        @triton.jit
        def mbarrier_kernel(
            input_ptr,
            output_ptr,
            N,
            BLOCK_SIZE: tl.constexpr,
        ):
            pid = tl.program_id(0)

            bar = smt.mbarrier(flag=0, expect_count=3)

            offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
            mask = offs < N
            x = tl.load(input_ptr + offs, mask=mask, other=0.0)

            for s in smt.parallel(0, 3):
                smt.barrier_arrive(bar)
                smt.barrier_wait(bar, flag=1)

            tl.store(output_ptr + offs, x * 2.0, mask=mask)

        torch.manual_seed(42)
        device = "cpu"
        N = BLOCK_SIZE * 2
        input_data = torch.randn((N, ), dtype=torch.float16, device=device)
        output = torch.zeros((N, ), dtype=torch.float16, device=device)

        grid = (triton.cdiv(N, BLOCK_SIZE), )

        mbarrier_kernel[grid](
            input_data,
            output,
            N,
            BLOCK_SIZE=BLOCK_SIZE,
        )

        return output, input_data

    output, input_data = run_simple_test()

    output, _ = run_simple_test()
    print("\n=== without SPINE_TRITON_USE_REF_PIPELINE ===")
    print("output:", output)

    expected = input_data * 2.0

    assert torch.allclose(output, expected, rtol=1e-5, atol=1e-8), "Output doesn't match expected"

    print("\n✅ mbarrier simple test passed!")


@pytest.mark.parametrize("BLOCK_SIZE", [
    (8),
    (16),
])
def test_global_mbarrier(BLOCK_SIZE):

    def run_simple_test():

        @triton.jit
        def global_mbarrier_kernel(
            input_ptr,
            output_ptr,
            N,
            BLOCK_SIZE: tl.constexpr,
        ):
            pid = tl.program_id(0)
            bar = smt.global_mbarrier(0)
            smt.barrier_set_expect(bar, 2)

            offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
            mask = offs < N
            x = tl.load(input_ptr + offs, mask=mask, other=0.0)
            x = x * 2.0
            smt.barrier_arrive(bar)
            smt.barrier_wait(bar, flag=1)

            tl.store(output_ptr + offs, x, mask=mask)

        torch.manual_seed(42)
        device = "cpu"
        N = BLOCK_SIZE * 2
        input_data = torch.randn((N, ), dtype=torch.float16, device=device)
        output = torch.zeros((N, ), dtype=torch.float16, device=device)

        grid = (triton.cdiv(N, BLOCK_SIZE), )

        global_mbarrier_kernel[grid](
            input_data,
            output,
            N,
            BLOCK_SIZE=BLOCK_SIZE,
        )

        return output, input_data

    output, input_data = run_simple_test()

    print("\n=== global_mbarrier and barrier_set_expect test ===")
    print("output:", output)

    expected = input_data * 2.0

    assert torch.allclose(output, expected, rtol=1e-5, atol=1e-8), \
        "Output doesn't match expected"

    print(f"\n✅ global_mbarrier and barrier_set_expect test passed! (BLOCK_SIZE={BLOCK_SIZE})")


@pytest.mark.parametrize("M, N, K, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, MICRO_M, MICRO_N, MICRO_K", [
    (512, 256, 256, 32, 32, 512, 16, 32, 8),
])
def test_mmt4d(M, N, K, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, MICRO_M, MICRO_N, MICRO_K):

    def run_mmt4d():

        @triton.jit
        def mmt4d(
            a_ptr,
            b_ptr,
            c_ptr,
            M,
            N,
            K,
            stride_am,
            stride_ak,
            stride_bk,
            stride_bn,
            stride_cm,
            stride_cn,
            BLOCK_SIZE_M: tl.constexpr,
            BLOCK_SIZE_N: tl.constexpr,
            BLOCK_SIZE_K: tl.constexpr,
            MICRO_M: tl.constexpr,
            MICRO_K: tl.constexpr,
            MICRO_N: tl.constexpr,
        ):
            pid_m = tl.program_id(0)
            pid_n = tl.program_id(1)

            a_block_ptr = tl.make_block_ptr(
                base=a_ptr,
                shape=[M, K],
                strides=[stride_am, stride_ak],
                offsets=[pid_m * BLOCK_SIZE_M, 0],
                block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_K],
                order=[1, 0],
            )

            a_view = smt.view(a_block_ptr, (0, 0), (BLOCK_SIZE_M, BLOCK_SIZE_K), (MICRO_M, MICRO_K))

            b_block_ptr = tl.make_block_ptr(
                base=b_ptr,
                shape=[K, N],
                strides=[stride_bk, stride_bn],
                offsets=[0, pid_n * BLOCK_SIZE_N],
                block_shape=[BLOCK_SIZE_K, BLOCK_SIZE_N],
                order=[1, 0],
            )

            b_view = smt.view(b_block_ptr, (0, 0), (BLOCK_SIZE_K, BLOCK_SIZE_N), (MICRO_K, MICRO_N))

            a = tl.load(a_view)
            b = tl.load(b_view)

            c = smt.dot(a, b)
            c = smt.view(c, (0, 0), (BLOCK_SIZE_M, BLOCK_SIZE_N), (1, 1))
            c_block_ptr = tl.make_block_ptr(
                base=c_ptr,
                shape=[M, N],
                strides=[stride_cm, stride_cn],
                offsets=[pid_m * BLOCK_SIZE_M, pid_n * BLOCK_SIZE_N],
                block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_N],
                order=[1, 0],
            )
            tl.store(c_block_ptr, c, boundary_check=(0, 1))

        torch.manual_seed(42)
        # K = BLOCK_SIZE_K
        device = "cpu"
        a = torch.randn((M, K), dtype=torch.float16, device=device)
        b = torch.randn((K, N), dtype=torch.float16, device=device)

        c = torch.empty((M, N), device=a.device, dtype=a.dtype)
        # launch kernel
        grid = lambda META: (
            triton.cdiv(M, META["BLOCK_SIZE_M"]),
            triton.cdiv(N, META["BLOCK_SIZE_N"]),
        )
        # BLOCK_SIZE_K = triton.next_power_of_2(K)

        mmt4d[grid](
            a,
            b,
            c,
            M,
            N,
            K,
            a.stride(0),
            a.stride(1),
            b.stride(0),
            b.stride(1),
            c.stride(0),
            c.stride(1),
            BLOCK_SIZE_M=128,
            BLOCK_SIZE_N=128,
            BLOCK_SIZE_K=BLOCK_SIZE_K,
            MICRO_M=MICRO_M,
            MICRO_N=MICRO_N,
            MICRO_K=MICRO_K,
        )
        return c

    os.environ['SPINE_TRITON_USE_REF_PIPELINE'] = '1'
    os.environ['TRITON_ALWAYS_COMPILE'] = '1'
    output_ref = run_mmt4d()
    print("=== set SPINE_TRITON_USE_REF_PIPELINE ===")
    print("output_ref shape:", output_ref.shape)
    print("output_ref sum:", output_ref.sum().item())
    print("output_ref mean:", output_ref.mean().item())
    print("output_ref:", output_ref)

    del os.environ['SPINE_TRITON_USE_REF_PIPELINE']
    output = run_mmt4d()
    print("\n=== unset SPINE_TRITON_USE_REF_PIPELINE ===")
    print("Output shape:", output.shape)
    print("Output sum:", output.sum().item())
    print("Output mean:", output.mean().item())
    print("Output:", output)

    print("\n=== Result Comparison ===")
    print("Shapes are identical:", output_ref.shape == output.shape)
    print("Values are exactly the same:", torch.allclose(output_ref, output))
    print("Maximum absolute difference:", torch.max(torch.abs(output_ref - output)).item())
    print("Mean absolute difference:", torch.mean(torch.abs(output_ref - output)).item())

    assert output_ref.shape == output.shape, "Shapes are not identical"
    assert torch.allclose(output_ref, output, rtol=1e-5, atol=1e-8), "Values are not exactly the same"

    max_diff = torch.max(torch.abs(output_ref - output)).item()
    mean_diff = torch.mean(torch.abs(output_ref - output)).item()

    print(f"✅ All checks passed! Max diff: {max_diff:.2e}, Mean diff: {mean_diff:.2e}")
    print("✅ Continuing execution...")
