import torch

import triton
from triton.backends.compiler import GPUTarget
from triton.backends.ztc.driver import CPUDriver
import triton.language as tl


# Not used for testing but serves as a template to generate the lit test at
# test/Conversion/TritonToStructured/ridiculously_nested_loops.mlir
@triton.jit
def nested_who_knows_how_many_levels(in_ptr, out_ptr, stride_m, stride_n):
    offs_am = tl.arange(0, 2)
    offs_an = tl.arange(0, 2)
    a_ptrs = in_ptr + (offs_am[:, None] * stride_m +
                        offs_an[None, :] * stride_n)

    offs_cm = tl.arange(0, 2)
    offs_cn = tl.arange(0, 2)
    c_ptrs = out_ptr + stride_m * offs_cm[:, None] + stride_n * offs_cn[
        None, :]

    for i1 in range(0, 2):
        a1 = tl.load(a_ptrs)

        for j1 in range(0, 2):
            a_ptrs += 2 * stride_n
            a2 = tl.load(a_ptrs)

            for k1 in range(0, 2):
                a_ptrs += 2 * stride_n
                a3 = tl.load(a_ptrs)
                tl.store(c_ptrs, a1)
                c_ptrs += 2 * stride_n

                tl.store(c_ptrs, a2)
                c_ptrs += 2 * stride_n
                tl.store(c_ptrs, a3)
                c_ptrs += 2 * stride_n

                for i2 in range(0, 2):
                    a1 = tl.load(a_ptrs)

                    for j2 in range(0, 2):
                        a_ptrs += 2 * stride_n
                        a2 = tl.load(a_ptrs)

                        for k2 in range(0, 2):
                            a_ptrs += 2 * stride_n
                            a3 = tl.load(a_ptrs)
                            tl.store(c_ptrs, a1)
                            c_ptrs += 2 * stride_n

                            tl.store(c_ptrs, a2)
                            c_ptrs += 2 * stride_n
                            tl.store(c_ptrs, a3)
                            c_ptrs += 2 * stride_n

                            for i3 in range(0, 2):
                                a1 = tl.load(a_ptrs)

                                for j3 in range(0, 2):
                                    a_ptrs += 2 * stride_n
                                    a2 = tl.load(a_ptrs)

                                    for k3 in range(0, 2):
                                        a_ptrs += 2 * stride_n
                                        a3 = tl.load(a_ptrs)
                                        tl.store(c_ptrs, a1)
                                        c_ptrs += 2 * stride_n

                                        tl.store(c_ptrs, a2)
                                        c_ptrs += 2 * stride_n
                                        tl.store(c_ptrs, a3)
                                        c_ptrs += 2 * stride_n

                                        for i4 in range(0, 2):
                                            a1 = tl.load(a_ptrs)

                                            for j4 in range(0, 2):
                                                a_ptrs += 2 * stride_n
                                                a2 = tl.load(a_ptrs)

                                                for k4 in range(0, 2):
                                                    a_ptrs += 2 * stride_n
                                                    a3 = tl.load(a_ptrs)
                                                    tl.store(c_ptrs, a1)
                                                    c_ptrs += 2 * stride_n

                                                    tl.store(c_ptrs, a2)
                                                    c_ptrs += 2 * stride_n
                                                    tl.store(c_ptrs, a3)
                                                    c_ptrs += 2 * stride_n

                for i5 in range(0, 2):
                    a1 = tl.load(a_ptrs)

                    for j5 in range(0, 2):
                        a_ptrs += 2 * stride_n
                        a2 = tl.load(a_ptrs)

                        for k5 in range(0, 2):
                            a_ptrs += 2 * stride_n
                            a3 = tl.load(a_ptrs)
                            tl.store(c_ptrs, a1)
                            c_ptrs += 2 * stride_n

                            tl.store(c_ptrs, a2)
                            c_ptrs += 2 * stride_n
                            tl.store(c_ptrs, a3)
                            c_ptrs += 2 * stride_n

            a_ptrs += 2 * stride_n

        for i6 in range(0, 2):
            a1 = tl.load(a_ptrs)

            for j6 in range(0, 2):
                a_ptrs += 2 * stride_n
                a2 = tl.load(a_ptrs)

                for k6 in range(0, 2):
                    a_ptrs += 2 * stride_n
                    a3 = tl.load(a_ptrs)
                    tl.store(c_ptrs, a1)
                    c_ptrs += 2 * stride_n

                    tl.store(c_ptrs, a2)
                    c_ptrs += 2 * stride_n
                    tl.store(c_ptrs, a3)
                    c_ptrs += 2 * stride_n
            a_ptrs += 2 * stride_n


        a_ptrs += 2 * stride_n


@triton.jit
def nested_use_same_level_loop_results(in_ptr, out_ptr, stride_m, stride_n):
    offs_am = tl.arange(0, 2)
    offs_an = tl.arange(0, 2)
    a_ptrs = in_ptr + (offs_am[:, None] * stride_m +
                        offs_an[None, :] * stride_n)

    offs_cm = tl.arange(0, 2)
    offs_cn = tl.arange(0, 2)
    c_ptrs = out_ptr + stride_m * offs_cm[:, None] + stride_n * offs_cn[
        None, :]

    for i1 in range(0, 2):
        a1 = tl.load(a_ptrs)

        for j1 in range(0, 2):
            a_ptrs += 2 * stride_n

        for i6 in range(0, 2):
            a1 = tl.load(a_ptrs)
            a_ptrs += 2 * stride_n
            a3 = tl.load(a_ptrs)
            tl.store(c_ptrs, a1)
            c_ptrs += 2 * stride_n

            c_ptrs += 2 * stride_n
            tl.store(c_ptrs, a3)
            c_ptrs += 2 * stride_n
            a_ptrs += 2 * stride_n

        a_ptrs += 2 * stride_n

@triton.jit
def nested2_complex_body(a_ptr, c_ptr, stride_m, stride_n):
    offs_am = tl.arange(0, 2)
    offs_an = tl.arange(0, 2)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_m +
                        offs_an[None, :] * stride_n)

    offs_cm = tl.arange(0, 2)
    offs_cn = tl.arange(0, 2)
    c_ptrs = c_ptr + stride_m * offs_cm[:, None] + stride_n * offs_cn[
        None, :]

    for i in range(0, 2):
        a_ptrs_copy = a_ptrs
        c_ptrs_copy = c_ptrs

        a_ptrs += 1
        c_ptrs += 1

        for j in range(0, 2):
            a2 = tl.load(a_ptrs)
            tl.store(c_ptrs, a2)
            a_ptrs += 3
            c_ptrs += 3

        a_ptrs = a_ptrs_copy + 2 * stride_m + 1
        c_ptrs = c_ptrs_copy + 2 * stride_m + 1



@triton.jit
def nested2_use_loop_results(in_ptr, out_ptr, stride_m, stride_n):
    offs_am = tl.arange(0, 2)
    offs_an = tl.arange(0, 2)
    a_ptrs = in_ptr + (offs_am[:, None] * stride_m +
                        offs_an[None, :] * stride_n)

    offs_cm = tl.arange(0, 2)
    offs_cn = tl.arange(0, 2)
    c_ptrs = out_ptr + stride_m * offs_cm[:, None] + stride_n * offs_cn[
        None, :]

    for i in range(0, 2):
        a2 = tl.load(a_ptrs)
        tl.store(c_ptrs, a2)

        a_ptrs += 4 * stride_n
        c_ptrs += 4 * stride_n


        for j in range(0, 2):
            a2 = tl.load(a_ptrs)
            tl.store(c_ptrs, a2)
            a_ptrs += 4 * stride_n
            c_ptrs += 4 * stride_n


@triton.jit
def nested3(in_ptr, out_ptr, stride_m, stride_n):
    offs_am = tl.arange(0, 2)
    offs_an = tl.arange(0, 2)
    a_ptrs = in_ptr + (offs_am[:, None] * stride_m +
                        offs_an[None, :] * stride_n)

    offs_cm = tl.arange(0, 2)
    offs_cn = tl.arange(0, 2)
    c_ptrs = out_ptr + stride_m * offs_cm[:, None] + stride_n * offs_cn[
        None, :]

    for i in range(0, 2):
        a1 = tl.load(a_ptrs)

        for j in range(0, 2):
            a_ptrs += 2 * stride_n
            a2 = tl.load(a_ptrs)

            for k in range(0, 2):
                a_ptrs += 2 * stride_n
                a3 = tl.load(a_ptrs)
                tl.store(c_ptrs, a1)
                c_ptrs += 2 * stride_n

                tl.store(c_ptrs, a2)
                c_ptrs += 2 * stride_n
                tl.store(c_ptrs, a3)
                c_ptrs += 2 * stride_n


        a_ptrs += 2 * stride_n

def test_nested3():
    n_rows = 4
    n_cols = 48
    expected = torch.tensor([[ 0,  1,  2,  3,  4,  5,  0,  1,  2,  3,  6,  7,  0,  1,
          8,  9, 10, 11,  0,  1,  8,  9, 12, 13, 14, 15, 16, 17,
         18, 19, 14, 15, 16, 17, 20, 21, 14, 15, 22, 23, 24, 25,
         14, 15, 22, 23, 26, 27],
        [48, 49, 50, 51, 52, 53, 48, 49, 50, 51, 54, 55, 48, 49,
         56, 57, 58, 59, 48, 49, 56, 57, 60, 61, 62, 63, 64, 65,
         66, 67, 62, 63, 64, 65, 68, 69, 62, 63, 70, 71, 72, 73,
         62, 63, 70, 71, 74, 75],
        [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
          0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
          0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
          0,  0,  0,  0,  0,  0],
        [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
          0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
          0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
          0,  0,  0,  0,  0,  0]], dtype=torch.int32, device='cpu')
    triton.runtime.driver.set_active(CPUDriver())
    x = torch.arange(0, n_rows * n_cols, device="cpu", dtype=torch.int32).reshape([n_rows, n_cols])
    output = torch.zeros([n_rows, n_cols], device=x.device, dtype=x.dtype)
    grid = lambda meta: (n_cols // 4,)

    print('before:')
    print(x)
    print(output)

    nested3[grid](x, output, x.stride(0), x.stride(1))
    print(output)
    torch.testing.assert_close(output, expected, rtol=0.001, atol=1e-5)
    print("Pass!")

    src = triton.compiler.ASTSource(
        fn=nested3,
        signature="*fp32,*fp32,i32,i32",
    )
    ret = triton.compile(
        src,
    )
    print(ret.asm["ttir"])
    print('Pass')


def test_nested2_use_loop_results():
    n_rows = 4
    n_cols = 32
    expected = torch.tensor([[ 0,  1,  0,  0,  4,  5,  0,  0,  8,  9,  0,  0, 12, 13,  0,  0, 16, 17,
          0,  0, 20, 21,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
        [32, 33,  0,  0, 36, 37,  0,  0, 40, 41,  0,  0, 44, 45,  0,  0, 48, 49,
          0,  0, 52, 53,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
        [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
          0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
        [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
          0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0]],
       device='cpu', dtype=torch.int32)
    # x = torch.arange(0, n_rows * n_cols, device="cuda", dtype=torch.int32).reshape([n_rows, n_cols])
    triton.runtime.driver.set_active(CPUDriver())
    x = torch.arange(0, n_rows * n_cols, device="cpu", dtype=torch.int32).reshape([n_rows, n_cols])
    output = torch.zeros([n_rows, n_cols], device=x.device, dtype=x.dtype)
    grid = lambda meta: (n_cols // 4,)

    print('before:')
    print(x)
    print(output)

    nested2_use_loop_results[grid](x, output, x.stride(0), x.stride(1))
    print(output)
    torch.testing.assert_close(output, expected, rtol=0.001, atol=1e-5)
    print("Pass!")

    src = triton.compiler.ASTSource(
        fn=nested2_use_loop_results,
        signature="*fp32,*fp32,i32,i32",
    )
    ret = triton.compile(
        src,
    )
    print(ret.asm["ttir"])
    print('Pass')


def test_nested2_complex_body():
    n_rows = 4
    n_cols = 8
    grid = lambda meta: (n_cols // 4,)
    expected = torch.tensor([[ 0,  1,  2,  0,  4,  5,  0,  0],
        [ 0,  9, 10,  0, 12, 13,  0,  0],
        [ 0,  0, 18, 19,  0, 21, 22,  0],
        [ 0,  0, 26, 27,  0, 29, 30,  0]], device='cpu', dtype=torch.int32)


    x = torch.arange(0, n_rows * n_cols, device="cpu", dtype=torch.int32).reshape([n_rows, n_cols])
    triton.runtime.driver.set_active(CPUDriver())
    output = torch.zeros([n_rows, n_cols], device=x.device, dtype=x.dtype)


    print('before:')
    print(x)
    print(output)

    nested2_complex_body[grid](x, output, x.stride(0), x.stride(1))
    print(output)
    torch.testing.assert_close(output, expected, rtol=0.001, atol=1e-5)
    print("Pass!")

    src = triton.compiler.ASTSource(
        fn=nested2_complex_body,
        signature="*fp32,*fp32,i32,i32",
    )
    ret = triton.compile(
        src,
    )
    print(ret.asm["ttir"])
    print('Pass')

def test_nested2_use_same_level_loop_result():
    n_rows = 4
    n_cols = 32
    grid = lambda meta: (n_cols // 4,)
    expected = torch.tensor([[ 4,  5,  0,  0,  6,  7,  8,  9,  0,  0, 10, 11, 18, 19,  0,  0, 20, 21,
         22, 23,  0,  0, 24, 25,  0,  0,  0,  0,  0,  0,  0,  0],
        [36, 37,  0,  0, 38, 39, 40, 41,  0,  0, 42, 43, 50, 51,  0,  0, 52, 53,
         54, 55,  0,  0, 56, 57,  0,  0,  0,  0,  0,  0,  0,  0],
        [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
          0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
        [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
          0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0]],
       device='cpu', dtype=torch.int32)


    x = torch.arange(0, n_rows * n_cols, device="cpu", dtype=torch.int32).reshape([n_rows, n_cols])
    triton.runtime.driver.set_active(CPUDriver())
    output = torch.zeros([n_rows, n_cols], device=x.device, dtype=x.dtype)


    print('before:')
    print(x)
    print(output)

    nested_use_same_level_loop_results[grid](x, output, x.stride(0), x.stride(1))
    print(output)
    torch.testing.assert_close(output, expected, rtol=0.001, atol=1e-5)
    print("Pass!")

    src = triton.compiler.ASTSource(
        fn=nested_use_same_level_loop_results,
        signature="*fp32,*fp32,i32,i32",
    )
    ret = triton.compile(
        src,
    )
    print(ret.asm["ttir"])
    print('Pass')
