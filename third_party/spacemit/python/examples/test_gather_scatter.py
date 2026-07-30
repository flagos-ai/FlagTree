import torch

import triton
import triton.language as tl

from triton.backends.triton_shared.driver import CPUDriver


@triton.jit
def gather_simple_no_mask(in0, out0):
    offs = tl.arange(0, 64)
    out_offs = tl.arange(0, 64)
    for i in range(0, 2):
        offs = offs // 10 + (i + 1 * 5) % 64
        a = tl.load(in0 + offs)
        tl.store(out0 + out_offs, a)
        offs += 64
        out_offs += 64


@triton.jit
def gather_simple_mask_no_other(in0, out0):
    offs = tl.arange(0, 64)
    out_offs = tl.arange(0, 64)
    mask_bound = 8
    for i in range(0, 2):
        gather_offs = offs // 4
        a = tl.load(in0 + gather_offs, mask=gather_offs < mask_bound)
        tl.store(out0 + out_offs, a)
        mask_bound += 16
        offs += 64
        out_offs += 64


@triton.jit
def gather_simple_mask_with_other(in0, out0):
    offs = tl.arange(0, 64)
    out_offs = tl.arange(0, 64)
    mask_bound = 8
    for i in range(0, 2):
        gather_offs = offs // 4
        a = tl.load(in0 + gather_offs, mask=gather_offs < mask_bound, other=-1)
        tl.store(out0 + out_offs, a)
        mask_bound += 16
        offs += 64
        out_offs += 64


@triton.jit
def masked_gather_scatter(in0, out0):
    offs = tl.arange(0, 64)
    out_offs = tl.arange(0, 64)
    for i in range(0, 2):
        offs = offs // 12
        mask = offs < 64
        store_mask = out_offs < 77
        a = tl.load(in0 + offs, mask=mask, other=99)
        tl.store(out0 + out_offs, a, mask=store_mask)
        offs += 64
        out_offs += 64


@triton.jit
def complex_gather_scatter(in0, out0):
    offs = tl.arange(0, 32)
    out_offs = tl.arange(0, 32)
    for i in range(0, 2):
        offs = offs // 3 + i
        mask = offs < 32
        a = tl.load(in0 + offs, mask=mask)
        tl.store(out0 + offs, a)
        offs += 32
        out_offs += 32

        for j in range(0, 2):
            offs = offs // ((i + 1) * (j + 1)) + i
            mask = offs < 32
            a = tl.load(in0 + offs, mask=mask)
            tl.store(out0 + offs, a)
            offs += 32
            out_offs += 32

        offs = offs // 3 + i
        mask = offs < 32
        a = tl.load(in0 + offs, mask=mask)
        tl.store(out0 + offs, a)
        offs += 32
        out_offs += 32


def run_test(triton_kernel, device, expected_output):
    SIZE = 128
    input = torch.arange(2, SIZE + 2, device=device, dtype=torch.int32)
    output = torch.full((SIZE, ), -1, device=device, dtype=torch.int32)

    if device == 'cpu':
        triton.runtime.driver.set_active(CPUDriver())

    grid = lambda meta: (1, )

    print(output)
    triton_kernel[grid](input, output)
    print(input)
    print(output)
    torch.testing.assert_close(output, expected_output)


def test_gather_simple_no_mask(device):
    expected_output = torch.tensor([
        7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 10, 10, 10, 10, 10,
        10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 13, 13, 13,
        13, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15,
        15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15,
        15, 15, 15, 15, 15, 15, 15, 15, 15
    ], device=device, dtype=torch.int32)
    run_test(gather_simple_no_mask, device, expected_output)


def test_gather_simple_mask_no_other(device):
    expected_output = torch.tensor([
        2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8, 9, 9, 9, 9, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 18, 18, 18, 18, 19, 19, 19, 19,
        20, 20, 20, 20, 21, 21, 21, 21, 22, 22, 22, 22, 23, 23, 23, 23, 24, 24, 24, 24, 25, 25, 25, 25, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    ], device=device, dtype=torch.int32)
    run_test(gather_simple_mask_no_other, device, expected_output)


def test_gather_simple_mask_with_other(device):
    expected_output = torch.tensor([
        2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8, 9, 9, 9, 9, -1, -1, -1, -1,
        -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
        18, 18, 18, 18, 19, 19, 19, 19, 20, 20, 20, 20, 21, 21, 21, 21, 22, 22, 22, 22, 23, 23, 23, 23, 24, 24, 24, 24,
        25, 25, 25, 25, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
        -1, -1, -1, -1, -1, -1, -1, -1
    ], device=device, dtype=torch.int32)
    run_test(gather_simple_mask_with_other, device, expected_output)


def test_masked_gather_scatter(device):
    expected_output = torch.tensor([
        2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5,
        5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
        7, 7, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
        -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1
    ], device=device, dtype=torch.int32)
    run_test(masked_gather_scatter, device, expected_output)


def test_complex_gather_scatter(device):
    expected_output = torch.tensor([
        2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, -1, -1, -1, -1, 17, 18, -1, 20, 21, -1, 23, 24, 25, -1, -1, 28, -1, -1, -1,
        -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
        -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
        -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
        -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1
    ], device=device, dtype=torch.int32)
    run_test(complex_gather_scatter, device, expected_output)
