import torch

import triton
import triton.language as tl

from triton.backends.triton_shared.driver import CPUDriver


@triton.jit
def gather_row(in0, out0, X: tl.constexpr = 2, Y: tl.constexpr = 4, Z: tl.constexpr = 8):
    offs_2d_x = tl.arange(0, X * Y)
    offs_2d_y = tl.arange(0, Z)

    offs_3d_x = offs_2d_x[:, None] // Y
    offs_3d_y = offs_2d_x[:, None] % Y

    offs_3d_z = offs_2d_y[None, :]

    stride_x = Y * Z
    stride_y = Z
    stride_z = 1

    offs = offs_3d_x * stride_x + offs_3d_y * stride_y + offs_3d_z * stride_z
    a = tl.load(in0 + offs)
    tl.store(out0 + offs, a)


def test_gather_row(device):
    SIZE = 64
    input = torch.arange(2, SIZE + 2, device=device, dtype=torch.int32)
    input3D = input.reshape(2, 4, 8)
    output = torch.full((SIZE, ), -1, device=device, dtype=torch.int32)

    if device == "cpu":
        triton.runtime.driver.set_active(CPUDriver())

    gather_row[
        1,
    ](input3D, output)
    torch.testing.assert_close(output, input)


@triton.jit
def gather_row_ld_mask(in0, out0, X: tl.constexpr = 2, Y: tl.constexpr = 4, Z: tl.constexpr = 8):
    offs_2d_x = tl.arange(0, X * Y)
    offs_2d_y = tl.arange(0, Z)

    offs_3d_x = offs_2d_x[:, None] // Y
    offs_3d_y = offs_2d_x[:, None] % Y

    offs_3d_z = offs_2d_y[None, :]

    stride_x = Y * Z
    stride_y = Z
    stride_z = 1

    offs = offs_3d_x * stride_x + offs_3d_y * stride_y + offs_3d_z * stride_z
    a = tl.load(in0 + offs, mask=offs_2d_y[None, :] < 4, other=0)
    tl.store(out0 + offs, a)


def test_gather_row_ld_mask(device):
    SIZE = 64
    input = torch.arange(2, SIZE + 2, device=device, dtype=torch.int32)
    input3D = input.reshape(2, 4, 8)
    output = torch.full((SIZE, ), -1, device=device, dtype=torch.int32)

    if device == "cpu":
        triton.runtime.driver.set_active(CPUDriver())

    gather_row_ld_mask[
        1,
    ](input3D, output)
    ref = input3D
    ref[:, :, 4:] = 0
    ref = ref.reshape(SIZE, )
    torch.testing.assert_close(output, ref)


@triton.jit
def gather_row_st_mask(in0, out0, X: tl.constexpr = 2, Y: tl.constexpr = 4, Z: tl.constexpr = 8):
    offs_2d_x = tl.arange(0, X * Y)
    offs_2d_y = tl.arange(0, Z)

    offs_3d_x = offs_2d_x[:, None] // Y
    offs_3d_y = offs_2d_x[:, None] % Y

    offs_3d_z = offs_2d_y[None, :]

    stride_x = Y * Z
    stride_y = Z
    stride_z = 1

    offs = offs_3d_x * stride_x + offs_3d_y * stride_y + offs_3d_z * stride_z
    a = tl.load(in0 + offs)
    tl.store(out0 + offs, a, mask=offs_2d_y[None, :] < 4)


def test_gather_row_st_mask(device):
    SIZE = 64
    input = torch.arange(2, SIZE + 2, device=device, dtype=torch.int32)
    input3D = input.reshape(2, 4, 8)
    output = torch.full((SIZE, ), -1, device=device, dtype=torch.int32)

    if device == "cpu":
        triton.runtime.driver.set_active(CPUDriver())

    gather_row_st_mask[
        1,
    ](input3D, output)
    ref = input3D
    ref[:, :, 4:] = -1
    ref = ref.reshape(SIZE, )
    torch.testing.assert_close(output, ref)


@triton.jit
def gather_column(in0, out0, X: tl.constexpr = 2, Y: tl.constexpr = 4, Z: tl.constexpr = 8):
    offs_2d_x = tl.arange(0, X)
    offs_2d_y = tl.arange(0, Y * Z)

    offs_3d_x = offs_2d_x[:, None]

    offs_3d_y = offs_2d_y[None, :] // Z
    offs_3d_z = offs_2d_y[None, :] % Z

    stride_x = Y * Z
    stride_y = Z
    stride_z = 1

    offs = offs_3d_x * stride_x + offs_3d_y * stride_y + offs_3d_z * stride_z
    a = tl.load(in0 + offs)
    tl.store(out0 + offs, a)


def test_gather_column(device):
    SIZE = 64
    input = torch.arange(2, SIZE + 2, device=device, dtype=torch.int32)
    input3D = input.reshape(2, 4, 8)
    output = torch.full((SIZE, ), -1, device=device, dtype=torch.int32)

    if device == "cpu":
        triton.runtime.driver.set_active(CPUDriver())

    gather_column[
        1,
    ](input3D, output)
    torch.testing.assert_close(output, input)


@triton.jit
def gather_column_ld_mask(in0, out0, X: tl.constexpr = 2, Y: tl.constexpr = 4, Z: tl.constexpr = 8):
    offs_2d_x = tl.arange(0, X)
    offs_2d_y = tl.arange(0, Y * Z)

    offs_3d_x = offs_2d_x[:, None]

    offs_3d_y = offs_2d_y[None, :] // Z
    offs_3d_z = offs_2d_y[None, :] % Z

    stride_x = Y * Z
    stride_y = Z
    stride_z = 1

    offs = offs_3d_x * stride_x + offs_3d_y * stride_y + offs_3d_z * stride_z
    a = tl.load(in0 + offs, mask=offs_2d_x[:, None] < 1, other=0)
    tl.store(out0 + offs, a)


def test_gather_column_ld_mask(device):
    SIZE = 64
    input = torch.arange(2, SIZE + 2, device=device, dtype=torch.int32)
    input3D = input.reshape(2, 4, 8)
    output = torch.full((SIZE, ), -1, device=device, dtype=torch.int32)

    if device == "cpu":
        triton.runtime.driver.set_active(CPUDriver())

    gather_column_ld_mask[
        1,
    ](input3D, output)

    ref = input3D
    ref[1:, :, :] = 0
    ref = ref.reshape(SIZE, )
    torch.testing.assert_close(output, ref)


@triton.jit
def gather_column_st_mask(in0, out0, X: tl.constexpr = 2, Y: tl.constexpr = 4, Z: tl.constexpr = 8):
    offs_2d_x = tl.arange(0, X)
    offs_2d_y = tl.arange(0, Y * Z)

    offs_3d_x = offs_2d_x[:, None]

    offs_3d_y = offs_2d_y[None, :] // Z
    offs_3d_z = offs_2d_y[None, :] % Z

    stride_x = Y * Z
    stride_y = Z
    stride_z = 1

    offs = offs_3d_x * stride_x + offs_3d_y * stride_y + offs_3d_z * stride_z
    a = tl.load(in0 + offs)
    tl.store(out0 + offs, a, mask=offs_2d_x[:, None] < 1)


def test_gather_column_st_mask(device):
    SIZE = 64
    input = torch.arange(2, SIZE + 2, device=device, dtype=torch.int32)
    input3D = input.reshape(2, 4, 8)
    output = torch.full((SIZE, ), -1, device=device, dtype=torch.int32)

    if device == "cpu":
        triton.runtime.driver.set_active(CPUDriver())
    gather_column_st_mask[
        1,
    ](input3D, output)
    ref = input3D
    ref[1:, :, :] = -1
    ref = ref.reshape(SIZE, )
    torch.testing.assert_close(output, ref)


@triton.jit
def gather_block(
    in0,
    out0,
    X: tl.constexpr = 2,
    Y: tl.constexpr = 4,
    Z: tl.constexpr = 8,
    W: tl.constexpr = 16,
):
    offs_3d_x = tl.arange(0, X)
    offs_3d_y = tl.arange(0, Y * Z)
    offs_3d_z = tl.arange(0, W)

    offs_4d_x = offs_3d_x[:, None, None]

    offs_4d_y = offs_3d_y[None, :, None] // Z

    offs_4d_z = offs_3d_y[None, :, None] % Z

    offs_4d_w = offs_3d_z[None, None, :]

    stride_x = Y * Z * W
    stride_y = Z * W
    stride_z = W
    stride_w = 1

    offs = (offs_4d_x * stride_x + offs_4d_y * stride_y + offs_4d_z * stride_z + offs_4d_w * stride_w)

    a = tl.load(in0 + offs)
    tl.store(out0 + offs, a)


def test_gather_block(device):
    SIZE = 1024
    input = torch.arange(2, SIZE + 2, device=device, dtype=torch.int32)
    input4D = input.reshape(2, 4, 8, 16)
    output = torch.full((SIZE, ), -1, device=device, dtype=torch.int32)

    if device == "cpu":
        triton.runtime.driver.set_active(CPUDriver())

    gather_block[
        1,
    ](input4D, output)
    torch.testing.assert_close(output, input)


@triton.jit
def gather_block_ld_mask(
    in0,
    out0,
    X: tl.constexpr = 2,
    Y: tl.constexpr = 4,
    Z: tl.constexpr = 8,
    W: tl.constexpr = 16,
):
    offs_3d_x = tl.arange(0, X)
    offs_3d_y = tl.arange(0, Y * Z)
    offs_3d_z = tl.arange(0, W)

    offs_4d_x = offs_3d_x[:, None, None]

    offs_4d_y = offs_3d_y[None, :, None] // Z

    offs_4d_z = offs_3d_y[None, :, None] % Z

    offs_4d_w = offs_3d_z[None, None, :]

    stride_x = Y * Z * W
    stride_y = Z * W
    stride_z = W
    stride_w = 1

    offs = (offs_4d_x * stride_x + offs_4d_y * stride_y + offs_4d_z * stride_z + offs_4d_w * stride_w)

    a = tl.load(in0 + offs, mask=offs_4d_x < 1 and offs_4d_w < 8, other=0)
    tl.store(out0 + offs, a)


def test_gather_block_ld_mask(device):
    SIZE = 1024
    input = torch.arange(2, SIZE + 2, device=device, dtype=torch.int32)
    input4D = input.reshape(2, 4, 8, 16)
    output = torch.full((SIZE, ), -1, device=device, dtype=torch.int32)

    if device == "cpu":
        triton.runtime.driver.set_active(CPUDriver())

    gather_block_ld_mask[
        1,
    ](input4D, output)

    ref = input4D
    ref[:, :, :, 8:] = 0
    ref[1:, :, :, :] = 0
    ref = ref.reshape(SIZE, )
    torch.testing.assert_close(output, ref)


@triton.jit
def gather_block_st_mask(
    in0,
    out0,
    X: tl.constexpr = 2,
    Y: tl.constexpr = 4,
    Z: tl.constexpr = 8,
    W: tl.constexpr = 16,
):
    offs_3d_x = tl.arange(0, X)
    offs_3d_y = tl.arange(0, Y * Z)
    offs_3d_z = tl.arange(0, W)

    offs_4d_x = offs_3d_x[:, None, None]

    offs_4d_y = offs_3d_y[None, :, None] // Z

    offs_4d_z = offs_3d_y[None, :, None] % Z

    offs_4d_w = offs_3d_z[None, None, :]

    stride_x = Y * Z * W
    stride_y = Z * W
    stride_z = W
    stride_w = 1

    offs = (offs_4d_x * stride_x + offs_4d_y * stride_y + offs_4d_z * stride_z + offs_4d_w * stride_w)

    a = tl.load(in0 + offs)
    tl.store(out0 + offs, a, mask=offs_4d_x < 1 and offs_4d_w < 8)


def test_gather_block_st_mask(device):
    SIZE = 1024
    input = torch.arange(2, SIZE + 2, device=device, dtype=torch.int32)
    input4D = input.reshape(2, 4, 8, 16)
    output = torch.full((SIZE, ), -1, device=device, dtype=torch.int32)

    if device == "cpu":
        triton.runtime.driver.set_active(CPUDriver())

    gather_block_st_mask[
        1,
    ](input4D, output)

    ref = input4D
    ref[:, :, :, 8:] = -1
    ref[1:, :, :, :] = -1
    ref = ref.reshape(SIZE, )
    torch.testing.assert_close(output, ref)
