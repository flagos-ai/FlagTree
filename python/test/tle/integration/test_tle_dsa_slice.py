# flagtree tle
"""TLE DSA slice family tests on tsingmicro (txda).

Covers `tle.dsa.extract_slice`/`insert_slice` (element-level strided slicing)
and the `tle.dsa.extract_tile`/`insert_tile` grid-coordinate wrappers that
resolve to the same dsa.extract_slice / dsa.insert_slice IR ops.
"""

import pytest
import torch

import triton
import triton.experimental.tle.language as tle
import triton.language as tl


def _is_txda():
    try:
        import torch_txda  # noqa: F401
    except ImportError:
        return False
    target = triton.runtime.driver.active.get_current_target()
    return getattr(target, "backend", None) == "txda"


pytestmark = pytest.mark.skipif(
    not _is_txda(), reason="TLE DSA tests require TsingMicro (txda) backend"
)


@triton.jit
def _idx(shape: tl.constexpr):
    return (tl.arange(0, shape[0])[:, None, None, None] * shape[1] * shape[2] * shape[3]
            + tl.arange(0, shape[1])[None, :, None, None] * shape[2] * shape[3]
            + tl.arange(0, shape[2])[None, None, :, None] * shape[3]
            + tl.arange(0, shape[3])[None, None, None, :])


TILE_SRC = (32, 32, 32, 16)  # tile-test source shape
TILE = (16, 16, 16, 8)       # tile shape -> grid [2, 2, 2, 2]
SLICE = (16, 16, 16, 16)     # slice-test source shape


# -------------------------------- tile kernels -------------------------------

@triton.jit
def extract_scalar(x_ptr, out_ptr, shape: tl.constexpr, tile_shape: tl.constexpr,
                   LIN: tl.constexpr):
    x = tl.load(x_ptr + _idx(shape))
    tile = tle.dsa.extract_tile(x, index=LIN, tile_shape=tile_shape)
    tl.store(out_ptr + _idx(tile_shape), tile)


@triton.jit
def extract_dyn_multi(x_ptr, out_ptr, shape: tl.constexpr, tile_shape: tl.constexpr,
                      I0: tl.constexpr, I1: tl.constexpr, I2: tl.constexpr,
                      I3: tl.constexpr):
    x = tl.load(x_ptr + _idx(shape))
    r = tl.full((), I0, tl.int32)
    c = tl.full((), I1, tl.int32)
    tile = tle.dsa.extract_tile(x, index=[r, c, I2, I3], tile_shape=tile_shape)
    tl.store(out_ptr + _idx(tile_shape), tile)


@triton.jit
def insert_multi(x_ptr, out_ptr, shape: tl.constexpr, tile_shape: tl.constexpr,
                 SI0: tl.constexpr, SI1: tl.constexpr, SI2: tl.constexpr,
                 SI3: tl.constexpr, DI0: tl.constexpr, DI1: tl.constexpr,
                 DI2: tl.constexpr, DI3: tl.constexpr):
    x = tl.load(x_ptr + _idx(shape))
    tile = x.extract_tile(index=[SI0, SI1, SI2, SI3], tile_shape=tile_shape)
    tile = tile + 1.0
    y = x.insert_tile(tile, index=[DI0, DI1, DI2, DI3])
    tl.store(out_ptr + _idx(shape), y)


@triton.jit
def insert_oop(x_ptr, out_ptr, shape: tl.constexpr, tile_shape: tl.constexpr,
               DI0: tl.constexpr, DI1: tl.constexpr, DI2: tl.constexpr,
               DI3: tl.constexpr):
    x = tl.load(x_ptr + _idx(shape))
    tile = tle.dsa.extract_tile(x, index=[0, 0, 0, 0], tile_shape=tile_shape)
    t2 = tile + 1.0
    s = tl.sum(tile)  # forces out-of-place mk.addvs: original tile still needed
    y = tle.dsa.insert_tile(x, t2, index=[DI0, DI1, DI2, DI3])
    y = y + s
    tl.store(out_ptr + _idx(shape), y)


@triton.jit
def insert_dyn_scalar(x_ptr, idx_src_ptr, idx_dst_ptr, out_ptr,
                      shape: tl.constexpr, tile_shape: tl.constexpr):
    x = tl.load(x_ptr + _idx(shape))
    idx_src = tl.load(idx_src_ptr)
    idx_dst = tl.load(idx_dst_ptr)
    tile = tle.dsa.extract_tile(x, index=idx_src, tile_shape=tile_shape)
    tile = tile + 1.0
    y = tle.dsa.insert_tile(x, tile, index=idx_dst)
    tl.store(out_ptr + _idx(shape), y)


# -------------------------------- slice kernels ------------------------------

@triton.jit
def extract_static(x_ptr, out_ptr, shape: tl.constexpr, offsets: tl.constexpr,
                   sizes: tl.constexpr, strides: tl.constexpr):
    x = tl.load(x_ptr + _idx(shape))
    sub = tle.dsa.extract_slice(x, offsets=offsets, sizes=sizes, strides=strides)
    tl.store(out_ptr + _idx(sizes), sub)


@triton.jit
def extract_dyn(x_ptr, o0_ptr, o1_ptr, out_ptr, shape: tl.constexpr,
                sizes: tl.constexpr):
    x = tl.load(x_ptr + _idx(shape))
    o0 = tl.load(o0_ptr)
    o1 = tl.load(o1_ptr)
    sub = tle.dsa.extract_slice(x, offsets=(o0, o1, 0, 0), sizes=sizes,
                                strides=(1, 1, 1, 1))
    tl.store(out_ptr + _idx(sizes), sub)


@triton.jit
def extract_mixed(x_ptr, o0_ptr, out_ptr, shape: tl.constexpr, sizes: tl.constexpr,
                  O1: tl.constexpr):
    x = tl.load(x_ptr + _idx(shape))
    o0 = tl.load(o0_ptr)
    sub = tle.dsa.extract_slice(x, offsets=(o0, O1, 0, 0), sizes=sizes,
                                strides=(1, 1, 1, 1))
    tl.store(out_ptr + _idx(sizes), sub)


@triton.jit
def insert_default(x_ptr, out_ptr, shape: tl.constexpr, offsets: tl.constexpr):
    x = tl.load(x_ptr + _idx(shape))
    tile = tle.dsa.extract_slice(x, offsets=(0, 0, 0, 0), sizes=(4, 4, 4, 4),
                                 strides=(1, 1, 1, 1))
    tile = tile + 1.0
    y = tle.dsa.insert_slice(x, tile, offsets=offsets)
    tl.store(out_ptr + _idx(shape), y)


@triton.jit
def insert_strided(x_ptr, out_ptr, shape: tl.constexpr, offsets: tl.constexpr):
    x = tl.load(x_ptr + _idx(shape))
    tile = tle.dsa.extract_slice(x, offsets=(0, 0, 0, 0), sizes=(4, 4, 4, 4),
                                 strides=(1, 1, 1, 1))
    tile = tile + 1.0
    y = tle.dsa.insert_slice(x, tile, offsets=offsets, sizes=(4, 4, 4, 4),
                             strides=(2, 2, 2, 2))
    tl.store(out_ptr + _idx(shape), y)


@triton.jit
def member_roundtrip(x_ptr, out_ptr, shape: tl.constexpr, offsets: tl.constexpr):
    x = tl.load(x_ptr + _idx(shape))
    sub = x.extract_slice(offsets=(0, 0, 0, 0), sizes=(8, 8, 8, 8),
                          strides=(1, 1, 1, 1))
    sub = sub + 1.0
    y = x.insert_slice(sub, offsets=offsets)
    tl.store(out_ptr + _idx(shape), y)


class TestTile:

    def test_extract_scalar(self):
        torch.manual_seed(42)
        x = torch.randn(*TILE_SRC, device="txda", dtype=torch.float32)

        # LIN=15 == linear id of [1,1,1,1]
        out = torch.zeros(*TILE, device="txda", dtype=torch.float32)
        extract_scalar[(1,)](x, out, TILE_SRC, TILE, 15)
        torch.testing.assert_close(out, x[16:32, 16:32, 16:32, 8:16])

    def test_extract_dyn_multi(self):
        torch.manual_seed(42)
        x = torch.randn(*TILE_SRC, device="txda", dtype=torch.float32)

        # mixed: dims 0,1 dynamic, dims 2,3 static
        out = torch.zeros(*TILE, device="txda", dtype=torch.float32)
        extract_dyn_multi[(1,)](x, out, TILE_SRC, TILE, 1, 1, 0, 0)
        torch.testing.assert_close(out, x[16:32, 16:32, 0:16, 0:8])

    def test_insert_multi(self):
        torch.manual_seed(44)
        shape, tile = (16, 16, 16, 16), (8, 8, 8, 8)
        x = torch.randn(*shape, device="txda", dtype=torch.float32)

        # identity: extract [0,0,0,0], +1.0, insert back at [0,0,0,0]
        out = torch.zeros(*shape, device="txda", dtype=torch.float32)
        insert_multi[(1,)](x, out, shape, tile, 0, 0, 0, 0, 0, 0, 0, 0)
        expected = x.clone()
        expected[0:8, 0:8, 0:8, 0:8] += 1.0
        torch.testing.assert_close(out, expected)

        # relocate: extract [1,1,1,1], +1.0, insert at [0,0,0,0]
        out = torch.zeros(*shape, device="txda", dtype=torch.float32)
        insert_multi[(1,)](x, out, shape, tile, 1, 1, 1, 1, 0, 0, 0, 0)
        expected = x.clone()
        expected[0:8, 0:8, 0:8, 0:8] = x[8:16, 8:16, 8:16, 8:16] + 1.0
        torch.testing.assert_close(out, expected)

    def test_insert_oop(self):
        torch.manual_seed(47)
        shape, tile = (16, 16, 16, 16), (8, 8, 8, 8)
        x2 = torch.randn(*shape, device="txda", dtype=torch.float32)
        out = torch.zeros(*shape, device="txda", dtype=torch.float32)
        insert_oop[(1,)](x2, out, shape, tile, 1, 1, 1, 1)
        src = x2[0:8, 0:8, 0:8, 0:8]
        expected = x2.clone()
        expected[8:16, 8:16, 8:16, 8:16] = src + 1.0
        expected = expected + src.sum()
        torch.testing.assert_close(out, expected)

    def test_insert_dyn_scalar(self):
        torch.manual_seed(50)
        shape, tile = (16, 16, 16, 16), (8, 8, 8, 8)

        # Runtime indices (5=[0,1,0,1], 10=[1,0,1,0]) exercise the div/mod path.
        x = torch.randn(*shape, device="txda", dtype=torch.float32)
        idx_src = torch.tensor(5, device="txda", dtype=torch.int32)
        idx_dst = torch.tensor(10, device="txda", dtype=torch.int32)
        out = torch.zeros(*shape, device="txda", dtype=torch.float32)
        insert_dyn_scalar[(1,)](x, idx_src, idx_dst, out, shape, tile)
        expected = x.clone()
        expected[8:16, 0:8, 8:16, 0:8] = x[0:8, 8:16, 0:8, 8:16] + 1.0
        torch.testing.assert_close(out, expected)


class TestSlice:

    def test_extract_static(self):
        torch.manual_seed(42)
        x = torch.randn(*SLICE, device="txda", dtype=torch.float32)

        # stride 1
        out = torch.zeros(8, 8, 8, 8, device="txda", dtype=torch.float32)
        extract_static[(1,)](x, out, SLICE, (4, 4, 4, 4), (8, 8, 8, 8), (1, 1, 1, 1))
        torch.testing.assert_close(out, x[4:12, 4:12, 4:12, 4:12])

        # stride 2
        out = torch.zeros(8, 8, 8, 8, device="txda", dtype=torch.float32)
        extract_static[(1,)](x, out, SLICE, (0, 0, 0, 0), (8, 8, 8, 8), (2, 2, 2, 2))
        torch.testing.assert_close(out, x[0:16:2, 0:16:2, 0:16:2, 0:16:2])

    def test_extract_dyn(self):
        torch.manual_seed(42)
        x = torch.randn(*SLICE, device="txda", dtype=torch.float32)
        o0 = torch.tensor(4, device="txda", dtype=torch.int32)
        o1 = torch.tensor(4, device="txda", dtype=torch.int32)

        # all-dynamic offsets on dims 0,1
        out = torch.zeros(8, 8, 8, 8, device="txda", dtype=torch.float32)
        extract_dyn[(1,)](x, o0, o1, out, SLICE, (8, 8, 8, 8))
        torch.testing.assert_close(out, x[4:12, 4:12, 0:8, 0:8])

        # mixed: dynamic dim0, static dim1 (=8)
        out = torch.zeros(8, 8, 8, 8, device="txda", dtype=torch.float32)
        extract_mixed[(1,)](x, o0, out, SLICE, (8, 8, 8, 8), 8)
        torch.testing.assert_close(out, x[4:12, 8:16, 0:8, 0:8])

    def test_insert_default(self):
        torch.manual_seed(44)
        x = torch.randn(*SLICE, device="txda", dtype=torch.float32)

        out = torch.zeros(*SLICE, device="txda", dtype=torch.float32)
        insert_default[(1,)](x, out, SLICE, (8, 8, 8, 8))
        expected = x.clone()
        expected[8:12, 8:12, 8:12, 8:12] = x[0:4, 0:4, 0:4, 0:4] + 1.0
        torch.testing.assert_close(out, expected)

    def test_insert_strided(self):
        torch.manual_seed(44)
        x = torch.randn(*SLICE, device="txda", dtype=torch.float32)

        out = torch.zeros(*SLICE, device="txda", dtype=torch.float32)
        insert_strided[(1,)](x, out, SLICE, (4, 4, 4, 4))
        expected = x.clone()
        expected[4:12:2, 4:12:2, 4:12:2, 4:12:2] = x[0:4, 0:4, 0:4, 0:4] + 1.0
        torch.testing.assert_close(out, expected)

    def test_member(self):
        torch.manual_seed(44)
        x = torch.randn(*SLICE, device="txda", dtype=torch.float32)

        out = torch.zeros(*SLICE, device="txda", dtype=torch.float32)
        member_roundtrip[(1,)](x, out, SLICE, (8, 8, 8, 8))
        expected = x.clone()
        expected[8:16, 8:16, 8:16, 8:16] = x[0:8, 0:8, 0:8, 0:8] + 1.0
        torch.testing.assert_close(out, expected)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
