import math

import pytest

import triton
import triton.language as tl
import triton.experimental.tle.language as tle


@triton.jit(noinline=True)
def _group_exchange(values, output, mesh: tl.constexpr, WIDTH: tl.constexpr, STRIDE: tl.constexpr,
                    DOMAIN: tl.constexpr):
    pid = tl.program_id(0)
    coord = pid // STRIDE % DOMAIN
    first = pid - coord % WIDTH * STRIDE
    for epoch in range(16):
        tl.store(values + pid, pid + epoch)
        tle.distributed_barrier(mesh)
        total = tl.full((), 0, tl.int32)
        for member in tl.static_range(WIDTH):
            total += tl.load(values + first + member * STRIDE, volatile=True)
        tl.store(output + epoch * tl.num_programs(0) + pid, total)
        tle.distributed_barrier(mesh)


@triton.jit
def _group_exchange_kernel(values, output, mesh: tl.constexpr, WIDTH: tl.constexpr, STRIDE: tl.constexpr,
                           DOMAIN: tl.constexpr):
    _group_exchange(values, output, mesh, WIDTH, STRIDE, DOMAIN)
    _group_exchange(values, output, mesh, WIDTH, STRIDE, DOMAIN)


@pytest.mark.skip_if_tle_unsupported("distributed_barrier")
@pytest.mark.parametrize("shape,axis,width", [((2, 4), 0, 2), ((2, 4), 1, 2), ((2, 1, 2), 0, 2), ((2, 1, 2), 2, 2)])
def test_grid_axis_groups_through_helper_calls(shape, axis, width, with_allocator):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available() or torch.version.cuda is None:
        pytest.skip("requires CUDA grid barrier lowering")
    names = tuple(f"block_{i}" for i in range(len(shape)))
    mesh = tle.device_mesh({"block": list(zip(names, shape))})
    group = mesh.axis_group(names[axis], group_shape=(width, ))
    count = math.prod(shape)
    stride = math.prod(shape[axis + 1:])
    values = torch.empty((count, ), device="cuda", dtype=torch.int32)
    output = torch.empty((16, count), device="cuda", dtype=torch.int32)
    for _ in range(3):
        compiled = _group_exchange_kernel[(count, )](values, output, group, width, stride, shape[axis], num_warps=4)
        expected = [[
            sum(pid - (pid // stride % shape[axis]) % width * stride + member * stride + epoch
                for member in range(width))
            for pid in range(count)
        ]
                    for epoch in range(16)]
        torch.testing.assert_close(output.cpu(), torch.tensor(expected, dtype=torch.int32), rtol=0, atol=0)
    assert compiled.metadata.launch_cooperative_grid
    assert 'group_kind = "grid_axis"' in compiled.asm["ttgir"]
    assert "atom.add.release.gpu.u32" in compiled.asm["ptx"]
