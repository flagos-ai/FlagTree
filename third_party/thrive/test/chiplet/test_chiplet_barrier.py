"""Cross-die distributed barrier ordering check.

Each block writes its die's value to all peers' buffers with putmem, then a
chiplet-space distributed_barrier, then reads back all buffers to confirm the
barrier made every peer's write visible to every die.
"""
import torch
import triton
import triton.experimental.tle.language as tle
import triton.language as tl

N_DIES = 4
NB = N_DIES
FILL = 42
BLOCK4 = 4
BLOCK16 = 16


@triton.jit
def barrier_kernel(buf_ptr, src_ptr, out_ptr, mesh: tl.constexpr, block: tl.constexpr):
    pid = tl.program_id(0)
    pe = tle.shard_id(mesh, "chiplet")
    npes = tle.n_pes(None)
    for peer in range(npes):
        tle.chiplet.putmem(buf_ptr + pe, src_ptr, 1, peer)
    tle.distributed_barrier(space="chiplet")
    offs = tl.arange(0, block)
    vals = tl.load(buf_ptr + offs, mask=offs < npes, other=0)
    tl.store(out_ptr + offs, vals, mask=offs < npes)


def test_barrier():
    mesh = tle.device_mesh({"chiplet": N_DIES})
    spec = tle.sharding(mesh)
    buf_dev = tle.make_sharded_tensor(torch.zeros(N_DIES, dtype=torch.int64), spec)
    src_dev = tle.make_sharded_tensor(torch.full((1, ), FILL, dtype=torch.int64), spec)
    out_dev = tle.make_sharded_tensor(torch.zeros(N_DIES, dtype=torch.int64), spec)

    barrier_kernel[(NB, )](buf_dev, src_dev, out_dev, mesh, block=BLOCK4)

    out = out_dev.cpu()
    assert (out == FILL).all(), \
        f"barrier did not order cross-die writes (want all {FILL}): {out}"


if __name__ == "__main__":
    test_barrier()
