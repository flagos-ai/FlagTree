"""Producer-consumer queue across chiplet dies with notify/wait signaling.

Producer blocks write a slot into a ring buffer and notify the peer die via a
signal; consumer blocks wait on the signal, then copy the slot to output.
"""
import torch
import triton
import triton.experimental.tle.language as tle
import triton.language as tl

N_DIES = 4
NUM_INPUTS = 64
BLOCK_SIZE = 256
QUEUE_SIZE = 4
NUM_PRODUCERS = 2
NUM_CONSUMERS = 2
NUM_BLOCKS = NUM_PRODUCERS + NUM_CONSUMERS
NUM_REPEATS = 5


@triton.jit
def producer_consumer_kernel(
    input_ptr,
    queue_ptr,
    signal_ptr,
    output_ptr,
    mesh: tl.constexpr,
    num_inputs: tl.constexpr,
    queue_size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    NUM_PRODUCERS: tl.constexpr,
    NUM_CONSUMERS: tl.constexpr,
):
    pid = tl.program_id(0)
    pe = tle.shard_id(mesh, "chiplet")
    npes = tle.n_pes(None)
    peer = (pe + 1) % npes

    if pid < NUM_PRODUCERS:
        i = pid
        while i < num_inputs:
            qoff = i % queue_size
            rep = i // queue_size
            remote_signal = tle.remote(signal_ptr, peer, scope=mesh, space="chiplet") + qoff
            tle.chiplet.wait(remote_signal, wait_value=rep * 2)
            tle.chiplet.putmem(queue_ptr + qoff * BLOCK_SIZE, input_ptr + i * BLOCK_SIZE, BLOCK_SIZE, peer)
            tle.chiplet.notify(signal_ptr + qoff, peer, signal=rep * 2 + 1, sig_op="set")
            i += NUM_PRODUCERS
    elif pid < NUM_PRODUCERS + NUM_CONSUMERS:
        i = pid - NUM_PRODUCERS
        while i < num_inputs:
            qoff = i % queue_size
            rep = i // queue_size
            tle.chiplet.wait(signal_ptr + qoff, wait_value=rep * 2 + 1)
            offs = tl.arange(0, BLOCK_SIZE)
            data = tl.load(queue_ptr + qoff * BLOCK_SIZE + offs)
            tl.store(output_ptr + i * BLOCK_SIZE + offs, data)
            tle.chiplet.notify(signal_ptr + qoff, pe, signal=rep * 2 + 2, sig_op="set")
            i += NUM_CONSUMERS


def test_producer_consumer():
    mesh = tle.device_mesh({"chiplet": N_DIES})
    per_die_input = NUM_INPUTS // N_DIES
    assert NUM_INPUTS % N_DIES == 0

    sharded_spec = tle.sharding(mesh, split=(("chiplet", ), ))
    replicated_spec = tle.sharding(mesh)

    for _ in range(NUM_REPEATS):
        queue_dev = tle.make_sharded_tensor(torch.zeros(QUEUE_SIZE * BLOCK_SIZE, dtype=torch.float32), replicated_spec)
        signal_dev = tle.make_sharded_tensor(torch.zeros(QUEUE_SIZE, dtype=torch.int64), replicated_spec)
        input_dev = tle.make_sharded_tensor(torch.randn(NUM_INPUTS * BLOCK_SIZE, dtype=torch.float32), sharded_spec)
        output_dev = tle.make_sharded_tensor(torch.zeros(NUM_INPUTS * BLOCK_SIZE, dtype=torch.float32), sharded_spec)

        producer_consumer_kernel[(NUM_BLOCKS, )](
            input_dev,
            queue_dev,
            signal_dev,
            output_dev,
            mesh,
            num_inputs=per_die_input,
            queue_size=QUEUE_SIZE,
            BLOCK_SIZE=BLOCK_SIZE,
            NUM_PRODUCERS=NUM_PRODUCERS,
            NUM_CONSUMERS=NUM_CONSUMERS,
        )

        output_host = output_dev.cpu()
        input_host = input_dev.cpu()
        passed, failed = 0, 0
        for consumer_pid in range(NUM_CONSUMERS):
            i_local = consumer_pid
            while i_local < per_die_input:
                for pe_dst in range(N_DIES):
                    src_die = (pe_dst - 1 + N_DIES) % N_DIES
                    g_dst = pe_dst * per_die_input + i_local
                    g_src = src_die * per_die_input + i_local
                    expected = input_host[g_src * BLOCK_SIZE:(g_src + 1) * BLOCK_SIZE]
                    actual = output_host[g_dst * BLOCK_SIZE:(g_dst + 1) * BLOCK_SIZE]
                    if torch.allclose(expected, actual):
                        passed += 1
                    else:
                        failed += 1
                        if failed <= 3:
                            print(f"[FAIL] pe_dst={pe_dst} i_local={i_local}: "
                                  f"expected {expected[:4]}, got {actual[:4]}")
                i_local += NUM_CONSUMERS
        assert failed == 0, f"verification failed: {failed}/{passed + failed}"


if __name__ == "__main__":
    test_producer_consumer()
