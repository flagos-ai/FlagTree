"""Apply pipelining to Q_B while retaining every ordered K16 MMA update."""
from pathlib import Path
from types import SimpleNamespace

from jinja2 import Environment
from local_optimizations.qb_n32 import KEY, transform


def original_source():
    return (Path(__file__).resolve().parents[6] / 'python/triton/flagmega/codegen/triton' / KEY).read_text()


def test_actual_qb_loop_uses_three_stages_and_two_updates():
    source = transform(original_source())
    line = next(line for line in source.splitlines() if 'for dense_partition_k in tl.range' in line)
    env = Environment()
    selected = SimpleNamespace(partition_k=1536, block_k=128, qb_n32_original_load_warps_n=4)
    rendered = env.from_string(line).render(call=selected)
    assert rendered.strip() == 'for dense_partition_k in tl.range(0, 1536, 128, num_stages=3, loop_unroll_factor=2):'
    ordinary = SimpleNamespace(partition_k=1536, block_k=128)
    assert env.from_string(line).render(call=ordinary).strip() == 'for dense_partition_k in tl.range(0, 1536, 128):'


def test_pipeline_keeps_all_original_load_dot_add_and_store_expressions():
    before = original_source()
    after = transform(before)
    start = before.index('                    dense_local_k_offsets =')
    assert after[after.index('                    dense_local_k_offsets ='):] == before[start:]
