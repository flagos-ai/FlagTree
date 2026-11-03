"""Keep all Native FP8 K128 updates while grouping the launch pipeline."""
from pathlib import Path
from local_optimizations.stage5 import KEY, transform


def raw_source():
    return (Path(__file__).resolve().parents[6] / 'python/triton/flagmega/codegen/triton' / KEY).read_text()


def test_actual_complete_k_template_groups_two_updates_with_four_pipeline_stages():
    changed = transform(raw_source())
    assert changed.count('num_stages=4, loop_unroll_factor=2') == 1
    assert 'tl.zeros(({{ call.matrix_rows }}, 8), tl.float32)' in changed
    assert '_fm_column = tl.arange(0, 8)' in changed


def test_all_original_k128_update_scale_and_store_expressions_remain():
    original = raw_source()
    baseline = original.replace('num_stages={{ call.num_stages }}', 'num_stages=8')
    baseline = baseline.replace('tl.zeros(({{ call.matrix_rows }}, 16), tl.float32)',
                                'tl.zeros(({{ call.matrix_rows }}, 8), tl.float32)')
    baseline = baseline.replace('_fm_column = tl.arange(0, 16)', '_fm_column = tl.arange(0, 8)')
    changed = transform(original)
    assert changed.replace('num_stages=4, loop_unroll_factor=2', 'num_stages=8') == baseline
