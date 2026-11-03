"""Pair full K64 iterations while preserving every original tail recurrence."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from local_optimizations.native_latent_probability_template import probability_template, KEY


def test_selected_long_consumer_has_complete_pair_and_ordered_tail():
    root = Path(__file__).resolve().parents[6] / 'python/triton/flagmega/codegen/triton'
    source = probability_template((root / KEY).read_text())
    assert '_fm_dsa_bulk_end = _fm_dsa_count // 128 * 128' in source
    assert 'tl.range(0, _fm_dsa_bulk_end, 64, num_stages=5, loop_unroll_factor=2)' in source
    assert 'tl.range(_fm_dsa_bulk_end, _fm_dsa_count, 64, num_stages=1)' in source
    assert 'if _fm_dsa_count >= 1024:' in source


def test_pairs_and_tail_cover_original_K64_iterations_once_in_order():
    for count in range(0, 2052):
        end = count // 128 * 128
        assert list(range(0, end, 64))+list(range(end, count, 64)) == list(range(0, count, 64))


def test_incomplete_paired_schedule_is_rejected():
    import pytest
    from triton.flagmega.errors import CodegenError
    from local_optimizations.paired_pv_loop import SCHEDULE, MIN_COUNT, validate, transform
    p = dict(latent_pv_loop_schedule=SCHEDULE, latent_pv_pair_min_count=MIN_COUNT,
             latent_long_column_tile=32)
    validate(dict(parameters=p))
    for field, value in [('latent_pv_pair_min_count', 256), ('latent_long_column_tile', 64),
                         ('latent_pv_loop_schedule', 'unproven')]:
        with pytest.raises(CodegenError):
            validate(dict(parameters={**p, field: value}))
    with pytest.raises(CodegenError):
        transform('')
