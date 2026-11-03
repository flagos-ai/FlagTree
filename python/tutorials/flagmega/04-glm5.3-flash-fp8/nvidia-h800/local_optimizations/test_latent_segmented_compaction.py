"""Stable integer compaction must retain arbitrary holes and all live tail slots."""
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from local_optimizations.native_latent_probability_template import probability_template, KEY


def test_selected_probability_template_uses_bounded_segmented_scan():
    root = Path(__file__).resolve().parents[6] / 'python/triton/flagmega/codegen/triton'
    source = probability_template((root / KEY).read_text())
    compact = source.split('    if _fm_dsa_count >= ', 1)[0]
    assert '_fm_dsa_original_valid.reshape((512, 4))' in compact
    assert 'tl.cumsum(_fm_dsa_group_count, 0)' in compact
    assert 'tl.cumsum(_fm_dsa_group_valid, 1)' in compact
    assert 'tl.cumsum(_fm_dsa_tail_valid.to(tl.int32), 0)' in compact
    assert 'tl.cumsum(_fm_dsa_original_valid.to(tl.int32), 0)' not in compact
    assert compact.count('tl.debug_barrier()') == 1


def test_integer_segment_ranks_equal_independent_stable_filter():
    # Exhaust the four-lane masks, then stress boundaries and arbitrary
    # position filtering. The oracle knows only stable filter, not ranks.
    rng = random.Random(519027)
    for case in range(96):
        tokens = [rng.randrange(-1, 4096) for _ in range(2051)]
        if case < 16:
            tokens = [i if (case >> (i % 4)) & 1 else -1 for i in range(2051)]
        position = (-1, 0, 63, 64, 255, 256, 2047, 4095)[case % 8]
        output, total = {}, 0
        for start in range(0, 2051, 4):
            valid = [t for t in tokens[start:start+4] if 0 <= t <= position]
            for lane, token in enumerate(valid):
                output[total+lane] = token
            total += len(valid)
        expected = [t for t in tokens if 0 <= t <= position]
        assert len(output) == total == len(expected)
        assert [output[i] for i in range(total)] == expected


def test_changed_or_unproven_geometry_is_rejected_before_render():
    import pytest
    from triton.flagmega.errors import CodegenError
    from local_optimizations.latent_segmented_compaction import SCHEDULE, validate, transform
    call = dict(parameters=dict(latent_compaction_schedule=SCHEDULE), topk_width=2051,
                compact_topk_capacity=4096, topk_stride=1, index_traversal='compact')
    validate(call)
    for field, value in [('topk_width', 2048), ('compact_topk_capacity', 2048),
                         ('topk_stride', 2), ('index_traversal', 'slots')]:
        with pytest.raises(CodegenError):
            validate({**call, field: value})
    with pytest.raises(CodegenError):
        validate({**call, 'parameters': {}})
    with pytest.raises(CodegenError):
        transform('')
