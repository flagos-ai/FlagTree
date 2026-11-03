"""Exercise the actual rendered FP32 head tree and complete pool ownership."""
import ast
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from repro_dsa_pool_score_batch import kernel_source


@pytest.mark.parametrize("ctas", [64, 96])
def test_eight_pool_groups_cover_all1024_slots_once(ctas):
    owners = [pool for pid in range(ctas)
              for start in range(pid * 8, 1024, ctas * 8)
              for pool in range(start, min(start + 8, 1024))]
    assert sorted(owners) == list(range(1024))
    assert len(owners) == len(set(owners))


def test_rendered_head_tree_retains_original_rounding_order():
    tree = ast.parse(kernel_source(96))
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                    and n.name == 'repro__pool_scores__batched')
    names = ('fold16', 'fold1', 'fold8', 'fold4', 'score')
    expressions = {n.targets[0].id: n for n in ast.walk(function)
                   if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)
                   and n.targets[0].id in names}
    snippet = ast.Module(body=[expressions[name] for name in names], type_ignores=[])
    values = np.random.default_rng(113732).standard_normal((8, 32)).astype(np.float32)
    values *= np.exp2(np.arange(32) % 24).astype(np.float32)
    values[:, ::7] *= -1
    # Original eight-warp layout: register offset 16, lane offset 1,
    # then shared warp-reduction offsets 8, 4, 2.
    lanes = values[:, :16] + values[:, 16:]
    warps = lanes[:, ::2] + lanes[:, 1::2]
    warps = warps[:, :4] + warps[:, 4:]
    warps = warps[:, :2] + warps[:, 2:]
    expected = warps[:, 0] + warps[:, 1]

    class ArrayLanguage:
        @staticmethod
        def sum(value, axis):
            return np.sum(value, axis=axis, dtype=np.float32)

    namespace = dict(tl=ArrayLanguage, B=8, hc0=values[:, :16], hc16=values[:, 16:])
    exec(compile(ast.fix_missing_locations(snippet), '<actual rendered head tree>', 'exec'), namespace)
    np.testing.assert_array_equal(namespace['score'].view(np.uint32), expected.view(np.uint32))
    assert not np.array_equal(np.sum(values, axis=1).view(np.uint32), expected.view(np.uint32))
