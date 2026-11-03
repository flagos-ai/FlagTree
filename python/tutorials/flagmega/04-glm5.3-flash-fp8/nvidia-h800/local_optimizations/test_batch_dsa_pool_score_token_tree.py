"""The actual four-token FP32 tree must survive changing warp ownership."""
import ast
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from repro_dsa_pool_score_batch import kernel_source


def test_rendered_token_denominator_pairs_are_adjacent_first():
    function = next(n for n in ast.parse(kernel_source(96)).body
                    if isinstance(n, ast.FunctionDef) and n.name.endswith("__batched"))
    names = ("denominator_pairs", "denominator")
    expressions = {n.targets[0].id: n for n in ast.walk(function)
                   if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)
                   and n.targets[0].id in names}
    assert set(expressions) == set(names), "The token tree must be explicit for register-owned tokens"
    values = np.random.default_rng(113737).random((8, 4, 128)).astype(np.float32)
    values *= np.array([1, 1e-5, 1e-7, 1e-8], np.float32)[None, :, None]
    expected = (values[:, 0] + values[:, 1]) + (values[:, 2] + values[:, 3])
    wrong = (values[:, 0] + values[:, 2]) + (values[:, 1] + values[:, 3])
    assert not np.array_equal(wrong.view(np.uint32), expected.view(np.uint32))
    class ArrayLanguage:
        @staticmethod
        def sum(value, axis): return np.sum(value, axis=axis, dtype=np.float32)
    namespace = dict(tl=ArrayLanguage, B=8, exponentials=values)
    snippet = ast.Module(body=[expressions[n] for n in names], type_ignores=[])
    exec(compile(ast.fix_missing_locations(snippet), "<actual token tree>", "exec"), namespace)
    np.testing.assert_array_equal(namespace["denominator"].view(np.uint32), expected.view(np.uint32))
