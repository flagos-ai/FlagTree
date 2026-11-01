"""Render the real sorter and exercise its dispatch at pool growth boundaries."""
import ast
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from local_optimizations import joint_dsa_absorbed_pool_scores
from local_optimizations.native_dsa_pool_scores import KEY
from triton.flagmega.codegen.triton.templates import TritonTemplateRegistry


@pytest.fixture(scope="module")
def dispatch():
    registry = TritonTemplateRegistry()
    raw = registry.environment.loader.get_source(registry.environment, KEY)[0]
    module_tree = ast.parse(Path(joint_dsa_absorbed_pool_scores.__file__).read_text())
    tile_assignments = [
        n for n in ast.walk(module_tree)
        if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Subscript)
        and isinstance(n.targets[0].slice, ast.Constant)
        and n.targets[0].slice.value == "joint_dsa_sort_tiles"
    ]
    assert len(tile_assignments) == 1
    tiles = ast.literal_eval(tile_assignments[0].value)
    call = dict(
        family="dsa_indexer", variant="decode", symbol="dsa",
        signature="workspace, weights, state, sequence, layer_arg, advance, key, gate, norm, result, device_dptr",
        state_layer_ids=(3,), layer_id="tl.load(layer_arg)", state="state",
        sequence_lengths="sequence", result="result", state_layer_stride=1052672,
        state_token_stride=257, state_value_stride=1, index_dim=128, index_heads=32,
        q_width=4096, q="workspace + 26240", q_stride=1, weights="weights",
        weights_stride=1, compress_ape="norm", ape_row_stride=128, ape_dim_stride=1,
        key="key", key_stride=1, gate="gate", gate_stride=1,
        k_norm_weight="key", k_norm_bias="gate", norm_stride=1,
        writer_active="True", advance_sequence="advance", epsilon=1e-5,
        scale=0.08838834764831845, state_capacity=4096, state_tile=4096,
        pool=4, pool_tile=1024, index_topk=2048, output_width=2051,
        output_tile=4096, value_type="tl.bfloat16",
        joint_dsa_pool_scores=dict(
            index="tl.program_id(0)", count=64,
            long_count=joint_dsa_absorbed_pool_scores.LONG_SCORE_CTAS,
            long_min_base=joint_dsa_absorbed_pool_scores.LONG_MIN_BASE,
        ),
        joint_dsa_sort_tiles=tiles,
    )
    context = dict(
        render_calls=[call], distributed_entry=False, grid_mesh=False, mesh_rank=3,
        mesh_hierarchy=(1, 1, 1), mesh_levels="cbb",
    )
    source = registry.environment.from_string(
        '{% import "entrypoints/_common.py.jinja" as entry with context %}\n'
        + joint_dsa_absorbed_pool_scores.helpers(raw)
    ).render(context)
    select = next(
        n for n in ast.parse(source).body
        if isinstance(n, ast.FunctionDef) and n.name == "dsa__select"
    )
    branch = next(
        n for n in ast.walk(select)
        if isinstance(n, ast.If) and ast.unparse(n.test) == "pool_count <= 16"
    )
    function = ast.parse(
        "def select_boundary(pool_count, result, first, visible_count, base):\n    pass\n"
    ).body[0]
    function.body = [branch]
    calls = []
    namespace = dict(
        dsa__sort_tile=lambda result, first, visible_count, base, tile: calls.append(("tile", tile)),
        dsa__sort_merge=lambda result, first, visible_count, base: calls.append(("merge", 528)),
    )
    exec(compile(ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])),
                 "<rendered DSA dispatch>", "exec"), namespace)

    def invoke(count):
        calls.clear()
        namespace["select_boundary"](count, object(), 0, count * 4, count * 4 - 1)
        assert len(calls) == 1
        return calls[0]

    return invoke


@pytest.mark.parametrize("count, expected", [
    (0, ("tile", 16)), (16, ("tile", 16)), (17, ("tile", 64)),
    (64, ("tile", 64)), (65, ("tile", 256)), (256, ("tile", 256)),
    (257, ("tile", 512)), (512, ("tile", 512)),
    (513, ("merge", 528)), (514, ("merge", 528)), (528, ("merge", 528)),
    (529, ("tile", 1024)), (1024, ("tile", 1024)),
])
def test_sort_dispatch_preserves_every_pool_growth_boundary(dispatch, count, expected):
    assert dispatch(count) == expected

