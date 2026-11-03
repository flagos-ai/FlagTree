"""Check the destructive result lifetime in the real rendered helper."""
import ast
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from local_optimizations.parallel_dsa_pool_sort import HELPER, selection
from triton.flagmega.codegen.triton.templates import TritonTemplateRegistry


def context():
    return dict(call=dict(symbol='dsa', signature='result, sequence, device_dptr',
                          result='result', sequence_lengths='sequence', index_topk=2048,
                          parallel_dsa_sort=True),
                distributed_entry=True, grid_mesh=True, mesh_rank=3,
                mesh_hierarchy=(4, 8, 16), mesh_levels='cbb',
                mesh_axis_names=('device', 'block_y', 'block_x'),
                device_context_abi='runtime-handles/v2')


def test_every_score_read_precedes_barrier_and_destructive_writes():
    registry = TritonTemplateRegistry()
    source = registry.environment.from_string(
        '{% import "entrypoints/_common.py.jinja" as entry with context %}\n'+HELPER
    ).render(context())
    fn = ast.parse(source).body[0]
    barriers = [i for i, n in enumerate(fn.body) if isinstance(n, ast.Expr)
                and isinstance(n.value, ast.Call)
                and ast.unparse(n.value.func) == 'tle.distributed_barrier']
    assert len(barriers) == 1
    before, after = fn.body[:barriers[0]], fn.body[barriers[0]+1:]
    def calls(nodes, name):
        return [n for node in nodes for n in ast.walk(node)
                if isinstance(n, ast.Call) and ast.unparse(n.func) == name]
    assert len(calls(before, 'tl.load')) == 3
    assert not calls(before, 'tl.store')
    assert not calls(after, 'tl.load')
    assert len(calls(after, 'tl.store')) == 2
    rank = next(n for n in before if isinstance(n, ast.Assign)
                and ast.unparse(n.targets[0]) == 'rank')
    assert ast.unparse(rank.value).startswith('tl.sum(')
    stores = calls(after, 'tl.store')
    assert 'rank < 512' in ast.unparse(stores[0])
    assert 'own_ids < Tile' in ast.unparse(stores[0])
    assert ast.unparse(stores[1].args[0]) == 'SortResult + 2048 + tail'


@pytest.mark.parametrize('tile', [512, 528, 1024])
def test_pool_owners_cover_complete_population_once(tile):
    owners = [(pid, pool) for pid in range(128)
              for pool in range(pid*8, pid*8+8) if pool < tile]
    assert sorted(pool for _, pool in owners) == list(range(tile))
    assert len(set(pool for _, pool in owners)) == tile


def test_rank_counts_use_one_warp_per_original_pool_owner():
    # Eight owners must occupy eight distinct warps. The original default
    # [8,4]/[8,1] layout duplicates owners across warps and serializes columns.
    source = TritonTemplateRegistry().environment.from_string(
        '{% import "entrypoints/_common.py.jinja" as entry with context %}\n'+HELPER
    ).render(context())
    fn = ast.parse(source).body[0]
    layouts = [n for n in ast.walk(fn) if isinstance(n, ast.Call)
               and ast.unparse(n.func) == 'tle.gpu.set_layout']
    assert len(layouts) == 2
    for layout in layouts:
        assert ast.unparse(layout.args[1]) == 'tle.gpu.BlockEncoding([1, 1], [1, 32], [8, 1], [1, 0])'
        comparison = layout.args[0].func.value
        assert isinstance(comparison, ast.Compare)
        assert isinstance(comparison.ops[0], ast.Gt)
        assert ast.unparse(comparison.comparators[0]) == 'own_keys[:, None]'
    assert 'own_ids = pid * 8 + tl.arange(0, 8)' in source
    assert 'all_keys = all_order.to(tl.uint64)' in source


def test_selection_uses_original_small_sort_and_uniform_large_dispatch():
    registry = TritonTemplateRegistry()
    source = registry.environment.from_string('def select(result, sequence, device_dptr):\n'+selection()).render(context())
    fn = ast.parse(source).body[0]
    branch = fn.body[-1]
    assert ast.unparse(branch.test) == '_sort_count >= 257'
    large = ast.unparse(branch.body[0])
    assert '_sort_count <= 512' in large
    assert '_sort_count <= 528' in large
    assert all('device_dptr, '+str(tile) in large for tile in (512, 528, 1024))
    assert 'tl.program_id' not in large
    assert 'dsa__select' in ast.unparse(branch.orelse[0])


def test_bounded_tail_ranking_cannot_discard_any_top512_owner():
    # The score phase initializes the whole 1024 population. Beyond the
    # bounded count every key is invalid, with ties resolved by lower ID.
    for count in (257, 258, 511, 512, 513, 514, 528):
        tile = 512 if count <= 512 else 528
        for valid in (0, 1, count-1, count):
            scores = [1 if i < valid else 0 for i in range(1024)]
            complete = sorted(range(1024), key=lambda i: (-scores[i], i))[:512]
            bounded = sorted(range(tile), key=lambda i: (-scores[i], i))[:512]
            assert complete == bounded
