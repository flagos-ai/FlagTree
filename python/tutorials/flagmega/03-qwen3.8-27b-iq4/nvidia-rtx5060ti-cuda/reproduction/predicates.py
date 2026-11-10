"""Exact original acceptance predicates; callers supply current verified bindings.

These functions compile the retained statements. A caller must independently
bind and inspect actual outputs before treating their execution as acceptance.
"""
import ast
import hashlib
import math
from pathlib import Path
import statistics

from .provider import checked, pin, require

HERE = Path(__file__).resolve().parent


def scientific55(origin, reset_count):
    """Keep every original scientific statement; update one reported census."""
    if type(reset_count) is not int or reset_count not in (275, 49):
        raise ValueError('Only the frozen baseline275/candidate49 census is supported')
    tree = ast.parse(checked(origin))
    if len(tree.body) != 55:
        raise ValueError('Original scientific statement count changed')
    matches = []
    for node in tree.body:
        if (isinstance(node, ast.Assert) and isinstance(node.test, ast.Compare)
                and len(node.test.comparators) == 1
                and isinstance(node.test.comparators[0], ast.List)
                and ast.literal_eval(node.test.comparators[0]) == [64, 33, 869, 275]):
            matches.append(node.test.comparators[0])
    if len(matches) != 1:
        raise ValueError('Original reporting census is not unique')
    matches[0].elts[-1] = ast.Constant(reset_count)
    return ast.fix_missing_locations(tree)



def components(reset_count=49):
    roots = HERE / 'methods'
    paths = {'science':'ordinary-science.py', 'resources':'ordinary-resources.py',
             'reference':'reference-full.py', 'aggregation':'aggregation.py',
             'statistics':'statistics.py', 'process_distribution':'process-distribution.py'}
    trees = {'science':scientific55(pin(roots / paths['science']), reset_count)}
    for key, count in (('resources',21), ('reference',53)):
        trees[key] = ast.parse((roots / paths[key]).read_bytes())
        require(len(trees[key].body) == count, 'Original predicate count changed: ' + key)
    tree = ast.parse((roots / paths['aggregation']).read_bytes())
    changes = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and node.value == 't24':
            node.value = 't27'
            changes += 1
    require(changes == 3, 'Original matched candidate label shape changed')
    trees['aggregation'] = ast.fix_missing_locations(tree)
    for key in ('statistics','process_distribution'):
        trees[key] = ast.parse((roots / paths[key]).read_bytes())
    return trees


def aggregate_qualified(*, all_results, raw_samples, all_rounds, warmups, resource_reports):
    trees = components()
    ns = dict(math=math, statistics=statistics, all_results=all_results, raw_samples=raw_samples,
              all_rounds=all_rounds, warmups=warmups, resource_reports=resource_reports,
              decision={'original_matrix': {'samples_per_backend': {'synthetic32':192, 'chinese64':384}}})
    for key in ('statistics','process_distribution','aggregation'):
        exec(compile(trees[key],str(HERE / 'methods' / key),'exec'), ns)
    return dict(scenarios=ns['gaps'], strict_lower_mean_both=ns['strict_win'],
                resource_reports_required=4, performance_accepted=False)
