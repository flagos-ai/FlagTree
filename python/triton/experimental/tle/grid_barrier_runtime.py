# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Host admission for the explicit flags32 TLE grid-barrier protocol."""
import json
import operator

_KEY = 'tle_flags_launch_contract'
_LIMIT = (1 << 31) - 1


def _dimensions(values, *, rank=None):
    if not isinstance(values, (tuple, list)) or not values or (rank is not None and len(values) != rank):
        raise ValueError('flags32 launch domain/grid has invalid rank')
    result = []
    product = 1
    for value in values:
        if isinstance(value, bool):
            raise ValueError('flags32 launch dimensions must be positive integers')
        try:
            value = operator.index(value)
        except TypeError as error:
            raise ValueError('flags32 launch dimensions must be static integers') from error
        if value <= 0 or value > _LIMIT // product:
            raise ValueError('flags32 launch domain/grid exceeds positive signed i32')
        result.append(value)
        product *= value
    return tuple(result), product


def _parse(value):
    try:
        contract = json.loads(value)
    except (TypeError, ValueError) as error:
        raise ValueError('invalid flags32 launch contract') from error
    if not isinstance(contract, dict) or set(contract) != {'schema', 'domain', 'grid'
                                                           } or contract['schema'] != 'tle.flags32/v1':
        raise ValueError('invalid flags32 launch contract schema')
    domain, population = _dimensions(contract['domain'])
    grid, launched = _dimensions(contract['grid'], rank=3)
    if population != launched:
        raise ValueError('flags32 launch grid and logical domain populations differ')
    return domain, grid


def _contract(metadata):
    value = getattr(metadata, _KEY, None)
    if value is None:
        return None
    _, grid = _parse(value)
    if (getattr(metadata, 'num_ctas', None) != 1 or tuple(getattr(metadata, 'cluster_dims', (1, 1, 1))) != (1, 1, 1)
            or getattr(metadata, 'launch_cooperative_grid', None) is not True):
        raise ValueError('flags32 requires a single-CTA cluster and cooperative grid launch')
    return grid


def validate_launch_contract(metadata, grid):
    expected = _contract(metadata)
    if expected is not None:
        actual, _ = _dimensions(grid, rank=3)
        if actual != expected:
            raise ValueError(f'flags32 launch grid {actual} does not match compiled grid {expected}')
