# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Opt-in fixed-slot observations using the disjoint hidden profile buffer."""
import triton.language as tl
from triton.language.core import builtin


def _constant(value, name, low, high):
    value = tl.core._unwrap_if_constexpr(value)
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f'census {name} must be a constant integer in [{low}, {high}]')
    return value


def census_occurrence(value, semantic):
    value = semantic.to_tensor(value)
    if value.type.is_block() or value.dtype != tl.int32:
        raise ValueError('census occurrence must be a scalar signed i32')
    return value.handle


@builtin
def census_configure(occurrences, ctas=128, _semantic=None):
    """Select census/v1; no automatic warp-runtime records are enabled.

    The normal launch grid must contain exactly ``ctas`` CTAs. Occurrence IDs
    are unique within one launch. The host supplies a separately allocated
    profile buffer with a 64-byte header and enables it at header offset zero.
    """
    occurrences = _constant(occurrences, 'occurrences', 1, 65536)
    ctas = _constant(ctas, 'ctas', 1, 65536)
    if not hasattr(_semantic.builder, 'configure_tle_census'):
        raise ValueError('census requires a rebuilt NVIDIA TLE compiler')
    _semantic.builder.configure_tle_census(occurrences, ctas)


@builtin
def census_mark(occurrence, point, _semantic=None):
    """Record one globaltimer observation by actual role-local thread zero.

    A mark is an observation at this program point, not a synchronization or a
    hardware-active interval. Out-of-range runtime occurrences leave missing
    slots, which the mandatory host completeness check must reject.
    """
    point = _constant(point, 'point', 0, 3)
    occurrence = census_occurrence(occurrence, _semantic)
    if not hasattr(_semantic.builder, 'create_tle_census_mark'):
        raise ValueError('census requires a rebuilt NVIDIA TLE compiler')
    _semantic.builder.create_tle_census_mark(occurrence, point, False)


@builtin
def census_mapping(role, point, _semantic=None):
    """Observe entry/exit SM IDs; role0 is consumer and role1 is producer."""
    role = _constant(role, 'role', 0, 1)
    point = _constant(point, 'mapping point', 0, 1)
    if not hasattr(_semantic.builder, 'create_tle_census_mark'):
        raise ValueError('census requires a rebuilt NVIDIA TLE compiler')
    _semantic.builder.create_tle_census_mark(_semantic.to_tensor(role).handle, point, True)
