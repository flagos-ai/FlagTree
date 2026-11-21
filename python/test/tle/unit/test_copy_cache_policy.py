# Copyright 2025-     FlagOS Contributors
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from dataclasses import FrozenInstanceError

import pytest
import triton.language as tl
from triton.experimental.tle.language.gpu import CachePolicy, FractionalEvictionPolicy


@pytest.mark.parametrize("fraction", [0, -1, 1.01, float("nan"), float("inf"), True, "0.5"])
def test_invalid_fraction(fraction):
    with pytest.raises(ValueError):
        FractionalEvictionPolicy("evict_last", fraction)


@pytest.mark.parametrize("fields", [
    {},
    {"cache_modifier": "unknown"},
    {"cache_modifier": "cs", "l1": "evict_first"},
    {"l1": "persist"},
    {"l2_prefetch_size": True},
    {"l2_prefetch_size": 32},
])
def test_invalid_nvidia_policy(fields):
    with pytest.raises(ValueError):
        CachePolicy(**fields)


@pytest.mark.parametrize("policy,descriptor,load", [
    (CachePolicy(cache_modifier="wb"), False, True),
    (CachePolicy(cache_modifier="ca"), False, False),
    (CachePolicy(l1="evict_last"), True, True),
    (CachePolicy(cache_modifier="cg"), True, False),
])
def test_invalid_copy_direction(policy, descriptor, load):
    with pytest.raises(ValueError):
        policy.validate_copy(descriptor, load)


def test_constexpr_identity_and_immutability():
    a = CachePolicy(l2=FractionalEvictionPolicy(tl.constexpr("evict_last"), tl.constexpr(1.0)))
    b = CachePolicy(l2=FractionalEvictionPolicy("evict_last", 1.0))
    c = CachePolicy(l2=FractionalEvictionPolicy("evict_first", 1.0))
    assert a == b and hash(a) == hash(b) and a.type.mangle() == b.type.mangle()
    assert a != c and a.type.mangle() != c.type.mangle()
    with pytest.raises(FrozenInstanceError):
        a.l1 = "evict_last"


@pytest.mark.parametrize("kwargs", [
    {"fraction": 0.5},
    {"secondary": "evict_first"},
])
def test_removed_fractional_controls(kwargs):
    with pytest.raises(ValueError, match="Only fraction=1"):
        FractionalEvictionPolicy("evict_last", **kwargs)


@pytest.mark.parametrize("size", [64, 128, 256])
def test_removed_prefetch(size):
    with pytest.raises(ValueError, match="prefetch is not supported"):
        CachePolicy(l2_prefetch_size=size)


def test_removed_independent_l1_l2_combination():
    with pytest.raises(ValueError, match="cannot be combined"):
        CachePolicy(l1="evict_first", l2=FractionalEvictionPolicy("evict_last"))
