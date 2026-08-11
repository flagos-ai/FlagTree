# Copyright 2018-2020 Philippe Tillet
# Copyright 2020-2022 OpenAI
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
"""Core public interfaces used by FlagTune config proposal."""

from __future__ import annotations

from dataclasses import dataclass, field as dataclass_field
from itertools import product
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

BenchmarkFn = Callable[[Dict[str, Any], Optional[int]], List[float]]
"""Benchmark callable accepted by a :data:`ConfigProposer`.

The first argument is a complete tunable-parameter dictionary and the optional
second argument is a requested run count.  The result is a list of latency
samples whose first element is currently used for ranking.  Implementations may
ignore the run count, as the FlagGems adapter does.
"""

ConfigProposer = Callable[
    [
        Optional[BenchmarkFn],
        Dict[str, Any],
        List[Dict[str, Any]],
        Dict[str, Any],
    ],
    List[Dict[str, Any]],
]
"""Candidate proposer callable contract.

Arguments are ``benchmark_fn``, runtime shape mapping, initial configuration
dictionaries, and caller metadata.  ``benchmark_fn=None`` requests prediction
without online benchmarking.  The return value is an ordered list of complete
parameter dictionaries.  The current FlagTune implementation accepts initial
configs and metadata for integration compatibility but derives candidates from
the registered variant's parameter space.
"""


@dataclass
class ParameterField:
    """Define one tunable parameter and its explicit legal values.

    Values retain YAML order and may be any hashable/comparable Python values
    accepted by the downstream kernel.  Registration requires a non-empty
    value list but direct construction of this low-level class does not enforce
    that invariant.
    """

    name: str
    legal_values: List[Any]


@dataclass
class ParameterSpace:
    """Represent a finite Cartesian product of tunable parameters.

    Attributes:
        fields: Ordered parameter definitions.  Order controls product
            enumeration and public name lists.
        constraints: Optional predicates applied to complete configuration
            dictionaries.  Config-driven registration currently supplies no
            parameter constraints, but the core type supports them.

    Notes:
        Product size is the multiplication of all field cardinalities.  The
        iterator is lazy, but consumers such as model prediction may materialize
        the whole product and should avoid unbounded spaces.
    """

    fields: List[ParameterField]
    constraints: List[Callable[[Dict[str, Any]], bool]] = dataclass_field(default_factory=list)

    def field_values(self) -> Dict[str, List[Any]]:
        """Return an ordered name-to-values mapping for all fields.

        The lists are the original mutable lists, not defensive copies.  Treat
        the result as read-only to avoid changing an active search space.
        """
        return {field.name: field.legal_values for field in self.fields}

    def iter_configs(self) -> Iterable[Dict[str, Any]]:
        """Lazily yield legal configurations in Cartesian-product order.

        Constraints are evaluated only after a full combination is assembled;
        exceptions raised by constraint callables propagate to the caller.
        """
        names = [field.name for field in self.fields]
        for values in product(*(field.legal_values for field in self.fields)):
            config = dict(zip(names, values))
            if all(constraint(config) for constraint in self.constraints):
                yield config

    def validate(self, config: Dict[str, Any]) -> bool:
        """Return whether all fields are present, legal, and constraint-valid.

        Extra keys are ignored.  Constraint exceptions are not converted to
        ``False`` because they normally indicate an invalid constraint
        implementation rather than an invalid candidate.
        """
        for field in self.fields:
            if field.name not in config or config[field.name] not in field.legal_values:
                return False
        return all(constraint(config) for constraint in self.constraints)

    def config_key(self, config: Dict[str, Any]) -> Tuple[Tuple[str, Any], ...]:
        """Build a stable, name-sorted key from fields present in ``config``.

        Missing fields are omitted rather than rejected.  Call
        :meth:`validate` first when a key must represent a complete candidate.
        Values must be hashable if the key will be inserted into a set or dict.
        """
        return tuple(sorted((field.name, config[field.name]) for field in self.fields if field.name in config))

    def active_field_names(self) -> Tuple[str, ...]:
        """Return all field names as an immutable declaration-ordered tuple."""
        return tuple(field.name for field in self.fields)

    @property
    def all_field_names(self) -> List[str]:
        """Return all field names as a new declaration-ordered list."""
        return [field.name for field in self.fields]
