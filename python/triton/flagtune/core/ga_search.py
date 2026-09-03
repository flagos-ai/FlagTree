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
"""Generic genetic search within one config-defined parameter space."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from triton.flagtune.core.interfaces import ParameterSpace


@dataclass
class GAParams:
    """Control bounded genetic candidate generation.

    Attributes:
        generations: Number of generations to attempt.
        population_size: Maximum number of ranked known entries used as the
            parent pool.
        elite_size: Reserved compatibility setting.  The current generic
            search ranks all known entries and does not separately truncate to
            this value.
        offspring_per_generation: Target unique children attempted per
            generation and default maximum number returned.
        mutation_rate: Independent probability of replacing each field after
            crossover or random initialization.
        random_rate: Probability of creating a child without crossover.
        max_evaluations: Optional cap used to reduce the returned batch after
            accounting for the initial population; zero means no extra cap.

    Notes:
        Direct construction does not clamp probabilities or reject negative
        counts.  Non-positive generation/offspring settings simply produce no
        candidates in :meth:`GASearcher.generate`.
    """

    generations: int
    population_size: int
    elite_size: int
    offspring_per_generation: int
    mutation_rate: float
    random_rate: float
    max_evaluations: int = 0


def _parse_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(float(str(value).strip()))
    except (ValueError, TypeError):
        return None


class GASearcher:
    """Generate unseen candidates inside one :class:`ParameterSpace`.

    The search consumes benchmark-history entries, selects parents by measured
    latency and prediction rank, then applies field-wise crossover, mutation,
    and random sampling.  It generates configurations only; the caller is
    responsible for benchmarking children and feeding useful scores into any
    later search.

    Args:
        param_space: Finite legal parameter space used for validation and
            random sampling.
        ga_params: Generation and sampling controls.
        seed: Private pseudo-random seed.  Identical inputs and seed make
            candidate generation deterministic.

    Notes:
        History entries may contain parameters directly or under ``config``;
        legacy ``META`` nesting is also recognized.  Tunable values are parsed
        through integer conversion, so this searcher is currently suitable for
        integer-valued kernel parameters rather than arbitrary categorical
        spaces.
    """

    def __init__(self, param_space: ParameterSpace, ga_params: GAParams, seed: int = 42) -> None:
        """Initialize an isolated deterministic genetic search state."""
        self.param_space = param_space
        self.ga_params = ga_params
        self._rng = random.Random(seed)

    def generate(self, entries: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate a bounded batch of unique candidate history entries.

        Args:
            entries: Seed history.  Each entry contains a configuration either
                directly or under ``config`` and may include ``latency_ms``,
                ``ga_latency_ms``, and ``candidate_rank`` metadata.

        Returns:
            Newly generated entries carrying ``config``, ``ga_generation``, and
            ``ga_source``.  The list is empty when seeds are absent/invalid,
            generation is disabled, the evaluation cap is exhausted, or the
            finite parameter space has no unseen candidate.

        Notes:
            At most 50 attempts per requested child (with a floor of 100 per
            generation) are made, preventing a nearly exhausted search space
            from looping indefinitely.  ``generate`` does not benchmark or
            score returned candidates.
        """
        if not entries or self.ga_params.generations <= 0 or self.ga_params.offspring_per_generation <= 0:
            return []
        population = self._initial_population(entries)
        if not population:
            return []

        generated_limit = self.ga_params.offspring_per_generation
        if self.ga_params.max_evaluations > 0:
            generated_limit = min(generated_limit, max(0, self.ga_params.max_evaluations - len(population)))
        if generated_limit <= 0:
            return []

        known_entries = list(population)
        known_keys = {self._entry_key(entry) for entry in known_entries}
        generated_history: List[Dict[str, Any]] = []
        base_entry = entries[0]
        for generation in range(1, self.ga_params.generations + 1):
            offspring = self._next_generation(
                base_entry,
                known_entries,
                known_keys,
                generation,
                self.ga_params.offspring_per_generation,
            )
            for entry in offspring:
                known_keys.add(self._entry_key(entry))
            known_entries.extend(offspring)
            generated_history.extend(offspring)
        return generated_history[-generated_limit:]

    def crossover(self, parent_a: Dict[str, Any], parent_b: Dict[str, Any]) -> Dict[str, Any]:
        """Choose every active field independently from one of two parents.

        Both parents must flatten to complete configurations.  Missing fields
        raise ``KeyError`` instead of being synthesized.
        """
        flat_a = self._flatten_config(parent_a.get("config", parent_a))
        flat_b = self._flatten_config(parent_b.get("config", parent_b))
        return {
            field: (flat_a if self._rng.random() < 0.5 else flat_b)[field]
            for field in self.param_space.active_field_names()
        }

    def mutate(self, flat: Dict[str, Any]) -> Dict[str, Any]:
        """Return a copied candidate with probabilistic legal-value mutations.

        Any missing field is always populated.  A mutation may randomly select
        the same value, so mutation does not guarantee a different key.
        """
        result = dict(flat)
        for field, choices in self.param_space.field_values().items():
            if field not in result or self._rng.random() < self.ga_params.mutation_rate:
                result[field] = self._rng.choice(choices)
        return result

    def _flatten_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        meta = config.get("META", {}) if isinstance(config.get("META"), dict) else {}
        result: Dict[str, Any] = {}
        for field in self.param_space.all_field_names:
            value = config.get(field, meta.get(field))
            parsed = _parse_int(value)
            if parsed is not None:
                result[field] = parsed
        return result

    def _entry_key(self, entry: Dict[str, Any]) -> Tuple[Tuple[str, Any], ...]:
        return self.param_space.config_key(self._flatten_config(entry.get("config", entry)))

    def _initial_population(self, entries: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        population: List[Dict[str, Any]] = []
        seen: Set[Tuple[Tuple[str, Any], ...]] = set()
        ordered = sorted(
            entries,
            key=lambda item: (
                _parse_int(item.get("candidate_rank")) is None,
                _parse_int(item.get("candidate_rank")) or 0,
            ),
        )
        for entry in ordered:
            flat = self._flatten_config(entry.get("config", entry))
            if not self.param_space.validate(flat):
                continue
            cloned = self._clone_entry(entry, flat, 0, "topk", _parse_int(entry.get("candidate_rank")))
            key = self._entry_key(cloned)
            if key not in seen:
                seen.add(key)
                population.append(cloned)
        return population

    @staticmethod
    def _clone_entry(
        base: Dict[str, Any],
        config: Dict[str, Any],
        generation: int,
        source: str,
        candidate_rank: Optional[int] = None,
    ) -> Dict[str, Any]:
        entry = dict(base)
        entry["config"] = dict(config)
        entry["ga_generation"] = generation
        entry["ga_source"] = source
        if candidate_rank is not None:
            entry["candidate_rank"] = candidate_rank
        elif generation > 0:
            entry.pop("candidate_rank", None)
        return entry

    def _random_config(self) -> Dict[str, Any]:
        return {field: self._rng.choice(choices) for field, choices in self.param_space.field_values().items()}

    def _parent_pool(self, known_entries: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        parents: List[Dict[str, Any]] = []
        seen = set()
        ranked = sorted(
            known_entries,
            key=lambda item: (
                item.get("ga_latency_ms") is None,
                float(item.get("ga_latency_ms") or 0.0),
                _parse_int(item.get("candidate_rank")) is None,
                _parse_int(item.get("candidate_rank")) or 0,
            ),
        )
        for entry in ranked:
            key = self._entry_key(entry)
            if key not in seen:
                seen.add(key)
                parents.append(entry)
            if len(parents) >= max(1, self.ga_params.population_size):
                break
        return parents or list(known_entries[:1])

    def _next_generation(
        self,
        base_entry: Dict[str, Any],
        known_entries: Sequence[Dict[str, Any]],
        known_keys: Set[Tuple[Tuple[str, Any], ...]],
        generation: int,
        target_count: int,
    ) -> List[Dict[str, Any]]:
        parents = self._parent_pool(known_entries)
        offspring: List[Dict[str, Any]] = []
        batch_keys = set()
        attempts = 0
        while len(offspring) < target_count and attempts < max(100, target_count * 50):
            attempts += 1
            if self._rng.random() < self.ga_params.random_rate or len(parents) == 1:
                child = self._random_config()
                source = "random"
            else:
                child = self.crossover(*self._rng.sample(parents, 2))
                source = "crossover"
            child = self.mutate(child)
            if not self.param_space.validate(child):
                continue
            key = self.param_space.config_key(child)
            if key in known_keys or key in batch_keys:
                continue
            batch_keys.add(key)
            offspring.append(self._clone_entry(base_entry, child, generation, source))
        return offspring
