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
"""Turn an exact FlagTune model bundle into ranked kernel configurations.

This is the runtime bridge used by FlagTune integration layers.  It loads the
identity-matched model bundle, normalizes a runtime shape with its compiled
variant, enumerates legal parameter configurations, builds the ordered feature
matrix, and ranks candidates with the bundled XGBoost model.  If a benchmark
callback is supplied, it times the predicted seeds, asks the GA searcher for
additional candidates, then returns the unique lowest-latency Top-K results.

Environment variables:
  * ``FLAGTUNE_TOP_K``: integer number of returned candidates; defaults
    to ``10`` and is read once per process. Invalid integers raise on first use.
  * ``FLAGTUNE_DISABLE_OPS``: comma-separated ``*``, ``op_id``,
    ``op_id/variant``, or full artifact-key selectors. A matching proposer
    returns no candidates so the caller can apply its normal fallback.

``_MODEL_MANAGER`` is a lazy process-wide manager shared by bundle inspection
and proposer construction, preventing duplicate archive/XGBoost loads.
``_TOP_K_CACHE`` preserves the first environment value for reproducibility and
avoids repeated parsing.  Neither cache is synchronized or reset automatically.
The full parameter Cartesian product is materialized before prediction, so
large parameter spaces can consume substantial memory and time.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

from triton.flagtune._dependencies import require_optional_dependency
from triton.flagtune.core.interfaces import BenchmarkFn, ConfigProposer
from triton.flagtune.contract.identity import ModelIdentity
from triton.flagtune.contract.operator_schema import VariantInfo

np = require_optional_dependency(
    "numpy",
    distribution_name="numpy",
    feature="FlagTune candidate proposal",
)

_MODEL_MANAGER: Optional[Any] = None
_TOP_K_CACHE: Optional[int] = None


def _get_model_manager() -> Any:
    """Return the lazy process-wide model manager shared by public entry points."""
    global _MODEL_MANAGER
    if _MODEL_MANAGER is None:
        from triton.flagtune.runtime.model_loader import FlagTuneModelManager

        _MODEL_MANAGER = FlagTuneModelManager()
    return _MODEL_MANAGER


def _top_k() -> int:
    """Read and memoize the requested candidate count for this process."""
    global _TOP_K_CACHE
    if _TOP_K_CACHE is None:
        _TOP_K_CACHE = int(os.environ.get("FLAGTUNE_TOP_K", "10"))
    return _TOP_K_CACHE


def _config_key(config: Dict[str, Any], fields: List[str]) -> Tuple[Tuple[str, Any], ...]:
    """Build a comparison key from direct or ``META`` parameter values."""
    meta = config.get("META", {}) if isinstance(config.get("META"), dict) else {}
    return tuple(sorted((name, config.get(name, meta.get(name))) for name in fields))


def _strip_config(config: Dict[str, Any], fields: List[str]) -> Dict[str, Any]:
    """Keep only declared parameter fields, accepting legacy ``META`` placement."""
    meta = config.get("META", {}) if isinstance(config.get("META"), dict) else {}
    return {name: config.get(name, meta.get(name)) for name in fields if name in config or name in meta}


def _in_history(history: List[Dict[str, Any]], config: Dict[str, Any], fields: List[str]) -> bool:
    """Return whether a candidate has already been benchmarked in this proposal."""
    target = _config_key(config, fields)
    return any(_config_key(entry.get("config", entry), fields) == target for entry in history)


def _disabled(identity: ModelIdentity) -> bool:
    """Return whether the operator, exact pair, or global wildcard is disabled."""
    raw = os.environ.get("FLAGTUNE_DISABLE_OPS", "").strip()
    if not raw:
        return False
    disabled = {item.strip() for item in raw.split(",") if item.strip()}
    pair = f"{identity.op_id}/{identity.variant}"
    return ("*" in disabled or identity.op_id in disabled or pair in disabled or identity.artifact_key in disabled)


def load_model_bundle(
    op_id: str,
    variant: str,
    *,
    platform_key: str,
    dtype_key: str,
    model_version: Optional[str] = None,
) -> Any:
    """Resolve and cache all runtime data required by one operator variant.

    The returned bundle contains its compiled model config and XGBoost predictor.
    It is shared with :func:`make_config_proposer` through the module-level model
    manager, so integration layers can inspect parameter metadata without loading
    the model twice.
    """
    return _get_model_manager().load(
        op_id,
        variant,
        platform_key=platform_key,
        dtype_key=dtype_key,
        model_version=model_version,
    )


def make_config_proposer(
    op_id: str,
    variant: str,
    *,
    platform_key: str,
    dtype_key: str,
    model_version: Optional[str] = None,
) -> ConfigProposer:
    """Create a model-backed proposer for one canonical operator variant.

    Args:
        op_id: Globally namespaced logical operator identifier.
        variant: Safe single-segment implementation/model variant.

    Returns:
        A :data:`~triton.flagtune.core.interfaces.ConfigProposer` with signature
        ``(benchmark_fn, shape, initial_configs, meta) -> config_dicts``.  When
        ``benchmark_fn`` is ``None``, it returns XGBoost-ranked Top-K candidates.
        Otherwise it benchmarks predicted seeds, generates GA candidates,
        benchmarks those, and returns the lowest-latency unique Top-K.

    Raises:
        FileNotFoundError: If the model bundle cannot be resolved locally or
            remotely.
        IncompatibleModelError: If the model and bundled config identity, digest,
            version, feature names, or feature count are inconsistent.
        ImportError: If a required model dependency such as XGBoost is missing.

    Notes:
        ``FLAGTUNE_TOP_K`` is parsed once per process on first use and
        defaults to 10. ``FLAGTUNE_DISABLE_OPS`` accepts ``*`` or an exact
        ``op_id`` or exact ``op_id/variant``; a disabled model returns an empty proposer so
        integration layers can use their normal fallback.

        The returned callable currently ignores ``initial_configs`` and
        ``meta``.  They remain part of the stable proposer interface for Triton
        integration. Candidate enumeration materializes the full
        parameter Cartesian product before prediction.
    """
    identity = ModelIdentity(platform_key, op_id, variant, dtype_key)
    if _disabled(identity):
        return lambda _fn, _shape, _initial, _meta: []

    loaded = load_model_bundle(
        op_id,
        variant,
        platform_key=platform_key,
        dtype_key=dtype_key,
        model_version=model_version,
    )
    variant_info = loaded.variant
    model = loaded.predictor

    top_k = _top_k()
    fields = variant_info.param_names

    from triton.flagtune.core.ga_search import GAParams, GASearcher

    ga_searcher = GASearcher(
        variant_info.param_space,
        GAParams(
            generations=5,
            population_size=20,
            elite_size=5,
            offspring_per_generation=10,
            mutation_rate=0.3,
            random_rate=0.2,
        ),
        seed=42,
    )

    def propose(
        fn: Optional[BenchmarkFn],
        shape: Dict[str, Any],
        _initial_configs: List[Dict[str, Any]],
        _meta: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Propose ranked candidates for one runtime shape.

        ``shape`` may contain unrelated kernel arguments; variant input
        normalization selects declared inputs and evaluates defaults.  A shape
        failing ``when`` returns an empty list.  Benchmark exceptions and empty
        sample lists are recorded as infinite latency rather than propagated.
        GA failures are likewise treated as no generated candidates.
        """
        inputs = variant_info.normalize_inputs(shape)
        if not variant_info.matches(inputs):
            return []

        predicted = _predict_config_dicts(variant_info, model, inputs, top_k)
        if fn is None:
            return predicted

        history: List[Dict[str, Any]] = []
        for rank, config in enumerate(predicted, start=1):
            stripped = _strip_config(config, fields)
            if len(stripped) != len(fields) or _in_history(history, stripped, fields):
                continue
            try:
                samples = fn(stripped, None)
                latency = float(samples[0]) if samples else float("inf")
            except Exception:
                latency = float("inf")
            history.append({
                "config": stripped,
                "latency_ms": latency,
                "ga_latency_ms": latency,
                "candidate_rank": rank,
            })

        try:
            generated = ga_searcher.generate(history) if history else []
        except Exception:
            generated = []

        for entry in generated:
            stripped = _strip_config(entry.get("config", entry), fields)
            if len(stripped) != len(fields) or _in_history(history, stripped, fields):
                continue
            try:
                samples = fn(stripped, None)
                latency = float(samples[0]) if samples else float("inf")
            except Exception:
                latency = float("inf")
            entry["config"] = stripped
            entry["latency_ms"] = latency
            entry["ga_latency_ms"] = latency
            history.append(entry)

        return _best_from_history(history, fields, top_k)

    return propose


def _best_from_history(history: List[Dict[str, Any]], fields: List[str], top_k: int) -> List[Dict[str, Any]]:
    """Deduplicate benchmark history and return its finite or infinite Top-K order."""
    scored = []
    for entry in history:
        config = _strip_config(entry.get("config", entry), fields)
        latency = float(entry.get("latency_ms", entry.get("ga_latency_ms", float("inf"))))
        scored.append((latency, config))
    scored.sort(key=lambda item: item[0])

    result: List[Dict[str, Any]] = []
    seen = set()
    for _, config in scored:
        key = _config_key(config, fields)
        if key in seen:
            continue
        seen.add(key)
        result.append(config)
        if len(result) >= top_k:
            break
    return result


def _predict_config_dicts(
    variant: VariantInfo,
    model: Any,
    inputs: Dict[str, Any],
    top_k: int,
) -> List[Dict[str, Any]]:
    """Enumerate all legal configs, score their ordered feature matrix, and rank Top-K."""
    configs = list(variant.iter_configs())
    if not configs:
        return []

    matrix = variant.build_feature_matrix(inputs, configs)
    scores = np.asarray(model.predict(matrix), dtype=float)
    if len(scores) != len(configs):
        raise RuntimeError(f"model {variant.op_id}/{variant.name!s} returned "
                           f"{len(scores)} scores for {len(configs)} candidates")

    order = np.argsort(-scores, kind="stable")[:top_k]
    return [configs[int(index)] for index in order]
