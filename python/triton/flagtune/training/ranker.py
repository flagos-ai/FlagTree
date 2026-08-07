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
"""Generic offline XGBoost ranking training for FlagTune model configs.

The module intentionally has no knowledge of FlagGems kernels, shape-config
files, or benchmark databases.  A producer writes one JSON object per measured
configuration with ``inputs``, ``config``, and ``latency_ms`` fields.  This
module validates those records against a compiled :class:`VariantInfo`, builds
the exact feature order declared by that variant, trains an ``XGBRanker``, and
exports a self-contained model/config bundle consumed by the runtime manager.

Benchmark data is read twice.  The first pass counts finite ranking rows and the
second fills one preallocated float32 feature matrix.  This avoids retaining a
large Python object graph.  XGBoost ranking is still a single global fit: use
``max_configs_per_shape`` to cap memory without changing group boundaries.
"""

from __future__ import annotations

import json
import math
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Optional, Tuple

import numpy as np

from triton.flagtune._version import __version__ as flagtune_version
from triton.flagtune.contract.operator_schema import (
    VariantInfo,
    load_operator_config,
    model_config_sha256,
    variant_to_model_config,
)
from triton.flagtune.contract.identity import ModelIdentity
from triton.flagtune.contract.archive import MODEL_ARCHIVE_NAME, validate_model_version, write_model_archive


class TrainingDataError(ValueError):
    """Report malformed, insufficient, or incompatible benchmark data."""


@dataclass(frozen=True)
class ExportedModel:
    """Describe one immutable versioned archive emitted by model training."""

    model_path: Path
    model_config: Dict[str, Any]


def load_training_variant(config_path: Path | str, variant: str) -> VariantInfo:
    """Load a training YAML and select its target variant by name.

    Args:
        config_path: Multi-variant FlagTune training configuration read with
            ``yaml.safe_load``.
        variant: Safe variant name declared in the operator configuration.

    Returns:
        The compiled variant used for data validation, feature construction,
        parameter enumeration, training, and artifact export.

    Raises:
        ImportError: If PyYAML is unavailable.
        OSError: If the configuration cannot be read.
        FlagTuneConfigError: If the configuration is malformed or unsafe.
        KeyError: If the requested variant is absent.

    Notes:
        Loading is stateless and does not register the operator globally. Only
        FlagTune's built-in safe expression DSL is accepted.
    """
    operator = load_operator_config(config_path)
    return operator.get_variant(variant)


@dataclass(frozen=True)
class XGBoostTrainingOptions:
    """Configure data preparation and one global XGBoost ranking-model fit.

    The XGBoost fields map directly to :class:`xgboost.XGBRanker`. Data controls
    set the minimum usable corpus, deterministic per-shape sampling limit, and
    progress visibility. ``max_configs_per_shape`` bounds the dense feature
    matrix but does not turn training into an incremental or mini-batch fit.
    Callers should validate device memory independently because XGBoost may
    allocate working memory beyond the float32 matrix reported by this module.
    """

    n_estimators: int = 1200
    max_depth: int = 8
    learning_rate: float = 0.03
    subsample: float = 0.95
    colsample_bytree: float = 0.95
    reg_lambda: float = 1.5
    reg_alpha: float = 0.0
    min_child_weight: float = 1.0
    gamma: float = 0.0
    max_bin: int = 512
    n_jobs: int = 4
    seed: int = 2026
    min_train_rows: int = 8
    max_configs_per_shape: Optional[int] = None
    show_progress: bool = True


@dataclass(frozen=True)
class PreparedRankingData:
    """Hold dense ranking arrays arranged in contiguous per-shape groups.

    ``features`` and ``labels`` have ``row_count`` rows. ``group_sizes`` gives
    the consecutive query-group boundaries required by ``XGBRanker.fit`` and
    sums to ``row_count``. Groups with fewer than two finite rows are excluded,
    while the two skip counters retain audit information about discarded input.
    Arrays are float32 to limit memory and match FlagTune inference features.
    """

    features: np.ndarray
    labels: np.ndarray
    group_sizes: List[int]
    shape_count: int
    row_count: int
    skipped_nonfinite_rows: int
    sampled_out_rows: int


def _iter_jsonl(path: Path) -> Iterator[Tuple[int, Dict[str, Any]]]:
    """Yield non-empty JSONL objects with one-based source line numbers.

    Args:
        path: UTF-8 benchmark JSONL file to stream.

    Yields:
        ``(line_number, record)`` pairs for decoded JSON objects.

    Raises:
        TrainingDataError: If a non-empty line is invalid JSON or is not an
            object. Blank lines are ignored and the complete file is not loaded
            into memory.
    """
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise TrainingDataError(f"invalid JSON in {path} at line {line_number}: {exc}") from exc
            if not isinstance(record, dict):
                raise TrainingDataError(f"{path}:{line_number} must contain a JSON object")
            yield line_number, record


def _ranking_group_key(record: Mapping[str, Any], line_number: int) -> str:
    """Return the canonical key for one structured Schema v2 ranking group."""
    group = record.get("ranking_group")
    if not isinstance(group, Mapping):
        raise TrainingDataError(f"benchmark data line {line_number} has no ranking_group mapping")
    required = ("operator_id", "variant", "dimensions", "model_dtype_key")
    missing = [name for name in required if name not in group]
    if missing:
        raise TrainingDataError(f"benchmark data line {line_number} ranking_group is missing "
                                f"{', '.join(missing)}")
    if not isinstance(group["dimensions"], Mapping):
        raise TrainingDataError(f"benchmark data line {line_number} ranking_group dimensions "
                                "must be a mapping")
    try:
        return json.dumps(dict(group), sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise TrainingDataError(f"benchmark data line {line_number} has invalid ranking_group: {exc}") from exc


def _finite_latency(record: Mapping[str, Any]) -> Optional[float]:
    """Return a positive finite latency, or ``None`` for an unusable target.

    Missing, non-numeric, non-finite, zero, and negative values are filtered
    rather than rejected so exhaustive benchmark failures remain auditable in
    the same JSONL corpus without poisoning rank labels.
    """
    value = record.get("latency_ms")
    if value is None:
        return None
    try:
        latency = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(latency) or latency <= 0:
        return None
    return latency


def _group_records(path: Path, ) -> Iterator[Tuple[str, List[Tuple[int, Dict[str, Any]]]]]:
    """Yield contiguous ranking groups and reject later duplicate groups.

    Args:
        path: Benchmark JSONL ordered by structured ``ranking_group``.

    Yields:
        Each canonical ranking-group key and its line-numbered records in
        original file order.

    Raises:
        TrainingDataError: If a key reappears after another group has started.

    Limitation:
        One group's records are retained as Python dictionaries at a time. The
        producer must therefore keep every shape contiguous and should avoid an
        unbounded number of configs for a single shape.
    """
    current_key: Optional[str] = None
    current: List[Tuple[int, Dict[str, Any]]] = []
    completed: set[str] = set()
    for line_number, record in _iter_jsonl(path):
        key = _ranking_group_key(record, line_number)
        if current_key is None:
            current_key = key
        if key != current_key:
            completed.add(current_key)
            yield current_key, current
            if key in completed:
                raise TrainingDataError(f"ranking_group {key!r} is not contiguous in {path}")
            current_key = key
            current = []
        current.append((line_number, record))
    if current_key is not None:
        yield current_key, current


def _selected_positions(
    valid_count: int,
    limit: Optional[int],
    seed: int,
    group_index: int,
) -> np.ndarray:
    """Choose stable source-order positions for deterministic group sampling.

    All rows are returned when ``limit`` is absent or large enough. Otherwise a
    NumPy generator seeded by the global seed and group index samples without
    replacement, then sorts the selected positions to preserve source order.
    """
    if limit is None or valid_count <= limit:
        return np.arange(valid_count, dtype=np.int64)
    rng = np.random.default_rng(np.random.SeedSequence([seed, group_index]))
    return np.sort(rng.choice(valid_count, size=limit, replace=False))


def _group_plan(
    path: Path,
    options: XGBoostTrainingOptions,
) -> Tuple[List[int], int, int, int]:
    """Plan dense allocation sizes in the first streaming pass over JSONL.

    Returns:
        ``(group_sizes, train_rows, skipped_rows, sampled_out_rows)``. Non-finite
        rows and finite groups too small for pairwise ranking count as skipped;
        deterministic sampling exclusions are reported separately.

    Note:
        The file is opened again by :func:`prepare_ranking_data`. A concurrent
        writer can invalidate this plan and is detected after the second pass.
    """
    group_sizes: List[int] = []
    skipped = 0
    sampled_out = 0
    platform_keys: set[str] = set()
    architectures: set[str] = set()
    dtype_keys: set[str] = set()
    for _group_index, (_key, records) in enumerate(_group_records(path)):
        for line_number, record in records:
            model_identity = record.get("model_identity")
            dtypes = record.get("dtypes")
            device = record.get("device")
            metadata = device.get("metadata") if isinstance(device, Mapping) else None
            architecture = metadata.get("architecture") if isinstance(metadata, Mapping) else None
            if architecture is not None:
                if not isinstance(architecture, str):
                    raise TrainingDataError(f"benchmark data line {line_number} has invalid device architecture")
                architectures.add(architecture)
            identity_fields = (
                model_identity.get("platform_key") if isinstance(model_identity, Mapping) else None,
                model_identity.get("dtype_key") if isinstance(model_identity, Mapping) else None,
                dtypes.get("inputs") if isinstance(dtypes, Mapping) else None,
                dtypes.get("outputs") if isinstance(dtypes, Mapping) else None,
            )
            if any(value is not None for value in identity_fields):
                platform_key, dtype_key, input_dtypes, output_dtypes = identity_fields
                if not isinstance(platform_key, str) or not isinstance(dtype_key, str):
                    raise TrainingDataError(f"benchmark data line {line_number} has incomplete platform/dtype identity")
                if not isinstance(input_dtypes, list) or not isinstance(output_dtypes, list):
                    raise TrainingDataError(f"benchmark data line {line_number} must contain dtype lists")
                from triton.flagtune.contract.identity import make_dtype_key

                if make_dtype_key([*input_dtypes, *output_dtypes]) != dtype_key:
                    raise TrainingDataError(f"benchmark data line {line_number} has inconsistent "
                                            "model_identity.dtype_key")
                ranking_group = record.get("ranking_group")
                if (not isinstance(ranking_group, Mapping) or ranking_group.get("model_dtype_key") != dtype_key):
                    raise TrainingDataError(f"benchmark data line {line_number} has inconsistent "
                                            "ranking_group.model_dtype_key")
                platform_keys.add(platform_key)
                dtype_keys.add(dtype_key)
        finite_count = sum(_finite_latency(record) is not None for _, record in records)
        skipped += len(records) - finite_count
        selected_count = finite_count
        if options.max_configs_per_shape is not None:
            selected_count = min(selected_count, options.max_configs_per_shape)
        sampled_out += finite_count - selected_count
        # Ranking groups with one row cannot contribute pairwise comparisons.
        if selected_count >= 2:
            group_sizes.append(selected_count)
        else:
            skipped += selected_count
    if len(platform_keys) > 1:
        raise TrainingDataError(f"benchmark data mixes platform identities: {sorted(platform_keys)}")
    if len(architectures) > 1:
        raise TrainingDataError(f"benchmark data mixes device architectures: {sorted(architectures)}")
    if len(dtype_keys) > 1:
        raise TrainingDataError(f"benchmark data mixes dtype identities: {sorted(dtype_keys)}")
    return group_sizes, sum(group_sizes), skipped, sampled_out


def prepare_ranking_data(
    variant: VariantInfo,
    benchmark_path: Path | str,
    options: XGBoostTrainingOptions,
) -> PreparedRankingData:
    """Build a compact feature matrix and per-shape relevance labels.

    Lower latency receives a larger non-negative label, matching FlagTune's
    inference convention that larger XGBoost scores are better.

    Args:
        variant: Registered operator variant that validates configs, normalizes
            feature order, and constructs the numeric feature matrix.
        benchmark_path: JSONL containing contiguous groups with
            ``ranking_group``, ``inputs``, ``config``, and ``latency_ms`` fields.
        options: Training and deterministic sampling controls.

    Returns:
        Preallocated float32 features, rank labels, XGBoost group sizes, and
        discard statistics in :class:`PreparedRankingData`.

    Raises:
        FileNotFoundError: If the benchmark corpus does not exist.
        ValueError: If data-control options are invalid.
        TrainingDataError: For insufficient rows, malformed groups/configs,
            inconsistent inputs, or non-finite/generated feature values.

    Implementation and limitations:
        The file is streamed twice: first to size allocation and again to fill
        it. Only one shape's JSON objects are retained, but the full dense matrix
        is resident for the single global XGBoost fit. Tied latencies inherit
        stable source order rather than sharing a relevance label.
    """
    path = Path(benchmark_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"benchmark JSONL not found: {path}")
    if options.min_train_rows <= 0:
        raise ValueError("min_train_rows must be positive")
    if (options.max_configs_per_shape is not None and options.max_configs_per_shape < 2):
        raise ValueError("max_configs_per_shape must be at least 2")

    planned_groups, row_count, skipped, sampled_out = _group_plan(path, options)
    if row_count < options.min_train_rows:
        raise TrainingDataError(f"not enough finite ranking rows: {row_count} < {options.min_train_rows}")

    feature_count = len(variant.feature_names)
    features = np.empty((row_count, feature_count), dtype=np.float32)
    labels = np.empty(row_count, dtype=np.float32)
    group_sizes: List[int] = []
    offset = 0

    for group_index, (_key, records) in enumerate(_group_records(path)):
        finite = [(line_number, record, latency)
                  for line_number, record in records
                  if (latency := _finite_latency(record)) is not None]
        positions = _selected_positions(len(finite), options.max_configs_per_shape, options.seed, group_index)
        if len(positions) < 2:
            continue
        selected = [finite[int(position)] for position in positions]
        first_inputs = selected[0][1].get("inputs")
        if not isinstance(first_inputs, Mapping):
            raise TrainingDataError(f"benchmark data line {selected[0][0]} has no inputs mapping")
        ranking_group = selected[0][1].get("ranking_group")
        if not isinstance(ranking_group, Mapping):
            raise TrainingDataError(f"benchmark data line {selected[0][0]} has no ranking_group mapping")
        if (ranking_group.get("operator_id") != variant.op_id or ranking_group.get("variant") != variant.name):
            raise TrainingDataError(f"benchmark data line {selected[0][0]} ranking_group does not "
                                    f"match {variant.op_id}/{variant.name}")
        if ranking_group.get("dimensions") != first_inputs:
            raise TrainingDataError(f"benchmark data line {selected[0][0]} ranking_group dimensions "
                                    "do not match inputs")
        configs: List[Mapping[str, Any]] = []
        latencies = np.empty(len(selected), dtype=np.float64)
        for index, (line_number, record, latency) in enumerate(selected):
            inputs = record.get("inputs")
            config = record.get("config")
            if inputs != first_inputs:
                raise TrainingDataError(f"ranking group contains inconsistent inputs at line {line_number}")
            if not isinstance(config, Mapping):
                raise TrainingDataError(f"benchmark data line {line_number} has no config mapping")
            if not variant.param_space.validate(dict(config)):
                raise TrainingDataError(f"benchmark data line {line_number} has a config outside "
                                        f"{variant.op_id}/{variant.name}'s parameter space")
            configs.append(config)
            latencies[index] = float(latency)

        matrix = variant.build_feature_matrix(first_inputs, configs)
        if matrix.shape != (len(selected), feature_count):
            raise TrainingDataError(f"feature matrix has shape {matrix.shape}, expected "
                                    f"({len(selected)}, {feature_count})")
        if not np.isfinite(matrix).all():
            raise TrainingDataError(f"non-finite feature value for ranking group {_key!r}")
        end = offset + len(selected)
        features[offset:end] = matrix
        order = np.argsort(latencies, kind="stable")
        group_labels = np.empty(len(selected), dtype=np.float32)
        group_labels[order] = np.arange(len(selected) - 1, -1, -1, dtype=np.float32)
        labels[offset:end] = group_labels
        group_sizes.append(len(selected))
        offset = end

    if offset != row_count or group_sizes != planned_groups:
        raise TrainingDataError("benchmark data changed while it was being read; retry with a stable file")
    return PreparedRankingData(
        features=features,
        labels=labels,
        group_sizes=group_sizes,
        shape_count=len(group_sizes),
        row_count=row_count,
        skipped_nonfinite_rows=skipped,
        sampled_out_rows=sampled_out,
    )


def estimate_dense_matrix_bytes(row_count: int, feature_count: int) -> int:
    """Return bytes required by the preallocated float32 feature matrix.

    This estimate excludes labels, group metadata, per-group Python objects, and
    XGBoost's internal histogram/booster allocations, so it is a lower bound on
    peak process memory rather than a complete training-memory prediction.
    """
    return int(row_count) * int(feature_count) * np.dtype(np.float32).itemsize


def _progress_callback(total: int, enabled: bool) -> Tuple[List[Any], Any]:
    """Create XGBoost callbacks with tqdm or a flushed text fallback.

    Args:
        total: Expected boosting rounds, normally ``n_estimators``.
        enabled: Return no callbacks and produce no output when false.

    Returns:
        A callback list and an optional tqdm object that the caller must close.
        Without tqdm, the callback prints the first tree, approximately every
        five percent, and the final tree with rate and ETA.

    Limitation:
        Progress counts completed boosting iterations, not feature preparation
        or time spent before XGBoost invokes its first callback.
    """
    if not enabled:
        return [], None
    from xgboost.callback import TrainingCallback

    try:
        from tqdm.auto import tqdm
    except ImportError:
        tqdm = None

    if tqdm is None:
        interval = max(1, total // 20)
        started = time.perf_counter()

        class _ConsoleTrainingCallback(TrainingCallback):
            """Emit portable text progress when tqdm is unavailable."""

            def after_iteration(self, model: Any, epoch: int, evals_log: Any) -> bool:
                """Print selected completed rounds and continue training.

                ``model`` and ``evals_log`` are accepted for XGBoost callback
                compatibility but are not inspected because this reporter has
                no validation dataset or metric display contract.
                """
                del model, evals_log
                completed = epoch + 1
                if completed == 1 or completed % interval == 0 or completed >= total:
                    elapsed = max(time.perf_counter() - started, 1e-9)
                    rate = completed / elapsed
                    remaining = max(total - completed, 0)
                    eta = remaining / rate if rate > 0 else float("inf")
                    print(
                        f"XGBoost progress: {completed}/{total} trees "
                        f"rate={rate:.2f}_tree/s eta={eta:.1f}s",
                        flush=True,
                    )
                return False

        return [_ConsoleTrainingCallback()], None

    progress = tqdm(total=total, desc="XGBoost", unit="tree")

    class _TqdmTrainingCallback(TrainingCallback):
        """Advance one externally owned tqdm bar after each boosting round."""

        def after_iteration(self, model: Any, epoch: int, evals_log: Any) -> bool:
            """Synchronize the bar with XGBoost's zero-based epoch and continue."""
            del model, evals_log
            progress.update(max(0, epoch + 1 - progress.n))
            return False

    return [_TqdmTrainingCallback()], progress


def train_xgboost_ranker(
    variant: VariantInfo,
    benchmark_path: Path | str,
    options: Optional[XGBoostTrainingOptions] = None,
) -> Tuple[Any, Dict[str, Any]]:
    """Prepare benchmark data and fit one global XGBoost ranking model.

    Args:
        variant: Registered FlagTune variant defining legal configs/features.
        benchmark_path: Stable JSONL corpus produced by exhaustive collection.
        options: Optional hyperparameters and data controls; defaults are used
            when omitted.

    Returns:
        The fitted ``XGBRanker`` and a JSON-serializable training summary.

    Raises:
        ImportError: If the optional XGBoost dependency is unavailable.
        ValueError: For invalid core hyperparameters or malformed training data.

    Implementation and limitation:
        Feature preparation precedes one ``tree_method='hist'`` pairwise-ranking
        fit with per-shape query groups. This API does not perform validation,
        early stopping, checkpoint continuation, or incremental batch training.
    """
    resolved = options or XGBoostTrainingOptions()
    if resolved.n_estimators <= 0:
        raise ValueError("n_estimators must be positive")
    if resolved.max_depth <= 0:
        raise ValueError("max_depth must be positive")
    if resolved.learning_rate <= 0:
        raise ValueError("learning_rate must be positive")

    try:
        from xgboost import XGBRanker
    except ImportError as exc:
        raise ImportError("FlagTune model training requires xgboost; install the optional "
                          "training dependency before running this API") from exc

    prepare_start = time.perf_counter()
    data = prepare_ranking_data(variant, benchmark_path, resolved)
    prepare_elapsed = time.perf_counter() - prepare_start
    callbacks, progress = _progress_callback(resolved.n_estimators, resolved.show_progress)
    model = XGBRanker(
        n_estimators=resolved.n_estimators,
        max_depth=resolved.max_depth,
        learning_rate=resolved.learning_rate,
        subsample=resolved.subsample,
        colsample_bytree=resolved.colsample_bytree,
        reg_lambda=resolved.reg_lambda,
        reg_alpha=resolved.reg_alpha,
        min_child_weight=resolved.min_child_weight,
        gamma=resolved.gamma,
        max_bin=resolved.max_bin,
        n_jobs=resolved.n_jobs,
        objective="rank:pairwise",
        eval_metric="ndcg",
        tree_method="hist",
        random_state=resolved.seed,
        callbacks=callbacks or None,
    )
    fit_start = time.perf_counter()
    try:
        model.fit(data.features, data.labels, group=data.group_sizes, verbose=False)
    finally:
        if progress is not None:
            progress.close()
    fit_elapsed = time.perf_counter() - fit_start

    summary = {
        "op_id": variant.op_id,
        "variant": variant.name,
        "feature_cols": list(variant.feature_names),
        "feature_count": len(variant.feature_names),
        "train_shape_count": data.shape_count,
        "train_row_count": data.row_count,
        "skipped_nonfinite_rows": data.skipped_nonfinite_rows,
        "sampled_out_rows": data.sampled_out_rows,
        "dense_feature_matrix_bytes": int(data.features.nbytes),
        "prepare_elapsed_s": prepare_elapsed,
        "xgboost_fit_elapsed_s": fit_elapsed,
        "xgboost_options": asdict(resolved),
    }
    return model, summary


def export_ranker_model(
    model: Any,
    variant: VariantInfo,
    output_root: Path | str,
    training_summary: Mapping[str, Any],
    *,
    identity: ModelIdentity,
    dtypes: List[str],
    gpu: Mapping[str, Any],
    model_version: str,
) -> ExportedModel:
    """Write one self-contained ``model.tar.gz`` archive for package staging.

    Args:
        model: Fitted object exposing XGBoost's ``save_model`` method.
        variant: Compiled training definition used for features and legal configs.
        output_root: Root under which the operator/variant/dtype path is derived.
        training_summary: JSON-serializable audit metadata from training.

    Returns:
        The final archive path and exact single-model configuration.

    Outputs and pitfalls:
        The three required files exist only as archive members. The archive is
        reproducible and atomically replaces an existing archive at the same
        staging path. Version identity remains in the child config and outer
        platform package. The canonical config SHA-256 is stored inside the
        Booster and verified at load time.
    """
    try:
        import yaml
    except ImportError as exc:
        raise ImportError("FlagTune model export requires PyYAML") from exc

    version = validate_model_version(model_version)
    target = (Path(output_root).expanduser().resolve() / Path(*identity.op_id.split("/")) / identity.variant /
              identity.dtype_key)
    target.mkdir(parents=True, exist_ok=True)
    config = variant_to_model_config(variant, identity, dtypes, gpu, version)
    config["flagtune_version_min"] = flagtune_version
    config_digest = model_config_sha256(config)

    booster = model.get_booster()
    booster.set_attr(flagtune_config_sha256=config_digest)
    booster.feature_names = list(variant.feature_names)
    summary = dict(training_summary)
    summary["op_id"] = variant.op_id
    summary["variant"] = variant.name
    summary["feature_cols"] = list(variant.feature_names)
    summary["feature_count"] = len(variant.feature_names)
    summary["model_config_sha256"] = config_digest
    summary["model_version"] = version
    with tempfile.TemporaryDirectory(prefix="flagtune-export-") as temporary_dir:
        loose_model_path = Path(temporary_dir) / "xgboost_ranker.json"
        model.save_model(str(loose_model_path))
        members = {
            "xgboost_ranker.json": loose_model_path.read_bytes(),
            "flagtune_config.yaml": yaml.safe_dump(config, sort_keys=False, allow_unicode=True).encode("utf-8"),
            "training_summary.json": json.dumps(summary, indent=2, sort_keys=True).encode("utf-8"),
        }
    model_path = write_model_archive(target / MODEL_ARCHIVE_NAME, members)
    for loose_name in members:
        loose_path = target / loose_name
        if loose_path.is_file():
            loose_path.unlink()
    return ExportedModel(
        model_path=model_path,
        model_config=config,
    )
