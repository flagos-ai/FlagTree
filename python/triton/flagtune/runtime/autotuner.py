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
"""
Flagtuner — Autotuner subclass with FlagTune config prediction.

Usage:
    from triton.flagtune import flagtune

    @flagtune(
        configs=[...],
        key=["M", "N", "K"],
        op_id="flaggems/mm",
        variant="general_tma",
        prune_configs_by={"perf_model": estimate_time, "top_k": 10},
    )
    @triton.jit
    def kernel(...):
        ...

When op_id and variant are set and FLAGTUNE_ENABLE=1, the tuner
replaces the pruned config list with XGBoost-predicted Top-K configs
via the ConfigProposer API.  Otherwise behaves exactly like Autotuner.
"""

from __future__ import annotations

import logging
import warnings
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

from triton.runtime.autotuner import Autotuner
from triton.flagtune.runtime.benchmark_protocol import BenchmarkMode, resolve_benchmarker

logger = logging.getLogger(__name__)


def _infer_tensor_dtypes(values: Iterable[Any]) -> Tuple[Any, ...]:
    """Return tensor dtypes in argument order using LibTuner's contract.

    Only ``torch.Tensor`` values and Triton ``TensorDescriptor`` values whose
    ``base`` is a tensor contribute to a model identity. In particular, an
    arbitrary object merely exposing a ``dtype`` attribute is not a tensor
    argument. This exactly matches FlagGems LibTuner's default identity and
    cache-key extraction; callers with a different identity order or source
    must provide ``flagtune_dtype_resolver`` explicitly.

    Returns:
        A tuple of dtypes in the supplied argument order. An empty tuple is
        returned when PyTorch is unavailable or no supported tensor argument
        is present.
    """
    try:
        import torch
    except ImportError:
        return ()

    try:
        from triton.tools.tensor_descriptor import TensorDescriptor
    except ImportError:
        TensorDescriptor = ()

    dtypes = []
    for value in values:
        if isinstance(value, torch.Tensor):
            dtypes.append(value.dtype)
        elif isinstance(value, TensorDescriptor) and isinstance(value.base, torch.Tensor):
            dtypes.append(value.base.dtype)
    return tuple(dtypes)


def _configs_to_dicts(configs: List[Any], param_fields: List[str]) -> List[Dict[str, Any]]:
    """Convert Triton configs into the dictionary form accepted by proposers.

    Only requested kernel parameter fields and Triton launch metadata are
    copied.  Values are coerced to integers, incomplete configs are retained,
    empty results are omitted, and config hooks are intentionally not carried
    into model features or predictions.
    """
    result = []
    for cfg in configs:
        d: Dict[str, Any] = {}
        if hasattr(cfg, "kwargs"):
            for f in param_fields:
                if f in cfg.kwargs:
                    d[f] = int(cfg.kwargs[f])
        if hasattr(cfg, "num_warps"):
            d["num_warps"] = int(cfg.num_warps)
        if hasattr(cfg, "num_stages"):
            d["num_stages"] = int(cfg.num_stages)
        if hasattr(cfg, "num_ctas"):
            d["num_ctas"] = int(cfg.num_ctas)
        if d:
            result.append(d)
    return result


class Flagtuner(Autotuner):
    """Triton autotuner with optional config-driven FlagTune prediction.

    This subclass first executes Triton's normal ``prune_configs`` logic.  If
    ``op_id`` and ``variant`` are supplied and
    ``FLAGTUNE_ENABLE=1``, it lazily creates a FlagTune proposer and replaces
    the pruned list with predicted candidates. When explicitly enabled,
    initialization, prediction, and config-conversion failures propagate.

    Args:
        fn: JIT kernel passed to :class:`triton.runtime.autotuner.Autotuner`.
        arg_names: Kernel argument names.
        configs: Baseline Triton configurations used by the normal autotuner
            and as the fallback set.
        key: Runtime argument names forming Triton's autotune cache key.
        reset_to_zero: Output arguments zeroed before benchmark runs.
        restore_value: Arguments restored after benchmark runs.
        pre_hook: Optional autotuner pre-hook.
        post_hook: Optional autotuner post-hook.
        prune_configs_by: Triton early-pruning/performance-model settings.
        warmup: Benchmark warmup iterations.
        rep: Benchmark measurement iterations.
        benchmark_mode: Architecture-neutral ``replay`` or ``event`` timing.
            The effective default is ``replay``.
        benchmark_retries: Timed replay samples sharing the total ``rep``
            measurement budget.
        use_cuda_graph: Deprecated compatibility alias for
            ``benchmark_mode``.
        op_id: Globally namespaced logical operator identifier.
        variant: Single-segment implementation/model variant.
        flagtune_dtype_resolver: Optional trusted code-side callable receiving
            runtime arguments and returning tensor dtypes in identity order.

    Notes:
        Variant eligibility is loaded from the bundle and validated for every
        runtime shape; no operator registration or selection occurs.
        The global enable flag is cached by :func:`triton.flagtune.is_enabled`
        on first access, so changing the environment later in the process has
        no effect unless that cache is explicitly reset for testing.
    """

    def __init__(
        self,
        fn,
        arg_names,
        configs,
        key,
        reset_to_zero,
        restore_value,
        pre_hook=None,
        post_hook=None,
        prune_configs_by: Optional[Dict] = None,
        warmup=25,
        rep=100,
        use_cuda_graph=None,
        benchmark_mode: Optional[str] = None,
        benchmark_retries: int = 10,
        op_id: Optional[str] = None,
        variant: Optional[str] = None,
        flagtune_dtype_resolver: Optional[Callable[[Dict[str, Any]], Any]] = None,
    ):
        """Initialize Triton's baseline tuner and lazy FlagTune state."""
        if benchmark_mode is not None and use_cuda_graph is not None:
            raise ValueError("benchmark_mode and deprecated use_cuda_graph cannot be supplied together")
        if use_cuda_graph is not None:
            warnings.warn(
                "use_cuda_graph is deprecated; use benchmark_mode='replay' or "
                "benchmark_mode='event'",
                DeprecationWarning,
                stacklevel=2,
            )
            selected_mode = (BenchmarkMode.REPLAY if use_cuda_graph else BenchmarkMode.EVENT)
        else:
            selected_mode = BenchmarkMode(benchmark_mode if benchmark_mode is not None else "replay")
        resolved_benchmark = resolve_benchmarker(
            selected_mode,
            warmup_ms=warmup,
            measurement_ms=rep,
            n_retries=benchmark_retries,
        )
        super().__init__(
            fn,
            arg_names,
            configs,
            key,
            reset_to_zero,
            restore_value,
            pre_hook=pre_hook,
            post_hook=post_hook,
            prune_configs_by=prune_configs_by,
            do_bench=resolved_benchmark.benchmark,
        )
        self.benchmark_protocol = resolved_benchmark.protocol

        if (op_id is None) != (variant is None):
            raise ValueError("FlagTune op_id and variant must be supplied together")
        self._flagtune_op_id = op_id
        self._flagtune_variant = variant
        self._flagtune_dtype_resolver = flagtune_dtype_resolver
        self._flagtune_models: Dict[Any, Any] = {}

    def _runtime_identity(self, kwargs: Dict[str, Any]):
        """Build the trusted GPU/dtype identity for the current kernel call."""
        from triton.flagtune.contract.identity import (
            ModelIdentity,
            discover_gpu_metadata,
            make_dtype_key,
        )

        arguments = {**(self.nargs or {}), **kwargs}
        if self._flagtune_dtype_resolver is None:
            dtypes = _infer_tensor_dtypes(arguments[name] for name in self.arg_names if name in arguments)
        else:
            dtypes = tuple(self._flagtune_dtype_resolver(arguments))
        if not dtypes:
            raise ValueError("FlagTune dtype resolver returned no tensor dtypes")
        gpu = discover_gpu_metadata()
        return ModelIdentity(
            str(gpu["platform_key"]),
            self._flagtune_op_id,
            self._flagtune_variant,
            make_dtype_key(dtypes),
        )

    def _ensure_flagtune(self, identity):
        """Lazily initialize this tuner's proposer and variant metadata once.

        Disabled or unnamed tuners leave normal Triton pruning active. Enabled
        initialization and model contract failures propagate to the caller.
        """
        if identity in self._flagtune_models:
            return self._flagtune_models.get(identity)
        if not self._flagtune_op_id or not self._flagtune_variant:
            return None

        from triton.flagtune import is_enabled as _is_enabled

        if not _is_enabled():
            return None

        from triton.flagtune.runtime.proposer import load_model_bundle, make_config_proposer

        loaded = load_model_bundle(
            identity.op_id,
            identity.variant,
            platform_key=identity.platform_key,
            dtype_key=identity.dtype_key,
        )
        result = (
            make_config_proposer(
                identity.op_id,
                identity.variant,
                platform_key=identity.platform_key,
                dtype_key=identity.dtype_key,
            ),
            loaded.variant,
        )
        self._flagtune_models[identity] = result
        return result

    def prune_configs(self, kwargs):
        """Return FlagTune candidates when available, otherwise Triton candidates.

        Predicted dictionaries are converted to fresh ``triton.Config`` objects.
        Individual conversion failures are skipped.  Triton's configured early
        pruning is applied again to predicted configs; an empty result at any
        stage restores the original pruned list.
        """
        pruned = super().prune_configs(kwargs)
        if not self._flagtune_op_id or not self._flagtune_variant:
            return pruned
        from triton.flagtune import is_enabled as _is_enabled

        if not _is_enabled():
            return pruned
        identity = self._runtime_identity(kwargs)
        model = self._ensure_flagtune(identity)
        proposer, variant_info = model

        param_fields = variant_info.param_names
        initial = _configs_to_dicts(pruned, param_fields)
        meta = {
            "op_id": identity.op_id,
            "variant": identity.variant,
            "platform_key": identity.platform_key,
            "dtype_key": identity.dtype_key,
        }

        config_dicts = proposer(None, self.nargs, initial, meta)

        if not config_dicts:
            raise RuntimeError(f"FlagTune proposer returned no configs for {identity.artifact_key}")

        result = []
        for d in config_dicts:
            result.append(variant_info.to_config(d))

        if not result:
            raise RuntimeError(f"FlagTune proposer produced no usable configs for {identity.artifact_key}")

        if self.early_config_prune:
            result = self.early_config_prune(result, self.nargs, **kwargs)

        if not result:
            raise RuntimeError(f"FlagTune configs were all pruned for {identity.artifact_key}")
        return result


def flagtune(
    configs,
    key,
    *,
    op_id: Optional[str] = None,
    variant: Optional[str] = None,
    flagtune_dtype_resolver: Optional[Callable[[Dict[str, Any]], Any]] = None,
    prune_configs_by: Optional[Dict] = None,
    reset_to_zero=None,
    restore_value=None,
    pre_hook=None,
    post_hook=None,
    warmup=25,
    rep=100,
    use_cuda_graph=None,
    benchmark_mode: Optional[str] = None,
    benchmark_retries: int = 10,
):
    """Decorate a Triton kernel with :class:`Flagtuner`.

    Args:
        configs: Baseline/fallback ``triton.Config`` objects.
        key: Runtime argument names used for autotune cache keys.
        op_id: Globally namespaced logical operator identifier.
        variant: Single-segment implementation/model variant.
        flagtune_dtype_resolver: Optional trusted code-side dtype resolver. It
            cannot be supplied by YAML.
        prune_configs_by: Standard Triton pruning configuration.
        reset_to_zero: Kernel arguments zeroed before benchmarking.
        restore_value: Kernel arguments restored after benchmarking.
        pre_hook: Optional benchmark pre-hook.
        post_hook: Optional benchmark post-hook.
        warmup: Benchmark warmup iterations.
        rep: Benchmark repetitions.
        benchmark_mode: Architecture-neutral ``replay`` or ``event`` timing.
            Omission defaults to ``replay``.
        benchmark_retries: Timed replay samples sharing the total ``rep``
            measurement budget.
        use_cuda_graph: Deprecated compatibility alias for
            ``benchmark_mode``.

    Returns:
        A decorator that replaces a JIT kernel with a :class:`Flagtuner`.

    Notes:
        This API is behavior-compatible with Triton's autotune decorator when
        FlagTune is disabled or unnamed. Model loading is lazy and requires no
        prior registration; enabled contract failures propagate.
    """

    def decorator(fn):
        """Wrap ``fn`` while preserving its declared Triton argument names."""
        return Flagtuner(
            fn,
            fn.arg_names,
            configs,
            key,
            reset_to_zero,
            restore_value,
            pre_hook=pre_hook,
            post_hook=post_hook,
            prune_configs_by=prune_configs_by,
            warmup=warmup,
            rep=rep,
            use_cuda_graph=use_cuda_graph,
            benchmark_mode=benchmark_mode,
            benchmark_retries=benchmark_retries,
            op_id=op_id,
            variant=variant,
            flagtune_dtype_resolver=flagtune_dtype_resolver,
        )

    return decorator
