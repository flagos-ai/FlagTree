from __future__ import annotations

import hashlib
import logging
import time
import warnings

from ._layout_autotune import (
    FallbackVariantExecutionError,
    GluonLayoutCompiledVariant,
    GluonLayoutVariantBundle,
    _CandidateCompileFailure,
    _IsolatedCompileRequest,
    _compile_variant_isolated,
    _make_launcher_abi_descriptor,
    _validate_compile_metrics,
    parse_layout_manifest,
)
from ._layout_autotune_factory import create_layout_autotuner


_LOGGER = logging.getLogger(__name__)


class GluonLayoutAutotuneRuntimeMixin:
    """Add whole-module layout candidate compilation to a Gluon JIT function.

    Keeping the candidate protocol in this mixin leaves the ordinary Gluon AST
    and binder path independent of MetaX layout autotuning.  The only generic
    runtime contract is ``JITFunction._prepare_kernel_for_launch``.
    """

    def __init__(self, *args, layout_autotune=True, **kwargs):
        if not isinstance(layout_autotune, bool):
            raise TypeError("layout_autotune must be a bool")
        self.layout_autotune_enabled = layout_autotune
        super().__init__(*args, **kwargs)
        self._layout_autotune = (
            create_layout_autotuner(self) if layout_autotune else None
        )

    @staticmethod
    def _validate_compiled_layout_variant(
        kernel, manifest, specification, runtime_contract
    ):
        metadata = getattr(kernel, "metadata", None)
        domain_digest = getattr(metadata, "gluon_layout_domain_digest", None)
        variant_digest = getattr(metadata, "gluon_layout_variant_digest", None)
        if (
            domain_digest != manifest["digest"]
            or variant_digest != specification.digest
        ):
            raise RuntimeError(
                f"compiled Gluon layout variant {specification.digest} "
                "changed its identity"
            )
        candidate_contract = getattr(
            metadata, "gluon_layout_runtime_contract", None
        )
        if candidate_contract != runtime_contract:
            raise RuntimeError(
                f"compiled Gluon layout variant {specification.digest} "
                "changed its effect contract"
            )

    @staticmethod
    def _bind_compiled_launcher_abi(kernel, source_kernel, specification):
        """Bind a file-compiled executable to the outer JIT launch signature.

        A TTGIR ``IRSource`` exposes only device ABI arguments, while the outer
        Gluon ``ASTSource`` launcher also accepts constexpr arguments.  Rebind
        the source before lazy launcher initialization so every candidate uses
        the exact same host ABI without altering its independently built binary.
        """
        source = getattr(source_kernel, "src", None)
        if (
            source is None
            or not hasattr(source, "fn")
            or not isinstance(getattr(source, "signature", None), dict)
            or not isinstance(getattr(source, "constants", None), dict)
        ):
            raise RuntimeError(
                "outer Gluon compilation does not expose its AST launch signature"
            )
        if any(
            getattr(kernel, field, None) is not None
            for field in ("module", "function", "_run")
        ):
            raise RuntimeError(
                f"compiled Gluon layout variant {specification.digest} "
                "initialized its launcher too early"
            )
        kernel.src = source
        _LOGGER.debug(
            "bound Gluon layout variant launcher digest=%s file=%s "
            "to outer AST signature",
            specification.digest,
            specification.mlir_file,
        )

    @staticmethod
    def _get_compiled_device_signature(kernel, specification):
        signature = getattr(getattr(kernel, "src", None), "signature", None)
        if (
            not isinstance(signature, dict)
            or set(signature) != set(range(len(signature)))
        ):
            raise RuntimeError(
                f"compiled Gluon layout variant {specification.digest} "
                "has an invalid device signature"
            )
        return tuple(str(signature[index]) for index in range(len(signature)))

    def _compile_layout_variant_bundle(
        self, fallback, target, options, env_vars=None
    ):
        if not getattr(self, "layout_autotune_enabled", True):
            return fallback
        metadata = getattr(fallback, "metadata", None)
        manifest_raw = getattr(metadata, "gluon_layout_manifest", None)
        runtime_contract = getattr(
            metadata, "gluon_layout_runtime_contract", None
        )
        # Effect metadata can exist without candidate files.  The manifest is
        # the explicit opt-in boundary for runtime variant consumption.
        if manifest_raw is None:
            return fallback
        if not isinstance(manifest_raw, dict) or not isinstance(
            runtime_contract, dict
        ):
            raise RuntimeError(
                "compiled Gluon fallback exposes incomplete layout metadata"
            )

        from triton.runtime.cache import get_cache_manager

        manifest = parse_layout_manifest(manifest_raw)
        cache = get_cache_manager(fallback.hash)
        variant_paths = {}
        for specification in manifest.variants:
            path = cache.get_file(specification.mlir_file)
            if not path:
                raise RuntimeError(
                    f"missing Gluon layout variant file "
                    f"{specification.mlir_file!r}"
                )
            with open(path, "rb") as handle:
                actual_sha256 = hashlib.sha256(handle.read()).hexdigest()
            if actual_sha256 != specification.mlir_sha256:
                raise RuntimeError(
                    f"Gluon layout variant {specification.digest} MLIR SHA "
                    f"mismatch: expected {specification.mlir_sha256}, "
                    f"got {actual_sha256}"
                )
            variant_paths[specification.digest] = path

        variants = {}
        compile_metrics = {}
        fallback_device_signature = None
        launcher_abi = _make_launcher_abi_descriptor(fallback)

        def compile_variant(specification):
            started_ns = time.perf_counter_ns()
            try:
                if specification == manifest.fallback:
                    compile_kwargs = {"target": target, "options": options}
                    if env_vars is not None:
                        compile_kwargs["_env_vars"] = env_vars
                    variant_kernel = self.compile(
                        variant_paths[specification.digest], **compile_kwargs
                    )
                    isolated_device_signature = None
                    metrics = _validate_compile_metrics(
                        {
                            "mode": "parent-fallback",
                            "outcome": "success",
                            "wall_us": (
                                time.perf_counter_ns() - started_ns
                            )
                            // 1000,
                        }
                    )
                else:
                    request = _IsolatedCompileRequest(
                        variant_paths[specification.digest],
                        target,
                        dict(options),
                        dict(env_vars) if env_vars is not None else None,
                        manifest.digest,
                        specification.digest,
                        runtime_contract,
                        launcher_abi,
                    )
                    (
                        variant_kernel,
                        isolated_device_signature,
                        metrics,
                    ) = _compile_variant_isolated(request, fallback)
            except Exception as error:
                if isinstance(error, _CandidateCompileFailure):
                    raise
                raise _CandidateCompileFailure(
                    "compile-error",
                    f"standalone candidate compilation failed: {error}",
                ) from error

            # Identity, effects, and ABI are protocol invariants, not candidate
            # quality.  A mismatch invalidates the domain instead of becoming a
            # cached experimental failure.
            self._validate_compiled_layout_variant(
                variant_kernel, manifest_raw, specification, runtime_contract
            )
            device_signature = (
                isolated_device_signature
                if isolated_device_signature is not None
                else self._get_compiled_device_signature(
                    variant_kernel, specification
                )
            )
            if (
                fallback_device_signature is not None
                and device_signature != fallback_device_signature
            ):
                raise RuntimeError(
                    f"compiled Gluon layout variant {specification.digest} "
                    "changed the device ABI signature: "
                    f"expected {fallback_device_signature}, "
                    f"got {device_signature}"
                )
            self._bind_compiled_launcher_abi(
                variant_kernel, fallback, specification
            )
            compile_metrics[specification.digest] = metrics
            _LOGGER.debug(
                "compiled Gluon layout variant digest=%s mode=%s wall_us=%s",
                specification.digest,
                metrics.get("mode"),
                metrics.get("wall_us"),
            )
            return GluonLayoutCompiledVariant(
                specification.digest,
                specification.mlir_file,
                specification.mlir_sha256,
                device_signature,
                variant_kernel,
            )

        try:
            compiled_fallback = compile_variant(manifest.fallback)
        except _CandidateCompileFailure as error:
            raise RuntimeError(
                "failed to independently compile the mandatory Gluon layout "
                "fallback"
            ) from error
        fallback_device_signature = compiled_fallback.device_signature
        variants[manifest.fallback.digest] = compiled_fallback
        _LOGGER.debug(
            "prepared Gluon layout domain digest=%s variants=%d fallback=%s",
            manifest.digest,
            len(manifest.variants),
            manifest.fallback.digest,
        )
        return GluonLayoutVariantBundle(
            manifest_raw,
            variants,
            runtime_contract,
            source_kernel=fallback,
            sources=variant_paths,
            compile_variant=compile_variant,
            compile_metrics=compile_metrics,
        )

    def _do_compile(
        self, key, signature, device, constexprs, options, attrs, warmup
    ):
        from triton import knobs
        from triton._C.libtriton import get_cache_invalidating_env_vars
        from triton.runtime import _async_compile
        from triton.runtime.cache import get_cache_key

        kernel_cache, _, target, backend, _ = self.device_caches[device]
        if self._call_hook(
            knobs.runtime.jit_cache_hook,
            key,
            signature,
            device,
            constexprs,
            options,
            [attrs],
            warmup,
        ):
            return None
        src = self.ASTSource(self, signature, constexprs, attrs)

        def finalize_compile(bundle):
            kernel_cache[key] = bundle
            self._call_hook(
                knobs.runtime.jit_post_compile_hook,
                key,
                signature,
                device,
                constexprs,
                options,
                [attrs],
                warmup,
            )

        env_vars = get_cache_invalidating_env_vars()
        async_mode = _async_compile.active_mode.get()
        if async_mode is not None:
            cache_key = get_cache_key(src, backend, options, env_vars)

            def compile_bundle():
                fallback = self.compile(
                    src,
                    target=target,
                    options=options.__dict__,
                    _env_vars=env_vars,
                )
                return self._compile_layout_variant_bundle(
                    fallback, target, options.__dict__, env_vars
                )

            return async_mode.submit(
                cache_key, compile_bundle, finalize_compile
            )

        fallback = self.compile(
            src,
            target=target,
            options=options.__dict__,
            _env_vars=env_vars,
        )
        bundle = self._compile_layout_variant_bundle(
            fallback, target, options.__dict__, env_vars
        )
        finalize_compile(bundle)
        return bundle

    def _prepare_kernel_for_launch(self, kernel, grid, stream, bound_args):
        if not getattr(self, "layout_autotune_enabled", True):
            if isinstance(kernel, GluonLayoutVariantBundle):
                return kernel.fallback_kernel
            return kernel
        try:
            return self._layout_autotune.prepare_for_launch(
                kernel, grid, stream, bound_args
            )
        except FallbackVariantExecutionError:
            raise
        except Exception as error:
            warnings.warn(
                "Gluon layout variant preparation fell back to the compiler "
                f"default: {error}",
                RuntimeWarning,
            )
            if isinstance(kernel, GluonLayoutVariantBundle):
                return kernel.fallback_kernel
            return kernel
