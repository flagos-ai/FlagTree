# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Runtime adapter for artifacts whose package was generated as C++.

A Triton package is launched by compiling the generated Python at prepare time
and calling into Triton's launcher. A C++ package is already compiled: the
artifact carries a shared library, and launching means calling its plain-C
``flagmega_launch`` with an array of device addresses.

Everything above the launch -- pool allocation, rdata, external-argument
validation, workspace diagnostics, the paged-attention state ABI -- is backend
independent. This subclasses the Triton call-graph module for launch and applies
the C++ package's readonly device-layout contract after canonical rdata loading.
"""

from __future__ import annotations

import ctypes
from pathlib import Path
from typing import Mapping

from triton.flagmega.artifacts.manifest import resolve_artifact_path
from triton.flagmega.errors import ArtifactError, RuntimeContractError


class _CppPackageLibrary:
    """One loaded generated library and its validated C entry points."""

    def __init__(self, path: Path, *, expected_arguments: int) -> None:
        self.path = path
        try:
            self._library = ctypes.CDLL(str(path))
        except OSError as error:
            raise ArtifactError(
                f"Cannot load generated C++ package {path}: {error}."
            ) from error
        for name, restype, argtypes in (
            ("flagmega_launch", ctypes.c_int,
             (ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p)),
            ("flagmega_argument_count", ctypes.c_int, ()),
            ("flagmega_abi_version", ctypes.c_int, ()),
        ):
            try:
                symbol = getattr(self._library, name)
            except AttributeError as error:
                raise ArtifactError(
                    f"Generated C++ package {path} has no {name!r}; it was not "
                    "built from a FlagMega C++ package source."
                ) from error
            symbol.restype = restype
            symbol.argtypes = list(argtypes)
        from triton.flagmega.codegen.cpp.renderer import CPP_RENDERER_VERSION

        abi = int(self._library.flagmega_abi_version())
        if abi != _EXPECTED_ABI_VERSION:
            raise ArtifactError(
                f"Generated C++ package {path} declares launch ABI version "
                f"{abi}; this runtime ({CPP_RENDERER_VERSION}) speaks version "
                f"{_EXPECTED_ABI_VERSION}. Recompile the artifact."
            )
        count = int(self._library.flagmega_argument_count())
        if count != expected_arguments:
            raise ArtifactError(
                f"Generated C++ package {path} takes {count} argument(s) but "
                f"the artifact's runtime binding has {expected_arguments}. The "
                "library and the manifest are out of sync."
            )
        self.argument_count = count

    def launch(self, pointers: tuple[int, ...], stream: int) -> None:
        if len(pointers) != self.argument_count:
            raise RuntimeContractError(
                f"Generated C++ package expects {self.argument_count} device "
                f"pointer(s), got {len(pointers)}."
            )
        array = (ctypes.c_void_p * self.argument_count)(*pointers)
        status = int(
            self._library.flagmega_launch(array, ctypes.c_void_p(stream))
        )
        if status != 0:
            raise RuntimeContractError(
                f"Launching the generated C++ package failed with backend "
                f"error {status}."
            )


# Must match FLAGMEGA_CPP_ABI_VERSION in ftt/runtime.h.
_EXPECTED_ABI_VERSION = 1


class _CppLaunchMixin:
    """Replace Triton JIT prepare/launch with the generated library's."""

    def _load_rdata(self, torch, device: str, expected_nbytes: int):
        pool = super()._load_rdata(torch, device, expected_nbytes)
        contract = self.codegen.get("readonly_device_layout")
        if contract is not None:
            from triton.flagmega.codegen.cpp.readonly_layout import apply_readonly_layout
            apply_readonly_layout(pool, contract, self.ir_module)
        return pool

    def _cpp_library(self) -> _CppPackageLibrary:
        library = getattr(self, "_library_handle", None)
        if library is None:
            name = self.codegen.get("library")
            if not isinstance(name, str) or not name:
                raise ArtifactError(
                    "C++ artifact codegen descriptor names no built library."
                )
            library = _CppPackageLibrary(
                resolve_artifact_path(Path(self.artifact), name),
                expected_arguments=len(self.runtime_binding["signature"]),
            )
            self._library_handle = library
        return library

    def _prepare_external(self, external_arguments: tuple[object, ...]) -> None:
        """Validate arguments and bind the library; nothing is JIT-compiled.

        The Triton path compiles one specialization here and checks its
        register/shared/spill usage against the target's resource contract.
        A C++ package was compiled ahead of time by the vendor driver, which
        does not report those counters through this path, so there is nothing
        to check -- stating that rather than reporting a contract that was
        never evaluated.
        """

        self._require_loaded()
        self._validate_external_arguments(external_arguments)
        self._cpp_library()
        self._prepared = ()
        self.prepare_count += 1
        self._mark_prepared()

    @property
    def prepared(self) -> bool:
        # The base class treats `()` as "not prepared" because a Triton
        # artifact always has at least one prepared kernel. A C++ artifact has
        # none, so prepared-ness is tracked by the library binding instead.
        return getattr(self, "_library_handle", None) is not None

    @property
    def resource_report(self):
        return None

    def _launch_external(
        self, external_arguments: tuple[object, ...], *, stream=None
    ) -> None:
        self._require_prepared()
        self._validate_external_arguments(external_arguments)
        pool_arguments = tuple(
            self._pool_values[str(pool["name"])]
            for pool in self.runtime_binding["pools"]
        )
        if stream is None:
            import torch

            stream = torch.cuda.current_stream(self._device).cuda_stream
        else:
            stream = getattr(stream, "cuda_stream", stream)
        self._cpp_library().launch(
            tuple(
                _device_pointer(value)
                for value in (*external_arguments, *pool_arguments)
            ),
            int(stream),
        )


def _device_pointer(value: object) -> int:
    pointer = getattr(value, "data_ptr", None)
    if pointer is None:
        raise RuntimeContractError(
            "A C++ package launch argument must be a device tensor; got "
            f"{type(value).__name__}."
        )
    return int(pointer())


def create_cpp_tir_runtime(artifact, manifest: Mapping[str, object], ir_module, kernel):
    """Build the C++ runtime adapter matching the artifact's entry ABI.

    Which adapter an entry needs (raw call graph, paged-attention model, gated
    delta net, single tensor) depends only on that entry's structural ABI, so
    this reuses ``tir_runtime_type`` and mixes the C++ launch path into
    whichever class it names. Enumerating backend-specific subclasses instead
    would silently lose any adapter added to the Triton path later.

    `kernel` is None for a C++ package: there is no JIT function to resolve.
    The loader passes it through only to keep one factory signature across both
    backends.
    """

    if kernel is not None:
        raise ArtifactError(
            "A C++ package has no JIT kernel object; the loader must pass None."
        )
    from triton.flagmega.runtime.module import tir_runtime_type

    base = tir_runtime_type(ir_module)
    adapter = _CPP_ADAPTERS.get(base)
    if adapter is None:
        adapter = type(f"GeneratedCpp{base.__name__.removeprefix('Generated')}",
                       (_CppLaunchMixin, base), {})
        _CPP_ADAPTERS[base] = adapter
    return adapter(artifact, manifest, ir_module, kernel)


# Composed adapters are cached so repeated loads of one artifact shape produce
# one class, keeping `isinstance` checks and tracebacks stable.
_CPP_ADAPTERS: dict[type, type] = {}


__all__ = ["create_cpp_tir_runtime"]
