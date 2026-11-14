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

"""Backend-neutral registry of post-compile kernel init hooks.

A dialect that needs per-module initialisation on the *loaded* module (e.g.
nvshmem/efshmem/dushmem `*_cumodule_init`) registers a callable here and stamps
its name into the kernel's MLIR module attribute `KERNEL_INIT_HOOK_ATTR`.
`triton.runtime.jit` reads that attribute after every compile and calls
`run_kernel_init_hooks`, which dispatches back to the registered callable.

The registry itself has no backend dependency — keep it that way so backends
(stripped or not) can use it without importing each other.
"""

from __future__ import annotations

from typing import Callable

# MLIR module string attribute listing comma-separated hook names. Stamped by
# `CUDAJITFunction` / `HCUJITFunction` / `TOPSJITFunction.mark_kernel_init_hook`,
# read back through `kernel.metadata.kernel_init_hooks`.
KERNEL_INIT_HOOK_ATTR = "tle.raw.kernel_init_hooks"

_KERNEL_INIT_HOOKS: dict[str, Callable] = {}


def register_kernel_init_hook(name: str, hook: Callable) -> None:
    """Register `hook` under `name`. Re-registering the same callable is a no-op."""
    existing = _KERNEL_INIT_HOOKS.get(name)
    if existing is not None:
        if existing is hook:
            return
        raise RuntimeError(f"kernel init hook {name!r} is already registered")
    _KERNEL_INIT_HOOKS[name] = hook


def run_kernel_init_hooks(kernel) -> None:
    for name in getattr(kernel.metadata, "kernel_init_hooks", ()):
        hook = _KERNEL_INIT_HOOKS.get(name)
        if hook is None:
            raise RuntimeError(f"kernel init hook {name!r} is not registered")
        hook(kernel)
