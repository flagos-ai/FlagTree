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
"""Resolve native per-access policies against the compilation target and path.

Backend support is declared in one table rather than an if/elif chain, and
policies normalize to the existing cache and evict instruction fields.
"""
from importlib import import_module

from triton.language.core import _unwrap_if_constexpr

from ._policy_base import DESCRIPTOR, POINTER

#: backend -> (module relative to this package, {copy path: accepted classes}).
#:
#: A backend absent from this table rejects every explicit policy. A path
#: absent from a backend's entry rejects every policy on that path, which is
#: how descriptor policies stay unsupported on HCU and MetaX.
#:
#: Listing several classes per path replaces the previous reliance on class
#: inheritance to express "this policy is also valid here". PPU's AIU policy
#: drives both descriptor copies and supported block-pointer loads, so it
#: appears on both paths without subclassing the unrelated LSU policy.
_BACKEND_POLICIES = {
    "nvidia": ("._nvidia_cache_policy", {
        POINTER: ("CachePolicy", ),
        DESCRIPTOR: ("CachePolicy", ),
    }),
    "mthreads": (".mthreads.types", {
        POINTER: ("LSUCachePolicy", ),
        DESCRIPTOR: ("TMECachePolicy", ),
    }),
}

_NOT_SUPPORTED = "copy cache_policy is not supported by this backend/path"


def _target_backend(semantic, mthreads_enabled, iluvatar_enabled):
    """Name the compilation target this policy must match."""
    from triton._flagtree_backend import FLAGTREE_BACKEND

    options = getattr(semantic.builder, "options", None)
    target = getattr(options, "backend_name", None)
    if target == "hip":
        return "hcu" if FLAGTREE_BACKEND == "hcu" else "amd"
    if target == "cuda":
        return "iluvatar" if iluvatar_enabled else "nvidia"
    if target is not None:
        return {"musa": "mthreads", "maca": "metax"}.get(target, target)
    # Older builders may not expose backend_name. Keep their established routing.
    if mthreads_enabled:
        return "mthreads"
    if iluvatar_enabled:
        return "iluvatar"
    return FLAGTREE_BACKEND or "nvidia"


def _expected_policy_types(backend, path):
    module_name, paths = _BACKEND_POLICIES[backend]
    class_names = paths.get(path)
    if not class_names:
        raise ValueError(f"{_NOT_SUPPORTED}; backend {backend} has no {path} copy policy")
    try:
        module = import_module(f"{__package__}{module_name}")
        return tuple(getattr(module, name) for name in class_names)
    except (ModuleNotFoundError, AttributeError) as exc:  # pragma: no cover - packaging error
        raise ValueError(f"copy cache_policy lowering is not implemented for backend {backend}") from exc


def _check_architecture(backend, semantic):
    """Reject a policy whose backend name matches but whose target does not."""
    if backend != "nvidia":
        return
    arch = getattr(getattr(semantic.builder, "options", None), "arch", None)
    if arch is not None and not (isinstance(arch, int) or str(arch).startswith("sm")):
        raise ValueError("NVIDIA copy cache policy requires a CUDA compilation target")


def resolve_policy(policy, is_descriptor, is_load, semantic, mthreads_enabled, iluvatar_enabled, common_ir):
    """Validate a policy object against the target, copy path, and direction."""
    policy = _unwrap_if_constexpr(policy)
    if policy is None:
        return None
    backend = _target_backend(semantic, mthreads_enabled, iluvatar_enabled)
    if common_ir:
        raise ValueError(f"{_NOT_SUPPORTED}: the CommonIR pipeline carries no native cache policies")
    if backend not in _BACKEND_POLICIES:
        raise ValueError(f"{_NOT_SUPPORTED}: backend {backend} declares no cache policy types")
    _check_architecture(backend, semantic)
    expected = _expected_policy_types(backend, DESCRIPTOR if is_descriptor else POINTER)
    if not isinstance(policy, expected):
        names = " or ".join(cls.__name__ for cls in expected)
        raise ValueError(f"{_NOT_SUPPORTED}; expected {names}")
    policy.validate_copy(is_descriptor, is_load)
    return policy


def resolve_memory_policy(policy, pointer, is_load, cache_modifier, eviction_policy, semantic):
    """Resolve a policy without changing an ordinary load/store's memory path.

    Like upstream Gluon's policy normalization, an object and legacy string
    controls are mutually exclusive. Keep the existing vendor encoding shared
    with copy, including its target and direction checks.
    """
    policy = _unwrap_if_constexpr(policy)
    if policy is None:
        return None
    if _unwrap_if_constexpr(cache_modifier) or _unwrap_if_constexpr(eviction_policy):
        raise ValueError("cache_policy cannot be combined with cache_modifier or eviction_policy")
    if getattr(policy, "copy_only", False):
        raise ValueError(f"{type(policy).__name__} requires a global-to-shared copy; "
                         "tl.load/store use ordinary memory policies")
    if getattr(pointer.dtype, "address_space", None) != 1:
        raise ValueError("cache_policy requires a global-memory pointer")
    # An ordinary load/store is always a pointer access, so a descriptor-only
    # policy is reported through the same path diagnostic as copy.
    return resolve_policy(policy, False, is_load, semantic, False, False, False)
