# Copyright 2026 FlagOS Contributors
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

import functools
import os

_IMPL_REGISTRY = {}


def get_available_triton_backend():
    return "nvt" if os.environ.get("FLAGTREE_USE_TILEIR", "0") == "1" else "oait"


def get_current_backend():
    return get_available_triton_backend()


def is_backend_available(name):
    return name in {"triton", "nvt", "oait", "cutile", "tilecpp"}


def register_impl(name=None, backend=None):

    def deco(fn):
        impl_name = name or fn.__name__
        impl_backend = backend or "triton"
        _IMPL_REGISTRY[(impl_name, impl_backend)] = fn
        return fn

    if callable(name) and backend is None:
        fn = name
        name = None
        return deco(fn)
    return deco


def mark_perf_ready(*args, **_kwargs):
    if args and callable(args[0]):
        return args[0]

    def deco(fn):
        return fn

    return deco


def dispatch(name, fallback_backend=None):
    """Dispatch to a registered implementation.

    Lookup prefers the explicit dispatch name over the decorated function name,
    and the current backend over the fallback backend over generic ``triton``.
    """

    def deco(fn):

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            backend = get_current_backend()
            candidates = [
                (name, backend),
                (name, fallback_backend or "triton"),
                (name, "triton"),
                (fn.__name__, backend),
                (fn.__name__, fallback_backend or "triton"),
                (fn.__name__, "triton"),
            ]
            for key in candidates:
                impl = _IMPL_REGISTRY.get(key)
                if impl is not None:
                    return impl(*args, **kwargs)
            return fn(*args, **kwargs)

        return wrapper

    return deco
