# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

import os

from .autotuner import get_max_configs, max_autotune


def _patch_autotune():
    try:
        import triton
    except ImportError:
        return

    from .autotuner import autotune

    triton.autotune = autotune


def _patch_config_init():
    """Allow Config to round-trip the ubtune_cfg attribute through the disk cache.

    UB-tuner dynamically injects ubtune_cfg onto Config objects via setattr to
    fix UB-overflow compile failures. The disk cache serializes config.__dict__
    to JSON, and on cache hit rebuilds configs with Config(**config_dict).
    The upstream Config.__init__ does not accept ubtune_cfg, so the attribute
    would be lost (or crash) on a disk-cache round-trip. This patch restores it.
    """
    from triton.runtime.autotuner import Config

    _original_config_init = Config.__init__
    _default_num_stages = object()

    def _patched_config_init(self, kwargs, num_warps=4, num_stages=_default_num_stages, num_ctas=1, maxnreg=None,
                             pre_hook=None, ir_override=None, **extra):
        has_mode = kwargs.get("multibuffer_mode") is not None
        if num_stages is _default_num_stages:
            num_stages = None if has_mode else 3
        if has_mode:
            if num_stages is not None:
                raise ValueError("num_stages and multibuffer_mode cannot be specified together")
            from triton.backends.ascend.utils import _multibuffer_mode_to_tuple
            kwargs = dict(kwargs, multibuffer_mode=_multibuffer_mode_to_tuple(kwargs["multibuffer_mode"]))
        _original_config_init(self, kwargs, num_warps, num_stages, num_ctas, maxnreg, pre_hook, ir_override)
        if "ubtune_cfg" in extra:
            setattr(self, "ubtune_cfg", extra["ubtune_cfg"])

    Config.__init__ = _patched_config_init


def _patch_cache_invalidating_env_vars():
    """Include TRITON_ENABLE_UBTUNER in the autotune disk cache key.

    UB-tuner may rescue configs that would otherwise fail to compile, changing
    the benchmark outcome. Toggling it must therefore invalidate cached
    autotune results; otherwise a cache hit from a run without UB-tuner would
    silently skip configs that UB-tuner could have rescued.
    """
    import triton.runtime.autotuner as _autotuner_module

    _original_get_env_vars = _autotuner_module.get_cache_invalidating_env_vars

    def _patched_get_env_vars():
        env_vars = _original_get_env_vars()
        ubtuner_mode = os.environ.get("TRITON_ENABLE_UBTUNER", "")
        if ubtuner_mode:
            env_vars["TRITON_ENABLE_UBTUNER"] = ubtuner_mode
        return env_vars

    _autotuner_module.get_cache_invalidating_env_vars = _patched_get_env_vars


_patch_autotune()
_patch_config_init()
_patch_cache_invalidating_env_vars()
