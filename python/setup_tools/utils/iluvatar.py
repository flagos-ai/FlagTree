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

import inspect
import sys

from setuptools import find_packages

OPS_PYTHON_ROOT = "third_party/iluvatar/python"
OPS_DISCOVERY_ROOT = f"{OPS_PYTHON_ROOT}/triton"
OPS_PACKAGE = "triton.ops"


def _ops_packages():
    return [f"triton.{package}" for package in find_packages(where=OPS_DISCOVERY_ROOT, include=["ops", "ops.*"])]


def get_extra_install_packages():
    return _ops_packages()


def get_package_dir():
    return {package: f"{OPS_PYTHON_ROOT}/{package.replace('.', '/')}" for package in _ops_packages()}


def register_cache(cache, flagtree_backend, check_env, set_llvm_env):
    cache.store(
        file="iluvatar-llvm22-x86_64",
        condition=("iluvatar" == flagtree_backend),
        url="https://baai-cp-web.ks3-cn-beijing.ksyuncs.com/trans/iluvatar-llvm22-x86_64_v0.6.1.tar.gz",
        pre_hook=lambda: check_env("LLVM_SYSPATH"),
        post_hook=set_llvm_env,
    )


def _build_setup_hook():
    patched_attr = "_iluvatar_ops_packages_patched"

    def wrap_setup(original_setup):
        if getattr(original_setup, patched_attr, False):
            return original_setup

        def setup_with_iluvatar_ops(*args, **kwargs):
            packages = list(kwargs.get("packages", []))
            for package in _ops_packages():
                if package not in packages:
                    packages.append(package)
            kwargs["packages"] = packages

            package_dir = dict(kwargs.get("package_dir", {}))
            package_dir.update(get_package_dir())
            kwargs["package_dir"] = package_dir
            return original_setup(*args, **kwargs)

        setattr(setup_with_iluvatar_ops, patched_attr, True)
        setup_with_iluvatar_ops._iluvatar_ops_original_setup = original_setup
        return setup_with_iluvatar_ops

    return wrap_setup


def _patch_setup(wrap_setup):
    patched = False
    frame = inspect.currentframe()
    while frame is not None:
        setup_func = frame.f_globals.get("setup")
        if callable(setup_func):
            frame.f_globals["setup"] = wrap_setup(setup_func)
            patched = True
        frame = frame.f_back

    main_module = sys.modules.get("__main__")
    if main_module is not None and hasattr(main_module, "setup"):
        main_module.setup = wrap_setup(main_module.setup)
        patched = True

    if not patched:
        raise RuntimeError("iluvatar setup hook could not find setup() to patch")


_patch_setup(_build_setup_hook())
