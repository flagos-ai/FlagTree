import inspect
import os
import shutil
import sys
from pathlib import Path

from setuptools import find_packages

FLAGTREE_BACKEND = os.environ.get("FLAGTREE_BACKEND", "iluvatar")
PYTHON_ROOT = f"third_party/{FLAGTREE_BACKEND}/python"
FLAGTREE_PYTHON_ROOT = "python"
TLE_PACKAGE = "triton.experimental.tle"
SKIP_BACKEND_PACKAGES_IN_PACKAGE_DIR = True


def _is_backend_package(package):
    return package == "triton.backends" or package.startswith("triton.backends.")


def _is_language_extra_package(package):
    return package == "triton.language.extra" or package.startswith("triton.language.extra.")


def _build_setup_hooks(backend, python_root, skip_backend_packages_in_package_dir=False):
    patched_attr = f"_{backend}_python_root_patched"

    def merge_packages(existing_packages):
        packages = []
        seen = set()

        def add(package):
            if package not in seen:
                packages.append(package)
                seen.add(package)

        for package in find_packages(where=python_root, include=["triton", "triton.*"]):
            add(package)

        for package in find_packages(where=FLAGTREE_PYTHON_ROOT, include=[TLE_PACKAGE, f"{TLE_PACKAGE}.*"]):
            add(package)

        for package in existing_packages:
            if (not package.startswith("triton.") or _is_backend_package(package) or _is_language_extra_package(package)
                    or package == "triton.profiler" or package.startswith("triton.profiler.")):
                add(package)

        return packages

    def merge_package_dir(existing_package_dir):
        package_dir = dict(existing_package_dir or {})
        package_dir[""] = python_root

        for package in find_packages(where=python_root, include=["triton", "triton.*"]):
            if skip_backend_packages_in_package_dir and package.startswith("triton.backends."):
                continue
            rel_package_path = package.replace(".", "/")
            package_dir[package] = f"{python_root}/{rel_package_path}"

        for package in find_packages(where=FLAGTREE_PYTHON_ROOT, include=[TLE_PACKAGE, f"{TLE_PACKAGE}.*"]):
            rel_package_path = package.replace(".", "/")
            package_dir[package] = f"{FLAGTREE_PYTHON_ROOT}/{rel_package_path}"

        return package_dir

    def patch_cmdclass(existing_cmdclass):
        cmdclass = dict(existing_cmdclass or {})
        original_build_py = cmdclass.get("build_py")
        if original_build_py is None:
            return cmdclass

        class BackendBuildPy(original_build_py):

            def run(self):
                self.force = True
                build_triton_dir = Path(self.build_lib) / "triton"
                if build_triton_dir.exists():
                    shutil.rmtree(build_triton_dir)
                return super().run()

        cmdclass["build_py"] = BackendBuildPy
        return cmdclass

    def wrap_setup(original_setup):
        if getattr(original_setup, patched_attr, False):
            return original_setup

        def setup_with_backend_python_root(*args, **kwargs):
            kwargs["packages"] = merge_packages(kwargs.get("packages", []))
            kwargs["package_dir"] = merge_package_dir(kwargs.get("package_dir", {}))
            kwargs["cmdclass"] = patch_cmdclass(kwargs.get("cmdclass", {}))
            return original_setup(*args, **kwargs)

        setattr(setup_with_backend_python_root, patched_attr, True)
        setup_with_backend_python_root._backend_python_root_original_setup = original_setup
        return setup_with_backend_python_root

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
        raise RuntimeError(f"{FLAGTREE_BACKEND} setup hook could not find setup() to patch "
                           f"(python root: {PYTHON_ROOT})")


_wrap_setup = _build_setup_hooks(
    FLAGTREE_BACKEND,
    PYTHON_ROOT,
    skip_backend_packages_in_package_dir=SKIP_BACKEND_PACKAGES_IN_PACKAGE_DIR,
)
_patch_setup(_wrap_setup)
