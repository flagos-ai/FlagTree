import shutil
from pathlib import Path

from setuptools import find_packages

MTHREADS_PYTHON_ROOT = "third_party/mthreads/python"
FLAGTREE_PYTHON_ROOT = "python"
TLE_PACKAGE = "triton.experimental.tle"


def skip_package_dir(package):
    return package == "triton" or package.startswith("triton.")


def get_package_dir():
    return {
        "": MTHREADS_PYTHON_ROOT,
    }


def _is_backend_package(package):
    return package == "triton.backends" or package.startswith("triton.backends.")


def _is_language_extra_package(package):
    return package == "triton.language.extra" or package.startswith("triton.language.extra.")


def _merge_mthreads_packages(existing_packages):
    packages = []
    seen = set()

    def add(package):
        if package not in seen:
            packages.append(package)
            seen.add(package)

    for package in find_packages(where=MTHREADS_PYTHON_ROOT, include=["triton", "triton.*"]):
        add(package)

    for package in find_packages(where=FLAGTREE_PYTHON_ROOT, include=[TLE_PACKAGE, f"{TLE_PACKAGE}.*"]):
        add(package)

    for package in existing_packages:
        if (not package.startswith("triton.") or _is_backend_package(package) or _is_language_extra_package(package)
                or package == "triton.profiler" or package.startswith("triton.profiler.")):
            add(package)

    return packages


def _merge_mthreads_package_dir(existing_package_dir):
    package_dir = dict(existing_package_dir or {})
    package_dir[""] = MTHREADS_PYTHON_ROOT

    for package in find_packages(where=MTHREADS_PYTHON_ROOT, include=["triton", "triton.*"]):
        rel_package_path = package.replace(".", "/")
        package_dir[package] = f"{MTHREADS_PYTHON_ROOT}/{rel_package_path}"

    for package in find_packages(where=FLAGTREE_PYTHON_ROOT, include=[TLE_PACKAGE, f"{TLE_PACKAGE}.*"]):
        rel_package_path = package.replace(".", "/")
        package_dir[package] = f"{FLAGTREE_PYTHON_ROOT}/{rel_package_path}"

    return package_dir


def _patch_mthreads_cmdclass(existing_cmdclass):
    cmdclass = dict(existing_cmdclass or {})
    original_build_py = cmdclass.get("build_py")
    if original_build_py is None:
        return cmdclass

    class MthreadsBuildPy(original_build_py):

        def run(self):
            self.force = True
            build_triton_dir = Path(self.build_lib) / "triton"
            if build_triton_dir.exists():
                shutil.rmtree(build_triton_dir)
            return super().run()

    cmdclass["build_py"] = MthreadsBuildPy
    return cmdclass


def apply_mthreads_setup_args(kwargs):
    """Apply mthreads package overrides to setup() kwargs.
    Called from setup.py when FLAGTREE_BACKEND=mthreads and not editable install."""
    kwargs["packages"] = _merge_mthreads_packages(kwargs.get("packages", []))
    kwargs["package_dir"] = _merge_mthreads_package_dir(kwargs.get("package_dir", {}))
    kwargs["cmdclass"] = _patch_mthreads_cmdclass(kwargs.get("cmdclass", {}))
    return kwargs
