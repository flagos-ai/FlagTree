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

import os
import shutil
from pathlib import Path


def register_cache(cache, flagtree_backend, check_env, set_llvm_env):
    is_sunrise = "sunrise" == flagtree_backend

    def configure_llvm(path):
        set_llvm_env(path)
        sunrise_cp_bc_files(path)

    cache.store(
        file="sunrise_llvm22_dev_release",
        condition=is_sunrise,
        url="https://baai-cp-web.ks3-cn-beijing.ksyuncs.com/trans/llvm-71bd243f-triton-v3.6.x.tar.gz",
        pre_hook=lambda: check_env("LLVM_SYSPATH"),
        post_hook=configure_llvm,
    )
    cache.store(
        file="sunriseTritonPlugin.so",
        condition=is_sunrise and not os.environ.get("FLAGTREE_PLUGIN"),
        url="https://baai-cp-web.ks3-cn-beijing.ksyuncs.com/trans/sunriseTritonPlugin_v0.6.0.4.tar.gz",
        md5_digest="4c77b8c0",
    )


# sunrise
def sunrise_cp_bc_files(path):
    # mkdir -p third_party/sunrise/backend/lib
    lib_dir = Path("third_party/sunrise/backend/lib")
    os.makedirs(lib_dir, exist_ok=True)
    # cp ${LLVM_SYSPATH}/stpu/bitcode/*.bc third_party/sunrise/backend/lib
    bc_dir = Path(path) / "stpu" / "bitcode"
    for bc_file in bc_dir.glob("*.bc"):
        shutil.copy(bc_file, lib_dir)


# pybind11 ABI the prebuilt sunriseTritonPlugin.so was built against. The plugin
# .so hard-encodes a PYBIND11_INTERNALS_VERSION (an embedded __pybind11_internals_v<N>
# symbol); libtriton must be built against the exact pybind11 the plugin was built
# with, otherwise `import triton._C.libtriton` fails or segfaults in the pybind11
# dispatcher.
_PYBIND11_INTERNALS_TO_PIP = {
    5: "pybind11>=2.12,<3.0",
    11: "pybind11==3.0.1",
    12: "pybind11==3.1.0",
}


def _read_pybind11_internals_from_so(plugin_path):
    """Return the PYBIND11_INTERNALS_VERSION embedded in the prebuilt plugin .so."""
    import mmap
    import re
    pat = re.compile(rb"__pybind11_internals_v(\d+)")
    try:
        with open(plugin_path, "rb") as fh, mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            m = pat.search(mm)
            if m:
                return int(m.group(1))
    except (OSError, ValueError):
        pass
    return None


def _installed_pybind11_internals():
    """(internals_version, pybind11_version) for the pybind11 the build will use."""
    import re
    try:
        import pybind11
    except Exception:
        return None, None
    version = getattr(pybind11, "__version__", None)
    hdr = os.path.join(pybind11.get_include(), "pybind11", "detail", "internals.h")
    try:
        m = re.search(r"#\s*define\s+PYBIND11_INTERNALS_VERSION\s+(\d+)", Path(hdr).read_text())
    except OSError:
        return None, version
    return (int(m.group(1)) if m else None), version


def check_pybind11_abi(cache):
    """Verify the env pybind11 ABI matches the prebuilt sunriseTritonPlugin.so."""
    # Under FLAGTREE_PLUGIN the plugin is compiled from source, so its pybind11 ABI
    # matches the env by construction; only the downloaded prebuilt .so needs checking.
    if os.environ.get("FLAGTREE_PLUGIN"):
        return
    try:
        plugin_path = Path(cache.get("sunriseTritonPlugin.so"))
    except KeyError:
        return
    if plugin_path.is_dir():
        scan_paths = sorted(plugin_path.rglob("*.so"))
    elif plugin_path.is_file():
        scan_paths = [plugin_path]
    else:
        scan_paths = []

    required = None
    for path in scan_paths:
        required = _read_pybind11_internals_from_so(path)
        if required is not None:
            break
    if required is None:
        return
    installed, version = _installed_pybind11_internals()

    # The internals version must match, AND the patch release must be the exact one
    # the objects were compiled against (see _PYBIND11_INTERNALS_TO_PIP comment).
    required_pip = _PYBIND11_INTERNALS_TO_PIP.get(required)
    if required_pip is not None and required_pip.startswith("pybind11=="):
        required_version = required_pip[len("pybind11=="):]
        if installed == required and version == required_version:
            print(f"[sunrise] pybind11 ABI OK: env pybind11 {version} (internals v{installed}) "
                  f"matches prebuilt sunriseTritonPlugin.so (internals v{required})")
            return
    elif installed == required:
        print(f"[sunrise] pybind11 ABI OK: env pybind11 {version} (internals v{installed}) "
              f"matches prebuilt sunriseTritonPlugin.so (internals v{required})")
        return

    pip_spec = _PYBIND11_INTERNALS_TO_PIP.get(required)
    detail = (
        f"[sunrise] pybind11 ABI mismatch: prebuilt sunriseTritonPlugin.so requires "
        f"PYBIND11_INTERNALS_VERSION={required}" +
        (f" with pybind11=={required_version}" if required_pip and required_pip.startswith("pybind11==") else "") +
        f", but the environment's pybind11 {version} provides {installed}." +
        " Building against a mismatched pybind11 makes `import " +
        "triton._C.libtriton` fail or segfault in the pybind11 dispatcher " +
        "('Cannot overload existing non-function object ... with a function " + "of the same name').")
    hint = (f" Install a matching pybind11 first, e.g. `pip install '{pip_spec}'`, then rebuild."
            if pip_spec else " No known pybind11 release maps to that internals version.")
    raise RuntimeError(detail + hint)
