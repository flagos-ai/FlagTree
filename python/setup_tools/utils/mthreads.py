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

import json
import os
from pathlib import Path
import re
import shutil
import subprocess

from python.setup_tools.utils.default import FlagCXRegistrar, printinfo


def skip_package_dir(package):
    return package == "triton" or package.startswith("triton.")


def get_package_dir():
    return {}


def relocate_flagcx():
    from python.setup_tools.utils import flagtree_submodules
    from python.setup_tools.utils.tools import flagtree_configs

    dependency = flagtree_submodules.get("flagcx")
    if dependency is None:
        return
    source_dir = os.environ.get("FLAGCX_SOURCE_DIR")
    dependency.dst_path = str(Path(source_dir).expanduser().resolve()) if source_dir else str(
        Path(flagtree_configs.flagtree_submodule_dir) / "mthreads/third_party/flagcx")


def _device_arch(sdk):
    requested = os.environ.get("DEVICE_BITCODE_ARCH", "").strip()
    if requested and requested != "native":
        if not re.fullmatch(r"mp_\d+", requested):
            raise RuntimeError(f"Invalid MUSA DEVICE_BITCODE_ARCH: {requested!r}")
        return requested

    # Prefer an explicit single-arch environment, then probe the installed SDK.
    for tool in (sdk / "bin/musa_arch", sdk / "bin/mcc"):
        if tool.name == "musa_arch" and tool.is_file():
            result = subprocess.run([str(tool)], text=True, capture_output=True, check=False)
            arches = set(re.findall(r"\bmp_\d+\b", result.stdout))
            if result.returncode == 0 and len(arches) == 1:
                return arches.pop()
    try:
        import torch
        if hasattr(torch, "musa") and torch.musa.is_available():
            arches = set()
            for i in range(torch.musa.device_count()):
                props = torch.musa.get_device_properties(i)
                name = getattr(props, "gcnArchName", None) or getattr(props, "name", "")
                match = re.search(r"\bmp_\d+\b", str(name))
                if match:
                    arches.add(match.group(0))
            if len(arches) == 1:
                return arches.pop()
    except (ImportError, AttributeError, RuntimeError):
        pass
    # Default to the FlagCX/MUSA CI architecture (MTT S5000).
    return os.environ.get("MUSA_ARCH", "mp_31")


def _toolchain():
    sdk = Path(os.environ.get("DEVICE_HOME") or os.environ.get("MUSA_HOME") or "/usr/local/musa").expanduser().resolve()
    if not (sdk / "bin/mcc").is_file():
        raise RuntimeError(f"MUSA FlagCX requires {sdk / 'bin/mcc'}. Set DEVICE_HOME to the MUSA SDK.")
    return sdk


class MThreadsFlagCXRegistrar(FlagCXRegistrar):
    """MThreads-specific FlagCX build and installation using the shared runtime layout."""

    def _set_path(self, external):
        self.backend_name = external["configs"].flagtree_backend or "mthreads"
        self.flagcx_src_dir = external["backend"].dst_path
        self.flagtree_dir = external["configs"].flagtree_root_dir
        self.src_lib_dir = Path(self.flagcx_src_dir) / "build/lib"

    def _json_include_dir(self):
        source = Path(self.flagcx_src_dir)
        override = os.environ.get("JSON_INCLUDE_DIR")
        candidates = []
        if override:
            candidates.append(Path(override).expanduser().resolve())
        candidates.append(source / "third-party/json/single_include")
        for candidate in candidates:
            if (candidate / "nlohmann/json.hpp").is_file():
                return candidate
        if override:
            raise RuntimeError(f"nlohmann/json.hpp is missing under JSON_INCLUDE_DIR={override}")
        try:
            subprocess.run(["git", "submodule", "update", "--init", "--", "third-party/json"],
                           cwd=source, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass
        candidate = source / "third-party/json/single_include"
        if not (candidate / "nlohmann/json.hpp").is_file():
            raise RuntimeError("FlagCX needs nlohmann/json.hpp. Install nlohmann-json3-dev, initialize "
                               "FlagCX's third-party/json submodule, or set JSON_INCLUDE_DIR.")
        return candidate

    def get_compile_cmds(self):
        source = Path(self.flagcx_src_dir)
        if os.environ.get("COMPILE_KERNEL", "1") != "1":
            raise RuntimeError("MUSA FlagCX device bitcode requires COMPILE_KERNEL=1.")
        if not (source / "bindings/ir/musa/Makefile").is_file():
            raise RuntimeError(f"{source} lacks the MUSA bitcode build. Use the shared pinned FlagCX revision, "
                               "or set FLAGCX_SOURCE_DIR to a compatible checkout.")
        sdk = _toolchain()
        self.arch = _device_arch(sdk)
        build_dir = source / "build/mthreads" / self.arch
        self.src_lib_dir = build_dir / "lib"
        json_dir = self._json_include_dir()
        cmd = [
            "make",
            "-j",
            os.environ.get("MAX_JOBS", "8"),
            "USE_MUSA=1",
            "COMPILE_KERNEL=1",
            f"DEVICE_HOME={sdk}",
            f"CCL_HOME={os.environ.get('CCL_HOME') or sdk}",
            f"DEVICE_BITCODE_ARCH={self.arch}",
            f"BUILDDIR={build_dir}",
            f"JSON_INCLUDE_DIR={json_dir}",
        ]
        for key in ("HOST_COMPILER", "USE_UCX", "UCX_HOME", "USE_IBUC", "USE_MPI", "MPI_HOME", "MUSA_ARCH"):
            if os.environ.get(key):
                cmd.append(f"{key}={os.environ[key]}")
        cmd.append("device_bitcode")
        return {self.shared_lib_name: cmd, self.bitcode_name: cmd}

    def _compile_and_cache(self):
        cmds = self.get_compile_cmds()
        for cmd in dict.fromkeys(tuple(cmd) for cmd in cmds.values()):
            printinfo(f"Compiling MUSA FlagCX in {self.flagcx_src_dir}")
            subprocess.run(list(cmd), cwd=self.flagcx_src_dir, check=True)
        for name in cmds:
            source = self.src_lib_dir / name
            if not source.is_file():
                raise FileNotFoundError(f"Expected FlagCX build artifact not found: {source}")
        for name in cmds:
            destination = self._get_runtime_path(name)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(self.src_lib_dir / name, destination)
        metadata = {"backend": self.backend_name, "arch": self.arch}
        path = self._get_runtime_path(self.bitcode_name).parent / "metadata.json"
        path.write_text(json.dumps(metadata, indent=2) + "\n")

    def _get_required_src_files_paths(self):
        whl_path = self._get_flagcx_wheel_path()
        if whl_path is None:
            source = Path(self.flagcx_src_dir)
            wrappers = (source / "plugin/interservice/flagcx_wrapper.py", source / "src/flagcx/api.py")
            wrapper = next((path for path in wrappers if path.is_file()), wrappers[0])
            return {"wrapper": wrapper, "include": self.src_lib_dir.parent / "include"}
        include = whl_path / "include"
        if not include.is_dir():
            include = whl_path / "lib" / "include"
        return {"wrapper": whl_path / "api.py", "include": include}

    def _copy_required_files(self):
        src_paths = self._get_required_src_files_paths()
        dst_paths = self._get_required_dst_files_paths()
        wrapper_src = src_paths["wrapper"]
        include_src = src_paths["include"]
        if not wrapper_src.is_file():
            raise FileNotFoundError(f"FlagCX Python wrapper not found: {wrapper_src}")
        manifest = include_src / ".flagcx_device_headers"
        if manifest.is_file():
            names = manifest.read_text().splitlines()
            if not names or any(not name or Path(name).name != name or not name.endswith(".h") for name in names):
                raise RuntimeError(f"Invalid or empty FlagCX header manifest: {manifest}")
            headers = [include_src / name for name in names]
            for header in headers:
                if not header.is_file():
                    raise FileNotFoundError(f"FlagCX consumer header not found: {header}")
            for wrapper_dst in dst_paths["wrapper"]:
                wrapper_dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(wrapper_src, wrapper_dst)
                printinfo(f"flagcx_wrapper.py copied from {wrapper_src} to {wrapper_dst}")
            for include_dst in dst_paths["include"]:
                if include_dst.exists():
                    shutil.rmtree(include_dst)
                include_dst.mkdir(parents=True)
                for header in headers:
                    shutil.copy2(header, include_dst / header.name)
                printinfo(f"FlagCX headers copied from {include_src} to {include_dst}")
            return
        # Wheel/source trees without a header manifest: copy the full include tree.
        super()._copy_required_files()


def handle_flagcx(*args, **kwargs):
    MThreadsFlagCXRegistrar(kwargs).run()


def get_package_data(backends):
    return {
        "triton._C": ["libflagcx.so"],
        "triton.backends.mthreads": ["lib/libflagcx_device.bc", "lib/metadata.json"],
        "triton.experimental.tle.language": ["include/*.h"],
    }


def register_cache(cache, flagtree_backend, check_env, set_llvm_env):
    is_mthreads = "mthreads" == flagtree_backend
    cache.store(
        file="mthreads-llvm22",
        condition=is_mthreads,
        url="https://baai-cp-web.ks3-cn-beijing.ksyuncs.com/trans/mthreads-llvm22-x64_v0.7.1.tar.gz",
        pre_hook=lambda: check_env("LLVM_SYSPATH"),
        post_hook=set_llvm_env,
    )
    cache.store(
        file="mthreads_local_binary",
        condition=is_mthreads,
        url="https://baai-cp-web.ks3-cn-beijing.ksyuncs.com/trans/mthreads_local_binary_v0.7.1.tar.gz",
    )
    cache.store(
        files=("ld.lld", "llc"),
        condition=is_mthreads,
        copy_src_path=f"{cache.dir_path}/{flagtree_backend}/mthreads_local_binary",
        copy_dst_path=f"third_party/{flagtree_backend}/backend/bin",
    )
