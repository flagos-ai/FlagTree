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


_LLVM_TOOLS = ("opt", "llvm-as", "llvm-dis")


def relocate_flagcx():
    from python.setup_tools.utils import flagtree_submodules
    from python.setup_tools.utils.tools import flagtree_configs

    dependency = flagtree_submodules.get("flagcx")
    if dependency is None:
        return
    source_dir = os.environ.get("FLAGCX_SOURCE_DIR")
    dependency.dst_path = str(Path(source_dir).expanduser().resolve()) if source_dir else str(
        Path(flagtree_configs.flagtree_submodule_dir) / "hcu/third_party/flagcx")


def _device_arch(sdk):
    requested = os.environ.get("DEVICE_BITCODE_ARCH", "").strip()
    if requested and requested != "native":
        if not re.fullmatch(r"gfx[0-9a-f]+", requested):
            raise RuntimeError(f"Invalid HCU DEVICE_BITCODE_ARCH: {requested!r}")
        return requested

    for tool in (sdk / "aillvm/bin/amdgpu-arch", sdk / "llvm/bin/amdgpu-arch", sdk / "bin/amdgpu-arch"):
        if tool.is_file():
            result = subprocess.run([str(tool)], text=True, capture_output=True, check=False)
            arches = set(re.findall(r"\bgfx[0-9a-f]+\b", result.stdout))
            if result.returncode == 0 and len(arches) == 1:
                return arches.pop()
    try:
        import torch
        if torch.version.hip and torch.cuda.is_available():
            arches = {
                torch.cuda.get_device_properties(i).gcnArchName.split(":")[0]
                for i in range(torch.cuda.device_count())
            }
            if len(arches) == 1:
                return arches.pop()
    except (ImportError, AttributeError, RuntimeError):
        pass
    raise RuntimeError("Cannot determine the HCU GPU architecture. Set DEVICE_BITCODE_ARCH explicitly "
                       "(for example gfx936 on BW1000); cross builds must name the deployment GPU.")


def _llvm_major(executable):
    output = subprocess.check_output([str(executable), "--version"], text=True, stderr=subprocess.STDOUT)
    match = re.search(r"(?:clang|LLVM) version\s+(\d+)", output)
    if not match:
        raise RuntimeError(f"Cannot determine the LLVM version of {executable}")
    return int(match.group(1))


def _llvm_executables(directory, name):
    versioned = sorted(directory.glob(f"{name}-[0-9]*"))
    for executable in (directory / name, *versioned):
        if executable.is_file() and os.access(executable, os.X_OK):
            yield executable


def _llvm_config_bindirs(directories):
    for directory in dict.fromkeys(directories):
        for config in _llvm_executables(directory, "llvm-config"):
            try:
                output = subprocess.check_output([str(config), "--bindir"], text=True, stderr=subprocess.DEVNULL)
            except (OSError, subprocess.CalledProcessError):
                continue
            bindir = Path(output.strip())
            if bindir.is_absolute():
                yield bindir.resolve()


def _find_llvm_tools(directories, compiler_major, *, allow_split=False):
    suites = {}
    split_suites = {}
    for directory in dict.fromkeys(directories):
        for name in _LLVM_TOOLS:
            for executable in _llvm_executables(directory, name):
                try:
                    major = _llvm_major(executable)
                except (OSError, subprocess.CalledProcessError, RuntimeError):
                    continue
                if major < compiler_major:
                    continue
                suites.setdefault((major, directory), {}).setdefault(name, executable)
                if allow_split:
                    split_suites.setdefault(major, {}).setdefault(name, executable)

    # Prefer a complete suite from one directory before combining tools from different paths.
    compatible = [(major, tools) for (major, _), tools in suites.items() if len(tools) == len(_LLVM_TOOLS)]
    if not compatible:
        compatible = [(major, tools) for major, tools in split_suites.items() if len(tools) == len(_LLVM_TOOLS)]
    if compatible:
        # A nearby reader version minimizes bitcode format changes. The DU build
        # verifies the processed IR before FlagTree links it into a kernel.
        return min(compatible, key=lambda entry: entry[0])[1]
    return None


def _toolchain():
    sdk = Path(os.environ.get("DEVICE_HOME") or os.environ.get("HIP_PATH") or "/opt/dtk").expanduser().resolve()
    if not (sdk / "bin/hipcc").is_file():
        raise RuntimeError(f"HCU FlagCX requires {sdk / 'bin/hipcc'}. Set DEVICE_HOME to the DTK installation.")

    compiler_major = _llvm_major(sdk / "bin/hipcc")
    override = os.environ.get("FLAGCX_HCU_LLVM_BIN")
    if override:
        directory = Path(override).expanduser().resolve()
        tools = _find_llvm_tools((directory,), compiler_major)
        searched = f"FLAGCX_HCU_LLVM_BIN={directory}"
    else:
        dtk_dirs = (sdk / "llvm/bin", sdk / "aillvm/bin", sdk / "bin")
        tools = _find_llvm_tools(dtk_dirs, compiler_major)
        if tools is None:
            path_dirs = tuple(dict.fromkeys(Path(entry).resolve() for entry in os.get_exec_path()))
            config_dirs = _llvm_config_bindirs((*dtk_dirs, *path_dirs))
            tools = _find_llvm_tools((*dtk_dirs, *config_dirs, *path_dirs), compiler_major, allow_split=True)
        searched = f"the DTK at {sdk}, llvm-config, and PATH"
    if tools is None:
        raise RuntimeError("Cannot find opt, llvm-as and llvm-dis with the same LLVM major version "
                           f">= {compiler_major} for HCU FlagCX. Checked {searched}. "
                           "Set FLAGCX_HCU_LLVM_BIN to a compatible LLVM bin directory.")
    return sdk, tools


class HCUFlagCXRegistrar(FlagCXRegistrar):
    """HCU-specific FlagCX build and installation using the shared runtime layout."""

    def _set_path(self, external):
        # HCU uses make's incremental build tree, not the common artifact cache.
        self.backend_name = external["configs"].flagtree_backend or "hcu"
        self.flagcx_src_dir = external["backend"].dst_path
        self.flagtree_dir = external["configs"].flagtree_root_dir
        self.src_lib_dir = Path(self.flagcx_src_dir) / "build/lib"

    def _json_include_dir(self):
        """Find the header-only JSON dependency, initializing it if necessary."""
        source = Path(self.flagcx_src_dir)
        override = os.environ.get("JSON_INCLUDE_DIR")
        candidates = [Path(override).expanduser()] if override else [
            source / "third-party/json/single_include",
            Path("/usr/include"),
        ]
        for candidate in candidates:
            if (candidate / "nlohmann/json.hpp").is_file():
                return candidate.resolve()
        if override:
            raise RuntimeError(f"nlohmann/json.hpp is missing under JSON_INCLUDE_DIR={override}")
        if (source / ".git").exists():
            subprocess.run(["git", "submodule", "update", "--init", "--", "third-party/json"], cwd=source, check=True)
        candidate = source / "third-party/json/single_include"
        if not (candidate / "nlohmann/json.hpp").is_file():
            raise RuntimeError("FlagCX needs nlohmann/json.hpp. Install nlohmann-json3-dev, initialize "
                               "FlagCX's third-party/json submodule, or set JSON_INCLUDE_DIR.")
        return candidate.resolve()

    def get_compile_cmds(self):
        source = Path(self.flagcx_src_dir)
        if os.environ.get("COMPILE_KERNEL", "1") != "1":
            raise RuntimeError("HCU FlagCX device bitcode requires COMPILE_KERNEL=1.")
        if not (source / "bindings/ir/du/Makefile").is_file():
            raise RuntimeError(f"{source} lacks the Hygon bitcode build. Use the shared pinned FlagCX revision, "
                               "or set FLAGCX_SOURCE_DIR to a compatible checkout.")
        sdk, llvm_tools = _toolchain()
        self.arch = _device_arch(sdk)
        json_dir = self._json_include_dir()
        build_dir = source / "build/hcu" / self.arch
        self.src_lib_dir = build_dir / "lib"
        cmd = [
            "make",
            "-j",
            os.environ.get("MAX_JOBS", "8"),
            "USE_DU=1",
            "COMPILE_KERNEL=1",
            f"DEVICE_HOME={sdk}",
            f"CCL_HOME={os.environ.get('CCL_HOME') or sdk}",
            f"DEVICE_BITCODE_ARCH={self.arch}",
            f"DTK_LLVM_BIN={llvm_tools['opt'].parent}",
            f"OPT={llvm_tools['opt']}",
            f"LLVM_AS={llvm_tools['llvm-as']}",
            f"LLVM_DIS={llvm_tools['llvm-dis']}",
            f"BUILDDIR={build_dir}",
            f"JSON_INCLUDE_DIR={json_dir}",
        ]
        for key in ("HOST_COMPILER", "USE_UCX", "UCX_HOME", "USE_IBUC", "USE_MPI", "MPI_HOME"):
            if os.environ.get(key):
                cmd.append(f"{key}={os.environ[key]}")
        cmd.append("device_bitcode")
        # Both artifacts are built together; the HCU build hook runs this target once.
        return {self.shared_lib_name: cmd, self.bitcode_name: cmd}

    def _compile_and_cache(self):
        # Always invoke make so DTK/build changes are checked without reusing
        # the common cache, which does not identify the HCU configuration.
        cmds = self.get_compile_cmds()
        for cmd in dict.fromkeys(tuple(cmd) for cmd in cmds.values()):
            printinfo(f"Compiling HCU FlagCX in {self.flagcx_src_dir}")
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
        return {"wrapper": whl_path / "api.py", "include": whl_path / "include"}

    def _copy_required_files(self):
        src_paths = self._get_required_src_files_paths()
        dst_paths = self._get_required_dst_files_paths()
        wrapper_src = src_paths["wrapper"]
        include_src = src_paths["include"]
        if not wrapper_src.is_file():
            raise FileNotFoundError(f"FlagCX Python wrapper not found: {wrapper_src}")
        manifest = include_src / ".flagcx_device_headers"
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


def handle_flagcx(*args, **kwargs):
    HCUFlagCXRegistrar(kwargs).run()


def get_package_data(backends):
    return {
        "triton._C": ["libflagcx.so"],
        "triton.backends.hcu": ["lib/libflagcx_device.bc", "lib/metadata.json"],
        "triton.experimental.tle.language": ["include/*.h"],
    }


def register_cache(cache, flagtree_backend, check_env, set_llvm_env):
    cache.store(
        file="hcu-llvm22-188a510-glibc2.35-glibcxx3.4.30-ubuntu-x86_64",
        condition=("hcu" == flagtree_backend),
        url=("https://baai-cp-web.ks3-cn-beijing.ksyuncs.com/trans/"
             "hcu-llvm22-188a510-glibc2.35-glibcxx3.4.30-ubuntu-x86_64.tar.gz"),
        pre_hook=lambda: check_env("LLVM_SYSPATH"),
        post_hook=set_llvm_env,
    )
    cache.store(file="libdushmem_device_gfx936.bc", condition=("hcu" == flagtree_backend),
                url="https://baai-cp-web.ks3-cn-beijing.ksyuncs.com/trans/"
                "libdushmem_device_gfx936_bc.tar.gz", copy_dst_path="/opt/dtk/dushmem/lib/", md5_digest="f6260239")


def install_extension(*args, **kargs):
    return
