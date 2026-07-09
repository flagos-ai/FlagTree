#
# Copyright 2024 Enflame. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
import os
from pathlib import Path
import subprocess
from typing import Any

device_name = "gcu"
datadir = os.getenv("TRITON_GCU_PATH") or "/opt/triton_gcu"

if not os.path.exists(datadir):
    raise Exception("Cannot find data directory in " + datadir)

TOOLKIT_PATH = os.path.join(datadir, "bin")
RUNTIME_PATH = os.path.join(datadir, "lib")

PY_TOOLS_PATH = Path(__file__).parent


# toolkit
def _run_command(cmd, content, *args):
    if not isinstance(content, str):
        content = str(content)
    result = subprocess.run([os.path.join(TOOLKIT_PATH, cmd)] + list(args), input=content, capture_output=True,
                            text=True, encoding="utf-8")
    if result.returncode != 0:
        raise Exception(result.stderr)
    # print(__file__, "run command: \n", [os.path.join(TOOLKIT_PATH, cmd)] + list(args))
    # print subprocess std::cerr << log to terminator
    print(result.stderr)
    return result.stdout


def _run_command2(cmd, content, *args):
    if not isinstance(content, str):
        content = str(content)
    result = subprocess.run([PY_TOOLS_PATH / cmd] + list(args), input=content, capture_output=True, text=True,
                            encoding="utf-8")
    if result.returncode != 0:
        raise Exception(result.stderr)
    # print(__file__, "run command: \n", [os.path.join(PY_TOOLS_PATH, cmd)] + list(args))
    # print(subprocess.stderr)
    print(result.stderr)
    return result.stdout


# Module-level cache to avoid reloading .so files for each arch
_gcu_opt_module_cache: dict[str, Any] = {}

# ===== GCU500 Target Configuration =====


def get_gcu500_target_constants():
    """Query EFGCU target triple/mcpu from the C++ backend.
    Returns: (triple, mcpu) tuple
    Falls back to hardcoded values when extension is not available.
    """
    try:
        gcu500 = _load_gcu_opt_module('gcu500')
        if gcu500 is not None:
            return gcu500.TARGET_TRIPLE, gcu500.TARGET_MCPU
    except (ImportError, AttributeError):
        pass
    return "efgcu-enflame-tops", "efgcu500"


def get_gcu500_triple():
    """Get EFGCU target triple."""
    return get_gcu500_target_constants()[0]


def get_gcu500_mcpu():
    """Get EFGCU target mcpu."""
    return get_gcu500_target_constants()[1]


def get_tops_home():
    """Get TOPS installation home directory."""
    return os.environ.get("TOPS_HOME", "/opt/tops")


def resolve_gcu500_tool(name: str) -> str:
    """Resolve the full path to a GCU500 tool (llc, lld, clang-offload-bundler, etc.).

    Search order:
    1. Environment variable TRITON_ENFLAME_{TOOL}_PATH
    2. Bundled in wheel: python/triton/bin/{tool}
    3. TOPS LLVM18: $TOPS_LLVM18_PATH/bin/{tool}
    4. TOPS bin: $TOPS_HOME/bin/{tool}

    Raises FileNotFoundError if tool not found.
    """
    env_key = f"TRITON_ENFLAME_{name.upper().replace('-', '_')}_PATH"
    env_path = os.environ.get(env_key)
    if env_path and os.path.isfile(env_path):
        return env_path

    bundled = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bin", name)
    if os.path.isfile(bundled):
        return bundled

    system = os.path.join(get_tops_home(), "bin", name)
    if os.path.isfile(system):
        return system

    tops_bin = os.path.join(get_tops_home(), "bin", name)
    if os.path.isfile(tops_bin):
        return tops_bin

    raise FileNotFoundError(f"Cannot find '{name}'. Set {env_key}, install TOPS LLVM18, "
                            f"or rebuild the wheel on a system with TOPS installed.")


# ===== GCU Pipeline API =====


class Pipeline:
    """Pure-Python wrapper around the C pipeline_* functions.

    Uses flat functions (no custom Python types) so that Python's GC
    never traverses C-owned memory.
    """

    def __init__(self, mod):
        self._mod = mod
        self._handle = mod.pipeline_create()

    def add_pass(self, name, options=''):
        self._mod.pipeline_add_pass(self._handle, name, options)

    def run(self, input_ir):
        return self._mod.pipeline_run(self._handle, input_ir)

    def __del__(self):
        if getattr(self, '_handle', 0):
            self._mod.pipeline_destroy(self._handle)
            self._handle = 0


def get_gcu_pipeline_class(arch):
    """Get a factory that creates Pipeline instances for the given arch.

    Returns a callable (not a class) so existing call sites still work:
        PipelineClass = get_gcu_pipeline_class('gcu300')
        pipeline = PipelineClass()   # -> Pipeline wrapping gcu300
    """
    mod = _load_gcu_opt_module(arch)

    def _factory():
        return Pipeline(mod)

    return _factory


def _load_gcu_opt_module(arch):
    """Load the in-process pybind11 .so for the given arch (cached)."""
    if arch in _gcu_opt_module_cache:
        return _gcu_opt_module_cache[arch]

    _dir = os.path.dirname(__file__)
    import ctypes, glob as _g, importlib.util as _iu, sysconfig as _sc
    tag = arch  # gcu300, gcu400, gcu500
    for _core in sorted(_g.glob(os.path.join(_dir, f'libtriton_{tag}_core.so*'))):
        ctypes.CDLL(_core, mode=ctypes.RTLD_GLOBAL)
        break
    try:
        mod = __import__(f'triton_gcu.triton._triton_{tag}', fromlist=[f'_triton_{tag}'])
        _gcu_opt_module_cache[arch] = mod
        return mod
    except ImportError:
        _ext_suffix = _sc.get_config_var('EXT_SUFFIX') or ''
        _exact = os.path.join(_dir, f'_triton_{tag}{_ext_suffix}')
        if os.path.isfile(_exact):
            spec = _iu.spec_from_file_location(f'_triton_{tag}', _exact, submodule_search_locations=[])
            mod = _iu.module_from_spec(spec)
            spec.loader.exec_module(mod)
            _gcu_opt_module_cache[arch] = mod
            return mod
        for _so in sorted(_g.glob(os.path.join(_dir, f'_triton_{tag}*.so'))):
            if '_core' in os.path.basename(_so):
                continue
            try:
                spec = _iu.spec_from_file_location(f'_triton_{tag}', _so, submodule_search_locations=[])
                mod = _iu.module_from_spec(spec)
                spec.loader.exec_module(mod)
                _gcu_opt_module_cache[arch] = mod
                return mod
            except ImportError:
                continue
        raise


def triton_gcu_opt(content, *args, arch):
    passes = ["-mlir-print-op-generic"] + list(args)
    if arch == "gcu410":
        arch = "gcu400"

    # Prefer in-process .so (no subprocess overhead).
    try:
        mod = _load_gcu_opt_module(arch)
        return mod.run_opt(str(content) if not isinstance(content, str) else content, passes)
    except (ImportError, AttributeError):
        pass

    return _run_command2(f"triton-{arch}-opt", content, *passes)


def gcu_compiler_opt(content, *args):
    passes = ["-mlir-print-op-generic"] + list(args)
    return _run_command("gcu-compiler-opt", content, *passes)


def compile(content, *args):
    return _run_command("gcu-compiler-compile", content, *args)


# Return the boolean value of an environment variable.
#
# Helpful environment variables:
#
# - "MLIR_ENABLE_DUMP=1` dumps the IR before every MLIR pass Triton runs and
# the IR after every MLIR pass GCU runs.
def get_bool_env(env, defaultValue=False):
    s = os.getenv(env, "").lower()
    if (s == "1" or s == "true" or s == "on"):
        return True
    if (s == "0" or s == "false" or s == "off"):
        return False
    return defaultValue


def compile_llir_to_fatbin_gcu500(llir_str: str, kernel_name: str = "kernel") -> dict[str, bytes | str | None]:
    """Compile LLVM IR to fatbin using EFGCU toolchain (llc + lld + bundler).

    Args:
        llir_str: LLVM IR text (must be already patched for TOPS compatibility)
        kernel_name: Kernel name for debug output

    Returns:
        Dictionary with 'fatbin' (bytes) and 'asm' (str | None) keys

    Pipeline:
        1. llc: LLVM IR -> .o + .s
        2. llvm-objcopy: strip .eh_frame sections
        3. lld: .o -> .so
        4. clang-offload-bundler: .so -> .fatbin
    """
    import tempfile
    import subprocess
    import shutil

    triple = get_gcu500_triple()
    mcpu = get_gcu500_mcpu()

    llc_bin = resolve_gcu500_tool("llc")
    lld_bin = resolve_gcu500_tool("lld")
    bundler_bin = resolve_gcu500_tool("clang-offload-bundler")

    with tempfile.TemporaryDirectory() as tmpdir:
        ll_path = os.path.join(tmpdir, "kernel.ll")
        obj_path = os.path.join(tmpdir, "kernel.o")
        asm_path = os.path.join(tmpdir, "kernel.s")
        so_path = os.path.join(tmpdir, "kernel.so")
        bin_path = os.path.join(tmpdir, "kernel.fatbin")

        with open(ll_path, "w", encoding="utf-8") as f:
            f.write(llir_str)

        llc_common_flags = [
            llc_bin,
            f"-mtriple={triple}",
            f"-mcpu={mcpu}",
            "-O3",
            "--relocation-model=pic",
            "-vectorize-slp=false",
            "--vectorize-loops=false",
            "-mattr=multi-addrspaces",
        ]

        result = subprocess.run(
            llc_common_flags + ["--filetype=obj", "-o", obj_path, ll_path],
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
        if result.returncode != 0:
            raise RuntimeError(f"llc (gcu500) failed:\n{result.stderr}")
        if result.stderr:
            print(result.stderr)

        result_asm = subprocess.run(
            llc_common_flags + ["--filetype=asm", "-o", asm_path, ll_path],
            capture_output=True,
            text=True,
            encoding="utf-8",
        )

        tops_objcopy = os.path.join(get_tops_home(), "bin", "llvm-objcopy")
        objcopy = tops_objcopy if os.path.isfile(tops_objcopy) else (shutil.which("llvm-objcopy")
                                                                     or shutil.which("objcopy"))
        if objcopy:
            subprocess.run(
                [objcopy, "--remove-section=.eh_frame", "--remove-section=.rela.eh_frame", obj_path],
                capture_output=True,
                text=True,
                encoding="utf-8",
            )

        lld_args = [
            lld_bin,
            "-flavor",
            "gnu",
            "-o",
            so_path,
            obj_path,
            "--Ttext=0",
            "-shared",
            "-z",
            "notext",
            "--entry=0",
        ]

        result = subprocess.run(
            lld_args,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
        if result.returncode != 0:
            raise RuntimeError(f"lld (gcu500 link) failed:\n{result.stderr}")

        target_str = "tops-dtu-enflame-tops--efgcu500"
        result = subprocess.run(
            [
                bundler_bin, "-type=o", f"-output={bin_path}", f"-targets=host-x86_64-unknown-linux,{target_str}",
                f"-input=/dev/null", f"-input={so_path}"
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
        if result.returncode != 0:
            raise RuntimeError(f"clang-offload-bundler failed:\n{result.stderr}")

        asm_content = None
        if result_asm.returncode == 0 and os.path.isfile(asm_path):
            with open(asm_path, "r", encoding="utf-8") as f:
                asm_content = f.read()

        with open(bin_path, "rb") as f:
            fatbin = f.read()

        return {"fatbin": fatbin, "asm": asm_content}
