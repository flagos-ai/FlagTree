from __future__ import annotations
import os
import re
import shlex
import struct
from pathlib import Path
import subprocess
from typing import Any, Final

import torch
import tempfile
import signal
from triton import knobs
from triton.runtime.errors import PTXASError
from functools import partial

from triton._C.libtriton import llvm  # pyright: ignore[reportMissingImports]
from triton._C.libtriton.tle.llvm import parse_llvm_ir  # pyright: ignore[reportMissingImports]
from triton.experimental.tle.raw.source_store import register_source

# TODO: We use cli tools to compile CUDA code temporarily, and plan to replace it with LLVM components Python bindings in the future.
CLANG = os.getenv("CLANG", "clang")
CLANG_FLAGS = shlex.split(os.getenv("CLANG_FLAGS", ""))

NVCC = os.getenv("NVCC", "nvcc")
NVCC_FLAGS = shlex.split(os.getenv("NVCC_FLAGS", ""))

PTXAS = os.getenv("PTXAS", "ptxas")

NVLINK = os.getenv("NVLINK", "nvlink")
NVLINK_FLAGS = shlex.split(os.getenv("NVLINK_FLAGS", ""))

OPT = os.getenv("OPT", "opt")


def _sanitize_clang_ir(ir: str) -> str:
    # Newer clang emits attributes that this Triton branch's LLVM parser does
    # not understand yet. They are not needed by TLE raw device function import.
    ir = ir.replace(" nocreateundeforpoison", "")
    ir = ir.replace(" contract", "")

    def _replace_hex_float(match: re.Match[str]) -> str:
        hex_digits = match.group(1)
        bits = int(hex_digits, 16)
        if len(hex_digits) == 16:
            value = struct.unpack("!d", bits.to_bytes(8, byteorder="big"))[0]
        elif len(hex_digits) == 8:
            value = struct.unpack("!f", bits.to_bytes(4, byteorder="big"))[0]
        else:
            return match.group(0)
        return repr(value)

    return re.sub(r"f0x([0-9A-Fa-f]+)", _replace_hex_float, ir)


def _get_cuda_gpu_arch() -> str:
    arch = os.getenv("TLE_CUDA_ARCH")
    if arch:
        return f"--cuda-gpu-arch={arch}"
    major, minor = torch.cuda.get_device_capability()
    suffix = "a" if major >= 9 else ""
    return f"--cuda-gpu-arch=sm_{major}{minor}{suffix}"


def make_cubin_inspection_hook(cuda_self, triton_self, stages, options, language, capability):

    def make_cubin(self, src, metadata, opt, capability):
        fsrc_cuda = cuda_self.source_file
        arch = _get_cuda_gpu_arch().split('=')[1]
        fbin_cuda = tempfile.NamedTemporaryFile(delete=False, suffix='.o').name

        build = subprocess.run([NVCC, "-c", "-rdc=true", f"-arch={arch}", *NVCC_FLAGS, "-o", fbin_cuda, fsrc_cuda],
                               capture_output=True)
        assert build.returncode == 0, (f"nvcc failed\nstderr:\n{build.stderr.decode()}")

        with tempfile.NamedTemporaryFile(delete=False, mode='w', suffix='.ptx') as fsrc_triton, \
            tempfile.NamedTemporaryFile(delete=False, mode='r', suffix='.log') as flog:
            fsrc_triton.write(src)
            fsrc_triton.flush()
            fbin_triton = fsrc_triton.name + '.o'
            fbin_combined = fbin_triton + '.combined.cubin'

            compile_only_cmds = ['-c']
            line_info = ["-lineinfo", "-suppress-debug-info"] if knobs.compilation.disable_line_info else ["-lineinfo"]
            fmad = [] if opt.enable_fp_fusion else ['--fmad=false']
            disable_opt = ['--opt-level', '0'] if knobs.nvidia.disable_ptxas_opt else []
            ptx_extra_options = opt.ptx_options.split(" ") if opt.ptx_options else []

            ptxas_cmd = [
                PTXAS, *compile_only_cmds, *line_info, *fmad, '-v', *disable_opt, *ptx_extra_options,
                f'--gpu-name={arch}', fsrc_triton.name, '-o', fbin_triton
            ]

            try:
                subprocess.run(ptxas_cmd, check=True, close_fds=False, stderr=flog)
                if os.path.exists(fsrc_triton.name):
                    os.remove(fsrc_triton.name)
                if os.path.exists(flog.name):
                    os.remove(flog.name)
            except subprocess.CalledProcessError as e:
                with open(flog.name) as log_file:
                    log = log_file.read()
                if os.path.exists(flog.name):
                    os.remove(flog.name)

                if e.returncode == 255:
                    error = 'Internal Triton PTX codegen error'
                elif e.returncode == 128 + signal.SIGSEGV:
                    error = '`ptxas` raised SIGSEGV'
                else:
                    error = f'`ptxas` failed with error code {e.returncode}'
                raise PTXASError(f"{error}\n"
                                 f"`ptxas` stderr:\n{log}\n"
                                 f'Repro command: {" ".join(ptxas_cmd)}\n')

            nvlink_cmds = [
                NVLINK,
                f"-arch={arch}",
                *NVLINK_FLAGS,
                fbin_triton,
                fbin_cuda,
                "-o",
                fbin_combined,
            ]

            try:
                subprocess.run(nvlink_cmds, check=True, close_fds=False, stderr=flog)
            except Exception as e:
                import logging
                logging.error(f"error runing nvlink: {shlex.join(nvlink_cmds)}")
                logging.exception(e)

            with open(fbin_combined, 'rb') as f:
                cubin = f.read()
            if os.path.exists(fbin_combined):
                os.remove(fbin_combined)
            if os.path.exists(fbin_triton):
                os.remove(fbin_triton)
            if os.path.exists(fbin_cuda):
                os.remove(fbin_cuda)
        return cubin

    stages["cubin"] = lambda src, metadata: make_cubin(triton_self, src, metadata, options, triton_self.target.arch)


class CUDAJITFunction(object):

    def __init__(self, fn: Any, file: Path, *args, **kwargs) -> None:
        super().__init__(
            *args, **{
                k: v
                for k, v in kwargs.items()
                if k not in ("compiler", "target", "extern_file", "extern_func_name", "deferred")
            })
        self.fn: Final[Any] = fn
        self.code: Final[str] = file.read_text()
        self.region_dialect: Final[str] = "cuda"
        self.lowered_region_dialect: Final[str] = "llvm"
        self.arg_dialect: Final[str] = "llvm"
        self.source_file: Final[str] = str(file)
        self.compiler = kwargs.get("compiler", None)
        self.target = kwargs.get("target", None)
        self.extern_file = kwargs.get("extern_file", None)
        self.extern_func_name = kwargs.get("extern_func_name", None)
        self.deferred: Final[bool] = kwargs.get("deferred", False)
        self.__triton_builtin__: Final[bool] = True

        if self.compiler is not None and self.compiler.lower(
        ) == "nvcc" and knobs.runtime.add_stages_inspection_hook is None:
            nvcc_cuda_hook = partial(make_cubin_inspection_hook, self)
            knobs.runtime.add_stages_inspection_hook = nvcc_cuda_hook

    def register_pending_source(self, *, hint: str = "") -> str:
        if not self.extern_func_name:
            raise RuntimeError("deferred tle_raw CUDA source requires extern_func_name= "
                               "(the device function symbol in the .cu file)")
        return register_source(
            region_dialect=self.region_dialect,
            extern_func_name=self.extern_func_name,
            source=self.code,
            hint=hint,
            extra={"source_file": self.source_file},
        )

    def create_region_by_llvm(self, builder, llvm: str, handles, alias_indices, hint: str = ""):
        return builder.create_tle_raw_region_by_llvm_func(
            llvm,
            self.region_dialect,
            self.arg_dialect,
            handles,
            alias_indices,
            hint,
        )

    def create_region_deferred(self, builder, source_id: str, handles, alias_indices, hint: str = ""):
        return builder.create_tle_raw_region_deferred(
            source_id,
            self.region_dialect,
            self.arg_dialect,
            handles,
            alias_indices,
            hint,
        )

    def make_llvm(self, mlir_context) -> str:
        build = subprocess.run(
            [
                CLANG,
                "-x",
                "cuda",
                "--cuda-device-only",
                _get_cuda_gpu_arch(),
                "-emit-llvm",
                "-O2",
                "-S",
                "-",
                "-o",
                "-",
                *CLANG_FLAGS,
            ],
            input=self.code.encode(),
            capture_output=True,
        )
        assert build.returncode == 0, (f"clang failed\nstderr:\n{build.stderr.decode()}")
        llvm_context = llvm.context()
        module = parse_llvm_ir(_sanitize_clang_ir(build.stdout.decode()), llvm_context, mlir_context)
        return f"{module}"

    def make_bc(self, public_api_names=None):
        fbc_cuda_unopti = Path(self.source_file).with_suffix('.bc.unopti')
        fbc_cuda = Path(self.source_file).with_suffix('.bc')

        build = subprocess.run([
            CLANG, "-c", "-x", "cuda", "--cuda-device-only",
            _get_cuda_gpu_arch(), "-emit-llvm", "-fcuda-flush-denormals-to-zero", *CLANG_FLAGS, "-o", fbc_cuda_unopti,
            self.source_file
        ], capture_output=True)
        assert build.returncode == 0, (f"clang failed\nstderr:\n{build.stderr.decode()}")

        if public_api_names is None:
            public_api_names = [self.extern_func_name]
        elif isinstance(public_api_names, str):
            public_api_names = [public_api_names]
        else:
            public_api_names = list(public_api_names)
        if not public_api_names or any(not name for name in public_api_names):
            raise ValueError("make_bc requires at least one public API name")
        public_api_list = ",".join(dict.fromkeys(public_api_names))

        opt = subprocess.run([
            OPT, "--passes=internalize,inline,globaldce", f"-internalize-public-api-list={public_api_list}", "-o",
            fbc_cuda, fbc_cuda_unopti
        ], capture_output=True)
        assert opt.returncode == 0, (f"opt failed\nstderr:\n{opt.stderr.decode()}")

        if os.path.exists(fbc_cuda_unopti):
            os.remove(fbc_cuda_unopti)
        return fbc_cuda


def compile_deferred_pending_source(entry: dict, *, context) -> str:
    source_text = entry["source"]

    class _CudaSourceFile:

        def read_text(self):
            return source_text

    cuda_fn = CUDAJITFunction(
        fn=None,
        file=_CudaSourceFile(),
        extern_func_name=entry.get("extern_func_name"),
        deferred=True,
    )
    return cuda_fn.make_llvm(context)
