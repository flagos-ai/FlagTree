from __future__ import annotations
import os
from pathlib import Path
import subprocess
from typing import Any, Final
import torch

from functools import partial
from triton import knobs

from triton._C.libtriton import llvm  # pyright: ignore[reportMissingImports]
from triton._C.libtriton.tle.llvm import parse_llvm_ir  # pyright: ignore[reportMissingImports]

# TODO: We use cli tools to compile CUDA code temporarily, and plan to replace it with LLVM components Python bindings in the future.
CLANG = os.getenv("CLANG", "clang")
NVCC = os.getenv("NVCC", "nvcc")
PTXAS = os.getenv("PTXAS", "ptxas")
NVLINK = os.getenv("NVLINK", "nvlink")


def make_cubin_inspection_hook(cuda_self, triton_self, stages, options, language, capability):
    from triton.backends.nvidia.compiler import sm_arch_from_capability, get_ptxas
    def make_cubin(self, src, metadata, opt, capability):
        # cuda compile
        fsrc_cuda = cuda_self.file
        fbin_cuda = Path(fsrc_cuda).with_suffix('.o')
        
        include_flags = []
        link_flags = []
        if cuda_self.libs is not None:
            for _, lib_path in cuda_self.libs.items():
                include_flags.append(f"-I{os.path.join(lib_path, "include")}")
                link_flags.append(f"-L{os.path.join(lib_path, "lib")}")
        
        link_libs = [f"-l{lib}" for lib in cuda_self.links] if cuda_self.links is not None else []
        macro_flags = [f"-D{macro_name}={macro_value}" for macro_name, macro_value in cuda_self.macros.items()] if cuda_self.macros is not None else []
        
        prop = torch.cuda.get_device_properties(torch.cuda.current_device())
        capability = prop.major * 10 + prop.minor
        suffix = "a" if capability >= 90 else ""
        arch = f"sm_{capability}{suffix}"
        
        build = subprocess.run([
            NVCC, 
            "-rdc=true", 
            f"-arch={arch}", 
            *include_flags, 
            *macro_flags, 
            "--extended-lambda", 
            "-c", 
            "-o", 
            fbin_cuda, 
            fsrc_cuda],
            capture_output=True)
        assert build.returncode == 0, (f"nvcc failed\nstderr:\n{build.stderr.decode()}")
        
        # compile triton
        import tempfile
        import signal
        from triton.runtime.errors import PTXASError
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
            
            # nvlink
            nvlink_cmds = [
                NVLINK,
                f"-arch={arch}",
                *link_flags,
                *link_libs,
                fbin_triton,
                fbin_cuda,
                "-o",
                fbin_combined,
            ]
            
            try:
                subprocess.run(nvlink_cmds, check=True, close_fds=False, stderr=flog)
            except Exception as e:
                import logging
                logging.error(f"error runing nvlink: {nvlink_cmds}")
                logging.exception(e)
            
            with open(fbin_combined, 'rb') as f:
                cubin = f.read()
            if os.path.exists(fbin_combined):
                os.remove(fbin_combined)
            if os.path.exists(fbin_triton):
                os.remove(fbin_triton)
        
        return cubin
    
    stages["cubin"] = lambda src, metadata: make_cubin(triton_self, src, metadata, options, triton_self.target.arch)
    

class CUDAJITFunction(object):

    def __init__(self, fn: Any, file: Path, *args, **kwargs) -> None:
        super().__init__()
        self.fn: Final[Any] = fn
        self.code: Final[str] = file.read_text()
        self.file: Final[Path] = file
        self.compiler = kwargs.get("compiler", "clang")
        self.extern: Final[Path] = kwargs.get("extern", None)
        self.extern_func_name = kwargs.get("extern_func_name", None)
        self.libs = kwargs.get("libs", None)
        self.links = kwargs.get("links", None)
        self.macros = kwargs.get("macros", None)
        self.__triton_builtin__: Final[bool] = True
        
        if (self.compiler).lower() == "nvcc":
            bound_hook = partial(make_cubin_inspection_hook, self)
            knobs.runtime.add_stages_inspection_hook = bound_hook

    def make_llvm(self, mlir_context) -> str:
        prop = torch.cuda.get_device_properties(torch.cuda.current_device())
        capability = prop.major * 10 + prop.minor
        suffix = "a" if capability >= 90 else ""
        arch = f"sm_{capability}{suffix}"
        build = subprocess.run(
            [
                CLANG,
                "-x",
                "cuda",
                "--cuda-device-only",
                f"--cuda-gpu-arch={arch}",
                "-emit-llvm",
                "-I/home/zyuli/miniconda3/envs/flagtree/lib/python3.12/site-packages/torch/include",
                "-O2",
                "-S",
                "-",
                "-o",
                "-",
            ],
            input=self.code.encode(),
            capture_output=True,
        )
        assert build.returncode == 0, (f"clang failed\nstderr:\n{build.stderr.decode()}")
        llvm_context = llvm.context()
        module = parse_llvm_ir(build.stdout.decode(), llvm_context, mlir_context)
        return f"{module}"

    def make_cubin(self):
        ...