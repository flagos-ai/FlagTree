from triton.backends.compiler import BaseBackend, GPUTarget
from triton._C.libtriton import ir, passes
from triton.runtime.cache import get_cache_manager

from dataclasses import dataclass
from typing import Any, Dict, Tuple, Optional
from pathlib import Path
from types import ModuleType
import os
import re
import functools
import hashlib
import tempfile
import subprocess
import importlib
import shutil


def _get_libdevice_bc_path() -> str:
    return "/opt/thrive/lib/libdevice/libdevice.bc"


def _get_libdevice_shmem_bc_path() -> str:
    return "/opt/thrive/lib/libdevice/libdevice_shmem.bc"


def _get_thrive_libdevice_path() -> dict:
    libdevice_dir = "/opt/thrive/lib/libdevice"
    files = {
        "vpu_libdevice": os.path.join(libdevice_dir, "vpu_libdevice.bc"),
        "dmu_libdevice": os.path.join(libdevice_dir, "dmu_libdevice.bc"),
        "nou_libdevice": os.path.join(libdevice_dir, "nou_libdevice.bc"),
    }
    for name, path in files.items():
        if not os.path.isfile(path):
            raise FileNotFoundError(f"[thrive] {name} bitcode not found: {path}")
    return files


def _get_thrive_opt_path() -> str:
    return "/opt/thrive/bin/thrive-opt"


def _get_dfca_mlir_opt_path() -> str:
    return "/opt/thrive/bin/mlir-opt"


def _get_dfca_mlir_translate_path() -> str:
    return "/opt/thrive/bin/mlir-translate"


def _get_dfca_llvm_link_path() -> str:
    return "/opt/thrive/bin/llvm-link"


def _get_dfca_llvm_opt_path() -> str:
    return "/opt/thrive/bin/opt"


def _get_dfcacc_path() -> str:
    return "/opt/thrive/bin/dfcacc"


def _save_content_to_tmp(content, subdir, fname):
    tmp_root = os.path.join(os.getcwd(), "tmp")
    if not os.path.exists(tmp_root):
        return
    out_dir = os.path.join(tmp_root, subdir)
    os.makedirs(out_dir, exist_ok=True)
    dst_path = os.path.join(out_dir, fname)
    if (isinstance(content, (str, Path)) and os.path.isfile(content)):
        shutil.copy2(content, dst_path)
    else:
        with open(dst_path, "w") as f:
            f.write(content if isinstance(content, str) else str(content))


@functools.lru_cache(None)
def file_hash(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


@dataclass(frozen=True)
class ThriveOptions:
    # GPU-specific options
    num_warps: int = 4
    num_ctas: int = 4
    # Thrive-specific options
    num_cores: int = 2
    num_pes: int = 1
    aot_kernel: bool = False
    enable_vectorization = True
    # Common options
    backend_name: str = "thrive"
    launch_cooperative_grid: bool = False
    num_stages: int = 0
    cluster_dims: tuple = (1, 1, 1)
    extern_libs: dict = None
    debug: bool = False
    supported_fp8_dtypes: tuple[str] = ("fp8e5", "fp8e5b15", "fp8e4nv")
    deprecated_fp8_dtypes: tuple[str] = ()
    allow_fp8e4nv: bool = True
    allowed_dot_input_precisions: Tuple[str] = ("ieee", "tf32")
    max_num_imprecise_acc_default: int = 0
    enable_fp_fusion: bool = False
    enable_fast_math: bool = True
    sanitize_overflow: bool = False
    instrumentation_mode: str = ""

    def __post_init__(self):
        extern_libs = {} if self.extern_libs is None else dict(self.extern_libs)
        if not extern_libs.get("libdevice", None):
            extern_libs["libdevice"] = _get_libdevice_bc_path()
            extern_libs["libdevice_shmem.bc"] = _get_libdevice_shmem_bc_path()
            extern_libs.update(_get_thrive_libdevice_path())
        object.__setattr__(self, "extern_libs", tuple(extern_libs.items()))

    def hash(self):
        hash_dict = dict(self.__dict__)
        hash_dict["extern_libs"] = tuple((k, file_hash(v)) for k, v in sorted(hash_dict["extern_libs"]))
        key = "_".join([f"{name}-{val}" for name, val in sorted(hash_dict.items())])
        return hashlib.sha256(key.encode("utf-8")).hexdigest()


class ThriveBackend(BaseBackend):

    @staticmethod
    def supports_target(target: GPUTarget):
        return target.backend == 'thrive'

    def __init__(self, target: GPUTarget) -> None:
        super().__init__(target)
        self.binary_ext = "dfcafb"

    def parse_options(self, opts) -> Any:
        args = {k: opts[k] for k in ThriveOptions.__dataclass_fields__.keys() if k in opts if opts[k] is not None}
        num_cores = args.get("num_cores", 2)
        allowed_cores = {1, 2, 4}
        if num_cores not in allowed_cores:
            raise ValueError(f"The num_cores must be 1 or 2 or 4, but got {num_cores}.")
        return ThriveOptions(**args)

    def pack_metadata(self, metadata):
        return (
            metadata.num_cores,
            metadata.use_dshmem,
            metadata.fw_preempt,
        )

    def get_codegen_implementation(self, options):
        codegen_fns = {"min_dot_size": lambda lhsType, rhsType: (1, 1, 1)}
        return codegen_fns

    def load_dialects(self, ctx):
        pass

    def get_module_map(self) -> Dict[str, ModuleType]:
        from triton.language.extra.thrive import libdevice
        return {"triton.language.extra.libdevice": libdevice}

    @functools.lru_cache()
    def hash(self):
        return self.target

    @staticmethod
    def make_ttir(mod, metadata, options):
        pm = ir.pass_manager(mod.context)
        pm.enable_debug()
        passes.common.add_inliner(pm)
        passes.ttir.add_combine(pm)
        passes.common.add_canonicalizer(pm)
        passes.ttir.add_reorder_broadcast(pm)
        passes.common.add_cse(pm)
        passes.common.add_licm(pm)
        passes.common.add_symbol_dce(pm)
        pm.run(mod, 'make_ttir')
        metadata["name"] = mod.get_entry_func_name()
        _save_content_to_tmp(mod, metadata["name"], "output_ttir.mlir")
        return mod

    @staticmethod
    def make_thrir(mod, metadata, options):
        metadata["name"] = mod.get_entry_func_name()
        metadata["num_cores"] = options.num_cores
        mod_text = str(mod)
        passes = [
            f"--triton-convert-to-triton-thrive=num-cores={options.num_cores}",
            "--triton-thrive-convert-gemm-op",
            "--triton-thrive-layout-propagate",
            "--cse",
            "--canonicalize",
            "--symbol-dce",
            "--triton-thrive-software-pipeline",
            "--triton-thrive-op-to-linalg",
            "--linalg-generalize-named-ops",
            "--triton-thrive-linalg-fuse-elementwise-ops",
            "--triton-thrive-convert-linalg-to-dsa",
        ]
        if options.enable_vectorization:
            passes.append("--triton-thrive-wrap-vpu")
            passes.append("--triton-thrive-trans-bf16-vpu")
        passes.extend([
            "--symbol-dce",
            "--canonicalize",
        ])
        with tempfile.TemporaryDirectory() as tmpdir:
            src_path = os.path.join(tmpdir, "tt.mlir")
            dst_path = os.path.join(tmpdir, "thr.mlir")
            Path(src_path).write_text(mod_text)
            thrive_opt = _get_thrive_opt_path()
            subprocess.check_call([thrive_opt, src_path, *passes, "-o", dst_path])
            result = Path(dst_path).read_text()
            _save_content_to_tmp(result, metadata["name"], "output_thr.mlir")
            return result

    @staticmethod
    def make_thrmir(mod, metadata, options):
        mod_text = str(mod)
        passes = [
            "--triton-thrive-wrap-shareop",
            "--triton-thrive-synchronize",
            "--triton-thrive-op-to-memref",
            "--triton-thrive-memref-alloc-hoist",
            "--triton-thrive-memref-promote-sram",
            "--triton-thrive-convert-simt-alloc",
        ]
        if options.enable_vectorization:
            passes.append("--triton-thrive-outline-vector-pass")
        passes.extend([
            "--triton-thrive-lower-with-thread-id",
            "--symbol-dce",
            "--canonicalize",
            "--cse",
        ])
        with tempfile.TemporaryDirectory() as tmpdir:
            src_path = os.path.join(tmpdir, "thr.mlir")
            dst_path = os.path.join(tmpdir, "thrm.mlir")
            Path(src_path).write_text(mod_text)
            thrive_opt = _get_thrive_opt_path()
            subprocess.check_call([thrive_opt, src_path, *passes, "-o", dst_path])
            result = Path(dst_path).read_text()
            _save_content_to_tmp(result, metadata["name"], "output_thrm.mlir")
            return result

    @staticmethod
    def make_llir(mod, metadata, options):
        mod_text = str(mod)
        metadata["fw_preempt"] = 0
        metadata["shared"] = 0
        metadata["use_dshmem"] = 0

        passes = []
        if options.enable_vectorization:
            passes.append("--triton-thrive-tiling-pass")
            passes.append("--triton-thrive-rewrite-reduction-pass")
        passes.append("--convert-linalg-to-loops")
        if options.enable_vectorization:
            passes.append("--triton-thrive-virtual-vector-pass")
        passes.extend([
            "--triton-thrive-memref-alloc-to-global",
            "--triton-thrive-convert-memref-to-llvm",
            "--triton-thrive-memory-op-to-llvm",
            "--convert-scf-to-cf",
            "--triton-thrive-func-op-to-llvm",
            "--triton-thrive-dsa-op-to-libdevice",
            "--triton-thrive-get-program-id-op-to-llvm",
            "--triton-thrive-convert-fp-to-fp",
            "--triton-thrive-mulextend-op-to-llvm",
            "--convert-index-to-llvm",
            "--memref-expand",
            "--expand-strided-metadata",
            "--finalize-memref-to-llvm",
            "--convert-vector-to-llvm",
            "--lower-affine",
            "--convert-scf-to-cf",
            "--convert-arith-to-llvm",
            "--convert-math-to-llvm",
            "--convert-cf-to-llvm",
            "--convert-func-to-llvm",
            "--canonicalize",
            "--cse",
            "--symbol-dce",
        ])
        with tempfile.TemporaryDirectory() as tmpdir:
            thrmir_path = os.path.join(tmpdir, "thrm.mlir")
            tmp_llvmir_path = os.path.join(tmpdir, "tmp_llvm.mlir")
            Path(thrmir_path).write_text(mod_text)
            thrive_opt = _get_thrive_opt_path()
            subprocess.check_call([thrive_opt, thrmir_path, *passes, "-o", tmp_llvmir_path])

            tmp_llvm_mod_text = Path(tmp_llvmir_path).read_text()
            _save_content_to_tmp(tmp_llvm_mod_text, metadata["name"], "output_tmp_llvm.mlir")
            tmp_llvm_mod_text = tmp_llvm_mod_text.replace("inbounds|nuw", "inbounds")
            tmp_llvm_mod_text = re.sub(r'llvm.intr.assume\s+(%\d+)\s*:\s*i1', r'"llvm.intr.assume"(\1) : (i1) -> ()',
                                       tmp_llvm_mod_text)
            tmp_llvm_mod_text = tmp_llvm_mod_text.replace("llvm.intr.stepvector", "llvm.intr.experimental.stepvector")
            Path(tmp_llvmir_path).write_text(tmp_llvm_mod_text)

            llvmir_path = os.path.join(tmpdir, "llvm.mlir")
            dfca_mlir_opt = _get_dfca_mlir_opt_path()
            subprocess.check_call([dfca_mlir_opt, tmp_llvmir_path, "--dfcakernel-to-llvm-pipeline", "-o", llvmir_path])
            llvm_mod_text = Path(llvmir_path).read_text()
            _save_content_to_tmp(tmp_llvm_mod_text, metadata["name"], "output_llvm.mlir")
            return llvm_mod_text

    @staticmethod
    def make_fatbin(mod, metadata, options):
        mod_text = str(mod)
        kernel_name = metadata["name"]
        key = hashlib.sha256(mod_text.encode("utf-8")).hexdigest()
        cache = get_cache_manager(key)
        fatbin_path = cache._make_path(f"{kernel_name}.dfcafb")

        with tempfile.TemporaryDirectory() as tmpdir:
            mlir_path = os.path.join(tmpdir, "llvm.mlir")
            ll_path = os.path.join(tmpdir, "llvm.ll")
            Path(mlir_path).write_text(mod_text)
            dfca_mlir_translate = _get_dfca_mlir_translate_path()
            subprocess.check_call([dfca_mlir_translate, mlir_path, "--mlir-to-llvmir", "-o", ll_path])
            _save_content_to_tmp(ll_path, metadata["name"], "llvm.ll")

            dep_bc_path = [path for _, path in options.extern_libs]
            dst_bc_path = os.path.join(tmpdir, f"{kernel_name}_dst.bc")
            dfca_llvm_link = _get_dfca_llvm_link_path()
            subprocess.check_call(
                [dfca_llvm_link, ll_path, *dep_bc_path, "--only-needed", "-internalize", "-o", dst_bc_path])

            opt_bc_path = os.path.join(tmpdir, f"{kernel_name}_opt.bc")
            dfca_llvm_opt = _get_dfca_llvm_opt_path()
            subprocess.check_call([dfca_llvm_opt, dst_bc_path, "-mattr=+d,+f", "-passes=inline", "-o", opt_bc_path])
            _save_content_to_tmp(opt_bc_path, metadata["name"], f"{kernel_name}_opt.bc")

            dfcacc = _get_dfcacc_path()
            subprocess.check_call(
                [dfcacc, opt_bc_path, "-O1", "-dfca-bc", "--dfca-device-only", "-w", "-o", fatbin_path])

            return fatbin_path

    def add_stages(self, stages, options, language):
        stages["ttir"] = lambda src, metadata: self.make_ttir(src, metadata, options)
        stages["thrir"] = lambda src, metadata: self.make_thrir(src, metadata, options)
        stages["thrmir"] = lambda src, metadata: self.make_thrmir(src, metadata, options)
        stages["llir"] = lambda src, metadata: self.make_llir(src, metadata, options)
        stages["dfcafb"] = lambda src, metadata: self.make_fatbin(src, metadata, options)
