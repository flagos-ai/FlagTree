import functools
import hashlib
import os
import platform
import tempfile
from pathlib import Path

from dataclasses import dataclass
from types import ModuleType
from typing import Any, Dict, Optional, Tuple

from triton._C.libtriton import cpu, ir, llvm, passes
from triton.backends.compiler import BaseBackend, GPUTarget
import triton.backends.cpu.driver as cpu_driver


def min_dot_size(target: GPUTarget):
    # Allow M=1 for true GEMV decode workloads (M=1 token generation)
    # N and K still require >=4 for tl.dot tiling constraints
    return lambda lhsType, rhsType: (1, 4, 4)


VecLib = cpu.passes.ttcpuir.VecLib
Ukernels = cpu.passes.ttcpuir.Ukernels


@dataclass(frozen=True)
class CPUOptions:
    # GPU-compatible dummy options (ignored by CPU backend)
    backend_name: str = "cpu"
    num_warps: int = 0
    num_stages: int = 0
    num_ctas: int = 0
    # Max threads for parallel kernel execution (0 = all available cores)
    num_threads: int = 0
    cluster_dims: tuple = (1, 1, 1)
    extern_libs: dict = None
    debug: bool = False
    supported_fp8_dtypes: Tuple[str] = ("fp8e5", "fp8e5b16", "fp8e4nv")
    deprecated_fp8_dtypes: Tuple[str] = ()
    allowed_dot_input_precisions: Tuple[str] = ("ieee", "tf32", "tf32x3")
    allow_fp8e4nv: bool = True
    allow_fp8e4b15: bool = True
    enable_fp_fusion: bool = True
    launch_cooperative_grid: bool = False
    max_num_imprecise_acc_default: int = 0
    # ARM64 CPU always uses fast-math for NEON/SVE2 codegen performance.
    # No runtime toggle needed — accuracy-sensitive paths use TLE C kernels
    # with explicit precision control, not Triton JIT kernels.
    enable_fast_math: bool = True
    # Math vectorization library: 'libsleef', 'libmvec', or None
    vec_lib: Optional[str] = 'libsleef'
    sanitize_overflow: bool = False
    # Micro-kernel library: 'OneDNN', 'XSMM', or None
    ukernels: str = None

    def __post_init__(self):
        pass

    def hash(self):
        hash_dict = dict(self.__dict__)
        key = "_".join([f"{name}-{val}" for name, val in sorted(hash_dict.items())])
        return hashlib.sha256(key.encode("utf-8")).hexdigest()

    def get_vec_lib(self) -> VecLib:
        if self.vec_lib is None:
            return None
        vec_lib = VecLib.__members__.get(self.vec_lib, None)
        if vec_lib is None:
            raise ValueError(f"Unexpected value for vec_lib: {self.vec_lib}, "
                             f"should be one of {{{', '.join(VecLib.__members__.keys())}}}")
        return vec_lib

    def get_ukernels(self) -> Ukernels:
        raw_ukernels = os.getenv("TRITON_CPU_UKERNELS_LIB", self.ukernels)
        if raw_ukernels is None or raw_ukernels == "None":
            return None
        ukernels = Ukernels.__members__.get(raw_ukernels, None)
        if ukernels is None:
            raise ValueError(f"Unexpected value for ukernels: {raw_ukernels}, "
                             f"should be one of {{{', '.join(Ukernels.__members__.keys())}}}")

        if ukernels == Ukernels.OneDNN and not cpu.onednn_available():
            import warnings
            warnings.simplefilter('once', category=UserWarning)
            warnings.warn(
                "Warning! Triton build was made without OneDNN support. "
                "Check if CMAKE_PREFIX_PATH contains path to OneDNN during build.\n"
                "\t-------OneDNN will NOT be used-------", stacklevel=1)
            return None

        if ukernels == Ukernels.XSMM and not cpu.xsmm_available():
            import warnings
            warnings.simplefilter('once', category=UserWarning)
            warnings.warn(
                "Warning! Triton build was made without XSMM support. "
                "Check if CMAKE_PREFIX_PATH contains path to XSMM during build.\n"
                "\t-------XSMM will NOT be used-------", stacklevel=1)
            return None

        return ukernels


class CPUBackend(BaseBackend):

    @staticmethod
    def supports_target(target: GPUTarget):
        return target.backend == "cpu"

    def __init__(self, target: tuple) -> None:
        super().__init__(target)
        self.binary_ext = "so"
        self.cpu_arch = platform.machine()
        self.cpu_name = llvm.get_cpu_name()
        self.cpu_features = llvm.get_cpu_features()
        # LLVM get_cpu_features() on aarch64 misses several features (e.g. i8mm, bf16).
        # Supplement from /proc/cpuinfo 'Features' line on Linux.
        if platform.system() == "Linux" and self.cpu_arch == "aarch64":
            try:
                with open("/proc/cpuinfo") as f:
                    for line in f:
                        if line.startswith("Features"):
                            proc_feats = set(line.split(":")[1].split())
                            # Map proc cpuinfo feature names to LLVM-style names
                            _feat_map = {
                                "i8mm": "i8mm",
                                "svei8mm": "i8mm",
                                "bf16": "bf16",
                                "svebf16": "bf16",
                                "asimdbf16": "bf16",
                            }
                            for proc_feat, llvm_feat in _feat_map.items():
                                if proc_feat in proc_feats:
                                    self.cpu_features.add(llvm_feat)
                            break
            except OSError:
                pass
        if 'amx-tile' in self.cpu_features:
            if not cpu.enable_amx():
                import warnings
                warnings.warn("Warning! Couldn't enable AMX for the process. "
                              "AMX optimizations are disabled.")
                self.cpu_features.discard('amx-tile')
                self.cpu_features.discard('amx-int8')
                self.cpu_features.discard('amx-fp16')
                self.cpu_features.discard('amx-bf16')

    def parse_options(self, opts) -> Any:
        args = {k: opts[k] for k in CPUOptions.__dataclass_fields__.keys() if k in opts}
        if "supported_fp8_dtypes" not in args:
            supported_fp8_dtypes = set(CPUOptions.supported_fp8_dtypes)
            args["supported_fp8_dtypes"] = tuple(sorted(supported_fp8_dtypes))
        return CPUOptions(**args)

    def pack_metadata(self, metadata):
        return metadata

    def get_codegen_implementation(self, options=None):
        codegen_fns = {"min_dot_size": min_dot_size(self.target)}
        return codegen_fns

    def get_module_map(self) -> Dict[str, ModuleType]:
        from triton.language.extra.cpu import libdevice
        return {"triton.language.extra.libdevice": libdevice}

    def load_dialects(self, ctx):
        cpu.load_dialects(ctx)

    @staticmethod
    def make_ttir(mod, metadata, opt):
        # Shared with GPU backends: inline, combine, canonicalize, reorder_broadcast
        pm = ir.pass_manager(mod.context)
        # Note: pm.enable_debug() disabled to avoid "unknown data value for option"
        # error when printing VectorTransformsOptions in pass pipeline.
        passes.common.add_inliner(pm)
        passes.ttir.add_combine(pm)
        passes.common.add_canonicalizer(pm)
        passes.ttir.add_reorder_broadcast(pm)
        passes.common.add_cse(pm)
        passes.common.add_licm(pm)
        passes.common.add_symbol_dce(pm)
        pm.run(mod)
        return mod

    @staticmethod
    def make_ttcir(mod, metadata, opt):
        # TTIR -> TritonCPU IR (scalarization pass)
        pm = ir.pass_manager(mod.context)
        # Note: pm.enable_debug() disabled to avoid "unknown data value for option"
        cpu.passes.ttcpuir.add_scalarize(pm, True)
        cpu.passes.ttcpuir.add_convert_memory_ops(pm, True)
        cpu.passes.ttcpuir.add_convert_ptr_ops(pm)
        cpu.passes.ttcpuir.add_convert_elementwise_ops(pm)
        cpu.passes.ttcpuir.add_convert_elem_manip_ops(pm)
        cpu.passes.ttcpuir.add_convert_dot_op(pm)
        cpu.passes.ttcpuir.add_convert_histogram_op(pm)
        # use_multidim_reduction_op=True preserves vector.MultiDimReductionOp for
        # 2D reductions, which the ConvertDotProduct (BFDOT) pass in make_tttcir
        # matches to lower bf16 dot products. False would make BFDOT dead code.
        cpu.passes.ttcpuir.add_convert_reduction_op(pm, True, True)
        cpu.passes.ttcpuir.add_convert_scan_op(pm)
        cpu.passes.ttcpuir.add_convert_cf_ops(pm)
        cpu.passes.ttcpuir.add_convert_atomic_ops(pm)
        cpu.passes.ttcpuir.add_convert_debug_ops(pm)
        passes.common.add_cse(pm)
        passes.common.add_symbol_dce(pm)
        passes.common.add_canonicalizer(pm)
        pm.run(mod)
        metadata["cluster_dims"] = (opt.cluster_dims[0], opt.cluster_dims[1], opt.cluster_dims[2])
        return mod

    def make_tttcir(self, mod, metadata, opt):
        # TritonCPU IR -> target-optimized TritonCPU IR
        # (architecture-specific dot lowering, mask optimization, etc.)
        pm = ir.pass_manager(mod.context)
        # Note: pm.enable_debug() disabled to avoid "unknown data value for option"
        cpu.passes.ttcpuir.add_triton_cpu_canonicalizer(pm)
        cpu.passes.ttcpuir.add_optimize_masks(pm)
        passes.common.add_canonicalizer(pm)

        if (ukernels := opt.get_ukernels()):
            cpu.passes.ttcpuir.add_loop_invariant_code_motion(pm)
            cpu.passes.ttcpuir.add_convert_dot_to_ukernels(pm, ukernels)
            passes.common.add_canonicalizer(pm)
            passes.common.add_cse(pm)

        is_arm = self.cpu_arch == "aarch64" or self.cpu_arch == "armv8"

        # ARM NEON BFDOT: lowers bf16 dot-product reductions (MultiDimReductionOp)
        if is_arm and 'fp-armv8' in self.cpu_features and 'neon' in self.cpu_features:
            cpu.passes.ttcpuir.add_convert_dot_product(pm, True)

        # ARM SVE2 i8mm: lowers int8 matrix multiply (cpu::DotOp).
        # Independent of BFDOT above — different dtype/op, both active on SVE2+NEON parts.
        if is_arm and 'sve2' in self.cpu_features:
            cpu.passes.ttcpuir.add_convert_dot_to_sve2_i8mm(pm)

        # Intel AMX dot lowering
        if 'amx-tile' in self.cpu_features:
            amx_int8 = 'amx-int8' in self.cpu_features
            amx_fp16 = False  # FP16 AMX dialect support pending
            amx_bf16 = 'amx-bf16' in self.cpu_features
            cpu.passes.ttcpuir.add_convert_dot_to_amx(pm, amx_int8, amx_fp16, amx_bf16)

        # x86 AVX-512 FMA dot lowering
        if 'avx512f' in self.cpu_features:
            cpu.passes.ttcpuir.add_convert_dot_to_fma(pm)

        # Generic scalar fallback for remaining dot ops
        cpu.passes.ttcpuir.add_convert_dot_generic(pm)

        # Type promotion passes
        promote_bf16_to_fp32 = (self.cpu_arch == "x86_64" and "avx512bf16" not in self.cpu_features)
        convert_mixed_precision_matmul = True
        promote_lib_math_to_fp32 = True
        cpu.passes.ttcpuir.add_convert_unsupported_ops(pm, promote_bf16_to_fp32, convert_mixed_precision_matmul,
                                                       promote_lib_math_to_fp32)

        decompose_bf16_conv = (self.cpu_arch == "x86_64" and "avx512bf16" not in self.cpu_features)
        decompose_fp8_conv = True
        cpu.passes.ttcpuir.add_decompose_fp_conversions(pm, decompose_bf16_conv, decompose_fp8_conv)

        passes.common.add_cse(pm)
        passes.common.add_symbol_dce(pm)
        passes.common.add_canonicalizer(pm)
        pm.run(mod)
        return mod

    def make_llir(self, src, metadata, options):
        num_warp_groups = src.get_int_attr("triton_gpu.num-warp-groups-per-cta")
        if num_warp_groups is not None:
            metadata["num_warps"] *= num_warp_groups
        metadata["threads_per_warp"] = 1
        mod = src

        # TritonCPU IR -> LLVM IR (MLIR representation)
        pm = ir.pass_manager(mod.context)
        # Note: pm.enable_debug() disabled to avoid "unknown data value for option"

        if options.get_ukernels() == Ukernels.OneDNN:
            cpu.passes.ttcpuir.add_ukernels_to_onednn_llvmir(pm)
        if options.get_ukernels() == Ukernels.XSMM:
            cpu.passes.ttcpuir.add_ukernels_to_xsmm_llvmir(pm)

        # TLE-CPU: lower NeonSdotOp / SdotGemvOp / SwigluOp / FlashAttnDecodeOp /
        # FusedMlpOp / FusedTransformerLayerOp to LLVM runtime calls
        import platform
        if platform.machine() in ("aarch64", "arm64"):
            cpu.passes.ttcpuir.add_neon_sdot_to_llvmir(pm)

        cpu.passes.ttcpuir.add_lower_vector_multi_dim(pm)
        cpu.passes.ttcpuir.add_expand_strided_metadata(pm)
        cpu.passes.ttcpuir.add_vector_to_scf(pm, True, 1, False)
        cpu.passes.ttcpuir.add_lower_affine(pm)
        passes.convert.add_scf_to_cf(pm)
        passes.convert.add_index_to_llvmir(pm)
        cpu.passes.ttcpuir.add_func_op_to_llvmir(pm)
        cpu.passes.ttcpuir.add_program_id_to_llvmir(pm)
        cpu.passes.ttcpuir.add_memory_op_to_llvmir(pm)
        cpu.passes.ttcpuir.add_atomic_ops_to_llvmir(pm)
        cpu.passes.ttcpuir.add_debug_ops_to_llvmir(pm)

        # Vectorized math library substitution
        vec_lib_requirements = {
            VecLib.libsleef: {"neon", "sse", "avx"},
            VecLib.libmvec: {"avx512f"},
        }
        if (vec_lib := options.get_vec_lib()) and vec_lib_requirements[vec_lib] & self.cpu_features:
            cpu.passes.ttcpuir.add_math_to_vec_lib(pm, vec_lib, self.cpu_features)

        passes.convert.add_math_to_llvmir(pm)
        cpu.passes.ttcpuir.add_math_to_libm(pm)
        cpu.passes.ttcpuir.add_vector_to_llvmir(pm, options.enable_fast_math)
        cpu.passes.ttcpuir.add_memref_to_llvmir(pm)
        passes.convert.add_reconcile_unrealized(pm)
        passes.convert.add_arith_to_llvmir(pm)
        cpu.passes.ttcpuir.add_func_to_llvmir(pm)
        cpu.passes.ttcpuir.add_ub_to_llvmir(pm)
        passes.common.add_canonicalizer(pm)
        passes.common.add_cse(pm)
        passes.common.add_symbol_dce(pm)

        if os.environ.get("TRITON_DISABLE_LINE_INFO", "0") == "0":
            passes.llvmir.add_di_scope(pm)
        pm.run(mod)

        # Find kernel function name
        kernel_names = cpu.find_kernel_names(mod)
        assert len(kernel_names) == 1, \
            f"expected exactly 1 kernel in a module, got {kernel_names}"

        # LLVM IR (MLIR) -> LLVM IR (native LLVM)
        llvm.init_targets()
        context = llvm.context()
        llvm_mod = llvm.to_module(mod, context)
        if llvm_mod is None:
            raise RuntimeError("Failed to convert to LLVM IR")
        llvm.set_host_target(llvm_mod)
        llvm.optimize_module(llvm_mod, llvm.OPTIMIZE_O3)

        metadata["shared"] = 0
        metadata["name"] = kernel_names[0]
        ret = str(llvm_mod)
        del llvm_mod
        del context
        return ret

    @staticmethod
    def make_asm(src, metadata, options):
        return llvm.translate_to_host_asm(src, options.enable_fp_fusion)

    @staticmethod
    def make_so(src, metadata, options):
        with tempfile.TemporaryDirectory() as tmpdir:
            asm_path = os.path.join(tmpdir, "kernel.s")
            Path(asm_path).write_text(src)
            lib_dirs = cpu_driver.library_dirs
            libs = ["m", "TritonCPURuntime", "sleef"]
            # TLE-CPU: compile and link registered NEON C functions
            extra_objs = []
            try:
                from triton.language.extra.cpu.neon import get_all_object_files
                extra_objs = get_all_object_files()
            except ImportError:
                pass
            if extra_objs:
                import subprocess
                ar_path = os.path.join(tmpdir, "libtle_neon.a")
                subprocess.check_call(["ar", "rcs", ar_path] + list(extra_objs))
                lib_dirs = list(lib_dirs) + [tmpdir]
                libs.append("tle_neon")
                libs.append("gomp")
            from triton.runtime.build import _build
            so = _build("kernel", asm_path, tmpdir, lib_dirs, cpu_driver.include_dirs, libs)
            with open(so, "rb") as f:
                return f.read()

    def add_stages(self, stages, options):
        stages["ttir"] = lambda src, metadata: self.make_ttir(src, metadata, options)
        stages["ttcir"] = lambda src, metadata: self.make_ttcir(src, metadata, options)
        stages["tttcir"] = lambda src, metadata: self.make_tttcir(src, metadata, options)
        stages["llir"] = lambda src, metadata: self.make_llir(src, metadata, options)
        stages["asm"] = lambda src, metadata: self.make_asm(src, metadata, options)
        stages["so"] = lambda src, metadata: self.make_so(src, metadata, options)

    @functools.lru_cache()
    def hash(self):
        import platform
        return f"{platform.machine()}"
