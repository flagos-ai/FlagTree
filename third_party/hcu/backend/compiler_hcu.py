from triton.backends.compiler import BaseBackend, GPUTarget, Language
from triton._C.libtriton import ir, passes, llvm, hcu, distributed, tle
from triton import knobs
from triton.runtime.errors import HSACOError
from dataclasses import dataclass
from .memory_schedule import MemorySchedulePolicy, prepare_memory_schedule
from typing import Any, Dict, Optional, Tuple
from types import ModuleType
import hashlib
import tempfile
import re
import subprocess
import functools
import warnings
import os
from pathlib import Path


_REGISTER_OWNER_DEFAULT_POLICY = "gfx936_optimize_epilogue_register_owner_shared_vector_v1"


def _schedule_hints(value):
    return tuple(dict.fromkeys(
        hint for part in value.split(",")
        if (hint := part.strip().lower()) and hint != "none"
    ))


@functools.lru_cache(maxsize=16)
def _clang_version(path, file_identity):
    return subprocess.check_output([path, "--version"], encoding="utf-8")


def _hcu_clang_version():
    path = HIPBackend.path_to_rocm_clang().resolve()
    info = path.stat()
    return _clang_version(path, (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns))


def get_min_dot_size(target: GPUTarget):
    # We fallback to use FMA and cast arguments if certain configurations is
    # not supported natively by matrix core units.
    return lambda lhs_type, rhs_type: (1, 1, 1)


def is_pingpong_schedule_enabled(arch, use_async_copy, override=None):
    if override is not None:
        return override
    return (arch == "gfx942" or (arch == "gfx950" and use_async_copy is True)
            ) if knobs.hcu.use_block_pingpong is None else knobs.hcu.use_block_pingpong


def is_in_thread_transpose_enabled(arch):
    return (arch == "gfx942") if knobs.hcu.use_in_thread_transpose is None else knobs.hcu.use_in_thread_transpose


@dataclass(frozen=True)
class HIPOptions:
    num_warps: int = 4
    waves_per_eu: int = 1
    num_stages: int = 2
    num_ctas: int = 1
    extern_libs: dict = None
    debug: bool = False
    sanitize_overflow: bool = True
    arch: str = None
    # We have native support for OCP fp8 variants since CDNA4/RDNA4. For earlier generations,
    # we software emulate the support for them.
    # For leagcy HCU, enable software emulation for fp8e4nv conversions.
    supported_fp8_dtypes: Tuple[str] = ("fp8e4nv", "fp8e5")
    deprecated_fp8_dot_operand_dtypes: Tuple[str] = ()
    default_dot_input_precision: str = "ieee"
    allowed_dot_input_precisions: Tuple[str] = ("ieee", )
    enable_fp_fusion: bool = True
    # Suppress automatic LLVM unrolling; explicit LLVM unroll hints prevail.
    disable_llvm_loop_unroll: bool = False
    # Keep wave-private synchronous staging in the operator tuning space.
    wave_private_sync: bool = False
    # Coalesce proved terminal read-only LDS publication; diagnostic opt-in.
    terminal_readonly_publication: bool = False
    combine_shifted_disjoint_or: bool = False
    # Experimental wave-group MMA motion; native completion check is required.
    mma_phase_schedule: bool = False
    coalesce_lgkm_waits: bool = False
    # Experimental full LDS wait after compute; proves cross-wave WAR release.
    post_mma_wait: bool = False
    # Preserve proved LDS-DMA alias metadata through native lowering.
    lds_dma_alias_workaround: bool = False
    # Rebase proved uniform store offsets; dynamic guards may increase pressure.
    rebase_buffer_store: bool = False
    # Extract proved nonnegative constant byte displacements to raw store SOFFSET.
    buffer_store_scalar_offset: bool = False
    # Exact full padded MemDescIndex plane intervals, retaining CFG barriers.
    padded_slot_intervals: bool = False
    # Split proved constant bytes between MUBUF immediate and uniform SOFFSET.
    buffer_store_split_immediate: bool = False
    # Preserve the proved masked base with an exact tied i32 register identity.
    buffer_store_preserve_masked_base: bool = False
    # Post-O3 same-BB readonly GPR/LDS issue interleaving; no wait rewrite.
    interleave_global_lds_reads: bool = False
    interleave_global_lds_reads_fences: bool = False
    # Keep register-preserving fences in four-float tuples for MMAC allocation.
    tupled_register_fences: bool = False
    structured_register_prepack_address: bool = False
    structured_register_unclamped_column: bool = False
    priority_after_first_mmac: bool = False
    launch_cooperative_grid: bool = False
    matrix_instr_nonkdim: int = 0
    kpack: int = 1
    allow_flush_denorm: bool = False
    max_num_imprecise_acc_default: int = 0
    backend_name: str = 'hip'
    instrumentation_mode: str = ""
    optimize_epilogue: bool = True
    # Dataclass defaults stay conservative; parse_options resolves target policy.
    register_batch_owner: bool = False
    register_batch_owner_shared_vector: bool = False
    # 0: mfma
    # 1: mmac legacy
    # 2: mmac interleave
    # 3: mmac transpose
    # 4: mmac interleave and transpose
    mmac_layout_force: int = -1
    # 0 keeps the heuristic; otherwise split CTA waves as [M, num_warps/M].
    mmac_warps_m: int = 0
    # Permit native accumulator-tile transpose on eligible gfx936 dots.
    mmac_transposed: bool = False

    async_copy_use_single_buffer: bool = True

    # None preserves the existing environment/architecture policy.
    use_block_pingpong: Optional[bool] = None
    prefer_wide_async_copy: bool = False

    # The following option provides hints to the HCUGPU backend regarding instruction scheduling
    # for all `tt.dot` operations in a kernel. The "none" variant preserves the default
    # instruction scheduling of the HCUGPU backend which aims at maximizing occupancy.
    # The option is experimental and may change at any time regarding its semantics and/or may
    # be gone entirely anytime.
    #
    # Current experimental scheduling variants:
    #
    # local-prefetch: implements instruction scheduling similar to the one from the ROCm Composable
    #                 Kernel library. Note, this variant requires the use of buffer load/store ops
    #                 and a special software pipelining style - i.e., 1x LDS and 1x register
    #                 prefetch buffers for each GEMM tile.
    #                 On gfx936 this also enables LDS-to-register prefetch for
    #                 eligible synchronous two- or three-stage single-dot loops.
    #                 Three stages additionally prefetch global loads in registers.
    # attention: enables a bunch of optimizations for attention kernels, including:
    #            - iglp 2 and sched.barrier around it
    #            - sink-insts-to-avoid-spills flag to avoid register spills
    # memory-bound-attention: enables custom scheduling strategy in llvm backend,
    #            This option targets special FA variant, which is memory bound and
    #            has a lot of elementwise operations from fused operand dequantizations.
    #            Note that this option is highly experimental,
    #            and will be removed as soon as default sceduler algorithm is fixed.
    #
    # Option allows to set multiple variants divided by commas:
    # schedule_hint="attention,memory-bound-attention"
    schedule_hint: str = 'none'

    # Optional LDS ring depth for eligible gfx936 synchronous local-prefetch.
    # Other loops retain their existing allocation and barrier analysis.
    sync_lds_buffers: int = 1

    # Consume standard loop-unroll hints at the selected stage.
    loop_unroll_stage: str = "ttir"

    # Extend options for HCU(legacy), used for old arch like gfx928, gfx936
    # 1. set scheduling latency for mmac and ds:
    #    - none: use default scheduling latency which equal mmac1-ds5
    #    - see get_options_args() to get more options.
    sched_latency: str = 'none'

    # wasp options
    wasp_enabled: bool = False
    wdra_enabled: bool = False
    wasp_num_load_warps: int = None
    wasp_num_mma_warps: int = None
    wdra_num_load_regs: int = None
    wdra_num_mma_regs_main: int = None
    wdra_num_mma_regs_tail: int = None

    def __post_init__(self):
        object.__setattr__(self, "schedule_hint", ",".join(_schedule_hints(self.schedule_hint)) or "none")
        if type(self.register_batch_owner) is not bool:
            raise ValueError("register_batch_owner must be boolean")
        if self.register_batch_owner and self.arch != "gfx936":
            raise ValueError("register_batch_owner requires gfx936")
        if self.register_batch_owner and not self.optimize_epilogue:
            raise ValueError("register_batch_owner requires optimize_epilogue")
        if type(self.register_batch_owner_shared_vector) is not bool:
            raise ValueError("register_batch_owner_shared_vector must be boolean")
        if self.register_batch_owner_shared_vector and not self.register_batch_owner:
            raise ValueError("register_batch_owner_shared_vector requires register_batch_owner")
        if type(self.prefer_wide_async_copy) is not bool:
            raise ValueError("prefer_wide_async_copy must be boolean")
        if self.use_block_pingpong is not None and type(self.use_block_pingpong) is not bool:
            raise ValueError("use_block_pingpong must be bool or None")
        MemorySchedulePolicy.from_options(self)
        if type(self.structured_register_prepack_address) is not bool:
            raise ValueError("structured_register_prepack_address must be boolean")
        if self.structured_register_prepack_address and self.arch != "gfx936":
            raise ValueError("structured_register_prepack_address requires gfx936")
        if type(self.structured_register_unclamped_column) is not bool:
            raise ValueError("structured_register_unclamped_column must be boolean")
        if self.structured_register_unclamped_column and self.arch != "gfx936":
            raise ValueError("structured_register_unclamped_column requires gfx936")
        if type(self.priority_after_first_mmac) is not bool:
            raise ValueError("priority_after_first_mmac must be boolean")
        if self.priority_after_first_mmac and self.arch != "gfx936":
            raise ValueError("priority_after_first_mmac requires gfx936")
        if type(self.combine_shifted_disjoint_or) is not bool:
            raise ValueError("combine_shifted_disjoint_or must be boolean")
        if self.combine_shifted_disjoint_or and self.arch != "gfx936":
            raise ValueError("combine_shifted_disjoint_or requires gfx936")
        if type(self.terminal_readonly_publication) is not bool:
            raise ValueError("terminal_readonly_publication must be boolean")
        if self.terminal_readonly_publication and self.arch != "gfx936":
            raise ValueError("terminal_readonly_publication requires gfx936")
        if type(self.wave_private_sync) is not bool:
            raise ValueError("wave_private_sync must be boolean")
        if self.wave_private_sync and self.arch != "gfx936":
            raise ValueError("wave_private_sync requires gfx936")
        if type(self.padded_slot_intervals) is not bool:
            raise ValueError("padded_slot_intervals must be boolean")
        if self.padded_slot_intervals and self.arch != "gfx936":
            raise ValueError("padded_slot_intervals requires gfx936")
        if type(self.buffer_store_scalar_offset) is not bool:
            raise ValueError("buffer_store_scalar_offset must be boolean")
        if self.buffer_store_scalar_offset and self.arch != "gfx936":
            raise ValueError("buffer_store_scalar_offset requires gfx936")
        if type(self.buffer_store_split_immediate) is not bool:
            raise ValueError("buffer_store_split_immediate must be boolean")
        if self.buffer_store_split_immediate and self.arch != "gfx936":
            raise ValueError("buffer_store_split_immediate requires gfx936")
        if type(self.buffer_store_preserve_masked_base) is not bool:
            raise ValueError("buffer_store_preserve_masked_base must be boolean")
        if self.buffer_store_preserve_masked_base and self.arch != "gfx936":
            raise ValueError("buffer_store_preserve_masked_base requires gfx936")
        if type(self.interleave_global_lds_reads) is not bool:
            raise ValueError("interleave_global_lds_reads must be boolean")
        if type(self.interleave_global_lds_reads_fences) is not bool:
            raise ValueError("interleave_global_lds_reads_fences must be boolean")
        if (self.interleave_global_lds_reads or self.interleave_global_lds_reads_fences) and self.arch != "gfx936":
            raise ValueError("interleave_global_lds_reads requires gfx936")
        if self.interleave_global_lds_reads_fences and not self.interleave_global_lds_reads:
            raise ValueError("interleave_global_lds_reads_fences requires interleave_global_lds_reads")
        if type(self.tupled_register_fences) is not bool:
            raise ValueError("tupled_register_fences must be boolean")
        if self.tupled_register_fences and self.arch != "gfx936":
            raise ValueError("tupled_register_fences requires gfx936")
        if type(self.rebase_buffer_store) is not bool:
            raise ValueError("rebase_buffer_store must be boolean")
        if self.rebase_buffer_store and self.arch != "gfx936":
            raise ValueError("rebase_buffer_store requires gfx936")
        if type(self.lds_dma_alias_workaround) is not bool:
            raise ValueError("lds_dma_alias_workaround must be a bool")
        if self.lds_dma_alias_workaround and self.arch != "gfx936":
            raise ValueError("lds_dma_alias_workaround requires gfx936")
        if self.loop_unroll_stage not in ("ttir", "post-pipeline"):
            raise ValueError("loop_unroll_stage must be ttir or post-pipeline")
        if self.loop_unroll_stage == "post-pipeline" and self.arch != "gfx936":
            raise ValueError("post-pipeline loop unrolling requires gfx936")
        gfx_major = int(self.arch[3:-2])  # Drop "gfx" prefix and minor/patch number
        warp_size = 32 if gfx_major >= 10 else 64
        object.__setattr__(self, 'warp_size', warp_size)
        assert self.num_warps > 0 and (self.num_warps & (self.num_warps - 1)) == 0, \
            "num_warps must be a power of 2"
        if self.mmac_transposed and self.arch != "gfx936":
            raise ValueError("mmac_transposed is supported only on gfx936")

        if self.mmac_warps_m:
            m = self.mmac_warps_m
            if (self.arch != "gfx936" or m < 1 or m & (m - 1)
                    or m > self.num_warps or self.num_warps % m):
                raise ValueError("mmac_warps_m requires gfx936 and a power-of-two divisor of num_warps")

        if (self.arch == 'gfx950') and (self.kpack != 1):
            warnings.warn(
                f"kpack is deprecated starting from gfx950 and will be removed in later releases. So for now kpack = {self.kpack} will be overwritten to 1 to make transitioning easier."
            )
            object.__setattr__(self, 'kpack', 1)

        # default_libdir = Path(__file__).parent / 'lib'
        # HCU toolchain uses this path.
        default_libdir = Path(HIPBackend.path_to_rocm()) / 'amdgcn/bitcode/'
        extern_libs = {} if self.extern_libs is None else dict(self.extern_libs)
        for lib in ["ocml", "ockl", "hip", "opencl"]:  # Add hip and opencl for HCU toolchain.
            extern_libs[lib] = str(default_libdir / f'{lib}.bc')
        # rocshmem_device_lib = str(default_libdir / 'librocshmem_device.bc')

        object.__setattr__(self, 'extern_libs', tuple(extern_libs.items()))
        # object.__setattr__(self, 'rocshmem_device_lib', rocshmem_device_lib)

    def hash(self):
        key = '_'.join([f'{name}-{val}' for name, val in self.__dict__.items()])
        return hashlib.sha256(key.encode("utf-8")).hexdigest()


class HIPBackend(BaseBackend):
    instrumentation = None
    supports_native_tensor_specialization = False

    @staticmethod
    def supports_target(target: GPUTarget):
        return target.backend == 'hip'

    def __init__(self, target: GPUTarget) -> None:
        super().__init__(target)
        assert isinstance(target.arch, str)
        self.binary_ext = "hsaco"

    def get_target_name(self, options) -> str:
        return f"hip:{options.arch}"

    def parse_options(self, opts) -> Any:
        legacy_sched_variant = opts.pop("instruction_sched_variant", None)
        args = {'arch': knobs.runtime.override_arch or self.target.arch}

        if opts.get("num_ctas", 1) > 1 and not hcu.supports_multi_cta_launch(self.target.arch):
            raise ValueError(f"num_ctas > 1 not supported on {self.target.arch}")

        # Enable XF32 (TF32) for CDNA3 GPUs
        if self.target.arch == 'gfx942':
            allowed_dot_input_precisions = set(HIPOptions.allowed_dot_input_precisions)
            allowed_dot_input_precisions.update({'tf32'})
            args["allowed_dot_input_precisions"] = tuple(sorted(allowed_dot_input_precisions))

        if "supported_fp8_dtypes" not in opts:
            args["supported_fp8_dtypes"] = tuple(sorted(HIPOptions.supported_fp8_dtypes))

        if self.target.arch == 'gfx950':
            deprecated_fp8_dot_operand_dtypes = set(HIPOptions.deprecated_fp8_dot_operand_dtypes)
            deprecated_fp8_dot_operand_dtypes.update({"fp8e5b16", "fp8e4b8"})
            args["deprecated_fp8_dot_operand_dtypes"] = tuple(sorted(deprecated_fp8_dot_operand_dtypes))

        if "enable_fp_fusion" not in opts:
            args["enable_fp_fusion"] = knobs.language.default_fp_fusion

        if "optimize_epilogue" not in opts:
            args["optimize_epilogue"] = knobs.hcu.optimize_epilogue

        # dtk triton compatibility: consume legacy compiler.py options and ignore.
        opts.pop("num_ldmatrixes", None)
        opts.pop("enable_mmacfuse", None)
        opts.pop("reorder_instr", None)

        args.update({k: opts[k] for k in HIPOptions.__dataclass_fields__.keys() \
                     if k in opts and opts[k] is not None})

        # Use the effective arch after explicit overrides; explicit False remains an ablation.
        if "terminal_readonly_publication" not in args:
            args["terminal_readonly_publication"] = args["arch"] == "gfx936"

        # Persistent Config caches may represent booleans as binary integers.
        # Canonicalize only declared boolean options; retain strict validation
        # for other integers/types and keep compiler hashes stable on reload.
        for name, field in HIPOptions.__dataclass_fields__.items():
            value = args.get(name)
            if (field.type in (bool, Optional[bool]) and type(value) is int
                    and value in (0, 1)):
                args[name] = bool(value)

        # Resolve defaults after effective arch/epilogue and bool normalization.
        # Explicit False remains an ablation. No shape/config policy is used.
        register_owner_default = (args["arch"] == "gfx936"
                                  and args.get("optimize_epilogue", HIPOptions.optimize_epilogue) is True)
        if "register_batch_owner" not in args:
            args["register_batch_owner"] = register_owner_default
        if "register_batch_owner_shared_vector" not in args:
            args["register_batch_owner_shared_vector"] = (
                register_owner_default and args["register_batch_owner"] is True)

        # Consume the legacy `instruction_sched_variant` option and map it to
        # `schedule_hint`. If both are present, keep the explicit schedule_hint.
        if "schedule_hint" not in args and legacy_sched_variant is not None:
            args["schedule_hint"] = legacy_sched_variant

        # gfx936 has four SIMD units per CU. Split computation into two
        # full-SIMD wave groups only when the launch supplies enough waves.
        # This is an initial scheduling heuristic; actual loop/queue proofs
        # and final native readiness still decide whether it can be used.
        # Explicit options preserve independent manual experiments and off.
        if "post_mma_wait" not in args and "mma_phase_schedule" not in args:
            waves = args.get("num_warps", HIPOptions.num_warps)
            paired = (args["arch"] == "gfx936"
                      and type(waves) is int and waves >= 8 and waves % 8 == 0
                      and args.get("prefer_wide_async_copy", False) is True
                      and "local-prefetch" in _schedule_hints(args.get("schedule_hint", "none")))
            args.update(post_mma_wait=paired, mma_phase_schedule=paired)

        if args.get("wasp_enabled"):
            if args.get("wdra_enabled"):
                assert args["wasp_num_load_warps"] == 4
                assert args["wasp_num_mma_warps"] in [4, 8]
            else:
                args.pop("wdra_num_load_regs", None)
                args.pop("wdra_num_mma_regs_main", None)
                args.pop("wdra_num_mma_regs_tail", None)
        else:
            assert not args.get("wdra_enabled"), "wdra_enabled is only supported when wasp_enabled is True"
            args.pop("wasp_num_load_warps", None)
            args.pop("wasp_num_mma_warps", None)
            args.pop("wdra_num_load_regs", None)
            args.pop("wdra_num_mma_regs_main", None)
            args.pop("wdra_num_mma_regs_tail", None)

        if args.get("sync_lds_buffers", 1) not in (1, 2):
            raise ValueError("sync_lds_buffers must be 1 or 2")
        if args.get("sync_lds_buffers", 1) == 2 and args["arch"] != "gfx936":
            raise ValueError("sync_lds_buffers=2 requires gfx936")
        return HIPOptions(**args)

    def pack_metadata(self, metadata):
        return (
            metadata.num_warps,
            metadata.num_ctas,
            metadata.shared,
        )

    def get_codegen_implementation(self, options):
        return {"min_dot_size": get_min_dot_size(self.target)}

    def get_module_map(self) -> Dict[str, ModuleType]:
        from triton.language.extra.hip import libdevice
        # from triton.language.extra.hip import librocshmem_device
        from triton.language.extra.hip import libnvshmem_device

        return {
            "triton.language.extra.libdevice": libdevice, "triton.language.extra.libshmem_device": libnvshmem_device
        }

    def load_dialects(self, ctx):
        distributed.ir.load_dialects(ctx)
        hcu.load_dialects(ctx)
        if HIPBackend.instrumentation:
            HIPBackend.instrumentation.load_dialects(ctx)

    @staticmethod
    def is_within_2gb(arg):
        import torch

        MAX_INT_32 = 2**31 - 1
        if hasattr(arg, "ptr_range"):
            return arg.ptr_range() <= MAX_INT_32
        if isinstance(arg, torch.Tensor) and hasattr(arg, "untyped_storage"):
            return arg.untyped_storage().size() <= MAX_INT_32
        return False

    @staticmethod
    def parse_attr(desc):
        ret = BaseBackend.parse_attr(desc)
        if "S" in desc:
            ret += [["tt.pointer_range", 32]]
        return ret

    @staticmethod
    def get_tensor_specialization(arg, **kwargs):
        ret = BaseBackend.get_tensor_specialization(arg, **kwargs)
        if knobs.hcu.use_buffer_ops and HIPBackend.is_within_2gb(arg):
            ret += "S"
        return ret

    @staticmethod
    def path_to_rocm():
        rocm_path = os.getenv("ROCM_PATH")
        if rocm_path is not None:
            return rocm_path

        candidates = [Path("/opt/rocm"), Path("/opt/dtk")]
        for candidate in candidates:
            if candidate.is_dir():
                return str(candidate)

        return str(candidates[0])

    @staticmethod
    def path_to_rocm_llvm():
        rocm_path = Path(HIPBackend.path_to_rocm())
        llvm_subdir = "aillvm" if rocm_path.name == "dtk" else "llvm"
        return rocm_path / llvm_subdir

    @staticmethod
    def path_to_rocm_lld():
        llvm_path = HIPBackend.path_to_rocm_llvm()
        # Check env path for ld.lld
        lld_env_path = knobs.hcu.lld_path
        if lld_env_path is not None:
            lld = Path(lld_env_path)
            if lld.is_file():
                return lld
        # Check backend for ld.lld (used for pytorch wheels)
        lld = Path(__file__).parent / "llvm/bin/ld.lld"
        if lld.is_file():
            return lld
        lld = llvm_path / "bin/ld.lld"
        if lld.is_file():
            return lld
        lld = Path("/usr/bin/ld.lld")
        if lld.is_file():
            return lld
        raise Exception(f"ROCm linker not found under {llvm_path}/bin/ld.lld. "
                        "Set 'TRITON_HIP_LLD_PATH' to its path.")

    @staticmethod
    def path_to_rocm_clang():
        llvm_path = HIPBackend.path_to_rocm_llvm()
        # Check env path for clang
        clang_env_path = os.getenv("TRITON_HIP_CLANG_PATH",
                                   # By default, use clang-18
                                   str(llvm_path / "bin/clang-18"))
        if clang_env_path is not None:
            clang = Path(clang_env_path)
            if clang.is_file():
                return clang
        clang = llvm_path / "bin/clang"
        if clang.is_file():
            return clang
        clang = Path("/usr/bin/clang")
        if clang.is_file():
            return clang
        raise Exception(f"ROCm compiler not found under {llvm_path}/bin/clang. "
                        "Set 'TRITON_HIP_CLANG_PATH' to its path.")

    @staticmethod
    def _get_clang_args(metadata, options):
        arch_args = {
            "gfx928": [
                "-mllvm=-support-512-vgprs=true",
            ],
            "gfx936": [
                "-mllvm=-support-768-vgprs=true",
            ],
            "gfx938": [
                "-mllvm=-support-768-vgprs=true",
            ],
            "gfx946": [
                "-mllvm=-support-512-vgprs=true",
            ],
            "gfx92a": [
                "-mllvm=-support-512-vgprs=true",
            ],
        }
        if options.arch in arch_args:
            options_args = arch_args[options.arch]
        else:
            raise ValueError(f"Unknown arch: {options.arch}")

        version_args = {
            "18": [
                "-mllvm=-enable-hcu-approx-func-fp-math=true",
                "-mllvm=-hcu-update-wait-by-reverse-search=true",
            ],
        }
        clang_out = subprocess.check_output([HIPBackend.path_to_rocm_clang(), "--version"])
        match = re.search(r"version\s*(?P<major>\d+)\.(?P<minor>\d+)([\d.]+)?", clang_out.decode())
        clang_major = match.group("major")
        clang_minor = match.group("minor")
        if clang_major in version_args:
            options_args.extend(version_args[clang_major])

        if options.sched_latency != 'none':
            sched_latency_args = {
                "mmac5-ds10":
                ["-mllvm=-enable-latency-hack=true", "-mllvm=-mmac-latency=5", "-mllvm=-ds-load-store-latency=10"],
                "mmac5-ds6":
                ["-mllvm=-enable-latency-hack=true", "-mllvm=-mmac-latency=5", "-mllvm=-ds-load-store-latency=6"],
            }
            if options.sched_latency in sched_latency_args:
                options_args.extend(sched_latency_args[options.sched_latency])
            else:
                raise ValueError(f"Unsupported scheduling latency: {options.sched_latency}")

        clang_args = [
            "-target",
            hcu.TARGET_TRIPLE,
            f"-mcpu={options.arch}:xnack-",
            "-mllvm=-check-valu-data-forward-hazards=0",
            "-mllvm=-disable-cluster-lds-memops=true",
            # Note: when register spill after ds_read_matrix, result is wrong for compiler backend. disable current.
            "-mllvm=-hcu-pre-emit-load-store-opt=false",
            "-mllvm=-vgpr-greedy-alloc-mode=local-wave" if options.wdra_enabled else "",
            *options_args,
            "-O3",
        ]

        return clang_args

    @staticmethod
    def make_ttir(mod, metadata, options):
        if options.arch == "gfx936" and options.prefer_wide_async_copy:
            # Preserve block-pointer descriptors before scalar pointer expansion.
            mod.set_attr("ttg.hcu.prefer_wide_async_copy",
                         ir.builder(mod.context).get_int32_attr(1))
        pm = ir.pass_manager(mod.context)
        pm.enable_debug()
        passes.common.add_inliner(pm)
        passes.ttir.add_rewrite_tensor_pointer(pm)
        passes.ttir.add_rewrite_tensor_descriptor_to_pointer(pm)
        passes.common.add_canonicalizer(pm)
        passes.ttir.add_combine(pm)
        passes.ttir.add_reorder_broadcast(pm)
        passes.common.add_cse(pm)
        passes.ttir.add_triton_licm(pm)
        passes.common.add_symbol_dce(pm)
        if options.loop_unroll_stage == "ttir":
            passes.ttir.add_loop_unroll(pm)
        pm.run(mod, 'make_ttir')
        return mod

    @staticmethod
    def make_ttgir(mod, metadata, options):
        metadata["hcu_register_owner_default_policy"] = _REGISTER_OWNER_DEFAULT_POLICY
        if options.register_batch_owner:
            builder = ir.builder(mod.context)
            mod.set_attr("ttg.hcu.register_batch_owner", builder.get_bool_attr(True))
            mod.set_attr("ttg.hcu.register_batch_owner_count", builder.get_int32_attr(0))
        if options.register_batch_owner_shared_vector:
            builder = ir.builder(mod.context)
            mod.set_attr("ttg.hcu.register_batch_owner_shared_vector", builder.get_bool_attr(True))
            mod.set_attr("ttg.hcu.register_batch_owner_shared_vector_count", builder.get_int32_attr(0))
        if options.prefer_wide_async_copy:
            mod.set_attr("ttg.hcu.prefer_wide_async_copy",
                         ir.builder(mod.context).get_int32_attr(1))
        if options.wave_private_sync:
            if options.arch != "gfx936":
                raise ValueError("wave_private_sync requires gfx936")
            mod.set_attr("ttg.hcu.wave_private_sync", ir.builder(mod.context).get_int32_attr(1))
        if options.arch == "gfx936" and "local-prefetch" in _schedule_hints(options.schedule_hint):
            mod.set_attr("ttg.hcu.local_prefetch", ir.builder(mod.context).get_int32_attr(1))
        if options.sync_lds_buffers == 2:
            mod.set_attr("ttg.hcu.sync_lds_buffers", ir.builder(mod.context).get_int32_attr(2))
        if options.mmac_transposed:
            mod.set_attr("ttg.hcu.mmac_transposed",
                         ir.builder(mod.context).get_int32_attr(1))
        if options.mmac_warps_m:
            mod.set_attr("ttg.hcu.mmac_warps_m",
                         ir.builder(mod.context).get_int32_attr(options.mmac_warps_m))
        pm = ir.pass_manager(mod.context)
        pm.enable_debug()
        passes.ttir.add_convert_to_ttgpuir(pm, f"hip:{options.arch}", options.num_warps, options.warp_size,
                                           options.num_ctas)
        # flagtree tle: lower tle.extract_tile
        tle.passes.add_lower_extract_tile(pm)
        # flagtree tle: lower tle.insert_tile
        tle.passes.add_lower_insert_tile(pm)
        # Fuse global loads stored through local_ptr into the final shared
        # buffer before local-pointer lowering obscures the copy pattern.
        tle.passes.add_optimize_local_pointer_async_stores(pm)
        # TritonDistributed Extension
        # distributed.passes.ttir.add_convert_to_ttgpuir_ext(pm, f"hip:{options.arch}", options.num_warps,
        #                                                    options.warp_size, options.num_ctas)
        pm.run(mod, 'make_ttgir_early')
        pm = ir.pass_manager(mod.context)
        pm.enable_debug()
        emuTF32 = False
        passes.ttgpuir.add_coalesce(pm)
        passes.ttgpuir.add_process_shared_memory_hint(pm)  # flagtree hints
        passes.ttgpuir.add_f32_dot_tc(pm, emuTF32)
        passes.ttgpuir.add_remove_layout_conversions(pm)
        passes.ttgpuir.add_optimize_thread_locality(pm)
        # begin flagtree tle
        tle.passes.add_select_encodings(pm)
        tle.passes.add_insert_local_pointer_barriers(pm)
        tle.passes.add_optimize_local_pointer_loads(pm)
        tle.passes.add_optimize_local_pointer_stores(pm)
        # end flagtree tle
        hcu.passes.ttgpuir.add_accelerate_matmul(pm, options.arch, options.matrix_instr_nonkdim, options.kpack,
                                                 options.mmac_layout_force)
        passes.ttgpuir.add_remove_layout_conversions(pm)
        if options.optimize_epilogue:
            hcu.passes.ttgpuir.add_optimize_epilogue(pm)
        hcu.passes.ttgpuir.add_optimize_dot_operands(pm, options.arch)

        hcu.passes.ttgpuir.add_mls_encoding_insertion(pm)
        passes.ttgpuir.add_remove_layout_conversions(pm)

        hcu.passes.ttgpuir.add_hoist_layout_conversions(pm)

        passes.ttgpuir.add_fuse_nested_loops(pm)
        passes.common.add_canonicalizer(pm)
        passes.ttir.add_triton_licm(pm)
        passes.common.add_canonicalizer(pm)

        global_prefetch = getattr(knobs.hcu, "global_prefetch", 0)
        local_prefetch = getattr(knobs.hcu, "local_prefetch", 0)
        use_async_copy = knobs.hcu.use_async_copy

        # The `local-prefetch` scheduling variant requires turning on buffer ops.
        if "local-prefetch" in _schedule_hints(options.schedule_hint):
            global_prefetch = 1

        async_copy_single_buffer = options.async_copy_use_single_buffer and (options.num_stages == 2
                                                                             and not global_prefetch)
        hcu.passes.ttgpuir.add_mls_stream_pipeline(pm, options.num_stages, global_prefetch, async_copy_single_buffer)

        use_block_pingpong = is_pingpong_schedule_enabled(options.arch, use_async_copy, options.use_block_pingpong)
        hcu.passes.ttgpuir.add_schedule_loops(pm, options.num_stages)

        if not options.wasp_enabled:
            if hasattr(hcu.passes.ttgpuir, "add_stream_pipeline"):
                hcu.passes.ttgpuir.add_stream_pipeline(pm, options.num_stages, global_prefetch, local_prefetch,
                                                       use_async_copy, use_block_pingpong)
            else:
                hcu.passes.ttgpuir.add_pipeline(pm, use_async_copy, use_block_pingpong)
        # Keep the legacy knob as the default policy, but allow
        # tle.load(is_async=...) to override it per operation.  The HCU
        # stream pipeline consumes eligible loads first; this pass handles
        # explicit async loads that remain outside a pipelined loop.
        # Pipelining leaves dead source tt.loads until canonicalization. Drop
        # them before TLE lowering can turn them into side-effecting copies.
        passes.common.add_canonicalizer(pm)
        # Pipelining only consumes loop loads. Lower remaining descriptors
        # (e.g. fully unrolled/peeled dots) before pointer canonicalization
        # changes the scalar pointer ABI that matrix lowering still needs.
        hcu.passes.ttgpuir.add_mls_lowering_pass(pm)
        tle.passes.add_lower_async_load(pm)
        # Form actual BUFFER copies before GLOBAL-only width contraction.
        # Failed conversions remain GLOBAL and retain their original limits.
        early_buffer = (options.arch == "gfx936"
                and knobs.hcu.use_buffer_ops
                and not options.wasp_enabled and not use_block_pingpong
                and not is_in_thread_transpose_enabled(options.arch))
        # Ping-pong must still see synchronous tt.loads, including explicit
        # per-operation False overrides of the global async policy.
        pp_async_buffer = (options.arch == "gfx936"
                and knobs.hcu.use_buffer_ops
                and not options.wasp_enabled and use_block_pingpong
                and not is_in_thread_transpose_enabled(options.arch))
        early_buffer = early_buffer or pp_async_buffer
        if early_buffer:
            hcu.passes.ttgpuir.add_canonicalize_pointers(pm)
            passes.common.add_canonicalizer(pm)
            hcu.passes.ttgpuir.add_convert_to_buffer_ops(pm, options.arch,
                knobs.hcu.use_buffer_atomics, knobs.hcu.buffer_ops_analyze_small_tensor_range,
                async_only=pp_async_buffer)
        hcu.passes.ttgpuir.add_coalesce_async_copy(pm, options.arch)
        tle.passes.add_downgrade_invalid_async_copy(pm)
        passes.common.add_canonicalizer(pm)
        for hint in _schedule_hints(options.schedule_hint):
            hcu.passes.ttgpuir.insert_instruction_sched_hints(pm, hint)
        passes.ttgpuir.add_remove_layout_conversions(pm)

        # Descriptor-preserving MLS selection can expose the final consumer
        # layout only after the earlier operand pass. Retarget the original
        # register read before data-duplication lowering allocates LDS staging.
        if options.arch == "gfx936" and options.prefer_wide_async_copy:
            hcu.passes.ttgpuir.add_optimize_dot_operands(pm, options.arch)
        passes.ttgpuir.add_reduce_data_duplication(pm)
        if options.wave_private_sync:
            # Ordinary dot-operand conversions become local allocations here.
            # Reuse the existing producer coalescing/provenance policy after
            # those stores exist, so wave-private staging can see the full path.
            hcu.passes.ttgpuir.add_coalesce_async_copy(pm, options.arch)
            passes.ttgpuir.add_remove_layout_conversions(pm)
        if is_in_thread_transpose_enabled(options.arch):
            hcu.passes.ttgpuir.add_in_thread_transpose(pm)
            passes.ttgpuir.add_remove_layout_conversions(pm)
        hcu.passes.ttgpuir.add_reorder_instructions(pm)
        if use_block_pingpong and options.num_stages > 1:
            hcu.passes.ttgpuir.add_block_pingpong(pm, options.num_stages)

        if options.wasp_enabled:
            passes.ttgpuir.add_warp_specialize_hcu(pm, 2, options.wdra_enabled, options.wasp_num_load_warps,
                                                   options.wasp_num_mma_warps)
            hcu.passes.ttgpuir.add_accelerate_matmul(pm, options.arch, options.matrix_instr_nonkdim, options.kpack,
                                                     options.mmac_layout_force)
            passes.ttgpuir.add_remove_layout_conversions(pm)
            if options.optimize_epilogue:
                hcu.passes.ttgpuir.add_optimize_epilogue(pm)
            passes.ttgpuir.add_optimize_dot_operands(pm, True)
            hcu.passes.ttgpuir.add_hoist_layout_conversions(pm)

        if knobs.hcu.use_buffer_ops:
            # Existing BUFFER operations already have canonical scalar bases.
            # Rebuilding fat pointers would visit unsupported BUFFER users.
            # Keep the late conversion for any register-fallback tt.loads.
            if not early_buffer:
                hcu.passes.ttgpuir.add_canonicalize_pointers(pm)
            passes.common.add_canonicalizer(pm)
            hcu.passes.ttgpuir.add_convert_to_buffer_ops(
                pm,
                options.arch,
                knobs.hcu.use_buffer_atomics,
                knobs.hcu.buffer_ops_analyze_small_tensor_range,
            )

        hcu.passes.ttgpuir.add_fold_true_cmpi(pm)
        passes.common.add_canonicalizer(pm)
        passes.common.add_cse(pm)
        passes.common.add_symbol_dce(pm)
        # Layout propagation can expose a view conversion on a pipeline's
        # peeled result. Retarget its original completed shared reads only
        # after the final SSA cleanup has exposed unused carried arguments.
        hcu.passes.ttgpuir.add_mls_lowering_pass(pm)
        if 1:  #use_async_copy:
            hcu.passes.ttgpuir.add_update_async_wait_count(pm, options.arch)
        pm.run(mod, 'make_ttgir')
        metadata["tensordesc_meta"] = mod.get_tensordesc_metadata()
        return mod

    @staticmethod
    def gluon_to_ttgir(src, metadata, options):
        mod = src
        pm = ir.pass_manager(mod.context)
        pm.enable_debug()

        passes.gluon.add_inliner(pm)
        passes.gluon.add_resolve_auto_encodings(pm)
        passes.common.add_sccp(pm)
        passes.ttir.add_loop_aware_cse(pm)
        passes.gluon.add_canonicalizer(pm)
        passes.ttgpuir.add_combine_tensor_select_and_if(pm)

        pm.run(mod, 'gluon_to_ttgir')
        metadata["tensordesc_meta"] = mod.get_tensordesc_metadata()
        return mod

    @staticmethod
    def make_llir(src, metadata, options):
        mod = src
        if options.priority_after_first_mmac:
            # Typed fences and the MMAC lowering own the SSA scheduling
            # boundary. Never move final assembly instructions as text.
            mod.set_attr("ttg.hcu.priority_after_first_mmac",
                         ir.builder(mod.context).get_bool_attr(True))
        if options.terminal_readonly_publication:
            mod.set_attr("ttg.hcu.terminal_readonly_publication",
                         ir.builder(mod.context).get_bool_attr(True))
        # Gluon skips make_ttgir; apply this existing opt-in for both frontends.
        if options.wave_private_sync:
            mod.set_attr("ttg.hcu.wave_private_sync",
                         ir.builder(mod.context).get_int32_attr(1))
        manual_ws = (options.arch == "gfx936" and
                     not options.wasp_enabled and not options.wdra_enabled and
                     (hcu.has_warp_specialize(mod) or hcu.has_cta_barrier(mod)))
        explicit_ws = manual_ws
        base_warps = mod.get_int_attr("ttg.num-warps") if explicit_ws else 0
        if options.structured_register_prepack_address:
            mod.set_attr("ttg.hcu.structured_register_prepack_address",
                         ir.builder(mod.context).get_bool_attr(True))
        if options.structured_register_unclamped_column:
            mod.set_attr("ttg.hcu.structured_register_unclamped_column",
                         ir.builder(mod.context).get_bool_attr(True))
        if options.rebase_buffer_store:
            mod.set_attr("ttg.hcu.rebase_buffer_store",
                         ir.builder(mod.context).get_bool_attr(True))
        if options.padded_slot_intervals:
            mod.set_attr("ttg.hcu.padded_slot_intervals",
                         ir.builder(mod.context).get_bool_attr(True))
        if options.buffer_store_scalar_offset:
            mod.set_attr("ttg.hcu.buffer_store_scalar_offset",
                         ir.builder(mod.context).get_bool_attr(True))
        if options.buffer_store_split_immediate:
            mod.set_attr("ttg.hcu.buffer_store_split_immediate",
                         ir.builder(mod.context).get_bool_attr(True))
        if options.buffer_store_preserve_masked_base:
            mod.set_attr("ttg.hcu.buffer_store_preserve_masked_base",
                         ir.builder(mod.context).get_bool_attr(True))
        if options.loop_unroll_stage == "post-pipeline":
            late_pm = ir.pass_manager(mod.context)
            late_pm.enable_debug()
            passes.ttir.add_loop_unroll(late_pm)
            passes.common.add_canonicalizer(late_pm)
            passes.common.add_cse(late_pm)
            late_pm.run(mod, "loop_unroll_post_pipeline")
        if options.disable_llvm_loop_unroll:
            hcu.disable_llvm_loop_unroll(mod)
        # TritonGPU -> LLVM-IR (MLIR)
        pm = ir.pass_manager(mod.context)
        pm.enable_debug()
        hcu.passes.ttgpuir.add_update_async_wait_count(pm, options.arch)
        # custom_lds_size is an experimental parameter that defines amount of LDS available
        # for one thread block. Measured in bytes.
        #
        # If custom_lds_size = 0, pass will consider all LDS is available for one threads block,
        # LDS size is determined by provided arch name.
        custom_lds_size = 0
        hcu.passes.ttgpuir.add_optimize_lds_usage(pm, options.arch, custom_lds_size)
        if explicit_ws:
            # Fresh SCF trace and producer start IDs precede type/thread lowering.
            hcu.passes.ttgpuir.add_warp_specialize_to_llvm(
                pm, options.arch, 4, base_warps, False, 0, 0, 0)
        passes.convert.add_scf_to_cf(pm)
        if options.arch == "gfx936" and (
            options.loop_unroll_stage == "post-pipeline" or options.wave_private_sync
        ):
            # Normalize constant ring-buffer paths before allocation and the
            # wave-private provenance proof, independently of unroll placement.
            passes.common.add_sccp(pm)
            passes.common.add_canonicalizer(pm)
            passes.common.add_cse(pm)
            passes.common.add_canonicalizer(pm)
        passes.gluon.add_inliner(pm)
        passes.convert.add_index_to_llvmir(pm)

        hcu.passes.ttgpuir.add_allocate_shared_memory(pm)
        # instrumentation point here so we can override IRs above (e.g., ttir and ttgir)
        if HIPBackend.instrumentation:
            HIPBackend.instrumentation.patch("ttgpuir_to_llvmir", pm, mod.context)
        ## __HIP_FTZ is used to control the denorm flushing behavior of exp2 op as follows:
        ## 1. If __HIP_FTZ = 1, exp2 flushes denorms in input and output regardless
        ##    of the value of kernel arg `allow_flush_denorm`.
        ## 2. If __HIP_FTZ = 0, whether exp2 flushes denorms in input and output
        ##    depends on the value of kernel arg `allow_flush_denorm`.
        ## 3. __HIP_FTZ is default to 1 and not exposed as a kernel argument.
        ##    For now it is used as a controller for developers only.
        __HIP_FTZ = True
        hcu.passes.ttgpuir.add_to_llvmir(pm, options.arch, __HIP_FTZ)
        hcu.passes.ttgpuir.set_func_inline_attrs(pm, options.arch)
        # TritonDistributed Extension: distributed -> llvm
        distributed.passes.ttgpuir.hcu.add_distributed_to_llvm(pm, options.arch, __HIP_FTZ)
        passes.common.add_canonicalizer(pm)
        passes.common.add_cse(pm)

        passes.convert.add_cf_to_llvmir(pm)
        passes.convert.add_arith_to_llvmir(pm)
        passes.common.add_canonicalizer(pm)
        passes.common.add_cse(pm)
        passes.common.add_symbol_dce(pm)

        if explicit_ws:
            # Consume the same-process pre-CF proof and flatten verified regions.
            hcu.passes.ttgpuir.add_warp_specialize_to_llvm(
                pm, options.arch, 4, base_warps, False, 0, 0, 0)
            passes.convert.add_arith_to_llvmir(pm)
            passes.common.add_canonicalizer(pm)
            passes.common.add_cse(pm)
        if options.wasp_enabled:
            hcu.passes.ttgpuir.add_warp_specialize_to_llvm(pm, options.arch, options.wasp_num_load_warps,
                                                           options.wasp_num_mma_warps, options.wdra_enabled,
                                                           options.wdra_num_load_regs or 0,
                                                           options.wdra_num_mma_regs_main or 0,
                                                           options.wdra_num_mma_regs_tail or 0)
            passes.convert.add_arith_to_llvmir(pm)
            passes.common.add_canonicalizer(pm)
            passes.common.add_cse(pm)
            passes.common.add_symbol_dce(pm)

        if _schedule_hints(options.schedule_hint):
            hcu.passes.ttgpuir.lower_instruction_sched_hints(pm, options.arch, options.num_stages)

        # This can not be moved below the di_scope pass
        if HIPBackend.instrumentation:
            HIPBackend.instrumentation.patch("llvmir_to_llvm", pm, mod.context)

        if not knobs.compilation.disable_line_info and not knobs.compilation.dump_ir_extract_di_local_variables:
            passes.llvmir.add_di_scope(pm)

        hcu.passes.ttgpuir.add_builtin_func_to_llvmir(pm, __HIP_FTZ)
        # TritonDistributed Extension: libdevice -> llvm
        distributed.passes.ttgpuir.hcu.add_lib_device_to_llvmir(pm, __HIP_FTZ)
        pm.run(mod, 'make_llir')
        if options.register_batch_owner:
            metadata["hcu_register_batch_owner_count"] = (
                mod.get_int_attr("ttg.hcu.register_batch_owner_count") or 0)
        if options.register_batch_owner_shared_vector:
            metadata["hcu_register_batch_owner_shared_vector_count"] = (
                mod.get_int_attr("ttg.hcu.register_batch_owner_shared_vector_count") or 0)

        if knobs.compilation.dump_ir_extract_di_local_variables:
            # comments below on why separate it
            if not knobs.compilation.disable_line_info:
                pm = ir.pass_manager(mod.context)
                pm.enable_debug()
                passes.llvmir.add_di_scope(pm)
                pm.run(mod, 'make_llir.disable_line_info')

            # insert dbg intrinsic with several DI Attribute including source
            # var name and type info note: unknown reason for now, but this
            # pass and add_di_scope has to be run separately, otherwise if we
            # put them into previous pipline, it trigger a segmentfault without
            # any error message; could be due to a bug in mlir or pybind11
            pm = ir.pass_manager(mod.context)
            pm.enable_debug()
            passes.llvmir.add_di_local_variable(pm)
            pm.run(mod, 'make_llir.dump_ir_extract_di_local_variables')

        # LLVM-IR (MLIR) -> LLVM-IR (LLVM)
        llvm.init_targets()
        context = llvm.context()
        llvm_mod = llvm.to_module(mod, context)
        hcu.attach_target_triple(llvm_mod)
        target_features = ''
        if knobs.compilation.enable_asan:
            target_features = '+xnack'
        llvm.attach_datalayout(llvm_mod, hcu.TARGET_TRIPLE, options.arch, target_features)

        # Set various control constants on the LLVM module so that device
        # libraries can resolve references to them.
        hcu.set_isa_version(llvm_mod, options.arch)
        hcu.set_abi_version(llvm_mod, 500)
        hcu.set_bool_control_constant(llvm_mod, "__oclc_finite_only_opt", False)
        hcu.set_bool_control_constant(llvm_mod, "__oclc_correctly_rounded_sqrt32", True)
        hcu.set_bool_control_constant(llvm_mod, "__oclc_unsafe_math_opt", False)
        hcu.set_bool_control_constant(llvm_mod, "__oclc_wavefrontsize64", options.warp_size == 64)

        # WarpSpecialize Passes would set this attribute
        total_num_warps = src.get_int_attr("ttg.total-num-warps")
        total_num_warps = total_num_warps if total_num_warps is not None else options.num_warps
        if manual_ws:
            if not base_warps <= total_num_warps <= 16:
                raise RuntimeError("explicit warp-specialize launch exceeds the actual CTA domain")
            metadata["num_warps"] = total_num_warps
            metadata["hcu_manual_warp_specialize"] = True

        # Set kernel attributes first given this may affect later optimizations.
        fns = [fn for fn in llvm_mod.get_functions() if not fn.is_declaration()]
        # If wdra is enabled, this attribute is required by the LLVM backend.
        if options.wdra_enabled:
            fns[0].add_fn_attr("hcu-wdra-waves-per-tg", str(total_num_warps))
        # The public kernel should be kernel 0.
        fns[0].set_calling_conv(hcu.CALLING_CONV_AMDGPU_KERNEL)
        launch_warps = total_num_warps if explicit_ws else options.num_warps
        fns[0].add_fn_attr("amdgpu-flat-work-group-size", f"1,{launch_warps*options.warp_size}")
        if "memory-bound-attention" in _schedule_hints(options.schedule_hint):
            fns[0].add_fn_attr("amdgpu-sched-strategy", "iterative-ilp")
        fns[0].add_fn_attr("uniform-work-group-size", "true")
        # LLVM HCUGPU backend supports the attribute "hcugpu-waves-per-eu"="<min>[, <max>]".
        # This attribute may be attached to a kernel function definition and is an optimization hint.
        # <min> parameter specifies the requested minimum number of waves per EU, and optional <max> parameter
        # specifies the requested maximum number of waves per EU (must be >= <min> if specified).
        # If <max> is omitted, then there is no restriction on the maximum number of waves per EU other than
        # the one dictated by the hardware for which the kernel is compiled. Passing 0, 0 as <min>, <max>
        # implies the default behavior (no limits).
        # Specifying N, N forces LLVM to focus on a single register count, simplifies some heuristics
        # and may improve scheduling.
        # FIXME: Keep legacy HCU behavior from Triton 3.1.x, need check triton-llvm waves_per_eu {0} or {0, 0} why not eliminate
        # fns[0].add_fn_attr("amdgpu-waves-per-eu", f"{options.waves_per_eu}, {options.waves_per_eu}")
        fns[0].add_fn_attr("amdgpu-waves-per-eu", f"{options.waves_per_eu}")
        denormal_mode = "preserve-sign" if options.allow_flush_denorm else "ieee"
        fns[0].add_fn_attr("denormal-fp-math-f32", denormal_mode)
        if knobs.compilation.enable_asan:
            fns[0].add_fn_target_feature("+xnack")
            fns[0].add_fn_asan_attr()

        # Hint the compiler that we'd like the firmware to set the kernel arguments
        # to user SGPRs so that the kernel does not need to s_load its arguments
        # from memory.
        hcu.set_all_fn_arg_inreg(fns[0])
        metadata['use_nvshmem'] = False
        for k in llvm_mod.get_functions():
            if "nvshmem" in k.name and k.is_declaration():
                metadata['use_nvshmem'] = True
                break

        if knobs.compilation.enable_asan:
            default_libdir = Path(__file__).parent / 'lib'
            paths = [
                str(default_libdir / 'asanrtl.bc'),
                str(default_libdir / "ocml.bc"),
                str(default_libdir / "ockl.bc")
            ]
            llvm.link_extern_libs(llvm_mod, paths)
        elif options.extern_libs:
            paths = [path for (name, path) in options.extern_libs if hcu.need_extern_lib(llvm_mod, name)]
            if len(paths) > 0:
                llvm.link_extern_libs(llvm_mod, paths)

        # if options.rocshmem_device_lib and metadata['use_rocshmem']:
        if metadata['use_nvshmem']:
            default_libdir = Path(__file__).parent / 'lib'
            llvm.link_extern_libs(llvm_mod, [str(default_libdir / "libnvshmem_device.bc")])

        hcu.strip_synchronous_barrier_markers(llvm_mod, options.arch)
        # Consume dense descriptor destination extents before LLVM folds
        # dereferenceable(base + offset, size) into weaker base-only facts.
        # The existing wide-copy policy enables this path; no new tuning axis.
        # Object scopes still require independently proven physical bounds.
        if options.prefer_wide_async_copy:
            hcu.annotate_lds_object_scopes(llvm_mod, options.arch, options.warp_size)
        llvm.optimize_module(llvm_mod, llvm.OPTIMIZE_O3, options.arch, '', [], options.enable_fp_fusion)

        # Architectures with architected SGPRs store the workgroup id in ttmp9 (X) and ttmp7 (Y[15:0], Z[31:16]).
        # These attributes are used to determine if Z should be masked out when loading Y. They are inferred during
        # optimize_module from calls to @llvm.amdgcn.workgroup.id.x/y/z(). We cannot rely on this because a
        # dispatch dimensions might be used even if there is no program_id() call for it.
        if hcu.has_architected_sgprs(options.arch):
            fns[0].remove_fn_attr("amdgpu-no-workgroup-id-x")
            fns[0].remove_fn_attr("amdgpu-no-workgroup-id-y")
            fns[0].remove_fn_attr("amdgpu-no-workgroup-id-z")

        if knobs.hcu.scalarize_packed_fops:
            hcu.add_scalarize_packed_fops_llvm_pass(fns[0])

        # Get some metadata
        metadata["shared"] = src.get_int_attr("ttg.shared")
        metadata["profile_scratch_size"] = src.get_int_attr("ttg.profile_scratch_memory_size") or 0
        metadata["profile_scratch_align"] = src.get_int_attr("ttg.profile_scratch_memory_alignment") or 1

        hcu.cleanup_bitcode_metadata(llvm_mod)
        # Disable inlining of print related functions,
        # because inlining of these function could slow down compilation significantly
        hcu.disable_print_inline(llvm_mod)
        if options.arch == "gfx936" and "local-prefetch" in _schedule_hints(options.schedule_hint):
            hcu.prefer_ready_shared_loads(llvm_mod, options.arch)
        hcu.preserve_buffer_load_byte_offsets(llvm_mod, options.arch)
        if options.arch == "gfx936":
            native_version = _hcu_clang_version()
            if re.search(r"\bversion\s+18\.", native_version):
                hcu.preserve_lds_barrier_order(llvm_mod, options.arch)
        if options.loop_unroll_stage == "post-pipeline":
            hcu.annotate_lds_object_scopes(llvm_mod, options.arch, options.warp_size)
        hcu.lower_tagged_local_barriers(llvm_mod, options.arch)
        if options.lds_dma_alias_workaround:
            hcu.preserve_lds_dma_alias_scopes(llvm_mod, options.arch)
        if options.interleave_global_lds_reads:
            interleave = hcu.interleave_global_lds_reads(
                llvm_mod, options.arch, options.interleave_global_lds_reads_fences)
            for key, value in interleave.items():
                metadata["hcu_global_lds_read_interleave_" + key] = int(value)
        if options.tupled_register_fences:
            fences = hcu.tuple_register_fences(llvm_mod, options.arch)
            for key, value in fences.items():
                metadata["hcu_tupled_register_fences_" + key] = int(value)
        return str(llvm_mod)

    ## TODO: [HCU] integrate with rocshmem
    @staticmethod
    def make_amdgcn(src, metadata, options, _skip_schedules=False):
        amdgcn=HIPBackend._make_amdgcn_with_schedules(src,metadata,options,_skip_schedules)
        if options.coalesce_lgkm_waits:
            original = amdgcn
            try:
                from .waitcnt_coalesce import coalesce_lgkm_waits
                from .mma_phase import check_native_readiness
                candidate, report = coalesce_lgkm_waits(amdgcn, metadata["name"])
                metadata["hcu_lgkm_wait_coalesce_result"] = report["reason"]
                metadata["hcu_lgkm_wait_coalesce_deleted"] = 0
                metadata["hcu_lgkm_wait_coalesce_strengthened"] = 0
                if report["accepted"] and report["changed"]:
                    audit = check_native_readiness(candidate, metadata["name"])
                    metadata["hcu_lgkm_wait_coalesce_completion"] = audit
                    if audit["accepted"]:
                        amdgcn = candidate
                        metadata["hcu_lgkm_wait_coalesce_deleted"] = report["deleted_count"]
                        metadata["hcu_lgkm_wait_coalesce_strengthened"] = report["strengthened_count"]
                    else:
                        metadata["hcu_lgkm_wait_coalesce_result"] = "fallback: completion: " + audit["reason"]
            except Exception as error:
                amdgcn = original
                metadata["hcu_lgkm_wait_coalesce_deleted"] = 0
                metadata["hcu_lgkm_wait_coalesce_strengthened"] = 0
                metadata["hcu_lgkm_wait_coalesce_result"] = "fallback: " + str(error)
        return amdgcn

    @staticmethod
    def _make_amdgcn_with_schedules(src, metadata, options, _skip_schedules=False):
        # Scheduling preparations are late: cached LLIR stays exact, and all paths
        # (including LLVM FileSource) use the same native completion check.
        stage_original = src
        schedule_original = None
        # Retired experiment IR cannot recover its pre-schedule source and
        # operand-overlap did not prove cross-wave WAR. Do not silently accept
        # those old private attributes under a different scheduling policy.
        if any(marker in src for marker in (
                "hcu.packet.wait.rotation", "hcu.packet.post_mma_wait",
                "hcu.packet.operand_overlap.experimental")):
            raise HSACOError("Retired experimental schedule IR; recompile the unscheduled source")
        imported_schedule = any(marker in src for marker in
                                ("hcu.mma.phase.loop", "hcu.memory.post_mma_wait"))
        validate_imported_schedule = imported_schedule

        def fallback_schedules(reason):
            if options.post_mma_wait:
                metadata["hcu_post_mma_wait_result"] = "fallback: " + reason
                metadata["hcu_post_mma_wait_loops"] = 0
            if options.mma_phase_schedule:
                metadata["hcu_mma_phase_result"] = "fallback: " + reason
                metadata["hcu_mma_phase_loops"] = 0
            result = HIPBackend._make_amdgcn_with_schedules(stage_original, metadata, options,
                                           _skip_schedules=True)
            # An imported already-scheduled FileSource has no earlier source
            # to recover. Never let recursive fallback bypass its native guard.
            if imported_schedule:
                from .mma_phase import check_native_readiness
                audit = check_native_readiness(result, metadata["name"])
                if not audit["accepted"]:
                    raise HSACOError("Imported schedule IR failed native completion: " + audit["reason"])
            return result

        if options.arch == "gfx936" and not _skip_schedules:
            try:
                src, prepared = prepare_memory_schedule(
                    src, metadata, MemorySchedulePolicy.from_options(options),
                    llvm, options.arch, options.num_warps)
                if prepared:
                    schedule_original = stage_original
            except (RuntimeError, ValueError) as error:
                return fallback_schedules("preparation: " + str(error))
        # Run after all late LLVM scheduling transforms, before native selection.
        # Explicit off does not call the binding or alter the LLVM string.
        if options.combine_shifted_disjoint_or:
            src, sites, reason = llvm.combine_hcu_shifted_disjoint_or(src, options.arch)
            metadata["hcu_shifted_disjoint_or_sites"] = sites
            metadata["hcu_shifted_disjoint_or_result"] = reason
        # Find kernel names (there should only be one)
        # We get the name at the last possible step to accommodate `triton.compile`
        # on user-provided LLVM
        names = re.findall(r"define amdgpu_kernel void @([a-zA-Z_][a-zA-Z0-9_]*)", src)
        assert len(names) == 1
        metadata["name"] = names[0]
        # llvm -> hsaco
        flags = []
        # The sink-insts-to-avoid-spills flag asks LLVM backend to sink instructions
        # into loops to avoid register spills in the MachineSinking pass, while it
        # can also lead to regression in some cases. But from current observation,
        # the regression is not significant. It would be better to have some heuristics.
        if "attention" in _schedule_hints(options.schedule_hint):
            flags.append('sink-insts-to-avoid-spills')
        # features = '-real-true16' if 'gfx11' in options.arch else ''
        # amdgcn = llvm.translate_to_asm(src, hcu.TARGET_TRIPLE, options.arch, features, flags, options.enable_fp_fusion,
        #                                False)
        llir_file = asm_file = None
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix=".ll", delete=False) as f:
                llir_file = f.name
                f.write(str(src))

            asm_file = tempfile.mktemp(suffix=".amdgcn")

            clang_path = HIPBackend.path_to_rocm_clang()
            clang_args = HIPBackend._get_clang_args(metadata, options) + flags

            # Compile to ASM
            asm_command = [clang_path] + clang_args + [llir_file, "-S", "-o", asm_file]
            result = subprocess.run(asm_command, check=True, capture_output=True, text=True)
            if options.wdra_enabled:
                log = result.stdout + result.stderr
                print(log, flush=True)

            with open(asm_file, "r") as fd_out:
                amdgcn = fd_out.read()

        except subprocess.CalledProcessError as e:
            if schedule_original is not None:
                return fallback_schedules("target compilation")
            print(f"Compilation failed: {e.stderr}")
            raise HSACOError(f"Compilation failed: {e.stderr}") from e
        except IOError as e:
            if schedule_original is not None:
                return fallback_schedules("target I/O: " + str(e))
            print(f"File operation failed: {str(e)}")
            raise HSACOError(f"File operation failed: {str(e)}") from e
        finally:
            # Clean up temporary files
            for file in [llir_file, asm_file]:
                if file is not None and os.path.exists(file):
                    os.remove(file)
        if schedule_original is not None or validate_imported_schedule:
            from .mma_phase import check_native_readiness
            try:
                audit = check_native_readiness(amdgcn, metadata["name"])
            except Exception as error:
                if schedule_original is not None:
                    return fallback_schedules("native checker: " + str(error))
                raise HSACOError("Imported schedule native checker failed: " + str(error)) from error
            # Readiness of the final native code is separate from whether
            # each requested IR scheduling preparation actually applied.
            metadata["hcu_schedule_native_readiness"] = audit
            if options.post_mma_wait and metadata.get("hcu_post_mma_wait_loops", 0):
                metadata["hcu_post_mma_wait_result"] = audit["reason"]
            if options.mma_phase_schedule:
                metadata["hcu_mma_phase_result"] = audit["reason"]
            if not audit["accepted"]:
                if schedule_original is None:
                    raise HSACOError("Imported schedule IR failed native completion: " + audit["reason"])
                return fallback_schedules("native completion: " + audit["reason"])
        if knobs.hcu.dump_amdgcn:
            print("// -----// HCUGCN Dump //----- //")
            print(amdgcn)
        return amdgcn

    @staticmethod
    def make_hsaco(src, metadata, options):
        # target_features = ''
        # if knobs.compilation.enable_asan:
        #     target_features = '+xnack'
        # hsaco = hcu.assemble_amdgcn(src, options.arch, target_features)
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix=".s", delete=False) as f:
                asm_file = f.name
                f.write(str(src))

            hsaco_file = tempfile.mktemp(suffix=".hsaco")

            clang_path = HIPBackend.path_to_rocm_clang()
            clang_args = HIPBackend._get_clang_args(metadata, options)

            # Compile to HSACO
            hsaco_command = [clang_path] + clang_args + [asm_file, "-x", "assembler", "-o", hsaco_file]
            subprocess.run(hsaco_command, check=True, capture_output=True, text=True)

            with open(hsaco_file, "rb") as fd_out:
                ret = fd_out.read()

        except subprocess.CalledProcessError as e:
            print(f"Compilation failed: {e.stderr}")
            raise HSACOError(f"Compilation failed: {e.stderr}") from e
        except IOError as e:
            print(f"File operation failed: {str(e)}")
            raise HSACOError(f"File operation failed: {str(e)}") from e

        return ret

    def add_stages(self, stages, options, language):
        if language == Language.TRITON:
            stages["ttir"] = lambda src, metadata: self.make_ttir(src, metadata, options)
            stages["ttgir"] = lambda src, metadata: self.make_ttgir(src, metadata, options)
        elif language == Language.GLUON:
            stages["ttgir"] = lambda src, metadata: self.gluon_to_ttgir(src, metadata, options)
        stages["llir"] = lambda src, metadata: self.make_llir(src, metadata, options)
        stages["amdgcn"] = lambda src, metadata: self.make_amdgcn(src, metadata, options)
        stages["hsaco"] = lambda src, metadata: self.make_hsaco(src, metadata, options)
        if knobs.runtime.add_stages_inspection_hook is not None:
            knobs.runtime.add_stages_inspection_hook(self, stages, options, language, None)

    @functools.lru_cache()
    def hash(self):
        version = _hcu_clang_version()
        return f'{version}-{self.target}'
