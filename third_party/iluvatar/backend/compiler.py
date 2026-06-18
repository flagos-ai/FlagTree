from triton.backends.compiler import BaseBackend, GPUTarget, Language
from triton._C.libtriton import ir, passes, llvm, iluvatar
from triton import knobs
from triton.runtime.errors import PTXASError

from dataclasses import dataclass
import functools
from typing import Any, Dict, Tuple, Optional
from types import ModuleType
import hashlib
import re
import tempfile
import signal
import os
import subprocess
from pathlib import Path


def min_dot_size(target: GPUTarget):

    def check_dot_compatibility(lhs_type, rhs_type) -> Tuple[int, int, int]:  # [m, n, k]
        lhs_bitwidth = lhs_type.scalar.primitive_bitwidth
        rhs_bitwidth = rhs_type.scalar.primitive_bitwidth
        assert lhs_bitwidth == rhs_bitwidth, "lhs and rhs bitwidth must be the same"
        # For small M/N the input we can still use tensorcores with padding.
        if lhs_bitwidth == 8:
            return (1, 1, 32)
        else:
            return (1, 1, 16)

    return check_dot_compatibility


def get_ptxas(arch: int) -> knobs.NvidiaTool:
    return knobs.nvidia.ptxas_blackwell if arch >= 100 else knobs.nvidia.ptxas


@functools.lru_cache()
def get_ptxas_version(arch: int = 80):
    mock_ver = knobs.nvidia.mock_ptx_version
    if mock_ver is not None:
        return mock_ver  # This is not really a version of ptxas, but it is good enough for testing
    version = subprocess.check_output([get_ptxas(arch).path, "--version"]).decode("utf-8")
    return version


@functools.lru_cache()
def ptx_get_version(cuda_version) -> int:
    '''
    Get the highest PTX version supported by the current CUDA driver.
    '''
    assert isinstance(cuda_version, str)
    major, minor = map(int, cuda_version.split('.'))
    if major == 12:
        if minor < 6:
            return 80 + minor
        else:
            return 80 + minor - 1
    if major == 11:
        return 70 + minor
    if major == 10:
        return 63 + minor

    if major >= 13:
        base_ptx = 90
        return base_ptx + (major - 13) * 10 + minor

    raise RuntimeError("Triton only support CUDA 10.0 or higher, but got CUDA version: " + cuda_version)


def get_ptx_version_from_options(options, arch: int):
    ptx_version = options.ptx_version
    if ptx_version is None:
        cuda_version = get_ptxas(arch).version
        ptx_version = ptx_get_version(cuda_version)
    return ptx_version


@functools.lru_cache()
def get_features(options, arch: int):
    ptx_version = get_ptx_version_from_options(options, arch)

    # PTX 8.6 is the max version supported by llvm c1188642.
    #
    # To check if a newer PTX version is supported, increase this value
    # and run a test.  If it's not supported, LLVM will print a warning
    # like "+ptx8.4 is not a recognized feature for this target".
    llvm_ptx_version = min(86, ptx_version)
    features = f'+ptx{llvm_ptx_version}'
    return features


@functools.lru_cache(None)
def file_hash(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def sm_arch_from_capability(capability: int):
    if capability == 71:
        return "ivcore11"
    elif capability == 80:
        return "ivcore20"
    else:
        raise ValueError(f"Unsupported capability: {capability}")


@dataclass(frozen=True)
class CUDAOptions:
    num_warps: int = 4
    num_ctas: int = 1
    num_stages: int = 3
    warp_size: int = 64
    # maxnreg corresponds to the ptx parameter .maxnreg, which controls the
    # maximum number of 32-bit registers used by one thread.
    maxnreg: Optional[int] = None
    ptx_version: int = None
    ptx_options: Optional[str] = knobs.nvidia.ptxas_options
    ir_override: Optional[str] = None  # filename of a user-defined IR (*.{ttir|ttgir|llir|ptx})
    enable_fp_fusion: bool = True
    enable_reflect_ftz: bool = True  # ftz in libdevice
    launch_cooperative_grid: bool = False
    launch_pdl: bool = False
    supported_fp8_dtypes: Tuple[str] = ("fp8e5", "fp8e4b15")
    deprecated_fp8_dot_operand_dtypes: Tuple[str] = ()
    default_dot_input_precision: str = "tf32"
    allowed_dot_input_precisions: Tuple[str] = ("tf32", "tf32x3", "ieee", 'bf16x3', 'bf16x6')
    max_num_imprecise_acc_default: bool = None
    extern_libs: dict = None
    debug: bool = False
    backend_name: str = 'corex'
    sanitize_overflow: bool = True
    arch: str = None
    instrumentation_mode: str = ""
    use_sme: int = 0

    def __post_init__(self):
        default_libdir = Path(__file__).parent / 'lib'
        extern_libs = {} if self.extern_libs is None else dict(self.extern_libs)
        if not extern_libs.get('libdevice', None):
            extern_libs[
                'libdevice'] = knobs.iluvatar.libdevice_path or knobs.iluvatar.libcuda_path + '/nvvm/libdevice/libdevice.compute_bi.10.bc'

        object.__setattr__(self, 'extern_libs', tuple(extern_libs.items()))
        assert self.num_warps > 0 and (self.num_warps & (self.num_warps - 1)) == 0, \
               "num_warps must be a power of 2"

    def hash(self):
        hash_dict = dict(self.__dict__)
        hash_dict["extern_libs"] = tuple((k, file_hash(v)) for k, v in sorted(hash_dict["extern_libs"]))
        key = "_".join([f"{name}-{val}" for name, val in sorted(hash_dict.items())])
        return hashlib.sha256(key.encode("utf-8")).hexdigest()


class CUDABackend(BaseBackend):
    instrumentation = None

    @staticmethod
    def supports_target(target: GPUTarget):
        return target.backend == 'corex'

    def _parse_arch(self, arch):
        pattern = r"^sm(\d+)$"
        match = re.fullmatch(pattern, arch)
        if not match:
            raise ValueError(f"TRITON_OVERRIDE_ARCH must have the form {pattern}")
        return int(match.group(1))

    def get_target_name(self, options) -> str:
        capability = self._parse_arch(options.arch)
        return f"cuda:{capability}"

    def __init__(self, target: GPUTarget) -> None:
        super().__init__(target)
        self.binary_ext = "cubin"

    def parse_options(self, opts) -> Any:
        # Enable debug mode for ConSan, so device-side assertions are not optimized out
        if "instrumentation_mode" in opts and opts["instrumentation_mode"] == "consan":
            opts["debug"] = True

        args = {'arch': knobs.runtime.override_arch or f"sm{self.target.arch}"}
        args.update({k: opts[k] for k in CUDAOptions.__dataclass_fields__.keys() if k in opts if opts[k] is not None})
        capability = int(self._parse_arch(args["arch"]))

        if args.get("num_ctas", 1) > 1 and capability < 90:
            raise ValueError((f"num_ctas > 1 requires NVIDIA SM90+ (Hopper). "
                              f"Current target is sm_{capability}. This configuration will fail. "
                              f"Please set num_ctas=1 or target an SM90+ GPU."))

        if "supported_fp8_dtypes" not in args:
            supported_fp8_dtypes = set(CUDAOptions.supported_fp8_dtypes)
            if capability >= 89:
                supported_fp8_dtypes.add("fp8e4nv")
            args["supported_fp8_dtypes"] = tuple(sorted(supported_fp8_dtypes))

        if "deprecated_fp8_dot_operand_dtypes" not in args:
            if capability >= 90:
                args["deprecated_fp8_dot_operand_dtypes"] = ("fp8e4b15", )

        if "enable_fp_fusion" not in args:
            args["enable_fp_fusion"] = knobs.language.default_fp_fusion

        args["max_num_imprecise_acc_default"] = 2**30 if capability == 90 else 0

        return CUDAOptions(**args)

    def pack_metadata(self, metadata):
        return (
            metadata.num_warps,
            metadata.num_ctas,
            metadata.shared,
        )

    def get_codegen_implementation(self, options):
        import triton.language.extra.corex as corex
        capability = int(self._parse_arch(options.arch))
        codegen_fns = {
            "convert_custom_types":
            corex.convert_custom_float8_sm80 if capability >= 80 else corex.convert_custom_float8_sm70, "min_dot_size":
            min_dot_size(self.target)
        }
        return codegen_fns

    def get_module_map(self) -> Dict[str, ModuleType]:
        from triton.language.extra.corex import libdevice
        return {"triton.language.extra.libdevice": libdevice}

    def load_dialects(self, ctx):
        iluvatar.load_dialects(ctx)
        if CUDABackend.instrumentation:
            CUDABackend.instrumentation.load_dialects(ctx)

    @staticmethod
    def make_ttir(mod, metadata, opt, capability):
        pm = ir.pass_manager(mod.context)
        pm.enable_debug()
        passes.common.add_inliner(pm)
        passes.ttir.add_rewrite_tensor_pointer(pm)
        if capability // 10 < 9:
            passes.ttir.add_rewrite_tensor_descriptor_to_pointer(pm)
        passes.common.add_canonicalizer(pm)
        passes.ttir.add_combine(pm)
        passes.ttir.add_reorder_broadcast(pm)
        passes.common.add_cse(pm)
        passes.common.add_symbol_dce(pm)
        passes.ttir.add_loop_unroll(pm)
        pm.run(mod, 'make_ttir')
        return mod

    @staticmethod
    def make_ttgir(mod, metadata, opt, capability):
        # Set maxnreg on all kernels, if it was provided.
        if opt.maxnreg is not None:
            mod.set_attr("ttg.maxnreg", ir.builder(mod.context).get_int32_attr(opt.maxnreg))

        pm = ir.pass_manager(mod.context)
        dump_enabled = pm.enable_debug()
        emuTF32 = (capability // 10 >= 8)
        passes.ttir.add_convert_to_ttgpuir(pm, f"cuda:{capability}", opt.num_warps, opt.warp_size, opt.num_ctas)
        # optimize TTGIR
        passes.ttgpuir.add_coalesce(pm)
        passes.ttgpuir.add_f32_dot_tc(pm, emuTF32)
        passes.ttgpuir.add_remove_layout_conversions(pm)
        passes.ttgpuir.add_optimize_thread_locality(pm)
        if hasattr(iluvatar.passes, "tle"):
            iluvatar.passes.tle.add_insert_local_pointer_barriers(pm)
            iluvatar.passes.tle.add_optimize_local_pointer_loads(pm)
            iluvatar.passes.tle.add_optimize_local_pointer_stores(pm)
        iluvatar.passes.ttgpuir.add_accelerate_matmul(pm, opt.use_sme)
        passes.ttgpuir.add_remove_layout_conversions(pm)
        iluvatar.passes.ttgpuir.add_mma_reduce_thread_locality(pm)
        iluvatar.passes.ttgpuir.add_optimize_epilogue(pm)
        passes.ttgpuir.add_optimize_dot_operands(pm, capability >= 71)
        passes.ttir.add_loop_aware_cse(pm)
        if capability // 10 in [7, 8, 9]:
            passes.ttgpuir.add_fuse_nested_loops(pm)
            passes.common.add_canonicalizer(pm)
            passes.ttir.add_triton_licm(pm)
            passes.common.add_canonicalizer(pm)
            passes.ttgpuir.add_combine_tensor_select_and_if(pm)
            passes.ttgpuir.add_assign_latencies(pm, opt.num_stages)
            passes.ttgpuir.add_schedule_loops(pm)
        elif capability // 10 >= 10:
            passes.ttgpuir.add_fuse_nested_loops(pm)
            passes.common.add_canonicalizer(pm)
            passes.ttir.add_triton_licm(pm)
            passes.ttgpuir.add_optimize_accumulator_init(pm)
            passes.ttgpuir.add_hoist_tmem_alloc(pm, False)
            passes.ttgpuir.add_assign_latencies(pm, opt.num_stages)
            passes.ttgpuir.add_schedule_loops(pm)
            passes.ttgpuir.add_warp_specialize(pm, opt.num_stages)
            passes.ttgpuir.add_pipeline(pm, opt.num_stages, dump_enabled)
            passes.ttgpuir.add_optimize_partition_warps(pm)
            passes.ttgpuir.add_combine_tensor_select_and_if(pm)
            # hoist again and allow hoisting out of if statements
            passes.ttgpuir.add_hoist_tmem_alloc(pm, True)
        else:
            passes.ttir.add_triton_licm(pm)
        passes.common.add_canonicalizer(pm)
        passes.ttir.add_loop_aware_cse(pm)
        iluvatar.passes.ttgpuir.add_matmul_smeload(pm, capability)
        passes.ttgpuir.add_remove_layout_conversions(pm)
        passes.ttgpuir.add_prefetch(pm)
        passes.ttgpuir.add_optimize_dot_operands(pm, capability >= 71)
        passes.ttgpuir.add_coalesce_async_copy(pm)
        passes.ttgpuir.add_remove_layout_conversions(pm)
        passes.ttgpuir.add_reduce_data_duplication(pm)
        passes.ttgpuir.add_reorder_instructions(pm)
        passes.ttir.add_loop_aware_cse(pm)
        passes.common.add_symbol_dce(pm)
        passes.common.add_sccp(pm)
        passes.common.add_cse(pm)
        passes.common.add_canonicalizer(pm)

        pm.run(mod, 'make_ttgir')
        metadata["tensordesc_meta"] = mod.get_tensordesc_metadata()
        return mod

    def gluon_to_ttgir(self, src, metadata, options, capability):
        mod = src
        pm = ir.pass_manager(mod.context)
        pm.enable_debug()

        passes.gluon.add_inliner(pm)
        passes.gluon.add_infer_coalesced_encodings(pm)
        passes.gluon.add_resolve_auto_encodings(pm)
        passes.gluon.add_canonicalizer(pm)
        passes.common.add_sccp(pm)
        passes.ttir.add_loop_aware_cse(pm)
        passes.gluon.add_canonicalizer(pm)
        passes.ttgpuir.add_combine_tensor_select_and_if(pm)

        pm.run(mod, 'gluon_to_ttgir')
        metadata["tensordesc_meta"] = mod.get_tensordesc_metadata()
        return mod

    def make_llir(self, src, metadata, options, capability):
        mod = src
        # TritonGPU -> LLVM-IR (MLIR)
        pm = ir.pass_manager(mod.context)
        pm.enable_debug()

        passes.ttgpuir.add_combine_tensor_select_and_if(pm)
        passes.ttgpuir.add_allocate_warp_groups(pm)
        passes.convert.add_scf_to_cf(pm)
        passes.gluon.add_inliner(pm)
        passes.ttgpuir.add_allocate_shared_memory(pm)
        if knobs.compilation.instrumentation_mode == "consan":
            # Call ConcurrencySanitizerPass here, before allocating global scratch memory but after allocating tensor and shared
            passes.ttgpuir.add_concurrency_sanitizer(pm)
        passes.ttgpuir.add_allocate_global_scratch_memory(pm)
        if CUDABackend.instrumentation:
            CUDABackend.instrumentation.patch("ttgpuir_to_llvmir", pm, mod.context)
        proc = sm_arch_from_capability(capability)
        iluvatar.passes.ttgpuir.add_to_llvmir(pm, proc, options.enable_reflect_ftz)
        passes.common.add_canonicalizer(pm)
        passes.common.add_cse(pm)
        passes.common.add_canonicalizer(pm)
        passes.common.add_cse(pm)
        passes.common.add_symbol_dce(pm)
        passes.convert.add_nvvm_to_llvm(pm)

        if not knobs.compilation.disable_line_info and not knobs.compilation.dump_ir_extract_di_local_variables:
            passes.llvmir.add_di_scope(pm)

        if CUDABackend.instrumentation:
            CUDABackend.instrumentation.patch("llvmir_to_llvm", pm, mod.context)

        pm.run(mod, 'make_llir')

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
        iluvatar.attach_target_triple(llvm_mod)
        target_features = ''
        triple = iluvatar.TARGET_TRIPLE
        llvm.attach_datalayout(llvm_mod, triple, proc, target_features)

        fns = [fn for fn in llvm_mod.get_functions() if not fn.is_declaration()]
        if fns:
            fns[0].set_calling_conv(iluvatar.CALLING_CONV_ILUVATAR_KERNEL)
        if options.maxnreg and options.maxnreg > 0:
            fns[0].add_fn_attr("iluvatar-num-vgpr", f"{options.maxnreg}")

        if options.enable_reflect_ftz:
            iluvatar.set_nvvm_reflect_ftz(llvm_mod)

        if options.extern_libs and iluvatar.has_extern_deps(llvm_mod):
            paths = [path for (name, path) in options.extern_libs]
            llvm.link_extern_libs(llvm_mod, paths)

        llvm.optimize_module(llvm_mod, llvm.OPTIMIZE_O3)

        # Get some metadata
        # warp-specialization mutates num_warps
        total_num_warps = src.get_int_attr("ttg.total-num-warps")
        if total_num_warps is not None:
            metadata["num_warps"] = total_num_warps
        metadata["shared"] = src.get_int_attr("ttg.shared")
        metadata["tmem_size"] = src.get_int_attr("ttg.tensor_memory_size")
        metadata["global_scratch_size"] = src.get_int_attr("ttg.global_scratch_memory_size")
        metadata["global_scratch_align"] = src.get_int_attr("ttg.global_scratch_memory_alignment")
        metadata["profile_scratch_size"] = src.get_int_attr("ttg.profile_scratch_memory_size") or 0
        metadata["profile_scratch_align"] = src.get_int_attr("ttg.profile_scratch_memory_alignment") or 1
        ret = str(llvm_mod)
        del llvm_mod
        del context
        return ret

    def make_asm(self, src, metadata, opt, capability):
        triple = iluvatar.TARGET_TRIPLE
        proc = sm_arch_from_capability(capability)
        asm = llvm.translate_to_asm(src, triple, proc, '', [], opt.enable_fp_fusion, False)
        return asm

    def make_cubin(self, src, metadata, opt, capability):
        names = re.findall(r"define iluvatar_kernel void @([a-zA-Z_][a-zA-Z0-9_]*)", src)
        assert len(names) == 1
        metadata["name"] = names[0]
        triple = iluvatar.TARGET_TRIPLE
        proc = sm_arch_from_capability(capability)
        cubin = iluvatar.translate_llvmir_to_cubin(src, triple, proc, '', [], opt.enable_fp_fusion, False)
        return cubin

    def add_stages(self, stages, options, language):
        capability = self._parse_arch(options.arch)
        if language == Language.TRITON:
            stages["ttir"] = lambda src, metadata: self.make_ttir(src, metadata, options, capability)
            stages["ttgir"] = lambda src, metadata: self.make_ttgir(src, metadata, options, capability)
        elif language == Language.GLUON:
            stages["ttgir"] = lambda src, metadata: self.gluon_to_ttgir(src, metadata, options, capability)
        stages["llir"] = lambda src, metadata: self.make_llir(src, metadata, options, capability)
        stages["asm"] = lambda src, metadata: self.make_asm(src, metadata, options, capability)
        stages["cubin"] = lambda src, metadata: self.make_cubin(src, metadata, options, self.target.arch)
        if knobs.runtime.add_stages_inspection_hook is not None:
            knobs.runtime.add_stages_inspection_hook(self, stages, options, language, capability)

    @functools.lru_cache()
    def hash(self):
        return f'{self.target.arch}'
