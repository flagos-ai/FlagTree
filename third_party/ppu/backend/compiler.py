# Copyright (c) 2026 T-Head Semiconductor Co., Ltd. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files
# (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge,
# publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
# CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
# SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

from triton.backends.compiler import BaseBackend, DotCap, DotSupport, GPUTarget, Language
from triton._C.libtriton import ir, passes, llvm, ppu
try:
    from triton._C.libtriton import tle
except ImportError:
    tle = None
from triton import knobs

from dataclasses import dataclass
import functools
from typing import Any, Dict, Tuple, Optional
from types import ModuleType
import hashlib
import re
import tempfile
import os
import subprocess
from pathlib import Path

# TLE ops the PPU backend has no lowering for, mapped to the TLE API that emits
# them. Checked on the frontend-emitted TTIR so the error names the API the user
# called instead of surfacing as an opaque MLIR legalization failure much later.
# ttg.warp_specialize and ttg.tma_copy reach TTIR only from the TLE builtins:
# tl.range(warp_specialize=True) merely sets a loop attribute, and PPU never runs
# the pass that would turn it into an op.
PPU_UNSUPPORTED_TLE_OPS = {
    "ttg.warp_specialize": "tle.warp_specialize",
    "ttg.tma_copy": "tle.gpu.copy.tensor_descriptor",
    "tle.distributed_barrier": "tle.distributed_barrier",
    "tle.remote_pointers": "tle.remote",
}


def reject_unsupported_tle(mod):
    features = set()

    # mod.walk takes a void callback and cannot be interrupted from Python, so
    # collect during the walk and report every unsupported feature at once.
    def visit(op):
        feature = PPU_UNSUPPORTED_TLE_OPS.get(op.get_name())
        if feature is not None:
            features.add(feature)

    mod.walk(visit)
    if features:
        raise NotImplementedError("unsupported TLE feature on the 'ppu' backend: " + ", ".join(sorted(features)))


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


# Whole-byte dtypes covered by the inherited async-copy/descriptor movement paths
_MOVEMENT_DTYPES = ("fp8e4nv", "fp8e5", "fp8e4b15", "int8", "uint8", "int16", "uint16", "int32", "uint32", "int64",
                    "uint64", "fp16", "bf16", "fp32", "fp64")

# fp8e4b15 is upcast to f16 by the common semantic layer before resolve_dot
# runs (upstream legacy behavior), so it never reaches these rules
_FP8_DOT_DTYPES = ("fp8e4nv", "fp8e5")
_NATIVE_SAME_TYPE_DOT_DTYPES = ("fp16", "bf16", "fp32", "fp64")


def _product_name(capability: int) -> str:
    return "PPU 810E (cap80)" if capability == 80 else f"PPU (cap{capability})"


def _make_resolve_dot(capability: int):
    """resolve_dot rule: INT8xINT8->INT32 is native; FP8 dot is native
    from cap89, and a preserved FP16-promotion path (EMULATED) below;
    every other combination is a compile-time error."""
    product = _product_name(capability)

    def resolve_dot(a_dtype, b_dtype, acc_dtype, M, N, K):
        if a_dtype == "int8" and b_dtype == "int8":
            return DotCap(DotSupport.NATIVE)
        if a_dtype in _FP8_DOT_DTYPES and b_dtype in _FP8_DOT_DTYPES:
            if capability >= 89:
                return DotCap(DotSupport.NATIVE)
            return DotCap(DotSupport.EMULATED, diag=f"FP8 dot on {product} is not native: "
                          "emulated via FP16 promotion (native=false)")
        if a_dtype == b_dtype and a_dtype in _NATIVE_SAME_TYPE_DOT_DTYPES:
            return DotCap(DotSupport.NATIVE)
        return DotCap(
            DotSupport.UNSUPPORTED, diag=f"tl.dot: {a_dtype} x {b_dtype} is not supported on {product}; "
            "native alternatives: int8 / fp16 / bf16 / fp32 / fp64 dot")

    return resolve_dot


def _make_resolve_dot_scaled(capability: int):
    """resolve_dot_scaled rule: cap89 has a native scaled-MMA path for
    mxfp4; everything else decomposes to a promoted fp16/bf16 dot and is
    declared EMULATED with a compile-time warning."""
    product = _product_name(capability)

    def resolve_dot_scaled(lhs_format, rhs_format):
        if capability >= 89 and "e2m1" in (lhs_format, rhs_format):
            return DotCap(DotSupport.NATIVE)
        return DotCap(
            DotSupport.EMULATED, diag=f"tl.dot_scaled ({lhs_format} x {rhs_format}) on {product} is not native: "
            "decomposed to a promoted fp16/bf16 dot (native=false)")

    return resolve_dot_scaled


def get_irformatter():
    binary = "llvm-irformatter"
    paths = [
        os.environ.get("TRITON_IR_FORMATTER_PATH", ""),
        os.path.join(os.path.dirname(__file__), "bin", binary),
        os.path.join(os.environ.get("PPU_SDK"), "bin", binary)
    ]
    for bin in paths:
        if os.path.exists(bin) and os.path.isfile(bin):
            return bin
    raise RuntimeError("Cannot find irformatter")


@functools.lru_cache()
def get_ppu_llc():
    binary = "ppu-llc"
    paths = [
        os.environ.get("TRITON_PPU_LLC_PATH", ""),
        os.path.join(os.path.dirname(__file__), "bin", binary),
        os.path.join(os.environ.get("PPU_SDK"), "bin", binary)
    ]
    for bin in paths:
        if os.path.exists(bin) and os.path.isfile(bin):
            return bin
    raise RuntimeError("Cannot find ppu-llc")


@functools.lru_cache()
def get_ppu_llc_version():
    version = subprocess.run([get_ppu_llc(), "--version"], check=False, capture_output=True, text=True).stdout
    return version


@functools.lru_cache(None)
def file_hash(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def sm_arch_from_capability(capability: int):
    return f"sm_{capability}"


def llir_get_kernel_name(llir: str) -> str:
    '''
    Get kernel name from LLIR code.
    This Kernel name is required when launching the kernel.
    '''
    assert llir
    # Prefer functions with an explicit calling-convention kernel marker
    # (e.g. "ptx_kernel") over plain "define void @..." — TLE-Raw DSL
    # regions emit helper functions as plain "define void @edsl(...)" that
    # would otherwise shadow the real kernel entry point.
    for line in llir.split('\n'):
        line = line.strip()
        if re.match(r'define \w+_kernel void @', line):
            return line.split('(')[0].split('@')[1]
    for line in llir.split('\n'):
        line = line.strip()
        if line.startswith('define void @'):
            return line.split('(')[0][13:]


@dataclass(frozen=True)
class HGGCOptions:
    num_warps: int = 4
    num_ctas: int = 1
    num_stages: int = 3
    warp_size: int = 32
    # maxnreg corresponds to the tix parameter .maxnreg, which controls the
    # maximum number of 32-bit registers used by one thread.
    maxnreg: Optional[int] = None
    ppu_llc_options: Optional[str] = knobs.ppu.ppu_llc_options
    ir_override: Optional[str] = None  # filename of a user-defined IR (*.{ttir|ttgir|llir|tix})
    enable_fp_fusion: bool = True
    enable_reflect_ftz: bool = True  # ftz in libdevice
    launch_cooperative_grid: bool = False
    launch_pdl: bool = False
    supported_fp8_dtypes: Tuple[str] = ("fp8e5", "fp8e4b15")
    supported_fp8_cast_dtypes: Tuple[str] = ()
    custom_cast_fp8_dtypes: Tuple[str] = ("fp8e4b15", )
    async_copy_dtypes: Tuple[str] = ()
    descriptor_dtypes: Tuple[str] = ()
    deprecated_fp8_dot_operand_dtypes: Tuple[str] = ()
    default_dot_input_precision: str = "tf32"
    allowed_dot_input_precisions: Tuple[str] = ("tf32", "tf32x3", "ieee", 'bf16x3', 'bf16x6')
    max_num_imprecise_acc_default: bool = None
    extern_libs: dict = None
    debug: bool = False
    backend_name: str = 'ppu'
    sanitize_overflow: bool = True
    arch: str = None
    instrumentation_mode: str = ""

    def __post_init__(self):
        default_libdir = Path(__file__).parent / 'lib'
        extern_libs = {} if self.extern_libs is None else dict(self.extern_libs)
        if not extern_libs.get('libdevice', None):
            extern_libs['libdevice'] = knobs.ppu.libdevice_path or str(default_libdir / 'libdevice.ppu.bc')

        object.__setattr__(self, 'extern_libs', tuple(extern_libs.items()))
        assert self.num_warps > 0 and (self.num_warps & (self.num_warps - 1)) == 0, \
               "num_warps must be a power of 2"

    def hash(self):
        hash_dict = dict(self.__dict__)
        hash_dict["extern_libs"] = tuple((k, file_hash(v)) for k, v in sorted(hash_dict["extern_libs"]))
        key = "_".join([f"{name}-{val}" for name, val in sorted(hash_dict.items())])
        return hashlib.sha256(key.encode("utf-8")).hexdigest()


class PPUBackend(BaseBackend):
    instrumentation = None

    @staticmethod
    def supports_target(target: GPUTarget):
        return target.backend == "cuda"

    def _parse_arch(self, arch):
        pattern = r"^sm(\d+)$"
        match = re.fullmatch(pattern, arch)
        if not match:
            raise ValueError(f"TRITON_OVERRIDE_ARCH must have the form {pattern}")
        return int(match.group(1))

    def get_target_name(self, options) -> str:
        capability = self._parse_arch(options.arch)
        return f"ppu:{capability}"

    def __init__(self, target: GPUTarget) -> None:
        super().__init__(target)
        self.binary_ext = "hgbin"

    def parse_options(self, opts) -> Any:
        # Enable debug mode for ConSan, so device-side assertions are not optimized out
        if "instrumentation_mode" in opts and opts["instrumentation_mode"] == "consan":
            opts["debug"] = True

        args = {'arch': knobs.runtime.override_arch or f"sm{self.target.arch}"}
        args.update({k: opts[k] for k in HGGCOptions.__dataclass_fields__.keys() if k in opts if opts[k] is not None})
        capability = int(self._parse_arch(args["arch"]))

        if args.get("num_ctas", 1) > 1 and capability < 90:
            raise ValueError((f"num_ctas > 1 requires SM90+. "
                              f"Current target is sm_{capability}. This configuration will fail. "
                              f"Please set num_ctas=1 or target an SM90+ GPU."))

        if "supported_fp8_dtypes" not in args:
            supported_fp8_dtypes = set(HGGCOptions.supported_fp8_dtypes)
            if capability >= 80:
                # E4M3FN: cast is native from cap89, software on cap80
                supported_fp8_dtypes.add("fp8e4nv")
            args["supported_fp8_dtypes"] = tuple(sorted(supported_fp8_dtypes))

        if "supported_fp8_cast_dtypes" not in args:
            # every declared fp8 dtype has a hardware or software cast lowering
            args["supported_fp8_cast_dtypes"] = args["supported_fp8_dtypes"]

        if "custom_cast_fp8_dtypes" not in args:
            # cap80 has no e4m3 cvt instructions; fp8e4nv casts go through software
            args["custom_cast_fp8_dtypes"] = ("fp8e4b15", "fp8e4nv") if capability < 89 else ("fp8e4b15", )

        if "async_copy_dtypes" not in args:
            args["async_copy_dtypes"] = _MOVEMENT_DTYPES
        if "descriptor_dtypes" not in args:
            args["descriptor_dtypes"] = _MOVEMENT_DTYPES

        if "deprecated_fp8_dot_operand_dtypes" not in args:
            if capability >= 90:
                args["deprecated_fp8_dot_operand_dtypes"] = ("fp8e4b15", )

        if "enable_fp_fusion" not in args:
            args["enable_fp_fusion"] = knobs.language.default_fp_fusion

        args["max_num_imprecise_acc_default"] = 2**30 if capability == 90 else 0

        return HGGCOptions(**args)

    def pack_metadata(self, metadata):
        return (
            metadata.num_warps,
            metadata.num_ctas,
            metadata.shared,
        )

    def get_codegen_implementation(self, options):
        import triton.language.extra.ppu as ppu
        capability = int(self._parse_arch(options.arch))
        sw_cast = "fp8e4nv" in options.custom_cast_fp8_dtypes
        codegen_fns = {
            "convert_custom_types": ppu.convert_custom_float8_sub89 if sw_cast else ppu.convert_custom_float8,
            "min_dot_size": min_dot_size(self.target),
            "resolve_dot": _make_resolve_dot(capability),
            "resolve_dot_scaled": _make_resolve_dot_scaled(capability),
        }
        return codegen_fns

    def get_module_map(self) -> Dict[str, ModuleType]:
        from triton.language.extra.ppu import libdevice
        return {"triton.language.extra.libdevice": libdevice}

    def load_dialects(self, ctx):
        ppu.load_dialects(ctx)
        if PPUBackend.instrumentation:
            PPUBackend.instrumentation.load_dialects(ctx)

    @staticmethod
    def make_ttir(mod, metadata, opt, capability):
        reject_unsupported_tle(mod)
        pm = ir.pass_manager(mod.context)
        pm.enable_debug()
        passes.common.add_inliner(pm)
        ppu.passes.ttppugpuir.add_tle_promote_async_load_to_aiu(pm)
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
        passes.ttir.add_convert_to_ttgpuir(pm, f"ppu:{capability}", opt.num_warps, 32, opt.num_ctas)
        if tle is not None:
            tle.raw_passes.add_tle_convert_arg_to_memdesc(pm)
            tle.raw_passes.add_tle_remove_redundant_copy(pm)
            tle.passes.add_lower_extract_tile(pm)
            tle.passes.add_lower_insert_tile(pm)
            tle.passes.add_optimize_local_pointer_async_stores(pm)
        # optimize TTGIR
        passes.ttgpuir.add_coalesce(pm)
        passes.ttgpuir.add_f32_dot_tc(pm, emuTF32)
        passes.ttgpuir.add_remove_layout_conversions(pm)
        passes.ttgpuir.add_optimize_thread_locality(pm)
        if tle is not None:
            tle.passes.add_early_assign_memory_space(pm)
            tle.passes.add_select_encodings(pm)
            tle.passes.add_insert_local_pointer_barriers(pm)
            tle.passes.add_optimize_local_pointer_loads(pm)
            tle.passes.add_optimize_local_pointer_stores(pm)
        ppu.passes.ttgpuir.add_accelerate_matmul(pm)
        passes.ttgpuir.add_remove_layout_conversions(pm)
        passes.ttgpuir.add_optimize_dot_operands(pm, capability >= 80)
        if tle is not None:
            tle.passes.add_promote_local_store_staging(pm)
        passes.ttir.add_loop_aware_cse(pm)
        if capability // 10 in [8, 9]:
            passes.ttgpuir.add_fuse_nested_loops(pm)
            passes.common.add_canonicalizer(pm)
            passes.ttir.add_triton_licm(pm)
            passes.common.add_canonicalizer(pm)
            passes.ttgpuir.add_combine_tensor_select_and_if(pm)
            passes.ttgpuir.add_assign_latencies(pm, opt.num_stages)
            passes.ttgpuir.add_schedule_loops(pm)
            passes.ttgpuir.add_pipeline(pm, opt.num_stages, dump_enabled)
        else:
            passes.ttir.add_triton_licm(pm)
        passes.common.add_canonicalizer(pm)
        passes.ttir.add_loop_aware_cse(pm)
        passes.ttgpuir.add_prefetch(pm)
        passes.ttgpuir.add_optimize_dot_operands(pm, capability >= 80)
        passes.ttgpuir.add_coalesce_async_copy(pm)
        if tle is not None:
            tle.passes.add_downgrade_invalid_async_copy(pm)
        ppu.passes.ttppugpuir.add_aiu_lowering(pm)
        passes.ttgpuir.add_remove_layout_conversions(pm)
        passes.ttgpuir.add_reduce_data_duplication(pm)
        passes.ttgpuir.add_reorder_instructions(pm)
        if tle is not None:
            tle.passes.add_lower_async_load(pm)
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
        ppu.passes.ttppugpuir.add_aiu_lowering(pm)
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
        ppu.passes.ttgpuir.add_allocate_shared_memory_ppu(pm, capability)
        if knobs.compilation.instrumentation_mode == "consan":
            # Call ConcurrencySanitizerPass here, before allocating global scratch memory but after allocating tensor and shared
            passes.ttgpuir.add_concurrency_sanitizer(pm)
        passes.ttgpuir.add_allocate_global_scratch_memory(pm)
        if tle is not None:
            # Inline TLE DSL regions before TritonGPU->LLVM lowering so no
            # tle.dsl_region op survives into the conversion pipeline.
            tle.raw_passes.add_tle_dsl_region_inline(pm)
        # instrumentation point here so we can override IRs above (e.g., ttir and ttgir)
        if PPUBackend.instrumentation:
            PPUBackend.instrumentation.patch("ttgpuir_to_llvmir", pm, mod.context)
        ppu.passes.ttgpuir.add_to_llvmir(pm, capability)
        ppu.passes.ttgpuir.add_convert_libdevice_func_to_ppu(pm)
        passes.common.add_canonicalizer(pm)
        passes.common.add_cse(pm)
        ppu.passes.ttppugpuir.add_ppugpu_to_llvm(pm)
        passes.common.add_canonicalizer(pm)
        passes.common.add_cse(pm)
        passes.common.add_symbol_dce(pm)
        passes.convert.add_nvvm_to_llvm(pm)

        if not knobs.compilation.disable_line_info and not knobs.compilation.dump_ir_extract_di_local_variables:
            passes.llvmir.add_di_scope(pm)

        if PPUBackend.instrumentation:
            PPUBackend.instrumentation.patch("llvmir_to_llvm", pm, mod.context)

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
        if knobs.compilation.enable_asan:
            raise RuntimeError(
                "Address Sanitizer Error: Address sanitizer is currently only supported on the AMD backend")
        llvm_mod = llvm.to_module(mod, context)
        ppu.attach_datalayout(llvm_mod)

        if options.enable_reflect_ftz:
            ppu.set_reflect_ftz(llvm_mod)

        shared_size = src.get_int_attr("ttg.shared")
        for k in llvm_mod.get_functions():
            if not k.is_declaration() and k.is_external_linkage():
                ppu.set_reqntid(k)
                ppu.set_smemsize(k, shared_size)

        if options.extern_libs and ppu.has_extern_deps(llvm_mod):
            paths = [path for (name, path) in options.extern_libs]
            llvm.link_extern_libs(llvm_mod, paths)

        llvm.optimize_module(llvm_mod, llvm.OPTIMIZE_O3)

        # Get some metadata
        total_num_warps = src.get_int_attr("ttg.total-num-warps")
        if total_num_warps is not None:
            metadata["num_warps"] = total_num_warps
        metadata["shared"] = src.get_int_attr("ttg.shared")
        metadata["global_scratch_size"] = src.get_int_attr("ttg.global_scratch_memory_size")
        metadata["global_scratch_align"] = src.get_int_attr("ttg.global_scratch_memory_alignment")
        metadata["profile_scratch_size"] = src.get_int_attr("ttg.profile_scratch_memory_size") or 0
        metadata["profile_scratch_align"] = src.get_int_attr("ttg.profile_scratch_memory_alignment") or 1
        ret = str(llvm_mod)
        metadata["name"] = llir_get_kernel_name(ret)
        del llvm_mod
        del context
        return ret

    def make_hgbin(self, src, metadata, opt, capability):
        # Restore PPU target intrinsics that were disguised as ordinary external
        # symbols so the build-time LLVM could round-trip the TLE raw-DSL region
        # (see triton/experimental/tle/raw/cuda/runtime.py)
        src = src.replace("__flagtree_ppu_intrinsic__", "llvm.ppu.")
        with tempfile.NamedTemporaryFile(delete=False, mode="w", suffix=".tix") as fsrc:
            fsrc.write(src)
            fsrc.flush()

            irformatter = get_irformatter()
            fsrcformatted = fsrc.name + ".trans"
            format_cmd = f"{irformatter} {fsrc.name} -S -o {fsrcformatted}"
            try:
                subprocess.run(format_cmd, shell=True, check=True, capture_output=True, text=True)
            except subprocess.CalledProcessError as e:
                error_msg = (f"IR Formatter error: `{e.cmd}` failed with return code {e.returncode}\n"
                             f"IR Formatter stderr:\n{e.stderr or ''}\n"
                             f"IR Formatter reproduce command: {format_cmd}\n")
                print(f"""

================================================================
{error_msg}

{src}
================================================================
please share the reproducer above with Triton project.
""")
                raise RuntimeError(error_msg) from e

            ppullc = get_ppu_llc()

            debug_info = []
            if knobs.compilation.disable_line_info:
                # This option is ignored if used without -lineinfo
                debug_info += ["-lineinfo", "-suppress-debug-info"]
            elif knobs.ppu.disable_ppu_llc_opt:
                # Synthesize complete debug info
                debug_info += ["-g"]
            else:
                # Only emit line info
                debug_info += ["-lineinfo"]

            fmad = [] if opt.enable_fp_fusion else ["--fmad=false"]

            extra_options = [
                "--ppu-backend-options",
                "--ppu-blksync-schedule-boundary=false",
                "--ppu-backend-options",
                "--ppu-enable-rewrite-partial-reg-uses=true",
                "--ppu-backend-options",
                "--ppu-max-vreg-count=256",
                "--ppu-backend-options",
                "--max-analysis-recursion-depth=7",
                "--ppu-backend-options",
                "--enable-threadIdx-x-div32-always-uniform=true",
            ]

            # Disable ppu-llc optimizations if requested
            disable_opt = ["--opt-level", "0"] if knobs.ppu.disable_ppu_llc_opt else []

            # Accept more ppu-llc options if provided
            ppu_llc_extra_options = opt.ppu_llc_options.split(" ") if opt.ppu_llc_options else []

            arch = sm_arch_from_capability(capability)

            fsrc.name = fsrcformatted

            fbin = fsrc.name + ".o"
            log_file = fsrc.name + ".log"

            ppullc_cmd = [
                ppullc,
                *debug_info,
                *fmad,
                *extra_options,
                "-v",
                *disable_opt,
                *ppu_llc_extra_options,
                f"--gpu-name={arch}",
                fsrc.name,
                "-o",
                fbin,
            ]
            ppullc_cmd = " ".join(ppullc_cmd)
            try:
                subprocess.run(ppullc_cmd, shell=True, check=True, capture_output=True, text=True)
                if knobs.ppu.dump_compile_log:
                    with open(log_file, "a") as f:
                        f.write(f"ppu-llc reproduce command: {ppullc_cmd}\n")
            except subprocess.CalledProcessError as e:
                error_msg = (f"ppu-llc error: `{e.cmd}` failed with return code {e.returncode}\n"
                             f"ppu-llc stderr:\n{e.stderr or ''}\n"
                             f"ppu-llc reproduce command: {ppullc_cmd}\n")
                with open(log_file, "a") as f:
                    f.write(error_msg)
                print(f"""

================================================================
{error_msg}

{src}
================================================================
please share the reproducer above with Triton project.
""")
                raise RuntimeError(error_msg) from e

            if os.path.exists(fsrc.name):
                os.remove(fsrc.name)

            with open(fbin, "rb") as f:
                hgbin = f.read()
            if os.path.exists(fbin):
                os.remove(fbin)
        return hgbin

    def add_stages(self, stages, options, language):
        capability = self._parse_arch(options.arch)
        if language == Language.TRITON:
            stages["ttir"] = lambda src, metadata: self.make_ttir(src, metadata, options, capability)
            stages["ttgir"] = lambda src, metadata: self.make_ttgir(src, metadata, options, capability)
        elif language == Language.GLUON:
            stages["ttgir"] = lambda src, metadata: self.gluon_to_ttgir(src, metadata, options, capability)
        stages["llir"] = lambda src, metadata: self.make_llir(src, metadata, options, capability)
        stages["hgbin"] = lambda src, metadata: self.make_hgbin(src, metadata, options, self.target.arch)
        if knobs.runtime.add_stages_inspection_hook is not None:
            knobs.runtime.add_stages_inspection_hook(self, stages, options, language, capability)

    @functools.lru_cache()
    def hash(self):
        version = get_ppu_llc_version()
        return f'{version}-{self.target.arch}'
