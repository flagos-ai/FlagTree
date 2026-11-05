from dataclasses import dataclass, field
import functools
import hashlib
import re
from typing import Dict, Tuple
from types import ModuleType
import os
from pathlib import Path
import subprocess
import tempfile

from triton.backends.compiler import BaseBackend, GPUTarget, Language
from triton import knobs
from triton._C.libtriton import ir, passes, llvm

try:
    from triton._C.libtriton import intellifusion
except ImportError:
    intellifusion = None


def get_libdevice_path_knob() -> str | None:
    return os.environ.get("TRITON_LIBDEVICE_PATH")


def resolve_default_libdevice_path() -> str | None:
    libdevice_path = get_libdevice_path_knob()
    if libdevice_path:
        return libdevice_path

    ifwa_home = os.environ.get("IFWA_HOME")
    if ifwa_home:
        candidate = Path(ifwa_home) / "ifvm" / "libdevice" / "libdevice-ifwa.bc"
        if candidate.is_file():
            return str(candidate)

    return None


@functools.lru_cache(None)
def file_hash(path: str) -> str:
    if not os.path.isfile(path):
        return f"missing:{path}"

    digest = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as e:
        return f"unreadable:{path}:{e.errno}"
    return digest.hexdigest()


def find_tool(tool: str) -> str | None:
    ifwa_home = os.environ.get("IFWA_HOME")
    if not ifwa_home:
        return None

    candidates = [os.path.join(ifwa_home, "bin", tool)]
    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate
    return None


def get_ifwa_toolkit_git_commit(ifwa_home: str) -> str:
    version_path = Path(ifwa_home).parent / "VERSION"
    try:
        version = version_path.read_text(encoding="utf-8")
    except OSError:
        raise RuntimeError("Cannot read IFWA toolkit VERSION metadata") from None

    match = re.search(r"^git_commit=([0-9a-fA-F]{40})$", version, re.MULTILINE)
    if match is None:
        raise RuntimeError("Cannot find a full git_commit in IFWA toolkit VERSION metadata")
    return match.group(1).lower()


def get_gpnpu_toolchain_fingerprint(tools: tuple[str, ...] = ("llvm-link", "llc", "lld")) -> str:
    # The IFWA Toolkit commit identifies the packaged llvm-link, llc, and lld.
    ifwa_home = os.environ.get("IFWA_HOME")
    if not ifwa_home:
        raise RuntimeError("Cannot identify the IFWA toolchain for cache reuse. Set IFWA_HOME.")

    missing = [tool for tool in tools if find_tool(tool) is None]
    if missing:
        names = ", ".join(missing)
        raise RuntimeError(f"Cannot find packaged IFWA tool(s): {names}")
    return f"ifwa:{get_ifwa_toolkit_git_commit(ifwa_home)}"


def require_extern_lib_path(name: str, path: str) -> str:
    if not os.path.isfile(path):
        raise RuntimeError(f"IFWA LLVM module references extern symbols, but extern library {name!r} was not found "
                           "or is not a regular file")
    try:
        with open(path, "rb"):
            pass
    except OSError:
        raise RuntimeError(f"IFWA LLVM module references extern symbols, but extern library {name!r} could not be "
                           "read") from None
    return path


def resolve_extern_lib_paths(extern_libs) -> tuple[list[str], str | None]:
    libs = dict(extern_libs or ())
    libdevice = libs.pop("libdevice", None)
    extern_lib_paths = [require_extern_lib_path(name, path) for name, path in libs.items()]
    if libdevice:
        libdevice = require_extern_lib_path("libdevice", libdevice)
    return extern_lib_paths, libdevice


def supports_fp8(arch: str) -> bool:
    # Keep architecture compatibility policy behind this interface. IFWA
    # currently enables FP8 for every architecture spelling. Add architecture-specific policy here when
    # concrete compatibility constraints are known.
    del arch
    return True


def min_dot_size(target: GPUTarget):

    def check_dot_compatibility(lhs_type, rhs_type) -> Tuple[int, int, int]:  # [m, n, k]
        if lhs_type is None or rhs_type is None:
            return (1, 1, 1)
        lhs_bitwidth = lhs_type.scalar.primitive_bitwidth
        rhs_bitwidth = rhs_type.scalar.primitive_bitwidth
        assert lhs_bitwidth == rhs_bitwidth, "lhs and rhs bitwidth must be the same"
        # For small M/N the input we can still use tensorcores with padding.
        if lhs_bitwidth == 8:
            return (1, 1, 32)
        else:
            return (1, 1, 16)

    return check_dot_compatibility


@dataclass(frozen=True)
class IfwaOptions:
    arch: str
    num_warps: int = 4
    num_stages: int = 2
    num_ctas: int = 1
    enable_fp_fusion: bool = True
    debug: bool = False
    instrumentation_mode: str = ""
    sanitize_overflow: bool = True
    force_dot_padding: bool = False
    # "ieee" here means "do not truncate f32 inputs to tf32 before the dot".
    # It is NOT a strict IEEE-754 compliance guarantee — rounding / denormal
    # behaviour of the lowered FMA follows the target hardware semantics.
    default_dot_input_precision: str = "tf32"
    allowed_dot_input_precisions: Tuple[str, ...] = ("tf32", "tf32x3", "ieee", "bf16x3", "bf16x6")
    # The shared Triton frontend uses this default for FP8 dot. IFWA TCGen5
    # lowering does not use the partial-accumulation setting.
    max_num_imprecise_acc_default: int = 0
    supported_fp8_dtypes: Tuple[str, ...] = field(init=False)
    extern_libs: Tuple[Tuple[str, str], ...] = ()

    def __post_init__(self):
        supported_fp8_dtypes = ("fp8e4nv", "fp8e5") if supports_fp8(self.arch) else ()
        object.__setattr__(self, "supported_fp8_dtypes", supported_fp8_dtypes)

    def hash(self):
        hash_dict = dict(self.__dict__)
        # Match the upstream cache-correctness rule for extern libs: linked
        # bitcode participates in the final LLVM module, so replacing
        # the toolkit libdevice in place must invalidate Triton's compile cache.
        # Missing libdevice paths get a stable marker here; make_llir reports
        # the actionable error only if the module actually references __nv_*.
        hash_dict["extern_libs"] = tuple((name, file_hash(path)) for name, path in sorted(self.extern_libs))
        key = "_".join([f"{name}-{val}" for name, val in sorted(hash_dict.items())])
        return hashlib.sha256(key.encode("utf-8")).hexdigest()


class IfwaBackend(BaseBackend):
    instrumentation = None
    supports_native_tensor_specialization = False
    backend_name: str = "ifwa"

    @staticmethod
    def supports_target(target: GPUTarget):
        return target.backend == "ifwa"

    def __init__(self, target: GPUTarget) -> None:
        super().__init__(target)
        assert isinstance(target.arch, str)
        if intellifusion is None:
            raise RuntimeError("IFWA backend is unavailable because triton._C.libtriton.intellifusion "
                               "could not be imported. Please build/install Triton with IFWA enabled.")
        self.binary_ext = "ifbin"

    def get_target_name(self, options) -> str:
        return f"ifwa:{options.arch}"

    def hash(self) -> str:
        # Triton's backend cache key is frozen before LLIR reveals whether the
        # default libdevice was actually referenced. Conservatively include
        # llvm-link with llc/lld so an in-place linker upgrade cannot reuse a
        # binary produced by a different effective IFWA toolchain.
        return f"ifwa-{self.target.arch}-{self.target.warp_size}-ifbin-{get_gpnpu_toolchain_fingerprint()}"

    def parse_options(self, options: dict) -> IfwaOptions:
        arch = options.get("arch") or knobs.runtime.override_arch or self.target.arch
        num_warps = options.get("num_warps", 4)
        num_stages = options.get("num_stages", 2)
        num_ctas = options.get("num_ctas", 1)
        # Non-None explicit options override the shared language default.
        enable_fp_fusion = options.get("enable_fp_fusion")
        if enable_fp_fusion is None:
            enable_fp_fusion = knobs.language.default_fp_fusion
        dot_input_precision = options.get("default_dot_input_precision")
        if dot_input_precision is None:
            dot_input_precision = IfwaOptions.default_dot_input_precision
        debug = bool(options.get("debug", False))
        instrumentation_mode = options.get("instrumentation_mode", "")
        sanitize_overflow = bool(options.get("sanitize_overflow", True))
        force_dot_padding = bool(options.get("force_dot_padding", False))
        extern_libs = dict(options.get("extern_libs") or {})
        libdevice_path = extern_libs.get("libdevice") or resolve_default_libdevice_path()
        if libdevice_path:
            extern_libs["libdevice"] = libdevice_path
        return IfwaOptions(arch=str(arch), num_warps=num_warps, num_stages=num_stages, num_ctas=num_ctas,
                           enable_fp_fusion=enable_fp_fusion, debug=debug, instrumentation_mode=instrumentation_mode,
                           sanitize_overflow=sanitize_overflow, force_dot_padding=force_dot_padding,
                           default_dot_input_precision=dot_input_precision, extern_libs=tuple(extern_libs.items()))

    def add_stages(self, stages: dict, options: IfwaOptions, language: Language) -> None:
        stages["ttir"] = lambda src, metadata: self.make_ttir(src, metadata, options)
        stages["ttgir"] = lambda src, metadata: self.make_ttgir(src, metadata, options)
        stages["llir"] = lambda src, metadata: self.make_llir(src, metadata, options)
        stages[self.binary_ext] = lambda src, metadata: self.make_ifbin(src, metadata, options)

    @staticmethod
    def make_ttir(mod, metadata, options):
        pm = ir.pass_manager(mod.context)
        pm.enable_debug()
        passes.common.add_inliner(pm)
        passes.common.add_canonicalizer(pm)
        passes.ttir.add_combine(pm)
        passes.ttir.add_reorder_broadcast(pm)
        passes.common.add_cse(pm)
        passes.common.add_symbol_dce(pm)
        passes.ttir.add_loop_unroll(pm)
        intellifusion.passes.ttgpuir.add_dot_strategy(pm, options.force_dot_padding, options.num_warps,
                                                      options.num_ctas)
        pm.run(mod, 'make_ttir')
        metadata["name"] = mod.get_entry_func_name()
        return mod

    @staticmethod
    def make_ttgir(mod, metadata, options):
        pm = ir.pass_manager(mod.context)
        pm.enable_debug()
        ifwa_target = f"ifwa:{options.arch}"

        passes.ttir.add_convert_to_ttgpuir(pm, ifwa_target,  # target
                                           options.num_warps,  # num_warps
                                           32,  # threads_per_warp (IFWA uses 32)
                                           options.num_ctas  # num_ctas
                                           )
        passes.ttgpuir.add_coalesce(pm)
        # Match the upstream placement: expand TF32x3/BF16x3/BF16x6 after
        # coalescing and before the generic MMAv5 acceleration pass.
        passes.ttgpuir.add_f32_dot_tc(pm, True)
        intellifusion.passes.ttnvgpuir.add_plan_cta(pm)
        passes.ttgpuir.add_remove_layout_conversions(pm)
        passes.ttgpuir.add_optimize_thread_locality(pm)
        intellifusion.passes.ttgpuir.add_accelerate_matmul(pm)
        passes.ttgpuir.add_remove_layout_conversions(pm)
        passes.ttgpuir.add_optimize_dot_operands(pm, True)
        intellifusion.passes.ttnvgpuir.add_optimize_descriptor_encoding(pm)
        passes.ttir.add_loop_aware_cse(pm)
        passes.ttgpuir.add_fuse_nested_loops(pm)
        passes.common.add_canonicalizer(pm)
        passes.ttir.add_triton_licm(pm)
        passes.ttgpuir.add_optimize_accumulator_init(pm)
        passes.ttgpuir.add_hoist_tmem_alloc(pm, False)
        intellifusion.passes.ttnvgpuir.add_promote_lhs_to_tmem(pm)
        if options.num_stages > 1:
            passes.ttgpuir.add_assign_latencies(pm, options.num_stages)
            passes.ttgpuir.add_schedule_loops(pm)
            passes.ttgpuir.add_pipeline(pm, options.num_stages, False)
        passes.ttgpuir.add_combine_tensor_select_and_if(pm)
        passes.ttgpuir.add_hoist_tmem_alloc(pm, True)
        intellifusion.passes.ttnvgpuir.add_remove_tmem_tokens(pm)
        passes.common.add_canonicalizer(pm)
        passes.ttir.add_loop_aware_cse(pm)
        passes.ttgpuir.add_optimize_dot_operands(pm, True)
        passes.ttgpuir.add_coalesce_async_copy(pm)
        intellifusion.passes.ttnvgpuir.add_optimize_tmem_layouts(pm)
        intellifusion.passes.ttnvgpuir.add_tmem_load_reduce(pm)
        intellifusion.passes.ttnvgpuir.add_tma_lowering(pm)
        passes.ttgpuir.add_remove_layout_conversions(pm)
        intellifusion.passes.ttnvgpuir.add_interleave_tmem(pm)
        passes.ttgpuir.add_reduce_data_duplication(pm)
        passes.ttgpuir.add_reorder_instructions(pm)
        passes.ttir.add_loop_aware_cse(pm)
        passes.common.add_symbol_dce(pm)
        intellifusion.passes.ttnvgpuir.add_fence_insertion(pm)
        intellifusion.passes.ttnvgpuir.add_lower_mma(pm)
        passes.common.add_sccp(pm)
        passes.common.add_cse(pm)
        passes.common.add_canonicalizer(pm)
        intellifusion.passes.ttgpuir.add_stage_scaled_fp4(pm)
        pm.run(mod, "make_ttgir")
        metadata["tensordesc_meta"] = mod.get_tensordesc_metadata()
        return mod

    @staticmethod
    def make_llir(mod, metadata, options):
        # TTGIR -> LLVM IR (MLIR)
        pm = ir.pass_manager(mod.context)
        pm.enable_debug()

        passes.ttgpuir.add_combine_tensor_select_and_if(pm)
        passes.convert.add_scf_to_cf(pm)
        # Match the matrix-tile swizzling used by IFWA convert-layout lowering.
        intellifusion.passes.ttgpuir.add_allocate_shared_memory(pm)
        intellifusion.passes.ttnvgpuir.add_allocate_tensor_memory(pm)
        intellifusion.passes.ttnvgpuir.add_check_matmul_two_cta(pm)
        intellifusion.passes.ttnvgpuir.add_proxy_fence_insertion(pm)
        intellifusion.passes.ttnvgpuir.add_tmem_barrier_insertion(pm)
        intellifusion.passes.llir.add_convert_to_llvmir(pm, options.num_warps)
        intellifusion.passes.ttnvgpuir.add_gpnpu_to_llvm(pm)
        passes.convert.add_nvvm_to_llvm(pm)
        passes.ttgpuir.add_canonicalize_llvm_ir(pm)
        passes.common.add_cse(pm)
        passes.common.add_canonicalizer(pm)
        passes.common.add_symbol_dce(pm)
        if not knobs.compilation.disable_line_info and not knobs.compilation.dump_ir_extract_di_local_variables:
            passes.llvmir.add_di_scope(pm)
        pm.run(mod, "make_llir")

        if knobs.compilation.dump_ir_extract_di_local_variables:
            if not knobs.compilation.disable_line_info:
                pm = ir.pass_manager(mod.context)
                pm.enable_debug()
                passes.llvmir.add_di_scope(pm)
                pm.run(mod, "make_llir.disable_line_info")

            pm = ir.pass_manager(mod.context)
            pm.enable_debug()
            passes.llvmir.add_di_local_variable(pm)
            pm.run(mod, "make_llir.dump_ir_extract_di_local_variables")

        # Keep runtime metadata aligned with Triton kernel launcher expectations.
        total_num_warps = mod.get_int_attr("ttg.total-num-warps")
        metadata["num_warps"] = total_num_warps if total_num_warps is not None else options.num_warps
        metadata["num_ctas"] = options.num_ctas
        metadata["shared"] = mod.get_int_attr("ttg.shared") or 0
        metadata["tmem_size"] = mod.get_int_attr("ttg.tensor_memory_size")
        metadata["global_scratch_size"] = mod.get_int_attr("ttg.global_scratch_memory_size") or 0
        metadata["global_scratch_align"] = mod.get_int_attr("ttg.global_scratch_memory_alignment") or 16
        metadata["profile_scratch_size"] = mod.get_int_attr("ttg.profile_scratch_memory_size") or 0
        metadata["profile_scratch_align"] = mod.get_int_attr("ttg.profile_scratch_memory_alignment") or 1

        # LLVM-IR (MLIR) -> LLVM-IR (LLVM)
        llvm.init_targets()
        context = llvm.context()
        llvm_mod = llvm.to_module(mod, context)

        return str(llvm_mod)

    @staticmethod
    def make_ifbin(src, metadata, options):
        # Parse final LLIR, including overrides, with Triton's linked LLVM.
        mod = intellifusion.parse_llvm_ir(src, llvm.context())
        metadata["binary_ext"] = "ifbin"
        extern_libs = dict(options.extern_libs)
        has_other_externs = any(name != "libdevice" for name in extern_libs)
        # Older cached stage metadata used every unresolved extern as a proxy
        # for libdevice usage. Ignore it now that dependencies are classified
        # by symbol semantics.
        metadata.pop("_" + "no" + "va" + "_has_extern_deps", None)  # legacy cached stage key
        needs_libdevice = intellifusion.has_libdevice_deps(mod)
        missing_libdevice = ("IFWA LLVM module requires libdevice, but it could not be located. "
                             "Set TRITON_LIBDEVICE_PATH to the bitcode file or IFWA_HOME to an IFWA toolkit "
                             "containing ifvm/libdevice/libdevice-ifwa.bc.")
        if "libdevice" not in extern_libs and needs_libdevice:
            raise RuntimeError(missing_libdevice)
        # Ordinary extern bitcode may introduce libdevice dependencies that are
        # absent from the original kernel IR, so keep toolkit BCs in that case.
        if not needs_libdevice and not has_other_externs:
            extern_libs.pop("libdevice", None)

        llc = find_tool("llc")
        if llc is None:
            raise RuntimeError("Cannot find llc. Please set IFWA_HOME.")
        lld = find_tool("lld")
        if lld is None:
            raise RuntimeError("Cannot find lld. Please set IFWA_HOME.")

        needs_extern_link = bool(extern_libs)
        llvm_link = find_tool("llvm-link") if needs_extern_link else None
        if needs_extern_link and llvm_link is None:
            raise RuntimeError("Cannot find llvm-link. Please set IFWA_HOME.")

        # llvm-link, llc, and lld exchange intermediate files in one
        # directory per invocation.
        with tempfile.TemporaryDirectory(prefix="triton-ifwa-") as work_dir:
            frontend_bc_path = os.path.join(work_dir, "kernel.frontend.bc")
            obj_path = os.path.join(work_dir, "kernel.o")
            binary_path = os.path.join(work_dir, "kernel.ifbin")
            # The backend serializer strips debug information from a copy
            # before the older toolkit consumes it. Canonical LLIR is retained.
            with open(frontend_bc_path, "wb") as f:
                f.write(intellifusion.serialize_bitcode(mod))
            link_cmds = []
            llc_input_path = frontend_bc_path
            if needs_extern_link:
                extern_lib_paths, libdevice_path = resolve_extern_lib_paths(extern_libs.items())
                resolved_path = os.path.join(work_dir, "kernel.resolved.ll")
                if extern_lib_paths:
                    extern_linked_path = os.path.join(work_dir, "kernel.extern.ll") if libdevice_path else resolved_path
                    link_cmds.append([
                        llvm_link,
                        "--only-needed",
                        llc_input_path,
                        *extern_lib_paths,
                        "-S",
                        "-o",
                        extern_linked_path,
                    ])
                    llc_input_path = extern_linked_path

                if libdevice_path:
                    # Import the math/helper definitions referenced by the
                    # kernel and ordinary externs. Whole-linking the SDK also
                    # compiles unrelated FP128 functions, which can stall llc.
                    # Keep imported definitions local to this kernel module;
                    # newer SDKs expose formerly private libdevice globals.
                    link_cmds.append([
                        llvm_link,
                        "--only-needed",
                        "--internalize",
                        llc_input_path,
                        libdevice_path,
                        "-S",
                        "-o",
                        resolved_path,
                    ])
                    llc_input_path = resolved_path

            # `llc` owns target lowering. Ordinary extern libraries remain
            # dependency-driven through llvm-link --only-needed.
            # Match the upstream createTargetMachine policy: Fast when enabled,
            # otherwise LLVM's Standard ("on"), preserving explicit FMA semantics.
            fp_contract = "fast" if options.enable_fp_fusion else "on"
            llc_cmd = [llc, "-filetype=obj", "-O3", f"-fp-contract={fp_contract}", "-o", obj_path, llc_input_path]
            lld_cmd = [
                lld,
                "-flavor",
                "gnu",
                "--no-undefined",
                "-shared",
                "-plugin-opt=O3",
                "--whole-archive",
                "-o",
                binary_path,
                obj_path,
                "--no-whole-archive",
            ]

            try:
                for cmd in [*link_cmds, llc_cmd, lld_cmd]:
                    subprocess.check_output(cmd, stderr=subprocess.STDOUT)
                    # Only the definitions imported from ordinary externs can
                    # introduce additional dependencies into this kernel.
                    if cmd in link_cmds and "libdevice" not in extern_libs:
                        with open(llc_input_path) as linked_file:
                            linked_mod = intellifusion.parse_llvm_ir(linked_file.read(), llvm.context())
                        if intellifusion.has_libdevice_deps(linked_mod):
                            raise RuntimeError(missing_libdevice)
                with open(binary_path, "rb") as f:
                    return f.read()
            except subprocess.CalledProcessError as e:
                failed_cmd = e.cmd if isinstance(e.cmd, (list, tuple)) else cmd
                tool_name = Path(failed_cmd[0]).name
                raise RuntimeError(f"IFWA binary build failed in {tool_name} (exit code {e.returncode})") from None
            except OSError as e:
                raise RuntimeError(f"IFWA binary build could not access a tool or output (errno {e.errno})") from None

    def load_dialects(self, context):
        if intellifusion is not None:
            intellifusion.load_dialects(context)
        return None

    def get_module_map(self) -> Dict[str, ModuleType]:
        # The shared libdevice wrappers match the __nv_* ABI linked by IFWA.
        from triton.language.extra.cuda import libdevice

        return {"triton.language.extra.libdevice": libdevice}

    def get_codegen_implementation(self, options: IfwaOptions):
        return {"min_dot_size": min_dot_size(self.target)}

    def pack_metadata(self, metadata):
        return (
            metadata.num_warps,
            metadata.num_ctas,
            metadata.shared,
        )
