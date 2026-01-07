import ctypes
import functools
import hashlib
import os
import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, Optional, Tuple, Union

from triton._C.libtriton import ir, passes, ascend
from triton.backends.ascend.utils import (
    _check_bishengir_api_change,
    _check_bishengir_able_save_ir,
    _check_bishengir_is_regbased,
    _enable_unpublished_feature,
    _enable_print_ub_bits,
    _get_kernel_target,
    _get_llvm_path,
    _get_mlir_path,
    _get_npucompiler_path,
    _get_triton_adapter_opt_path,
    _is_ascend_sanitizer_enabled,
    _is_debug_line_info_disabled,
    _is_auto_map_parallel_blocks_enabled,
    downgrade_llir,
    force_disable_ffts,
)
from triton.backends.ascend.driver import (
    NPUUtils
)
from triton.backends.compiler import (
    AttrsDescriptor,
    BaseBackend,
    GPUTarget,
    register_descriptor,
)
from triton.runtime import driver
from triton.runtime.cache import get_dump_manager

try:
    import acl
    is_compile_on_910_95 = acl.get_soc_name().startswith("Ascend910_95")
except Exception as e:
    is_compile_on_910_95 = False

# TODO: materialize the concrete min shape
def min_dot_size(target: GPUTarget):
    return lambda lhsType, rhsType: (1, 1, 1)


def make_ttir(mod, metadata, opt):
    if "hash" not in metadata:
        metadata["hash"] = hashlib.sha256(f"{mod}-{metadata}".encode()).hexdigest()
    # the same optimize pass for triton-ir as all other backends
    pm = ir.pass_manager(mod.context)
    pm.enable_debug()
    passes.common.add_inliner(pm)
    passes.ttir.add_combine(pm)
    passes.common.add_canonicalizer(pm)
    passes.ttir.add_reorder_broadcast(pm)
    passes.common.add_cse(pm)
    passes.common.add_licm(pm)
    passes.common.add_symbol_dce(pm)
    passes.ttir.add_loop_unroll(pm)
    pm.run(mod)
    if opt.debug:
        dump_manager = get_dump_manager(metadata["hash"])
        print(f"Dumping intermediate results to {dump_manager.cache_dir}")
        dump_manager.put(str(mod), "kernel.ttir.mlir", binary=False)

    return mod


def ttir_to_linalg(mod, metadata, opt, *, named_ops=False):
    # use triton_adapter to lower Triton-MLIR to linalg
    # Get Triton-MLIR as string
    pm = ir.pass_manager(mod.context)
    pm.enable_debug()

    #  metadata.setdefault("enable_select_analysis", False)
    #  metadata.setdefault("compile_on_910_95", False)
    #  metadata.setdefault("force_simt_template", False)

    enable_nd2nz_on_vector = metadata["enable_nd2nz_on_vector"]
    enable_select_analysis = metadata["enable_select_analysis"]
    compile_on_910_95 = metadata["compile_on_910_95"]
    force_simt_template = metadata["force_simt_template"]
    enable_linearize = metadata["enable_linearize"]

    # Add pass here.
    # ascend.passes.convert.add_triton_to_linalg_pipeline(pm)
    if enable_linearize:
        ascend.passes.convert.add_triton_linearize(pm)
    ascend.passes.convert.add_triton_discretemaskaccessconversion(
    pm,
    compile_on_910_95 = compile_on_910_95,
    force_simt_template = force_simt_template,
    )
    ascend.passes.convert.add_triton_to_annotation(pm)
    ascend.passes.convert.add_triton_to_unstructure_incubated(
    pm,
    compile_on_910_95 = compile_on_910_95,
    force_simt_template = force_simt_template,
    )
    ascend.passes.convert.add_triton_to_hivm(pm)
    ascend.passes.convert.add_triton_to_hfusion(pm)
    ascend.passes.convert.add_triton_to_llvm(pm)
    ascend.passes.convert.add_bubble_up_operation(pm)
    ascend.passes.convert.add_triton_to_linalg_incubated(
    pm,
    global_kernel = False,
    named_ops = named_ops,
    enable_nd2nz_on_vector = enable_nd2nz_on_vector,
    enable_select_analysis = enable_select_analysis,
    compile_on_910_95 = compile_on_910_95,
    )
    pm.run(mod)
    return str(mod)

def linalg_to_llir(linalg: str, metadata, opt):
    with tempfile.TemporaryDirectory() as tmpdir:
        ttadapter_path = os.path.join(tmpdir, "kernel.ttadapter.mlir")
        llmlir_path = os.path.join(tmpdir, "kernel.llir.mlir")
        llir_path = os.path.join(tmpdir, "kernel.ll")
        Path(ttadapter_path).write_text(linalg)
        mlir_opt_path = _get_mlir_path("bin", "mlir-opt")
        # TritonAdapter-MLIR to LLVM-MLIR
        subprocess.check_call(
            [
                mlir_opt_path,
                ttadapter_path,
                "--convert-linalg-to-affine-loops",
                "--eliminate-empty-tensors",
                "--empty-tensor-to-alloc-tensor",
                "--one-shot-bufferize=allow-return-allocs-from-loops=true",
                "--lower-affine",
                "--convert-linalg-to-loops",
                "--convert-scf-to-cf",
                "--convert-cf-to-llvm",
                "--convert-arith-to-llvm",
                "--convert-math-to-llvm",
                "--convert-complex-to-llvm",
                "--convert-vector-to-llvm",
                "--convert-index-to-llvm",
                "--memref-expand",
                "--expand-strided-metadata",
                "--finalize-memref-to-llvm",
                "--convert-func-to-llvm",
                # Lowering memrefs creates more affine.apply ops.
                # Lowering these affine ops again creates further arith ops,
                # so we have to run these two passes again here.
                "--lower-affine",
                "--convert-arith-to-llvm",
                # Remove all unrealized casts created
                "--reconcile-unrealized-casts",
                "-o",
                llmlir_path,
            ]
        )
        if opt.debug:
            dump_manager = get_dump_manager(metadata["hash"])
            dump_manager.put(
                Path(llmlir_path).read_text(), "kernel.llir.mlir", binary=False
            )

        # LLVM-MLIR to LLVM-IR
        mlir_translate_path = _get_mlir_path("bin", "mlir-translate")
        subprocess.check_call(
            [mlir_translate_path, llmlir_path, "--mlir-to-llvmir", "-o", llir_path]
        )
        if opt.debug:
            dump_manager = get_dump_manager(metadata["hash"])
            dump_manager.put(Path(llir_path).read_text(), "kernel.ll", binary=False)

        return Path(llir_path).read_text()


def llir_to_cpuasm(llir: str, metadata, opt):
    # add metadata at final stage
    # Note: Compiled Kernel requires to estimate size of shared memory to occupy
    # Currently, CPU backend requires no limit on shared memory size
    metadata["shared"] = 1
    # We can get a function name (C naming) from
    # LLVM-IR by getting the first "define void @".
    fn_name = llir.split("define void @")[1].split("(")[0].strip()
    metadata["name"] = fn_name + " cpu"
    with tempfile.TemporaryDirectory() as tmpdir:
        src_path = os.path.join(tmpdir, "kernel.ll")
        linked_path = os.path.join(tmpdir, "kernel_linked.ll")
        dst_path = os.path.join(tmpdir, "kernel.s")

        llir = downgrade_llir(llir)
        if opt.debug:
            dump_manager = get_dump_manager(metadata["hash"])
            dump_manager.put(llir, "kernel_downgrade.ll", binary=False)

        Path(src_path).write_text(llir)

        linker_path = _get_llvm_path("bin", "llvm-link")
        libclc_path = _get_llvm_path("lib", "clc", "libspirv-aarch64--.bc")
        subprocess.check_call(
            [
                linker_path,
                src_path,
                libclc_path,
                "--only-needed",
                "-S",
                "-o",
                linked_path,
            ]
        )
        if opt.debug:
            dump_manager = get_dump_manager(metadata["hash"])
            dump_manager.put(
                Path(linked_path).read_text(), "kernel_linked.ll", binary=False
            )

        llc_path = _get_llvm_path("bin", "llc")
        subprocess.check_call([llc_path, linked_path, "-o", dst_path])
        if opt.debug:
            dump_manager = get_dump_manager(metadata["hash"])
            dump_manager.put(Path(dst_path).read_text(), "kernel.s", binary=False)

        # Actually it's text-format assembly.  Use read_text().
        return Path(dst_path).read_text()


def __get_metadata_attr_by_callback(lib, postfix: str, metadata, meta_key: str):
    func_symbol = metadata["kernel_name"] + postfix
    if hasattr(lib, func_symbol):
        callback_func = getattr(lib, func_symbol)
        callback_func.restype = ctypes.c_int64
        callback_func.argtypes = []
        metadata[meta_key] = callback_func()


def _parse_linalg_metadata(linalg: str, metadata: dict):
    """
    Parse Linalg IR to extract metadata required for NPU compilation.
    Extracts and updates the following fields in metadata:
      - mix_mode
      - kernel_name
      - tensor_kinds
      - shared (currently hardcoded)
      - name (combined kernel_name and mix_mode)

    Additionally, removes the mix_mode attribute from the IR.
    """
    # --- Regular expressions and examples ---

    # Example: mix_mode = "aiv" -> aiv
    MIX_MODE_REGEX = r'mix_mode\s*=\s*"([^"]+)"'

    # Example: parallel_mode = "mix_simd_simt" -> mix_simd_simt
    PARALLEL_MODE_REGEX = r'parallel_mode\s*=\s*"([^"]+)"'

    # Example: func.func @gather_sorted_kernel(%arg0: ...) -> gather_sorted_kernel
    KERNEL_NAME_REGEX = r"func\.func\s+@(\w+)"

    # Example: %arg1: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32} -> ('1', '0')
    TENSOR_KIND_REGEX = r'%arg(\d+):[^,)]*?\{[^}]*?tt\.tensor_kind\s*=\s*([^:\s}]+)\s*:[^}]*?\}'

    # Example removal:   ', mix_mode = "aiv"' → ''
    REMOVE_MIX_MODE_REGEX = r', mix_mode\s*=\s*"[^"]*"'

    # Note: Compiled Kernel requires to estimate size of shared memory to occupy
    # Currently, NPU backend does not limit on shared memory
    metadata["shared"] = 1
    # the mix mode is also encoded into metadata['name'] for runtime to distinguish
    metadata["mix_mode"] = re.search(MIX_MODE_REGEX, linalg).group(1)
    metadata["parallel_mode"] = re.search(PARALLEL_MODE_REGEX, linalg).group(1)
    metadata["kernel_name"] = re.search(KERNEL_NAME_REGEX, linalg).group(1)
    # Use while space to split kernel_name and mix_mode.
    # Check the function load_binary in npu_driver.py.
    metadata["name"] = metadata["kernel_name"] + " " + metadata["mix_mode"]
    # Parse all tensor kinds from arguments
    metadata["tensor_kinds"] = [int(kind) for _, kind in re.findall(TENSOR_KIND_REGEX, linalg)]
    # init the ub bits of triton kernel for inductor autotune using
    metadata["required_ub_bits"] = 0
    # remove the mix_mode attribute
    linalg = re.sub(REMOVE_MIX_MODE_REGEX, "", linalg)
    return linalg, metadata


def _parse_ttir_metadata(ttir: str, metadata: dict):
    """
    Parse TTIR to extract metadata required for NPU compilation.
    Extracts and updates the following fields in metadata:
      - kernel_name
      - shared (currently hardcoded)
    """
    # --- Regular expressions and examples ---
    # Example: tt.func @gather_sorted_kernel(%arg0: ...) -> gather_sorted_kernel
    KERNEL_NAME_REGEX = r"tt\.func\spublic\s+@(\w+)"

    # Example: %arg1: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32} -> ('1', '0')
    TENSOR_KIND_REGEX = r'%arg(\d+):[^,)]*?\{[^}]*?tt\.tensor_kind\s*=\s*([^:\s}]+)\s*:[^}]*?\}'

    # Note: Compiled Kernel requires to estimate size of shared memory to occupy
    # Currently, NPU backend does not limit on shared memory
    metadata["shared"] = 1
    # Note: Currently, for TTIR inputs, we only support vector kernels.
    metadata["mix_mode"] = "aiv"
    metadata["kernel_name"] = re.search(KERNEL_NAME_REGEX, ttir).group(1)
    metadata["name"] = metadata["kernel_name"] + " " + metadata["mix_mode"]
    # Parse all tensor kinds from arguments
    metadata["tensor_kinds"] = [int(kind) for _, kind in re.findall(TENSOR_KIND_REGEX, ttir)]
    return metadata


def get_common_bishengir_compile_options(metadata):
    bishengir_target = metadata['target'].arch
    bishengir_target_opt = f"--target={bishengir_target}"
    return [bishengir_target_opt]

def linalg_to_bin_enable_npu_compile_910_95(linalg: str, metadata, opt):
    linalg, metadata = _parse_linalg_metadata(linalg, metadata)
    with tempfile.TemporaryDirectory() as tmpdir:
        ttadapter_path = os.path.join(tmpdir, "kernel.ttadapter.mlir")
        Path(ttadapter_path).write_text(linalg)
        bin_file = os.path.join(tmpdir, "kernel")
        if _check_bishengir_api_change():
            bin_file_with_ext = "kernel.o"
        else:
            bin_file_with_ext = "kernel_reloc.o"
        bin_path = os.path.join(tmpdir, bin_file_with_ext)
        callback_path = os.path.join(tmpdir, "libkernel.so")
        _compile_option_list = get_common_bishengir_compile_options(metadata)
        multibuffer = metadata["multibuffer"]
        if multibuffer is not None:
            _compile_option_list += [
                f"--enable-auto-multi-buffer={multibuffer}",
            ]
        enable_auto_bind_sub_block = metadata["enable_auto_bind_sub_block"]
        if enable_auto_bind_sub_block is not None:
            _compile_option_list += [
                f"--enable-auto-bind-sub-block={enable_auto_bind_sub_block}",
            ]
        if force_disable_ffts():
            _compile_option_list += ["--disable-ffts"]
        if _is_ascend_sanitizer_enabled():
            _compile_option_list += ["--enable-sanitizer=true"]
        if not _is_debug_line_info_disabled():
            _compile_option_list += ["--enable-debug-info=true"]

        if _enable_print_ub_bits():
            _compile_option_list += ["--enable-print-memory-allocated-size"]

        enable_hivm_auto_cv_balance = metadata["enable_hivm_auto_cv_balance"]
        if enable_hivm_auto_cv_balance is not None:
            _compile_option_list += \
                [f"--enable-hivm-auto-cv-balance={enable_hivm_auto_cv_balance}"]

        unit_flag = metadata["unit_flag"]
        if unit_flag is not None:
            _compile_option_list += \
                [f"--enable-hivm-unit-flag-sync={unit_flag}"]

        inject_barrier_all = metadata["inject_barrier_all"]
        if inject_barrier_all is not None:
            _compile_option_list += \
                [f"--enable-hivm-inject-barrier-all-sync={inject_barrier_all}"]

        limit_auto_multi_buffer_only_for_local_buffer = metadata["limit_auto_multi_buffer_only_for_local_buffer"]
        if limit_auto_multi_buffer_only_for_local_buffer is not None:
            _compile_option_list += \
                [f"--limit-auto-multi-buffer-only-for-local-buffer={limit_auto_multi_buffer_only_for_local_buffer}"]

        set_workspace_multibuffer = metadata["set_workspace_multibuffer"]
        if set_workspace_multibuffer is not None:
            _compile_option_list += \
                [f"--set-workspace-multibuffer={set_workspace_multibuffer}"]

        tile_mix_vector_loop = metadata["tile_mix_vector_loop"]
        if tile_mix_vector_loop is not None:
            _compile_option_list += \
                [f"--tile-mix-vector-loop={tile_mix_vector_loop}"]

        tile_mix_cube_loop = metadata["tile_mix_cube_loop"]
        if tile_mix_cube_loop is not None:
            _compile_option_list += \
                [f"--tile-mix-cube-loop={tile_mix_cube_loop}"]

        auto_multi_buffer = metadata["limit_auto_multi_buffer_of_local_buffer"]
        if auto_multi_buffer is not None:
            _compile_option_list += \
                [f"--limit-auto-multi-buffer-of-local-buffer={auto_multi_buffer}"]

        if _is_auto_map_parallel_blocks_enabled():
            _compile_option_list += ["--enable-auto-blockify-loop"]
        npu_compiler_path = _get_npucompiler_path()
        if npu_compiler_path.endswith("bishengir-compile"):
            _compile_option_list += [
                "--enable-hfusion-compile=true",
                "--enable-triton-kernel-compile=true",
            ]
        cmd_list = (
            [npu_compiler_path, ttadapter_path]
            + _compile_option_list
            + ["-o", bin_file]
        )

        ret = subprocess.run(cmd_list, capture_output=True, check=True)
        match = re.search(r'UB\s+size\s*=\s*(\d+)\s*bits', ret.stdout.decode('utf-8'))
        if match:
            # get the ub bits of triton kernel from bisheng for inductor autotune using
            metadata["required_ub_bits"] = int(match.group(1))
        if Path(callback_path).is_file():
            lib = ctypes.CDLL(callback_path)
            __get_metadata_attr_by_callback(lib, "_infer_workspace_shape_function", metadata, "workspace_size")
            __get_metadata_attr_by_callback(lib, "_infer_sync_block_lock_num_function", metadata, "lock_num")
            __get_metadata_attr_by_callback(lib, "_infer_sync_block_lock_init_function", metadata, "lock_init_val")

        return Path(bin_path).read_bytes()


def linalg_to_bin_enable_npu_compile_A2_A3(linalg: str, metadata, opt):
    linalg, metadata = _parse_linalg_metadata(linalg, metadata)
    with tempfile.TemporaryDirectory() as tmpdir:
        ttadapter_path = os.path.join(tmpdir, "kernel.ttadapter.mlir")
        Path(ttadapter_path).write_text(linalg)
        bin_file = os.path.join(tmpdir, "kernel")
        if _check_bishengir_api_change():
            bin_file_with_ext = "kernel.o"
        else:
            bin_file_with_ext = "kernel_reloc.o"
        if _check_bishengir_is_regbased():
            bishengir_hivm_opt = "--reg-based=true"
        else:
            bishengir_hivm_opt = "--enable-hivm-compile=true"
        bin_path = os.path.join(tmpdir, bin_file_with_ext)
        callback_path = os.path.join(tmpdir, "libkernel.so")
        _compile_option_list = [
            f"--target={NPUUtils().get_arch()}",
        ]
        multibuffer = metadata["multibuffer"]
        if multibuffer is not None:
            _compile_option_list += [
                f"--enable-auto-multi-buffer={multibuffer}",
            ]
        enable_auto_bind_sub_block = metadata["enable_auto_bind_sub_block"]
        if enable_auto_bind_sub_block is not None:
            _compile_option_list += [
                f"--enable-auto-bind-sub-block={enable_auto_bind_sub_block}",
            ]
        if _is_ascend_sanitizer_enabled():
            _compile_option_list += ["--enable-sanitizer=true"]
        if not _is_debug_line_info_disabled():
            _compile_option_list += ["--enable-debug-info=true"]

        if _enable_print_ub_bits():
            _compile_option_list += ["--enable-print-memory-allocated-size"]

        enable_hivm_auto_cv_balance = metadata["enable_hivm_auto_cv_balance"]
        if enable_hivm_auto_cv_balance is not None:
            _compile_option_list += \
                [f"--enable-hivm-auto-cv-balance={enable_hivm_auto_cv_balance}"]

        sync_solver = metadata["sync_solver"]
        if sync_solver is not None:
            _compile_option_list += \
                [f"--enable-hivm-graph-sync-solver={sync_solver}"]

        unit_flag = metadata["unit_flag"]
        if unit_flag is not None:
            _compile_option_list += \
                [f"--enable-hivm-unit-flag-sync={unit_flag}"]

        inject_barrier_all = metadata["inject_barrier_all"]
        if inject_barrier_all is not None:
            _compile_option_list += \
                [f"--enable-hivm-inject-barrier-all-sync={inject_barrier_all}"]

        inject_block_all = metadata["inject_block_all"]
        if inject_block_all is not None:
            _compile_option_list += \
                [f"--enable-hivm-inject-block-all-sync={inject_block_all}"]

        limit_auto_multi_buffer_only_for_local_buffer = metadata["limit_auto_multi_buffer_only_for_local_buffer"]
        if limit_auto_multi_buffer_only_for_local_buffer is not None:
            _compile_option_list += \
                [f"--limit-auto-multi-buffer-only-for-local-buffer={limit_auto_multi_buffer_only_for_local_buffer}"]

        set_workspace_multibuffer = metadata["set_workspace_multibuffer"]
        if set_workspace_multibuffer is not None:
            _compile_option_list += \
                [f"--set-workspace-multibuffer={set_workspace_multibuffer}"]

        tile_mix_vector_loop = metadata["tile_mix_vector_loop"]
        if tile_mix_vector_loop is not None:
            _compile_option_list += \
                [f"--tile-mix-vector-loop={tile_mix_vector_loop}"]

        tile_mix_cube_loop = metadata["tile_mix_cube_loop"]
        if tile_mix_cube_loop is not None:
            _compile_option_list += \
                [f"--tile-mix-cube-loop={tile_mix_cube_loop}"]

        auto_multi_buffer = metadata["limit_auto_multi_buffer_of_local_buffer"]
        if auto_multi_buffer is not None:
            _compile_option_list += \
                [f"--limit-auto-multi-buffer-of-local-buffer={auto_multi_buffer}"]

        disable_auto_inject_block_sync = metadata["disable_auto_inject_block_sync"]
        if disable_auto_inject_block_sync is not None:
            _compile_option_list += \
                [f"--disable-auto-inject-block-sync={disable_auto_inject_block_sync}"]

        if _is_auto_map_parallel_blocks_enabled():
            _compile_option_list += ["--enable-auto-blockify-loop"]
        npu_compiler_path = _get_npucompiler_path()
        if npu_compiler_path.endswith("bishengir-compile"):
            _compile_option_list += [
                "--enable-hfusion-compile=true",
                bishengir_hivm_opt,
                "--enable-triton-kernel-compile=true",
            ]
        cmd_list = (
            [npu_compiler_path, ttadapter_path]
            + _compile_option_list
            + ["-o", bin_file]
        )
        ret = subprocess.run(cmd_list, capture_output=True, check=True)
        match = re.search(r'UB\s+size\s*=\s*(\d+)\s*bits', ret.stdout.decode('utf-8'))
        if match:
            # get the ub bits of triton kernel from bisheng for inductor autotune using
            metadata["required_ub_bits"] = int(match.group(1))
        if Path(callback_path).is_file():
            lib = ctypes.CDLL(callback_path)
            __get_metadata_attr_by_callback(lib, "_infer_workspace_shape_function", metadata, "workspace_size")
            __get_metadata_attr_by_callback(lib, "_infer_sync_block_lock_num_function", metadata, "lock_num")
            __get_metadata_attr_by_callback(lib, "_infer_sync_block_lock_init_function", metadata, "lock_init_val")

        return Path(bin_path).read_bytes()


@dataclass(frozen=True)
class NPUOptions:
    debug: bool = False
    sanitize_overflow: bool = True
    llvm_version: int = 15
    kernel_name: str = "triton_"

    cluster_dims: tuple = (1, 1, 1)
    num_warps: int = 4
    num_ctas: int = 1
    num_stages: int = 1
    warp_size: int = 32
    num_buffers_warp_spec: int = 0
    num_consumer_groups: int = 0
    reg_dec_producer: int = 0
    reg_inc_consumer: int = 0

    compile_on_910_95: bool = is_compile_on_910_95
    enable_linearize: bool = False
    enable_warp_specialization: bool = False
    enable_nd2nz_on_vector: bool = False
    enable_persistent: bool = False
    optimize_epilogue: bool = False
    enable_fp_fusion: bool = True
    allow_fp8e4nv: bool = False
    allowed_dot_input_precisions: Tuple[str] = ("ieee", "hf32")
    max_num_imprecise_acc_default: bool = None
    extern_libs: dict = None

    multibuffer: bool = not is_compile_on_910_95
    enable_auto_bind_sub_block: bool = not is_compile_on_910_95
    enable_select_analysis: bool = is_compile_on_910_95
    enable_hivm_auto_cv_balance: bool = None
    sync_solver: bool = None
    unit_flag: bool = None
    inject_barrier_all: bool = None
    inject_block_all: bool = None
    limit_auto_multi_buffer_only_for_local_buffer: bool = None
    limit_auto_multi_buffer_of_local_buffer: str = None
    set_workspace_multibuffer: int = None
    tile_mix_vector_loop: int = None
    tile_mix_cube_loop: int = None
    disable_auto_inject_block_sync: bool = None

    stream: int = None
    parallel_mode: str = "simd"
    force_simt_only: bool = False
    force_simt_template: bool = False
    # compile_mode: "simd" (default), "unstructured_in_simt", "simt_only"
    # When compile_mode is provided, it automatically sets other fields
    compile_mode: str = "simd"

    def __post_init__(self):
        # Parse compile_mode and set related fields
        if self.compile_mode == "simd":
            object.__setattr__(self, "parallel_mode", "simd")
        elif self.compile_mode == "unstructured_in_simt":
            # For historical compatibility reasons, force_simt_template will still be used.
            object.__setattr__(self, "force_simt_template", True)
        elif self.compile_mode == "simt_only":
            object.__setattr__(self, "force_simt_only", True)
            object.__setattr__(self, "parallel_mode", "simt")

    def hash(self):
        key = "_".join([f"{name}-{val}" for name, val in self.__dict__.items()])
        return hashlib.sha256(key.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class CPUOptions:
    debug: bool = False
    llvm_version: int = 15
    kernel_name: str = "triton_"

    cluster_dims: tuple = (1, 1, 1)
    num_warps: int = -1
    num_ctas: int = -1
    num_stages: int = -1

    enable_warp_specialization: bool = False
    enable_persistent: bool = False
    optimize_epilogue: bool = False
    enable_fp_fusion: bool = True
    allow_fp8e4nv: bool = False
    max_num_imprecise_acc_default: bool = None
    extern_libs: dict = None

    def hash(self):
        key = "_".join([f"{name}-{val}" for name, val in self.__dict__.items()])
        return hashlib.sha256(key.encode("utf-8")).hexdigest()


@register_descriptor
class AscendAttrsDescriptor(AttrsDescriptor):

    # For now we collect shapes of tensor at runtime.
    # We comment out the following func but keep it for future reference.
    def _add_backend_properties(self, params=None, values=None):
        pass


def ttir_to_npubin(mod, metadata, opt):
    # Get Triton-MLIR as string
    ttir_code = str(mod)
    metadata = _parse_ttir_metadata(ttir_code, metadata)
    with tempfile.TemporaryDirectory() as tmpdir:
        # prepare input
        src_path = os.path.join(tmpdir, "kernel.ttir.mlir")
        Path(src_path).write_text(ttir_code)
        # prepare output
        bin_file = os.path.join(tmpdir, "kernel")
        bin_path = os.path.join(tmpdir, "kernel.o")
        # build compile options
        _compile_option_list = get_common_bishengir_compile_options(metadata)
        if opt.force_simt_only:
            _compile_option_list += ["--enable-triton-ir-compile"]
            _compile_option_list += ["--pure-simt"]
            _compile_option_list += [f"--num-warps={opt.num_warps}"]
            _compile_option_list += [f"--threads-per-warp={opt.warp_size}"]

        npu_compiler_path = _get_npucompiler_path()
        cmd_list = (
            [npu_compiler_path, src_path]
            + _compile_option_list
            + ["-o", bin_file]
        )
        ret = subprocess.run(cmd_list, capture_output=True, check=True)
        return Path(bin_path).read_bytes()


class AscendBackend(BaseBackend):

    @staticmethod
    def supports_target(target: GPUTarget):
        return target.backend == "cpu" or target.backend == "npu"

    def __init__(self, target: GPUTarget) -> None:
        super().__init__(target)
        if target.backend == "cpu":
            self.binary_ext = "cpuasm"
        elif target.backend == "npu":
            self.binary_ext = "npubin"

    def parse_options(self, opts) -> Any:
        # TODO: get available targets when building options?
        if self.target.backend == "npu":
            args = {
                k: opts[k]
                for k in NPUOptions.__dataclass_fields__.keys()
                if k in opts
            }
            options = NPUOptions(**args)
        else:
            args = {
                k: opts[k]
                for k in CPUOptions.__dataclass_fields__.keys()
                if k in opts
            }
            options = CPUOptions(**args)
        return options

    def pack_metadata(self, metadata):
        # collect necessary metadata to launch kernels
        # TORCHINDUCTOR_UNIQUE_KERNEL_NAMES=1 could set unique name.
        # Get this name as the kernel_name to CANN runtime.
        # kernel_name is unique to Ascend backend and should not be public.
        # CANN runtime limits the length of kernel name <= 50.
        # Considering '\n' is appended, thus the real kernel name <= 49.
        KERNEL_NAME_MAX_LEN = 49
        kernel_name_orig, mix_mode = metadata.name.split()
        if len(kernel_name_orig) > KERNEL_NAME_MAX_LEN:
            kernel_name = kernel_name_orig[-KERNEL_NAME_MAX_LEN:]
        else:
            kernel_name = kernel_name_orig
        return {
            "kernel_name": kernel_name,
            "hash": metadata.hash,
            "debug": metadata.debug,
            "tensor_kinds": metadata.tensor_kinds,
        }

    def get_codegen_implementation(self):
        # Note: a dict of functions is required to generate vendor-specific code piecies
        #       e.g. convert custom types like fp8e4b15
        codegen_fns = {"min_dot_size": min_dot_size(self.target)}
        return codegen_fns

    def load_dialects(self, ctx):
        pass

    def get_attrs_descriptor(self, params, args):
        return AscendAttrsDescriptor(params, args)

    def add_stages(self, stages, options):
        if self.target.backend == "npu":
            stages["ttir"] = lambda src, metadata: make_ttir(src, metadata, options)
            if options.force_simt_only:
                stages["npubin"] = (
                    lambda src, metadata: ttir_to_npubin(
                        src, metadata, options
                    )
                )
                return
            stages["ttadapter"] = lambda src, metadata: ttir_to_linalg(
                src, metadata, options, named_ops=True
            )
            if options.compile_on_910_95:
                stages["npubin"] = (
                    lambda src, metadata: linalg_to_bin_enable_npu_compile_910_95(
                        src, metadata, options
                    )
                )
            else:
                stages["npubin"] = (
                    lambda src, metadata: linalg_to_bin_enable_npu_compile_A2_A3(
                        src, metadata, options
                    )
                )
        else:
            stages["ttir"] = lambda src, metadata: make_ttir(src, metadata, options)
            stages["ttadapter"] = lambda src, metadata: ttir_to_linalg(
                src, metadata, options
            )
            stages["llir"] = lambda src, metadata: linalg_to_llir(
                src, metadata, options
            )
            stages["cpuasm"] = lambda src, metadata: llir_to_cpuasm(
                src, metadata, options
            )

    @functools.lru_cache()
    def hash(self):
        # TODO fetch compiler version
        version_key = self.target
        return str(version_key)

    def get_module_map(self) -> Dict[str, ModuleType]:
        return {}
