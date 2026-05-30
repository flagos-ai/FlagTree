# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

import ctypes
import functools
import hashlib
import glob
import os
import re
import shlex
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
    _enable_dump_memory_info,
    _enable_msdebug,
    _get_kernel_target,
    _get_llvm_path,
    _get_mlir_path,
    _get_npucompiler_path,
    _get_triton_adapter_opt_path,
    _get_triton_mlir_opt_path,
    _get_triton_opt_path,
    _get_bishengir_opt_path,
    _is_ascend_sanitizer_enabled,
    _is_debug_line_info_disabled,
    _is_auto_map_parallel_blocks_enabled,
    downgrade_llir,
    force_disable_ffts,
    triton_enable_libdevice_simt,
    get_cann_version_file_hash,
)
from triton.backends.ascend.driver import (NPUUtils)
from triton.backends.compiler import (
    BaseBackend,
    GPUTarget,
)
from triton.runtime import driver
from triton.runtime.cache import get_dump_manager
from triton.tools.get_ascend_devices import is_compile_on_910_95


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
    ttir_code = str(mod)
    with tempfile.TemporaryDirectory() as tmpdir:
        src_path = os.path.join(tmpdir, "kernel.ttir.mlir")
        dst_path = os.path.join(tmpdir, "kernel.ttadapter.mlir")
        Path(src_path).write_text(ttir_code)
        triton_adapter_opt_path = _get_triton_adapter_opt_path()

        enable_nd2nz_on_vector = metadata["enable_nd2nz_on_vector"]
        enable_select_analysis = metadata["enable_select_analysis"]
        compile_on_910_95 = metadata["compile_on_910_95"]
        force_simt_template = metadata["force_simt_template"]
        enable_sync_block_lock = metadata["enable_sync_block_lock"]
        enable_mask_fallback_conversion = metadata["enable_mask_fallback_conversion"]
        optimize_dynamic_offset = metadata["optimize_dynamic_offset"]
        auto_blockify_size = metadata["auto_blockify_size"]
        if not _is_auto_map_parallel_blocks_enabled():
            auto_blockify_size = 1
        pm = ir.pass_manager(mod.context)
        pm.enable_debug()
        ascend.passes.ttir.add_auto_blockify(pm, auto_blockify_size)
        if (metadata["add_auto_scheduling"]):
            ascend.passes.ttir.add_dag_sync(pm)
            ascend.passes.ttir.add_dag_scope(pm)
            passes.common.add_cse(pm)
            passes.common.add_canonicalizer(pm)
            ascend.passes.ttir.add_dag_ssbuffer(pm)
            passes.common.add_cse(pm)
            passes.common.add_canonicalizer(pm)

        ascend.passes.ttir.add_triton_to_structure(pm, enable_mask_fallback_conversion, optimize_dynamic_offset)
        ascend.passes.ttir.add_discrete_mask_access_conversion(pm, compile_on_910_95, force_simt_template,
                                                               enable_sync_block_lock)
        ascend.passes.ttir.add_triton_to_annotation(pm)
        ascend.passes.ttir.add_triton_to_unstructure(pm, compile_on_910_95, force_simt_template)
        ascend.passes.ttir.add_triton_to_hivm(pm)
        ascend.passes.ttir.add_triton_to_hfusion(pm)
        ascend.passes.ttir.add_triton_to_llvm(pm)
        ascend.passes.ttir.add_bubble_up_operation(pm)
        ascend.passes.ttir.add_triton_to_structure(pm, enable_mask_fallback_conversion, optimize_dynamic_offset)
        ascend.passes.ttir.add_triton_to_linalg(pm, False, named_ops, enable_nd2nz_on_vector, enable_select_analysis,
                                                compile_on_910_95)
        if metadata["enable_dynamic_cv_pipeline"]:
            ascend.passes.ttir.add_dynamic_cv_pipeline(pm, compile_on_910_95)

        _val = metadata.get("intra_cache_num")
        if _val is not None:
            ascend.passes.ttir.set_buffer_count(0, _val)

        _val = metadata.get("inter_cache_num")
        if _val is not None:
            ascend.passes.ttir.set_buffer_count(1, _val)

        _val = metadata.get("load_cache_num")
        if _val is not None:
            ascend.passes.ttir.set_buffer_count(2, _val)

        if opt.debug:
            # Print the equivalent triton-opt command line so the pass
            # pipeline can be reproduced and debugged outside of Python.
            cmd = [
                _get_triton_opt_path(), src_path, f"--pass-pipeline={pm.get_pipeline_str()}", "--mlir-print-debuginfo",
                "-o", dst_path
            ]
            print(f"[DEBUG] cmd list: {shlex.join(cmd)}")

        pm.run(mod)

        if opt.debug:
            dump_manager = get_dump_manager(metadata["hash"])
            dump_manager.put(str(mod), "kernel.ttadapter.mlir", binary=False)

        return str(mod)


def linalg_to_bc_by_triton_mlir_opt(linalg: str, metadata, opt):
    """
    Convert Linalg IR to MLIR Bytecode format using triton-mlir-opt.
    This function supports both MLIR and BishengIR ops.
    Args:
        linalg: Linalg IR in text format
        metadata: Compilation metadata
        opt: Compilation options
    Returns:
        Bytecode data as bytes (not file path, to avoid temp directory cleanup issues)
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        ttadapter_path = os.path.join(tmpdir, "kernel.ttadapter.mlir")
        bc_path = os.path.join(tmpdir, "kernel.mlirbc")
        Path(ttadapter_path).write_text(linalg)

        triton_mlir_opt_path = _get_triton_mlir_opt_path()

        # The --emit-bytecode flag ensures output is in BC format
        subprocess.run([
            triton_mlir_opt_path,
            ttadapter_path,
            "--emit-bytecode",
            "-o",
            bc_path,
        ], check=True, capture_output=True, text=True)

        # Read bytecode as binary before temp directory is cleaned up
        with open(bc_path, "rb") as f:
            bc_data = f.read()

        if opt.debug:
            dump_manager = get_dump_manager(metadata["hash"])
            dump_manager.put(bc_data, "kernel.mlirbc", binary=True)

        return bc_data


def bc_to_linalg_by_bishengir_opt(bc_data: bytes, metadata, opt):
    """
    Convert MLIR Bytecode to MLIR text format using bishengir-opt.
    Args:
        bc_data: Bytecode data as bytes
        metadata: Compilation metadata
        opt: Compilation options
    Returns:
        MLIR text as string
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        bc_path = os.path.join(tmpdir, "kernel.mlirbc")
        mlir_path = os.path.join(tmpdir, "kernel.mlir")

        # Write bytecode data to temporary file
        with open(bc_path, "wb") as f:
            f.write(bc_data)

        bishengir_opt_path, env = _get_bishengir_opt_path()

        subprocess.run([
            bishengir_opt_path,
            bc_path,
            "--mlir-print-debuginfo",
            "-o",
            mlir_path,
        ], env=env, capture_output=True, check=True, text=True)

        # Read the generated MLIR text
        linalg_text = Path(mlir_path).read_text()

        if opt.debug:
            dump_manager = get_dump_manager(metadata["hash"])
            dump_manager.put(linalg_text, "kernel.mlir", binary=False)

        return linalg_text


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
      - name (kernel_name)

    Additionally, removes the mix_mode attribute from the IR.
    """
    # --- Regular expressions and examples ---

    DISABLE_AUTO_TILE_AND_BIND_SUBBLOCK_REGEX = r'hivm.disable_auto_tile_and_bind_subblock'

    # Example: mix_mode = "aiv" -> aiv
    MIX_MODE_REGEX = r'mix_mode\s*=\s*"([^"]+)"'

    # Example: parallel_mode = "mix_simd_simt" -> mix_simd_simt
    PARALLEL_MODE_REGEX = r'parallel_mode\s*=\s*"([^"]+)"'

    # Example: func.func @gather_sorted_kernel(%arg0: ...) -> gather_sorted_kernel
    KERNEL_NAME_REGEX = r"func\.func\s+@(\w+)"

    # Example: %arg1: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32} -> ('1', '0')
    TENSOR_KIND_REGEX = r'%arg(\d+):[^,)]*?\{[^}]*?tt\.tensor_kind\s*=\s*([^:\s}]+)\s*:[^}]*?\}'

    # Example: bitcode = "a.bc"
    BITCODES_REGEX = r'bitcode\s*=\s*(?:"([^"]+)"|\'([^\']+)\'|(\w+))'

    # Note: Compiled Kernel requires to estimate size of shared memory to occupy
    # Currently, NPU backend does not limit on shared memory
    metadata["shared"] = 1
    # Force disable auto tile and bind subblock if attribute is present in module
    metadata["auto_tile_and_bind_subblock"] = not re.search(DISABLE_AUTO_TILE_AND_BIND_SUBBLOCK_REGEX, linalg)
    metadata["mix_mode"] = re.search(MIX_MODE_REGEX, linalg).group(1)
    metadata["parallel_mode"] = re.search(PARALLEL_MODE_REGEX, linalg).group(1)
    metadata["kernel_name"] = re.search(KERNEL_NAME_REGEX, linalg).group(1)
    metadata["name"] = metadata["kernel_name"]
    # Parse all tensor kinds from arguments
    metadata["tensor_kinds"] = [int(kind) for _, kind in re.findall(TENSOR_KIND_REGEX, linalg)]
    # init the ub bits of triton kernel for inductor autotune using
    metadata["required_ub_bits"] = 0

    # Parse all bitcode paths
    bitcodes = re.findall(BITCODES_REGEX, linalg)
    metadata["bitcodes"] = [val for group in bitcodes for val in group if val]
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
    metadata["name"] = metadata["kernel_name"]
    # Parse all tensor kinds from arguments
    metadata["tensor_kinds"] = [int(kind) for _, kind in re.findall(TENSOR_KIND_REGEX, ttir)]
    return metadata


def get_common_bishengir_compile_options(metadata):
    bishengir_target = metadata['target'].arch
    bishengir_target_opt = f"--target={bishengir_target}"
    return [bishengir_target_opt]


def get_auto_bind_sub_block_option(metadata):
    # auto_tile_and_bind_subblock is read from the module.
    # enable_auto_bind_sub_block is set by the user and has a higher priority.
    enable_auto_bind_sub_block = metadata["enable_auto_bind_sub_block"]
    return (metadata["auto_tile_and_bind_subblock"]
            if enable_auto_bind_sub_block is None else enable_auto_bind_sub_block)


def _save_npuir_debug_output(stdout_bytes: bytes, stderr_bytes: bytes, tmpdir: str, metadata_hash: str):
    stdout = stdout_bytes.decode('utf-8') if stdout_bytes else ''
    stderr = stderr_bytes.decode('utf-8') if stderr_bytes else ''
    combined = stdout + stderr
    if not combined.strip():
        combined = "No output captured."
    output_path = os.path.join(tmpdir, "kernel.npuir.mlir")
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(combined)

    dump_manager = get_dump_manager(metadata_hash)
    dump_manager.put(Path(output_path).read_text(encoding='utf-8'), "kernel.npuir.mlir", binary=False)


def linalg_to_bin_enable_npu_compile_910_95(linalg: str, metadata, opt):
    linalg, metadata = _parse_linalg_metadata(linalg, metadata)
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_file_name = "kernel.mlir" if opt.use_bytecode else "kernel.ttadapter.mlir"
        ttadapter_path = os.path.join(tmpdir, tmp_file_name)
        Path(ttadapter_path).write_text(linalg)
        bin_file = os.path.join(tmpdir, "kernel")
        if _check_bishengir_api_change():
            bin_file_with_ext = "kernel.o"
        else:
            bin_file_with_ext = "kernel_reloc.o"
        bin_path = os.path.join(tmpdir, bin_file_with_ext)
        callback_path = os.path.join(tmpdir, "libkernel.so")
        _compile_option_list = get_common_bishengir_compile_options(metadata)

        multibuffer = metadata.get("multibuffer")
        num_stages = metadata.get("num_stages")

        if multibuffer is not None or num_stages is not None:
            multi_buffer_value = True
            if multibuffer is not None and not multibuffer:
                multi_buffer_value = False
            elif num_stages is not None and num_stages == 1:
                multi_buffer_value = False
            _compile_option_list += [
                f"--enable-auto-multi-buffer={multi_buffer_value}",
            ]

        disable_tightly_coupled_buffer_reuse = metadata["disable_tightly_coupled_buffer_reuse"]
        if disable_tightly_coupled_buffer_reuse:
            _compile_option_list += ["--disable-tightly-coupled-buffer-reuse"]

        _compile_option_list += [
            f"--enable-auto-bind-sub-block={get_auto_bind_sub_block_option(metadata)}",
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

        auto_multi_buffer = metadata["limit_auto_multi_buffer_of_local_buffer"]
        if auto_multi_buffer is not None:
            _compile_option_list += \
                [f"--limit-auto-multi-buffer-of-local-buffer={auto_multi_buffer}"]

        enable_mixed_cv = metadata["enable_mixed_cv"]
        if enable_mixed_cv is not None:
            _compile_option_list += \
                [f"--enable-mixed-cv={enable_mixed_cv}"]

        enable_cce_vf_auto_sync = metadata["enable_cce_vf_auto_sync"]
        if enable_cce_vf_auto_sync is not None:
            _compile_option_list += \
                [f"--append-bisheng-options=-mllvm --cce-vf-auto-sync={enable_cce_vf_auto_sync}"]

        enable_cce_vf_remove_membar = metadata["enable_cce_vf_remove_membar"]
        if enable_cce_vf_remove_membar is not None:
            _compile_option_list += \
                [f"--append-bisheng-options=-mllvm --cce-vf-remove-membar={enable_cce_vf_remove_membar}"]

        # TEMP: Allow TRITON_ENABLE_VF_FUSION env var to override metadata configuration
        env_vf = os.getenv("TRITON_ENABLE_VF_FUSION")
        enable_vf_fusion = (env_vf.lower() in ("true", "1", "yes") if env_vf is not None else metadata.get(
            "enable_vf_fusion", False))
        if enable_vf_fusion:
            _compile_option_list += ["--enable-vf-fusion"]

        enable_drop_unit_dims = metadata["enable_drop_unit_dims"]
        if enable_drop_unit_dims is not None:
            _compile_option_list += \
                [f"--enable-drop-unit-dims={enable_drop_unit_dims}"]

        enable_flatten = metadata["enable_flatten"]
        if enable_flatten is not None:
            _compile_option_list += \
                [f"--enable-flatten={enable_flatten}"]

        enable_auto_vectorize_v2 = metadata["enable_auto_vectorize_v2"]
        if enable_auto_vectorize_v2 is not None:
            _compile_option_list += \
                [f"--enable-auto-vectorize-v2={enable_auto_vectorize_v2}"]
        auto_vectorize_v2_max_fused_ops_num = metadata["auto_vectorize_v2_max_fused_ops_num"]
        if auto_vectorize_v2_max_fused_ops_num is not None:
            _compile_option_list += \
                [f"--hfusion-max-fused-ops-in-auto-vectorize-v2={auto_vectorize_v2_max_fused_ops_num}"]
        prevec_max_fused_ops_num = metadata["prevec_max_fused_ops_num"]
        if prevec_max_fused_ops_num is not None:
            _compile_option_list += \
                [f"--hfusion-max-fused-elementwise-ops={prevec_max_fused_ops_num}"]

        disable_auto_inject_block_sync = metadata["disable_auto_inject_block_sync"]
        if disable_auto_inject_block_sync is not None:
            _compile_option_list += \
                [f"--disable-auto-inject-block-sync={disable_auto_inject_block_sync}"]

        bitcodes = metadata["bitcodes"]
        if bitcodes is not None:
            for bitcode in bitcodes:
                _compile_option_list += \
                    [f"--link-aicore-bitcode={bitcode}"]

        enable_auto_blockify = metadata["enable_auto_blockify"]
        if _is_auto_map_parallel_blocks_enabled():
            if (enable_auto_blockify is None or enable_auto_blockify):
                _compile_option_list += ["--enable-auto-blockify-loop"]
        else:
            if enable_auto_blockify:
                _compile_option_list += ["--enable-auto-blockify-loop"]

        npu_compiler_path, env = _get_npucompiler_path()
        if npu_compiler_path.endswith("bishengir-compile"):
            _compile_option_list += [
                "--enable-hfusion-compile=true",
                "--enable-triton-kernel-compile=true",
            ]
        bisheng_options = metadata["bisheng_options"]
        if bisheng_options is not None:
            _compile_option_list += [f"--append-bisheng-options={bisheng_options}"]
        mix_mode = opt.mix_mode
        if mix_mode in ["aic"]:
            _compile_option_list += ["--disable-hfusion-vectorize=true"]

        if opt.debug:
            _compile_option_list += ["--bishengir-print-ir-after=hivm-graph-sync-solver"]

        cmd_list = ([npu_compiler_path, ttadapter_path] + _compile_option_list + ["-o", bin_file])
        vf_merge_level = metadata["vf_merge_level"]
        if vf_merge_level is not None and vf_merge_level != 1:
            cmd_list += [f"--enable-vf-merge-level={vf_merge_level}"]

        hfusion_enable_multiple_consumer_fusion = metadata["hfusion_enable_multiple_consumer_fusion"]
        if hfusion_enable_multiple_consumer_fusion:
            cmd_list += [f"--hfusion-enable-multiple-consumer-fusion={hfusion_enable_multiple_consumer_fusion}"]

        if opt.debug:
            print(f"[DEBUG] cmd_list: {' '.join(cmd_list)}")

        try:
            ret = subprocess.run(cmd_list, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        except subprocess.CalledProcessError as e:
            if opt.debug:
                _save_npuir_debug_output(e.stdout, e.stderr, tmpdir, metadata["hash"])
            raise

        if opt.debug:
            _save_npuir_debug_output(ret.stdout, ret.stderr, tmpdir, metadata["hash"])

        stdout_str = ret.stdout.decode('utf-8') if ret.stdout else ''
        match = re.search(r'UB\s+size\s*=\s*(\d+)\s*bits', stdout_str)
        if match:
            # get the ub bits of triton kernel from bisheng for inductor autotune using
            metadata["required_ub_bits"] = int(match.group(1))

        if not Path(bin_path).exists():
            error_msg = ret.stderr.decode('utf-8') if ret.stderr else ''
            print(f"[DEBUG] {bin_path} is not found")
            print(f"[DEBUG] Stderr:\n{error_msg}")
            raise subprocess.CalledProcessError(ret.returncode, cmd_list, ret.stdout, ret.stderr)

        if Path(callback_path).is_file():
            lib = ctypes.CDLL(callback_path)
            __get_metadata_attr_by_callback(lib, "_infer_task_type_function", metadata, "bs_task_type")
            __get_metadata_attr_by_callback(lib, "_infer_workspace_shape_function", metadata, "workspace_size")
            __get_metadata_attr_by_callback(lib, "_infer_sync_block_lock_num_function", metadata, "lock_num")
            __get_metadata_attr_by_callback(lib, "_infer_sync_block_lock_init_function", metadata, "lock_init_val")

        return Path(bin_path).read_bytes()


def linalg_to_bin_enable_npu_compile_A2_A3(linalg: str, metadata, opt):
    linalg, metadata = _parse_linalg_metadata(linalg, metadata)
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_file_name = "kernel.mlir" if opt.use_bytecode else "kernel.ttadapter.mlir"
        ttadapter_path = os.path.join(tmpdir, tmp_file_name)
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

        multibuffer = metadata.get("multibuffer")
        num_stages = metadata.get("num_stages")

        if multibuffer is not None or num_stages is not None:
            multi_buffer_value = True
            if multibuffer is not None and not multibuffer:
                multi_buffer_value = False
            elif num_stages is not None and num_stages == 1:
                multi_buffer_value = False
            _compile_option_list.append(f"--enable-auto-multi-buffer={multi_buffer_value}")

        enable_ubuf_saving = metadata["enable_ubuf_saving"]
        if enable_ubuf_saving is not None:
            _compile_option_list += [
                f"--enable-ubuf-saving={enable_ubuf_saving}",
            ]

        enable_preload = metadata["enable_preload"]
        if enable_preload is not None:
            _compile_option_list += [
                f"--enable-preload={enable_preload}",
            ]

        _compile_option_list += [
            f"--enable-auto-bind-sub-block={get_auto_bind_sub_block_option(metadata)}",
        ]

        if _is_ascend_sanitizer_enabled():
            _compile_option_list += ["--enable-sanitizer=true"]
        if not _is_debug_line_info_disabled():
            _compile_option_list += ["--enable-debug-info=true"]

        if _enable_print_ub_bits():
            _compile_option_list += ["--enable-print-memory-allocated-size"]

        if _enable_dump_memory_info():
            _compile_option_list += ["--enable-memory-display=true"]

        if _enable_msdebug():
            _compile_option_list += ["--enable-ms-debug=true"]

        enable_hivm_auto_cv_balance = metadata["enable_hivm_auto_cv_balance"]
        if enable_hivm_auto_cv_balance is not None:
            _compile_option_list += \
                [f"--enable-hivm-auto-cv-balance={enable_hivm_auto_cv_balance}"]

        sync_solver = metadata["sync_solver"]
        if sync_solver is not None:
            _compile_option_list += [
                f"--enable-hivm-graph-sync-solver={sync_solver}",
                f"--enable-hivm-cross-core-gss={sync_solver}",
            ]

        unit_flag = metadata["unit_flag"]
        if unit_flag is not None:
            _compile_option_list += \
                [f"--enable-hivm-unit-flag-sync={unit_flag}"]

        enable_drop_unit_dims = metadata["enable_drop_unit_dims"]
        if enable_drop_unit_dims is not None:
            _compile_option_list += \
                [f"--enable-drop-unit-dims={enable_drop_unit_dims}"]

        enable_flatten = metadata["enable_flatten"]
        if enable_flatten is not None:
            _compile_option_list += \
                [f"--enable-flatten={enable_flatten}"]

        enable_auto_vectorize_v2 = metadata["enable_auto_vectorize_v2"]
        if enable_auto_vectorize_v2 is not None:
            _compile_option_list += \
                [f"--enable-auto-vectorize-v2={enable_auto_vectorize_v2}"]

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

        bitcodes = metadata["bitcodes"]
        if bitcodes is not None:
            for bitcode in bitcodes:
                _compile_option_list += \
                    [f"--link-aicore-bitcode={bitcode}"]

        _compile_option_list += [f"--link-aicore-bitcode={get_libdevice()}"]

        disable_size_align_for_cast = metadata["disable_size_align_for_cast"]
        if disable_size_align_for_cast is not None:
            _compile_option_list += \
                [f"--disable-size-align-for-cast={disable_size_align_for_cast}"]

        if _is_auto_map_parallel_blocks_enabled():
            _compile_option_list += ["--enable-auto-blockify-loop"]
        npu_compiler_path, env = _get_npucompiler_path()
        if npu_compiler_path.endswith("bishengir-compile"):
            _compile_option_list += [
                "--enable-hfusion-compile=true",
                bishengir_hivm_opt,
                "--enable-triton-kernel-compile=true",
            ]

        if opt.debug:
            _compile_option_list += ["--mlir-print-ir-after-failure"]
            _compile_option_list += ["--bishengir-print-ir-after=hivm-graph-sync-solver"]
        cmd_list = ([npu_compiler_path, ttadapter_path] + _compile_option_list + ["-o", bin_file])
        if opt.debug:
            print(f"[DEBUG] cmd_list: {' '.join(cmd_list)}")

        try:
            ret = subprocess.run(cmd_list, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        except subprocess.CalledProcessError as e:
            if opt.debug:
                _save_npuir_debug_output(e.stdout, e.stderr, tmpdir, metadata["hash"])
            raise

        if opt.debug:
            _save_npuir_debug_output(ret.stdout, ret.stderr, tmpdir, metadata["hash"])

        stdout_str = ret.stdout.decode('utf-8') if ret.stdout else ''
        match = re.search(r'UB\s+size\s*=\s*(\d+)\s*bits', stdout_str)
        if match:
            metadata["required_ub_bits"] = int(match.group(1))

        if not Path(bin_path).exists():
            error_msg = ret.stderr.decode('utf-8') if ret.stderr else ''
            print(f"[DEBUG] {bin_path} is not found")
            print(f"[DEBUG] Stderr:\n{error_msg}")
            raise subprocess.CalledProcessError(ret.returncode, cmd_list, ret.stdout, ret.stderr)

        if Path(callback_path).is_file():
            lib = ctypes.CDLL(callback_path)
            __get_metadata_attr_by_callback(lib, "_infer_task_type_function", metadata, "bs_task_type")
            __get_metadata_attr_by_callback(lib, "_infer_workspace_shape_function", metadata, "workspace_size")
            __get_metadata_attr_by_callback(lib, "_infer_sync_block_lock_num_function", metadata, "lock_num")
            __get_metadata_attr_by_callback(lib, "_infer_sync_block_lock_init_function", metadata, "lock_init_val")

        return Path(bin_path).read_bytes()


def get_libdevice():
    current = os.path.dirname(__file__)
    return os.path.join(current, "lib/libdevice.10.bc")


@dataclass(frozen=True)
class NPUOptions:
    debug: bool = False
    sanitize_overflow: bool = True
    llvm_version: int = 15
    kernel_name: str = "triton_"
    arch: str = ""

    cluster_dims: tuple = (1, 1, 1)
    num_warps: int = 32
    num_ctas: int = 1
    num_stages: int = 1 if is_compile_on_910_95 else 2
    warp_size: int = 32
    num_buffers_warp_spec: int = 0
    num_consumer_groups: int = 0
    reg_dec_producer: int = 0
    reg_inc_consumer: int = 0

    auto_blockify_size: int = 1
    enable_auto_blockify: bool = None
    compile_on_910_95: bool = is_compile_on_910_95
    optimize_dynamic_offset: bool = False
    enable_mask_fallback_conversion: bool = False
    enable_warp_specialization: bool = False
    enable_nd2nz_on_vector: bool = False
    enable_persistent: bool = False
    optimize_epilogue: bool = False
    enable_fp_fusion: bool = True
    allow_fp8e4nv: bool = False
    auto_tile_and_bind_subblock: bool = True
    supported_fp8_dtypes: Tuple[str] = ("fp8e5", "fp8e4b15", "fp8e4nv", "fp8e4b8", "fp8e5b16")
    deprecated_fp8_dtypes: Tuple[str] = ()
    vf_merge_level: int = 1
    default_dot_input_precision: str = "ieee"
    allowed_dot_input_precisions: Tuple[str] = ("ieee", "hf32")
    max_num_imprecise_acc_default: int = 0
    extern_libs: dict = None
    bisheng_options: str = "-cce-link-aicore-ll-module " + get_libdevice()

    multibuffer: bool = not is_compile_on_910_95
    enable_ubuf_saving: bool = None
    enable_preload: bool = None
    enable_auto_bind_sub_block: bool = None
    disable_tightly_coupled_buffer_reuse: bool = False
    enable_select_analysis: bool = True
    enable_hivm_auto_cv_balance: bool = None
    sync_solver: bool = None
    unit_flag: bool = None
    enable_cce_vf_auto_sync: bool = None
    enable_cce_vf_remove_membar: bool = None
    enable_drop_unit_dims: bool = None
    enable_flatten: bool = None
    enable_auto_vectorize_v2: bool = None
    auto_vectorize_v2_max_fused_ops_num: int = None
    prevec_max_fused_ops_num: int = None
    inject_barrier_all: bool = None
    inject_block_all: bool = None
    disable_size_align_for_cast: bool = None
    limit_auto_multi_buffer_only_for_local_buffer: bool = None
    limit_auto_multi_buffer_of_local_buffer: str = None
    set_workspace_multibuffer: int = None
    tile_mix_vector_loop: int = None
    tile_mix_cube_loop: int = None
    disable_auto_inject_block_sync: bool = None
    enable_mixed_cv: bool = None
    enable_vf_fusion: bool = False
    # todo: this code will be removed in version 530.
    add_auto_scheduling: bool = False
    enable_dynamic_cv_pipeline: bool = False
    hfusion_enable_multiple_consumer_fusion: bool = False

    stream: int = None
    parallel_mode: str = "simd"
    force_simt_only: bool = False
    force_simt_template: bool = False
    enable_sync_block_lock: bool = False
    # only take effect on the simt-only & simd-simt-mix scenarios
    shared_mem_dynamic_size: int = None
    # enable_bishengir_simt_optimization is passed as
    # -enable-bishengir-simt-optimization flag to bishengir-compile.
    enable_bishengir_simt_optimization: int = 000
    # compile_mode: "simd" (default), "unstructured_in_simt", "simt_only"
    # When compile_mode is provided, it automatically sets other fields
    compile_mode: str = "simd"
    mix_mode: str = ""
    simt_stack_limit: int = None
    # use_bytecode:
    # If True, the compilation flow is:
    #   Linalg IR → MLIR Bytecode (via triton-mlir-opt)
    #            → LLIR (via bishengir-opt)
    #            → Binary (via bishengir-compile)
    #
    # If False, the compilation flow is:
    #   Linalg IR → LLIR → Binary (via bishengir-compile directly)
    use_bytecode: bool = True
    # take effect on the reorder instruction pattern for SIMT. The pattern is disabled by default.
    enable_simt_reorder_instruction: bool = False
    # disable simt fma optimization to get high precision
    disable_fma: bool = False

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

        if self.force_simt_only:
            if self.shared_mem_dynamic_size is None:
                object.__setattr__(self, "shared_mem_dynamic_size", 122880)
        else:
            object.__setattr__(self, "shared_mem_dynamic_size", 221184)

    def hash(self):
        key = "_".join([f"{name}-{val}" for name, val in self.__dict__.items()])
        key = "_".join([key, get_cann_version_file_hash()])
        return hashlib.sha256(key.encode("utf-8")).hexdigest()


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
            _compile_option_list += ["--enable-hivm-compile=false"]
            _compile_option_list += ["--enable-triton-ir-compile"]
            _compile_option_list += ["--pure-simt"]
            _compile_option_list += [f"--num-warps={opt.num_warps}"]
            _compile_option_list += [f"--threads-per-warp={opt.warp_size}"]
            if opt.enable_bishengir_simt_optimization != 000:
                _compile_option_list += [
                    f"--enable-bishengir-simt-optimization={opt.enable_bishengir_simt_optimization}"
                ]
            if opt.simt_stack_limit:
                _compile_option_list += [f"--simt-stack-limit={opt.simt_stack_limit}"]
            if opt.shared_mem_dynamic_size is not None:
                _compile_option_list += [f"--shared-mem-dynamic-size={opt.shared_mem_dynamic_size}"]
            if opt.enable_simt_reorder_instruction:
                _compile_option_list += ["--enable-simt-reorder-instruction=true"]
            if opt.disable_fma:
                _compile_option_list += [f"--disable-fma"]
            enable_libdevice_simt = triton_enable_libdevice_simt()
            if (enable_libdevice_simt):
                bisheng_options = metadata["bisheng_options"]
                if bisheng_options is not None:
                    _compile_option_list += [f"--append-bisheng-options={bisheng_options}"]

            # Enable SIMT auto-blockify if user opted in, or if the env var is
            # set and the user didn't explicitly opt out (matches the SIMD path
            # at line ~541).
            enable_auto_blockify = opt.enable_auto_blockify
            if _is_auto_map_parallel_blocks_enabled():
                if enable_auto_blockify is None or enable_auto_blockify:
                    _compile_option_list += ["--enable-auto-blockify-loop"]
            else:
                if enable_auto_blockify:
                    _compile_option_list += ["--enable-auto-blockify-loop"]

        npu_compiler_path, env = _get_npucompiler_path()
        cmd_list = ([npu_compiler_path, src_path] + _compile_option_list + ["-o", bin_file])
        ret = subprocess.run(cmd_list, env=env, capture_output=True, check=True)
        if not Path(bin_path).exists():
            error_msg = ret.stderr.decode('utf-8')
            print(f"[DEBUG] {bin_path} is not found")
            print(f"[DEBUG] Stderr:\n{error_msg}")
            raise subprocess.CalledProcessError(ret.returncode, cmd_list, ret.stdout, ret.stderr)
        return Path(bin_path).read_bytes()


class AscendBackend(BaseBackend):

    @staticmethod
    def supports_target(target: GPUTarget):
        return target.backend == "npu"

    def __init__(self, target: GPUTarget) -> None:
        super().__init__(target)
        if target.backend == "npu":
            self.binary_ext = "npubin"
            # Include all binary file extensions (mlirbc is used in bytecode mode)
            self.binary_extensions = {"npubin", "mlirbc"}

    def parse_options(self, opts) -> Any:
        # TODO: get available targets when building options?
        if self.target.backend == "npu":
            args = {k: opts[k] for k in NPUOptions.__dataclass_fields__.keys() if k in opts}
            args.setdefault("arch", self.target.arch)
            options = NPUOptions(**args)
        else:
            raise NotImplementedError(f"Backend '{self.target.backend}' is not supported. "
                                      "Please ensure the target backend is set to 'npu'.")
        return options

    def pack_metadata(self, metadata):
        # collect necessary metadata to launch kernels
        # TORCHINDUCTOR_UNIQUE_KERNEL_NAMES=1 could set unique name.
        # Get this name as the kernel_name to CANN runtime.
        # kernel_name is unique to Ascend backend and should not be public.
        # CANN runtime limits the length of kernel name <= 50.
        # Considering '\n' is appended, thus the real kernel name <= 49.
        KERNEL_NAME_MAX_LEN = 49
        kernel_name_orig = metadata.kernel_name
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

    def get_codegen_implementation(self, options):
        # Note: a dict of functions is required to generate vendor-specific code piecies
        #       e.g. convert custom types like fp8e4b15
        from triton.backends.ascend import _apply_ascend_patch
        _apply_ascend_patch()
        codegen_fns = {"min_dot_size": min_dot_size(self.target)}
        return codegen_fns

    def load_dialects(self, ctx):
        ascend.load_dialects(ctx)

    def add_stages(self, stages, options, language):
        if self.target.backend == "npu":
            stages["ttir"] = lambda src, metadata: make_ttir(src, metadata, options)
            if options.force_simt_only:
                stages["npubin"] = (lambda src, metadata: ttir_to_npubin(src, metadata, options))
                return
            stages["ttadapter"] = lambda src, metadata: ttir_to_linalg(src, metadata, options, named_ops=True)
            # Support BC mode: convert Linalg IR to Bytecode format, then back to MLIR
            if options.use_bytecode:
                # Step 1: Convert Linalg IR to Bytecode using triton-mlir-opt
                stages["mlirbc"] = lambda src, metadata: linalg_to_bc_by_triton_mlir_opt(src, metadata, options)
                # Step 2: Convert Bytecode back to MLIR text using bishengir-opt
                stages["bcmlir"] = lambda src, metadata: bc_to_linalg_by_bishengir_opt(src, metadata, options)
            if options.compile_on_910_95:
                stages["npubin"] = (
                    lambda src, metadata: linalg_to_bin_enable_npu_compile_910_95(src, metadata, options))
            else:
                stages["npubin"] = (
                    lambda src, metadata: linalg_to_bin_enable_npu_compile_A2_A3(src, metadata, options))
        else:
            raise NotImplementedError(f"Backend '{self.target.backend}' is not supported. "
                                      "Please ensure the target backend is set to 'npu'.")

    @functools.lru_cache()
    def hash(self):
        # TODO fetch compiler version
        version_key = self.target
        return str(version_key)

    def get_module_map(self) -> Dict[str, ModuleType]:
        return {}
