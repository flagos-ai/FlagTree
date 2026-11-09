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
import json
import os
import re
import shlex
import subprocess
import tempfile
import warnings
from dataclasses import InitVar, dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, Optional, Tuple, Union
from .debug_line_rewriter import rewrite_debug_line

from triton._C.libtriton import ir, passes, ascend, buffer_ir
from triton._C.libtriton.ascend import ir as ascend_ir
try:
    from triton._C.libtriton import distributed
except ImportError:
    distributed = None
from triton.backends.ascend.utils import (
    _multibuffer_mode_to_tuple,
    _check_bishengir_api_change,
    _check_bishengir_able_save_ir,
    _check_bishengir_is_regbased,
    _enable_print_ub_bits,
    _enable_dump_memory_info,
    _enable_msdebug,
    _get_kernel_target,
    _get_npucompiler_path,
    _get_triton_adapter_opt_path,
    _get_triton_mlir_opt_path,
    _get_triton_opt_path,
    _get_bishengir_opt_path,
    _is_ascend_sanitizer_enabled,
    _is_debug_line_info_disabled,
    _is_auto_map_parallel_blocks_enabled,
    _get_auto_blockify_blacklist_reasons,
    _warn_auto_blockify_disabled,
    _remove_deprecated_npu_options,
    _warn_deprecated_npu_option,
    _warn_deprecated_ascend_env_vars,
    downgrade_llir,
    force_disable_ffts,
    graph_ub_budget_bytes_for_arch,
    ub_size_in_kbytes_for_arch,
    get_cann_version_file_hash,
    is_cann_version_at_least,
)
from triton.backends.ascend.driver import (NPUUtils)
from triton.backends.ascend.program_grid import (
    DEFAULT_GRAPH_OPTIMIZATION_RULE_MASK,
    PROGRAM_GRID_TRANSFORMS_ATTR,
    ProgramGridContractError,
    get_persistent_transform,
    normalize_graph_optimization_rule_mask,
    normalize_program_grid_transforms,
)
from triton.backends.compiler import (
    BaseBackend,
    GPUTarget,
)
from triton.runtime.cache import get_dump_manager


# TODO: materialize the concrete min shape
def min_dot_size(target: GPUTarget):
    return lambda lhsType, rhsType: (1, 1, 1)


# Get result code saved in module {attr_name = rc}
def _get_then_remove_rc(mod, attr_name: str) -> int:
    get_int_attr = getattr(ascend.ir, "get_int_attr", None)
    remove_attr = getattr(ascend.ir, "remove_attr", None)

    if get_int_attr is None:
        return -1
    try:
        attr_value = get_int_attr(mod, attr_name)
    except TypeError:
        return -1

    if remove_attr:
        remove_attr(mod, attr_name)

    if not isinstance(attr_value, int):
        return -1

    return attr_value


def _get_then_remove_program_grid_transforms(mod):
    get_transforms = getattr(ascend.ir, "get_program_grid_transforms", None)
    remove_attr = getattr(ascend.ir, "remove_attr", None)
    if get_transforms is None:
        if PROGRAM_GRID_TRANSFORMS_ATTR in str(mod):
            raise RuntimeError("hacc.program_grid_transforms requires the matching Ascend C++ binding")
        return None
    try:
        raw = get_transforms(mod)
    except TypeError:
        if PROGRAM_GRID_TRANSFORMS_ATTR in str(mod):
            raise RuntimeError("hacc.program_grid_transforms requires an MLIR module accepted by "
                               "the Ascend C++ binding")
        return None
    if raw is not None and remove_attr:
        remove_attr(mod, PROGRAM_GRID_TRANSFORMS_ATTR)
    return raw


def _export_coalesce_metadata(mod, metadata, *, require_row_contract=False):
    # Tile/strided coalescing (TritonToLinalg) records the chosen coalesce factor
    # H and the grid axis it applies to as module attrs hacc.coalesce_factor /
    # hacc.coalesce_axis. In the full-TA design the *launcher* (driver.py) owns
    # the grid division: it divides grid[axis] by H before launch, and bishengir
    # no longer interprets the attrs. Read them into metadata here and strip them
    # from the module so the hacc.* attrs never reach hivmc (which rejects
    # unknown module attrs). Absent attrs -> factor 1 (no-op) / axis -1.
    # RowCoalescing also records whether the launcher should use ceil-div when
    # shrinking the grid, because its runtime valid-count guard handles tails.
    # A Row result is only valid as the complete triple.  The regular T2L
    # Axis/Chunk ABI predates ceil-div and deliberately remains compatible with
    # a missing ceil-div attr.
    factor = _get_then_remove_rc(mod, "hacc.coalesce_factor")
    axis = _get_then_remove_rc(mod, "hacc.coalesce_axis")
    ceil_div = _get_then_remove_rc(mod, "hacc.coalesce_grid_ceil_div")
    has_any_contract_attr = any(value != -1 for value in (factor, axis, ceil_div))
    valid_factor = isinstance(factor, int) and factor > 1
    valid_axis = isinstance(axis, int) and axis in (0, 1, 2)
    valid_ceil_div = isinstance(ceil_div, int) and ceil_div > 0
    if has_any_contract_attr and (not valid_factor or not valid_axis):
        raise RuntimeError("invalid hacc.coalesce launch contract")
    if require_row_contract and has_any_contract_attr and not valid_ceil_div:
        raise RuntimeError("RowCoalescing requires hacc.coalesce_grid_ceil_div")

    metadata["coalesce_factor"] = factor if valid_factor else 1
    metadata["coalesce_axis"] = axis if valid_axis else -1
    metadata["coalesce_grid_ceil_div"] = valid_ceil_div
    metadata["row_coalescing_applied"] = metadata["coalesce_factor"] > 1


def _export_program_grid_metadata(mod, metadata, *, require_row_contract=False):
    raw_transforms = _get_then_remove_program_grid_transforms(mod)
    if raw_transforms is not None:
        factor = _get_then_remove_rc(mod, "hacc.coalesce_factor")
        axis = _get_then_remove_rc(mod, "hacc.coalesce_axis")
        ceil_div = _get_then_remove_rc(mod, "hacc.coalesce_grid_ceil_div")
        has_legacy_attrs = any(value != -1 for value in (factor, axis, ceil_div))
        if has_legacy_attrs:
            raise RuntimeError("hacc.program_grid_transforms conflicts with legacy hacc.coalesce_* metadata")
        try:
            transforms = normalize_program_grid_transforms(raw_transforms)
        except ProgramGridContractError as error:
            raise RuntimeError(f"invalid hacc.program_grid_transforms: {error}") from error
        metadata["program_grid_transforms"] = transforms
        metadata["program_grid_mapping_applied"] = True
        metadata["coalesce_factor"] = 1
        metadata["coalesce_axis"] = -1
        metadata["coalesce_grid_ceil_div"] = False
        metadata["row_coalescing_applied"] = False
        return

    metadata["program_grid_transforms"] = None
    metadata["program_grid_mapping_applied"] = False
    _export_coalesce_metadata(
        mod,
        metadata,
        require_row_contract=require_row_contract,
    )


def _finalize_program_launch_policy(metadata, opt):
    required_fields = (
        "program_grid_transforms",
        "program_grid_mapping_applied",
        "row_coalescing_applied",
        "has_auto_blockify_blacklist_op",
        "mix_mode",
    )
    missing = [name for name in required_fields if name not in metadata]
    if missing:
        raise RuntimeError("cannot finalize program launch policy; missing metadata: " + ", ".join(missing))

    raw_transforms = metadata["program_grid_transforms"]
    mapping_applied = metadata["program_grid_mapping_applied"]
    row_coalescing_applied = metadata["row_coalescing_applied"]
    has_auto_blockify_blacklist_op = metadata["has_auto_blockify_blacklist_op"]
    if not isinstance(mapping_applied, bool):
        raise RuntimeError("program_grid_mapping_applied must be a boolean")
    if not isinstance(row_coalescing_applied, bool):
        raise RuntimeError("row_coalescing_applied must be a boolean")
    if not isinstance(has_auto_blockify_blacklist_op, bool):
        raise RuntimeError("has_auto_blockify_blacklist_op must be a boolean")

    transforms = None
    if raw_transforms is not None:
        try:
            transforms = normalize_program_grid_transforms(raw_transforms)
        except ProgramGridContractError as error:
            raise RuntimeError(f"invalid exported program_grid_transforms: {error}") from error
    if mapping_applied != (transforms is not None):
        raise RuntimeError("program_grid_mapping_applied disagrees with program_grid_transforms")
    if mapping_applied and row_coalescing_applied:
        raise RuntimeError("program-grid mapping conflicts with legacy RowCoalescing")

    persistent_transform = get_persistent_transform(transforms) if transforms is not None else None
    ptsm_cap_authorized = False
    if persistent_transform is not None:
        if metadata["mix_mode"] != "aiv":
            raise RuntimeError("persistent program-grid transform requires final mix_mode=aiv")
        if not (persistent_transform["persistent_coverage"] and persistent_transform["grid_stride_abi_verified"]):
            raise RuntimeError("persistent program-grid transform lacks coverage/ABI verification")
        ptsm_cap_authorized = True

    if opt.is_pure_simt:
        auto_blockify_enabled = _is_auto_map_parallel_blocks_enabled() and not row_coalescing_applied
    else:
        auto_blockify_enabled = (_is_auto_map_parallel_blocks_enabled() and not has_auto_blockify_blacklist_op
                                 and not ptsm_cap_authorized)

    if auto_blockify_enabled and ptsm_cap_authorized:
        raise RuntimeError("AutoBlockify and persistent-grid cap cannot both be enabled")

    metadata["auto_blockify_enabled"] = auto_blockify_enabled
    metadata["ptsm_cap_authorized"] = ptsm_cap_authorized


def _adjust_metadata_by_module_result(mod, metadata, opt, **kwargs):
    rc = _get_then_remove_rc(mod, "triton_ascend.dynamic_cv_pipeline.rc")
    if rc == 3:
        metadata["disable_vf_operand_substitution"] = True
        return
    if rc != -1 and rc > 0:
        # When the option dynamic_cv_pipeline is set to False,
        # these options should also reverted.
        metadata["enable_dynamic_cv_pipeline"] = False
        metadata["enable_mixed_cv"] = kwargs["enable_mixed_cv"]
        metadata["disable_auto_inject_block_sync"] = kwargs["disable_auto_inject_block_sync"]
        metadata["set_workspace_multibuffer"] = kwargs["set_workspace_multibuffer"]
        if opt.debug:
            print(f"SSBUFFER return code={rc}, will fallback to enable_dynamic_cv_pipeline=False")


def _get_dump_paths(hash_key: str, src_path: str, dst_path: str) -> Tuple[str, str]:
    dump_manager = get_dump_manager(hash_key)
    return (dump_manager._make_path(os.path.basename(src_path)), dump_manager._make_path(os.path.basename(dst_path)))


def _with_debug_line(npubin_stage, options):
    """Wrap an npubin-producing stage so the emitted kernel binary gets
    .debug_line cleanup for msdebug stepping. No-op unless
    LLVM_EXTRACT_DI_LOCAL_VARIABLES is set; never raises (failure returns the
    artifact unchanged), so it cannot break a build."""

    def stage(src, metadata):
        artifact = npubin_stage(src, metadata)
        return rewrite_debug_line(artifact, metadata=metadata, options=options)

    return stage


# Stage extensions named after the BiShengIR step they drive, for diagnostics.
_STAGE_NAMES = {
    "ttadapter": "ConvertTritonIRToLinalgIR",
    "npubin": "ConvertLinalgIRToBinary",
    "bcmlir": "BytecodeToLinalgIRByBishengirOpt",
    "mlirbc": "LinalgIRToBytecodeByTritonMLIROpt",
}


def _kernel_cache_hint(ext, metadata):
    """The cache location upstream appends to a stage failure.

    Core compile() has fn_cache_manager in scope; a backend stage does not, so
    rebuild it from metadata["hash"] through the same get_cache_manager used at
    compiler.py:279 (knobs.cache.manager_class or FileCacheManager), which
    yields the same cache_dir. A custom manager class need not expose one, and
    the core's file_name is not reachable here, so that case names the stage
    artifact only.
    """
    from triton.runtime.cache import FileCacheManager, get_cache_manager

    hash_key = (metadata or {}).get("hash")
    if hash_key:
        try:
            manager = get_cache_manager(hash_key)
        except Exception:
            # A cache manager that cannot be built must not replace the real
            # compilation diagnostic with its own failure.
            manager = None
        if isinstance(manager, FileCacheManager) and manager.cache_dir:
            return f"\n\n[INFO]: The compiled kernel cache is in {manager.cache_dir}\n\n"
    return f"\n\n[INFO]: The compiled kernel cache is kernel.{ext}\n\n"


def _with_tool_diagnostics(ext, stage):
    """Wrap an Ascend compilation stage so a tool failure keeps its diagnostic.

    These stages shell out, and CalledProcessError.str() carries only the
    command line while the diagnostic sits in stderr. The tuners decide
    whether a failure is recoverable by matching the compiler's message
    (runtime/ubtuner.py matches "ub overflow") and autotuner classifies
    failures by isinstance(exc, MLIRCompilationError), so both the text and
    the type have to travel with the exception.

    triton-ascend does this by patching the stage loop in core compiler.py to
    catch Exception. FlagTree does not apply that patch, so the handling lives
    here, leaving core Triton untouched the way the nvidia backend formats its
    own ptxas failures. The catch is deliberately narrowed to SubprocessError
    rather than Exception: autotuner skips a config on MLIRCompilationError, so
    wrapping an in-process Python defect would let it pass for "this config
    does not compile" and be visible only through the exception chain (the risk
    flagged in docs/backend/ascend-mlir-compilation-error.md). SubprocessError
    still covers every external-tool failure, including TimeoutExpired.
    """
    # Imported lazily: triton.compiler imports triton.backends, which loads
    # this module, so a module-level import would be circular.
    from triton.compiler.errors import MLIRCompilationError

    stage_name = _STAGE_NAMES.get(ext, "MLIRCompile")

    def run(src, metadata):
        try:
            return stage(src, metadata)
        except subprocess.SubprocessError as e:
            detail = getattr(e, "stderr", None)
            if detail:
                if isinstance(detail, bytes):
                    detail = detail.decode("utf-8", errors="replace")
                else:
                    detail = str(detail)
            else:
                # A tool that failed without captured stderr still belongs to
                # the stage; upstream reports str(e) for exactly this case.
                detail = str(e)
            detail += _kernel_cache_hint(ext, metadata)
            raise MLIRCompilationError(stage_name, detail) from e

    return run


def _graph_optimize_kwargs(opt):
    kwargs = {
        "ub_capacity_bytes": graph_ub_budget_bytes_for_arch(opt.target_arch),
        "compile_mode": opt.compile_mode,
    }
    rule_mask = getattr(opt, "rule_mask", DEFAULT_GRAPH_OPTIMIZATION_RULE_MASK)
    if rule_mask != DEFAULT_GRAPH_OPTIMIZATION_RULE_MASK:
        kwargs["rule_mask"] = rule_mask
        kwargs["ub_safety_percent"] = 80
        kwargs["reserved_ub_bytes"] = 0
        kwargs["mapping_ub_capacity_bytes"] = (ub_size_in_kbytes_for_arch(opt.target_arch) * 1024)
    return kwargs


def _serialize_module(mod, opt):
    if getattr(opt, "debug", False) or not _is_debug_line_info_disabled() or _enable_msdebug():
        return str(mod)
    return mod.str_nodebug()


def make_ttir(mod, metadata, opt):
    if "hash" not in metadata:
        metadata["hash"] = hashlib.sha256(f"{mod}-{metadata}".encode()).hexdigest()
    # the same optimize pass for triton-ir as all other backends
    pm = ir.pass_manager(mod.context)
    pm.enable_debug()
    passes.common.add_inliner(pm)
    # passes.ttir.add_rewrite_tensor_pointer(pm)
    passes.ttir.add_rewrite_tensor_descriptor_to_pointer(pm)
    passes.ttir.add_combine(pm)
    passes.common.add_canonicalizer(pm)
    passes.ttir.add_reorder_broadcast(pm)
    # commonir: lane-vectorize merges per-lane tensor computations into a single
    # tensor with a synthesized leading lane dimension.
    ascend.passes.ttir.add_lane_vectorize(pm)
    passes.common.add_cse(pm)
    passes.common.add_licm(pm)
    passes.common.add_symbol_dce(pm)
    passes.ttir.add_loop_unroll(pm)
    if opt.enable_graph_optimize:
        ascend.passes.ttir.add_graph_optimize(pm, **_graph_optimize_kwargs(opt))
    pm.run(mod, 'make_ttir')
    if opt.debug:
        dump_manager = get_dump_manager(metadata["hash"])
        print(f"Dumping intermediate results to {dump_manager.cache_dir}")
        dump_manager.put(str(mod), "kernel.ttir.mlir", binary=False)

    return mod


def ttir_to_linalg(mod, metadata, opt, *, named_ops=True):
    # use triton_adapter to lower Triton-MLIR to linalg
    # Get Triton-MLIR as string
    ttir_code = _serialize_module(mod, opt)
    auto_map_parallel_blocks_enabled = _is_auto_map_parallel_blocks_enabled()
    # This is compiler-derived safety metadata, never a user compile option.
    # Derive it even when the feature is currently disabled so a later runtime
    # environment change cannot enable AutoBlockify for unsafe TTIR.
    blacklist_reasons = _get_auto_blockify_blacklist_reasons(ttir_code)
    has_auto_blockify_blacklist_op = bool(blacklist_reasons)
    metadata["has_auto_blockify_blacklist_op"] = has_auto_blockify_blacklist_op
    if auto_map_parallel_blocks_enabled and has_auto_blockify_blacklist_op and blacklist_reasons:
        kernel_name = re.search(r"tt\.func\spublic\s+@(\w+)", ttir_code).group(1)
        _warn_auto_blockify_disabled(kernel_name or "<unknown>", blacklist_reasons)
    with tempfile.TemporaryDirectory() as tmpdir:
        src_path = os.path.join(tmpdir, "kernel.ttir.mlir")
        dst_path = os.path.join(tmpdir, "kernel.ttadapter.mlir")
        if opt.debug:
            Path(src_path).write_text(ttir_code)
        triton_adapter_opt_path = _get_triton_adapter_opt_path()

        # Select analysis is a fixed lowering policy, not a user compile option.
        enable_select_analysis = True
        compile_on_910_95 = metadata["compile_on_910_95"]
        compile_mode = opt.compile_mode
        metadata["compile_mode"] = compile_mode
        enable_mixed_cv = metadata.get("enable_mixed_cv")
        disable_auto_inject_block_sync = metadata.get("disable_auto_inject_block_sync")
        set_workspace_multibuffer = metadata.get("set_workspace_multibuffer")

        # Inject grid tile-count hint for ChunkCoalescing. When the kernel
        # has no boundary mask but grid[axis] is known at compile time (e.g.
        # from constexpr nchunks), the pass uses this to safely choose H.
        grid_num_tiles = metadata.get("grid_num_tiles")
        if isinstance(grid_num_tiles, int) and grid_num_tiles > 0:
            try:
                _builder = ascend.ir.ascendnpu_ir_builder(mod.context, opt.target_arch)
                mod.set_attr("hacc.grid_num_tiles", _builder.parse_attr(f"{grid_num_tiles} : i32"))
            except Exception:
                pass  # graceful fallback: pass runs without hint

        pm = ir.pass_manager(mod.context)
        pm.enable_debug()
        if distributed is not None:
            distributed.ascend_passes.ttgpuir.add_convert_triton_distributed_to_hivm(pm)

        ascend.passes.ttir.add_triton_control_flow_opt(pm)
        ascend.passes.ttir.add_triton_to_structure(pm, False, False)
        ascend.passes.ttir.add_discrete_mask_access_conversion(pm, compile_on_910_95, compile_mode)
        ascend.passes.ttir.add_triton_to_annotation(pm)
        ascend.passes.ttir.add_triton_to_unstructure(pm, compile_on_910_95, compile_mode)
        ascend.passes.ttir.add_triton_to_hivm(pm)
        ascend.passes.ttir.add_triton_to_hfusion(pm, compile_on_910_95)
        ascend.passes.ttir.add_triton_to_llvm(pm)
        ascend.passes.ttir.add_bubble_up_operation(pm)
        ascend.passes.ttir.add_triton_to_structure(pm, False, False)

        # commonir: fold tile.* ops away before the linalg lowering, which does
        # not know the CommonIR dialect. Followed by the inliner because the
        # lowering can leave unresolved !tile.buf function signatures.
        ascend.passes.ttir.add_commonir_to_hivm(pm)
        passes.common.add_inliner(pm)
        passes.common.add_canonicalizer(pm)

        ascend.passes.ttir.add_reduce_sum_strength(pm, opt.enable_reduce_sum_strength, 2)

        ascend.passes.ttir.add_triton_to_linalg(pm, False, named_ops, False, enable_select_analysis,
                                                compile_on_910_95, compile_mode)

        # commonir: a second run, since the linalg lowering can reintroduce
        # tile.* ops from buffer-typed values.
        ascend.passes.ttir.add_commonir_to_hivm(pm)
        passes.common.add_canonicalizer(pm)
        passes.common.add_cse(pm)
        # Restricted to 910_95/950. The merged buffer is written by two disjoint
        # memref.copy ops, and on 910_9362 the generated code only makes the
        # copy next to the surviving to_tensor visible, so the other half of the
        # tile reads back the padding value. The IR is unchanged through
        # bishengir-opt, so the loss happens in code generation.
        if compile_on_910_95:
            ascend.passes.ttir.add_merge_concat_load_buffer(pm)
        if metadata["enable_dynamic_cv_pipeline"]:
            metadata["set_workspace_multibuffer"] = 0
            metadata["enable_mixed_cv"] = True
            metadata["disable_auto_inject_block_sync"] = True

            # Must run before add_dynamic_cv_pipeline because the driven
            # AddMultiBufferInnerScope pass reads the module-level
            # `ssbuffer.insertionOptimization` attribute (set here) at run time.
            # Keep the existing default-on buffer insertion behavior.
            ascend.passes.ttir.set_enable_buffer_insert_optimization(mod)
            ascend.passes.ttir.add_dynamic_cv_pipeline(pm, compile_on_910_95)

        if _enable_msdebug():
            ascend.passes.ttir.add_normalize_debug_line_locations(pm)

        _intra_val = metadata.get("buf_slot_num_of_veccore")
        if _intra_val is not None:
            ascend.passes.ttir.set_buffer_count(mod, "INTRA", _intra_val)

        _inter_val = metadata.get("buf_slot_num_of_crosscore")
        if _inter_val is not None:
            ascend.passes.ttir.set_buffer_count(mod, "INTER", _inter_val)

        _load_val = metadata.get("buf_slot_num_of_gm")
        if _load_val is not None:
            ascend.passes.ttir.set_buffer_count(mod, "LOAD", _load_val)

        if opt.debug:
            # Print a command for replaying the pass pipeline outside Python.
            # Module attributes set above are not carried by --pass-pipeline.
            print_src_path, print_dst_path = _get_dump_paths(metadata["hash"], src_path, dst_path)
            try:
                triton_opt_path = _get_triton_opt_path()
            except EnvironmentError:
                # Wheels bundle this tool, but it is only named here, not
                # executed. A missing debug tool must not abort compilation.
                triton_opt_path = "triton-opt"
            cmd = [
                triton_opt_path,
                print_src_path,
                f"--pass-pipeline={pm.get_pipeline_str()}",
                "--mlir-print-debuginfo",
                "-o",
                print_dst_path,
            ]
            print(f"[DEBUG] cmd list: {shlex.join(cmd)}")

        pm.run(mod, 'ttir_to_linalg')
        _adjust_metadata_by_module_result(mod, metadata, opt, enable_mixed_cv=enable_mixed_cv,
                                          disable_auto_inject_block_sync=disable_auto_inject_block_sync,
                                          set_workspace_multibuffer=set_workspace_multibuffer)
        _export_program_grid_metadata(mod, metadata)

        if opt.debug:
            dump_manager = get_dump_manager(metadata["hash"])
            dump_manager.put(str(mod), "kernel.ttadapter.mlir", binary=False)

        return _serialize_module(mod, opt)


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

        # The --emit-bytecode flag ensures output is in BC format.
        # Pin v4 for the LLVM 22 producer / LLVM 19 consumer bridge:
        # the consumer cannot read the producer's default v6 location encoding.
        # Override only when the selected consumer supports that bytecode version.
        bytecode_version = os.environ.get("TRITON_ASCEND_BYTECODE_VERSION", "4")
        subprocess.run([
            triton_mlir_opt_path,
            ttadapter_path,
            "--emit-bytecode",
            f"--emit-bytecode-version={bytecode_version}",
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

        cmd = [
            bishengir_opt_path,
            bc_path,
        ]
        if opt.debug:
            cmd += ["--mlir-print-debuginfo"]
        cmd += ["-o", mlir_path]
        proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(
                f"bishengir-opt failed with exit code {proc.returncode}\n"
                f"command: {shlex.join(cmd)}\n"
                f"stdout:\n{proc.stdout}\n"
                f"stderr:\n{proc.stderr}")

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

    # Inserted by DiscreteMaskAccessConversionPass / MemOpConverter when discrete
    # masked stores need cross-block exclusion (e.g. hivm.sync_block_lock).
    SYNC_BLOCK_LOCK_REGEX = r'\bsync_block_lock\b'

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
    module_attrs = re.search(r'\bmodule(?:\s+@\w+)?\s+attributes\s*\{([^}]*)\}', linalg)
    metadata["has_full_row_copy"] = bool(module_attrs and re.search(r'\btt\.full_row_copy\b', module_attrs.group(1)))
    # Force disable auto tile and bind subblock if attribute is present in module
    metadata["auto_tile_and_bind_subblock"] = not re.search(DISABLE_AUTO_TILE_AND_BIND_SUBBLOCK_REGEX, linalg)
    # Turn off auto-blockify only for the ORDERED (token-ring) sync_block_lock:
    if re.search(SYNC_BLOCK_LOCK_REGEX, linalg) and not re.search(r"sync_block_lock_unordered", linalg):
        metadata["has_auto_blockify_blacklist_op"] = True
    # Mixed kernels use (block, subblock) as the unordered-lock participant
    # identity. Pure AIV kernels keep subblock tiling disabled: they neither
    # need it nor necessarily have the FFTS runtime resource it requires.
    has_unordered_sync_block_lock = re.search(r"sync_block_lock_unordered", linalg) is not None
    metadata["has_unordered_sync_block_lock"] = has_unordered_sync_block_lock
    if has_unordered_sync_block_lock:
        mix_mode = re.search(MIX_MODE_REGEX, linalg).group(1)
        if mix_mode != "mix":
            metadata["auto_tile_and_bind_subblock"] = False
        # One metadata cache line for runtime participant_num, plus one
        # choosing and one ticket cache line per participant. Each cache line is
        # 8 i64. This fallback is for one lock; the bishengir callback supplies
        # the exact total after lowering.
        metadata["sync_block_lock_layout"] = 1 << 32
        metadata["lock_init_val"] = 0
    # the mix mode is also encoded into metadata['name'] for runtime to distinguish
    metadata["mix_mode"] = re.search(MIX_MODE_REGEX, linalg).group(1)
    metadata["parallel_mode"] = re.search(PARALLEL_MODE_REGEX, linalg).group(1)
    metadata["kernel_name"] = re.search(KERNEL_NAME_REGEX, linalg).group(1)
    # Check the function load_binary in npu_driver.py.
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
    # Keep this as compiler-derived safety metadata.  In particular, do not
    # trust a caller-provided False value to override an unsafe TTIR pattern.
    metadata["has_auto_blockify_blacklist_op"] = bool(_get_auto_blockify_blacklist_reasons(ttir))
    # Parse all tensor kinds from arguments
    metadata["tensor_kinds"] = [int(kind) for _, kind in re.findall(TENSOR_KIND_REGEX, ttir)]
    return metadata


# 3.5 compatibility switches used by the TLE DSA tutorials and custom ops.
# The NPUOptions field names match the bishengir flag names with underscores.
_COMPAT_BISHENGIR_OPTIONS = (
    "disable_auto_cv_work_space_manage",
    "enable_legacy_insert_load_store_for_mix_cv",
)


def _append_compat_bishengir_options(options, metadata):
    for option in _COMPAT_BISHENGIR_OPTIONS:
        value = metadata.get(option)
        if value is not None:
            options.append(f"--{option.replace('_', '-')}={value}")


def _auto_multi_buffer_limit(metadata):
    """Return ``--limit-auto-multi-buffer-of-local-buffer`` value, if any.

    The implicit ``no-limit`` default only makes sense while auto multi-buffer
    is enabled. The 950 default disables it (see ``NPUOptions.multibuffer``) to
    stay within the 248 KiB UB, and then the limit flag must not be forwarded.
    """
    value = metadata["limit_auto_multi_buffer_of_local_buffer"]
    if value is None and metadata.get("multibuffer", True):
        value = "no-limit"
    return value


def get_common_bishengir_compile_options(metadata):
    bishengir_target = metadata['target'].arch
    options = [f"--target={bishengir_target}"]
    _append_compat_bishengir_options(options, metadata)
    return options


@functools.lru_cache()
def _supports_full_row_copy_inlining(compiler_path):
    try:
        result = subprocess.run([compiler_path, "--help"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                timeout=10, check=False)
    except (OSError, subprocess.SubprocessError):
        return False
    return result.returncode == 0 and "--enable-lib-call-no-inline" in result.stdout


def _full_row_copy_compile_options(metadata, compiler_path):
    # A shared noinline load helper loses constant-width specialization when
    # the masked tail also calls it. Scope this to compiler-versioned A5 SIMD
    # loads; other kernels retain the NPU compiler's default inlining policy.
    if (metadata.get("has_full_row_copy", False) and metadata["target"].arch.startswith("Ascend950")
            and metadata.get("mix_mode") == "aiv" and metadata.get("parallel_mode") == "simd"
            and _supports_full_row_copy_inlining(compiler_path)):
        return ["--enable-lib-call-no-inline=false"]
    return []


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


def _dump_kernel_binary(metadata_hash: str, bin_path: str):
    """Copy the compiled kernel object into the TRITON_DEBUG dump directory."""
    dump_manager = get_dump_manager(metadata_hash)
    dump_manager.put(Path(bin_path).read_bytes(), os.path.basename(bin_path), binary=True)


def try_compile_with_config(linalg: str, ub_config: Dict[str, Any], metadata: dict, opt) -> Tuple[bool, str]:
    """
    Try to compile with given UB config, return (success, error_msg).
    If compilation fails due to UB overflow or other errors, returns (False, error_msg).
    If compilation succeeds, returns (True, "").
    """
    # Must import from ubtuner.py - this is the single source of truth
    from triton.backends.ascend.runtime.ubtuner import UB_TO_NPU_OPTION_MAP
    option_mapping = UB_TO_NPU_OPTION_MAP

    metadata = dict(metadata)

    for ub_key, meta_key in option_mapping.items():
        if ub_key in ub_config:
            metadata[meta_key] = ub_config[ub_key]

    # Choose compile function based on options
    if opt.compile_on_910_95:
        compile_func = linalg_to_bin_enable_npu_compile_910_95
    else:
        compile_func = linalg_to_bin_enable_npu_compile_A2_A3

    try:
        compile_func(linalg, metadata, opt)
        return (True, "")
    except subprocess.CalledProcessError as e:
        error_msg = e.stderr.decode('utf-8') if e.stderr else str(e)
        return (False, error_msg)
    except Exception as e:
        return (False, str(e))


def linalg_to_bin_enable_npu_compile_910_95(linalg: str, metadata, opt):
    linalg, metadata = _parse_linalg_metadata(linalg, metadata)
    _finalize_program_launch_policy(metadata, opt)
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_file_name = "kernel.mlir"
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

        multibuffer_mode = metadata.get("multibuffer_mode")
        if multibuffer_mode is not None:
            mode_arg = "[" + ",".join(f"({level},{count})" for level, count in multibuffer_mode) + "]"
            _compile_option_list.append(f"--multibuffer-mode={mode_arg}")

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

        vf_fusion_mode = metadata["vf_fusion_mode"]
        if vf_fusion_mode is not None:
            _compile_option_list += [
                f"--vf-fusion-mode={vf_fusion_mode}",
            ]

        enable_preload = metadata["enable_preload"]
        if enable_preload is not None:
            _compile_option_list += [
                f"--enable-preload={enable_preload}",
            ]

        disable_tightly_coupled_buffer_reuse = metadata["disable_tightly_coupled_buffer_reuse"]
        if disable_tightly_coupled_buffer_reuse:
            _compile_option_list += ["--disable-tightly-coupled-buffer-reuse"]

        _compile_option_list += [
            f"--enable-auto-bind-sub-block={get_auto_bind_sub_block_option(metadata)}",
        ]

        _compile_option_list += [f"--enable-optimized-metaop={is_cann_version_at_least(9, 2, 0)}"]

        npu_utils = NPUUtils()
        if npu_utils.has_device_limit():
            _compile_option_list += [
                f"--custom-aic-number={npu_utils.get_aicore_num()}",
            ]
            _compile_option_list += [
                f"--custom-aiv-number={npu_utils.get_aivector_core_num()}",
            ]

        if force_disable_ffts(opt.target_arch):
            _compile_option_list += ["--disable-ffts"]
        if _is_ascend_sanitizer_enabled():
            _compile_option_list += ["--enable-sanitizer=true"]
        if not _is_debug_line_info_disabled():
            _compile_option_list += ["--enable-debug-info=true"]
        if _enable_msdebug():
            _compile_option_list += ["--enable-debug-variables=true"]
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

        auto_multi_buffer = _auto_multi_buffer_limit(metadata)
        if auto_multi_buffer is not None:
            _compile_option_list += \
                [f"--limit-auto-multi-buffer-of-local-buffer={auto_multi_buffer}"]
        auto_multi_buffer_buffer = metadata["limit_auto_multi_buffer_buffer"]
        if auto_multi_buffer_buffer is not None:
            _compile_option_list += \
                [f"--limit-auto-multi-buffer-buffer={auto_multi_buffer_buffer}"]

        enable_mixed_cv = metadata["enable_mixed_cv"]
        if enable_mixed_cv is not None:
            _compile_option_list += \
                [f"--enable-mixed-cv={enable_mixed_cv}"]

        enable_dynamic_cv_pipeline = metadata["enable_dynamic_cv_pipeline"]
        if enable_dynamic_cv_pipeline == True and not metadata.get("disable_vf_operand_substitution", False):
            _compile_option_list += [f"--enable-vf-operand-substitution=True"]

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

        if metadata["auto_blockify_enabled"]:
            _compile_option_list += ["--enable-auto-blockify-loop"]
        npu_compiler_path, env = _get_npucompiler_path()
        if npu_compiler_path.endswith("bishengir-compile"):
            _compile_option_list += [
                "--enable-hfusion-compile=true",
                "--enable-triton-kernel-compile=true",
            ]
            _compile_option_list += _full_row_copy_compile_options(metadata, npu_compiler_path)
            # Temporary until the NPU compiler enables batch matmul by default in Q4.
            if metadata.get("enable_hivm_batch_matmul"):
                _compile_option_list += ["--enable-hivm-batch-matmul"]
            if metadata.get("enable_vf_stack_limit"):
                _compile_option_list += ["--enable-vf-stack-limit"]
        bisheng_options = metadata["bisheng_options"]
        if bisheng_options is not None:
            _compile_option_list += [f"--append-bisheng-options={bisheng_options}"]
        if opt.debug:
            _compile_option_list += ["--mlir-print-ir-after-failure"]
            _compile_option_list += ["--mlir-print-stacktrace-on-diagnostic"]
            _compile_option_list += ["--bishengir-print-ir-after=hivm-graph-sync-solver"]

        vf_merge_level = metadata["vf_merge_level"]
        if vf_merge_level is not None:
            _compile_option_list += [f"--enable-vf-merge-level={vf_merge_level}"]

        hfusion_enable_multiple_consumer_fusion = metadata["hfusion_enable_multiple_consumer_fusion"]
        if hfusion_enable_multiple_consumer_fusion is not None:
            _compile_option_list += [
                f"--hfusion-enable-multiple-consumer-fusion={hfusion_enable_multiple_consumer_fusion}"
            ]

        plan_memory_strategy = metadata["plan_memory_strategy"]
        if plan_memory_strategy is not None:
            _compile_option_list += [f"--plan-memory-strategy={plan_memory_strategy}"]

        cmd_list = ([npu_compiler_path, ttadapter_path] + _compile_option_list + ["-o", bin_file])

        if opt.debug or os.getenv("TRITON_PRINT_AUTOTUNING", None) == "1":
            print_cmd_list = cmd_list.copy()
            print_cmd_list[1], print_cmd_list[-1] = _get_dump_paths(metadata["hash"], ttadapter_path, bin_path)
            print(f"[DEBUG] cmd_list: {shlex.join(print_cmd_list)}")

        try:
            ret = subprocess.run(cmd_list, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        except subprocess.CalledProcessError as e:
            if opt.debug:
                _save_npuir_debug_output(e.stdout, e.stderr, tmpdir, metadata["hash"])
            stdout = (e.stdout or b"").decode("utf-8", errors="replace")
            stderr = (e.stderr or b"").decode("utf-8", errors="replace")
            failed_dir = os.path.join(tempfile.gettempdir(), f"triton_ascend_failed_{os.getpid()}")
            os.makedirs(failed_dir, exist_ok=True)
            failed_mlir = os.path.join(failed_dir, "kernel.mlir")
            Path(failed_mlir).write_text(Path(ttadapter_path).read_text())
            raise RuntimeError(
                f"bishengir-compile failed with exit code {e.returncode}\n"
                f"command: {shlex.join(cmd_list)}\n"
                f"failed MLIR: {failed_mlir}\n"
                f"stdout:\n{stdout}\n"
                f"stderr:\n{stderr}") from e

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

        if opt.debug:
            _dump_kernel_binary(metadata["hash"], bin_path)

        if Path(callback_path).is_file():
            lib = ctypes.CDLL(callback_path)
            __get_metadata_attr_by_callback(lib, "_infer_task_type_function", metadata, "bs_task_type")
            __get_metadata_attr_by_callback(lib, "_infer_workspace_shape_function", metadata, "workspace_size")
            __get_metadata_attr_by_callback(lib, "_infer_sync_block_lock_num_function", metadata,
                                            "sync_block_lock_layout")
            __get_metadata_attr_by_callback(lib, "_infer_sync_block_lock_init_function", metadata, "lock_init_val")

        return Path(bin_path).read_bytes()


def linalg_to_bin_enable_npu_compile_A2_A3(linalg: str, metadata, opt):
    linalg, metadata = _parse_linalg_metadata(linalg, metadata)
    _finalize_program_launch_policy(metadata, opt)
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_file_name = "kernel.mlir"
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
        _append_compat_bishengir_options(_compile_option_list, metadata)

        multibuffer_mode = metadata.get("multibuffer_mode")
        if multibuffer_mode is not None:
            mode_arg = "[" + ",".join(f"({level},{count})" for level, count in multibuffer_mode) + "]"
            _compile_option_list.append(f"--multibuffer-mode={mode_arg}")

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

        disable_size_align_for_cast = metadata["disable_size_align_for_cast"]
        if disable_size_align_for_cast is not None:
            _compile_option_list += [
                f"--disable-size-align-for-cast={disable_size_align_for_cast}",
            ]

        enable_preload = metadata["enable_preload"]
        if enable_preload is not None:
            _compile_option_list += [
                f"--enable-preload={enable_preload}",
            ]

        _compile_option_list += [
            f"--enable-auto-bind-sub-block={get_auto_bind_sub_block_option(metadata)}",
        ]

        _compile_option_list += [f"--enable-optimized-metaop=false"]

        if _is_ascend_sanitizer_enabled():
            _compile_option_list += ["--enable-sanitizer=true"]
        if not _is_debug_line_info_disabled():
            _compile_option_list += ["--enable-debug-info=true"]

        if _enable_print_ub_bits():
            _compile_option_list += ["--enable-print-memory-allocated-size"]

        if _enable_dump_memory_info():
            _compile_option_list += ["--enable-memory-display=true"]

        if _enable_msdebug():
            _compile_option_list += ["--enable-debug-variables=true"]

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

        enable_libdevice = os.getenv("TRITON_ENABLE_LIBDEVICE", False)
        if enable_libdevice:
            _compile_option_list += [f"--link-aicore-bitcode={get_libdevice()}"]

        if metadata["auto_blockify_enabled"]:
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
            _compile_option_list += ["--mlir-print-stacktrace-on-diagnostic"]
            _compile_option_list += ["--bishengir-print-ir-after=hivm-graph-sync-solver"]

        cmd_list = ([npu_compiler_path, ttadapter_path] + _compile_option_list + ["-o", bin_file])

        if opt.debug or os.getenv("TRITON_PRINT_AUTOTUNING", None) == "1":
            print_cmd_list = cmd_list.copy()
            print_cmd_list[1], print_cmd_list[-1] = _get_dump_paths(metadata["hash"], ttadapter_path, bin_path)
            print(f"[DEBUG] cmd_list: {shlex.join(print_cmd_list)}")

        try:
            ret = subprocess.run(cmd_list, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        except subprocess.CalledProcessError as e:
            if opt.debug:
                _save_npuir_debug_output(e.stdout, e.stderr, tmpdir, metadata["hash"])
            stdout = (e.stdout or b"").decode("utf-8", errors="replace")
            stderr = (e.stderr or b"").decode("utf-8", errors="replace")
            failed_dir = os.path.join(tempfile.gettempdir(), f"triton_ascend_failed_{os.getpid()}")
            os.makedirs(failed_dir, exist_ok=True)
            failed_mlir = os.path.join(failed_dir, "kernel.mlir")
            Path(failed_mlir).write_text(Path(ttadapter_path).read_text())
            raise RuntimeError(
                f"bishengir-compile failed with exit code {e.returncode}\n"
                f"command: {shlex.join(cmd_list)}\n"
                f"failed MLIR: {failed_mlir}\n"
                f"stdout:\n{stdout}\n"
                f"stderr:\n{stderr}") from e

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

        if opt.debug:
            _dump_kernel_binary(metadata["hash"], bin_path)

        if Path(callback_path).is_file():
            lib = ctypes.CDLL(callback_path)
            __get_metadata_attr_by_callback(lib, "_infer_task_type_function", metadata, "bs_task_type")
            __get_metadata_attr_by_callback(lib, "_infer_workspace_shape_function", metadata, "workspace_size")
            __get_metadata_attr_by_callback(lib, "_infer_sync_block_lock_num_function", metadata,
                                            "sync_block_lock_layout")
            __get_metadata_attr_by_callback(lib, "_infer_sync_block_lock_init_function", metadata, "lock_init_val")

        return Path(bin_path).read_bytes()


def get_libdevice():
    current = os.path.dirname(__file__)
    return os.path.join(current, "lib/libdevice.10.bc")


def _is_a5_target_arch(arch: str) -> bool:
    return isinstance(arch, str) and arch.startswith(("Ascend910_95", "Ascend950"))


def _get_libdevice_compile_state(arch: str) -> tuple[bool, bool]:
    return (
        bool(os.getenv("TRITON_ENABLE_LIBDEVICE", False)),
        bool(os.getenv("TRITON_ENABLE_LIBDEVICE_SIMT", False)) and _is_a5_target_arch(arch),
    )


_CANONICAL_COMPILE_MODES = ("simd", "simd_simt_template", "simt_only")
_COMPILE_MODE_ALIASES = {
    "unstructured_in_simt": "simd_simt_template",
}


def _normalize_compile_mode(compile_mode, arch: str) -> str:
    """Validate and canonicalize the public compile-mode contract.

    ``simd_simt_template`` is the portable default: it retains the ordinary
    SIMD pipeline on A2/A3 and enables the existing template-SIMT subpaths on
    Ascend 950.  ``unstructured_in_simt`` is an equivalent spelling of that mode.
    Pure-SIMT remains an Ascend 950-only explicit compile mode.
    """
    if not isinstance(compile_mode, str):
        raise ValueError("compile_mode must be a string; expected one of: " + ", ".join(_CANONICAL_COMPILE_MODES))

    canonical_mode = _COMPILE_MODE_ALIASES.get(compile_mode, compile_mode)
    if canonical_mode not in _CANONICAL_COMPILE_MODES:
        raise ValueError(f"invalid compile_mode={compile_mode!r}; expected one of: " +
                         ", ".join(_CANONICAL_COMPILE_MODES))

    if canonical_mode == "simt_only" and not _is_a5_target_arch(arch):
        raise ValueError('compile_mode="simt_only" is supported only on Ascend 950 targets.')
    return canonical_mode


@dataclass(frozen=True)
class NPUOptions:
    debug: bool = False
    sanitize_overflow: bool = True
    # Backend-only construction input.  AscendBackend.parse_options injects
    # GPUTarget.arch and never forwards a user-supplied compile option.
    arch: InitVar[str] = ""
    rule_mask: InitVar[int] = DEFAULT_GRAPH_OPTIMIZATION_RULE_MASK
    # This becomes compiler metadata, so its name must also be valid for the
    # namedtuple constructed by CompiledKernel on Python 3.10.
    target_arch: str = field(init=False, repr=False)

    cluster_dims: tuple = (1, 1, 1)
    num_warps: int = 32
    num_ctas: int = 1
    # Keep an omitted stage count distinct from an explicit mode.
    num_stages: Optional[int] = None
    # Ascend threads-per-warp is a backend capability, not a compile option.
    warp_size: int = field(default=32, init=False)
    ir_override: Optional[str] = None  # filename of a user-defined IR (*.{ttir|ttadapter|mlirbc|bcmlir|npubin})

    # Deprecated constructor-only compatibility input.  The supplied value is
    # ignored and replaced with the lowering selector derived from GPUTarget.arch.
    compile_on_910_95: Optional[bool] = field(default=None, repr=False, kw_only=True)
    enable_warp_specialization: bool = False
    enable_persistent: bool = False
    optimize_epilogue: bool = False
    enable_fp_fusion: bool = True
    launch_cooperative_grid: bool = False
    backend_name: str = 'cann'
    instrumentation_mode: str = ""
    enable_graph_optimize: bool = True
    supported_fp8_dtypes: Tuple[str] = ("fp8e5", "fp8e4b15", "fp8e4nv", "fp8e4b8", "fp8e5b16")
    deprecated_fp8_dtypes: Tuple[str] = ()
    vf_merge_level: int = 1
    default_dot_input_precision: str = "ieee"
    allowed_dot_input_precisions: Tuple[str] = ("ieee", "hf32")
    max_num_imprecise_acc_default: int = 0
    extern_libs: dict = None
    # Backend-managed BiSheng arguments. Keep the name discoverable so legacy
    # launch keywords are accepted, but never forward a user-supplied value.
    bisheng_options: str = field(default="-cce-link-aicore-ll-module " + get_libdevice(), init=False)
    # 3.5-compatible default: auto multi-buffer is disabled on the 950
    # generation, where the extra ring buffers can overflow the 248 KiB UB.
    multibuffer: Optional[bool] = None
    # Public dictionaries are stored as sorted pairs for hashing and JSON caches.
    multibuffer_mode: Optional[Union[Dict[str, int], Tuple[Tuple[str, int], ...]]] = None
    vf_fusion_mode: str = None
    enable_ubuf_saving: bool = None
    disable_size_align_for_cast: bool = None
    enable_preload: bool = None
    enable_auto_bind_sub_block: bool = None
    disable_tightly_coupled_buffer_reuse: bool = False
    # 3.5 compatibility switches used by the TLE DSA tutorials and custom ops.
    disable_auto_cv_work_space_manage: bool = None
    enable_legacy_insert_load_store_for_mix_cv: bool = None
    enable_hivm_auto_cv_balance: bool = None
    # Temporary 910_95 switch; the NPU compiler plans to make this default in Q4.
    enable_hivm_batch_matmul: bool = False
    # Only takes effect on the Ascend 950 non-pure-SIMT BiShengIR compilation path.
    enable_vf_stack_limit: bool = False
    sync_solver: bool = None
    unit_flag: bool = None
    enable_flatten: bool = None
    enable_auto_vectorize_v2: bool = None
    auto_vectorize_v2_max_fused_ops_num: int = None
    prevec_max_fused_ops_num: int = None
    inject_barrier_all: bool = None
    inject_block_all: bool = None
    limit_auto_multi_buffer_only_for_local_buffer: bool = None
    limit_auto_multi_buffer_of_local_buffer: str = None
    limit_auto_multi_buffer_buffer: str = None
    set_workspace_multibuffer: int = None
    tile_mix_vector_loop: int = None
    tile_mix_cube_loop: int = None
    disable_auto_inject_block_sync: bool = None
    enable_mixed_cv: bool = None
    enable_dynamic_cv_pipeline: bool = None
    enable_cube_block_merge: bool = False
    hfusion_enable_multiple_consumer_fusion: bool = None
    buf_slot_num_of_veccore: int = None
    buf_slot_num_of_crosscore: int = None
    buf_slot_num_of_gm: int = None

    # plan memory strategy: "default" (default) or "largest-first"
    plan_memory_strategy: str = None
    # Internal launch metadata.  The mode is initially derived from
    # compile_mode, then replaced with the mode emitted by TritonToLinalg.
    parallel_mode: str = field(default="simd", init=False)
    # Internal pure-SIMT state, derived from the effective lowering mode.
    is_pure_simt: bool = field(default=False, init=False)
    # Only takes effect on the pure-SIMT path.
    shared_mem_dynamic_size: int = None
    # Ascend 950 pure-SIMT-only option passed as -simt-optimization-mode
    # to bishengir-compile. Its value grammar belongs to the toolchain.
    # Individual digits are passed to various passes to control behavior,
    # and are parsed right-to-left.
    # For example, a value of 101 is interpreted as 0000101.
    # If left as 0, bishengir-compile sets this to 900101
    simt_optimization_mode: int = 0000000
    # Canonical modes: SIMD (D), SIMD with template-SIMT (P), and pure-SIMT
    # (T). ``unstructured_in_simt`` is an equivalent P spelling.
    compile_mode: str = "simd_simt_template"
    # disable simt fma optimization to get high precision
    disable_fma: bool = False

    # superblocking factor
    superblock_factor: int = 1

    # ChunkCoalescing: number of tiles along the outermost grid axis.
    # Auto-injected from static grid tuples; enables safe coalescing for
    # unmasked kernels whose grid dims are compile-time known.
    grid_num_tiles: int = None

    # Deprecated names remain visible to callers that use parse_options({})
    # to distinguish compile options from kernel arguments. Their public
    # values are ignored (or routed to canonical options) before construction.
    add_auto_scheduling: bool = field(default=False, init=False)
    allow_fp8e4nv: bool = field(default=False, init=False)
    auto_blockify_size: int = field(default=1, init=False)
    auto_tile_and_bind_subblock: bool = field(default=True, init=False)
    code_motion: Optional[bool] = field(default=None, init=False)
    enable_auto_blockify: Optional[bool] = field(default=None, init=False)
    enable_bishengir_simt_optimization: int = field(default=0, init=False)
    enable_buffer_insert_optimization: bool = field(default=True, init=False)
    enable_cce_vf_auto_sync: Optional[bool] = field(default=None, init=False)
    enable_cce_vf_remove_membar: Optional[bool] = field(default=None, init=False)
    enable_cross_if_fusion: bool = field(default=False, init=False)
    enable_drop_unit_dims: Optional[bool] = field(default=None, init=False)
    enable_linearize: Optional[bool] = field(default=None, init=False)
    enable_mask_fallback_conversion: bool = field(default=False, init=False)
    enable_nd2nz_on_vector: bool = field(default=False, init=False)
    enable_reduce_sum_strength: bool = field(default=False, init=False)
    enable_select_analysis: bool = field(default=True, init=False)
    enable_simt_reorder_instruction: bool = field(default=False, init=False)
    enable_sync_block_lock: Optional[bool] = field(default=None, init=False)
    enable_ub_refine_opt: bool = field(default=False, init=False)
    enable_vf_fusion: Optional[bool] = field(default=None, init=False)
    force_simt_only: bool = field(default=False, init=False)
    force_simt_template: bool = field(default=False, init=False)
    graph_optimize_emit_remarks: bool = field(default=False, init=False)
    graph_optimize_max_rewrites_per_function: int = field(default=64, init=False)
    graph_optimize_rule_mask: int = field(default=DEFAULT_GRAPH_OPTIMIZATION_RULE_MASK, init=False)
    graph_optimize_ub_capacity_bytes: Optional[int] = field(default=None, init=False)
    has_auto_blockify_blacklist_op: Optional[bool] = field(default=None, init=False)
    inter_cache_num: Optional[int] = field(default=None, init=False)
    intra_cache_num: Optional[int] = field(default=None, init=False)
    kernel_name: str = field(default="triton_", init=False)
    llvm_version: int = field(default=15, init=False)
    load_cache_num: Optional[int] = field(default=None, init=False)
    mix_mode: str = field(default="", init=False)
    ops_reorder: Optional[bool] = field(default=None, init=False)
    optimize_dynamic_offset: bool = field(default=False, init=False)
    simt_reorder_instruction: bool = field(default=False, init=False)
    storage_align: Optional[bool] = field(default=None, init=False)
    stream: Optional[int] = field(default=None, init=False)
    use_bytecode: bool = field(default=True, init=False)

    def __post_init__(self, arch, rule_mask):
        from triton.backends.ascend import _apply_ascend_patch

        _apply_ascend_patch()
        object.__setattr__(self, "target_arch", arch)
        # Plain init=False defaults live on the class. Materialize them in
        # the instance because both Inductor and JIT inspect options.__dict__.
        for name, option_field in self.__dataclass_fields__.items():
            if not option_field.init and name not in self.__dict__:
                object.__setattr__(self, name, option_field.default)
        # Keep the legacy name discoverable while its property and all
        # compiler decisions continue to use the injected target architecture.
        self.__dict__["arch"] = arch
        object.__setattr__(self, "multibuffer_mode", _multibuffer_mode_to_tuple(self.multibuffer_mode))
        if self.multibuffer_mode is not None and self.num_stages is not None:
            raise ValueError("num_stages and multibuffer_mode cannot be specified together")
        if self.multibuffer_mode is None and self.num_stages is None:
            object.__setattr__(self, "num_stages", 2)
        if self.compile_on_910_95 is not None:
            _warn_deprecated_npu_option("compile_on_910_95")
        object.__setattr__(
            self,
            "compile_on_910_95",
            isinstance(arch, str) and arch.startswith(("Ascend910_95", "Ascend950")),
        )
        if self.multibuffer is None:
            object.__setattr__(self, "multibuffer", not self.compile_on_910_95)
        try:
            normalized_rule_mask = normalize_graph_optimization_rule_mask(rule_mask)
        except ProgramGridContractError as error:
            raise ValueError(f"invalid GraphOptimize rule_mask: {error}") from error
        if normalized_rule_mask != DEFAULT_GRAPH_OPTIMIZATION_RULE_MASK:
            object.__setattr__(self, "rule_mask", normalized_rule_mask)

        compile_mode = str(_normalize_compile_mode(self.compile_mode, arch))
        object.__setattr__(self, "compile_mode", compile_mode)

        if compile_mode == "simt_only":
            object.__setattr__(self, "is_pure_simt", True)
            object.__setattr__(self, "parallel_mode", "simt")
        else:
            object.__setattr__(self, "parallel_mode", "simd")

        if self.is_pure_simt:
            if self.shared_mem_dynamic_size is None:
                object.__setattr__(self, "shared_mem_dynamic_size", 122880)
        else:
            object.__setattr__(self, "shared_mem_dynamic_size", 221184)

    def hash(self):
        key = "_".join([f"{name}-{val}" for name, val in self.__dict__.items()])
        key = "_".join([key, get_cann_version_file_hash()])
        return hashlib.sha256(key.encode("utf-8")).hexdigest()


def _get_npu_options_arch(options: NPUOptions) -> str:
    """Expose the injected target to the established lowering builder API.

    ``arch`` is an ``InitVar`` rather than a user compile option. Its legacy
    name is also serialized for option discovery, while this view always
    returns the target injected by ``AscendBackend.parse_options``.
    """
    return options.target_arch


NPUOptions.arch = property(_get_npu_options_arch)

_INTERNAL_NPU_OPTION_MARKERS = frozenset({
    "target_arch",
    "compile_on_910_95",
    "parallel_mode",
    "is_pure_simt",
})


def _is_internal_npu_options(options, target_arch: str) -> bool:
    """Recognize an NPUOptions metadata round trip by backend-only fields."""
    return (_INTERNAL_NPU_OPTION_MARKERS <= options.keys() and options.get("target_arch") == target_arch
            and options.get("compile_on_910_95") == _is_a5_target_arch(target_arch))


def _normalize_bishengir_simt_optimization_for_context(options: NPUOptions, raw_options) -> None:
    """Restrict the vendor SIMT optimization switch to its Ascend 950 pure-SIMT path."""
    option_name = "simt_optimization_mode"
    if option_name not in raw_options:
        return

    if _is_a5_target_arch(options.target_arch) and options.is_pure_simt:
        # The BiShengIR toolchain owns the option's value grammar; keep the
        # Python boundary free of numeric or range validation.
        return

    warnings.warn(
        "simt_optimization_mode only takes effect for Ascend 950 "
        "pure-SIMT compilation; ignoring the explicit value.",
        UserWarning,
        stacklevel=3,
    )
    object.__setattr__(options, option_name, 0)


def ttir_to_npubin(mod, metadata, opt):
    _export_program_grid_metadata(mod, metadata, require_row_contract=True)
    ttir_code = _serialize_module(mod, opt)
    metadata = _parse_ttir_metadata(ttir_code, metadata)
    _finalize_program_launch_policy(metadata, opt)

    with tempfile.TemporaryDirectory() as tmpdir:
        # prepare input
        src_path = os.path.join(tmpdir, "kernel.ttir.mlir")
        Path(src_path).write_text(ttir_code)
        # prepare output
        bin_file = os.path.join(tmpdir, "kernel")
        bin_path = os.path.join(tmpdir, "kernel.o")
        # build compile options
        _compile_option_list = get_common_bishengir_compile_options(metadata)
        if opt.is_pure_simt:
            _compile_option_list += ["--enable-hivm-compile=false"]
            _compile_option_list += ["--enable-triton-ir-compile"]
            _compile_option_list += ["--pure-simt"]
            _compile_option_list += [f"--num-warps={opt.num_warps}"]
            _compile_option_list += [f"--threads-per-warp={opt.warp_size}"]
            if opt.simt_optimization_mode != 0000000:
                _compile_option_list += [f"--simt-optimization-mode={opt.simt_optimization_mode}"]
            _compile_option_list += [f"--simt-stack-limit={get_simt_stack_limit()}"]
            if opt.shared_mem_dynamic_size is not None:
                _compile_option_list += [f"--shared-mem-dynamic-size={opt.shared_mem_dynamic_size}"]
            if opt.disable_fma:
                _compile_option_list += [f"--disable-fma"]
            if opt.compile_on_910_95:
                npu_utils = NPUUtils()
                if npu_utils.has_device_limit():
                    _compile_option_list += [
                        f"--custom-aic-number={npu_utils.get_aicore_num()}",
                        f"--custom-aiv-number={npu_utils.get_aivector_core_num()}",
                    ]

            bisheng_options = metadata["bisheng_options"]
            if bisheng_options is not None:
                _compile_option_list += [f"--append-bisheng-options={bisheng_options}"]

            if _is_auto_map_parallel_blocks_enabled() and not metadata.get("row_coalescing_applied", False):
                _compile_option_list += ["--enable-auto-blockify-loop"]
                if opt.superblock_factor > 1:
                    _compile_option_list += [f"--super-block-factor={opt.superblock_factor}"]

        npu_compiler_path, env = _get_npucompiler_path()
        cmd_list = ([npu_compiler_path, src_path] + _compile_option_list + ["-o", bin_file])
        ret = subprocess.run(cmd_list, env=env, capture_output=True, check=True)
        if not Path(bin_path).exists():
            error_msg = ret.stderr.decode('utf-8')
            print(f"[DEBUG] {bin_path} is not found")
            print(f"[DEBUG] Stderr:\n{error_msg}")
            raise subprocess.CalledProcessError(ret.returncode, cmd_list, ret.stdout, ret.stderr)
        return Path(bin_path).read_bytes()


def get_simt_stack_limit():
    # simt_stack_limit resolution precedence:
    #  1. torch_npu's acl_default.json "StackSize":{"simt_stack_size":N}.
    #  2. The kernel-time default simt_stack_limit=1152.
    _simt_stack_limit = 1152
    try:
        import torch_npu
        torch_npu_basic_path = os.path.dirname(torch_npu.__file__)
        _acl_cfg_path = os.path.join(torch_npu_basic_path, "acl_default.json")
        with open(_acl_cfg_path, "r") as f:
            _acl_cfg = json.load(f)
        _cfg_stack = _acl_cfg.get("StackSize", {}).get("simt_stack_size", None)
        if isinstance(_cfg_stack, int) and not isinstance(_cfg_stack, bool) and _cfg_stack > 0:
            _simt_stack_limit = _cfg_stack
    except Exception as e:
        print(f"[DEBUG] read acl_default.json failed: {e}")
    return _simt_stack_limit


class AscendBackend(BaseBackend):

    @staticmethod
    def supports_target(target: GPUTarget):
        return target.backend == "npu"

    @staticmethod
    def use_alignment_specialization(options: dict) -> bool:
        # The binder runs before parse_options. Normalize its option copy so
        # legacy selectors affect alignment specialization and the cache key.
        normalized = _remove_deprecated_npu_options(options, in_place=True)
        return normalized.get("compile_mode") == "simt_only"

    def __init__(self, target: GPUTarget) -> None:
        super().__init__(target)
        if target.backend == "npu":
            self.binary_ext = "npubin"
            # Include all binary file extensions (mlirbc is always emitted for normal kernels).
            self.binary_extensions = {"npubin", "mlirbc"}

    def parse_options(self, opts) -> Any:
        # TODO: get available targets when building options?
        if self.target.backend == "npu":
            _warn_deprecated_ascend_env_vars()
            option_names = {
                name
                for name, option_field in NPUOptions.__dataclass_fields__.items()
                if option_field.init and name not in {"arch", "compile_on_910_95"}
            }
            # Serialized NPUOptions include backend-only derived fields.  Use
            # those provenance markers instead of depending on every public
            # field being present, which changes whenever the dataclass evolves.
            internal_options = _is_internal_npu_options(opts, self.target.arch)
            # Normalize a copy, except for the legacy launch keyword below.
            # NPUOptions retains compatibility defaults for legacy names so
            # JIT and Inductor can still recognize them as compile options.
            normalized_opts = opts if internal_options else _remove_deprecated_npu_options(opts)
            args = {k: normalized_opts[k] for k in option_names if k in normalized_opts}
            options = NPUOptions(arch=self.target.arch, **args)
            # Lazy init enable_dynamic_cv_pipeline if not provided.
            # compile_on_910_95 is already resolved from the requested target.
            if options.enable_dynamic_cv_pipeline is None:
                object.__setattr__(options, "enable_dynamic_cv_pipeline", options.compile_on_910_95)
            if not internal_options:
                _normalize_bishengir_simt_optimization_for_context(options, normalized_opts)
                # Community JIT rejects the legacy launch keyword "stream"
                # even when options.__dict__ recognizes it. Consume it at
                # the Ascend boundary; the returned compatibility field stays
                # None and the runtime still selects the current stream.
                opts.pop("stream", None)
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
        # Ascend frontend initialization and with dispatch live in CodeGenerator.
        return {"min_dot_size": min_dot_size(self.target)}

    def load_dialects(self, ctx):
        from triton._C.libtriton import buffer_ir
        from triton._C.libtriton.ascend import ir as ascend_ir
        if distributed is not None:
            distributed.ir.load_dialects(ctx)
        buffer_ir.load_dialects(ctx)
        ascend_ir.load_dialects(ctx)
        ascend.load_dialects(ctx)

    def add_stages(self, stages, options, language):
        if self.target.backend != "npu":
            raise NotImplementedError(f"Backend '{self.target.backend}' is not supported. "
                                      "Please ensure the target backend is set to 'npu'.")
        stages["ttir"] = lambda src, metadata: make_ttir(src, metadata, options)
        if options.is_pure_simt:
            stages["npubin"] = (lambda src, metadata: ttir_to_npubin(src, metadata, options))
        else:
            stages["ttadapter"] = lambda src, metadata: ttir_to_linalg(src, metadata, options)
            # Normal kernels always convert Linalg IR to bytecode and back to MLIR text.
            stages["mlirbc"] = lambda src, metadata: linalg_to_bc_by_triton_mlir_opt(src, metadata, options)
            stages["bcmlir"] = lambda src, metadata: bc_to_linalg_by_bishengir_opt(src, metadata, options)
            if options.compile_on_910_95:
                stages["npubin"] = (
                    lambda src, metadata: linalg_to_bin_enable_npu_compile_910_95(src, metadata, options))
            else:
                stages["npubin"] = (
                    lambda src, metadata: linalg_to_bin_enable_npu_compile_A2_A3(src, metadata, options))
            stages["npubin"] = _with_debug_line(stages["npubin"], options)
        # Only the stages that shell out to BiShengIR tools. ttir runs the pass
        # manager in process, and stages the caller supplied are not ours to
        # rewrite. _STAGE_NAMES covers both paths, pure SIMT npubin included.
        for ext in _STAGE_NAMES:
            if ext in stages:
                stages[ext] = _with_tool_diagnostics(ext, stages[ext])

    @functools.lru_cache()
    def hash(self):
        # TODO fetch compiler version
        version_key = (self.target, _get_libdevice_compile_state(self.target.arch))
        return str(version_key)

    def get_module_map(self) -> Dict[str, ModuleType]:
        from triton.language.extra.cann import libdevice

        return {"triton.language.extra.libdevice": libdevice}
