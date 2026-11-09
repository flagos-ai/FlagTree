from triton.backends.compiler import BaseBackend, GPUTarget
from triton.backends import llvm19_toolchain
from triton._C.libtriton import ir, passes, xpu, llvm
from triton.runtime.cache import get_cache_manager
import subprocess
import tempfile
import re
import warnings
import logging
import os
import sys

from dataclasses import dataclass
import functools
from typing import Tuple, Optional
import hashlib
from pathlib import Path

# The desensitized LLVM 19 package ships no `opt`; the read-in step is
# `opt`-only, so disable it when the staged toolchain lacks it.  An explicit
# TRITON_LLVM19_READIN from the environment wins over this default.
try:
    _llvm19_bin = llvm19_toolchain.get_llvm19_bin_dir("xpu")
    if not os.path.exists(os.path.join(_llvm19_bin, "opt")):
        os.environ.setdefault("TRITON_LLVM19_READIN", "0")
except Exception:  # toolchain not staged yet -- leave the default alone
    pass


def run_cmd(cmd, env=None):
    result = subprocess.run(cmd, capture_output=True, env=env)

    if result.stderr:
        print(result.stderr, file=sys.stderr)

    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed ({result.returncode}): {cmd}\nstdout: {result.stdout}\nstderr: {result.stderr}")

    return result.stdout


def parse_floating_range_string(range_str: str) -> Optional[Tuple[Optional[float], Optional[float]]]:
    """Parse a "Start:End" floating-point range string such as "1.5:10.0".

    Returns:
        (start, end) with None for an omitted side, or None if the string
        is malformed.
    """

    parts = range_str.split(':')

    if len(parts) > 2:
        print(f"错误: 范围字符串 '{range_str}' 格式不正确，包含太多冒号。")
        return None

    full_parts = parts + [""] * (2 - len(parts))

    start_str = full_parts[0]
    end_str = full_parts[1]

    start_val: Optional[float] = None
    end_val: Optional[float] = None

    if start_str:
        try:
            start_val = float(start_str)
        except ValueError:
            print(f"错误: Start 部分 '{start_str}' 不是有效的浮点数。")
            return None

    if end_str:
        try:
            end_val = float(end_str)
        except ValueError:
            print(f"错误: End 部分 '{end_str}' 不是有效的浮点数。")
            return None

    # A bare number ("123.789") is an upper bound only: (None, 123.789).
    if len(parts) == 1 and start_val is not None:
        end_val = start_val
        start_val = None

    return start_val, end_val


@functools.lru_cache(None)
def file_hash(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


# Flags handed to the staged LLVM 19 llc by this backend.  Single source of
# truth for the codegen options: the call sites below build their `extra_args`
# from it, and XPUBackend.hash() folds it (together with
# llvm19_toolchain.LLC_ALWAYS_FLAGS) into the JIT cache key.  Without that the
# key would keep serving objects compiled with the old flags from a warm cache
# (KB E23).
_LLC_EXTRA = ("-O2", )


@functools.lru_cache(maxsize=None)
def _xtdk_lld_version(tool_dir: str) -> str:
    """`ld.lld --version` of the linker this backend's elfconv step drives.

    Toolchain identity for XPUBackend.hash(), the same ingredient the other two
    backends take from their own `path_to_xtdk_lld()`: the linker version moves
    the object bytes, so a warm cache must not outlive a toolchain change.  The
    tool dir is the one `path_to_xpu_compile_tool` resolves and that elfconv
    takes as `CLANG_PATH` (see make_xpubin).  Memoised because hash() runs once
    per compilation and an installed binary's version cannot change under a
    running process -- the effect the other two backends get from the lru_cache
    on hash() itself.

    **This must never raise**: it runs inside `hash()`, i.e. inside a cache key,
    the contract `llvm19_toolchain.codegen_key()` states for every ingredient it
    folds in.  `tool_dir` is not guaranteed to hold a linker either -- it is
    "the clang path" (`TRITON_XPU_CLANG_PATH`, a plain clang install has no
    `ld.lld`) or a per-arch directory that need not be staged (`xpu2`).  A
    missing tool is a fact about this toolchain, so it becomes the
    `lld:absent(<cause>)` placeholder -- the same shape `_table_digests()` uses
    for a package without a table -- which keeps the key honest: installing a
    linker later changes it.
    """
    try:
        return subprocess.check_output(
            [os.path.join(tool_dir, "ld.lld"), "--version"],
            encoding="utf-8",
            env=llvm19_toolchain.llvm19_env("xpu"),
        ).strip()
    except Exception as e:  # noqa: BLE001 - cache-key ingredient: a missing ld.lld must degrade, not raise
        return f"lld:absent({type(e).__name__})"


# @dataclass   create __dataclass_fields__ specical attribute
# frozen=True can't modift entry's attribute once it have been created
# [raise FrozenInstanceError]
@dataclass(frozen=True)
class XPUOptions:
    arch: int = 3
    assert arch in [2, 3, 4, 5], "Invalid XPU ARCH"
    grid: tuple = (1, 1, 1)  # grid launch params
    debug: bool = False
    instrumentation_mode: str = ""
    sanitize_overflow: bool = True

    cluster_num: int = 12 if arch == 3 else 8
    core_num: int = 64
    buffer_size_limit: int = int(os.environ.get("TRITONXPU_BUFFER_SIZE", 512))
    groups_per_cluster: int = int(os.environ.get("TRITONXPU_GROUPS_PER_CLUSTER", 1))
    unroll_num: int = int(os.environ.get("TRITONXPU_UNROLL_NUM", 2))
    vrf_budget: int = int(os.environ.get("TRITONXPU_VRF_BUDGET", 24))
    # On by default: measured no worse than the unroll-num knob anywhere and
    # clearly better on three geometries. layernorm bufSz=512 927.6->833.5us,
    # bufSz=2048 460.0->434.8us, softmax bufSz=512 2588.9->2385.4us.
    # Set TRITONXPU_BUDGET_TILING=0 to fall back to unroll_num.
    budget_tiling: bool = bool(int(os.environ.get("TRITONXPU_BUDGET_TILING", 1)))
    # Enabled on demand after a TLE stack overflow; the initial compile is unchanged.
    tle_relieve_pressure: bool = bool(int(os.environ.get("TRITONXPU_TLE_RELIEVE_PRESSURE", 0)))
    # The two pinned unroll constants in UnrollControl (bool-store-vectorize=4,
    # core-deal-multi-rows=1) short-circuit the tile decision before unroll_num
    # is read, so until this knob existed neither could be shown to be wrong:
    # boolfused emits the same binary at unroll_num 1/4/16. <0 keeps them (the
    # 0 (the default since step 3.5) ignores the two pins and lets the pressure
    # model decide, <0 restores them as an escape hatch, >0 overrides the
    # constant so its time curve can be swept. The model picks iterNum=1 at all
    # four pinned sites, where the pins pick 4/4/4/2
    # (xpu-tile-redesign/reference/findings.md 1.30), and on device the pinned
    # factor is ~19% slower at both sites with identical results (1.47).
    pin_unroll_num: int = int(os.environ.get("TRITONXPU_PIN_UNROLL_NUM", 0))
    # Off by default. A tree with no vector value of its own bypasses the whole
    # criteria chain and takes the legacy factor, which is derived from
    # unroll_num (a *width* knob, moving opposite to iterNum). Turning this on
    # hands those sites to the chain. Three sites in the probe suite reach that
    # entry and all three move 64 -> 1; the measured optimum on bool is the
    # interior point iterNum=16, so this is a measurement knob for step 2.6,
    # not a fix (xpu-tile-redesign/reference/findings.md 1.51).
    open_decision_entry: bool = bool(int(os.environ.get("TRITONXPU_OPEN_DECISION_ENTRY", 0)))
    # On by default since step 3.2n. Off, Vectorize's set comes from the
    # all-or-nothing closure walk; on, it comes from the Vector-Flow partition's
    # terminal state, and a segment that the criteria chain accepts is cut out of
    # the tree with its boundary materialised as unpack. The two things bought:
    # the four probes that change are all faster on device with max_abs=0
    # (truncint -21.8%, truncstore -29.8%, locont -29.2%, twoseg -3.1%,
    # findings 1.68), and the 68-operator FlagGems sweep is result-identical to
    # the closure path (5639 passed / 3 pre-existing failed on both, findings
    # 1.72). Set TRITONXPU_PER_OP_DECISION=0 to fall back to the closure walk.
    per_op_decision: bool = bool(int(os.environ.get("TRITONXPU_PER_OP_DECISION", 1)))
    is_use_mask_zero: bool = int(os.environ.get("TRITONXPU_IS_USE_MASK_ZERO", 0))
    extern_libs: dict = None
    is_sdnn: bool = False
    backend_name: str = "xpu"
    cluster_dims: tuple = (1, 1, 1)  # TODO: find mapping relationship
    max_num_imprecise_acc_default: int = 0

    isOpenCmpNan: bool = False
    isCloseOffsetAnalysis: bool = False
    isCloseCoreTiling: bool = False
    isCloseUnrollControl: bool = False
    isCloseVectorization: bool = bool(int(os.environ.get("TRITONXPU_IS_CLOSE_VECTORIZATION", 0)))
    isCloseMemoryCache: bool = False
    isCloseClusterLoopGrid: bool = False
    isClusterOneCoreActOnly: bool = False
    isCLOSE_TTXPU_O_ATOMIC_SIM: bool = False
    isCloseDtypeConvert: bool = False
    isCloseInterleave: bool = False
    isCloseMemoryAsync: bool = bool(int(os.environ.get("TRITONXPU_IS_CLOSE_MEMORY_ASYNC", 0)))
    isCloseTLESmemVec: bool = bool(int(os.environ.get("TRITONXPU_IS_CLOSE_TLE_SM_VEC", 0)))
    isCloseTLEVec: bool = bool(int(os.environ.get("TRITONXPU_IS_CLOSE_TLE_VEC", 0)))
    isAutoCoreTiling: bool = bool(int(os.environ.get("TRITONXPU_AUTO_CORE_TILING", 0)))

    enable_fp_fusion: bool = False
    allow_fp8e4nv: bool = True
    allow_fp8e4b15: bool = False
    supported_fp8_dtypes: Tuple[str] = ()
    default_dot_input_precision: str = "tf32"
    allowed_dot_input_precisions: Tuple[str] = ("ieee", "tf32")

    exp_range: str = ":0"

    num_warps: int = (-1)  # TODO: invalid value, just to keep num_warps function signature
    num_ctas: int = -1  # TODO: invalid value, just to keep num_ctas function signature
    num_stages: int = 1

    # use_int4_w4a8
    use_int4_w4a8: bool = False
    load_tile_size: int = 131072

    def __post_init__(self):
        default_libdir = Path(__file__).parent / f"xpu{self.arch}"
        extern_libs = {} if self.extern_libs is None else dict(self.extern_libs)
        if not extern_libs.get("libdevice", None):
            extern_libs["libdevice"] = os.getenv(
                "TRITON_LIBDEVICE_PATH",
                str(default_libdir / "lib" / f"libdevice-xpu{self.arch}.bc"),
            )
            if not os.path.exists(extern_libs["libdevice"]):
                warnings.warn(f'libdevice not found: {extern_libs["libdevice"]}', UserWarning)
                del extern_libs["libdevice"]

        if not extern_libs.get("libdevice-sdnn", None):
            extern_libs["libdevice-sdnn"] = os.getenv(
                "TRITON_LIBDEVICE_PATH",
                str(default_libdir / "lib" / f"libdevice-xpu{self.arch}s.bc"),
            )
            if not os.path.exists(extern_libs["libdevice-sdnn"]):
                warnings.warn(f'libdevice not found: {extern_libs["libdevice-sdnn"]}', UserWarning)
                del extern_libs["libdevice-sdnn"]

        object.__setattr__(self, "extern_libs", tuple(extern_libs.items()))

        invalid_params = []
        if self.num_warps != -1:
            invalid_params.append(f"num_warps={self.num_warps}")
        if self.num_ctas != -1:
            invalid_params.append(f"num_ctas={self.num_ctas}")
        if self.num_stages != -1 and (not self.is_sdnn):
            invalid_params.append(f"num_stages={self.num_stages}")
        if len(invalid_params) > 0:
            logging.debug(f"Invalid {', '.join(invalid_params)} in xpu arch")

    def hash(self):
        hash_dict = dict(self.__dict__)
        hash_dict["extern_libs"] = tuple((k, file_hash(v)) for k, v in sorted(hash_dict["extern_libs"]))
        key = "_".join([f"{name}-{val}" for name, val in sorted(hash_dict.items())])
        return hashlib.sha256(key.encode("utf-8")).hexdigest()


class XPUBackend(BaseBackend):

    def __init__(self, target: GPUTarget) -> None:
        super().__init__(target)
        assert isinstance(target.arch, int)
        self.binary_ext = "xpubin"
        self.buffer_len = 128

    @staticmethod
    def supports_target(target: GPUTarget):
        # Accept the target string configured via TRITON_XPU_TARGET_BACKEND
        # (see XPUDriver.get_current_target), plus "cuda"/"xpu" aliases so
        # CompiledKernel metadata cached under any prior setting still
        # deserializes and routes here. Only one backend is ever discovered in
        # this stack (FLAGTREE_BACKEND restriction), so the wide alias set
        # cannot cause cross-backend routing conflicts.
        configured = os.environ.get("TRITON_XPU_TARGET_BACKEND", "cuda")
        return target.backend in (configured, "cuda", "xpu")

    @staticmethod
    def path_to_xpu_compile_tool(opt):
        # Check env path for clang
        if "TRITON_XPU_CLANG_PATH" in os.environ:
            clang_path = os.getenv("TRITON_XPU_CLANG_PATH")
            return clang_path
        return os.path.join(Path(__file__).parent, f"xpu{opt.arch}", "bin")

    @staticmethod
    def make_ttir(mod, metadata, opt):
        pm = ir.pass_manager(mod.context)
        pm.enable_debug()
        passes.common.add_inliner(pm)
        passes.ttir.add_rewrite_tensor_pointer(pm)
        passes.ttir.add_combine(pm)
        passes.common.add_canonicalizer(pm)
        passes.ttir.add_reorder_broadcast(pm)
        passes.common.add_cse(pm)
        passes.common.add_licm(pm)
        passes.common.add_symbol_dce(pm)
        pm.run(mod, 'make_ttir')
        return mod

    @staticmethod
    def is_tle_kernel(mod):
        return xpu.is_tle_kernel(mod)

    @staticmethod
    def make_ttxir(mod, metadata, opt):
        metadata["xpu_arch"] = opt.arch
        metadata["is_tle"] = XPUBackend.is_tle_kernel(mod)
        metadata["is_sdnn"] = opt.is_sdnn or xpu.is_sdnn_kernel(mod)
        # tensor_args drives the launch-time scaled-buffer quantization (findmax +
        # cast_te) in device/xpu3/launch.cpp, which replaces the input pointer with
        # a fresh quantized buffer and APPENDS a scale param. That is only correct
        # when the kernel was actually compiled to consume those scale params, i.e.
        # when TritonConvertTypePass ran in TO_F16 (XMLIR_MATMUL_FAST_MODE=1) or
        # w4a8 mode and appended the matching max_a_ptr/max_b_ptr arguments (see
        # ConvertType.cpp convertBF16ToF16/convertI8ToI4). For the default TO_F32
        # path (plain bf16 dot) no scale param is added, so a non-empty tensor_args
        # makes launch.cpp append a spurious scale param, shifting gridX/Y/Z and
        # handing the kernel a wild quantized buffer base -> illegal memory access.
        # Gate on the same condition that selects the quantizing mode.
        _needs_scale_args = metadata["use_int4_w4a8"] or int(os.environ.get("XMLIR_MATMUL_FAST_MODE", 0)) == 1
        metadata["tensor_args"] = xpu.get_tensor_args(mod, []) if (metadata["is_sdnn"] is True
                                                                   and _needs_scale_args) else []
        metadata["shared"] = (-1)  # TODO: invalid value, just to keep CompiledKernel _init_handles() success

        max_buffer_size = metadata["buffer_size_limit"]
        tle_smem_vec = not metadata["isCloseTLESmemVec"]
        tle_vec = not metadata["isCloseTLEVec"]
        elem_bytes = int(os.environ.get("TRITONXPU_ELEMBYTES", 0))
        groups_per_cluster = metadata["groups_per_cluster"]
        unroll_num = metadata["unroll_num"]
        vrf_budget = metadata.get("vrf_budget", 24)
        XPUBackend.buffer_len = xpu.get_buffer_len(mod, max_buffer_size, elem_bytes)
        # print(f"XPUBackend.buffer_len = {XPUBackend.buffer_len}")
        core_num = metadata["core_num"]
        # F/O Prefix For Function/Optimization Macro
        is_use_mask_zero = metadata["is_use_mask_zero"]
        if is_use_mask_zero:
            warnings.warn(
                'XRE Version Must Be More than 5.0.21.37 (After 2025.07.22). And echo 1 > /proc/kunlun/dev4/dma_excp_mask',
                UserWarning)
        TTXPU_F_INTERLEAVE = 0 if metadata["grid"] != (12, 1, 1) else int(os.environ.get("TRITONXPU_INTERLEAVE", 1))
        TTXPU_F_OHTER_VALUE_SIM = int(os.environ.get("TRITONXPU_OTHER_SIM", 0))
        TTXPU_F_DTYPE_CONVERT = 0 if metadata["isCloseDtypeConvert"] else int(
            os.environ.get("TRITONXPU_DTYPE_CONVERT", 1))
        TTXPU_O_ATOMIC_SIM = 0 if metadata["isCLOSE_TTXPU_O_ATOMIC_SIM"] else int(
            os.environ.get("TRITONXPU_ATOMIC_SIM", 1))
        TTXPU_O_CLOSE_OPT = int(os.environ.get("TRITONXPU_CLOSE_OPTIMIZE", 0))
        TTSDNN_F_MATMUL_FAST_MODE = int(os.environ.get("XMLIR_MATMUL_FAST_MODE", 0))
        TTSDNN_F_DMA_MODE = int(os.environ.get("XMLIR_DMA_FAST_MODE", 0))
        TTSDNN_F_KILL_EW_FILL_MODE = int(os.environ.get("XMLIR_KILL_EW_FILL_MODE", 0))

        if opt.isClusterOneCoreActOnly and TTXPU_F_OHTER_VALUE_SIM:
            assert 0, "isClusterOneCoreActOnly and TRITONXPU_OTHER_SIM can't act simultaneously"

        pm = ir.pass_manager(mod.context)
        pm.enable_debug()
        passes.ttir.add_loop_aware_cse(pm)
        if metadata["is_tle"]:
            # TLE path: independent IR pipeline, skip all normcopy optimization passes.
            # TritonToTritonXPU adds ClusterLayout encoding needed for type conversion.
            # TLE-specific ops (tle_copy_g2l/l2g, tle_local_ptr) + triton.load/store
            # will be lowered directly in TritonXPUToLLVM.
            xpu.passes.ttxpuir.add_convert_triton_to_tritonxpu_pass(pm, opt.arch, XPUBackend.buffer_len, core_num,
                                                                    isTLE=True)
            if not metadata["isCloseCoreTiling"]:
                # Core-tile a 2D row-reduce (axis=1) across the core grid. Runs
                # before tle_legalize (reduce is still tt.reduce). RowTiled when
                # M % core_num == 0 (whole rows per core, local reduce); LargeN
                # when 2 <= M | core_num (row split across cores, cross-core
                # reduce). Safe no-op otherwise.
                xpu.passes.ttxpuir.add_tritonxpu_tle_core_tiling_pass(pm, 0, XPUBackend.buffer_len,
                                                                      core_num)  # dumpFlag=0
            # Software-pipeline the GM->LM/SM fills. Must run BEFORE tle_legalize,
            # which rewrites the loop body the rotation is built from. The budget
            # checks inside run even when the rewrite is disabled, so this call is
            # unconditional; only the rewrite is gated on TRITONXPU_TLE_PIPELINE.
            # The LM budget is the stricter of the XTDK stack-size limit (which
            # TRITON_TUNE_BUFFER_LM_SIZE moves, default 8000) and the 8KB physical
            # LM, since exceeding either one fails the build. num_stages stays 1
            # (= off) at module level: a loop opts in with tl.range(num_stages=2),
            # which lands as tt.num_stages on the scf.for.
            lm_bytes_limit = min(int(os.environ.get("TRITON_TUNE_BUFFER_LM_SIZE", 8000)), 8192)
            xpu.passes.ttxpuir.add_tritonxpu_tle_pipeline_pass(pm, bool(int(os.environ.get("TRITONXPU_TLE_PIPELINE",
                                                                                           0))), 1, 1, lm_bytes_limit)
            xpu.passes.ttxpuir.add_tritonxpu_tle_legalize_pass(pm)  # dumpFlag=0
            # Assign static SM byte offsets to cluster-shared (scope=smem) TLE
            # buffers before the LLVM lowering reads xpu.sm_offset in the
            # alloc / copy_g2l / local_ptr SM branches. No-op when there are no
            # smem buffers. Must run after tle_legalize (the smem local_alloc
            # ops still exist) and before make_llir.
            xpu.passes.ttxpuir.add_tritonxpu_tle_sm_alloc_pass(pm)
            # Promote bf16 compute to f32 and leave the bf16 <-> f32 conversion on
            # the LM boundary, where XPUTLETriton{Load,Store}OpConversion fuses it
            # (XPU3 has no bf16 ALU and no bf16 convert instruction). Must run
            # before normalize/vectorize so the chain is already f32 by then.
            xpu.passes.ttxpuir.add_tritonxpu_tle_dtype_convert_pass(pm)
            if not metadata["isCloseVectorization"]:
                compareFusion = int(os.environ.get("TRITONXPU_COMPARE_FUSION", 0))
                # Pre-vectorization normalization, split out of vectorize's own
                # prologue (step 1.5). Must stay immediately before it.
                xpu.passes.ttxpuir.add_tritonxpu_normalize_pass(
                    pm, 0, compareFusion) if not TTXPU_O_CLOSE_OPT else None  # dumpFlag=0
                xpu.passes.ttxpuir.add_tritonxpu_vectorize_pass(
                    pm, 0, compareFusion, opt.per_op_decision, tle_smem_vec,
                    tle_vec) if not TTXPU_O_CLOSE_OPT else None  # dumpFlag=0
                # Peak live-vector pressure is only defined after vectorization.
                xpu.passes.ttxpuir.add_tritonxpu_tile_analysis_pass(pm, vrf_budget)
            if not metadata["isCloseUnrollControl"] and metadata["tle_relieve_pressure"]:
                xpu.passes.ttxpuir.add_tritonxpu_unroll_control_pass(
                    pm, XPUBackend.buffer_len, core_num, is_use_mask_zero, unroll_num, vrf_budget,
                    metadata["budget_tiling"], metadata["pin_unroll_num"],
                    metadata["open_decision_entry"]) if not TTXPU_O_CLOSE_OPT else None
            if not metadata["isCloseClusterLoopGrid"]:
                # LoopGrid: wraps kernel body in a cluster loop and appends gridX/Y/Z
                # arguments that xpuLaunchKernel passes at the end of kernel_params.
                xpu.passes.ttxpuir.add_tritonxpu_cf_to_scf_pass(pm)
                xpu.passes.ttxpuir.add_tritonxpu_loop_grid_pass(pm)
            passes.common.add_canonicalizer(pm)
            passes.common.add_cse(pm)
            passes.common.add_symbol_dce(pm)
        elif metadata["is_sdnn"]:
            xpu.passes.ttsdnnir.add_tritonsdnn_strip_all_ops_pass(pm, opt.arch)
            if opt.arch < 4:
                if metadata["use_int4_w4a8"]:
                    xpu.passes.ttsdnnir.add_triton_convert_type_pass(pm, opt.arch, 2)
                else:
                    xpu.passes.ttsdnnir.add_triton_convert_type_pass(pm, opt.arch, TTSDNN_F_MATMUL_FAST_MODE)
            xpu.passes.ttsdnnir.add_convert_triton_to_tritonsdnn_pass(pm, opt.arch, opt.load_tile_size)
            passes.ttir.add_loop_aware_cse(pm)
            xpu.passes.ttsdnnir.add_linalg_to_tritonsdnn_pass(pm, opt.arch)
            passes.ttir.add_loop_aware_cse(pm)
            xpu.passes.ttsdnnir.add_tritonsdnn_legalize_pass(pm, opt.arch)
            xpu.passes.ttsdnnir.add_tritonsdnn_merge_extern_ew_pass(pm)
            xpu.passes.ttsdnnir.add_tritonsdnn_combine_before_pass(pm, opt.arch)
            xpu.passes.ttsdnnir.add_tritonsdnn_resolve_layout_conflict_pass(pm)
            xpu.passes.ttsdnnir.add_tritonsdnn_hoist_ds_pass(pm)
            xpu.passes.ttsdnnir.add_tritonsdnn_transpose_mma_pass(pm)
            xpu.passes.ttsdnnir.add_tritonsdnn_transpose_ew_pass(pm)
            xpu.passes.ttsdnnir.add_tritonsdnn_combine_before_pass(pm, opt.arch)
            # Wrap orphan linalg.generic into sdnn.core_specialize AFTER the
            # combine-before fusion, so elementwise generics are first fused into a
            # minimal number of ops (avoids fragmenting into many tiny
            # core_specialize regions). Must run before bufferize (which lowers the
            # wrapped linalg to loops).
            # FlagTree: the prebuilt v0.3.6.7.1 SDNN objects do not export this
            # binding yet (it ships with the sync source d116bdf4 rebuild).
            if hasattr(xpu.passes.ttsdnnir, "add_tritonsdnn_wrap_fallback_pass"):
                xpu.passes.ttsdnnir.add_tritonsdnn_wrap_fallback_pass(pm)
            xpu.passes.ttsdnnir.add_tritonsdnn_eliminate_mma_acc_zero_pass(pm, opt.arch)
            if opt.arch == 4:
                xpu.passes.ttsdnnir.add_tritonsdnn_fuse_mma_vector_bias_pass(pm, opt.arch)
            xpu.passes.ttsdnnir.add_tritonsdnn_optimize_rc_layout_pass(pm, opt.arch)
            if opt.arch == 4:
                xpu.passes.ttsdnnir.add_tritonsdnn_ewlite_scheduling_pass(pm, opt.arch)
                if TTSDNN_F_DMA_MODE:
                    xpu.passes.ttsdnnir.add_tritonsdnn_remove_ds_op_pass(pm, opt.arch)
            xpu.passes.ttsdnnir.add_tritonsdnn_dsa_copy_pass(pm)
            xpu.passes.ttsdnnir.add_tritonsdnn_bufferize_pass(pm, opt.arch)
            xpu.passes.ttsdnnir.add_tritonsdnn_mx_scale_layout_pass(pm)
            xpu.passes.ttsdnnir.add_tritonsdnn_fuse_relu_activation_pass(pm, opt.arch)
            if opt.exp_range != ":0":
                res = parse_floating_range_string(opt.exp_range)
                xpu.passes.ttsdnnir.add_tritonsdnn_ew_act_table_pass(pm, res)
            else:
                xpu.passes.ttsdnnir.add_tritonsdnn_ew_act_table_pass(pm, None)
            xpu.passes.ttsdnnir.add_tritonsdnn_loop_grid_pass(pm)
            xpu.passes.ttsdnnir.add_tritonsdnn_hoist_loop_invariant_dma_pass(pm, opt.arch)
            # FlagTree: defer_event_coloring (arch5-only wiring) needs the
            # v0.3.6.8.x SDNN objects; the 7.1 binding has no such kwarg.
            if opt.arch == 5:
                xpu.passes.ttsdnnir.add_tritonsdnn_pipeline_pass(pm, opt.arch, defer_event_coloring=True)
            else:
                xpu.passes.ttsdnnir.add_tritonsdnn_pipeline_pass(pm, opt.arch)
            if opt.arch == 4 and TTSDNN_F_KILL_EW_FILL_MODE:
                xpu.passes.ttsdnnir.add_tritonsdnn_kloop_acc_elimination_pass(pm)
            xpu.passes.ttsdnnir.add_tritonsdnn_multi_buffer_pass(pm, opt.arch, opt.num_stages)
            if opt.arch == 5 and not TTSDNN_F_SINGLE_CODE_MODE:
                xpu.passes.ttsdnnir.add_tritonsdnn_event_coloring_pass(pm, 0)
            xpu.passes.ttsdnnir.add_tritonsdnn_lower_rc_subview_pass(pm, opt.arch)
            passes.common.add_symbol_dce(pm)
            passes.common.add_canonicalizer(pm)
            passes.common.add_cse(pm)
        else:
            xpu.passes.ttxpuir.add_tritonxpu_legalize_extern_ew_pass(pm)
            xpu.passes.ttxpuir.add_convert_triton_to_tritonxpu_pass(pm, opt.arch, XPUBackend.buffer_len, core_num)
            xpu.passes.ttxpuir.add_tritonxpu_print_pass(pm)
            xpu.passes.ttxpuir.add_tritonxpu_gm2lm_pass(pm, opt.arch, TTXPU_O_ATOMIC_SIM, opt.isClusterOneCoreActOnly,
                                                        is_use_mask_zero)
            if not metadata["isCloseMemoryCache"]:
                xpu.passes.ttxpuir.add_tritonxpu_memory_cache_pass(pm, XPUBackend.buffer_len,
                                                                   core_num) if not TTXPU_O_CLOSE_OPT else None
            passes.common.add_canonicalizer(pm)
            if TTXPU_F_DTYPE_CONVERT:
                xpu.passes.ttxpuir.add_tritonxpu_dtype_convert_pass(pm, opt.arch)
            xpu.passes.ttxpuir.add_tritonxpu_vectorizability_analysis_pass(pm, True, tle_smem_vec, tle_vec, True)
            # M2/M3's decision half, taken here because the terminal states are
            # E-independent at this position. Tiers 1 and 2 need a fixed per-core
            # geometry, so they stay in unroll_control below and are recorded as
            # deferred. Writes only triton_xpu.tile_decision -- read back and
            # erased by add_tritonxpu_tile_analysis_pass -- and only under
            # TRITONXPU_TILE_DECIDE=1.
            xpu.passes.ttxpuir.add_tritonxpu_tile_decide_pass(pm, True)
            if not metadata["isCloseCoreTiling"]:
                xpu.passes.ttxpuir.add_tritonxpu_core_tiling_pass(
                    pm, 0, XPUBackend.buffer_len, core_num, groups_per_cluster,
                    metadata["isAutoCoreTiling"]) if not TTXPU_O_CLOSE_OPT else None  # dumpFlag=0
            # xpu.passes.ttxpuir.add_tritonxpu_lm_to_sm_pass(pm)
            passes.common.add_cse(pm)
            if not metadata["isCloseOffsetAnalysis"]:
                xpu.passes.ttxpuir.add_tritonxpu_scalar_analysis_pass(pm, False) if not TTXPU_O_CLOSE_OPT else None
                xpu.passes.ttxpuir.add_tritonxpu_offset_state_pass(
                    pm, 0, XPUBackend.buffer_len, is_use_mask_zero) if not TTXPU_O_CLOSE_OPT else None  # dumpFlag=0
                xpu.passes.ttxpuir.add_tritonxpu_scalar_analysis_pass(pm, True) if not TTXPU_O_CLOSE_OPT else None
            passes.common.add_canonicalizer(pm)
            xpu.passes.ttxpuir.add_tritonxpu_legalize_pass(pm, XPUBackend.buffer_len, core_num, groups_per_cluster,
                                                           is_use_mask_zero)
            passes.common.add_canonicalizer(pm)
            if not TTXPU_F_OHTER_VALUE_SIM:
                xpu.passes.ttxpuir.add_tritonxpu_mask_pass(pm, opt.isClusterOneCoreActOnly, is_use_mask_zero)
            passes.common.add_canonicalizer(pm)
            passes.common.add_cse(pm)
            passes.common.add_licm(pm)
            passes.common.add_symbol_dce(pm)
            if not metadata["isCloseInterleave"] and TTXPU_F_INTERLEAVE:
                xpu.passes.ttxpuir.add_tritonxpu_interleave_pass(pm) if not TTXPU_O_CLOSE_OPT else None
            passes.common.add_canonicalizer(pm)
            if not metadata["isCloseVectorization"]:
                compareFusion = int(os.environ.get("TRITONXPU_COMPARE_FUSION", 0))
                xpu.passes.ttxpuir.add_tritonxpu_normalize_pass(
                    pm, 0, compareFusion) if not TTXPU_O_CLOSE_OPT else None  # dumpFlag=0
                xpu.passes.ttxpuir.add_tritonxpu_vectorizability_analysis_pass(pm, True, tle_smem_vec, tle_vec, False)
                xpu.passes.ttxpuir.add_tritonxpu_vectorize_pass(
                    pm, 0, compareFusion, opt.per_op_decision, tle_smem_vec,
                    tle_vec) if not TTXPU_O_CLOSE_OPT else None  # dumpFlag=0
            passes.common.add_canonicalizer(pm)
            xpu.passes.ttxpuir.add_tritonxpu_alloca_pass(pm, XPUBackend.buffer_len, core_num)
            if not metadata["isCloseMemoryAsync"]:
                xpu.passes.ttxpuir.add_tritonxpu_async_load_schedule_pass(
                    pm, 0) if not TTXPU_O_CLOSE_OPT else None  # dumpFlag=0
            if not metadata["isCloseUnrollControl"]:
                xpu.passes.ttxpuir.add_tritonxpu_tile_analysis_pass(pm, vrf_budget)
                xpu.passes.ttxpuir.add_tritonxpu_unroll_control_pass(
                    pm, XPUBackend.buffer_len, core_num, is_use_mask_zero, unroll_num, vrf_budget,
                    metadata["budget_tiling"], metadata["pin_unroll_num"],
                    metadata["open_decision_entry"]) if not TTXPU_O_CLOSE_OPT else None
            xpu.passes.ttxpuir.add_tritonxpu_store_control_pass(pm) if not TTXPU_O_CLOSE_OPT else None
            if not TTXPU_F_OHTER_VALUE_SIM:
                xpu.passes.ttxpuir.add_tritonxpu_other_sim_pass(pm, XPUBackend.buffer_len, core_num)
            xpu.passes.ttxpuir.add_tritonxpu_memory_inplace_pass(pm, XPUBackend.buffer_len,
                                                                 core_num) if not TTXPU_O_CLOSE_OPT else None
            if not metadata["isCloseClusterLoopGrid"]:
                xpu.passes.ttxpuir.add_tritonxpu_cf_to_scf_pass(pm)
                xpu.passes.ttxpuir.add_tritonxpu_loop_grid_pass(pm)
                if int(os.environ.get("TRITONXPU_LOOP_INVARIANT_STAGING", 0)):
                    xpu.passes.ttxpuir.add_tritonxpu_loop_invariant_staging_pass(pm)
            passes.common.add_cse(pm)
            passes.common.add_licm(pm)
            passes.common.add_symbol_dce(pm)

        try:
            pm.run(mod, 'make_ttxir')
        except Exception as e:
            from triton import OutOfResources
            raise OutOfResources(0, 0, f"uni_sram {e}")

        return mod

    @staticmethod
    def make_ewtable(mod, metadata, opt):

        def get_range_table(fn, dtype, mode, size, min, interval):
            import numpy as np
            import torch

            def np_gelu(x):
                return torch.nn.functional.gelu(torch.tensor(x))

            fn = {
                "math.exp": np.exp,
                "math.exp2": np.exp2,
                "math.log2": np.log2,
                "tt.gelu": np_gelu,
            }[fn]
            dtype = {0: np.float32, 1: np.float16}[dtype]
            table = [0.0] * size * 2
            val_vec = []
            if mode == 0:  # KB
                for i in range(size + 1):
                    val_vec += [fn(min + i * interval)]
                for i in range(size):
                    table[i] = (val_vec[i + 1] - val_vec[i]) / interval  # k
                    table[i + size] = val_vec[i] - table[i] * (min + i * interval)  # b
            else:  # Interpolation
                for i in range(size + 1):
                    val_vec += [fn(min + (2 * i) * interval)]
                    val_vec += [fn(min + (2 * i + 1) * interval)]
                if fn == "math.exp" or fn == "math.exp2":
                    iter_table = val_vec
                    for iter in range(10):
                        for idx in range(size * 2 - 1, 0, -1):
                            iter_table[idx] = ((val_vec[idx + 1] - val_vec[idx - 1]) / interval) - \
                                            ((iter_table[idx + 1] + iter_table[idx - 1]) / 2.0)
                    val_vec = iter_table

                for i in range(size):
                    table[i] = val_vec[2 * i]
                    table[i + size] = val_vec[2 * i + 1]

            # for i in range(len(table)):
            #     print(f"cpu_table[{i}] = {table[i]}")

            # print(f'table = {table}')
            return np.array(table, dtype=dtype).tobytes()

        metadata["ewtable"] = ""
        lut_info = xpu.get_lut_info(mod)
        if not lut_info:
            return mod

        lut_info_str = ""
        for fn, info in lut_info.items():
            lut_info_str += f"{fn}-{info.dtype}-{info.mode}-{info.size}-{info.min},"
        key = hashlib.sha256(lut_info_str.encode()).hexdigest()
        name = "ewtable.dat"
        cache = get_cache_manager(key)
        cache_path = cache.get_file(name)
        if cache_path is None:
            k, b = b'', b''
            for fn, info in lut_info.items():
                table = get_range_table(fn, info.dtype, info.mode, info.size, info.min, info.interval)
                mid = len(table) // 2
                k += table[:mid]
                b += table[mid:]
            cache_path = cache.put(k + b, name, binary=True)

        metadata["ewtable"] = cache_path
        return mod

    @staticmethod
    def make_llir(mod, metadata, opt):
        XPUBackend.make_ewtable(mod, metadata, opt)
        # The emitters below (createIntrinsicCallByName / createCallIntrinsic)
        # need this process's table before they run, or the fact is not born on
        # the instruction -- landing B and B' can only patch up.  See
        # llvm19_toolchain.install_intrinsic_attr_table.
        llvm19_toolchain.install_intrinsic_attr_table()
        # Gate tensor_args on the same condition that selects the quantizing
        # ConvertType mode (see make_ttxir). Only TO_F16 (XMLIR_MATMUL_FAST_MODE=1)
        # or w4a8 kernels actually consume the launch-appended scale param; for
        # the default TO_F32 bf16 dot a non-empty tensor_args makes launch.cpp
        # append a spurious scale param and hand the kernel a wild quantized
        # buffer base -> illegal memory access.
        _needs_scale_args = metadata["use_int4_w4a8"] or int(os.environ.get("XMLIR_MATMUL_FAST_MODE", 0)) == 1
        metadata["tensor_args"] = xpu.get_tensor_args(mod, metadata["tensor_args"]) if (metadata["is_sdnn"] is True
                                                                                        and _needs_scale_args) else []

        # TritonXPU -> LLVM-IR (MLIR)
        pm = ir.pass_manager(mod.context)
        pm.enable_debug()
        # xpu.passes.ttxpuir.add_decompose_unsupported_conversions(pm, opt.arch)
        passes.convert.add_scf_to_cf(pm)  # cf->llvm exist  choose scf->cf->llvm
        # passes.convert.add_index_to_llvmir(pm) // TODO[dyq]: necessary?

        # Record this kernel's tle.raw payloads before the conversion replaces the
        # raw ops (`triton_sdnn.raw` / `triton_xpu.raw`) with calls: the payload is
        # compiled and merged into the kernel module at the LLVM 19 stage
        # (make_elf), so it never goes through the public LLVM 22 MLIR.
        #
        # Imported lazily: importing any `tle.raw` submodule runs the package
        # `__init__`, which pulls in `triton.language` -- and the backends are
        # themselves imported from `triton.runtime.jit`, so a module-level import
        # cycles.
        from triton.experimental.tle.raw.merge import record_raw_extern_libs, record_raw_payloads
        record_raw_payloads(mod, metadata)

        if metadata["is_sdnn"]:
            xpu.passes.ttsdnnir.add_convert_tritonsdnn_to_llvm_pass(pm, opt.arch)
        else:
            xpu.passes.ttxpuir.add_allocate_xpu_shared_memory(pm)
            xpu.passes.ttxpuir.add_convert_tritonxpu_to_llvm_pass(pm, opt.arch, XPUBackend.buffer_len,
                                                                  metadata["is_use_mask_zero"])
        passes.common.add_canonicalizer(pm)
        passes.common.add_cse(pm)

        # passes.convert.add_cf_to_llvmir(pm)
        # passes.convert.add_arith_to_llvmir(pm)
        # passes.common.add_canonicalizer(pm)
        # passes.common.add_cse(pm)
        passes.common.add_symbol_dce(pm)
        if os.environ.get("TRITON_DISABLE_LINE_INFO", "0") == "0":
            passes.llvmir.add_di_scope(pm)

        xpu.passes.llvmxpuir.insert_mfence_check(pm)

        pm.run(mod, 'make_llir')

        # LLVM-IR (MLIR) -> LLVM-IR (LLVM)
        llvm.init_targets()
        context = llvm.context()
        llvm_mod = llvm.to_module(mod, context)

        if opt.extern_libs:
            if metadata["is_sdnn"]:
                paths = [path for (name, path) in opt.extern_libs if "sdnn" in name and xpu.llvm.need_extern_lib(mod)]
            else:
                paths = [
                    path for (name, path) in opt.extern_libs if "sdnn" not in name and xpu.llvm.need_extern_lib(mod)
                ]
            assert (len(paths) <= 1), f"Expected 0/1 extern_lib path, but found {len(paths)}"
            llvm.link_extern_libs(llvm_mod, paths)
            # A payload merged in at the LLVM 19 stage resolves its device
            # library calls against the same library the kernel linked here.
            record_raw_extern_libs(metadata, paths, [path for (_name, path) in opt.extern_libs])

        # The XPU LLVM target triple/datalayout must be attached before
        # optimize_module. attach_datalayout sets the full
        # "xpu{arch}-baidu-none-gnu" triple on the module.
        #
        # Crucially, optimize_module must be called WITH the `arch` argument.
        # When arch is omitted it defaults to "", so optimize_module builds the
        # pass pipeline with a NULL TargetMachine (see python/src/llvm.cc:
        # `if (!arch.empty() ...) targetMachine = createTargetMachine(...)`).
        # Without the XPU TargetMachine, @llvm.sqrt.f64 / fp64 div are legalized
        # into external libm libcalls (`sqrt`), which then fail to link in
        # xpu{arch}-elfconv-triton (ld.lld: error: undefined symbol: sqrt).
        # libdevice-xpu{arch}s.bc is built with "target-features"="+bigcore,...",
        # so every helper linked in above (e.g. _ZN3xpu14findCacheIndexEPiii, which
        # every sdnn.get_cache_buffer lowers to) arrives tagged as a Bigcore
        # function while the kernel itself carries SDNN instructions. The XPU
        # backend rejects that mix with "SDNN and Bigcore kernel cannot be in same
        # module." -- and since the LLVM 22 upgrade the check also runs from
        # optimize_module's TargetMachine callbacks, i.e. before amend_func below
        # gets to stamp +cdnn{arch} on the whole module. Stamp the linked-in
        # definitions here so the module is uniformly SDNN going into O3; the
        # helpers are plain scalar code and get inlined away regardless.
        if metadata["is_sdnn"]:
            for func in llvm_mod.get_functions():
                if func.is_declaration():
                    continue
                func.remove_fn_attr("target-features")
                func.add_fn_target_feature(f"+cdnn{opt.arch}")

        # The XPU LLVM target triple must be attached before optimize_module is
        # invoked. optimize_module calls createTargetMachine which uses
        # lookupTarget(module->getTargetTriple()); without a triple set this
        # returns nullptr and segfaults at target->createTargetMachine(...).
        llvm_mod.set_target_triple(f"xpu{opt.arch}-baidu-none-gnu")
        llvm.attach_datalayout(llvm_mod, f"xpu{opt.arch}-baidu-none-gnu", f"xpu{opt.arch}", "")
        # Landing B': the same table, on the translated module.  It runs here --
        # after the device libraries are linked in and after the SDNN target-feature
        # fixup -- so every declaration the in-process O3 is about to see carries the
        # facts the table states.  Landing B (the MLIR pass) could not reach the XPU
        # dialect's declarations: `createIntrinsicCallByName` creates them inside
        # `to_module`, after that pass had already run.
        llvm19_toolchain.stamp_llvm_ir(llvm_mod)
        llvm19_toolchain.dump_pre_o3(llvm_mod, metadata)
        llvm.optimize_module(llvm_mod, llvm.OPTIMIZE_O3, f"xpu{opt.arch}")
        xpu.llvm.amend_func(llvm_mod, mod, context, opt.arch, metadata["is_sdnn"])

        del context
        return llvm_mod

    @staticmethod
    def make_elf(mod, metadata, opt):
        # Imported lazily: importing any `tle.raw` submodule runs the package
        # `__init__`, which pulls in `triton.language` -- and the backends are
        # themselves imported from `triton.runtime.jit`, so a module-level import
        # cycles.
        from triton.experimental.tle.raw.merge import merge_raw_payloads
        # Find kernel names (there should only be one)
        # We get the name at the last possible step to accomodate `triton.compile`
        # on user-provided LLVM
        metadata["name"] = xpu.llvm.get_kernel_name(mod)
        # print(f'metadata[name] = {metadata["name"]}')

        # Must go through the LLVM 22 -> 19 text-downgrade path; do NOT use the
        # in-process `xpu.llvm.translate_to_asm`: it runs the public LLVM 22
        # verifier, which rejects this kernel's cc 124 (AMDGPU_Gfx_WholeWave there
        # requires the first argument to be i1). The text path below never runs
        # the LLVM 22 verifier or backend.
        triple = f"xpu{opt.arch}-baidu-none-gnu"
        proc = f"xpu{opt.arch}"
        ir_text = llvm19_toolchain.downgrade_ir(mod._ir_for_lowering(), "xpu")
        # Merge the payloads in after the downgrade (they are LLVM 19 text).
        try:
            ir_text = merge_raw_payloads(ir_text, metadata, "xpu", opt.arch)
        except RuntimeError as e:
            if "llvm-link missing" in str(e):
                raise RuntimeError(f"{e}\n  tle.raw payload kernels need the LLVM 19 `llvm-link`, which "
                                   f"this toolchain package does not ship.") from e
            raise
        llc_extra = llvm19_toolchain.llc_flags(_LLC_EXTRA, metadata)
        ret_asm = llvm19_toolchain.llir_to_asm(ir_text, "xpu", triple, proc, extra_args=llc_extra)
        # The llc only emits the XPU_KERNEL_PARAM_SIZE_<name> object when it
        # recognised the function as an XPU kernel entry, which depends on
        # the kernel calling convention having survived the LLVM 22 -> 19
        # text downgrade (amdgpu_gfx_whole_wave -> xpu_kernel).  If that
        # rewrite ever stops matching (e.g. a future LLVM prebuilt renames
        # cc 124), llc silently emits a plain function: the ELF builds fine
        # but the kernel has no entry point.  Assert the symbol here so a
        # broken rewrite fails loudly at compile time instead of at launch.
        if f"XPU_KERNEL_PARAM_SIZE_{metadata['name']}" not in ret_asm:
            raise RuntimeError(f"llc did not emit XPU_KERNEL_PARAM_SIZE_{metadata['name']}: "
                               f"the kernel entry convention was lost in the LLVM 22 -> 19 "
                               f"downgrade. The kernel calling-convention rewrite in "
                               f"llvm_ir_downgrade.py is likely stale for this LLVM "
                               f"prebuilt.")
        fn_cache_manager = get_cache_manager(metadata["hash"])
        from triton.knobs import compilation as _compilation_knobs
        if not _compilation_knobs.store_binary_only:
            fn_cache_manager.put(ret_asm, f"{metadata['name']}.asm")
        ret_elf = llvm19_toolchain.llir_to_object(ir_text, "xpu", triple, proc, extra_args=llc_extra)

        del mod
        return ret_elf

    @staticmethod
    def make_xpubin(mod, metadata, opt):
        with tempfile.TemporaryDirectory() as tmpdir:
            clang_path = XPUBackend.path_to_xpu_compile_tool(opt)
            elfconv = os.path.join(clang_path, f"xpu{opt.arch}-elfconv-triton")
            objfile = os.path.join(tmpdir, "kernel.o")
            binfile = os.path.join(tmpdir, "kernel.bin")
            with open(objfile, "wb") as f:
                f.write(mod)
            cmd = [elfconv, objfile, binfile, clang_path]
            # The elfconv script drives the per-arch LLVM 19 tools (clang,
            # ld.lld, llvm-readelf, llvm-objcopy, llvm-objdump, xpu-xxd) and the
            # whole subtree inherits this: __lib gives them the shared
            # libLLVM.so.19.1 / libclang-cpp.so.19.1.  Without it the tools fall
            # back to an ambient libLLVM (the XPU runtime ships an older one on
            # $XCUDA/lib64) and silently lower the private intrinsics wrong.
            out_bytes = run_cmd(cmd, env=llvm19_toolchain.llvm19_env("xpu"))
            printf_buf_offset_res = re.search(rb"0x[0-9a-fA-F]+", out_bytes)
            if printf_buf_offset_res:
                printf_buf_offset_hex = printf_buf_offset_res.group(0)
                printf_buf_offset_hex_str = printf_buf_offset_hex.decode("utf-8")
                printf_buf_offset = int(printf_buf_offset_hex_str, 16)
            else:
                printf_buf_offset = 0
            metadata["printf_buf_offset"] = printf_buf_offset
            with open(binfile, "rb") as f:
                return f.read()

    @staticmethod
    def is_elf_stack_size_oob(mod, margin: int = 0) -> bool:
        stack_size_oob = llvm.is_elf_stack_size_oob(mod, margin)
        return stack_size_oob

    def hash(self) -> str:
        """Returns a unique identifier for this backend"""
        # Toolchain identity, the same ingredient the other backends' hashes take
        # from their staged `ld.lld`: the linker moves the object bytes, so a
        # warm cache must not outlive a toolchain change.  The llc flags do have
        # to be in the key for the same reason as in the other backends -- a warm
        # cache would otherwise keep objects built with the old flags -- and
        # `codegen_key()` also carries the stamp's table tag, the table digests
        # and the LLVM 19 bin dir the tools come from.
        tool_dir = XPUBackend.path_to_xpu_compile_tool(self.target)
        flags = " ".join(_LLC_EXTRA) + " " + llvm19_toolchain.codegen_key("xpu")
        return f"{_xtdk_lld_version(tool_dir)}-{self.target}-{flags}"

    def parse_options(self, options: dict) -> object:
        args = {"arch": self.target.arch}
        args.update({k: options[k] for k in XPUOptions.__dataclass_fields__.keys() if k in options})
        return XPUOptions(**args)

    def add_stages(self, stages: dict, options: object, language=None) -> None:
        stages["ttir"] = lambda src, metadata: self.make_ttir(src, metadata, options)
        stages["ttxir"] = lambda src, metadata: self.make_ttxir(src, metadata, options)
        stages["llir"] = lambda src, metadata: self.make_llir(src, metadata, options)
        stages["elf"] = lambda src, metadata: self.make_elf(src, metadata, options)
        stages["xpubin"] = lambda src, metadata: self.make_xpubin(src, metadata, options)

    def pack_metadata(self, metadata):
        return (
            metadata.num_warps,
            metadata.num_ctas,
            metadata.shared,
            metadata.cluster_dims[0],
            metadata.cluster_dims[1],
            metadata.cluster_dims[2],
        )

    def get_codegen_implementation(self, options=None):
        # XPU does not have GPU-style WMMA/MMA shape constraints; report the
        # most permissive lower bound (1, 1, 1) so `tl.dot` accepts arbitrary
        # M/N/K. Mirrors triton_shared and the interpreter default.
        def float_mod(input, other, _semantic):
            from triton.language.extra.xpu.libdevice import fmod

            return fmod(input, other, _semantic=_semantic)

        codegen_fns = {
            "min_dot_size": lambda lhsType, rhsType: (1, 1, 1),
            "float_mod": float_mod,
        }
        return codegen_fns

    def load_dialects(self, context):
        xpu.load_dialects(context)

    def get_module_map(self) -> "dict":
        from triton.language.extra.xpu import libdevice

        return {"triton.language.extra.libdevice": libdevice}
