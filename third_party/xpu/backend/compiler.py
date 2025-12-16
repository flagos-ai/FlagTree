from triton.backends.compiler import BaseBackend, GPUTarget
from triton._C.libtriton import ir, passes, xpu, llvm
from triton.runtime.cache import get_cache_manager
import subprocess
import tempfile
import re
import warnings
import logging
import os

from dataclasses import dataclass
import functools
from typing import Any, Tuple, Optional
import hashlib
from pathlib import Path


def parse_floating_range_string(range_str: str) -> Optional[Tuple[Optional[float], Optional[float]]]:
    """
    解析形如 "Start:End" 的浮点数范围字符串。

    参数:
        range_str (str): 范围字符串，例如 "1.5:10.0" 或 ":-3.14" 或 "123.789"。

    返回:
        Optional[Tuple[Optional[float], Optional[float]]]:
        (start, end) 的元组，如果解析失败则返回 None。
        None 表示该部分缺失。
    """

    # 1. 分割字符串
    parts = range_str.split(':')

    if len(parts) > 2:
        print(f"错误: 范围字符串 '{range_str}' 格式不正确，包含太多冒号。")
        return None

    # 确保 parts 包含两个元素，例如 ["50.5"] -> ["50.5", ""]
    # 或 [":10"] -> ["", "10"]
    full_parts = parts + [""] * (2 - len(parts))

    start_str = full_parts[0]
    end_str = full_parts[1]

    start_val: Optional[float] = None
    end_val: Optional[float] = None

    # 2. 解析 Start 部分
    if start_str:
        try:
            start_val = float(start_str)
        except ValueError:
            print(f"错误: Start 部分 '{start_str}' 不是有效的浮点数。")
            return None

    # 3. 解析 End 部分
    if end_str:
        try:
            end_val = float(end_str)
        except ValueError:
            print(f"错误: End 部分 '{end_str}' 不是有效的浮点数。")
            return None

    # 4. 特殊处理：如果只有一个数字 (例如 "123.789")
    # 此时 parts 长度为 1，start_val 有值，end_val 为 None。
    # 我们将其视为 range(0.0, 123.789)，即 start 缺失，end 为该数字。
    if len(parts) == 1 and start_val is not None:
        end_val = start_val
        start_val = None

    return start_val, end_val


@functools.lru_cache(None)
def file_hash(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


# @dataclass   create __dataclass_fields__ specical attribute
# frozen=True can't modift entry's attribute once it have been created
# [raise FrozenInstanceError]
@dataclass(frozen=True)
class XPUOptions:
    arch: int = int(os.environ.get("TRITON_XPU_ARCH", "3"))
    assert arch in [2, 3, 4, 5], "Invalid XPU ARCH"
    grid: tuple = (1, 1, 1)  # grid launch params
    debug: bool = False

    cluster_num: int = 12 if arch == 3 else 8
    core_num: int = 64
    buffer_size_limit: int = int(os.environ.get("TRITONXPU_BUFFER_SIZE", 512))
    groups_per_cluster: int = int(os.environ.get("TRITONXPU_GROUPS_PER_CLUSTER", 1))
    unroll_num: int = int(os.environ.get("TRITONXPU_UNROLL_NUM", 2))
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
    isCloseVectorization: bool = False
    isCloseMemoryCache: bool = False
    isCloseClusterLoopGrid: bool = False
    isClusterOneCoreActOnly: bool = False
    isCLOSE_TTXPU_O_ATOMIC_SIM: bool = False
    isCloseDtypeConvert: bool = False
    isCloseInterleave: bool = False
    isCloseMemoryAsync: bool = True

    enable_fp_fusion: bool = False
    allow_fp8e4nv: bool = True
    allow_fp8e4b15: bool = False
    default_dot_input_precision: str = "tf32"
    allowed_dot_input_precisions: Tuple[str] = ("ieee", "tf32")

    exp_range: str = ":0"

    num_warps: int = (-1)  # TODO: invalid value, just to keep num_warps function signature
    num_ctas: int = -1  # TODO: invalid value, just to keep num_ctas function signature
    num_stages: int = 1

    # use_int4_w4a8
    use_int4_w4a8: bool = False

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
        return target.backend == "xpu"

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
        pm.run(mod)
        return mod

    @staticmethod
    def make_ttxir(mod, metadata, opt):
        metadata["xpu_arch"] = opt.arch
        metadata["is_sdnn"] = opt.is_sdnn or xpu.is_sdnn_kernel(mod)
        metadata["tensor_args"] = xpu.get_tensor_args(mod, []) if metadata["is_sdnn"] is True else []
        metadata["shared"] = (-1)  # TODO: invalid value, just to keep CompiledKernel _init_handles() success

        max_buffer_size = metadata["buffer_size_limit"]
        elem_bytes = int(os.environ.get("TRITONXPU_ELEMBYTES", 0))
        groups_per_cluster = metadata["groups_per_cluster"]
        unroll_num = metadata["unroll_num"]
        XPUBackend.buffer_len = xpu.get_buffer_len(mod, max_buffer_size, elem_bytes)
        # print(f"XPUBackend.buffer_len = {XPUBackend.buffer_len}")
        core_num = metadata["core_num"]
        # F/O Prefix For Function/Optimization Macro
        is_use_mask_zero = metadata["is_use_mask_zero"]
        if is_use_mask_zero:
            warnings.warn(
                f'XRE Version Must Be More than 5.0.21.37 (After 2025.07.22). And echo 1 > /proc/kunlun/dev4/dma_excp_mask',
                UserWarning)
        TTXPU_F_INTERLEAVE = 0 if metadata["grid"] != (12, 1, 1) else int(os.environ.get("TRITONXPU_INTERLEAVE", 1))
        TTXPU_F_OHTER_VALUE_SIM = int(os.environ.get("TRITONXPU_OTHER_SIM", 0))
        TTXPU_F_STORE_MASK_SIM = int(os.environ.get("TRITONXPU_STORE_MASK_SIM", 0))
        TTXPU_F_DTYPE_CONVERT = 0 if metadata["isCloseDtypeConvert"] else int(
            os.environ.get("TRITONXPU_DTYPE_CONVERT", 1))
        TTXPU_O_ATOMIC_SIM = 0 if metadata["isCLOSE_TTXPU_O_ATOMIC_SIM"] else int(
            os.environ.get("TRITONXPU_ATOMIC_SIM", 1))
        TTXPU_O_CLOSE_OPT = int(os.environ.get("TRITONXPU_CLOSE_OPTIMIZE", 0))
        TTSDNN_F_SINGLE_CORE_MODE = int(os.environ.get("TRITON_SDNN_SINGLE_CODE_MODE", 0))
        TTSDNN_F_MATMUL_FAST_MODE = int(os.environ.get("XMLIR_MATMUL_FAST_MODE", 0))
        TTSDNN_F_DMA_MODE = int(os.environ.get("XMLIR_DMA_FAST_MODE", 0))

        if opt.isClusterOneCoreActOnly and TTXPU_F_OHTER_VALUE_SIM:
            assert 0, "isClusterOneCoreActOnly and TRITONXPU_OTHER_SIM can't act simultaneously"

        pm = ir.pass_manager(mod.context)
        pm.enable_debug()
        xpu.passes.ttsdnnir.add_tritonsdnn_strip_all_ops_pass(pm, opt.arch)
        if metadata["is_sdnn"]:
            if opt.arch < 4:
                if metadata["use_int4_w4a8"] == True:
                    xpu.passes.ttsdnnir.add_triton_convert_type_pass(pm, opt.arch, 2)
                else:
                    xpu.passes.ttsdnnir.add_triton_convert_type_pass(pm, opt.arch, TTSDNN_F_MATMUL_FAST_MODE)
            xpu.passes.ttsdnnir.add_convert_triton_to_tritonsdnn_pass(pm, opt.arch)
            xpu.passes.ttsdnnir.add_linalg_to_tritonsdnn_pass(pm, opt.arch)
            xpu.passes.ttsdnnir.add_tritonsdnn_legalize_pass(pm, opt.arch)
            xpu.passes.ttsdnnir.add_tritonsdnn_combine_before_pass(pm, opt.arch)
            xpu.passes.ttsdnnir.add_tritonsdnn_bufferize_pass(pm, opt.arch)
            xpu.passes.ttsdnnir.add_tritonsdnn_combine_pass(pm, opt.arch)
            if opt.exp_range != ":0":
                res = parse_floating_range_string(opt.exp_range)
                xpu.passes.ttsdnnir.add_tritonsdnn_ew_act_table_pass(pm, res)
            else:
                xpu.passes.ttsdnnir.add_tritonsdnn_ew_act_table_pass(pm, None)
            xpu.passes.ttsdnnir.add_tritonsdnn_loop_grid_pass(pm)
            if opt.arch == 4 and TTSDNN_F_DMA_MODE:
                xpu.passes.ttsdnnir.add_tritonsdnn_remove_ds_op_pass(pm, opt.arch)
            if not TTSDNN_F_SINGLE_CORE_MODE:
                xpu.passes.ttsdnnir.add_tritonsdnn_pipeline_pass(pm)

            xpu.passes.ttsdnnir.add_tritonsdnn_multi_buffer_pass(pm, opt.arch, opt.num_stages)
            passes.common.add_symbol_dce(pm)
            passes.common.add_canonicalizer(pm)
            passes.common.add_cse(pm)
        else:
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
            if not metadata["isCloseCoreTiling"]:
                xpu.passes.ttxpuir.add_tritonxpu_core_tiling_pass(
                    pm, 0, XPUBackend.buffer_len, core_num,
                    groups_per_cluster) if not TTXPU_O_CLOSE_OPT else None  # dumpFlag=0
            # xpu.passes.ttxpuir.add_tritonxpu_lm_to_sm_pass(pm)
            passes.common.add_cse(pm)
            if not metadata["isCloseOffsetAnalysis"]:
                xpu.passes.ttxpuir.add_tritonxpu_offset_state_pass(
                    pm, 0, XPUBackend.buffer_len, is_use_mask_zero) if not TTXPU_O_CLOSE_OPT else None  # dumpFlag=0
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
                xpu.passes.ttxpuir.add_tritonxpu_vectorize_pass(
                    pm, 0, compareFusion) if not TTXPU_O_CLOSE_OPT else None  # dumpFlag=0
            passes.common.add_canonicalizer(pm)
            xpu.passes.ttxpuir.add_tritonxpu_alloca_pass(pm, XPUBackend.buffer_len, core_num)
            if not metadata["isCloseMemoryAsync"]:
                xpu.passes.ttxpuir.add_tritonxpu_memory_async_pass(pm,
                                                                   0) if not TTXPU_O_CLOSE_OPT else None  # dumpFlag=0
            if not metadata["isCloseUnrollControl"]:
                xpu.passes.ttxpuir.add_tritonxpu_unroll_control_pass(pm, XPUBackend.buffer_len, core_num,
                                                                     is_use_mask_zero,
                                                                     unroll_num) if not TTXPU_O_CLOSE_OPT else None
            xpu.passes.ttxpuir.add_tritonxpu_store_control_pass(pm) if not TTXPU_O_CLOSE_OPT else None
            if not TTXPU_F_OHTER_VALUE_SIM:
                xpu.passes.ttxpuir.add_tritonxpu_other_sim_pass(pm, XPUBackend.buffer_len, core_num)
            xpu.passes.ttxpuir.add_tritonxpu_memory_inplace_pass(pm, XPUBackend.buffer_len,
                                                                 core_num) if not TTXPU_O_CLOSE_OPT else None
            if not metadata["isCloseClusterLoopGrid"]:
                xpu.passes.ttxpuir.add_tritonxpu_cf_to_scf_pass(pm)
                xpu.passes.ttxpuir.add_tritonxpu_loop_grid_pass(pm)
            passes.common.add_cse(pm)
            passes.common.add_licm(pm)
            passes.common.add_symbol_dce(pm)

        try:
            pm.run(mod)
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
        metadata["tensor_args"] = xpu.get_tensor_args(mod,
                                                      metadata["tensor_args"]) if metadata["is_sdnn"] is True else []

        # TritonXPU -> LLVM-IR (MLIR)
        pm = ir.pass_manager(mod.context)
        pm.enable_debug()
        # xpu.passes.ttxpuir.add_decompose_unsupported_conversions(pm, opt.arch)
        passes.convert.add_scf_to_cf(pm)  # cf->llvm exist  choose scf->cf->llvm
        # passes.convert.add_index_to_llvmir(pm) // TODO[dyq]: necessary?

        if metadata["is_sdnn"]:
            xpu.passes.ttsdnnir.add_convert_tritonsdnn_to_llvm_pass(pm, opt.arch)
        else:
            passes.ttgpuir.add_allocate_shared_memory(pm)
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

        pm.run(mod)

        # LLVM-IR (MLIR) -> LLVM-IR (LLVM)
        llvm.init_targets()
        context = llvm.context()
        llvm_mod = llvm.to_module(mod, context)

        if opt.extern_libs:
            paths = [path for (name, path) in opt.extern_libs if xpu.llvm.need_extern_lib(mod)]
            assert (len(paths) <= 2), f"Expected 0/1/2 extern_lib path, but found {len(paths)}"
            llvm.link_extern_libs(llvm_mod, paths)

        llvm.optimize_module(llvm_mod, llvm.OPTIMIZE_O3, f"xpu{opt.arch}")
        xpu.llvm.amend_func(llvm_mod, mod, context, opt.arch, metadata["is_sdnn"])

        del context
        return llvm_mod

    @staticmethod
    def make_elf(mod, metadata, opt):
        # Find kernel names (there should only be one)
        # We get the name at the last possible step to accomodate `triton.compile`
        # on user-provided LLVM
        metadata["name"] = xpu.llvm.get_kernel_name(mod)

        # llvm -> elf/asm
        triple = f"xpu{opt.arch}-baidu-none-gnu"
        proc = f"xpu{opt.arch}"
        flags = ["xpu-cmp-nan"] if metadata["isOpenCmpNan"] else []
        ret_asm = xpu.llvm.translate_to_asm(mod, triple, proc, "", flags, False, False)
        fn_cache_manager = get_cache_manager(metadata["hash"])
        fn_cache_manager.put(ret_asm, f"{metadata['name']}.asm")
        ret_elf = xpu.llvm.translate_to_asm(mod, triple, proc, "", [], False, True)

        del mod
        return ret_elf

    @staticmethod
    def make_xpubin(mod, metadata, opt):
        with tempfile.TemporaryDirectory() as tmpdir:
            clang_path = XPUBackend.path_to_xpu_compile_tool(opt)
            elfconv = os.path.join(Path(__file__).parent, f"xpu{opt.arch}-elfconv")
            objfile = os.path.join(tmpdir, "kernel.o")
            binfile = os.path.join(tmpdir, "kernel.bin")
            with open(objfile, "wb") as f:
                f.write(mod)
            cmd = [elfconv, objfile, binfile, clang_path]
            out = subprocess.run(cmd, check=True, capture_output=True)
            printf_buf_offset_res = re.search(rb"0x[0-9a-fA-F]+", out.stdout)
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
    def is_elf_stack_size_oob(mod) -> bool:
        stack_size_oob = llvm.is_elf_stack_size_oob(mod)
        return stack_size_oob

    def hash(self) -> str:
        """Returns a unique identifier for this backend"""
        # TODO:
        return f"1"

    def parse_options(self, options: dict) -> object:
        args = {"arch": self.target.arch}
        args.update({k: options[k] for k in XPUOptions.__dataclass_fields__.keys() if k in options})
        return XPUOptions(**args)

    def add_stages(self, stages: dict, options: object) -> None:
        stages["ttir"] = lambda src, metadata: self.make_ttir(src, metadata, options)
        stages["ttxir"] = lambda src, metadata: self.make_ttxir(src, metadata, options)
        stages["llir"] = lambda src, metadata: self.make_llir(src, metadata, options)
        stages["elf"] = lambda src, metadata: self.make_elf(src, metadata, options)
        stages["xpubin"] = lambda src, metadata: self.make_xpubin(src, metadata, options)

    def pack_metadata(self, metadata):
        return (
            metadata.cluster_dims[0],
            metadata.cluster_dims[1],
            metadata.cluster_dims[2],
        )

    def get_codegen_implementation(self):
        codegen_fns = dict()
        return codegen_fns

    def load_dialects(self, context):
        xpu.load_dialects(context)
