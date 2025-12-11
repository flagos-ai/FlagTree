# Copyright (c) 2025 by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
from triton.backends.compiler import BaseBackend, GPUTarget
from triton._C.libtriton import ir, passes, metax
try:
    #from triton._C.libtriton import distributed
    enable_dist = True
except ImportError:
    enable_dist = False
from dataclasses import dataclass
import functools
from typing import Any, Tuple, Optional
import hashlib
import re
import os
import subprocess
from pathlib import Path


@functools.lru_cache()
def _path_to_binary(binary: str):
    paths = [
        os.environ.get(f"TRITON_{binary.upper()}_PATH", ""),
        os.path.join(os.path.dirname(__file__), "bin", binary),
    ]

    for bin in paths:
        if os.path.exists(bin) and os.path.isfile(bin):
            result = subprocess.check_output([bin, "--version"], stderr=subprocess.STDOUT)
            if result is not None:
                version = re.search(r".*release (\d+\.\d+).*", result.decode("utf-8"), flags=re.MULTILINE)
                if version is not None:
                    return bin, version.group(1)
    raise RuntimeError(f"Cannot find {binary}")


@functools.lru_cache()
def get_ptxas_version():
    version = subprocess.check_output([_path_to_binary("ptxas")[0], "--version"]).decode("utf-8")
    return version


@functools.lru_cache()
def ptx_get_version(cuda_version) -> int:
    '''
    Get the highest PTX version supported by the current CUDA driver.
    '''
    assert isinstance(cuda_version, str)
    major, minor = map(int, cuda_version.split('.'))
    if major == 12:
        return 80 + minor
    if major == 11:
        return 70 + minor
    if major == 10:
        return 63 + minor
    raise RuntimeError("Triton only support CUDA 10.0 or higher")


@functools.lru_cache(None)
def file_hash(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def maca_get_kernel_name(src: str) -> str:
    '''
    Get kernel name from llvm ir.
    This Kernel name is required when launching the kernel.
    '''
    assert src
    import re
    for line in src.split('\n'):
        line = line.strip()
        if line.startswith('define metaxgpu_kernel void @'):
            return re.match(r"define metaxgpu_kernel void @(.+?)\(", line).groups()[0]


def parse_option(string):
    return [item for item in string.split(';') if item]


def get_lld_version():
    maca_path = os.environ.get('MACA_PATH')
    assert maca_path, "Not found MACA_PATH"
    lld_path = maca_path + "/mxgpu_llvm/bin/ld.lld"
    try:
        result = subprocess.run([lld_path, '--version'], capture_output=True, text=True, timeout=10)
        version_output = result.stdout

        version_match = re.search(r'LLD\s+(\d+)\.\d+\.\d+', version_output)
        if not version_match:
            version_match = re.search(r'lld\s+(\d+)\.\d+\.\d+', version_output, re.IGNORECASE)

        if version_match:
            major_version = int(version_match.group(1))
            return major_version
        else:
            return 0

    except FileNotFoundError:
        return 0
    except subprocess.TimeoutExpired:
        return 0
    except Exception:
        return 0


@dataclass(frozen=True)
class MACAOptions:
    num_warps: int = 4
    num_ctas: int = 1
    num_stages: int = 3
    # maxnreg corresponds to the ptx parameter .maxnreg, which controls the
    # maximum number of 32-bit registers used by one thread.
    maxnreg: Optional[int] = None
    cluster_dims: tuple = (1, 1, 1)
    ptx_version: int = None
    enable_fp_fusion: bool = True
    allow_fp8e4nv: bool = False
    allow_fp8e4b15: bool = False
    default_dot_input_precision: str = "tf32"
    allowed_dot_input_precisions: Tuple[str] = ("tf32", "tf32x3", "ieee")
    max_num_imprecise_acc_default: bool = None
    extern_libs: dict = None
    debug: bool = False
    backend_name: str = 'maca'
    # MACA: new args
    pipeline: str = "basic"
    scenario: str = ""
    arch: str = None
    extra_options: str = ""
    pipeline_load_num: int = -1

    def __post_init__(self):
        default_libdir = os.getenv("MACA_PATH") + '/lib'
        ext_default_libdir = Path(__file__).parent / 'lib'
        extern_libs = {} if self.extern_libs is None else dict(self.extern_libs)
        if not extern_libs.get('libdevice', None):
            # ext_maca_mathlib.bc
            env_ext_libdevice_path = os.getenv("TRITON_EXT_LIBDEVICE_PATH", None)
            ext_libdevice_path = env_ext_libdevice_path if env_ext_libdevice_path is not None else str(
                ext_default_libdir) + '/ext_maca_mathlib.bc'
            assert os.path.exists(ext_libdevice_path), "ext_maca_mathlib.bc do not exit, please check!"
            extern_libs['ext_libdevice'] = ext_libdevice_path
            # maca_kernellib.bc
            env_kernel_libdevice_path = os.getenv("TRITON_KERNEL_LIBDEVICE_PATH", None)
            kernel_libdevice_path = env_kernel_libdevice_path if env_kernel_libdevice_path is not None else default_libdir + '/maca_kernellib.bc'
            extern_libs['kernel_libdevice'] = kernel_libdevice_path
            # maca_mathlib.bc
            env_libdevice_path = os.getenv("TRITON_LIBDEVICE_PATH", None)
            libdevice_path = env_libdevice_path if env_libdevice_path is not None else default_libdir + '/maca_mathlib.bc'
            extern_libs['libdevice'] = libdevice_path
        object.__setattr__(self, 'extern_libs', tuple(extern_libs.items()))
        assert self.num_warps > 0 and self.num_warps <= 16 and (self.num_warps & (self.num_warps - 1)) == 0, \
                "num_warps must be a power of 2 or greater than 0 and less than or equal to 16"

    def hash(self):
        hash_dict = dict(self.__dict__)
        # hash_dict["extern_libs"] = tuple((k, file_hash(v)) for k, v in sorted(hash_dict["extern_libs"]))
        key = "_".join([f"{name}-{val}" for name, val in sorted(hash_dict.items())])
        return hashlib.sha256(key.encode("utf-8")).hexdigest()


class MACABackend(BaseBackend):

    @staticmethod
    def supports_target(target: GPUTarget):
        return target.backend == 'maca'

    def __init__(self, target: GPUTarget) -> None:
        super().__init__(target)
        self.capability = target.arch
        assert isinstance(self.capability, int)
        self.binary_ext = "mcfatbin"

    def parse_options(self, opts) -> Any:
        args = {k: opts[k] for k in MACAOptions.__dataclass_fields__.keys() if k in opts}
        # USE_MACA: support allow_fp8e4nv(i.e. float8_e4m3fn)
        args["allow_fp8e4nv"] = True
        # args["allow_fp8e4nv"] = False
        args["allow_fp8e4b15"] = False
        args["max_num_imprecise_acc_default"] = 2**30 if self.capability == 90 else 0
        return MACAOptions(**args)

    def pack_metadata(self, metadata):
        return (
            metadata.num_warps,
            metadata.num_ctas,
            metadata.shared,
            metadata.cluster_dims[0],
            metadata.cluster_dims[1],
            metadata.cluster_dims[2],
        )

    def get_codegen_implementation(self):
        import triton.language.extra.cuda as cuda
        codegen_fns = {
            "convert_custom_types":
            cuda.convert_custom_float8_sm80 if self.capability >= 80 else cuda.convert_custom_float8_sm70
        }
        return codegen_fns

    def load_dialects(self, ctx):
        metax.load_dialects(ctx)

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
    def make_ttgir(mod, metadata, opt, capability):
        assert opt.pipeline_load_num >= -1, "invalid pipeline_load_num value!"
        scenarios = parse_option(opt.scenario)
        disable_prefetch = "unprefetch" in scenarios
        fullstage = "fullstage" in scenarios
        store_coalesce = "storeCoalesce" in scenarios
        mla = "mla" in scenarios
        single_shm = "singleshm" in scenarios
        use_opt_maca_mma = True
        use_opt_maca_mma = (opt.pipeline != "" and not os.getenv("TRITON_DISABLE_MACA_OPT_MMA"))
        dot_operands_out_loop = ("dot_operands_out_loop" in scenarios
                                 or os.getenv("TRITON_ENABLE_MACA_OPT_MOVE_DOT_OPERANDS_OUT_LOOP"))
        cvt_shm_no_pad = ("cvt_shm_no_pad" in scenarios and not os.getenv("TRITON_DISABLE_MACA_OPT_CVT_SHM_NO_PAD"))
        merge_convert_layout = os.getenv("TRITON_ENABLE_MACA_MERGE_CONVERT_LAYOUT")
        # TTIR -> TTGIR
        pm = ir.pass_manager(mod.context)
        pm.enable_debug()
        passes.ttir.add_convert_to_ttgpuir(pm, f"cuda:{capability}", opt.num_warps, 64, opt.num_ctas)
        # optimize TTGIR
        passes.ttgpuir.add_coalesce(pm)
        if capability // 10 >= 8:
            passes.ttgpuir.add_f32_dot_tc(pm)
        passes.ttgpuir.add_remove_layout_conversions(pm)
        passes.ttgpuir.add_optimize_thread_locality(pm)

        if opt.pipeline == "cpasync":
            disable_prefetch = True
            metax.passes.ttgpuir.add_tritonmetaxgpu_change_layout_for_int8_pass(pm, opt.num_stages, opt.pipeline)
        metax.passes.ttgpuir.add_accelerate_matmul(pm, opt.num_stages, disable_prefetch, store_coalesce, capability)
        passes.ttgpuir.add_remove_layout_conversions(pm)
        if not os.getenv("TRITON_DISABLE_CONSTANCY_LOAD_LAYOUT_OPT"):
            metax.passes.ttgpuir.add_tritonmetaxgpu_change_layout_for_constancy_load_layout(pm)
            passes.ttgpuir.add_remove_layout_conversions(pm)
        if store_coalesce:
            metax.passes.ttgpuir.add_tritonmetaxgpu_change_layout_from_repn_to_elemn_pass(pm)
            metax.passes.ttgpuir.add_tritonmetaxgpu_optimize_cstore_pass(pm, opt.num_stages)
            passes.ttgpuir.add_remove_layout_conversions(pm)
        passes.ttgpuir.add_optimize_dot_operands(pm, capability >= 80)
        passes.common.add_cse(pm)
        if capability // 10 >= 8:
            passes.ttgpuir.add_combine_tensor_select_and_if(pm)
            if use_opt_maca_mma:
                if opt.pipeline == "basic":
                    if mla and single_shm:
                        # only mla=True and single_shm=True
                        metax.passes.ttgpuir.add_pipeline_maca(pm, opt.num_stages, opt.pipeline_load_num, fullstage,
                                                               True)
                    else:
                        metax.passes.ttgpuir.add_pipeline_maca(pm, opt.num_stages, opt.pipeline_load_num, fullstage,
                                                               False)
                elif opt.pipeline == "cpasync" and not mla:
                    metax.passes.ttgpuir.add_pipeline_async_tn(pm, opt.num_stages)
                    metax.passes.ttgpuir.add_pipeline_async_tt(pm, opt.num_stages)
                    metax.passes.ttgpuir.add_pipeline_async_base(pm, opt.num_stages, fullstage)
                elif mla and opt.num_stages == 2 and opt.pipeline == "cpasync":
                    metax.passes.ttgpuir.add_pipeline_async_multidot_mla_mixed(pm, opt.num_stages, fullstage,
                                                                               opt.pipeline_load_num, single_shm, True)
                elif mla and opt.num_stages == 2 and opt.pipeline == "mixed":
                    metax.passes.ttgpuir.add_pipeline_async_multidot_mla_mixed(pm, opt.num_stages, fullstage,
                                                                               opt.pipeline_load_num, single_shm, False)
                else:
                    print("no avalilable pipeline for maca")
            else:
                passes.ttgpuir.add_pipeline(pm, opt.num_stages)
        if use_opt_maca_mma and opt.pipeline == "basic" and "unprefetch" not in scenarios:
            metax.passes.ttgpuir.add_prefetch_maca(pm)
        elif not use_opt_maca_mma:
            passes.ttgpuir.add_prefetch(pm)
        passes.ttgpuir.add_optimize_dot_operands(pm, capability >= 80)
        passes.ttgpuir.add_remove_layout_conversions(pm)
        passes.ttgpuir.add_reduce_data_duplication(pm)
        passes.ttgpuir.add_reorder_instructions(pm)
        if dot_operands_out_loop:
            metax.passes.ttgpuir.add_tritonmetaxgpu_move_dot_operands_out_loop_pass(pm)
        if not os.getenv("TRITON_DISABLE_MACA_MERGE_EQUAL_SHARED_LAYOUT"):
            metax.passes.ttgpuir.add_tritonmetaxgpu_merge_equal_shared_layout_pass(pm)
        if cvt_shm_no_pad:
            metax.passes.ttgpuir.add_tritonmetaxgpu_change_convert_layout_attr(pm, False, False, True, False)
        if merge_convert_layout:
            metax.passes.ttgpuir.add_tritonmetaxgpu_merge_convert_layout_pass(pm)
        passes.common.add_cse(pm)
        passes.common.add_symbol_dce(pm)
        passes.common.add_canonicalizer(pm)
        pm.run(mod)
        return mod

    @staticmethod
    def make_mlir(src, metadata, options, capability):
        # warp-specialization mutates num_warps
        num_warp_groups = src.get_int_attr("triton_gpu.num-warp-groups-per-cta")
        if num_warp_groups is not None:
            metadata["num_warps"] *= num_warp_groups
        mod = src

        # TritonGPU -> LLVM-IR (MLIR)
        pm = ir.pass_manager(mod.context)
        pm.enable_debug()
        passes.ttgpuir.add_combine_tensor_select_and_if(pm)
        passes.convert.add_scf_to_cf(pm)
        passes.convert.add_index_to_llvmir(pm)
        passes.ttgpuir.add_allocate_shared_memory(pm)

        scenarios = parse_option(options.scenario)
        enSmIdxCache = ("smemOffsetCache" in scenarios or os.getenv("TRITON_ENABLE_SMEM_OFFSET_CACHE"))
        enSmIndexOpt = ("smemIndexOpt" in scenarios or os.getenv("TRITON_ENABLE_BSM_INDEX_OPT"))
        if not enSmIdxCache:
            isEnSmIdxCache = False
        else:
            isEnSmIdxCache = True
        if not enSmIndexOpt:
            isEnSmIndexOpt = False
        else:
            isEnSmIndexOpt = True
        metax.passes.ttgpuir.add_to_llvmir(pm, capability, isEnSmIdxCache, isEnSmIndexOpt)
        passes.convert.add_arith_to_llvmir(pm)
        passes.common.add_canonicalizer(pm)
        passes.common.add_cse(pm)
        passes.common.add_symbol_dce(pm)
        if os.environ.get("TRITON_DISABLE_LINE_INFO", "0") == "0":
            passes.llvmir.add_di_scope(pm)
        pm.run(mod)

        # Get some metadata
        metadata["shared"] = src.get_int_attr("triton_gpu.shared")
        ret = str(mod)
        return ret

    @staticmethod
    def make_llir(src, metadata, options, capability):
        mlir_opt_path = os.path.dirname(__file__) + "/bin/mlir-opt"
        maca_path = os.environ.get('MACA_PATH')
        assert maca_path, "Not found MACA_PATH"
        llvm_version = get_lld_version()
        opted_mlir = metax.mlir_opt(src, mlir_opt_path, llvm_version)
        llir = metax.translate_mlir_to_llir(opted_mlir, maca_path)
        if options.extern_libs:
            paths = [path for (name, path) in options.extern_libs]
            llir = metax.link_extern_libs(llir, paths, maca_path)
        metadata["name"] = maca_get_kernel_name(llir)
        return llir

    @staticmethod
    def make_mcfatbin(src, metadata, opt, capability):
        scenarios = parse_option(opt.scenario)
        opt_mxcc = os.environ.get("TRITON_COMPILER_OPT_PATH")
        mxcc_arch = os.environ.get('MACA_PATH') + "/mxgpu_llvm/bin/mxcc"
        if opt_mxcc:
            mxcc_arch = opt_mxcc + "/mxgpu_llvm/bin/mxcc"
        if mxcc_arch is None:
            raise RuntimeError('mxcc_arch is None (not specified)')
        compile_options = ""
        if (opt.pipeline == "basic" or opt.pipeline == "basic-prefetch") and "mla" not in scenarios:
            compile_options = " -mllvm -metaxgpu-sched-regpressure=false "
            if "fullstage" in scenarios:
                compile_options += " -mllvm -metaxgpu-vectorize-slp=true -mllvm -metaxgpu-igroup "
            else:
                compile_options += " -mllvm -metaxgpu-vectorize-slp=true -mllvm -metaxgpu-sched-mma-maxnum=3 "
            if "roll" not in scenarios:
                compile_options += " -mllvm -metaxgpu-mma-unroll-count=" + str(opt.num_stages) + " "
        elif opt.pipeline == "cpasync" and "mla" not in scenarios:
            compile_options = " -mllvm -metaxgpu-sched-regpressure=true "
            compile_options += " -mllvm -metaxgpu-sinkload=false -mllvm -metaxgpu-vectorize-slp=true -mllvm -metaxgpu-igroup -mllvm -metaxgpu-aggressive-4g-addr-opt=true \
                                -mllvm -metaxgpu-shl-add-combine=false -mllvm -misched-postra=true -mllvm -enable-post-misched=true "

            disable_int8_opt = ("disable_int8_opt" in scenarios) or (os.getenv("TRITON_DISABLE_MACA_COMPILER_INT8_OPT"))
            if not disable_int8_opt:
                compile_options += " -mllvm -metaxgpu-slp-vectorize-i8=true"

            if "unroll" in scenarios:
                compile_options += " -mllvm -metaxgpu-mma-unroll-count=" + str(opt.num_stages) + " "
        if "flashattn-fwd" in scenarios:
            compile_options = " -mllvm -metaxgpu-mma-sched=true -mllvm -metaxgpu-sched-select=metaxgpu-minreg -mllvm -map-use-pk-fma=1 "
        elif "flashattn-bwd" in scenarios:
            compile_options = " -mllvm -metaxgpu-sched-regpressure=true "
            compile_options += " -mllvm -metaxgpu-sinkload=false -mllvm -metaxgpu-vectorize-slp=true "
        if "mla" in scenarios:
            # maybe will change the compile options in mla later
            if opt.num_stages == 2:
                if opt.pipeline == "cpasync":
                    compile_options = " -mllvm -metaxgpu-sched-regpressure=true "
                    compile_options += " -mllvm -metaxgpu-sinkload=false -mllvm -metaxgpu-vectorize-slp=true -mllvm -metaxgpu-igroup -mllvm -metaxgpu-aggressive-4g-addr-opt=true \
                                        -mllvm -metaxgpu-shl-add-combine=false -mllvm -misched-postra=true -mllvm -enable-post-misched=true "

                    if "unroll" in scenarios:
                        compile_options += " -mllvm -metaxgpu-mma-unroll-count=" + str(opt.num_stages) + " "
                elif opt.pipeline == "basic" or opt.pipeline == "mixed":
                    compile_options = " -mllvm -metaxgpu-mma-sched=true -mllvm -map-use-pk-fma=1 -mllvm -metaxgpu-split-regalloc=true -mllvm -metaxgpu-aggressive-fold=true \
                                        -mllvm -metaxgpu-disable-licm=true "

                else:
                    assert False, "Please set pipeline for mla!"
            else:
                compile_options = " -mllvm -metaxgpu-mma-sched=true -mllvm -map-use-pk-fma=1 -mllvm -metaxgpu-split-regalloc=true -mllvm -metaxgpu-aggressive-fold=true "
        if opt.extra_options != "":
            compile_options = opt.extra_options
        if capability == 86:
            compile_options += " --offload-arch=xcore1500 "
        if capability == 89:
            compile_options += " --offload-arch=xcore1600 "
        # llvm19 workaround
        llvm_version = get_lld_version()
        if llvm_version == 19:
            compile_options += " -mllvm -metaxgpu-merge-copy-postra=false"
            compile_options += " -mllvm --vectorize-slp=false"
        # TODO: remove llvm19 workaround for aggressive-4g-addr-opt
        if ("noaddropt" in scenarios) or (os.getenv("TRITON_DISABLE_MACA_COMPILER_4G_ADDR_OPT")):
            compile_options = compile_options.replace("-mllvm -metaxgpu-aggressive-4g-addr-opt=true ", "")
        return metax.translate_llvmir_to_mcfatbin(src, mxcc_arch, os.environ.get('MACA_PATH'), compile_options)

    def add_stages(self, stages, options):
        stages["ttir"] = lambda src, metadata: self.make_ttir(src, metadata, options)
        stages["ttgir"] = lambda src, metadata: self.make_ttgir(src, metadata, options, self.capability)
        stages["mlir"] = lambda src, metadata: self.make_mlir(src, metadata, options, self.capability)
        stages["llir"] = lambda src, metadata: self.make_llir(src, metadata, options, self.capability)
        stages["mcfatbin"] = lambda src, metadata: self.make_mcfatbin(src, metadata, options, self.capability)

    @functools.lru_cache()
    def hash(self):
        mxcc_arch = os.environ.get('MACA_PATH') + "/mxgpu_llvm/bin/mxcc"
        if mxcc_arch is None:
            raise RuntimeError('mxcc_arch is None (not specified)')
        version = subprocess.check_output([mxcc_arch, "--version"]).decode("utf-8").split('\n', 1)[0]
        return f'{version}-{self.capability}'
