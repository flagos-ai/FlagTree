from triton.backends.compiler import BaseBackend, GPUTarget, Language
from triton._C.libtriton import ir, passes, llvm, sunrise
from triton import knobs
from dataclasses import dataclass
import functools
from typing import Any, Dict, Tuple
from types import ModuleType
import hashlib
import platform
import re
import tempfile
import os
import subprocess
from pathlib import Path

def min_dot_size(target: GPUTarget):
    return lambda lhsType, rhsType: (8, 8, 16) if lhsType.is_int8() else (8, 8, 4)

@functools.lru_cache(None)
def file_hash(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()

@dataclass(frozen=True)
class SunriseOptions:
    num_warps: int = 4
    num_ctas: int = 1
    num_stages: int = 3
    cluster_dims: tuple = (1, 1, 1)
    enable_fp_fusion: bool = True
    supported_fp8_dtypes: Tuple[str] = ("fp8e5", )
    deprecated_fp8_dot_operand_dtypes: Tuple[str] = ()
    default_dot_input_precision: str = "ieee"
    allowed_dot_input_precisions: Tuple[str] = ("ieee", )
    max_num_imprecise_acc_default: bool = None
    extern_libs: dict = None
    debug: bool = False
    backend_name: str = 'tang'
    sanitize_overflow: bool = True
    arch: str = None

    # 当前s2上没有响应的libdivice库,需要怎么编译出来？？
    def __post_init__(self):
        warp_size = 32
        object.__setattr__(self, 'warp_size', warp_size)
        default_libdir = Path(__file__).parent / 'lib'
        extern_libs ={} if self.extern_libs is None else dict(self.extern_libs)
        for lib in ["ocml", "ockl"]:
            extern_libs[lib] = str(default_libdir / f'{lib}.bc')
        object.__setattr__(self, 'extern_libs', tuple(extern_libs.items()))
        assert self.num_warps > 0 and (self.num_warps & (self.num_warps - 1)) == 0, \
               "num_warps must be a power of 2"

    def hash(self):
        hash_dict = dict(self.__dict__)
        hash_dict["extern_libs"] = tuple((k, file_hash(v)) for k, v in sorted(hash_dict["extern_libs"]))
        key = "_".join([f"{name}-{val}" for name, val in sorted(hash_dict.items())])
        return hashlib.sha256(key.encode("utf-8")).hexdigest()


class SunriseBackend(BaseBackend):

    @staticmethod
    def supports_target(target: GPUTarget):
        return target.backend == 'tang'

    def get_target_name(self, options) -> str:
        return f"tang:{options.arch}"  # tang:S2

    def __init__(self, target: GPUTarget) -> None:
        super().__init__(target)
        self.binary_ext = 'stcu'

    def parse_options(self, opts) -> Any:
        args = {'arch': knobs.runtime.override_arch or self.target.arch}
        if "enable_fp_fusion" not in opts:
            args["enable_fp_fusion"] = knobs.language.default_fp_fusion
        args["max_num_imprecise_acc_default"] = 0   # TODO
        args.update({k: opts[k] for k in SunriseOptions.__dataclass_fields__.keys() \
                     if k in opts and opts[k] is not None})
        return SunriseOptions(**args)

    def pack_metadata(self, metadata):
        return (
            metadata.num_warps,
            metadata.num_ctas,
            metadata.shared,
            metadata.cluster_dims[0],
            metadata.cluster_dims[1],
            metadata.cluster_dims[2],
        )

    def get_codegen_implementation(self, options):
        codegen_fns = {
            "min_dot_size": min_dot_size(self.target)
        }
        return codegen_fns

    def get_module_map(self) -> Dict[str, ModuleType]:
        from triton.language.extra.tang import libdevice
        return {"triton.language.extra.libdevice": libdevice}

    def load_dialects(self, ctx):
        sunrise.load_dialects(ctx)

    def path_to_clang_offload_bundler():
        lld_env_path = knobs.sunrise.lld_path
        if lld_env_path is not None:
            lld = Path(lld_env_path)
            if lld.is_file():
                return lld
        arch = platform.machine()
        lld = Path(f"/usr/local/tangrt/toolchains/llvm/prebuilt/linux-{arch}/bin/clang-offload-bundler")
        if lld.is_file():
            return lld
        raise Exception("clang-offload-bundler not found. Set 'TRITON_SUNRISE_LLD_PATH' to its path.")

    @staticmethod
    def get_triple():
        triple = knobs.sunrise.triple
        if triple is None or triple == '':
            return "stcu-unknown-tang"
        return triple

    @staticmethod
    def get_flag(metadata, opt):
        flag = knobs.sunrise.flag
        if flag is None or flag == []:
            flag = ['enable-predicate']
        if isinstance(flag, str):
            flag = flag.split()
        if metadata["num_warps"] > 16:
            flag.append('thread-regfile-size=64')
        for name, path in opt.extern_libs:
            if name == "ockl":
                flag.append('ocklPath='+path)
        return flag

    @staticmethod
    def get_optimization_level(llvm):
        opt = knobs.sunrise.opt_level
        if int(opt) == 0:
            return llvm.OPTIMIZE_O0
        if int(opt) == 1:
            return llvm.OPTIMIZE_O1
        if int(opt) == 2:
            return llvm.OPTIMIZE_O2
        return llvm.OPTIMIZE_O3

    @staticmethod
    def make_ttir(mod, metadata, opt):
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
        passes.ttir.add_loop_unroll(pm)
        pm.run(mod)
        return mod

    @staticmethod
    def make_ttgir(mod, metadata, opt, capability):
        num_stages = opt.num_stages if opt.num_stages <= 3 else 3
        # TTIR -> TTGIR
        pm = ir.pass_manager(mod.context)
        pm.enable_debug()
        passes.ttir.add_convert_to_ttgpuir(pm, f"tang:{capability}", opt.num_warps, 32, opt.num_ctas)
        # optimize TTGIR
        passes.ttgpuir.add_coalesce(pm)
        # TODO(Qingyi): Move PlanCTAPass to the front of CoalescePass
        # nvidia.passes.ttnvgpuir.add_plan_cta(pm, cluster_info)
        passes.ttgpuir.add_remove_layout_conversions(pm)
        passes.ttgpuir.add_optimize_thread_locality(pm)
        sunrise.passes.ttgpuir.add_combine_optimize(pm)
        if os.getenv('OFF_MMA', '0') == '1':
            print('not run accelerate_matmul pass')
        else:
            sunrise.passes.ttgpuir.add_accelerate_matmul(pm, 1, 0) # 版本：1.0
            sunrise.passes.ttgpuir.add_mma_direct_store(pm)
            passes.ttgpuir.add_remove_layout_conversions(pm)
        # passes.ttgpuir.add_optimize_dot_operands(pm, True)
        passes.common.add_cse(pm)
        if os.getenv('DFT_PP', '0') == '1':
            if os.getenv('OFF_ASYNC', '0') == '0':
                passes.ttgpuir.add_assign_latencies(pm, num_stages)
                passes.ttgpuir.add_schedule_loops(pm)
                passes.ttgpuir.add_pipeline(pm, num_stages, False )
            if os.getenv('OFF_PREF', '0') == '0':
                passes.ttir.add_loop_aware_cse(pm)
                passes.common.add_canonicalizer(pm)
                passes.ttir.add_loop_aware_cse(pm)
                passes.ttgpuir.add_prefetch(pm)
        else:
            if os.getenv('OFF_ASYNC', '0') == '0':
                passes.ttgpuir.add_assign_latencies(pm, num_stages)
                passes.ttgpuir.add_schedule_loops(pm)
                sunrise.passes.ttgpuir.add_pipeline(pm, num_stages, 1, 0) # 版本：1.0
            if os.getenv('OFF_PREF', '0') == '0':
                passes.ttir.add_loop_aware_cse(pm)
                passes.common.add_canonicalizer(pm)
                passes.ttir.add_loop_aware_cse(pm)
                passes.ttgpuir.add_prefetch(pm)
        # passes.ttgpuir.add_optimize_dot_operands(pm, True)
        passes.ttgpuir.add_remove_layout_conversions(pm)
        passes.ttgpuir.add_reduce_data_duplication(pm)
        passes.ttgpuir.add_reorder_instructions(pm)
        passes.common.add_cse(pm)
        passes.common.add_symbol_dce(pm)
        if os.getenv('K_OUTER', '0') == '1':
            print('not run split_dot pass because K_OUTER == 1')
        else:
            sunrise.passes.ttgpuir.add_split_dot(pm, 1, 0)
        # if capability // 10 >= 9:
        #     nvidia.passes.ttnvgpuir.add_fence_insertion(pm)
        passes.common.add_canonicalizer(pm)
        pm.run(mod)
        return mod

    def ttgir_opt(self, src, metadata, options, capability):
        mod = src
        pm = ir.pass_manager(mod.context)
        pm.enable_debug()

        passes.ttgpuir.add_inliner(pm)
        passes.common.add_sccp(pm)
        passes.ttir.add_loop_aware_cse(pm)
        passes.ttgpuir.add_canonicalizer(pm)
        passes.ttgpuir.add_combine_tensor_select_and_if(pm)

        pm.run(mod)
        # metadata["tensordesc_meta"] = mod.get_tensordesc_metadata()
        return mod

    @staticmethod
    def make_llir(src, metadata, options, capability):
        mod = src
        # TritonGPU -> LLVM-IR (MLIR)
        pm = ir.pass_manager(mod.context)
        pm.enable_debug()
        passes.ttgpuir.add_combine_tensor_select_and_if(pm)
        passes.convert.add_scf_to_cf(pm)
        passes.convert.add_index_to_llvmir(pm)
        passes.ttgpuir.add_allocate_shared_memory(pm)
        sunrise.passes.ttgpuir.add_to_llvmir(pm, capability)
        sunrise.passes.ttgpuir.add_remove_repeated_fence(pm)
        passes.convert.add_scf_to_cf(pm)
        passes.convert.add_cf_to_llvmir(pm)
        passes.convert.add_arith_to_llvmir(pm)
        passes.common.add_canonicalizer(pm)
        passes.common.add_cse(pm)
        passes.common.add_symbol_dce(pm)
        if not knobs.compilation.disable_line_info:
            passes.llvmir.add_di_scope(pm)
        pm.run(mod)
        # LLVM-IR (MLIR) -> LLVM-IR (LLVM)
        llvm.init_targets()
        context = llvm.context()
        llvm_mod = llvm.to_module(mod, context)
        llvm.attach_datalayout(llvm_mod, 'stcu-unknown-tang', '', '')
        # sunrise.set_nvvm_reflect_ftz(llvm_mod)   # 属性设置，可以不需要
        if options.extern_libs:
            for name, path in options.extern_libs:
                if name == "ocml":
                    llvm.link_extern_libs(llvm_mod, [path])
                # if name == "ockl":
                #     llvm.link_override_lib(llvm_mod, path)
        llvm.optimize_module(llvm_mod, SunriseBackend.get_optimization_level(llvm))

        # Get some metadata
        total_num_warps = src.get_int_attr("ttg.total-num-warps")
        if total_num_warps is not None:
            metadata["num_warps"] = total_num_warps
        metadata["shared"] = src.get_int_attr("ttg.shared")
        ret = str(llvm_mod)
        ret = ret.replace("define void @", "define dso_local cc200 void @")
        if knobs.sunrise.dump_stcu:
            with open('sunrise.ll', 'w') as f:
                f.write(ret)
        del llvm_mod
        del context
        return ret

    @staticmethod
    def make_stcu(src, metadata, opt, capability):
        names = re.findall(r"define dso_local cc200 void @([a-zA-Z_][a-zA-Z0-9_]*)", src)

        assert len(names) == 1
        metadata["name"] = names[0]
        proc = ''

        triple = SunriseBackend.get_triple()
        flag = SunriseBackend.get_flag(metadata, opt)
        if knobs.sunrise.dump_stcu:
            asm_debug = llvm.translate_to_asm(src, triple, proc, '', flag, opt.enable_fp_fusion,
                                              False)
            with open('sunrise.asm', 'w') as f:
                f.write(asm_debug)

        asm = llvm.translate_to_asm(src, triple, proc, '', flag, opt.enable_fp_fusion, True)
        if knobs.sunrise.dump_stcu:
            with open('sunrise.elf', 'wb') as f:
                f.write(asm)

        bundler = SunriseBackend.path_to_clang_offload_bundler()

        major = 0
        try:
            output = subprocess.check_output([bundler, "--version"], stderr=subprocess.STDOUT)
            version_str = output.decode("utf-8").strip()
            match = re.search(r"version\s+(\d+)\.(\d+)\.(\d+)", version_str)
            if match:
                major = int(match.group(1))
            else:
                print("Cannot parse clang-offload-bundler version\n")
        except Exception as e:
            print("Error getting version:", e)

        arch = platform.machine()

        with tempfile.NamedTemporaryFile() as tmp_out:
            with tempfile.NamedTemporaryFile() as tmp_in:
                with open(tmp_in.name, 'wb') as fd_in:
                    fd_in.write(asm)
                try:
                    cmd = f'{bundler} -type=o -targets=host-{arch}-unknown-linux,tang-stpu-unknown-tang -input=/dev/null -input={tmp_in.name} -output={tmp_out.name}'
                    subprocess.run(cmd, shell=True, check=True)
                except subprocess.CalledProcessError as e:
                    print(" run error\n")

            with open(tmp_out.name, 'rb') as fd_out:
                ret = fd_out.read()
        return ret

    def add_stages(self, stages, options, language):
        capability = 80 # options.arch
        if language == Language.TRITON:
            stages["ttir"] = lambda src, metadata: self.make_ttir(src, metadata, options)
            stages["ttgir"] = lambda src, metadata: self.make_ttgir(src, metadata, options, capability)
        elif language == Language.GLUON:
            stages["ttgir"] = lambda src, metadata: self.ttgir_opt(src, metadata, options, capability)
        stages["llir"] = lambda src, metadata: self.make_llir(src, metadata, options, capability)
        stages["stcu"] = lambda src, metadata: self.make_stcu(src, metadata, options, capability)

    @functools.lru_cache()
    def hash(self):
        version = subprocess.check_output([SunriseBackend.path_to_clang_offload_bundler(), "--version"], encoding='utf-8')
        return f'{version}-{self.target.arch}'
