from triton.backends.compiler import BaseBackend, GPUTarget
from triton._C.libtriton import ir, passes, spacemit
from dataclasses import dataclass
from typing import Any, Dict, Tuple
from types import ModuleType
import hashlib
import sys
import tempfile
import shutil
import os
import re
import subprocess
import functools
from pathlib import Path
from . import (
    get_spine_triton_opt_path,
    dump_ir_if_needed,
    get_llvm_bin_path,
    get_spine_mlir_opt_path,
    extract_kernel_name,
    get_cpu_name_from_arch_id,
    get_spine_mlir_opt_options,
    get_cpu_arch,
    get_target_arch,
    get_cross_toolchain,
)


def _ttir_to_linalgdir(mod, metadata):
    # Get Triton-MLIR as string
    ttir_code = str(mod)
    with tempfile.TemporaryDirectory() as tmpdir:
        src_path = os.path.join(tmpdir, "tt.mlir")
        dst_path = os.path.join(tmpdir, "linalg.mlir")
        Path(src_path).write_text(ttir_code)
        dump_ir_if_needed([src_path], metadata["name"])
        spine_triton_opt_path = get_spine_triton_opt_path()
        subprocess.check_call([
            spine_triton_opt_path,
            src_path,
            "--triton-to-linalg-experimental",
            "-o",
            dst_path,
        ])
        dump_ir_if_needed([dst_path], metadata["name"])
        return Path(dst_path).read_text()


def _optimize_linalgdir(linalgdir: str):
    # We don't apply any optimizations now, but we can add passes if needed.
    return linalgdir


def _spine_mlir_linalgdir_to_llir_ref(linalgdir: str, metadata):
    with tempfile.TemporaryDirectory() as tmpdir:
        linalg_path = os.path.join(tmpdir, "linalg.mlir")
        llmlir_path = os.path.join(tmpdir, "ll.mlir")
        llir_path = os.path.join(tmpdir, "ll.ir")
        Path(linalg_path).write_text(linalgdir)
        # SpineTriton-MLIR to LLVM-MLIR
        spine_mlir_path = get_spine_mlir_opt_path()

        pipeline_option_str = get_spine_mlir_opt_options()
        if pipeline_option_str == "":
            pipeline_option_str = "enable-always-tls=1"

        cmd_str = '{} {} --spine-triton-e2e-ref-pipeline="{}" -o {}'.format(spine_mlir_path, linalg_path,
                                                                            pipeline_option_str, llmlir_path)
        subprocess.check_call(
            cmd_str,
            shell=True,
        )

        # LLVM-MLIR to LLVM-IR
        mlir_translate_path = get_llvm_bin_path("mlir-translate")
        subprocess.check_call([mlir_translate_path, llmlir_path, "--mlir-to-llvmir", "-o", llir_path])
        dump_ir_if_needed([llmlir_path, llir_path], metadata["name"])
        return Path(llir_path).read_text()


def _spine_mlir_linalgdir_to_llir(linalgdir: str, metadata):
    with tempfile.TemporaryDirectory() as tmpdir:
        linalg_path = os.path.join(tmpdir, "linalg.mlir")
        llmlir_path = os.path.join(tmpdir, "ll.mlir")
        llir_path = os.path.join(tmpdir, ".ll")
        Path(linalg_path).write_text(linalgdir)
        spine_mlir_path = get_spine_mlir_opt_path()

        pipeline_option_str = get_spine_mlir_opt_options()
        if pipeline_option_str == "":
            pipeline_option_str = "enable-always-tls=1 enable-fuse-group=false"

        cmd_str = '{} {} --spine-triton-e2e-pipeline="{}" -o {}'.format(spine_mlir_path, linalg_path,
                                                                        pipeline_option_str, llmlir_path)
        subprocess.check_call(
            cmd_str,
            shell=True,
        )
        dump_ir_if_needed([llmlir_path], metadata["name"])

        llmlir_new_path = llmlir_path
        base_path = os.getenv("SPINE_TRITON_DUMP_PATH", "")
        if base_path:
            llmlir_new_path = os.path.join(tmpdir, "ll_with_debuginfo.mlir")
            subprocess.check_call([
                spine_mlir_path,
                os.path.join(
                    base_path,
                    metadata["name"] + "_" + os.path.basename(llmlir_path),
                ),
                "--ensure-debug-info-scope-on-llvm-func",
                "-mlir-print-debuginfo",
                "-o",
                llmlir_new_path,
            ])

        # LLVM-MLIR to LLVM-IR
        mlir_translate_path = get_llvm_bin_path("mlir-translate")
        subprocess.check_call([mlir_translate_path, llmlir_new_path, "--mlir-to-llvmir", "-o", llir_path])
        dump_ir_if_needed([llir_path], metadata["name"])
        return Path(llir_path).read_text()


def _optimize_llir(llir: str):
    # We don't apply any optimizations now, but we can add passes if needed.
    return llir


def _llir_to_so(llir: str, metadata):
    cpu_arch = get_cpu_arch()
    target_arch_id = metadata["target"].arch_id
    ai_cpu_arch = get_cpu_name_from_arch_id(target_arch_id)

    target_arch = get_target_arch()
    cross_toolchain = get_cross_toolchain()

    with tempfile.TemporaryDirectory() as tmpdir:
        src_path = os.path.join(tmpdir, ".ll")
        src_opt_path = os.path.join(tmpdir, ".opt.ll")
        asm_path = os.path.join(tmpdir, ".s")
        dst_path = os.path.join(tmpdir, ".o")
        Path(src_path).write_text(llir)

        llopt_path = get_llvm_bin_path("opt")
        llopt_flags = []
        if target_arch == "riscv64":
            llopt_flags.extend([
                "--march=riscv64", "-passes=loop-vectorize", "--pass-remarks-missed", "-force-vector-width=32",
                "-force-vector-interleave=2"
            ])

        subprocess.check_call([llopt_path, src_path, *llopt_flags, "-o", src_opt_path])

        llc_path = get_llvm_bin_path("llc")
        llc_flags = ["-O3", "--float-abi=hard", "--relocation-model=pic"]
        if target_arch == "riscv64":
            mattr_list = ["64bit", "a", "b", "c", "d", "f", "i", "m", "v", "zfh", "zvfh", "zicbop", "zicbom", "zicboz"]
            if ai_cpu_arch in {"spacemit-a200", "spacemit-a200m"}:
                mattr_list.extend(["xsmtvsfu", "zmatrix"])
            elif ai_cpu_arch in {"spacemit-a100"}:
                mattr_list.append("xsmtvdotii")
            elif ai_cpu_arch in {"spacemit-x100", "spacemit-x60", "spacemit-a60"}:
                mattr_list.append("xsmtvdoti")

            llc_flags.extend(["--march=riscv64", "--mattr=" + ",".join(mattr_list)])

        # Generate assembly for dump if SPINE_TRITON_DUMP_PATH exists, but still generate object file for the final output
        if (dum_dir := os.environ.get("SPINE_TRITON_DUMP_PATH", "")) != "" and os.path.exists(dum_dir):
            subprocess.check_call([llc_path, src_opt_path, *llc_flags, "-filetype=asm", "-o", asm_path])
            kernel_name = metadata.get("name", "unknown_kernel")
            asm_dump_path = os.path.join(dum_dir, "{}.s".format(kernel_name))
            shutil.copy(asm_path, asm_dump_path)

        subprocess.check_call([llc_path, src_opt_path, *llc_flags, "-filetype=obj", "-o", dst_path])
        # rpc_host = os.environ.get("SPINE_TRITON_RPC_HOST", "")
        # if rpc_host:
        #     # For RPC mode, we don't need to create a shared library
        #     with open(dst_path, "rb") as f:
        #         return f.read()

        dump_ir_if_needed([dst_path], metadata["name"])

        cpu_backend_path = Path(__file__).resolve().parent
        include_dir = os.path.join(cpu_backend_path, "include")
        so_path = os.path.join(tmpdir, ".so")
        runtime_lib_dir = os.path.join(cpu_backend_path.parent.parent, "_C")

        if target_arch == "riscv64" and cpu_arch != "riscv64":
            assert os.path.exists(cross_toolchain), "Cross-compilation toolchain path does not exist: {}".format(
                cross_toolchain)
            # Cross-compilation mode: use cross-compile toolchain
            cc = os.path.join(cross_toolchain, "bin", "clang++")
            sysroot = os.path.join(cross_toolchain, "sysroot")
            subprocess.check_call([
                cc,
                "--target=riscv64-unknown-linux-gnu",
                f"--sysroot={sysroot}",
                "-std=c++17",
                "-march=rv64gcv_zfh_zba_zicbop",
                "-mabi=lp64d",
                "-O3",
                dst_path,
                f"-I{include_dir}",
                "-shared",
                "-fPIC",
                "-fuse-ld=lld",
                "-nostdlib++",
                "-o",
                so_path,
            ])
        else:
            # Native compilation mode
            py_version = sys.version_info
            py_include_dir = os.path.join(
                sys.base_prefix,
                "include",
                f"python{sys.version_info.major}.{sys.version_info.minor}",
            )
            py_lib_dir = os.path.join(sys.base_prefix, "lib")
            py_lib = "{name}{major}.{minor}".format(name="python", major=py_version.major, minor=py_version.minor)

            gcc_flags = []
            if target_arch == "riscv64":
                gcc_flags.extend(["-march=rv64gcv_zfh_zba_zicbop", "-mabi=lp64d", "-O3"])
            subprocess.check_call([
                "g++",
                "-std=c++17",
                *gcc_flags,
                dst_path,
                f"-I{py_include_dir}",
                f"-I{include_dir}",
                f"-L{py_lib_dir}",
                f"-L{runtime_lib_dir}",
                "-shared",
                f"-l{py_lib}",
                "-lSpineTritonRuntime",  # spine-triton's own runtime: spine_assert, spine_print_unranked_memref, proton, etc.
                "-fPIC",
                "-o",
                so_path,
            ])

        dump_ir_if_needed([so_path], metadata["name"])
        with open(so_path, "rb") as f:
            return f.read()


@dataclass(frozen=True)
class CPUOptions:
    debug: bool = False
    arch: str = None
    num_warps: int = 0
    num_ctas: int = 0
    num_stages: int = 1
    enable_warp_specialization: bool = False
    enable_fp_fusion: bool = False
    extern_libs = None
    cluster_dims: tuple = (1, 1, 1)
    shared: bool = False
    # Disable FP8 here since this is a sample CPU backend.
    # Target specific backends can eanble it with supported types.
    supported_fp8_dtypes: Tuple[str] = ()
    allow_fp8e4nv: bool = False
    allowed_dot_input_precisions: Tuple[str] = ("ieee", )
    sanitize_overflow: bool = True

    def __post_init__(self):
        pass

    def hash(self):
        key = "_".join([f"{name}-{val}" for name, val in self.__dict__.items()])
        return hashlib.md5(key.encode("utf-8")).hexdigest()


class CPUBackend(BaseBackend):
    binary_ext = "so"

    @staticmethod
    def supports_target(target: GPUTarget):
        return target.backend == "cpu"

    def __init__(self, target: GPUTarget) -> None:
        super().__init__(target)

    def parse_options(self, opts) -> Any:
        if "instrumentation_mode" in opts:
            opts.pop("instrumentation_mode")
        args = {"arch": self.target.arch}
        args.update({k: opts[k] for k in CPUOptions.__dataclass_fields__.keys() if k in opts})
        return CPUOptions(**args)

    def get_codegen_implementation(self, options):
        codegen_fns = {"min_dot_size": lambda lhsType, rhsType: (1, 1, 1)}
        return codegen_fns

    def pack_metadata(self, metadata):
        # Note: We actually don't need any of these except for the name which is
        # used in the launch function in driver.py. Putting these in so we're
        # consistent with other backends
        return (
            metadata.num_warps,
            metadata.num_ctas,
            metadata.shared,
            metadata.cluster_dims[0],
            metadata.cluster_dims[1],
            metadata.cluster_dims[2],
            metadata.name,
        )

    # Our compilation pipeline isn't in python like nvidia or amd, no need to load
    # dialects. See `spine-triton.cc`
    def load_dialects(self, ctx):
        spacemit.load_dialects(ctx)

    @staticmethod
    def make_ttir(mod, metadata, opt):
        pm = ir.pass_manager(mod.context)
        pm.enable_debug()
        passes.common.add_inliner(pm)
        passes.ttir.add_combine(pm)
        passes.common.add_canonicalizer(pm)
        passes.ttir.add_reorder_broadcast(pm)
        passes.common.add_cse(pm)
        passes.common.add_licm(pm)
        passes.common.add_symbol_dce(pm)
        pm.run(mod, "make_ttir")
        num_threads = metadata['target'].num_threads
        attrs = []
        attrs.append(num_threads)
        arch_id = metadata['target'].arch_id
        attrs.append(arch_id)
        force_vector_interleave = metadata['target'].force_vector_interleave
        attrs.append(force_vector_interleave)
        builder = ir.builder(mod.context)
        mod.set_attr("tt.num_threads", builder.get_int32_attr(num_threads))
        mod.set_attr("tt.arch_id", builder.get_string_attr(arch_id))
        mod.set_attr("tt.force_vector_interleave", builder.get_int32_attr(force_vector_interleave))
        tt_pattern = r"tt\.func\s+public\s+@(\w+)\s*\("
        kernel_name = extract_kernel_name(tt_pattern, str(mod))
        metadata["name"] = kernel_name
        return mod

    def add_stages(self, stages, options, language):
        stages["ttir"] = lambda src, metadata: self.make_ttir(src, metadata, options)
        stages["linalgdir"] = lambda src, metadata: _optimize_linalgdir(_ttir_to_linalgdir(src, metadata))

        use_ref_pipeline = int(os.getenv("SPINE_TRITON_USE_REF_PIPELINE", "0")) > 0

        if not use_ref_pipeline:
            stages["llir"] = lambda src, metadata: _optimize_llir(_spine_mlir_linalgdir_to_llir(src, metadata))
        else:
            stages["llir"] = lambda src, metadata: _optimize_llir(_spine_mlir_linalgdir_to_llir_ref(src, metadata))

        stages["so"] = lambda src, metadata: _llir_to_so(src, metadata)

    @functools.lru_cache()
    def hash(self):
        return self.target

    # The CPU backend does not use any extra python modules, return an empty dictionary
    def get_module_map(self) -> Dict[str, ModuleType]:
        return {}


def get_cache_sizes():

    unit_map = {"k": 1024, "m": 1024**2, "g": 1024**3, "": 1}

    cache_cmd = {
        "L1": "lscpu | grep -E 'L1d|Cache|一级数据' | grep -v 'combined' | awk '{print $3, $4, $5, $6}'",
        "L2": "lscpu | grep -E 'L2|Cache|二级数据' | grep -v 'combined' | awk '{print $3, $4, $5, $6}'",
        "L3": "lscpu | grep -E 'L3|Cache|三级数据' | grep -v 'combined' | awk '{print $3, $4, $5, $6}'",
    }

    results = []
    for cache_level in ["L1", "L2", "L3"]:  # Enforce order
        try:
            output = subprocess.check_output(cache_cmd[cache_level], shell=True).decode()
        except Exception as e:
            print(f"Command execution failed: {e}")
            results.append(0)
            continue

        match = re.search(r"(\d+)\s*([KMG]?i?B)\s*\((\d+)\s*instances\)", output)
        matchNoinstances = re.search(r"(\d+)\s*([KMG]?i?B)", output)
        if match:
            total_size, unit, instances = match.groups()
        elif matchNoinstances:
            total_size, unit = matchNoinstances.groups()
            instances = 1
        else:
            results.append(0)
            continue

        unit = unit.lower().rstrip("ib")
        bytes_per_instance = (int(total_size) * unit_map[unit]) // int(instances)
        results.append(bytes_per_instance)

    return results  # Format: [L1_size, L2_size, L3_size] in bytes
