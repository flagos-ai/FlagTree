#
# Copyright 2024 Enflame. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
import re
import os
import tempfile
from pathlib import Path
from triton.backends.compiler import BaseBackend, GPUTarget
from triton.backends.enflame.backend import GCUBackend, _version_key
from triton.backends.enflame import toolkit
from triton.backends.enflame.toolkit import *

from dataclasses import dataclass
import functools
from typing import Any, Tuple
import hashlib
from triton.runtime.cache import get_cache_manager
from triton._C.libtriton import ir, passes, llvm
from triton.backends.enflame import passes as gcu_passes
from triton.backends.enflame.gcu_intrinsics import restore_intrinsics_from_placeholders as _restore_gcu_intrinsics
from typing import Dict
from types import ModuleType
from triton.runtime.errors import OutOfResources

if toolkit.get_bool_env("TRITON_GCU_COMPILE_TIME"):
    import triton.knobs as knobs
    from triton.compiler.compiler import ASTSource

    def gcu_listener(*, src, metadata, metadata_group, times, cache_hit):
        if cache_hit:
            print(f"==== Triton Cache Hit for function:  {src.name} ===")
            return
        print(f"\n=== Triton Compile Times for function: {src.name} ===")
        opt_keys = ("num_warps", "num_ctas", "num_stages", "warp_size", "vector_length", "arch")
        opt_parts = [f"{k}={metadata[k]}" for k in opt_keys if k in metadata]
        if opt_parts:
            print(f"  options: {', '.join(opt_parts)}")
        print(f"  IR init (AST->TTIR): {times.ir_initialization / 1000:.1f} ms")
        for stage_name, us in times.lowering_stages:
            print(f"  {stage_name}: {us / 1000:.1f} ms")
        print(f"  Store results: {times.store_results / 1000:.1f} ms")
        print(f"  Total lowering: {times.total_lowering / 1000:.1f} ms")
        print(f"  Total: {times.total / 1000:.1f} ms")
        if isinstance(src, ASTSource):
            print(f"\n  signature for function: {src.name}:")
            for name, ty in sorted(src.signature.items()):
                print(f"  {name}: {ty!r}")
            arg_names = src.fn.arg_names
            if src.constants:
                print(f"\n  constant args for function: {src.name}:")
                for key, val in sorted(src.constants.items()):
                    if isinstance(key, tuple) and len(key) == 1 and isinstance(key[0], int):
                        i = key[0]
                        pname = arg_names[i] if i < len(arg_names) else f"arg[{i}]"
                    else:
                        pname = str(key)
                    print(f"    {pname}: {val!r}")

    knobs.compilation.listener = gcu_listener


def _patch_kernel_for_gcuir(kernel):
    # add gpu module
    kernel = re.sub('module ([^\n]+)\n', 'module \\1\ngpu.module @triton {\n', kernel)
    pattern = r'#loc\d* = loc\(.*?\)\n'
    loc_lines = re.findall(pattern, kernel)
    kernel = re.sub(pattern, '', kernel)
    kernel = ''.join(loc_lines) + kernel.replace(pattern, '')
    kernel += '}\n'
    return kernel


def _patch_kernel_for_llir(kernel, arch):
    # strip arith.trunci's overflowFlags attribute
    kernel = re.sub(r'((?:")?arith\.trunci(?:")?\([^)]*\))\s*<\{overflowFlags\s*=\s*#arith\.overflow<[^>]*>\}>', r'\1',
                    kernel)
    kernel = re.sub(r'((?:")?arith\.trunci(?:")?\([^)]*\)\s*<\{)overflowFlags\s*=\s*#arith\.overflow<[^>]*>,\s*', r'\1',
                    kernel)
    _overflow_map = {'0': 'none', '1': 'nsw', '2': 'nuw', '3': 'nsw, nuw'}

    def _fix_overflow(m):
        val = m.group(1)
        return f'overflowFlags = #llvm.overflow<{_overflow_map.get(val, "none")}>'

    kernel = re.sub(r'overflowFlags\s*=\s*(\d+)\s*:\s*i32', _fix_overflow, kernel)
    return kernel


def make_ttir(mod, metadata, options):
    # Validate num_warps constraints
    if options.arch in ("gcu400", "gcu410") and options.num_warps > 4:
        print(mod)
        assert False, "num_warps must not exceed 4"
    elif options.arch in ("gcu300", "gcu500") and options.num_warps > 8:
        print(mod)
        assert False, "num_warps must not exceed 8"
    pm = ir.pass_manager(mod.context)
    pm.enable_debug()

    passes.common.add_inliner(pm)
    if options.arch == "gcu500":
        passes.ttir.add_rewrite_tensor_pointer(pm)
    passes.ttir.add_rewrite_tensor_descriptor_to_pointer(pm)
    passes.common.add_canonicalizer(pm)
    passes.ttir.add_combine(pm)
    passes.ttir.add_reorder_broadcast(pm)
    passes.common.add_cse(pm)
    passes.common.add_symbol_dce(pm)
    passes.ttir.add_loop_unroll(pm)
    pm.run(mod, "ttir")
    return mod


def make_ttgir(mod, metadata, options):
    pm = ir.pass_manager(mod.context)
    pm.enable_debug()
    # TritonToTritonGPU -- add GPU encoding (BlockedEncodingAttr).
    # GCU300/400 handle this later in make_gcuir; GCU500 does it here (like NV/AMD).
    if options.arch == "gcu500":
        passes.ttir.add_convert_to_ttgpuir(pm, f"gcu:{options.arch}", options.num_warps, options.warp_size,
                                           options.num_ctas)
    if options.arch in ("gcu400", "gcu410", "gcu500"):
        if options.arch == "gcu500":
            try:
                passes.ttgpuir.add_optimize_thread_locality(pm)
            except AttributeError:
                pass
    passes.ttgpuir.add_fuse_nested_loops(pm)
    passes.common.add_cse(pm)
    passes.common.add_symbol_dce(pm)
    passes.common.add_canonicalizer(pm)
    pm.run(mod, "ttgir")
    metadata["launch_cooperative_grid"] = getattr(options, "launch_cooperative_grid", False)
    metadata["cluster_dims"] = tuple(getattr(options, "cluster_dims", (1, 1, 1)))
    return mod


def make_gcuir(mod, metadata, options):
    patched_mod = _patch_kernel_for_gcuir(str(mod))
    metadata['name'] = re.search('tt.func public @(\\w+)\\(', patched_mod).group(1).strip()
    metadata['tle_raw'] = '"tle.dsl_region"' in patched_mod

    arch = options.arch if options.arch != "gcu410" else "gcu400"
    PipelineClass = toolkit.get_gcu_pipeline_class(arch)
    pm = PipelineClass()

    dump_enabled = False
    ws_inner_barrier_enabled = True
    if toolkit.get_bool_env("MLIR_ENABLE_DUMP"):
        gcu_passes.mlir.add_print_ir_after_all(pm)
        gcu_passes.mlir.add_disable_threading(pm)
        gcu_passes.mlir.add_print_ir_module_scope(pm)
        dump_enabled = True
    if toolkit.get_bool_env("MLIR_ENABLE_TIMING"):
        gcu_passes.mlir.add_timing(pm)
        gcu_passes.mlir.add_timing_display(pm, 'list')

    if options.arch == "gcu300":
        if toolkit.get_bool_env("ENABLE_I64_CHECK", True):
            gcu_passes.gcu300.add_gcu64_type_verifier(pm)

        support_stride0 = toolkit.get_bool_env("TRITON_GCU_ENABLE_STRIDE_BROADCAST")

        gcu_passes.gcu300.add_gcu_convert_triton_to_tritongpu(pm, options.num_warps, options.warp_size,
                                                              options.num_ctas, f'gcu:{options.arch}')
        gcu_passes.gcu300.add_tritongpu_remove_layout_conversions(pm)
        gcu_passes.gcu300.add_triton_gpu_to_triton_gcu(pm)
        gcu_passes.gcu300.add_convert_tensor_pointer(pm)
        gcu_passes.gcu300.add_triton_gcu_dot_layout_optimize(pm)
        gcu_passes.gcu300.add_tritongpu_remove_layout_conversions(pm)
        gcu_passes.gcu300.add_convert_triton_load_store_to_gcu_dma(pm, support_stride0)
        gcu_passes.mlir.add_canonicalize(pm)
        gcu_passes.mlir.add_loop_invariant_code_motion(pm)
        gcu_passes.gcu300.add_gcu_combine_ops(pm)
        gcu_passes.gcu300.add_gcu_triton_fusion(pm)
        gcu_passes.gcu300.add_triton_gcu_data_layout_optimize(pm)
        gcu_passes.mlir.add_canonicalize(pm)
        gcu_passes.gcu300.add_triton_gcu_pingpong(pm, options.num_stages)
        gcu_passes.gcu300.add_flatten_triton_func(pm)
        gcu_passes.gcu300.add_convert_triton_to_gcu(pm, options.vector_length)
        gcu_passes.mlir.add_cse(pm)
        gcu_passes.mlir.add_canonicalize(pm)

    elif options.arch == "gcu400" or options.arch == "gcu410":
        if toolkit.get_bool_env("ENABLE_I64_CHECK"):
            gcu_passes.gcu400.add_gcu64_type_verifier(pm)
        if metadata['tle_raw']:
            gcu_passes.gcu400.add_tle_convert_arg_to_memdesc(pm)
            gcu_passes.gcu400.add_tle_remove_redundant_copy(pm)
            gcu_passes.gcu400.add_tle_dslregion_inline(pm)
        gcu_passes.gcu400.add_gcu_convert_triton_to_tritongpu(pm, options.num_warps, options.warp_size,
                                                              options.num_ctas, f'gcu:{options.arch}')
        gcu_passes.gcu400.add_tritongpu_remove_layout_conversions(pm)
        gcu_passes.gcu400.add_tle_to_triton_gcu(pm, getattr(options, 'cluster_dims', (1, 1, 1)))
        gcu_passes.gcu400.add_triton_gpu_to_triton_gcu(pm)
        gcu_passes.gcu400.add_tritongcu_accelerate_matmul(pm)
        gcu_passes.mlir.add_cse(pm)
        gcu_passes.gcu400.add_gcu_warp_specialization(pm, options.num_stages, dump_enabled, ws_inner_barrier_enabled)
        gcu_passes.gcu400.add_triton_gcu_allocate_warp_groups(pm)
        gcu_passes.gcu400.add_convert_tensor_pointer(pm)
        gcu_passes.gcu400.add_convert_triton_load_store_to_gcu_dma(pm, options.enable_stride0, options.redundant_sip)
        gcu_passes.mlir.add_canonicalize(pm)
        gcu_passes.gcu400.add_triton_wgdot_to_gcu(pm)
        gcu_passes.gcu400.add_tritongpu_remove_layout_conversions(pm)
        gcu_passes.gcu400.add_triton_gcu_data_layout_optimize(pm)
        gcu_passes.mlir.add_loop_invariant_code_motion(pm)
        gcu_passes.gcu400.add_annotate_dot_acc_reuse(pm)
        gcu_passes.gcu400.add_gcu_combine_ops(pm)
        gcu_passes.gcu400.add_gcu_triton_fusion(pm, options.arch)
        gcu_passes.mlir.add_cse(pm)
        gcu_passes.mlir.add_canonicalize(pm)
        gcu_passes.gcu400.add_flatten_triton_func(pm)
        gcu_passes.gcu400.add_triton_gcu_local_mem_optimize(pm)
        gcu_passes.gcu400.add_gcu_tle_lower_async_load(pm)
        gcu_passes.gcu400.add_convert_triton_to_gcu(pm)
        gcu_passes.mlir.add_cse(pm)
        gcu_passes.mlir.add_canonicalize(pm)

    gcu_passes.mlir.add_print_op_generic(pm)

    return pm.run(patched_mod)


def make_llir(mod, metadata, options):
    mod = _patch_kernel_for_llir(str(mod), options.arch)
    passes = []
    if toolkit.get_bool_env("MLIR_ENABLE_DUMP"):
        passes.append('-mlir-print-ir-after-all')
    if not toolkit.get_bool_env("TRITON_DISABLE_LINE_INFO", True):
        passes.append('--ensure-debug-info-scope-on-llvm-func')
    if toolkit.get_bool_env("MLIR_ENABLE_TIMING"):
        passes.append('--mlir-timing')
        passes.append('--mlir-timing-display=list')
    passes += [
        '-insert-local-fence=arch=' + options.arch, '--convert-vector-to-scf=target-rank=1', '-lower-affine',
        '-convert-vector-to-gcu=vector-bit-width=' + str(options.vector_length * 8), '-canonicalize',
        '-convert-private-tag-to-gcu', '-convert-memref-to-gcu', '-kernel-memory-alloc=arch=' + options.arch +
        ' num-warps=' + str(options.num_warps), '-convert-warp-specialize-to-scf', '-loop-invariant-code-motion',
        '-convert-scf-to-cf', '-canonicalize', '-cse', '--symbol-dce', '-gcu-remove-transform-ir',
        '-convert-vector-to-gcu=vector-bit-width=' + str(options.vector_length * 8), '-canonicalize',
        '--expand-strided-metadata', '-lower-affine', '-canonicalize', '-cse',
        '--convert-gpu-to-gcu=chipset=' + options.arch + ' vector-bit-width=' + str(options.vector_length * 8),
        '--gcu-attach-target=arch=' + options.arch, '-convert-index-to-llvm', '-gpu-to-llvm', '-convert-llvm-to-gcu',
        '-alloca-to-entry', '-canonicalize'
    ]

    # Get some metadata
    # warp-specialization mutates num_warps
    m = re.search(r'"ttg\.total-num-warps"\s*=\s*(\d+)\s*:\s*i32', mod)
    if m:
        metadata["num_warps"] = int(m.group(1))

    ## Do nothing, until we figure out how to link .bc into triton_gcu.
    #if options.extern_libs:
    #    paths = [path for (name, path) in options.extern_libs]
    #    llvm.link_extern_libs(llvm_mod, paths)

    return toolkit.gcu_compiler_opt(mod, *passes)


def make_fatbin(mod, metadata, options):
    """Unified fatbin generation for all GCU architectures.

    For gcu500: uses EFGCU toolchain (llc + lld + bundler)
    For gcu300/400/410: uses gcu-compiler-compile
    """
    if options.arch == "gcu300":
        metadata['shared'] = int(re.search('gcu.shared_memory_size = (\\d+)', str(mod)).group(1).strip())
        local_mem_size = int(re.search('gcu.local_memory_size = (\\d+)', str(mod)).group(1).strip())
        if metadata['shared'] > options.max_shared:
            raise OutOfResources(metadata['shared'], options.max_shared, "shared memory")
        if local_mem_size > options.max_local:
            raise OutOfResources(local_mem_size, options.max_local, "local memory")
    elif options.arch == "gcu400" or options.arch == "gcu410":
        metadata['shared'] = int(re.search('gcu.shared_memory_size = (\\d+)', str(mod)).group(1).strip())
        dsm_mem_size = int(re.search('gcu.dsm_memory_size = (\\d+)', str(mod)).group(1).strip())
        if dsm_mem_size > options.max_dsm:
            raise OutOfResources(dsm_mem_size, options.max_dsm, "dsm memory")
        block_dsm_mem_size = int(match.group(1).strip()) if (match := re.search('gcu.block_dsm_memory_size = (\\d+)',
                                                                                str(mod))) else 0
        if block_dsm_mem_size > 7 * 1024 * 1024:
            raise OutOfResources(block_dsm_mem_size, 7 * 1024 * 1024, "block dsm memory")
    elif options.arch == "gcu500":
        if 'shared' not in metadata:
            metadata['shared'] = 0
        dsm_mem_size_match = re.search('gcu.dsm_memory_size = (\\d+)', str(mod))
        if dsm_mem_size_match:
            dsm_mem_size = int(dsm_mem_size_match.group(1).strip())
            if dsm_mem_size > options.max_dsm:
                raise OutOfResources(dsm_mem_size, options.max_dsm, "dsm memory")

    if options.arch == "gcu500":
        return _make_fatbin_gcu500(str(mod), metadata, options)
    else:
        if metadata['tle_raw']:
            mod = _restore_gcu_intrinsics(str(mod))
        with tempfile.TemporaryDirectory() as tmpdir:
            bin = os.path.join(tmpdir, "kernel.fatbin")
            compile_args = [
                "--device-only", "--is-triton-backend", f"--arch={options.arch}", f"--toolkit-path={toolkit.datadir}",
                f"--output={bin}"
            ]
            toolkit.compile(mod, *compile_args)
            with open(bin, "rb") as f:
                return f.read()


def _make_fatbin_gcu500(src_str, metadata, options):
    """LLVM IR text -> fatbin via EFGCU llc + lld + bundler.

    Pipeline (pure EFGCU, no gcu-compiler):
    1. Patch LLIR for TOPS LLVM18 compatibility
    2. Call toolkit.compile_llir_to_fatbin_gcu500 (llc + lld + bundler)
    Saves .ll and .s alongside fatbin in the cache directory.

    Note: metadata["shared"] should have been set by _make_llir_gcu500 already.
    All f32 intrinsics (maxnumf/minnumf/exp/exp2) are handled by C++ MLIR
    patterns in ConvertTritonGPUToEFGCU for both scalar and tensor types.
    Vector scalarization is unnecessary because the MLIR type converter
    produces struct-of-scalars, not LLVM vectors.
    """
    kernel_name = metadata.get("name", "kernel") if isinstance(metadata, dict) else "kernel"
    gcu500 = toolkit._load_gcu_opt_module('gcu500')
    patched = gcu500.patch_llir(src_str, kernel_name)

    cache = get_cache_manager(metadata.get("hash"))
    cache.put(patched, f"{kernel_name}.ll")

    result = toolkit.compile_llir_to_fatbin_gcu500(patched, kernel_name)

    if result["asm"]:
        cache.put(result["asm"], f"{kernel_name}.s")

    return result["fatbin"]


def _make_llir_gcu500(mod, metadata, options):
    """GCU500 LLIR stage: lower TTGIR directly to EFGCU LLVM IR.

    Uses the _triton_gcu500 nanobind extension (in-process pipeline).
    The pass pipeline mirrors the former triton-gcu500-opt pass list.
    """
    gcu500 = toolkit._load_gcu_opt_module('gcu500')

    ttgir_str = str(mod)
    m = re.search(r'tt\.func public @(\w+)\s*\(', ttgir_str)
    if m and 'name' not in metadata:
        metadata['name'] = m.group(1).strip()

    pipeline = gcu500.Pipeline()
    pipeline.add_allocate_gcu_shared_memory()
    pipeline.add_convert_scf_to_cf()
    pipeline.add_convert_index_to_llvm()
    pipeline.add_convert_triton_gpu_to_efgcu()
    pipeline.add_canonicalizer()
    pipeline.add_cse()
    pipeline.add_convert_cf_to_llvm()
    pipeline.add_convert_arith_to_llvm()
    pipeline.add_canonicalizer()
    pipeline.add_cse()
    pipeline.add_symbol_dce()

    result_mlir = pipeline.run(ttgir_str)
    kernel_name = metadata.get("name", "kernel")
    cache = get_cache_manager(metadata.get("hash"))
    cache.put(result_mlir, f"{kernel_name}.efvm.mlir")

    shared_match = re.search(r'ttg\.shared\s*=\s*(\d+)', result_mlir)
    if shared_match:
        metadata['shared'] = int(shared_match.group(1))

    gcu_shared_match = re.search(r'gcu\.shared_memory_size\s*=\s*(\d+)', result_mlir)
    if gcu_shared_match:
        gcu_shared = int(gcu_shared_match.group(1))
        metadata['shared'] = max(metadata.get('shared', 0), gcu_shared)

    llir = pipeline.translate_to_llvmir(result_mlir)
    return llir


@functools.lru_cache(None)
def file_hash(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def min_dot_size(target: GPUTarget):
    return lambda lhsType, rhsType: (1, 1, 1)


@dataclass()
class GCUOptions:
    num_warps: int = 4
    warp_size: int = 1
    num_ctas: int = 1
    num_stages: int = 3
    vector_length: int = 512
    debug: bool = False
    cluster_dims: tuple = (1, 1, 1)
    allow_fp8e4nv: bool = False
    allow_fp8e4b15: bool = False
    supported_fp8_dtypes: Tuple[str] = ()
    deprecated_fp8_dot_operand_dtypes: Tuple[str] = ()
    default_dot_input_precision: str = "ieee"
    allowed_dot_input_precisions: Tuple[str] = ("ieee", )
    backend_name: str = 'gcu'
    max_num_imprecise_acc_default: int = 0
    enable_fp_fusion: bool = True
    enable_stride0: bool = False
    redundant_sip: bool = False
    launch_cooperative_grid: bool = False
    extern_libs: dict = None
    sanitize_overflow: bool = False
    num_buffers_warp_spec: int = 0
    num_consumer_groups: int = 0
    reg_dec_producer: int = 0
    reg_inc_consumer: int = 0
    arch: str = None
    max_shared: int = 0
    max_local: int = 0
    max_dsm: int = 0
    instrumentation_mode: str = ""
    launch_pdl: bool = False

    def __post_init__(self):
        architecture = GCUBackend().get_architecture_descriptor()
        self.arch = "gcu" + str(architecture['version'])
        assert self.num_warps > 0 and (self.num_warps & (self.num_warps - 1)) == 0, \
               "num_warps must be a power of 2"
        if self.arch == "gcu400" or self.arch == "gcu410":
            self.vector_length = 2048
            self.allow_fp8e4nv = True
            self.allowed_dot_input_precisions: Tuple[str] = ("tf32", "tf32x3", "ieee")
            self.max_num_imprecise_acc_default = 2**30
            self.supported_fp8_dtypes: Tuple[str] = ("fp8e4nv", "fp8e5")
            self.deprecated_fp8_dot_operand_dtypes: Tuple[str] = ()
            self.max_dsm = 896 * 1024
            # Handle exceptions for the num_warps parameter
            if self.num_warps > 4:
                self.num_warps = 4
                if toolkit.get_bool_env("TRITON_GCU_DEBUG"):
                    print(f"warning: num_warps {self.num_warps}, "
                          f"exceeds triton-gcu limit (4). "
                          f"Triton-gcu automatically resets num_warps to 4!")
        elif self.arch == "gcu500":
            self.vector_length = 2048
            self.allow_fp8e4nv = True
            self.allowed_dot_input_precisions: Tuple[str] = ("tf32", "tf32x3", "ieee")
            self.max_num_imprecise_acc_default = 2**30
            self.supported_fp8_dtypes: Tuple[str] = ("fp8e4nv", "fp8e5")
            self.deprecated_fp8_dot_operand_dtypes: Tuple[str] = ()
            self.max_dsm = 312 * 1024  # Increased DSM for SIMT
            # GCU500 (5.0): 1 warp = 128 threads (equivalent to 128 CUDA threads per warp)
            self.warp_size = 128  # SIMT warp size (5.0: 128 threads per warp)
            self.simt_lane_size = 128  # SIMT lane size (same as warp size for GCU500)
            # Handle exceptions for the num_warps parameter (SIMT mode supports more warps)
            if self.num_warps > 8:
                self.num_warps = 8
                if toolkit.get_bool_env("TRITON_GCU_DEBUG"):
                    print(f"warning: num_warps {self.num_warps}, "
                          f"exceeds triton-gcu500 limit (8). "
                          f"Triton-gcu500 automatically resets num_warps to 8!")
        elif self.arch == "gcu300":
            self.max_local = (1024 * (1024 + 512) - 512)
            self.max_shared = 1024 * 1024 * 8 * self.num_warps
            # Handle exceptions for the num_warps parameter
            if self.num_warps > 8:
                self.num_warps = 8
                if toolkit.get_bool_env("TRITON_GCU_DEBUG"):
                    print(f"warning: num_warps {self.num_warps}, "
                          f"exceeds triton-gcu limit (8). "
                          f"Triton-gcu automatically resets num_warps to 8!")

        ## register the libdevice
        default_libdir = Path(__file__).parent / 'lib'
        extern_libs = {} if self.extern_libs is None else dict(self.extern_libs)
        if not extern_libs.get('libdevice', None):
            extern_libs['libdevice'] = os.getenv("TRITON_LIBDEVICE_PATH", str(default_libdir / 'libdevice.bc'))
        object.__setattr__(self, 'extern_libs', tuple(extern_libs.items()))

        pass

    def hash(self):
        ## Restore the code below, when we have libdevice.bc
        #hash_dict = dict(self.__dict__)
        #hash_dict["extern_libs"] = tuple((k, file_hash(v)) for k, v in sorted(hash_dict["extern_libs"]))
        version_key = _version_key()
        key = version_key + '_' + '_'.join([f'{name}-{val}' for name, val in self.__dict__.items()])
        return hashlib.sha256(key.encode("utf-8")).hexdigest()


class _GCUBackend(BaseBackend):

    def __init__(self, target: GPUTarget) -> None:
        super().__init__(target)
        self._backend = GCUBackend()
        self.binary_ext = "fatbin"

    @staticmethod
    def supports_target(target: GPUTarget):
        return target.backend == 'gcu'

    def get_target_name(self, options) -> str:
        return f"gcu:{options.arch}"

    def parse_options(self, opts) -> Any:
        args = {k: opts[k] for k in GCUOptions.__dataclass_fields__.keys() if k in opts}

        if "enable_fp_fusion" not in opts:
            args["enable_fp_fusion"] = toolkit.get_bool_env("TRITON_DEFAULT_FP_FUSION")

        if "ENABLE_STRIDE_GATHER" in opts:
            args["enable_stride0"] = opts["ENABLE_STRIDE_GATHER"]
        else:
            args["enable_stride0"] = toolkit.get_bool_env("TRITON_GCU_ENABLE_STRIDE_GATHER")
        args.update({k: opts[k] for k in GCUOptions.__dataclass_fields__.keys() if k in opts})

        return GCUOptions(**args)

    def load_dialects(self, ctx):
        self._backend.load_dialects(ctx)

    @functools.lru_cache()
    def hash(self):
        return self._backend.hash()

    def get_architecture_descriptor(self, **kwargs):
        return self._backend.get_architecture_descriptor(**kwargs)

    def get_codegen_implementation(self, options):
        codegen_fns = {"min_dot_size": min_dot_size(self.target)}
        return codegen_fns

    def pack_metadata(self, metadata):
        return (
            metadata.num_warps,
            metadata.num_ctas,
            metadata.shared,
            metadata.cluster_dims[0],
            metadata.cluster_dims[1],
            metadata.cluster_dims[2],
            int(getattr(metadata, "launch_cooperative_grid", False)),
        )

    def add_stages(self, stages, options, language):
        stages["ttir"] = lambda src, metadata: make_ttir(src, metadata, options)
        stages["ttgir"] = lambda src, metadata: make_ttgir(src, metadata, options)
        if options.arch == "gcu500":
            stages["llir"] = lambda src, metadata: _make_llir_gcu500(src, metadata, options)
        else:
            stages["gcuir"] = lambda src, metadata: make_gcuir(src, metadata, options)
            stages["llir"] = lambda src, metadata: make_llir(src, metadata, options)
        stages["fatbin"] = lambda src, metadata: make_fatbin(src, metadata, options)

    def get_module_map(self) -> Dict[str, ModuleType]:
        return self._backend.get_module_map()
