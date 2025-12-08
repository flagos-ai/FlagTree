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
from triton.backends.enflame.backend import GCUBackend
from triton.backends.enflame import toolkit
from triton.backends.enflame.toolkit import *
from dataclasses import dataclass
import functools
from typing import Any, Tuple
import hashlib
from triton._C.libtriton import ir, passes, llvm
from typing import Dict
from types import ModuleType


def _patch_kernel(kernel):
    # add gpu module
    kernel = re.sub('module ([^\n]+)\n', 'module \\1\ngpu.module @triton {\n', kernel)
    pattern = r'#loc\d* = loc\(.*?\)\n'
    loc_lines = re.findall(pattern, kernel)
    kernel = re.sub(pattern, '', kernel)
    kernel = ''.join(loc_lines) + kernel.replace(pattern, '')
    kernel += '}\n'
    return kernel


def make_ttir(mod, metadata, options):
    pm = ir.pass_manager(mod.context)
    pm.enable_debug()
    passes.common.add_inliner(pm)
    #passes.ttir.add_rewrite_tensor_pointer(pm)
    passes.ttir.add_combine(pm)
    passes.common.add_canonicalizer(pm)
    passes.ttir.add_reorder_broadcast(pm)
    passes.common.add_cse(pm)
    passes.common.add_licm(pm)
    passes.common.add_symbol_dce(pm)
    pm.run(mod)
    return mod


def make_ttgir(mod, metadata, options):
    pm = ir.pass_manager(mod.context)
    pm.enable_debug()
    passes.ttir.add_convert_to_ttgpuir(pm, f"gcu:{options.arch}", options.num_warps, options.warp_size,
                                       options.num_ctas)
    # passes.ttgpuir.add_coalesce(pm)
    passes.ttgpuir.add_remove_layout_conversions(pm)
    passes.ttgpuir.add_optimize_thread_locality(pm)
    passes.common.add_cse(pm)
    passes.common.add_symbol_dce(pm)
    passes.common.add_canonicalizer(pm)
    pm.run(mod)
    return mod


def make_gcuir(mod, metadata, options):
    patched_mod = _patch_kernel(str(mod))
    metadata['name'] = re.search('tt.func public @(\\w+)\\(', patched_mod).group(1).strip()
    passes = []
    if toolkit.get_bool_env("MLIR_ENABLE_DUMP"):
        passes.append('-mlir-print-ir-after-all')
    if toolkit.get_bool_env("MLIR_ENABLE_TIMING"):
        passes.append('--mlir-timing')
        passes.append('--mlir-timing-display=list')
    if options.arch == "gcu300":
        passes += [
            # '-mlir-disable-threading',
            # '-mlir-print-ir-module-scope',
            '-triton-gpu-to-triton-gcu', '-gcu64-type-verifier', '-convert-tensor-pointer',
            '-triton-gcu-dot-layout-optimize', '-tritongpu-remove-layout-conversions',
            '-convert-triton-load-store-to-gcu-dma', '-canonicalize', '-loop-invariant-code-motion',
            '-gcu-triton-fusion', '-triton-gcu-data-layout-optimize', '-canonicalize',
            '-triton-gcu-pingpong=' + 'num_stages=' + str(options.num_stages), '-flatten-triton-func',
            '-convert-triton-to-gcu=' + 'vector-length=' + str(options.vector_length), '-cse', '-canonicalize'
        ]
    elif options.arch == "gcu400" or options.arch == "gcu410":
        if toolkit.get_bool_env("ENABLE_I64_CHECK"):
            passes.append('-gcu64-type-verifier')
        LOAD_STORE_TO_DMA_PASS = '-convert-triton-load-store-to-gcu-dma'
        if toolkit.get_bool_env("TRITON_GCU_ENABLE_STRIDE_BROADCAST"):
            LOAD_STORE_TO_DMA_PASS += '=support_stride0=true'
        passes += [
            # '-mlir-disable-threading',
            # '-mlir-print-ir-module-scope',
            '-triton-gpu-to-triton-gcu', '-convert-tensor-pointer', LOAD_STORE_TO_DMA_PASS, '-canonicalize',
            '-loop-invariant-code-motion', '-gcu-combine-ops', '-gcu-triton-fusion=arch=' + options.arch,
            '-canonicalize', '-flatten-triton-func', '-convert-triton-to-gcu', '-cse', '-canonicalize'
        ]
    return toolkit.triton_gcu_opt(patched_mod, *passes, arch=options.arch)


def make_llir(mod, metadata, options):
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
        '-convert-memref-to-gcu', '-kernel-memory-alloc=arch=' + options.arch + ' num-warps=' + str(options.num_warps),
        '-loop-invariant-code-motion', '-convert-scf-to-cf', '-canonicalize', '-cse', '--symbol-dce',
        '-gcu-remove-transform-ir', '-convert-vector-to-gcu=vector-bit-width=' + str(options.vector_length * 8),
        '-canonicalize',
        '--convert-gpu-to-gcu=chipset=' + options.arch + ' vector-bit-width=' + str(options.vector_length * 8),
        '--gcu-attach-target=arch=' + options.arch, '-convert-index-to-llvm', '-gpu-to-llvm', '-convert-llvm-to-gcu',
        '-alloca-to-entry', '-canonicalize'
    ]

    ## Do nothing, until we figure out how to link .bc into triton_gcu.
    #if options.extern_libs:
    #    paths = [path for (name, path) in options.extern_libs]
    #    llvm.link_extern_libs(llvm_mod, paths)

    return toolkit.gcu_compiler_opt(mod, *passes)


def make_fatbin(mod, metadata, options):
    metadata['shared'] = int(re.search('gcu.shared_memory_size = (\\d+)', str(mod)).group(1).strip())
    with tempfile.TemporaryDirectory() as tmpdir:
        bin = os.path.join(tmpdir, "kernel.fatbin")
        toolkit.compile(mod, "--device-only", "--is-triton-backend", f"--arch={options.arch}",
                        f"--toolkit-path={datadir}", f"--output={bin}")
        with open(bin, "rb") as f:
            return f.read()


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
    arch: str = "gcu300"
    vector_length: int = 512
    debug: bool = False
    cluster_dims: tuple = (1, 1, 1)
    allow_fp8e4nv: bool = False
    allow_fp8e4b15: bool = False
    supported_fp8_dtypes: Tuple[str] = ()
    deprecated_fp8_dtypes: Tuple[str] = ()
    default_dot_input_precision: str = "ieee"
    allowed_dot_input_precisions: Tuple[str] = ("ieee", )
    backend_name: str = 'gcu'
    max_num_imprecise_acc_default: int = 0
    enable_fp_fusion: bool = True
    launch_cooperative_grid: bool = False
    extern_libs: dict = None
    sanitize_overflow: bool = False
    num_buffers_warp_spec: int = 0
    num_consumer_groups: int = 0
    reg_dec_producer: int = 0
    reg_inc_consumer: int = 0
    arch: str = None

    def __post_init__(self):
        architecture = GCUBackend().get_architecture_descriptor()
        self.arch = "gcu" + str(architecture['version'])
        assert self.num_warps > 0 and (self.num_warps & (self.num_warps - 1)) == 0, \
               "num_warps must be a power of 2"
        if self.arch == "gcu400" or self.arch == "gcu410":
            assert self.num_warps <= 4, "num_warps must not exceed 4"
            self.vector_length = 2048
            self.allow_fp8e4nv = True
            self.allowed_dot_input_precisions: Tuple[str] = ("tf32", "tf32x3", "ieee")
            self.max_num_imprecise_acc_default = 2**30
            self.supported_fp8_dtypes: Tuple[str] = ("fp8e4nv", "fp8e5")
            self.deprecated_fp8_dtypes: Tuple[str] = ()
            self.sanitize_overflow: bool = True

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

        key = '_'.join([f'{name}-{val}' for name, val in self.__dict__.items()])
        return hashlib.sha256(key.encode("utf-8")).hexdigest()


class _GCUBackend(BaseBackend):

    def __init__(self, target: GPUTarget) -> None:
        super().__init__(target)
        self._backend = GCUBackend()
        self.binary_ext = "fatbin"

    @staticmethod
    def supports_target(target: GPUTarget):
        return target.backend == 'gcu'

    def parse_options(self, opts) -> Any:
        args = {k: opts[k] for k in GCUOptions.__dataclass_fields__.keys() if k in opts}

        if "enable_fp_fusion" not in opts:
            args["enable_fp_fusion"] = os.getenv("TRITON_DEFAULT_FP_FUSION", "1") == "1"

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
        )

    def add_stages(self, stages, options):
        stages["ttir"] = lambda src, metadata: make_ttir(src, metadata, options)
        stages["ttgir"] = lambda src, metadata: make_ttgir(src, metadata, options)
        stages["gcuir"] = lambda src, metadata: make_gcuir(src, metadata, options)
        stages["llir"] = lambda src, metadata: make_llir(src, metadata, options)
        stages["fatbin"] = lambda src, metadata: make_fatbin(src, metadata, options)

    def get_module_map(self) -> Dict[str, ModuleType]:
        return self._backend.get_module_map()
