# Copyright 2025-     FlagOS Contributors
# SPDX-License-Identifier: MIT
"""EVAS frontend stages and SDK code generation."""

import hashlib
import os
import tempfile
import importlib.util
from pathlib import Path
import re
import shutil
import functools
import json
from dataclasses import dataclass
from typing import Any, Tuple, Dict
from types import ModuleType

from triton.runtime.cache import get_dump_manager, get_cache_manager
from triton.backends.compiler import BaseBackend, GPUTarget
from triton._C import libtriton
from triton._C.libtriton import ir, passes
from .utils import run_command, _cache_key
from .arg_access import argument_access
from .local_shards import LOCAL_ACCESS_VERSION


def get_sys_path(bin_name: str):
    path = shutil.which(bin_name)
    if path is None:
        raise RuntimeError(bin_name + " not found")
    return path


def _make_ttir(mod, metadata, opt):
    ttir_code = str(mod)
    pattern = r"tt\.func public @(.*?)(?=\()"
    matches = re.findall(pattern, ttir_code)
    metadata["name"] = matches[0]
    pm = ir.pass_manager(mod.context)
    _configure_pass_debug(pm, metadata, "ttir_ir")
    passes.common.add_inliner(pm)
    passes.ttir.add_rewrite_tensor_pointer(pm)  ## triton 3.7 deleted
    passes.ttir.add_rewrite_tensor_descriptor_to_pointer(pm)
    passes.common.add_canonicalizer(pm)
    passes.ttir.add_combine(pm)
    passes.ttir.add_reorder_broadcast(pm)
    passes.common.add_cse(pm)
    passes.ttir.add_triton_licm(pm)
    passes.common.add_symbol_dce(pm)
    passes.ttir.add_loop_unroll(pm)
    pm.run(mod, "make_ttir")
    if opt.auto_dual_die:
        metadata["buffer_access"] = argument_access(mod)
    return mod.str()


def _configure_pass_debug(pm, metadata, stage):
    pm.enable_debug()
    if os.environ.get("TRITON_KERNEL_DUMP", "0") == "1":
        directory = Path(get_dump_manager(metadata["hash"]).cache_dir) / stage
        directory.mkdir(parents=True, exist_ok=True)
        libtriton.evas.enable_ir_dump(pm, str(directory))


def _ttir_to_ttsharedir(ttir_code, metadata):
    # IRSource skips _make_ttir and supplies the parsed MLIR module directly.
    # Preserve that IR without rerunning AST-only passes, and record its entry.
    if not isinstance(ttir_code, str):
        metadata.setdefault("name", ttir_code.get_entry_func_name())
        if metadata.get("auto_dual_die", False):
            metadata["buffer_access"] = argument_access(ttir_code)
        ttir_code = ttir_code.str()
    context = ir.context()
    ir.load_dialects(context)
    libtriton.evas.load_dialects(context)
    module = libtriton.evas.parse_module(ttir_code, context)
    access = (json.loads(libtriton.evas.export_local_access(module)) if metadata.get("local_shards", False) else None)
    pm = ir.pass_manager(context)
    _configure_pass_debug(pm, metadata, "ttshared_ir")
    libtriton.evas.add_triton_to_evas_pipeline(pm)
    pm.run(module, "triton_to_evas")
    lowered = libtriton.evas.module_to_string(module)
    if access is not None:
        if access.get("version") != LOCAL_ACCESS_VERSION:
            raise ValueError("EVAS incompatible access graph version; rebuild the EVAS frontend")
        if not metadata.get("buffer_access", {}).get("supported", False):
            access.update(supported=False, reason="argument effects are not verified")
        access["abi"] = bool(access.get("supported"))
        metadata["local_access"] = access
    return lowered


def _rename_entry_func_to_kernel(ttsharedir: str, metadata) -> str:
    original_name = metadata.get("name")
    if original_name == "kernel":
        return ttsharedir
    if original_name:
        ttsharedir = re.sub(
            rf'(\bsym_name\s*=\s*)"{re.escape(original_name)}"',
            r'\1"kernel"',
            ttsharedir,
            count=1,
        )
        ttsharedir = re.sub(
            rf'(\bfunc\.func\s+(?:public\s+|private\s+)?@){re.escape(original_name)}\b',
            r'\1kernel',
            ttsharedir,
            count=1,
        )
    if 'sym_name = "kernel"' not in ttsharedir and "func.func @kernel" not in ttsharedir:
        ttsharedir = re.sub(r'(\bsym_name\s*=\s*)"[^"]+"', r'\1"kernel"', ttsharedir, count=1)
    return ttsharedir


def pack_header(ccsrc: str, metadata):

    def _add_device_to_func_def(ccsrc: str) -> str:
        # Find all function definitions that are not "kernel" and add __device__ prefix
        # This regex matches function definitions like "void func_name(" or "int func_name(" etc.
        # but excludes the kernel function
        # Add __device__ only to function definitions that start with "MCU ALWAYS_INLINE" and are not kernel
        pattern = r'(?!.*__device__)(\bMCU.+\([^;{]*\)\s*{)'
        ccsrc = re.sub(pattern,  # Match MCU ALWAYS_INLINE function def but not kernel
                       r'__device__ \1', ccsrc, flags=re.MULTILINE)
        return ccsrc

    def _extract_kernel_function_args(ccsrc: str):
        """
        Extract the kernel function arguments (names and types) from C/C++ code.

        Args:
            cc_code (str): The C/C++ code string

        Returns:
            dict: A dictionary where keys are argument names and values are their types
                Returns None if no kernel function is found
        """
        # Pattern to match the kernel function declaration
        # Looks for "void kernel(" or "MCU void kernel(" or similar patterns
        kernel_pattern = r'(?:__global__\s+)?(?:extern\s+"C"\s+)?(?:MCU\s+)?(?:__device__\s+)?void\s+kernel\s*\((.*?)\)'

        # Find the kernel function
        kernel_match = re.search(kernel_pattern, ccsrc, re.DOTALL)

        if not kernel_match:
            raise RuntimeError("No kernel function found in the provided C/C++ code.")

        # Extract the arguments section
        args_section = kernel_match.group(1)

        # Handle empty arguments case
        if args_section.strip() == "":
            raise RuntimeError("Kernel function has no arguments.")

        # Split multiple arguments
        args = []

        # Handle complex arguments with nested commas (like template parameters)
        depth = 0
        current_arg = ""

        for char in args_section:
            if char == ',' and depth == 0:
                args.append(current_arg.strip())
                current_arg = ""
            else:
                if char == '<':
                    depth += 1
                elif char == '>':
                    depth -= 1
                current_arg += char

        # Add the last argument
        if current_arg.strip():
            args.append(current_arg.strip())

        # Process each argument to extract type and name
        arg_dict = []

        for arg in args:
            # Pattern to match type and name: everything before the last word is the type
            arg = arg.strip()

            # Skip empty arguments
            if not arg:
                continue

            # Handle pointers and references in names
            name_pattern = r'(\w+)(?:\s*\[\s*\w*\s*\])*\s*$'
            name_match = re.search(name_pattern, arg)

            if name_match:
                name = name_match.group(1)
                # Extract the type by removing the name from the end
                type_part = arg[:arg.rfind(name)].strip()

                # Clean up any trailing spaces, asterisks should stick with the type
                if type_part.endswith('*'):
                    while type_part.endswith(' *'):
                        type_part = type_part[:-2] + '*'

                arg_dict.append({"name": name, "type": type_part})
            else:
                # For arguments that don't match the expected pattern
                raise ValueError(f"Could not parse argument: '{arg}'")

        return arg_dict

    def _delete_MCU(ccsrc: str) -> str:
        # Replace all occurrences of "MCU" with ""
        return re.sub(r'\bMCU\s*', '', ccsrc)

    def _replace_wt_data_setction(ccsrc: str) -> str:
        # Replace all occurrences of ".wtdata" with ".data"
        return ccsrc.replace('.wtdata', '.data')

    kernel_name = metadata["name"]
    ccsrc = _add_device_to_func_def(ccsrc)
    args = _extract_kernel_function_args(ccsrc)
    ccsrc, replacements = re.subn(r'\bvoid\s+kernel\s*\(', 'void kernel_block(', ccsrc, count=1)
    if replacements != 1:
        raise RuntimeError("Could not rename the generated kernel function.")
    ccsrc = _delete_MCU(ccsrc)
    ccsrc = _replace_wt_data_setction(ccsrc)
    return f"""
#ifdef __AC_DEVICE_COMPILE__
#include "visa_defs_v1.h"
#include "visa.h"
#include "operator/te/te.h"
#include "target/evamind/evamind-v1/visa_evamind_me.h"
#include "target/evamind/visa_evamind_te.h"
#include "target/evamind/visa_evamind_vcall.h"
#include "util/tuple.h"
#include "visa_matrix.h"
#include "fusion_op/gs_p_masked.h"

#if __has_include(<npurt.h>)
#include <npurt.h>
#else
#include <devrt.h>
#endif
using namespace visa::te;
using namespace visa::matrix;
using namespace visa;
#endif

#include <stdint.h>
#include <helper_hpe.h>
#ifndef INFINITY
#define INFINITY (__builtin_inff())
#endif
{ccsrc}

extern "C" __global__ void {kernel_name}({', '.join([f"{arg['type']} {arg['name']}" for arg in args])}) {{
    kernel_block({', '.join([arg['name'] for arg in args])});
}}
"""


def cc_to_kernel_elf_command(src_path: str, dst_path: str):
    visa_dir = os.environ.get("VISA_PATH")
    if not visa_dir:
        raise EnvironmentError("Source the SDK environment to set VISA_PATH before compiling device code.")
    hgss_root = os.environ.get("RISCV_HGSS")
    return [
        "evcc", "-O2", "-Wall", "-Werror", "-std=c++17", "-fno-omit-frame-pointer", "-Wno-unused-command-line-argument",
        "-Wno-implicitly-unsigned-literal", "-Wno-unused-variable", "-ftls-model=local-exec", "-fno-common",
        "-ffast-math", "-ffunction-sections", "-fdata-sections", "-Wl,--gc-sections", "-D__riscv_v_vlen=1024",
        "-D__evamind_v1", "-D__EVAMIND_DEVICE_V1", "--ac-device-only", "-emit-device-elf", "-x", "ac", "-o", dst_path,
        src_path, "-I", visa_dir + "/include", "-I", visa_dir + "/include/target/evamind/evamind-v1",
        *(["-I", os.path.join(hgss_root, "include"), "-I",
           os.path.join(hgss_root, "dev", "include", "devrt")] if hgss_root else []),
        *(["-I", os.path.join(hgss_root, "misc", "include")] if hgss_root else []), "-I",
        os.path.join(os.path.dirname(__file__), "include/evas")
    ]


def compile_elf(cc: bytes, metadata):
    dst_name = "kernel"
    with tempfile.TemporaryDirectory() as tmpdir:
        src_path = os.path.join(tmpdir, dst_name + ".cc")
        Path(src_path).write_bytes(cc)
        dst_path = os.path.join(tmpdir, dst_name + ".elf")
        command = cc_to_kernel_elf_command(src_path, dst_path)
        run_command(command)
        return Path(dst_path).read_bytes()


def epu_tool_paths():
    # A PATH override selects the SDK/compiler installation as a pair.
    # Never silently mix a partial PATH installation with a Python package.
    names = ("fc-opt", "fc-translate")
    found = [shutil.which(name) for name in names]
    if any(found):
        if not all(found):
            raise FileNotFoundError("Put both fc-opt and fc-translate from the same installation on PATH.")
        paths = [Path(path).resolve() for path in found]
        if paths[0].parent != paths[1].parent:
            raise ValueError("fc-opt and fc-translate on PATH must resolve to the same installation directory.")
    else:
        try:
            spec = importlib.util.find_spec("jax_plugins.epu")
        except ModuleNotFoundError:
            spec = None
        if spec is None or spec.origin is None:
            raise FileNotFoundError("Put fc-opt and fc-translate on PATH or install the SDK's jax_plugins.epu package.")
        epu_dir = Path(spec.origin).parent
        paths = [epu_dir / "fc-opt", epu_dir / "fc-translate"]
    for path in paths:
        if not path.is_file() or not os.access(path, os.X_OK):
            raise FileNotFoundError(f"EVAS compiler tool is not executable: {path}")
    return tuple(str(path.resolve()) for path in paths)


@functools.lru_cache(maxsize=32)
def _tool_digest(path, size, mtime_ns, ctime_ns):
    # Stat fields invalidate the digest when a developer replaces a binary.
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def epu_toolchain_key():
    identities = []
    for path in epu_tool_paths():
        stat = os.stat(path)
        identities.append(path + ":" + _tool_digest(path, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns))
    return hashlib.sha256("\n".join(identities).encode()).hexdigest()


def ev_opt_command(src: str, dst: str, metadata):
    ev_opt, _ = epu_tool_paths()
    hash_suffix = metadata["hash"][:5]
    incbin_dir = get_cache_manager(_cache_key(f"incbin/incbin_{hash_suffix}")).cache_dir
    pipeline_options = ("simt-level=simt "
                        "rank-collapse-level=necessary "
                        "simplify-line-break=true "
                        "allow-return-allocs-from-loops=true "
                        f"visa-incbin-dir={incbin_dir}")
    if metadata.get("auto_dual_die", False):
        pipeline_options += " auto-dual-die=true"
    if metadata.get("local_access", {}).get("abi", False):
        pipeline_options += " local-shards=true"
    ev_opt_command = [
        ev_opt,
        f"--evofc-flagtree-evas-codegen-pipeline={pipeline_options}",
        src,
        "-o",
        dst,
    ]
    enable_ir_dump = os.environ.get("TRITON_KERNEL_DUMP", "0") == "1"
    if enable_ir_dump:
        fn_dump_manager = get_dump_manager(metadata["hash"])
        ev_opt_command.extend([
            "--mlir-print-ir-after-all",
            "--mlir-print-ir-tree-dir=" + fn_dump_manager.cache_dir + "/evopt_ir",
        ])
    return ev_opt_command


def ev_trans_command(src: str, dst: str):
    _, ev_trans = epu_tool_paths()
    ev_trans_command = [ev_trans, "-mlir-to-cpp", src, "-o", dst]
    return ev_trans_command


def _ttshared_to_cc(ttsharedir: str, metadata):
    with tempfile.TemporaryDirectory() as tmpdir:
        src_path = os.path.join(tmpdir, "ttshared.mlir")
        Path(src_path).write_text(_rename_entry_func_to_kernel(ttsharedir, metadata))
        emitc_path = os.path.join(tmpdir, "emitc.mlir")
        cc_path = os.path.join(tmpdir, "dst.cc")

        run_command(ev_opt_command(src_path, emitc_path, metadata))
        run_command(ev_trans_command(emitc_path, cc_path))
        return pack_header(Path(cc_path).read_text(), metadata)


def _cc_to_evbin(ccsrc: str, metadata):
    return compile_elf(ccsrc.encode("utf-8"), metadata)


@dataclass(frozen=True)
class EvasOptions:
    debug: bool = False
    arch: str = None
    num_warps: int = 0
    num_ctas: int = 0
    num_stages: int = 1
    instrumentation_mode: str = ""
    one_tile_per_cta: bool = False
    enable_warp_specialization: bool = False
    enable_fp_fusion: bool = False
    extern_libs = None
    cluster_dims: tuple = (1, 1, 1)
    enable_dual_die: bool = False
    # Automatic scheduling keeps the ordinary kernel's global pointer/PID ABI.
    auto_dual_die: bool = False
    local_shards: bool = False
    toolchain_key: str = ""
    buffer_bridge_key: str = ""
    shared: bool = False
    allow_fp8e4nv: bool = True
    allowed_dot_input_precisions: Tuple[str] = ("ieee", )
    max_num_imprecise_acc_default: int = 0
    # TODO: just for lowering to extern elementwise op (libdevice.py)
    # some operations like libdevice.rsqrt and so on
    backend_name = "evas"
    to_ttsharedir: bool = False
    sanitize_overflow: bool = True
    # Permit FP8 IR generation for debugging; lowering support is incomplete.
    supported_fp8_dtypes: Tuple[str] = ("fp8e4nv", "fp8e5")
    ir_override: str = None

    def __post_init__(self):
        if not isinstance(self.enable_dual_die, bool):
            raise TypeError("EVAS enable_dual_die must be a bool")
        if not isinstance(self.auto_dual_die, bool):
            raise TypeError("EVAS auto_dual_die must be a bool")
        if self.auto_dual_die and not self.enable_dual_die:
            raise ValueError("EVAS auto_dual_die requires a dual-die launch")
        if not isinstance(self.local_shards, bool):
            raise TypeError("EVAS local_shards must be a bool")
        if self.local_shards and not self.auto_dual_die:
            raise ValueError("EVAS local_shards requires auto_dual_die")

    def hash(self):
        key = "_".join([f"{name}-{val}" for name, val in self.__dict__.items()])
        return hashlib.sha256(key.encode("utf-8")).hexdigest()


class EvasBackend(BaseBackend):
    binary_ext = "elf"

    @staticmethod
    def supports_target(target: GPUTarget):
        return target.backend == "epu"

    def parse_options(self, opts) -> Any:
        args = {"arch": self.target.arch}
        args.update({k: opts[k] for k in EvasOptions.__dataclass_fields__.keys() if k in opts})
        if args.get("auto_dual_die", False):
            if "enable_dual_die" in args and not args["enable_dual_die"]:
                raise ValueError("auto_dual_die conflicts with enable_dual_die=False")
            args["enable_dual_die"] = True
        # Compiler identity is derived here, never trusted from kernel kwargs.
        # Frontend-only debugging needs no SDK. The common cache hashes
        # libtriton and the backend Python modules for frontend identity.
        args["toolchain_key"] = "" if args.get("to_ttsharedir", False) else epu_toolchain_key()
        # Include the effect analysis and runtime ABI so an older compiled
        # kernel cannot silently lose output copy-back metadata.
        args["buffer_bridge_key"] = hashlib.sha256(b"".join(
            Path(__file__).with_name(name).read_bytes()
            for name in ("compiler.py", "driver.py", "launcher.py", "arg_access.py", "buffer_bridge.py",
                         "local_shards.py", "utils.py"))).hexdigest()
        return EvasOptions(**args)

    def get_codegen_implementation(self, options):
        codegen_fns = {"min_dot_size": lambda lhsType, rhsType: (1, 1, 1)}
        return codegen_fns

    def pack_metadata(self, metadata):
        # Keep Triton's common prefix followed by the EVAS launch contract.
        return (
            metadata.num_warps,
            metadata.num_ctas,
            metadata.shared,
            metadata.cluster_dims[0],
            metadata.cluster_dims[1],
            metadata.cluster_dims[2],
            metadata.name,
            metadata.enable_dual_die,
            metadata.auto_dual_die,
            getattr(metadata, "buffer_access", None),
            getattr(metadata, "local_access", None),
        )

    def load_dialects(self, ctx):
        # JIT language extensions use the EVAS bindings in this FlagTree build.
        libtriton.evas.load_dialects(ctx)

    def add_stages(self, stages, options, language=None):
        stages["ttir"] = lambda src, metadata: _make_ttir(src, metadata, options)
        if options.to_ttsharedir:
            stages["elf"] = _ttir_to_ttsharedir
        else:
            stages["ttsharedir"] = _ttir_to_ttsharedir
            stages["cc"] = _ttshared_to_cc
            stages["elf"] = _cc_to_evbin

    def get_module_map(self) -> Dict[str, ModuleType]:
        """
        Return a map of interface modules to their device-specific implementations
        """
        from triton.language.extra.evas import libdevice

        return {"triton.language.extra.libdevice": libdevice}

    @functools.lru_cache()
    def hash(self):
        key = "_".join([f"{name}-{val}" for name, val in self.__dict__.items()])
        return hashlib.sha256(key.encode("utf-8")).hexdigest()
