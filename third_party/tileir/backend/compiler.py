from triton.runtime.errors import OutOfResources
from triton.backends.tileir.errors import HitFallback
from triton.runtime.cache import get_cache_manager
from triton.backends.compiler import BaseBackend, GPUTarget, Language
from triton.backends.tileir.conf import TileIREnvConf
from triton._C.libtriton import ir, passes, tileir

# flagtree: added `field` import for the lazy tileiras lookup default_factory below.
from dataclasses import dataclass, field
import functools
from typing import Any, Dict, Tuple, Optional
from types import ModuleType
import hashlib
import re
import logging
import tempfile
import os
import subprocess
import sys
from pathlib import Path


def format_compute_capability(capability: int) -> str:
    """
    Format compute capability for GPU architecture.

    Args:
        capability: Numeric compute capability (e.g., 80, 90, 100)

    Returns:
        Formatted architecture string (e.g., "sm_80", "sm_90a", "sm_100a")

    Note:
        - Hopper (sm_90) and newer architectures get 'a' suffix
        - Ampere (sm_80) and older architectures have no suffix
    """
    if capability >= 90:  # Hopper and newer
        return f"sm_{capability}a"
    else:  # Ampere and older
        return f"sm_{capability}"


if sys.version_info >= (3, 12):
    TemporaryDirectory = tempfile.TemporaryDirectory
else:
    import shutil
    from contextlib import contextmanager

    @contextmanager
    def TemporaryDirectory(suffix=None, prefix=None, dir=None, delete=True):
        temp_dir = tempfile.mkdtemp(suffix, prefix, dir)
        try:
            yield temp_dir
        finally:
            if delete:
                shutil.rmtree(temp_dir)


@dataclass(frozen=True)
class TileIROptions:
    ########################## tileIR core options ##########################
    backend_name: str = "tileir"
    arch: str = None
    num_ctas: int = 1
    # tileir use num_stages to control the op cost, see <tileir_link>
    num_stages: int = 3
    # tileir use opt_level to control the optimization level, see <tileir_link>
    opt_level: int = 3
    # tileir use occupancy to control the register usage, see <tileir_link>
    occupancy: int = 1
    # tileir use enable_fp_fusion to control the fma fusion, see <tileir_link>
    enable_fp_fusion: bool = True
    # flagtree: defer the tileiras lookup to instantiation time so that merely
    # importing the tileir backend doesn't require the binary to be present
    # (e.g. AMD-only builds, CUDA < 13, or systems where tileiras isn't on PATH).
    # Upstream fork calls get_tileiras_path() at class-definition time and
    # raises on import when tileiras is missing.
    tileir_tileiras_path: str = field(default_factory=TileIREnvConf.get_tileiras_path)

    # type and precision control, compatibility with other backend
    supported_fp8_dtypes: Tuple[str] = ("fp8e5", "fp8e4b15")
    deprecated_fp8_dot_operand_dtypes: Tuple[str] = ()
    default_dot_input_precision: str = "tf32"
    allowed_dot_input_precisions: Tuple[str] = (
        "tf32",
        "tf32x3",
        "bf16x3",
        "bf16x6",
        "ieee",
    )

    ########################## compatibility with other backend ##########################
    # tileir doesn't need these flags, just for compatibility with other backend
    num_warps: int = 4
    cluster_dims: tuple = (1, 1, 1)
    instrumentation_mode: str = ""
    debug: bool = False
    sanitize_overflow: bool = True
    extern_libs: dict = None
    # maxnreg in tileir backend is just for compatibility with other backend
    # tileir use occupancy to control the register usage.
    maxnreg: Optional[int] = None
    launch_pdl: bool = False
    launch_cooperative_grid: bool = False
    max_num_imprecise_acc_default: bool = None
    # workaround for tileir memory model
    # currently we only autogen alias mem token, non-alias is not supported
    enable_autogen_alias_mem_token: bool = True

    # Dynamic environment-dependent properties
    # These properties influence the behavior of the tile compiler
    # and need to be updated automatically when accessed to reflect current environment settings
    @property
    def enable_ftz(self):
        return TileIREnvConf.enable_ftz()

    @property
    def enable_approx(self):
        return TileIREnvConf.enable_approx()

    def __post_init__(self):
        assert self.num_warps > 0 and (self.num_warps & (self.num_warps - 1)) == 0, ("num_warps must be a power of 2")

    def hash(self):
        hash_dict = dict(self.__dict__)
        # Get all property values from class __dict__
        for name, value in type(self).__dict__.items():
            if isinstance(value, property):
                hash_dict[name] = getattr(self, name)
        key = "_".join([f"{name}-{val}" for name, val in sorted(hash_dict.items())])
        return hashlib.sha256(key.encode("utf-8")).hexdigest()


def get_tileir_version():
    return "13.1"


class TileIRBackend(BaseBackend):

    @classmethod
    def route_target(cls, target, jit_fn):
        from triton.backends.tileir.router import pick_target_for_kernel

        candidate = pick_target_for_kernel(target, jit_fn)
        return candidate if candidate != target else None

    @classmethod
    def get_language_extension(cls):
        from triton.backends.tileir import extend_core

        return extend_core

    def make_ir(self, src, options, codegen_fns, module_map, context):
        if hasattr(src, "fn"):
            from triton.backends.tileir.code_generator import ast_to_ttir

            return ast_to_ttir(src.fn, src, context=context, options=options, codegen_fns=codegen_fns,
                               module_map=module_map)
        return super().make_ir(src, options, codegen_fns, module_map, context)

    def get_module_map(self):
        from triton.language.extra.cuda import libdevice

        return {"triton.language.extra.libdevice": libdevice}

    @staticmethod
    def supports_target(target: GPUTarget):
        return target.backend == "tileir"

    def _parse_arch(self, arch):
        pattern = r"^sm(\d+)$"
        match = re.fullmatch(pattern, arch)
        if not match:
            raise ValueError(f"TRITON_OVERRIDE_ARCH must have the form {pattern}")
        return int(match.group(1))

    def __init__(self, target: GPUTarget) -> None:
        super().__init__(target)
        self.binary_ext = "cubin"

    def parse_options(self, opts) -> Any:
        args = {"arch": os.getenv("TRITON_OVERRIDE_ARCH", f"sm{self.target.arch}")}
        args.update({k: opts[k] for k in TileIROptions.__dataclass_fields__.keys() if k in opts if opts[k] is not None})
        capability = int(self._parse_arch(args["arch"]))
        if "supported_fp8_dtypes" not in args:
            supported_fp8_dtypes = set(TileIROptions.supported_fp8_dtypes)
            # todo: sm90 or 89? oait uses 89, we use 90
            if capability >= 90:
                supported_fp8_dtypes.add("fp8e4nv")
                supported_fp8_dtypes.add("fp8e5")
            args["supported_fp8_dtypes"] = tuple(sorted(supported_fp8_dtypes))

        if "deprecated_fp8_dot_operand_dtypes" not in args:
            if capability >= 90:
                args["deprecated_fp8_dot_operand_dtypes"] = ("fp8e4b15", )

        if "enable_fp_fusion" not in args:
            args["enable_fp_fusion"] = os.getenv("TRITON_DEFAULT_FP_FUSION", "1") == "1"

        args["max_num_imprecise_acc_default"] = 2**30 if capability == 90 else 0
        return TileIROptions(**args)

    def pack_metadata(self, metadata):
        return (
            metadata.num_warps,
            metadata.num_ctas,
        )

    def get_codegen_implementation(self, options):
        import triton.language.extra.cuda as cuda

        capability = int(self._parse_arch(options.arch))
        codegen_fns = {
            "convert_custom_types":
            cuda.convert_custom_float8_sm80 if capability >= 80 else cuda.convert_custom_float8_sm70,
            "min_dot_size": lambda lhs, rhs: (1, 1, 1),
        }
        return codegen_fns

    def load_dialects(self, ctx):
        tileir.load_dialects(ctx)

    @staticmethod
    def call_tileiras(mod, metadata, opt: TileIROptions, capability):
        name = metadata["name"]
        fn_cache_manager = get_cache_manager(metadata["hash"])

        tileiras = opt.tileir_tileiras_path
        tileiras_cmd = [
            tileiras,
            f"--gpu-name=sm_{capability}",
            f"--opt-level={opt.opt_level}",
        ]
        # Save bytecode to cache
        bytecode = tileir.write_bytecode(mod)
        bytecode_cache_name = f"{name}.bytecode"
        bytecode_file = fn_cache_manager.put(bytecode, bytecode_cache_name)

        # Use temp file for cubin output to avoid race conditions.
        with (
                tempfile.NamedTemporaryFile(delete=False, mode="w", suffix=".log") as flog,
                tempfile.NamedTemporaryFile(delete=False, suffix=".cubin") as fbin,
        ):
            tileiras_cmd.append(bytecode_file)
            tileiras_cmd.append(f"-o")
            tileiras_cmd.append(fbin.name)

            try:
                try:
                    subprocess.run(tileiras_cmd, check=True, close_fds=False, stderr=flog)
                except subprocess.CalledProcessError as e:
                    with open(flog.name) as log_file:
                        log = log_file.read()

                    if "uses too much shared data" in log:
                        pattern = r"0x([0-9a-fA-F]+) bytes, 0x([0-9a-fA-F]+) max"
                        match = re.search(pattern, log)
                        if match:
                            used_smem = int(match.group(1), 16)
                            max_smem = int(match.group(2), 16)
                            raise OutOfResources(used_smem, max_smem, "shared memory")
                    if "allocated tmem out of resource" in log:
                        # "allocated tmem out of resource: <used> vs <max>"
                        pattern = (r"allocated tmem out of resource:\s*([0-9]+)\s*vs\s*([0-9]+)")
                        match = re.search(pattern, log)
                        if match:
                            used_tmem = int(match.group(1))
                            max_tmem = int(match.group(2))
                            raise OutOfResources(used_tmem, max_tmem, "tensor memory")
                    error = f"`tileiras` failed with error code {e.returncode}"
                    raise RuntimeError(f"{error}\n"
                                       f"`tileiras` stderr:\n{log}\n"
                                       f"Repro command: {' '.join(str(item) for item in tileiras_cmd)}\n")
                with open(fbin.name, "rb") as f:
                    cubin = f.read()
            finally:
                # flagtree: clean up both temp files on every path (success or error).
                # Previously the .log file leaked on success — long-running workloads
                # that JIT many kernels would accumulate one .log per compile in /tmp.
                if os.path.exists(flog.name):
                    os.remove(flog.name)
                if os.path.exists(fbin.name):
                    os.remove(fbin.name)
        return cubin

    @staticmethod
    def make_ttir(mod, metadata, opt: TileIROptions, capability):
        # TODO: check these transform passes
        # flagtree: the upstream-newer fork uses mod.name; flagtree's MLIR API
        # (older) exposes the entry function name via get_entry_func_name().
        metadata["name"] = mod.get_entry_func_name()
        pm = ir.pass_manager(mod.context)
        pm.enable_debug()
        passes.common.add_inliner(pm)
        passes.ttir.add_combine(pm)
        passes.common.add_canonicalizer(pm)
        passes.common.add_cse(pm)
        passes.common.add_licm(pm)
        passes.common.add_symbol_dce(pm)
        # passes.ttir.add_loop_unroll(pm)
        pm.run(mod, "make_ttir")
        return mod

    @staticmethod
    def make_tileir(mod, metadata, opt: TileIROptions, capability):
        pm = ir.pass_manager(mod.context)
        pm.enable_debug()
        # Inherit LiftControlflowToSCF from upstream to adapt to `ControlFlow` within `triton.func`
        tileir.passes.add_lift_tt_cf_to_scf(pm)
        # The root IR for ttir is builtin moduleOp and all
        # cuda-tile ir must under tileir_moduleOp.
        # So, we will insert an tileir moduleOp directly at the beginning of TritonToCudaTile pass.
        tileir.passes.add_assume_to_tileir(pm)
        tileir.passes.add_triton_to_cudatile(
            pm,
            opt.enable_approx,
            opt.enable_ftz,
            capability,
            metadata["num_ctas"],
            metadata["num_warps"],
            opt.occupancy,
            metadata["num_stages"],
        )
        tileir.passes.add_auto_gen_memtoken(pm, opt.enable_autogen_alias_mem_token)
        passes.common.add_inliner(pm)
        if opt.enable_fp_fusion:
            tileir.passes.add_fma_fusion(pm)
        tileir.passes.add_strip_debuginfo(pm)
        pm.run(mod, "make_tileir")
        if not tileir.only_contain_legal_dialects(mod):
            if os.environ.get("FLAGTREE_TILEIR_DUMP_FAILED_IR", "0") == "1":
                safe_name = re.sub(r"[^a-zA-Z0-9_]+", "_", metadata.get("name", "unknown"))
                with open(f"/tmp/flagtree_tileir_failed_{safe_name}.mlir", "w") as f:
                    f.write(str(mod))
            raise RuntimeError("Triton ttir to tileir ir failed. Some ttir ops cannot be converted to tileir.")

        pattern = r"entry @([a-zA-Z0-9_]*)\("
        match = re.findall(pattern, mod.__str__())
        if len(match) != 1:
            raise RuntimeError("Kernel Name matching fail")
        return mod

    @staticmethod
    def make_cubin(mod, metadata, opt: TileIROptions, capability):
        # TileIR kernels use statically allocated shared memory and request no
        # dynamic shared memory from the CUDA launcher.
        metadata["shared"] = 0
        return TileIRBackend.call_tileiras(mod, metadata, opt, capability)

    def add_stages(self, stages, options, language):
        assert language == Language.TRITON, "Only TRITON language is supported for now"
        capability = int(self._parse_arch(options.arch))
        stages["ttir"] = lambda src, metadata: self.make_ttir(src, metadata, options, capability)
        stages["tileir"] = lambda src, metadata: self.make_tileir(src, metadata, options, capability)
        stages["cubin"] = lambda src, metadata: self.make_cubin(src, metadata, options, capability)

    @functools.lru_cache()
    def hash(self):
        version = get_tileir_version()
        return f"{'tileir'}-{version}-{self.target.arch}"
