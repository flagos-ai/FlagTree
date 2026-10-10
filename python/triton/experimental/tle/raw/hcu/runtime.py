"""HIP-backed tle.raw runtime for HCU.

HCU raw defaults to deferred. The DSL region only calls the original symbol,
and the clang bitcode — still carrying ``target-cpu`` / ``target-features`` —
is linked in front of the assembler instead of being imported into the Triton
module.
"""

from __future__ import annotations

import os
import re
import shlex
import struct
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Final

import torch
from triton._C.libtriton import llvm
from triton._C.libtriton.tle.llvm import parse_llvm_ir
from triton.experimental.tle.raw.runtime import RawJITFunction
from triton.experimental.tle.raw.source_store import register_source

_LINK_SOURCES: list[dict[str, str]] = []


def take_link_sources() -> list[dict[str, str]]:
    sources = list(_LINK_SOURCES)
    _LINK_SOURCES.clear()
    return sources


def _parse_clang_major(clang: str) -> int | None:
    try:
        out = subprocess.check_output([clang, "--version"], text=True, stderr=subprocess.STDOUT)
    except (OSError, subprocess.CalledProcessError):
        return None
    match = re.search(r"clang version (\d+)\.", out)
    return int(match.group(1)) if match else None


def _resolve_clang() -> str:
    tried: list[str] = []
    candidates = [
        os.getenv("TRITON_HIP_CLANG_PATH"),
        "/opt/dtk/aillvm/bin/clang-18",
        "/opt/dtk/aillvm/bin/clang",
    ]
    for candidate in candidates:
        if not candidate or candidate in tried:
            continue
        tried.append(candidate)
        if Path(candidate).is_file() and _parse_clang_major(candidate):
            return candidate
    detail = ", ".join(tried) if tried else "<none>"
    raise RuntimeError(f"TLE raw HCU requires the DTK aillvm clang. Tried: {detail}. "
                       "Set TRITON_HIP_CLANG_PATH to /opt/dtk/aillvm/bin/clang-18.")


def _offload_arch() -> str:
    arch = os.getenv("HCU_OFFLOAD_ARCH")
    if arch:
        return arch.split(":")[0]
    name = getattr(torch.cuda.get_device_properties(0), "gcnArchName", "gfx936")
    return name.split(":")[0]


def _sanitize_clang_ir(ir: str) -> str:
    ir = ir.replace(" nocreateundeforpoison", "")
    ir = ir.replace(" contract", "")

    def _replace_hex_float(match: re.Match[str]) -> str:
        hex_digits = match.group(1)
        bits = int(hex_digits, 16)
        if len(hex_digits) == 16:
            value = struct.unpack("!d", bits.to_bytes(8, byteorder="big"))[0]
        elif len(hex_digits) == 8:
            value = struct.unpack("!f", bits.to_bytes(4, byteorder="big"))[0]
        else:
            return match.group(0)
        return repr(value)

    ir = re.sub(r"f0x([0-9A-Fa-f]+)", _replace_hex_float, ir)
    # HIP emits a compiler.used global that pins the device function and a
    # __hip_cuid marker. Neither belongs in the Triton module.
    ir = re.sub(r"(?m)^@__hip_cuid_\w+ = .*\n", "", ir)
    ir = re.sub(r"(?m)^@llvm\.compiler\.used = .*\n", "", ir)
    return ir


def _split_args(arglist: str) -> list[str]:
    args: list[str] = []
    depth = 0
    start = 0
    for index, char in enumerate(arglist):
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
        elif char == "," and depth == 0:
            args.append(arglist[start:index].strip())
            start = index + 1
    tail = arglist[start:].strip()
    if tail:
        args.append(tail)
    return args


_LLVM_PREFIX_WORDS = {
    "private",
    "internal",
    "available_externally",
    "linkonce",
    "weak",
    "common",
    "appending",
    "extern_weak",
    "linkonce_odr",
    "weak_odr",
    "external",
    "hidden",
    "protected",
    "dso_local",
    "unnamed_addr",
    "local_unnamed_addr",
}


def _arg_type(arg: str) -> str:
    match = re.match(r"\s*(ptr(?:\s+addrspace\(\d+\))?|i\d+|float|double|half|bfloat|%[\w.]+)", arg)
    if match is None:
        raise RuntimeError(f"cannot read an LLVM argument type from {arg!r}")
    return match.group(1)


def _return_type(prefix: str) -> str:
    tokens = prefix.split()
    while tokens and tokens[0] in _LLVM_PREFIX_WORDS:
        tokens.pop(0)
    if not tokens:
        raise RuntimeError(f"cannot read an LLVM return type from {prefix!r}")
    return " ".join(tokens)


def _define_signature(func_name: str, clang_ir: str) -> tuple[str, str]:
    match = re.search(rf"(?m)^define\s+(.+?)\s+@{re.escape(func_name)}\(", clang_ir)
    if match is None:
        raise RuntimeError(f"clang IR has no definition of {func_name}")
    start = match.end()
    depth = 1
    index = start
    while index < len(clang_ir) and depth:
        if clang_ir[index] == "(":
            depth += 1
        elif clang_ir[index] == ")":
            depth -= 1
        index += 1
    if depth:
        raise RuntimeError(f"clang IR definition of {func_name} has an unclosed argument list")
    return _return_type(match.group(1)), clang_ir[start:index - 1]


def _struct_typedef(ret_ty: str, clang_ir: str) -> str:
    if not ret_ty.startswith("%"):
        return ""
    match = re.search(rf"(?m)^{re.escape(ret_ty)} = type \{{.*\}}", clang_ir)
    if match is None:
        raise RuntimeError(f"clang IR is missing the type of {ret_ty}")
    return match.group(0) + "\n"


def _wrapper_llvm(func_name: str, clang_ir: str) -> tuple[str, str]:
    """Build a stub whose body is an alwaysinline call to ``func_name``.

    The stub is what Triton imports. The original clang bitcode, linked later,
    supplies ``func_name`` with its target attributes intact. Shared-memory
    descriptors keep ``ptr addrspace(3)`` and a struct return so ``call_smem``
    can pack the result back into a memdesc.
    """
    ret_ty, arglist = _define_signature(func_name, clang_ir)
    types = [_arg_type(arg) for arg in _split_args(arglist)]
    wrapper = f"{func_name}_raw"
    params = ", ".join(f"{ty} %{index}" for index, ty in enumerate(types))
    call_args = ", ".join(f"{ty} %{index}" for index, ty in enumerate(types))
    decl_params = ", ".join(types)
    typedef = _struct_typedef(ret_ty, clang_ir)
    # internal so make_llir still sees a single externally linked kernel.
    if ret_ty == "void":
        body = (f"  call void @{func_name}({call_args})\n"
                f"  ret void\n")
    else:
        body = (f"  %result = call {ret_ty} @{func_name}({call_args})\n"
                f"  ret {ret_ty} %result\n")
    stub = (f"{typedef}"
            f"declare {ret_ty} @{func_name}({decl_params})\n"
            f"define internal {ret_ty} @{wrapper}({params}) {{\n"
            f"{body}"
            f"}}\n")
    return wrapper, stub


def _clang_command(clang: str, arch: str, src: str, output: str, *, bitcode: bool, dushmem: bool = False) -> list[str]:
    command = [
        clang,
        "-x",
        "hip",
        "--cuda-device-only",
        "-emit-llvm",
        "-O3",
        f"--offload-arch={arch}",
        "-nogpulib",
        "-fno-exceptions",
        "-fno-rtti",
        "-I/opt/dtk/include",
        "-I/opt/dtk/hip/include",
    ]
    if dushmem:
        # libdushmem_device.a is code object v4, and its headers call the
        # warp-sync builtins that HIP hides unless this macro is set.
        command.extend([
            "-DHIP_ENABLE_WARP_SYNC_BUILTINS",
            "-mcode-object-version=4",
            "-I/opt/dtk/include/dushmem",
            "-I/opt/dtk/dushmem/include",
        ])
    if bitcode:
        # -fgpu-rdc keeps the device function externally visible so llvm-link
        # can resolve the declaration Triton emits.
        command.extend(["-fgpu-rdc", "-c", src, "-o", output])
    else:
        command.extend(["-S", src, "-o", output])
    return command


def _run_clang(command: list[str], src_path: str) -> str:
    build = subprocess.run(command, capture_output=True)
    if build.returncode != 0:
        raise RuntimeError(f"HCU clang failed to compile {src_path} (exit {build.returncode}).\n"
                           f"command: {shlex.join(command)}\n"
                           f"stderr:\n{build.stderr.decode(errors='replace')}")
    if command[-1] == "-":
        return build.stdout.decode()
    return Path(command[-1]).read_text()


def _register_dushmem_hook() -> None:
    # Cache hits skip tracing, so the hook has to be registered at import.
    from triton.experimental.tle.raw.cuda.runtime import register_kernel_init_hook
    from triton.experimental.tle.raw.hcu.dushmem import (
        DUSHMEM_CUMODULE_INIT_HOOK,
        initialize_dushmem_cumodule,
    )
    try:
        register_kernel_init_hook(DUSHMEM_CUMODULE_INIT_HOOK, initialize_dushmem_cumodule)
    except RuntimeError:
        pass


class HCUJITFunction(RawJITFunction):

    def __init__(self, fn: Any, file: Path, *args, **kwargs) -> None:
        # Eager import drops HCU target attributes. Deferred is the only path.
        kwargs["deferred"] = kwargs.get("deferred", True)
        if not kwargs["deferred"]:
            raise RuntimeError("tle_raw HCU only supports deferred mode")
        super().__init__(fn, **kwargs)
        if self.library not in ("", "dushmem"):
            raise RuntimeError(f"tle_raw library={self.library!r} is not supported on hcu")
        if self.library == "dushmem":
            _register_dushmem_hook()
        self.code: Final[str] = file.read_text()
        self.region_dialect: Final[str] = "hcu"
        self.lowered_region_dialect: Final[str] = "llvm"
        self.arg_dialect: Final[str] = "llvm"
        self.source_file: Final[str] = str(file)
        self.arch: Final[str] = _offload_arch()

    def register_pending_source(self, *, hint: str = "") -> str:
        if not self.extern_func_name:
            raise RuntimeError("deferred tle_raw HCU source requires extern_func_name= "
                               "(the device function symbol in the HIP file)")
        return register_source(
            region_dialect=self.region_dialect,
            extern_func_name=self.extern_func_name,
            source=self.code,
            hint=hint,
            extra={"source_file": self.source_file, "arch": self.arch, "library": self.library},
        )

    def mark_kernel_init_hook(self, semantic, generator) -> None:
        if self.library != "dushmem":
            return
        from triton.experimental.tle.raw.hcu.dushmem import DUSHMEM_CUMODULE_INIT_HOOK
        _register_dushmem_hook()
        operation = generator.module.get_operation()
        hooks = operation.get_str_attr("tle.raw.kernel_init_hooks")
        hook_names = set(hooks.split(",")) if hooks else set()
        hook_names.add(DUSHMEM_CUMODULE_INIT_HOOK)
        generator.module.set_attr(
            "tle.raw.kernel_init_hooks",
            semantic.builder.get_string_attr(",".join(sorted(hook_names))),
        )

    def create_region_by_llvm(self, builder, llvm_ir: str, handles, alias_indices, hint: str = "",
                              extern_func_name: str = ""):
        return super().create_region_by_llvm(builder, llvm_ir, handles, alias_indices, hint, extern_func_name)

    def create_region_deferred(self, builder, source_id: str, handles, alias_indices, hint: str = ""):
        return builder.create_tle_raw_region_deferred(
            source_id,
            self.region_dialect,
            self.arg_dialect,
            handles,
            alias_indices,
            hint,
        )

    def _compile_device_ir(self) -> str:
        clang = _resolve_clang()
        with tempfile.NamedTemporaryFile(suffix=".hip", mode="w", delete=False) as src_file:
            src_file.write(self.code)
            src_path = src_file.name
        try:
            command = _clang_command(clang, self.arch, src_path, "-", bitcode=False,
                                      dushmem=self.library == "dushmem")
            return _sanitize_clang_ir(_run_clang(command, self.source_file))
        finally:
            Path(src_path).unlink(missing_ok=True)

    def make_llvm(self, mlir_context) -> str:
        raise RuntimeError("tle_raw HCU uses deferred mode; eager import is disabled")
        # Eager imported the whole clang module. That path drops target-cpu /
        # target-features, so HCU does not use it.
        # if not self.extern_func_name:
        #     raise RuntimeError("tle_raw HCU source requires extern_func_name=")
        # clang_ir = self._compile_device_ir()
        # llvm_context = llvm.context()
        # try:
        #     module = parse_llvm_ir(clang_ir, llvm_context, mlir_context)
        # except Exception as exc:
        #     raise RuntimeError(f"failed to import the HCU clang IR of {self.source_file}.\n"
        #                        f"IR as emitted by clang:\n{clang_ir}") from exc
        # return f"{module}"


def compile_deferred_pending_source(entry: dict, *, context) -> str:
    """Import a call stub. The real clang bitcode is linked at amdgcn time."""
    source_text = entry["source"]
    func_name = entry.get("extern_func_name") or ""
    arch = entry.get("arch") or _offload_arch()
    source_file = entry.get("source_file", "<deferred hcu source>")

    class _HipSource:

        def read_text(self):
            return source_text

    library = entry.get("library") or ""
    hip_fn = HCUJITFunction(
        fn=None,
        file=_HipSource(),
        extern_func_name=func_name,
        deferred=True,
        library=library,
    )
    object.__setattr__(hip_fn, "arch", arch)
    object.__setattr__(hip_fn, "source_file", source_file)
    clang_ir = hip_fn._compile_device_ir()
    wrapper, stub = _wrapper_llvm(func_name, clang_ir)
    entry["extern_func_name"] = wrapper
    _LINK_SOURCES.append({
        "symbol": func_name,
        "source": source_text,
        "arch": arch,
        "file": source_file,
        "library": library,
    })
    llvm_context = llvm.context()
    try:
        module = parse_llvm_ir(stub, llvm_context, context)
    except Exception as exc:
        raise RuntimeError(f"failed to import the HCU raw stub for {func_name}.\n{stub}") from exc
    return f"{module}"
