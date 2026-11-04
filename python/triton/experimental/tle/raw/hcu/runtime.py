from __future__ import annotations

from pathlib import Path
from typing import Any, Final

import re
import sys
import json
import shlex
import subprocess

from triton._C.libtriton import hcu, llvm
from triton._C.libtriton.tle.llvm import parse_llvm_ir
from triton.backends.hcu.compiler import HIPBackend
from triton.experimental.tle.raw.runtime import RawJITFunction
from triton.experimental.tle.raw.source_store import register_source

from triton.experimental.tle.raw.cuda.runtime import register_kernel_init_hook
from triton.experimental.tle.raw.hcu.dushmem import (
    DUSHMEM_CUMODULE_INIT_HOOK,
    initialize_dushmem_cumodule,
)

from triton.experimental.tle.raw.hcu.utils import _offload_arch


def _sanitize_clang_ir(text: str) -> tuple[str, list[str]]:
    """Handle only known textual LLVM compatibility differences."""

    applied: list[str] = []

    # Keep this list intentionally small. In particular, never rewrite target
    # triple, data layout, calling convention or pointer address spaces.
    for attribute in (
            " nocreateundeforpoison",
            " contract",
    ):
        text, count = re.subn(
            re.escape(attribute),
            "",
            text,
        )
        if count:
            applied.append(f"dropped {count}x {attribute.strip()!r}")

    # Do not import private per-function subtarget feature set.
    patterns = (
        (
            r'\s+"target-cpu"="[^"]*"',
            "target-cpu",
        ),
        (
            r'\s+"target-features"="[^"]*"',
            "target-features",
        ),
    )

    for pattern, name in patterns:
        text, count = re.subn(
            pattern,
            "",
            text,
        )
        if count:
            applied.append(f"dropped {count}x {name}")

    return text, applied


def _extract_target_triple(text: str) -> str | None:
    match = re.search(
        r'^target triple\s*=\s*"([^"]+)"',
        text,
        re.MULTILINE,
    )
    return match.group(1) if match else None


def _dushmem_include_flags() -> list[str]:
    return [
        "-DHIP_ENABLE_WARP_SYNC_BUILTINS",
        "-mcode-object-version=4",
        "-I/opt/dtk/dushmem/include",
    ]


def _library_include_flags(library: str | None) -> list[str]:
    if library in (None, ""):
        return []
    elif library == "dushmem":
        return _dushmem_include_flags()
    raise RuntimeError("TLE-Raw HCU supports only library=None or library='dushmem'; "
                       f"got {library!r}")


_AST_SCAN_MACROS: Final[list[str]] = [
    "-D__device__=__attribute__((device))",
    "-D__host__=__attribute__((host))",
    "-D__global__=__attribute__((global))",
    "-D__shared__=__attribute__((shared))",
    "-D__constant__=__attribute__((constant))",
    "-D__forceinline__=inline __attribute__((always_inline))",
]

_STDIN_NAMES: Final[frozenset[str]] = frozenset({"<stdin>", "-"})


def _collect_device_functions(root: dict) -> list[str]:
    """Return names of __device__ functions with a body, defined in stdin."""
    # Clang's JSON dumper only emits "file" when it changes, so track it
    # in document order to know which file each node belongs to.
    state: dict[str, Any] = {"file": None}
    found: list[str] = []

    def track(loc: Any) -> None:
        if not isinstance(loc, dict):
            return
        if "spellingLoc" in loc or "expansionLoc" in loc:
            track(loc.get("spellingLoc"))
            track(loc.get("expansionLoc"))
        elif "file" in loc:
            state["file"] = loc["file"]

    def visit(node: Any) -> None:
        if not isinstance(node, dict):
            return
        track(node.get("loc"))
        loc_file = state["file"]
        rng = node.get("range") or {}
        track(rng.get("begin"))
        track(rng.get("end"))

        inner = node.get("inner") or []
        if (node.get("kind") == "FunctionDecl" and not node.get("isImplicit") and node.get("name")
                and loc_file in _STDIN_NAMES):
            kinds = {c.get("kind") for c in inner if isinstance(c, dict)}
            if "CUDADeviceAttr" in kinds and "CompoundStmt" in kinds:
                found.append(node["name"])

        for child in inner:
            visit(child)

    old_limit = sys.getrecursionlimit()
    sys.setrecursionlimit(max(old_limit, 10000))
    try:
        visit(root)
    finally:
        sys.setrecursionlimit(old_limit)

    return list(dict.fromkeys(found))


def _infer_device_function_name(source: str, source_file: str) -> str:
    clang = HIPBackend.path_to_rocm_clang()
    arch = _offload_arch()

    command = [
        str(clang),
        "-x",
        "hip",
        "--offload-device-only",
        f"--offload-arch={arch}",
        "-nogpulib",
        "-nogpuinc",
        "-fsyntax-only",
        "-Wno-everything",
        "-ferror-limit=0",
        *_AST_SCAN_MACROS,
        "-Xclang",
        "-ast-dump=json",
        "-",
    ]

    result = subprocess.run(
        command,
        input=source.encode("utf-8"),
        capture_output=True,
        check=False,
    )

    stdout = result.stdout.decode("utf-8", errors="replace")
    start = stdout.find("{")
    if start < 0:
        raise RuntimeError("TLE-Raw HCU AST scan produced no output\n"
                           f"source: {source_file}\n"
                           f"command: {shlex.join(command)}\n"
                           f"stderr:\n{result.stderr.decode(errors='replace')}")

    try:
        root, _ = json.JSONDecoder().raw_decode(stdout[start:])
    except json.JSONDecodeError as exc:
        raise RuntimeError("TLE-Raw HCU failed to parse Clang AST JSON\n"
                           f"source: {source_file}") from exc

    names = _collect_device_functions(root)

    if len(names) == 1:
        return names[0]

    if not names:
        raise RuntimeError("TLE-Raw HCU could not find a __device__ "
                           f"function with a body in {source_file}. "
                           "Specify extern_func_name= explicitly.")

    raise RuntimeError("TLE-Raw HCU found multiple __device__ "
                       f"functions in {source_file}: {names}. "
                       "Specify extern_func_name= explicitly.")


def _find_lowered_function_symbol(
    llvm_ir: str,
    source_name: str,
) -> str:
    definitions = re.findall(
        r"""
        ^\s*define\b
        [^{@\n]*
        @(?:
            "([^"]+)"
            |
            ([^\s(]+)
        )
        \s*\(
        """,
        llvm_ir,
        re.MULTILINE | re.VERBOSE,
    )

    names = [quoted or plain for quoted, plain in definitions]

    if source_name in names:
        return source_name

    itanium_prefix = (f"_Z{len(source_name)}{source_name}")

    candidates = [name for name in names if name.startswith(itanium_prefix)]

    if len(candidates) == 1:
        return candidates[0]

    raise RuntimeError("TLE-Raw HCU cannot uniquely resolve "
                       f"{source_name!r} to an LLVM symbol; "
                       f"definitions={names}, "
                       f"candidates={candidates}")


def _resolve_llvm_tool(name: str) -> Path:
    llvm_root = HIPBackend.path_to_rocm_llvm()
    tool = llvm_root / "bin" / name

    if not tool.is_file():
        raise RuntimeError(f"cannot find {name}: {tool}")

    return tool


def _extract_function_closure(
    llvm_ir: str,
    symbol: str,
) -> str:
    llvm_extract = _resolve_llvm_tool("llvm-extract")

    command = [
        str(llvm_extract),
        f"--func={symbol}",
        "--recursive",
        "-S",
        "-",
        "-o",
        "-",
    ]

    result = subprocess.run(
        command,
        input=llvm_ir.encode("utf-8"),
        capture_output=True,
        check=False,
    )

    if result.returncode != 0:
        raise RuntimeError("TLE-Raw HCU llvm-extract failed\n"
                           f"symbol: {symbol}\n"
                           f"command: {shlex.join(command)}\n"
                           f"stderr:\n"
                           f"{result.stderr.decode(errors='replace')}")

    extracted = result.stdout.decode(
        "utf-8",
        errors="strict",
    )

    if not re.search(
            rf"^\s*define\b[^\n@]*"
            rf"@{re.escape(symbol)}\s*\(",
            extracted,
            re.MULTILINE,
    ):
        raise RuntimeError("llvm-extract output does not contain "
                           f"the requested function {symbol!r}")

    return extracted


def _clang_flags(library: str | None = None) -> list[str]:
    return [
        "-fgpu-rdc",
        "-nogpulib",
        "-Wno-unused-command-line-argument",
        "-include",
        "hip/hip_runtime.h",
        f"-I{HIPBackend.path_to_rocm()}/cuda/cuda/include",
        *_library_include_flags(library),
    ]


class HCUJITFunction(RawJITFunction):
    """Compile CUDA/HIP-compatible device source into an eager raw region."""

    def __init__(self, fn: Any, file: Path, *args, **kwargs) -> None:
        super().__init__(fn, **kwargs)

        self.code: Final[str] = file.read_text()
        self.source_file: Final[str] = str(file)

        # The frontend decorator may say name=\"cuda\", but downstream IR must
        # describe the toolchain that actually compiled the source.
        self.region_dialect: Final[str] = "hcu"
        self.lowered_region_dialect: Final[str] = "llvm"
        self.arg_dialect: Final[str] = "llvm"

        self.opt_level: Final[str] = kwargs.get("opt_level", "-O3")

        if not self.extern_func_name:
            self.extern_func_name = (_infer_device_function_name(
                self.code,
                self.source_file,
            ))

    def make_llvm(self, mlir_context) -> str:
        clang = HIPBackend.path_to_rocm_clang()
        arch = _offload_arch()

        command = [
            str(clang),
            "-x",
            "hip",
            "--offload-device-only",
            f"--offload-arch={arch}",
            self.opt_level,
            "-emit-llvm",
            "-S",
            "-",
            "-o",
            "-",
            *_clang_flags(self.library),
        ]

        result = subprocess.run(
            command,
            input=self.code.encode(),
            capture_output=True,
            check=False,
        )

        if result.returncode != 0:
            raise RuntimeError("TLE-Raw HCU source compilation failed\n"
                               f"source: {self.source_file}\n"
                               f"architecture: {arch}\n"
                               f"command: {shlex.join(command)}\n"
                               f"stderr:\n"
                               f"{result.stderr.decode(errors='replace')}")

        text = result.stdout.decode(errors="strict")
        text, rewrites = _sanitize_clang_ir(text)

        emitted_triple = _extract_target_triple(text)
        expected_triple = str(hcu.TARGET_TRIPLE)

        if (emitted_triple is not None and emitted_triple != expected_triple):
            raise RuntimeError("HCU Clang emitted an incompatible LLVM target triple: "
                               f"{emitted_triple!r}; expected {expected_triple!r}. "
                               "Use the same DTK compiler family as compiler.py.")

        lowered_func_name = (_find_lowered_function_symbol(
            text,
            self.extern_func_name,
        ))

        text = _extract_function_closure(
            text,
            lowered_func_name,
        )

        # Keep dushmem calls/global declarations unresolved here. Only this
        # raw function closure crosses the LLVM -> MLIR import boundary.
        # The device library is linked by HIPBackend.make_hsaco(), never here.
        text, final_rewrites = _sanitize_clang_ir(text)
        rewrites.extend(final_rewrites)

        # findExternalLLVMFunc() in C++ must look up functions by their LLVM symbol,
        # not by the source-level kernel function name.
        self.extern_func_name = lowered_func_name

        # Validate that FlagTree's embedded LLVM and MLIR LLVM importer can
        # consume the textual IR. create_region_by_llvm() will import it again
        # while constructing tle.dsl_region.
        llvm_context = llvm.context()
        try:
            mlir_module = parse_llvm_ir(text, llvm_context, mlir_context)
        except Exception as exc:
            applied = (", ".join(rewrites) if rewrites else "none")
            raise RuntimeError("Failed to import HCU clang LLVM IR into "
                               "MLIR LLVM dialect\n"
                               f"source: {self.source_file}\n"
                               f"rewrites: {applied}\n"
                               f"LLVM IR:\n{text}") from exc

        return str(mlir_module)

    def mark_kernel_init_hook(self, semantic, generator) -> None:
        if self.library == "dushmem":
            from triton.experimental.tle.raw.hcu.dushmem import DUSHMEM_CUMODULE_INIT_HOOK
            operation = generator.module.get_operation()
            hooks = operation.get_str_attr("tle.raw.kernel_init_hooks")
            hook_names = set(hooks.split(",")) if hooks else set()
            hook_names.add(DUSHMEM_CUMODULE_INIT_HOOK)
            generator.module.set_attr(
                "tle.raw.kernel_init_hooks",
                semantic.builder.get_string_attr(",".join(sorted(hook_names))),
            )

    def register_pending_source(self, *, hint: str = "") -> str:
        if not self.extern_func_name:
            raise RuntimeError("Deferred TLE-Raw HCU requires "
                               "extern_func_name= to identify the device "
                               "function in the HIP/CUDA source")

        return register_source(
            region_dialect=self.region_dialect,
            extern_func_name=self.extern_func_name,
            source=self.code,
            hint=hint,
            extra={
                "source_file": self.source_file,
                "opt_level": self.opt_level,
                "library": self.library,
            },
        )

    def create_region_deferred(
        self,
        builder,
        source_id: str,
        handles,
        alias_indices,
        hint: str = "",
    ):
        return builder.create_tle_raw_region_deferred(
            source_id,
            self.region_dialect,
            self.arg_dialect,
            handles,
            alias_indices,
            hint,
        )


def compile_deferred_pending_source(entry: dict, *, context) -> str:
    source_text = entry["source"]

    class _HCUSourceFile:

        def read_text(self):
            return source_text

    hcu_fn = HCUJITFunction(
        fn=None,
        file=_HCUSourceFile(),
        extern_func_name=entry.get("extern_func_name", ""),
        deferred=True,
        opt_level=entry.get("opt_level", "-O3"),
        library=entry.get("library", ""),
    )

    llvm_ir = hcu_fn.make_llvm(context)
    # The materializer must find the lowered (possibly C++-mangled) symbol.
    entry["extern_func_name"] = hcu_fn.extern_func_name
    return llvm_ir


register_kernel_init_hook(DUSHMEM_CUMODULE_INIT_HOOK, initialize_dushmem_cumodule)
