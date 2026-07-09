from __future__ import annotations
import ast
import copy
from functools import cached_property
import inspect
import os
import re
import subprocess
import textwrap
from typing import Any, Dict, Final, List, Optional

from mlir import ir
from mlir.passmanager import PassManager

from ..mlir.codegen import MLIRCodeGenerator

_GCU_COMPILER_OPT = "/opt/triton_gcu/bin/gcu-compiler-opt"


class TOPSMLIRJITFunction(object):
    """TLE-Raw dialect for TOPS MLIR EDSL: writes MLIR that lowers to GCU-compatible LLVM IR.

    Usage:
        @dialect(name="tops_mlir", arch="gcu400")
        def edsl(x: Input[L["!llvm.ptr<1>"]], ...):
            ...  # MLIR Python bindings code

    When use_gcu_opt=True, lowering is delegated to gcu-compiler-opt which
    understands GCU dialect ops (gcu.ptr2memref, etc.) and gpu.* ops natively.
    The EDSL output is wrapped in gpu.module before passing to gcu-compiler-opt.
    """

    def __init__(self, fn: Any, pipeline: Optional[List[str]] = None, context: Optional[ir.Context] = None,
                 arch: str = "gcu400", use_gcu_opt: bool = True, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.fn: Final[Any] = fn
        self.arch: Final[str] = arch
        self.use_gcu_opt: Final[bool] = use_gcu_opt
        self.region_dialect: Final[str] = "tops"
        self.arg_dialect: Final[str] = "llvm"
        self.pipeline: Final[List[str]] = ([*pipeline] if pipeline is not None else [
            "convert-vector-to-scf{target-rank=1}",
            "convert-scf-to-cf",
            "convert-vector-to-llvm",
            "finalize-memref-to-llvm",
            "convert-arith-to-llvm",
            "convert-cf-to-llvm",
            "convert-func-to-llvm",
            "convert-index-to-llvm",
            "reconcile-unrealized-casts",
            "cse",
        ])
        ctx = ir.Context() if context is None else context
        ctx.allow_unregistered_dialects = True
        self.context: Final[ir.Context] = ctx
        self.__triton_builtin__: Final[bool] = True

    def __deepcopy__(self, memo: Dict[int, Any]) -> TOPSMLIRJITFunction:
        return self.__class__(
            copy.deepcopy(self.fn, memo),
            copy.deepcopy(self.pipeline, memo),
            self.context,
            self.arch,
            self.use_gcu_opt,
        )

    @cached_property
    def ast(self) -> ast.Module:
        return ast.parse(self.src)

    @cached_property
    def absfilename(self) -> str:
        return inspect.getabsfile(self.fn)

    @cached_property
    def fnname(self) -> str:
        return self.fn.__name__

    @cached_property
    def globals(self) -> Dict[str, Any]:
        g = {k: v for k, v in self.fn.__globals__.items() if not k.startswith("__")}
        if self.fn.__closure__:
            for name, cell in zip(self.fn.__code__.co_freevars, self.fn.__closure__):
                g[name] = cell.cell_contents
        return g

    @cached_property
    def codegen(self) -> MLIRCodeGenerator:
        return MLIRCodeGenerator(self.absfilename, {}, self.globals, self.context)

    @property
    def mlir_module(self) -> ir.Module:
        return self.codegen.visit(self.ast)

    @property
    def ll(self) -> ir.Module:
        mod: ir.Module = self.mlir_module
        with self.context:
            pm: PassManager = PassManager()
            pm.enable_verifier(True)
            if os.environ.get("MLIR_ENABLE_DUMP"):
                pm.enable_ir_printing()
            for p in self.pipeline:
                pm.add(p)
            pm.run(mod.operation)
            return mod

    def _lower_via_gcu_opt(self) -> str:
        """Lower MLIR to LLVM IR by calling gcu-compiler-opt as subprocess.

        Wraps the EDSL output in gpu.module (required by gcu-compiler-opt),
        then runs a GCU-aware lowering pipeline that handles gcu.* and gpu.* ops.
        """
        mod: ir.Module = self.mlir_module
        mlir_text = str(mod)

        inner = re.search(r"module\b[^{]*\{(.*)\}\s*$", mlir_text, re.DOTALL)
        body = inner.group(1) if inner else mlir_text

        wrapped = ('module attributes {gpu.container_module} {\n'
                   f'  gpu.module @triton {{\n{body}\n  }}\n'
                   '}\n')

        local_mem_size = TOPSMLIRJITFunction._compute_local_mem_size(wrapped)
        wrapped = self._fix_gcu_types(wrapped)

        chipset = self.arch if self.arch else "gcu400"
        vbw = str(2048 * 8 if "400" in chipset or "410" in chipset else 512 * 8)
        debug_passes: list[str] = []
        if os.environ.get("MLIR_ENABLE_DUMP"):
            debug_passes.append("--mlir-print-ir-after-all")
        if os.environ.get("TRITON_DISABLE_LINE_INFO", "1").lower() not in (
                "1",
                "true",
                "yes",
                "on",
        ):
            debug_passes.append("--ensure-debug-info-scope-on-llvm-func")
        if os.environ.get("MLIR_ENABLE_TIMING", "").lower() in (
                "1",
                "true",
                "yes",
                "on",
        ):
            debug_passes += ["--mlir-timing", "--mlir-timing-display=list"]
        passes = [
            f"-insert-local-fence=arch={chipset}",
            "--convert-vector-to-scf=target-rank=1",
            "-lower-affine",
            f"-convert-vector-to-gcu=vector-bit-width={vbw}",
            "-canonicalize",
            "-convert-private-tag-to-gcu",
            "-convert-memref-to-gcu",
            f"-kernel-memory-alloc=arch={chipset} num-warps=1",
            "-loop-invariant-code-motion",
            "-convert-scf-to-cf",
            "-canonicalize",
            "-cse",
            "--symbol-dce",
            "-gcu-remove-transform-ir",
            f"-convert-vector-to-gcu=vector-bit-width={vbw}",
            "-canonicalize",
            "--expand-strided-metadata",
            "-lower-affine",
            "-canonicalize",
            "-cse",
            f"--convert-gpu-to-gcu=chipset={chipset} vector-bit-width={vbw}",
            f"--gcu-attach-target=arch={chipset}",
            "-convert-index-to-llvm",
            "-gpu-to-llvm",
            "-convert-llvm-to-gcu",
            "-alloca-to-entry",
            "-canonicalize",
        ]

        cmd = [_GCU_COMPILER_OPT, *debug_passes, *passes]

        if os.environ.get("EDSL_DEBUG_OPT", "").lower() in ("1", "true"):
            dbg_cmd = [_GCU_COMPILER_OPT, "--mlir-print-ir-after-all", *passes]
            dbg_proc = subprocess.run(dbg_cmd, input=wrapped, capture_output=True, text=True, timeout=60)
            import sys
            print("=== gcu-compiler-opt STDERR (debug) ===", file=sys.stderr)
            for line in dbg_proc.stderr.splitlines():
                if any(kw in line
                       for kw in ("memref.alloc", "malloc", "IR Dump", "alloca", "local_memory_size", "address_space")):
                    print(line, file=sys.stderr)
            print("=== end debug ===", file=sys.stderr)

        proc = subprocess.run(cmd, input=wrapped, capture_output=True, text=True, timeout=60)
        if proc.returncode != 0:
            raise RuntimeError(f"gcu-compiler-opt failed (rc={proc.returncode}):\n"
                               f"STDIN:\n{wrapped}\n"
                               f"STDERR:\n{proc.stderr}")
        result = self._unwrap_gpu_module(proc.stdout)
        result = self._rename_local_mem(result, self.fnname)
        result = self._fix_local_mem_size(result, local_mem_size)
        return result

    @staticmethod
    def _fix_gcu_types(text: str) -> str:
        """Fix GCU dialect type syntax differences between Python MLIR and gcu-compiler-opt.

        Python MLIR (unregistered dialects) produces:
          !gcu.dte<private>   → gcu-compiler-opt expects !gcu.dte<<private>>
          memref<Nxf32, 9>    → gcu-compiler-opt expects memref<Nxf32, #gcu.address_space<local>>
          func.func           → gpu.func (required for gcu.dynamic_shared_memory lowering)
        Also computes gcu.local_memory_size from memref.view operations.
        """
        text = re.sub(r'!gcu\.dte<(\w+)>', r'!gcu.dte<<\1>>', text)
        text = re.sub(
            r'memref<([^>]*),\s*9\s*>',
            r'memref<\1, #gcu.address_space<local>>',
            text,
        )
        text = re.sub(
            r'func\.func\s+public\s+@(\w+)\(([^)]*)\)\s*\{',
            r'gpu.func @\1(\2) kernel {',
            text,
        )
        text = re.sub(
            r'^\s*return\s*$',
            '    gpu.return',
            text,
            flags=re.MULTILINE,
        )
        text = re.sub(
            r'<\{overflowFlags = #arith\.overflow<none>\}>',
            '',
            text,
        )
        return text

    @staticmethod
    def _compute_local_mem_size(text: str) -> int:
        """Compute required local memory from memref.view ops in MLIR text.

        Handles both assembly format:
          memref.view %base[%c16384][] : memref<?xi8, 9> to memref<4096xf32, 9>
        and generic format:
          "memref.view"(%base, %off) : (...) -> memref<4096xf32, 9>
        """
        type_sizes = {'f32': 4, 'f16': 2, 'bf16': 2, 'i32': 4, 'i8': 1, 'i16': 2, 'i64': 8, 'f64': 8}
        constants = {}
        for cm in re.finditer(r'(%\w+)\s*=\s*arith\.constant\s+(\d+)\s*:\s*index', text):
            constants[cm.group(1)] = int(cm.group(2))

        max_end = 0
        for m in re.finditer(
                r'memref\.view\s+%\w+\[(%\w+)\].*?'
                r'memref<(\d+)x(\w+),\s*(?:9|#gcu\.address_space<local>)>', text):
            offset_name = m.group(1)
            num_elems = int(m.group(2))
            elem_ty = m.group(3)
            elem_size = type_sizes.get(elem_ty, 4)
            view_size = num_elems * elem_size
            offset = constants.get(offset_name, 0)
            end = offset + view_size
            if end > max_end:
                max_end = end

        for m in re.finditer(r'"memref\.view".*?'
                             r'memref<(\d+)x(\w+),\s*(?:9|#gcu\.address_space<local>)>', text):
            num_elems = int(m.group(1))
            elem_ty = m.group(2)
            elem_size = type_sizes.get(elem_ty, 4)
            view_size = num_elems * elem_size
            if view_size > max_end:
                max_end = view_size

        align = 128
        if max_end > 0:
            max_end = ((max_end + align - 1) // align) * align
        return max_end

    @staticmethod
    def _rename_local_mem(text: str, suffix: str) -> str:
        """Rename local_mem global symbol to avoid collision with native pipeline.

        Only renames @local_mem references and sym_name = "local_mem",
        NOT attribute names like gcu.local_memory_size.
        """
        unique = f"edsl_local_mem_{suffix}"
        text = re.sub(r'@local_mem\b', f'@{unique}', text)
        text = text.replace('sym_name = "local_mem"', f'sym_name = "{unique}"')
        return text

    @staticmethod
    def _fix_local_mem_size(text: str, size: int) -> str:
        """Fix local memory size attribute after kernel-memory-alloc pass.

        Only fixes gcu.local_memory_size on functions. Does NOT touch
        gcu.dsm_memory_size (shared/L2), global variable sizes, or tensor types.
        """
        if size <= 0:
            return text
        text = re.sub(
            r'gcu\.local_memory_size\s*=\s*\d+\s*:\s*i64',
            f'gcu.local_memory_size = {size} : i64',
            text,
        )
        return text

    @staticmethod
    def _unwrap_gpu_module(text: str) -> str:
        """Extract llvm.func definitions from gpu.module wrapper.

        gcu-compiler-opt outputs (3 levels of braces):
          module attributes {gpu.container_module} {   <- level 1
            gpu.module @triton [...] {                 <- level 2
              llvm.func @kernel(...) { ... }           <- content
            }                                         <- close level 2
          }                                           <- close level 1

        The C++ backend expects a flat module with llvm.func:
          module {
            llvm.func @kernel(...) { ... }
          }
        """
        lines = text.splitlines()
        gpu_line = None
        for i, line in enumerate(lines):
            if "gpu.module" in line:
                gpu_line = i
                break
        if gpu_line is None:
            return text
        depth = 0
        start = None
        end = None
        for i in range(gpu_line, len(lines)):
            for ch in lines[i]:
                if ch == "{":
                    if depth == 0:
                        start = i + 1
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        end = i
                        break
            if end is not None:
                break
        if start is None or end is None:
            return text
        body = "\n".join(lines[start:end])
        return f"module {{\n{body}\n}}\n"

    def create_region_by_llvm(self, builder, llvm: str, handles, alias_indices, hint: str = ""):
        return builder.create_tle_raw_region_by_llvm_func(
            llvm,
            self.region_dialect,
            self.arg_dialect,
            handles,
            alias_indices,
            hint,
        )

    def make_llvm(self, context=None) -> str:
        if self.use_gcu_opt and os.path.isfile(_GCU_COMPILER_OPT):
            return self._lower_via_gcu_opt()
        return f"{self.ll}"

    @cached_property
    def src(self) -> str:
        return textwrap.dedent(inspect.getsource(self.fn))
