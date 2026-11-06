# Copyright 2025-     FlagOS Contributors
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import annotations
import ast
import copy
from functools import cached_property
import inspect
from typing import Any, Dict, Final, List, Optional

from mlir import ir
from mlir.passmanager import PassManager

from .codegen import MLIRCodeGenerator
from triton.experimental.tle.raw.runtime import RawJITFunction


class MLIRJITFunction(RawJITFunction):

    def __init__(self, fn: Any, pipeline: Optional[List[str]] = None, context: Optional[ir.Context] = None, *args,
                 **kwargs) -> None:
        super().__init__(fn, **kwargs)
        if self.deferred:
            raise RuntimeError("Deferred MLIR raw is not supported; use eager mode")
        self.absfilename: Final[str] = inspect.getabsfile(self.fn)
        self.pipeline: Final[List[str]] = ([*pipeline] if pipeline is not None else [
            "convert-scf-to-cf",
            "finalize-memref-to-llvm",
            "convert-arith-to-llvm",
            "convert-cf-to-llvm",
            "convert-func-to-llvm",
            "convert-index-to-llvm",
            "convert-nvvm-to-llvm",
            "cse",
        ])
        self.context: Final[ir.Context] = ir.Context() if context is None else context
        self.region_dialect: Final[str] = "mlir"
        self.arg_dialect: Final[str] = "llvm"

    def __deepcopy__(self, memo: Dict[int, Any]) -> MLIRJITFunction:
        return self.__class__(copy.deepcopy(self.fn, memo), copy.deepcopy(self.pipeline, memo), self.context)

    @cached_property
    def ast(self) -> ast.Module:
        return ast.parse(self.src)

    @cached_property
    def fnname(self) -> str:
        return self.fn.__name__

    @cached_property
    def globals(self) -> Dict[str, Any]:
        return {k: v for k, v in self.fn.__globals__.items() if not k.startswith("__")}

    @cached_property
    def codegen(self) -> MLIRCodeGenerator:
        return MLIRCodeGenerator(self.absfilename, {}, self.globals, self.context, func_name_override=self.fnname)

    @property
    def ir(self) -> ir.Module:
        mod: ir.Module = self.codegen.visit(self.ast)
        return mod

    @property
    def ll(self) -> ir.Module:
        mod: ir.Module = self.ir
        with self.context:
            pm: PassManager = PassManager()
            pm.enable_verifier(True)
            for p in self.pipeline:
                pm.add(p)
            pm.run(mod.operation)
            return mod

    def make_llvm(self, context=None) -> str:
        return f"{self.ll}"

    def create_region_deferred(self, *args, **kwargs):
        raise RuntimeError("Deferred MLIR raw is not supported, including tensor fragment arguments")

    def create_region_by_llvm(self, builder, llvm: str, handles, alias_indices, hint: str = "",
                              extern_func_name: str = ""):
        return super().create_region_by_llvm(builder, llvm, handles, alias_indices, hint, extern_func_name)

    @cached_property
    def src(self) -> str:
        return inspect.getsource(self.fn)
