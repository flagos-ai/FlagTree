from __future__ import annotations
import ast
import copy
from functools import cached_property
import inspect
from typing import Any, Dict, Final, List, Optional

from mlir import ir
from mlir.passmanager import PassManager

from .codegen import MLIRCodeGenerator


class MLIRJITFunction(object):

    def __init__(self, fn: Any, pipeline: Optional[List[str]] = None, context: Optional[ir.Context] = None, *args,
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.fn: Final[Any] = fn
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
        self.__triton_builtin__: Final[bool] = True

    def __deepcopy__(self, memo: Dict[int, Any]) -> MLIRJITFunction:
        return self.__class__(copy.deepcopy(self.fn, memo), copy.deepcopy(self.pipeline, memo), self.context)

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
        return {k: v for k, v in self.fn.__globals__.items() if not k.startswith("__")}

    @cached_property
    def codegen(self) -> MLIRCodeGenerator:
        return MLIRCodeGenerator(self.absfilename, {}, self.globals, self.context)

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

    @cached_property
    def src(self) -> str:
        return inspect.getsource(self.fn)
