from __future__ import annotations
import ast
import copy
from functools import cached_property
import inspect
from typing import Any, Dict, Final, List, Optional

from mlir import ir
from mlir.passmanager import PassManager

from .codegen import EdslMLIRCodeGenerator


class EdslMLIRJITFunction(object):

    def __init__(self, fn: Any, pipeline: Optional[List[str]] = None, context: Optional[ir.Context] = None, *args,
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.fn: Final[Any] = fn
        self.pipeline: Final[List[str]] = [
            "convert-scf-to-cf", "finalize-memref-to-llvm", "convert-arith-to-llvm", "convert-cf-to-llvm",
            "convert-func-to-llvm", "convert-index-to-llvm", "convert-nvvm-to-llvm"
        ] if pipeline is None else pipeline
        self.context: Final[ir.Context] = ir.Context() if context is None else context

    def __deepcopy__(self, memo: Dict[int, Any]) -> EdslMLIRJITFunction:
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

    def make_ir(self) -> ir.Module:
        codegen: EdslMLIRCodeGenerator = EdslMLIRCodeGenerator(self.absfilename, {}, self.globals, self.context)
        return codegen.visit(self.ast)

    def make_llir(self, module: Optional[ir.Module] = None) -> ir.Module:
        mod: ir.Module = module if module is not None else self.make_ir()
        context: ir.Context = module.context if module is not None else self.context
        with context:
            pm: PassManager = PassManager()
            pm.enable_verifier(True)
            for p in self.pipeline:
                pm.add(p)
            pm.run(mod.operation)
            return mod

    @cached_property
    def src(self) -> str:
        return inspect.getsource(self.fn)
