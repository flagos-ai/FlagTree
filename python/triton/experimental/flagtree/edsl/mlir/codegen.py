import ast
from typing import Any, Dict, Final, List, Optional, Sequence
from typing_extensions import override

from mlir import ir
from mlir.dialects import func


class UnknownSymbolError(Exception):

    def __init__(self, name: str, *args, **kwargs) -> None:
        super().__init__(f"unknown symbol {name}", *args, **kwargs)


class EdslMLIRCodeGenerator(ast.NodeVisitor):

    def __init__(self, absfilename: str, lscope: Optional[Dict[str, Any]] = None, gscope: Optional[Dict[str,
                                                                                                        Any]] = None,
                 context: Optional[ir.Context] = None, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.absfilename: Final[str] = absfilename
        self.lscope: Final[Dict[str, Any]] = {} if lscope is None else lscope
        self.gscope: Final[Dict[str, Any]] = {} if gscope is None else gscope
        self.context: Final[ir.Context] = ir.Context() if context is None else context

    def call_function(self, fn, args: Sequence[Any]) -> Any:
        return fn(*args)

    def lookup(self, name: str) -> Optional[Any]:
        for scope in self.lscope, self.gscope:
            if ret := scope.get(name):
                return ret
        return None

    @override
    def visit_Assign(self, node: ast.Assign) -> Any:
        [target] = node.targets
        ret = self.visit(node.value)
        self.lscope[target.id] = ret
        return ret

    @override
    def visit_Attribute(self, node: ast.Attribute) -> None:
        ret: Any = self.visit(node.value)
        if ret is not None:
            ret = getattr(ret, node.attr)
        return ret

    @override
    def visit_Call(self, node: ast.Call) -> Any:
        with ir.Location.file(self.absfilename, node.lineno, node.col_offset):
            fn = self.visit(node.func)
            args: List[ir.Value] = [self.visit(arg) for arg in node.args]
            return self.call_function(fn, args)

    @override
    def visit_Constant(self, node: ast.Constant) -> Any:
        return node.value

    @override
    def visit_FunctionDef(self, node: ast.FunctionDef) -> func.FuncOp:

        def convert_annotation_to_type(tynode: ast.Tuple) -> ir.Type:
            [_, tynode] = tynode.slice.dims
            return ir.Type.parse(self.visit(tynode))

        with self.context, ir.Location.file(self.absfilename, node.lineno, node.col_offset):
            fnty: ir.FunctionType = ir.FunctionType.get(
                [convert_annotation_to_type(arg.annotation) for arg in node.args.args], [])
            fn: func.FuncOp = func.FuncOp(node.name, fnty, visibility="public")
            block: ir.Block = fn.add_entry_block()
            for k, arg in zip(map(lambda arg: arg.arg, node.args.args), block.arguments):
                self.lscope[k] = arg
            with ir.InsertionPoint(block):
                for stmt in node.body:
                    self.visit(stmt)
                func.return_([])
            return fn

    @override
    def visit_List(self, node: ast.List) -> List[Any]:
        ret = [self.visit(elt) for elt in node.elts]
        return ret

    @override
    def visit_Module(self, node: ast.Module) -> ir.Module:
        [func] = node.body
        with self.context, ir.Location.file(self.absfilename, 0, 0):
            module: ir.Module = ir.Module.create()
            with ir.InsertionPoint(module.body):
                self.visit(func)
            return module

    @override
    def visit_Name(self, node: ast.Name) -> Any:
        if ret := self.lookup(node.id):
            return ret
        else:
            raise UnknownSymbolError(node.id)
