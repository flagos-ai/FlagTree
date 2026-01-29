import ast
from typing import Any, Dict, Final, List, Optional, Sequence
from typing_extensions import override

from mlir import ir
from mlir.dialects import func

from .utils import ExternalCall


class UnknownSymbolError(Exception):

    def __init__(self, name: str, *args, **kwargs) -> None:
        super().__init__(f"unknown symbol {name}", *args, **kwargs)


class EdslMLIRCodeGenerator(ast.NodeVisitor):

    def __init__(self, absfilename: str, lscope: Dict[str, Any] = None, gscope: Dict[str, Any] = {},
                 context: Optional[ir.Context] = None, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.absfilename: Final[str] = absfilename
        self.lscope: Final[Dict[str, Any]] = {**lscope}
        self.gscope: Final[Dict[str, Any]] = {**gscope}
        self.decls: Final[Dict[str, func.FuncOp]] = {}
        self.constants: Final[Dict[str, Any]] = {}
        self.context: Final[ir.Context] = ir.Context() if context is None else context
        self.module: Optional[ir.Module] = None

    def call_function(self, fn, args: Sequence[Any], kwargs: Dict[str, Any]) -> Any:
        return fn(*args, **kwargs)

    def lookup(self, name: str) -> Optional[Any]:
        for scope in self.lscope, self.gscope:
            if ret := scope.get(name):
                return ret
        return None

    @override
    def visit_Assign(self, node: ast.Assign) -> Any:
        [target] = node.targets
        ret = self.visit(node.value)
        if isinstance(target, ast.Name):
            self.lscope[target.id] = ret
        elif isinstance(target, ast.Tuple) or isinstance(target, ast.List):
            for elt, val in zip(target.elts, ret):
                self.lscope[elt.id] = val
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
            kwargs: Dict[str, ir.Value] = {keyword.arg: self.visit(keyword.value) for keyword in node.keywords}
            ret = self.call_function(fn, args, kwargs)
            if isinstance(ret, ExternalCall):
                ret = ret.call(self)
            return ret

    @override
    def visit_Constant(self, node: ast.Constant) -> Any:
        return node.value

    @override
    def visit_For(self, node: ast.For) -> None:
        with ir.Location.file(self.absfilename, node.lineno, node.col_offset):
            for iters in self.visit(node.iter):
                if isinstance(node.target, ast.Name):
                    self.lscope[node.target.id] = iters
                elif isinstance(node.target, ast.Tuple):
                    assert len(node.target.elts) == len(iters)
                    for elt, iter in zip(node.target.elts, iters):
                        self.lscope[elt.id] = iter
                else:
                    raise NotImplementedError(f"unsupported for target type: {type(node.target)}")
                for stmt in node.body:
                    self.visit(stmt)

    @override
    def visit_FunctionDef(self, node: ast.FunctionDef) -> func.FuncOp:
        with self.context, ir.Location.file(self.absfilename, node.lineno, node.col_offset):
            operand_tys: List[ir.Type] = []
            output_tys: List[ir.Type] = []
            output_indices: List[int] = []
            for idx, arg in enumerate(node.args.args):
                # issue#328 [bug]edsl InOut&Input anno F722 error
                # https://github.com/flagos-ai/FlagTree/issues/328
                # use while find method to fix the bug, 
                # remember replace below arg.annotation.slice.value with type_str
                '''
                slice_node = arg.annotation.slice
                if isinstance(slice_node, ast.Subscript):
                    type_str = slice_node.slice.value
                else:
                    type_str = slice_node.value
                '''
                if arg.annotation.value.id == "InOut":
                    ty: ir.Type = ir.Type.parse(arg.annotation.slice.value)
                    operand_tys += [ty]
                    output_tys += [ty]
                    output_indices += [idx]
                elif arg.annotation.value.id == "Input":
                    ty: ir.Type = ir.Type.parse(arg.annotation.slice.value)
                    operand_tys += [ty]
                elif arg.annotation.value.id == "Num":
                    ty: ir.Type = ir.Type.parse(arg.annotation.slice.value)
                    operand_tys += [ty]
                else:
                    raise NotImplementedError(f"unsupported argument annotation: {ast.dump(arg.annotation)}")
            fnty: ir.FunctionType = ir.FunctionType.get(operand_tys, output_tys)
            fn: func.FuncOp = func.FuncOp(node.name, fnty, visibility="public")
            block: ir.Block = fn.add_entry_block()
            for k, arg in zip(map(lambda arg: arg.arg, node.args.args), block.arguments):
                self.lscope[k] = arg
            with ir.InsertionPoint(block):
                for stmt in node.body:
                    self.visit(stmt)
                func.return_([block.arguments[idx] for idx in output_indices])
            return fn

    @override
    def visit_List(self, node: ast.List) -> List[Any]:
        ret = [self.visit(elt) for elt in node.elts]
        return ret

    @override
    def visit_Module(self, node: ast.Module) -> ir.Module:
        [fn] = node.body
        with self.context, ir.Location.file(self.absfilename, 0, 0):
            self.module = ir.Module.create()
            with ir.InsertionPoint(self.module.body):
                self.visit(fn)
            return self.module

    @override
    def visit_Name(self, node: ast.Name) -> Any:
        if ret := self.lookup(node.id):
            return ret
        else:
            raise UnknownSymbolError(node.id)

    @override
    def visit_Subscript(self, node: ast.Subscript) -> Any:
        lhs = self.visit(node.value)
        slices = self.visit(node.slice)
        return lhs[slices]

    @override
    def visit_UnaryOp(self, node) -> Any:
        operand = self.visit(node.operand)
        if isinstance(node.op, ast.USub):
            return -operand
        else:
            raise NotImplementedError(f"unsupported unary op: {type(node.op)}")

    @override
    def visit_With(self, node: ast.With) -> None:
        [item] = node.items
        with self.visit(item.context_expr):
            for stmt in node.body:
                self.visit(stmt)
