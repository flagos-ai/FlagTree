from __future__ import annotations
from abc import abstractmethod
from typing import TYPE_CHECKING, Any, Final, List
from typing_extensions import override

from mlir import ir
from mlir.dialects import arith, func, llvm, scf
import inspect
import os

if TYPE_CHECKING:
    from .codegen import EdslMLIRCodeGenerator


class ExternalCall(object):

    def __init__(self, keyword: str, args: List[Any] = [], *_args, **_kwargs) -> None:
        super().__init__(*_args, **_kwargs)
        self.keyword: Final[str] = keyword
        self.args: List[Any] = [*args]

    @abstractmethod
    def build(self) -> func.FuncOp:
        ...

    @abstractmethod
    def call(self, codegen: EdslMLIRCodeGenerator) -> func.CallOp:
        ...

    def decl(self, codegen: EdslMLIRCodeGenerator) -> func.FuncOp:
        with ir.InsertionPoint.at_block_begin(codegen.module.body):
            funcop: func.FuncOp = codegen.decls.get(self.keyword) or self.build()
        codegen.decls[self.keyword] = funcop
        return funcop

    def global_string(self, val: str, codegen: EdslMLIRCodeGenerator) -> llvm.GlobalOp:
        key: str = f"TleRaw_PrintFormat{len(codegen.constants)}"
        with ir.InsertionPoint.at_block_begin(codegen.module.body):
            op: ir.Operation = codegen.constants.get(val) or llvm.mlir_global(
                ir.Type.parse(f"!llvm.array<{len(val.encode())} x i8>"), key,
                ir.Attribute.parse("#llvm.linkage<internal>"), value=ir.StringAttr.get(val))
        codegen.constants[val] = op
        return op


class VPrintf(ExternalCall):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__("vprintf", *args, **kwargs)

    @override
    def build(self) -> func.FuncOp:
        return func.FuncOp(
            self.keyword,
            ir.FunctionType.get([ir.Type.parse("!llvm.ptr"), ir.Type.parse("!llvm.ptr")],
                                [ir.IntegerType.get_signless(32)]), visibility="private")

    @override
    def call(self, codegen: EdslMLIRCodeGenerator) -> func.CallOp:
        [format, *args] = self.args
        funcop: func.FuncOp = self.decl(codegen)
        format: llvm.GlobalOp = self.global_string(format, codegen)
        fptr: llvm.AddressOfOp = llvm.AddressOfOp(ir.Type.parse("!llvm.ptr"), format.sym_name.value)
        struct: ir.Type = ir.Type.parse("!llvm.struct<({})>".format(", ".join(map(lambda arg: f"{arg.type}", args))))
        size: ir.Value = arith.constant(ir.IntegerType.get_signless(32), len(args))
        alloca: llvm.AllocaOp = llvm.alloca(ir.Type.parse("!llvm.ptr"), size, struct)
        for i, arg in enumerate(args):
            ptr: llvm.GEPOp = llvm.getelementptr(ir.Type.parse("!llvm.ptr"), alloca, [], [i], arg.type, 0)
            llvm.store(arg, ptr)
        return func.call([ir.IntegerType.get_signless(32)], ir.FlatSymbolRefAttr.get(funcop.name.value), [fptr, alloca])


def vprintf(*args) -> VPrintf:
    return VPrintf(args)


class Assert(ExternalCall):

    def __init__(self, cond, msg, file_name, func_name, line_no, *args, **kwargs) -> None:
        dependencies = [cond] + list(args)
        super().__init__("__assertfail", dependencies, **kwargs)
        self.cond = cond
        self.msg = msg
        self.file_name = file_name
        self.func_name = func_name
        self.line_no = line_no
        self.print_args = args

    @override
    def build(self) -> func.FuncOp:
        ptr_type = ir.Type.parse("!llvm.ptr")
        i32_type = ir.IntegerType.get_signless(32)
        i64_type = ir.IntegerType.get_signless(64)

        return func.FuncOp(self.keyword, ir.FunctionType.get([ptr_type, ptr_type, i32_type, ptr_type, i64_type], []),
                           visibility="private")

    @override
    def call(self, codegen: EdslMLIRCodeGenerator) -> Any:
        func_op = self.decl(codegen)

        true_const = arith.constant(ir.IntegerType.get_signless(1), 1)
        is_false = arith.xori(self.cond, true_const)

        if_op = scf.IfOp(is_false)
        with ir.InsertionPoint(if_op.then_block):

            debug_args = [self.msg]
            if self.print_args:
                debug_args.extend(self.print_args)
            VPrintf(debug_args).call(codegen)

            # 1. Message String
            msg_global = self.global_string(self.msg, codegen)
            msg_ptr = llvm.AddressOfOp(ir.Type.parse("!llvm.ptr"), msg_global.sym_name.value)

            # 2. File Name String
            file_global = self.global_string(self.file_name, codegen)
            file_ptr = llvm.AddressOfOp(ir.Type.parse("!llvm.ptr"), file_global.sym_name.value)

            # 3. Line Number (Integer)
            line_val = arith.constant(ir.IntegerType.get_signless(32), self.line_no)

            # 4. Function Name String
            func_global = self.global_string(self.func_name, codegen)
            func_ptr = llvm.AddressOfOp(ir.Type.parse("!llvm.ptr"), func_global.sym_name.value)

            # 5. Char Size
            char_size_val = arith.constant(ir.IntegerType.get_signless(64), 1)

            #__assertfail
            func.call([], ir.FlatSymbolRefAttr.get(func_op.name.value),
                      [msg_ptr, file_ptr, line_val, func_ptr, char_size_val])

            scf.yield_([])

        return if_op


def vassert(cond, fmt, *args):
    frame = inspect.currentframe().f_back
    try:
        filename = os.path.basename(frame.f_code.co_filename)  # 只取文件名，不要绝对路径
        funcname = frame.f_code.co_name
        lineno = frame.f_lineno
    finally:
        del frame

    return Assert(cond, fmt, filename, funcname, lineno, *args)
