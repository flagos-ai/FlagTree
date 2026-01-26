from __future__ import annotations
from abc import abstractmethod
from typing import TYPE_CHECKING, Any, Final, List
from typing_extensions import override
from hashlib import md5
from mlir import ir
from mlir.dialects import arith, func, llvm

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
           funcop: func.FuncOp = codegen.decls.get(self.keyword) or (self.build())
        codegen.decls[self.keyword] = funcop
        return funcop

    def global_string(self, val: str, codegen: EdslMLIRCodeGenerator) -> llvm.GlobalOp:
        key: str = f"globalstr{md5(val.encode("utf-8")).hexdigest()[:6]}"
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
        print("vprintf called")
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
