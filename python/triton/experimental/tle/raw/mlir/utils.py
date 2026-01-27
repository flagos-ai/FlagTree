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
        # keyword 对应底层的符号名
        super().__init__("__assert_fail", *args, **kwargs)
        self.cond = cond
        # 保存元数据
        self.msg = msg
        self.file_name = file_name
        self.func_name = func_name
        self.line_no = line_no
        # args 依然保留，用于 vprintf 的动态参数打印
        self.print_args = args

    @override
    def build(self) -> func.FuncOp:
        """
        定义 __assert_fail 的函数签名，严格对齐 CUDA 标准。
        void __assert_fail(const char *message, const char *file, 
                           unsigned int line, const char *function, 
                           size_t charSize);
        """
        ptr_type = ir.Type.parse("!llvm.ptr") # i8*
        i32_type = ir.IntegerType.get_signless(32)
        i64_type = ir.IntegerType.get_signless(64)
        
        return func.FuncOp(
            self.keyword, 
            ir.FunctionType.get(
                [ptr_type, ptr_type, i32_type, ptr_type, i64_type], 
                []
            ),
            visibility="private"
        )

    @override
    def call(self, codegen: EdslMLIRCodeGenerator) -> Any:
        # 1. 获取函数声明
        func_op = self.decl(codegen)

        # 2. 条件取反 (Assert 是 "真则过"，底层是 "假则挂")
        true_const = arith.constant(ir.IntegerType.get_signless(1), 1)
        is_false = arith.xori(self.cond, true_const)

        # 3. 构建 If 结构 (对应 C++ 中的 splitBlock)
        if_op = scf.IfOp(is_false)
        with ir.InsertionPoint(if_op.then_block):
            
            # --- 步骤 A: 打印用户友好的动态信息 ---
            # 因为 __assert_fail 只能打印固定字符串，
            # 所以我们先调 vprintf 把变量值(args)打印出来给用户看
            if self.print_args:
                # 构造 vprintf 调用，传入格式化字符串和参数
                # 注意：这里我们重新把 self.msg 和 self.print_args 组合传给 VPrintf
                # 假设 VPrintf 接受 ([fmt, args...]) 形式
                VPrintf([self.msg, *self.print_args]).call(codegen)

            # --- 步骤 B: 准备 __assert_fail 的静态参数 ---
            
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
            
            # 5. Char Size (通常设为 1 或 sizeof(char))
            char_size_val = arith.constant(ir.IntegerType.get_signless(64), 1)

            # --- 步骤 C: 调用 __assert_fail ---
            func.call(
                [], 
                ir.FlatSymbolRefAttr.get(func_op.name.value), 
                [msg_ptr, file_ptr, line_val, func_ptr, char_size_val]
            )
            
            # 理论上 __assert_fail 不会返回，但在 MLIR 中加上 trap 更保险
            llvm.intr.trap()
            
            scf.yield_([])

        return if_op

# 对外接口 vassert
def vassert(cond, fmt, *args):
    # 自动获取调用者的位置信息
    frame = inspect.currentframe().f_back
    try:
        filename = os.path.basename(frame.f_code.co_filename) # 只取文件名，不要绝对路径
        funcname = frame.f_code.co_name
        lineno = frame.f_lineno
    finally:
        del frame # 避免循环引用内存泄漏

    return Assert(cond, fmt, filename, funcname, lineno, *args).call(EdslMLIRCodeGenerator.current)
