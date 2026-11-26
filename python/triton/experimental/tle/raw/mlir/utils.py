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
from abc import abstractmethod
import base64
from hashlib import blake2s
import inspect
import os
import re
from typing import TYPE_CHECKING, Any, Final, List
from typing_extensions import override
from triton.runtime.driver import driver

from mlir import ir
from mlir.dialects import arith, func, llvm, scf

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
        hdigest = blake2s(val.encode('utf-8'), digest_size=16)
        key: str = f"globalstr{base64.urlsafe_b64encode(hdigest.digest()).decode('ascii').rstrip('=')}"
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


def _is_hip_target() -> bool:
    forced = os.environ.get("TLE_RAW_PRINTF")  # "ockl" | "vprintf"
    if forced:
        return forced == "ockl"
    return driver.active.get_current_target().backend == "hip"


def _to_i64(val):
    i64 = ir.IntegerType.get_signless(64)
    f64 = ir.F64Type.get()
    s = str(val.type)
    if m := re.fullmatch(r"i(\d+)", s):
        return val if int(m.group(1)) == 64 else arith.extui(i64, val)
    if s == "f64":
        return arith.bitcast(i64, val)
    if s in ("f32", "f16", "bf16"):
        return arith.bitcast(i64, arith.extf(f64, val))
    if s == "index":
        return arith.index_cast(i64, val)
    if s.startswith("!llvm.ptr"):
        return llvm.ptrtoint(i64, val)
    raise NotImplementedError(f"vprintf: unsupported argument type {s}")


class AMDGPUPrintf(ExternalCall):
    """HIP device printf through the ROCm-Device-Libs hostcall (ockl)."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__("__ockl_printf_begin", *args, **kwargs)

    def build(self) -> func.FuncOp:
        i64 = ir.IntegerType.get_signless(64)
        return func.FuncOp(self.keyword, ir.FunctionType.get([i64], [i64]), visibility="private")

    @staticmethod
    def _decl(codegen, name, ins, outs):
        with ir.InsertionPoint.at_block_begin(codegen.module.body):
            fn = codegen.decls.get(name) or func.FuncOp(name, ir.FunctionType.get(ins, outs), visibility="private")
            codegen.decls[name] = fn
        return fn

    def call(self, codegen):
        fmt, *args = self.args
        i32, i64 = ir.IntegerType.get_signless(32), ir.IntegerType.get_signless(64)
        ptr = ir.Type.parse("!llvm.ptr")
        begin = self._decl(codegen, "__ockl_printf_begin", [i64], [i64])
        s_app = self._decl(codegen, "__ockl_printf_append_string_n", [i64, ptr, i64, i32], [i64])
        a_app = self._decl(codegen, "__ockl_printf_append_args", [i64, i32, *[i64] * 7, i32], [i64])
        sym = lambda f: ir.FlatSymbolRefAttr.get(f.name.value)
        c = lambda t, v: arith.constant(t, v)

        fmt_z = fmt + "\0"  # hostcall needs the NUL terminator
        g = self.global_string(fmt_z, codegen)
        fptr = llvm.AddressOfOp(ptr, g.sym_name.value)

        vals = [_to_i64(a) for a in args]
        desc = func.call([i64], sym(begin), [c(i64, 0)])
        desc = func.call([i64], sym(s_app), [desc, fptr, c(i64, len(fmt_z.encode())), c(i32, 0 if vals else 1)])
        zero = c(i64, 0)
        for off in range(0, len(vals), 7):
            chunk = vals[off:off + 7]
            pad = [zero] * (7 - len(chunk))
            last = int(off + 7 >= len(vals))
            desc = func.call([i64], sym(a_app), [desc, c(i32, len(chunk)), *chunk, *pad, c(i32, last)])
        return desc


def vprintf(*args):
    return AMDGPUPrintf(args) if _is_hip_target() else VPrintf(args)


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
        filename = os.path.basename(frame.f_code.co_filename)
        funcname = frame.f_code.co_name
        lineno = frame.f_lineno
    finally:
        del frame

    return Assert(cond, fmt, filename, funcname, lineno, *args)
