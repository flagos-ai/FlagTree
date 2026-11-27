# SPDX-FileCopyrightText: Copyright (c) 2025 SpacemiT. All rights reserved.
# SPDX-License-Identifier: MIT
"""LLVM-direct emitter: @smt_rvisa AST → top-level `llvm.func` module,
built with the upstream MLIR Python bindings (`mlir.dialects.llvm`).

All IR is constructed through bindings op builders — every op is verified by
MLIR at construction time and `Module.get_asm()` refuses custom-form printing
when the verifier fails, so malformed IR cannot slip through silently. There
is NO text emission path.

Why a parsed skeleton for the signature: `!llvm.func<...>` (LLVMFunctionType)
cannot be constructed from the Python bindings (no public wrapper; passing a
builtin FunctionType raises std::bad_cast), so the function *shell* (module
attributes + signature + empty body, plus the `@spine_grid` declaration in
sibling mode) is parsed once via `ir.Module.parse`. Only the shell is parsed;
the entire body is built with bindings.

Unlike SmtMLIRBuilderCodegen (which emits ops via the C++ builder API into a
`tle.dsl_region` that later inlines into a *func.func* — where LLVM ops trip
BufferDeallocation's "unknown memory side effects"), this backend produces a
standalone `llvm.func`. Pure llvm.func is a no-op for
BufferDeallocation/ConvertToScalableVector, so it bypasses spine-opt entirely
and feeds straight into mlir-translate → llc(riscv64).

ABI (matches build/.../driver.py `_launch`): each memref param is passed as
`(i64 rank, !llvm.ptr descriptor)`; scalars by value; then 3 trailing i32 =
gridX,gridY,gridZ (new spert ABI). Single-program mode ignores prog*.
"""
from __future__ import annotations

import ast
import inspect
import textwrap

from .codegen import _parse_signature
from ._mlir_loader import ensure_mlir_on_path

ensure_mlir_on_path()
try:
    from mlir import ir
    from mlir.dialects import llvm as dllvm
except ImportError as _e:  # pragma: no cover - environment guard
    raise ImportError(
        "smt_rvisa LLVM-direct emitter requires the MLIR Python bindings "
        "(importable 'mlir' package). They are vendored into "
        "triton/backends/<backend>/mlir_core by setup.py at package time; if that "
        "copy is missing, rebuild the wheel (third_party/spacemit/scripts/install_flagtree_plugin.sh) from an "
        "LLVM install that has python_packages/mlir_core.") from _e

# Descriptor struct for a memref arg. The driver (backend/driver.py `_launch`)
# passes each memref as (int64_t rank=0, void* &ptr_arg) where ptr_arg is a
# rank-0 StridedMemRefType<char,0> = {allocated_ptr, aligned_ptr, offset}.
# There are NO sizes/strides arrays (rank 0), so the data pointer is field [1]
# (aligned) and llvm_size is unavailable in this ABI.
_DESC = "!llvm.struct<(ptr, ptr, i64)>"

# Sentinel marking a dynamic index in GEPOp.rawConstantIndices
# (= std::numeric_limits<int32_t>::min(), LLVM::GEP dynamic-index convention).
_GEP_DYNAMIC = -2147483648

# llvm.icmp predicate "slt" (signed less-than), used by sr.range loops.
_ICMP_SLT = 2


def _parse_type(s: str) -> ir.Type:
    return ir.Type.parse(s)


def _const_i64_v(n: int) -> ir.Value:
    i64 = ir.IntegerType.get_signless(64)
    return dllvm.ConstantOp(i64, value=ir.IntegerAttr.get(i64, n)).result


class LLVMDirectCodegen:
    """Walk a @smt_rvisa fn and build a top-level `llvm.func` via bindings."""

    def __init__(self, sibling_abi: bool = False) -> None:
        self._env: dict[str, ir.Value] = {}  # py var -> SSA Value
        self._desc_cache: dict[str, ir.Value] = {}  # pyname -> loaded descriptor
        self._arch = '0xA064'
        self._num_threads = 4
        # sibling_abi=True: called from a func.func sibling (mixed mode). Each
        # memref param arrives as a single i64 (the aligned data pointer, cast
        # from index by the host bridge), recovered via llvm.inttoptr — NOT the
        # driver's (i64 rank, !llvm.ptr descriptor) pair. No trailing grid args;
        # one trailing ctx i64 instead. Proven by test_manual_mixed_ir.py.
        self._sibling_abi = sibling_abi
        self._ptr_i64: dict[str, ir.Value] = {}  # pyname -> i64 arg (data ptr)
        self._ctx_arg: ir.Value | None = None
        # build state (set up by _build_body)
        self._func = None  # the llvm.func Operation being filled
        self._cur_blk = None  # block that ops are appended to
        self._entry_args = None  # entry block arguments

    # --- emit helpers ---------------------------------------------------
    def _ip(self) -> ir.InsertionPoint:
        """Insertion point appending to the end of the current block."""
        return ir.InsertionPoint(self._cur_blk)

    @staticmethod
    def _elem_align(rt: ir.Type) -> int:
        """Byte alignment of a scalar/vector element type (vector<[4]xf16>->2).

        Takes the already-parsed result type and unwraps it structurally
        (VectorType.element_type down to the scalar); no text sniffing.
        """
        if rt is None:
            raise ValueError("llvm.load via call_intrinsic requires result_type=<type string>")
        t = rt
        while isinstance(t, ir.VectorType):
            t = ir.VectorType(t).element_type
        if isinstance(t, ir.FloatType):
            width = ir.FloatType(t).width
        elif isinstance(t, ir.IntegerType):
            width = ir.IntegerType(t).width
        else:
            width = 8  # index / unknown element: byte-granular, as before
        return width // 8

    # ------------------------------------------------------------------
    # Skeleton (signature shell) construction
    # ------------------------------------------------------------------
    def _skeleton(self, fn_name: str, arg_tys: list[str], *, module_attrs: str = "module ",
                  declare_spine_grid: bool = False) -> None:
        """Parse the module/func shell and set up build state.

        The shell is the ONLY parsed text (LLVMFunctionType has no Python
        constructor); the body is built entirely with bindings afterwards.
        """
        sig = ", ".join(f"%arg{i}: {t}" for i, t in enumerate(arg_tys))
        grid_decl = "llvm.func @spine_grid(%g0: i64, %g1: i64) -> i64\n  " if declare_spine_grid else ""
        src = (f"{module_attrs}{{\n  {grid_decl}llvm.func @{fn_name}({sig}) {{ llvm.return }}\n}}")
        self._module = ir.Module.parse(src)
        # last operation = our func (the grid decl, if any, comes first)
        self._func = list(self._module.body.operations)[-1]
        blk = self._func.regions[0].blocks[0]
        self._entry_args = list(blk.arguments)
        # Drop the skeleton terminator; the real llvm.return is emitted after
        # the body is generated.
        list(blk.operations)[-1].operation.erase()
        self._cur_blk = blk

    # ------------------------------------------------------------------
    # Public entry: @smt_rvisa fn -> module text with a top-level llvm.func
    # ------------------------------------------------------------------
    def emit_module(self, fn) -> str:
        params = _parse_signature(fn)
        self._params = params  # for program_id computation
        func_node = self._func_node(fn)

        # --- signature: memref -> (i64 rank, !llvm.ptr descriptor addr);
        #     scalar -> i64; then 3 trailing i32 (gridX/Y/Z, spert ABI) ---
        self._mem_ptr: dict[str, ir.Value] = {}
        arg_tys: list[str] = []
        arg_meta: list[tuple[str, str] | None] = []  # per arg: ("mem", pname) | None
        for pname, ann in params:
            if ann.kind == "mem":
                arg_tys += ["i64", "!llvm.ptr"]
                arg_meta += [None, ("mem", pname)]
            else:  # scalar (index) passed by value as i64
                arg_tys.append("i64")
                arg_meta.append(("scalar", pname))
        arg_tys += ["i32"] * 3  # num_programs gridX,gridY,gridZ
        arg_meta += [None] * 3

        hdr = ('module attributes {dlti.target_system_spec = '
               '#dlti.target_system_spec<"CPU" = #dlti.target_device_spec<'
               f'"arch_id" = "{self._arch}", "num_threads" = {self._num_threads} : i32>>, '
               'tt.force_vector_interleave = 2 : i32} ')
        with ir.Context(), ir.Location.unknown():
            self._skeleton(fn.__name__, arg_tys, module_attrs=hdr)
            for v, meta in zip(self._entry_args, arg_meta):
                if meta is not None and meta[0] == "mem":
                    self._mem_ptr[meta[1]] = v
                elif meta is not None:
                    self._env[meta[1]] = v
            self._gen_body(func_node)
            return self._module.operation.get_asm()

    # ------------------------------------------------------------------
    # Public entry: sibling llvm.func for mixed mode (no module wrapper)
    # ------------------------------------------------------------------
    def emit_func_for_inline(self, fn) -> tuple[str, list[str]]:
        """Build an llvm.func callable from a host func.func.

        Returns (func_text, param_types):
        - func_text: the llvm.func definition text (serialized via get_asm(),
          indented for splicing into the host module)
        - param_types: sibling ABI types, every param → "i64"
          (memref = data-ptr-as-i64, scalar = i64)
        """
        params = _parse_signature(fn)
        self._params = params
        func_node = self._func_node(fn)

        # Sibling ABI (called from func.func, see test_manual_mixed_ir.py):
        #   memref param → single i64 (aligned data ptr, cast from index by host)
        #   scalar param → single i64
        #   + one trailing ctx i64 (for program_id via spine_grid).
        arg_tys = ["i64"] * (len(params) + 1)
        param_types = ["i64"] * len(params)
        with ir.Context(), ir.Location.unknown():
            # The @spine_grid declaration must be present in the build module
            # for the llvm.call symbol reference to verify; the host-side
            # declaration is guaranteed by compiler._inject_mixed_llvm_llmlir.
            self._skeleton(fn.__name__, arg_tys, declare_spine_grid=True)
            for (pname, ann), v in zip(params, self._entry_args):
                if ann.kind == "mem":
                    self._ptr_i64[pname] = v
                else:  # scalar
                    self._env[pname] = v
            self._ctx_arg = self._entry_args[-1]
            self._gen_body(func_node)
            func_op = self._func
            # serialize just the func (module wrapper stripped)
            text = textwrap.indent(func_op.get_asm(), "  ")
        return text, param_types

    # ------------------------------------------------------------------
    @staticmethod
    def _func_node(fn) -> ast.FunctionDef:
        src = textwrap.dedent(inspect.getsource(fn))
        return next(n for n in ast.walk(ast.parse(src)) if isinstance(n, ast.FunctionDef))

    def _gen_body(self, func_node: ast.FunctionDef) -> None:
        for stmt in func_node.body:
            if isinstance(stmt, ast.Pass):
                continue
            if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant):
                continue  # docstring
            self._gen_stmt(stmt)
        with self._ip():
            dllvm.ReturnOp()

    # ------------------------------------------------------------------
    # Statements
    # ------------------------------------------------------------------
    def _gen_stmt(self, node) -> None:
        if isinstance(node, ast.Assign):
            assert len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
            v = self._gen_expr(node.value)
            self._env[node.targets[0].id] = v
        elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            self._gen_call(node.value)  # void call_intrinsic (vse etc.)
        elif isinstance(node, ast.For):
            self._gen_for(node)
        elif isinstance(node, (ast.Return, ast.Pass)):
            pass
        else:
            raise NotImplementedError(f"llvm-direct: unsupported stmt {ast.dump(node)}")

    def _gen_for(self, node: ast.For) -> None:
        """`for k in sr.range(lb[, ub, step]):` → llvm.br/cond_br loop.

        Vars reassigned in the body become iter-args (carried across the
        back-edge), mirroring host_ll.mlir's ^bb1(iv, acc...) header. Loop is
        [lb, ub) step. iv and iter-args are i64/typed block args.
        """
        assert isinstance(node.target, ast.Name)
        iv_name = node.target.id
        rargs = node.iter.args
        i64 = ir.IntegerType.get_signless(64)
        with self._ip():
            if len(rargs) == 1:
                lb = _const_i64_v(0)
                ub = self._gen_expr(rargs[0])
                step = _const_i64_v(1)
            else:
                lb = self._gen_expr(rargs[0])
                ub = self._gen_expr(rargs[1])
                step = self._gen_expr(rargs[2])

        # iter-args: vars defined before the loop and reassigned inside it
        outer = set(self._env)
        reassigned: list[str] = []
        for s in node.body:
            if isinstance(s, ast.Assign):
                for t in s.targets:
                    if isinstance(t, ast.Name) and t.id in outer and t.id not in reassigned:
                        reassigned.append(t.id)
        ia_init = [(v, self._env[v]) for v in reassigned]

        blocks = self._func.regions[0].blocks
        entry_blk = self._cur_blk
        header = blocks.append(*([i64] + [val.type for _, val in ia_init]))
        body = blocks.append()
        exit_ = blocks.append()

        # entry -> header with initial values
        with ir.InsertionPoint(entry_blk):
            dllvm.br([lb] + [val for _, val in ia_init], header)

        # header block: bind iv + iter-arg block args, test, cond_br
        hargs = list(header.arguments)
        iv_v = hargs[0]
        self._env[iv_name] = iv_v
        ia_hdr = list(zip(reassigned, hargs[1:]))
        for name, v in ia_hdr:
            self._env[name] = v
        with ir.InsertionPoint(header):
            cond = dllvm.icmp(_ICMP_SLT, iv_v, ub)
            dllvm.cond_br(cond, [], [], body, exit_)

        # body block
        self._cur_blk = body
        for s in node.body:
            self._gen_stmt(s)
        with self._ip():
            nxt = dllvm.add(iv_v, step, 0)  # 0 = no overflow flags
            dllvm.br([nxt] + [self._env[name] for name, _ in ia_hdr], header)

        # exit block: iter-args live on as their header block-arg values
        self._cur_blk = exit_
        for name, v in ia_hdr:
            self._env[name] = v

    # ------------------------------------------------------------------
    # Expressions -> ir.Value
    # ------------------------------------------------------------------
    def _gen_expr(self, node) -> ir.Value:
        if isinstance(node, ast.Call):
            v = self._gen_call(node)
            if v is None:
                raise ValueError("llvm-direct: void intrinsic used in expression position")
            return v
        if isinstance(node, ast.Name):
            return self._env[node.id]
        if isinstance(node, ast.Constant) and isinstance(node.value, int):
            with self._ip():
                return _const_i64_v(node.value)
        if isinstance(node, ast.BinOp):
            lv = self._gen_expr(node.left)
            rv = self._gen_expr(node.right)
            opn = {ast.Mult: dllvm.mul, ast.Add: dllvm.add, ast.Sub: dllvm.sub}.get(type(node.op))
            if opn is None:
                raise NotImplementedError(f"llvm-direct: unsupported binop {type(node.op).__name__}")
            with self._ip():
                return opn(lv, rv, 0)  # 0 = no overflow flags
        raise NotImplementedError(f"llvm-direct: unsupported expr {ast.dump(node)}")

    def _arg(self, node) -> ir.Value:
        """Resolve a call arg node to a Value.

        Bare string literals were a text-path escape hatch (raw SSA names);
        the bindings path has no such channel — every argument must be a
        real Value.
        """
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            raise ValueError("llvm-direct: string literal argument is not supported "
                             "(was a text-path passthrough); pass a Value-producing expression")
        return self._gen_expr(node)

    def _gen_call(self, node: ast.Call):
        fname = node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id
        h = getattr(self, f"_p_{fname}", None)
        if h is None:
            raise NotImplementedError(f"llvm-direct: unsupported primitive {fname!r}")
        return h(node)

    # ---- primitive handlers ------------------------------------------
    def _p_llvm_const(self, node):
        val = node.args[0].value
        typ_s = node.args[1].value
        typ = _parse_type(typ_s)
        with self._ip():
            if isinstance(typ, ir.VectorType):
                elem = ir.VectorType(typ).element_type
                if isinstance(elem, ir.FloatType):
                    eattr = ir.FloatAttr.get(elem, float(val))
                else:
                    eattr = ir.IntegerAttr.get(elem, int(val))
                return dllvm.ConstantOp(typ, value=ir.DenseElementsAttr.get_splat(typ, eattr)).result
            if isinstance(typ, ir.FloatType):
                return dllvm.ConstantOp(typ, value=ir.FloatAttr.get(typ, float(val))).result
            return dllvm.ConstantOp(typ, value=ir.IntegerAttr.get(typ, int(val))).result

    def _p_llvm_poison(self, node):
        typ = _parse_type(node.args[0].value)
        with self._ip():
            return dllvm.mlir_poison(typ)

    def _p_llvm_base_ptr(self, node):
        pname = node.args[0].id
        ptrT = _parse_type("!llvm.ptr")
        with self._ip():
            if self._sibling_abi:
                # Sibling ABI: memref arrives as a single i64 (aligned data
                # ptr). Recover the pointer via inttoptr, once, cached.
                base = self._desc_cache.get(pname)
                if base is None:
                    base = dllvm.inttoptr(ptrT, self._ptr_i64[pname])
                    self._desc_cache[pname] = base
                return base
            desc = self._desc_cache.get(pname)
            if desc is None:
                desc = dllvm.load(_parse_type(_DESC), self._mem_ptr[pname])
                self._desc_cache[pname] = desc
            return dllvm.extractvalue(ptrT, desc, [1])

    def _p_llvm_gep(self, node):
        base = self._gen_expr(node.args[0])
        off = self._gen_expr(node.args[1])
        elem_s = node.args[2].value if len(node.args) > 2 else "f16"
        ptrT = _parse_type("!llvm.ptr")
        with self._ip():
            return dllvm.GEPOp(ptrT, base, [off], [_GEP_DYNAMIC], _parse_type(elem_s), 0).result

    def _p_llvm_size(self, node):
        """llvm_size(mem[, dim]) — UNSUPPORTED in the llvm-direct driver ABI.

        The driver passes memrefs as rank-0 StridedMemRefType descriptors
        {allocated, aligned, offset} with no sizes/strides. There is no dim
        size to read. Pass shape info as an explicit scalar kernel parameter
        (e.g. K: sr.index) and use that for loop bounds instead.
        """
        raise NotImplementedError("llvm-direct: llvm_size is unavailable — the driver ABI passes rank-0 "
                                  "memref descriptors with no shape. Pass sizes as scalar params "
                                  "(e.g. K: sr.index) and use them for loop bounds.")

    def _p_program_id(self, node):
        """program_id(axis) -> i64 index of this program along `axis`.

        Sibling ABI (mixed mode): resolved at runtime via spine_grid(ctx, axis),
        the SAME lowering the host uses for tl.program_id (verified from a dumped
        _mv_sv_host_style2 ll.mlir: `%p = llvm.call @spine_grid(%arg0, %axis)`).
        The ctx handle is the trailing i64 arg the host bridge forwards (%arg0 of
        the host). Requires `llvm.func @spine_grid(i64, i64) -> i64` in the module;
        compiler._inject_mixed_llvm_llmlir guarantees the declaration is present.

        Standalone-module ABI (emit_module, no ctx): falls back to the trailing
        i32 grid arg (legacy single-module path, not multi-core mixed mode).
        """
        axis = node.args[0].value
        if axis not in (0, 1, 2):
            raise ValueError(f"program_id axis must be 0, 1, or 2; got {axis}")
        i64 = ir.IntegerType.get_signless(64)
        with self._ip():
            if self._sibling_abi:
                if self._ctx_arg is None:
                    raise RuntimeError("program_id in sibling mode requires a ctx arg; "
                                       "emit_func_for_inline must set codegen._ctx_arg.")
                ax = _const_i64_v(axis)
                return dllvm.CallOp(i64, [self._ctx_arg, ax], [], [], callee="spine_grid").result

            # Standalone module ABI: memref=2 args, scalar=1 arg; grid i32 trails.
            n_user_args = sum(2 if p[1].kind == "mem" else 1 for p in self._params)
            prog_v = self._entry_args[n_user_args + axis]
            return dllvm.sext(i64, prog_v)

    # Plain llvm-dialect ops reachable through sr.call_intrinsic (dispatch
    # table; anything unlisted fails loudly instead of guessing).
    _PLAIN_UNARY = {
        "llvm.fptrunc": staticmethod(lambda rt, ops: dllvm.fptrunc(rt, ops[0])),
        "llvm.fpext": staticmethod(lambda rt, ops: dllvm.fpext(rt, ops[0])),
    }
    _PLAIN_BINARY = {
        "llvm.fadd": staticmethod(lambda ops: dllvm.fadd(ops[0], ops[1])),
        "llvm.fsub": staticmethod(lambda ops: dllvm.fsub(ops[0], ops[1])),
        "llvm.fmul": staticmethod(lambda ops: dllvm.fmul(ops[0], ops[1])),
        "llvm.fdiv": staticmethod(lambda ops: dllvm.fdiv(ops[0], ops[1])),
    }

    def _p_call_intrinsic(self, node):
        intrin = node.args[0].value
        elts = node.args[1].elts
        rt_s = None
        for kw in node.keywords:
            if kw.arg == "result_type":
                rt_s = kw.value.value
        ops = [self._arg(e) for e in elts]
        rt = None if rt_s in (None, "()") else _parse_type(rt_s)

        with self._ip():
            # Native LLVM ops that have a real MLIR llvm-dialect op (NOT
            # llvm.call_intrinsic). Only ops with no MLIR equivalent
            # (llvm.riscv.*) stay wrapped in llvm.call_intrinsic, so the
            # emitted IR carries only the intended intrinsic calls.
            if intrin == "llvm.load":
                # llvm.load needs an explicit alignment (element size).
                return dllvm.load(rt, ops[0], alignment=self._elem_align(rt))
            if intrin == "llvm.store":
                dllvm.store(ops[0], ops[1])
                return None
            if intrin in ("llvm.vector.reduce.fadd", "llvm.intr.vector.reduce.fadd"):
                return dllvm.intr_vector_reduce_fadd(rt, ops[0], ops[1])
            if intrin in self._PLAIN_UNARY:
                return self._PLAIN_UNARY[intrin](rt, ops)
            if intrin in self._PLAIN_BINARY:
                return self._PLAIN_BINARY[intrin](ops)

            # Detect: plain LLVM op vs intrinsic (llvm.riscv.vle /
            # llvm.sadd.with.overflow). Heuristic (unchanged from the previous
            # emitter): a '.' after the leading "llvm." ⇒ intrinsic name.
            is_intrinsic = '.' in intrin[5:] if intrin.startswith("llvm.") else False
            if is_intrinsic:
                op = dllvm.CallIntrinsicOp(rt, intrin, ops, [], [])
                return None if rt is None else op.result
            raise NotImplementedError(f"llvm-direct: unsupported plain op {intrin!r} via call_intrinsic "
                                      f"(add an explicit bindings dispatch entry)")


def emit_llvm_func_for_inline(fn) -> tuple[str, list[str]]:
    """Emit an llvm.func that can be called from a host func.func.

    Unlike emit_llvm_direct_module (which wraps the llvm.func in a standalone
    module), this returns just the function text to be appended as a sibling
    in a mixed-mode module.

    Returns:
        (func_text, param_types) where:
        - func_text is the complete llvm.func definition (no module wrapper)
        - param_types is a list of MLIR type strings for the call site
          Format: ["i64", "i64", ...] (every sibling-ABI param is i64)
    """
    return LLVMDirectCodegen(sibling_abi=True).emit_func_for_inline(fn)


def emit_llvm_direct_module(fn) -> str:
    """Convenience: build a fresh codegen and return the module text."""
    return LLVMDirectCodegen().emit_module(fn)
