# SPDX-FileCopyrightText: Copyright (c) 2025 SpacemiT. All rights reserved.
# SPDX-License-Identifier: MIT
"""SmtMLIRBuilderCodegen — translates @smt_rvisa Python functions to MLIR ops.

Phase 1: AST visitor for the smt_rvisa eDSL subset.
Builds vector/arith/memref/scf ops straight through the C++ builder API
(create_tle_dsl_region_direct), with no MLIR text emission.
"""
from __future__ import annotations

import ast
import inspect
import textwrap
from typing import Callable

from .builtins import mma_cube as _mma_cube
from .types import (
    _TypedAnnotation,
    Ty,
    ScalarTy,
    VecTy,
    TensorTy,
    MemTy,
    StridedLayout,
    classify_type,
    BRIDGE,
    INDEX,
    I1,
    I64,
    F32,
    LLVM_PTR,
    LLVM_DESC,
    VOID,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_signature(fn: Callable) -> list[tuple[str, _TypedAnnotation]]:
    sig = inspect.signature(fn)
    result = []
    for pname, param in sig.parameters.items():
        ann = param.annotation
        if ann is inspect.Parameter.empty:
            raise ValueError(f"Parameter '{pname}' of @smt_rvisa function '{fn.__name__}' "
                             f"must have a sr.mem(...) or sr.index annotation.")
        if not isinstance(ann, _TypedAnnotation):
            raise ValueError(f"Parameter '{pname}' annotation must be sr.mem(...) or sr.index, got {ann!r}")
        result.append((pname, ann))
    return result


def _find_reassigned(body: list, outer_vars: set) -> set:
    """Variables in outer_vars that are assigned inside body (direct stmts only)."""
    found = set()
    for stmt in body:
        if isinstance(stmt, ast.Assign):
            for t in stmt.targets:
                if isinstance(t, ast.Name) and t.id in outer_vars:
                    found.add(t.id)
    return found


_SMT_RVISA_BUILTIN_NAMES = {
    "range",
    "proton_mark",
    "vconfig",
    "vzero",
    "vload",
    "vmacc",
    "vreduce_sum",
    "vreduce_max",
    "vreduce_min",
    "vreduce_mul",
    "vstore",
    "alloc",
    "pack",
    "vmadot",
    "vpack",
    "vbroadcast",
    "vshape",
    "spread",
    "imin",
    "vmin",
    "vmax",
    "sqrt",
    "rsqrt",
    "vexp",
    "vlog",
    "sload",
    "sstore",
    "viota",
    "abs",
    "cast",
    "select",
    # LLVM-direct LLVM-dialect primitives (call_intrinsic full-channel kernels)
    "call_intrinsic",
    "llvm_poison",
    "llvm_const",
    "llvm_base_ptr",
    "llvm_gep",
    "llvm_size",
}

# §6.4 binary operators → (float arith op, int arith op). None = not defined for
# that domain (e.g. bitwise on floats, true division on ints).
_BINOP_ARITH = {
    ast.Add: ("addf", "addi"),
    ast.Sub: ("subf", "subi"),
    ast.Mult: ("mulf", "muli"),
    ast.Div: ("divf", None),  # a / b  true division (float) → vfdiv
    ast.FloorDiv: (None, "divsi"),  # a // b integer floor division → vdiv
    ast.Mod: ("remf", "remsi"),  # a % b remainder → vrem
    ast.BitAnd: (None, "andi"),
    ast.BitOr: (None, "ori"),
    ast.BitXor: (None, "xori"),
    ast.LShift: (None, "shli"),
    ast.RShift: (None, "shrsi"),
}

# §6.4 comparisons → (arith.cmpf predicate, arith.cmpi predicate). Signed int.
_CMP_PRED = {
    ast.Lt: ("olt", "slt"),
    ast.LtE: ("ole", "sle"),
    ast.Gt: ("ogt", "sgt"),
    ast.GtE: ("oge", "sge"),
    ast.Eq: ("oeq", "eq"),
    ast.NotEq: ("one", "ne"),
}

# RVV vector config (SPEC §6.1). VLEN is the physical scalable-register width;
# SEW is the element width derived from the dtype (SPEC §3.1), not a vconfig
# parameter. The svector eDSL's element granularity is f16 (all svector kernels
# load f16 and count VL in f16 elements); f32 accumulators are the same element
# count at a wider LMUL group. VLMAX = lmul * VLEN / SEW.
_VLEN_BITS = 1024  # K3 scalable register width
_BASE_SEW_BITS = 16  # f16 element width (SPEC §3.1); the svector loop's VL granularity


def _vlmax(lmul: int, sew_bits: int = _BASE_SEW_BITS) -> int:
    """VLMAX (element count) for K3 at the given LMUL, per SPEC §6.1.

    VLMAX = lmul * VLEN / SEW. With VLEN=1024 and SEW=16 (f16):
      lmul=1 -> 64, lmul=2 -> 128, lmul=4 -> 256, lmul=8 -> 512.
    """
    return lmul * _VLEN_BITS // sew_bits


def _is_smt_rvisa_attr(node, attr: str, aliases: set | None = None) -> bool:
    """Check if node is <alias>.<attr> where alias is a smt_rvisa module import."""
    if not (isinstance(node, ast.Attribute) and node.attr == attr):
        return False
    if not isinstance(node.value, ast.Name):
        return False
    if aliases is not None:
        return node.value.id in aliases
    # fallback: accept any name when aliases not provided
    return True


_DTYPE_NAMES = {"f16", "f32", "bf16", "f64", "i8", "i16", "i32"}


def _resolve_dtype(node, default: str = "f16") -> str:
    """Resolve a dtype arg written as a string literal ("f16") or a bare name
    (f16 / f32 / bf16, the module-level dtype constants) to its MLIR string."""
    if node is None:
        return default
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.Name) and node.id in _DTYPE_NAMES:
        return node.id
    # last resort: literal_eval (raises for anything unexpected)
    return ast.literal_eval(node)


def _is_vec2d_of(t: Ty, elems: tuple) -> bool:
    """Rank-2 static-shape vector with a scalar element in `elems` (vmadot guard)."""
    return (isinstance(t, VecTy) and len(t.dims) == 2 and all(isinstance(d, int) for d in t.dims)
            and isinstance(t.elem, ScalarTy) and t.elem.name in elems)


# ---------------------------------------------------------------------------
# Builder-API codegen  (no string emission)
# ---------------------------------------------------------------------------


class SmtMLIRBuilderCodegen:
    """Translate @smt_rvisa fn → C++ builder API calls.

    generate_builder(fn) → (param_tys, body_builder)
    body_builder(b, block_args) is the callback for create_tle_dsl_region_direct.
    """

    def __init__(self):
        self._b = None
        self._env: dict[str, tuple] = {}
        self._const_index_cache: dict[int, object] = {}
        self._const_int_typed_cache: dict[tuple, object] = {}
        self._const_float_cache: dict[tuple, object] = {}
        self._constexpr_ints: dict[str, int] = {}
        self._constexpr_floats: dict[str, float] = {}
        self._active_vl: int | None = None
        self._active_valid = None
        self._aliases: set[str] = set()
        self._all_iter_arg_names: set[str] = set()
        self._loop_iter_args: set[str] = set()

    # --- Type helpers ---

    def _tt(self, ty: Ty):
        """Materialise a structured Ty at the C++ builder boundary."""
        return ty.build(self._b)

    def _user_ty(self, text: str) -> Ty:
        """Classify an LLVM-channel user type literal (e.g. "vector<[8]xf16>",
        "!llvm.ptr", "()") into a structured Ty.

        The literal is the only type *text* kernel authors write; it goes to
        the official MLIR parser (``builder.parse_type``) and is re-derived
        structurally by ``classify_type`` — never parsed in Python. "()" (void)
        has no MLIR type, so it is detected by text before parsing.
        """
        if text.strip() == "()":
            return VOID
        T = self._b.parse_type(text)
        if T is None:
            raise ValueError(f"_user_ty: MLIR parser rejected type literal {text!r}")
        return classify_type(self._b, T)

    # --- Constant helpers (no caching — caches cause dominance violations across regions) ---

    def _const_int(self, n: int):
        return self._b.create_arith_constant_index(n)

    def _const_int_typed(self, n: int, elem: ScalarTy):
        return self._b.create_arith_constant_int(n, self._tt(elem))

    def _const_float(self, v: float, ftype: ScalarTy = F32):
        return self._b.create_arith_constant_float(v, self._tt(ftype))

    # --- Env helpers ---

    def _bind(self, name: str, val, typ: Ty):
        self._env[name] = (val, typ)

    def _get(self, name: str) -> tuple:
        if name not in self._env:
            raise ValueError(f"Undefined variable: {name!r}")
        return self._env[name]

    def _require_vl(self) -> int:
        if self._active_vl is None:
            raise ValueError("smt_rvisa svector op used before vconfig() set VL")
        return self._active_vl

    def _try_const_int(self, node) -> int | None:
        if isinstance(node, ast.Constant) and isinstance(node.value, int):
            return node.value
        # Runtime params/locals (bound in _env) are never constants; a host
        # module global sharing the name must not be folded in their place
        # (see _gen_expr Name resolution for the same hazard).
        if isinstance(node, ast.Name) and node.id in self._env:
            return None
        if isinstance(node, ast.Name) and node.id in self._constexpr_ints:
            return self._constexpr_ints[node.id]
        if isinstance(node, ast.BinOp):
            l = self._try_const_int(node.left)
            r = self._try_const_int(node.right)
            if l is None or r is None:
                return None
            if isinstance(node.op, ast.Add): return l + r
            if isinstance(node.op, ast.Sub): return l - r
            if isinstance(node.op, ast.Mult): return l * r
            if isinstance(node.op, ast.FloorDiv): return l // r
        return None

    # --- broadcast helper ---

    def _broadcast_to(self, val, vec_ty: VecTy):
        return self._b.create_vector_broadcast(val, self._tt(vec_ty))

    def _match_operands(self, lv, lt: Ty, rv, rt: Ty):
        lv_is = isinstance(lt, VecTy)
        rv_is = isinstance(rt, VecTy)
        if lv_is and rv_is:
            if lt != rt:
                raise NotImplementedError(f"mismatched vectors {lt}/{rt}")
            return lv, rv, lt
        if lv_is and not rv_is:
            return lv, self._broadcast_to(rv, lt), lt
        if rv_is and not lv_is:
            return self._broadcast_to(lv, rt), rv, rt
        return lv, rv, None

    def _scalar_index_to_float(self, v, ftype: ScalarTy):
        """index → ftype scalar: index_cast to i64, then sitofp. `index` is not
        an integer type in MLIR so sitofp can't take it directly."""
        i64_v = self._b.create_arith_index_cast(v, self._tt(I64))
        return self._b.create_arith_sitofp(i64_v, self._tt(ftype))

    def _promote_scalar_pair(self, lv, lt: Ty, rv, rt: Ty):
        """Promote a pair of scalar operands to a common type, returning
        (lv, rv, result_type). Handles index↔float mixes (mean = sum / N)
        by lifting index to the float side; identical types pass through."""
        if lt == rt:
            return lv, rv, lt
        l_f = isinstance(lt, ScalarTy) and lt.is_float
        r_f = isinstance(rt, ScalarTy) and rt.is_float
        if l_f and rt == INDEX:
            return lv, self._scalar_index_to_float(rv, lt), lt
        if r_f and lt == INDEX:
            return self._scalar_index_to_float(lv, rt), rv, rt
        if l_f and r_f:
            # differing float widths: widen the narrower to the wider
            wide = lt if lt.bits >= rt.bits else rt
            if lt != wide:
                lv = self._b.create_arith_extf(lv, self._tt(wide))
            if rt != wide:
                rv = self._b.create_arith_extf(rv, self._tt(wide))
            return lv, rv, wide
        raise NotImplementedError(f"scalar promote between {lt!r} and {rt!r}")

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def generate_builder(self, fn):
        """Return (param_tys, body_builder) for create_tle_dsl_region_direct.

        ``param_tys`` are structured Ty objects; call_registry materialises
        them through ``Ty.build(builder)`` — no type-text channel.
        """
        src = textwrap.dedent(inspect.getsource(fn))
        tree = ast.parse(src)
        func_nodes = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
        if not func_nodes:
            raise ValueError(f"No function definition in {fn.__name__!r}")
        func_node = func_nodes[0]

        params = _parse_signature(fn)
        param_tys = [ann.ty for _, ann in params]

        # Detect smt_rvisa module aliases
        try:
            import smt_rvisa as _sr_mod
        except ModuleNotFoundError:
            try:
                from triton.language.extra import smt_rvisa as _sr_mod
            except (ModuleNotFoundError, ImportError):
                _sr_mod = None
        aliases: set[str] = set()
        if _sr_mod is not None:
            for k, v in (fn.__globals__ or {}).items():
                if v is _sr_mod:
                    aliases.add(k)
        if not aliases:
            aliases = {"smt_rvisa", "sr"}

        # Closure / global constexprs
        freevars: dict[str, object] = {}
        if getattr(fn, "__closure__", None):
            for nm, cell in zip(fn.__code__.co_freevars, fn.__closure__):
                try:
                    freevars[nm] = cell.cell_contents
                except ValueError:
                    pass
        constexpr_ints: dict[str, int] = {}
        constexpr_floats: dict[str, float] = {}
        for nm, val in {**(fn.__globals__ or {}), **freevars}.items():
            if isinstance(val, int) and not isinstance(val, bool):
                constexpr_ints.setdefault(nm, val)
            elif isinstance(val, float):
                constexpr_floats.setdefault(nm, val)

        # Pre-scan: find iter_args for all top-level for loops
        all_iter_arg_names: set[str] = set()
        defined_so_far: set[str] = set(p for p, _ in params)
        for stmt in func_node.body:
            if isinstance(stmt, ast.Assign):
                for t in stmt.targets:
                    if isinstance(t, ast.Name):
                        defined_so_far.add(t.id)
            elif isinstance(stmt, ast.For):
                all_iter_arg_names |= _find_reassigned(stmt.body, defined_so_far)

        def body_builder(b, block_args):
            # Fresh state for each invocation
            self.__init__()
            self._b = b
            self._aliases = aliases
            self._constexpr_ints = dict(constexpr_ints)
            self._constexpr_floats = dict(constexpr_floats)
            self._all_iter_arg_names = all_iter_arg_names
            # Bind params to block args
            for (pname, _ann), pty, barg in zip(params, param_tys, block_args):
                self._env[pname] = (barg, pty)
            # Generate body statements
            for stmt in func_node.body:
                if isinstance(stmt, ast.Pass):
                    continue
                if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant):
                    continue
                self._gen_stmt(stmt)

        return param_tys, body_builder

    # ------------------------------------------------------------------
    # Statement generators
    # ------------------------------------------------------------------

    def _gen_stmt(self, node):
        if isinstance(node, ast.Assign):
            self._gen_assign(node)
        elif isinstance(node, ast.For):
            self._gen_for(node)
        elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            self._gen_call_stmt(node.value)
        elif isinstance(node, (ast.Return, ast.Pass)):
            pass
        else:
            raise NotImplementedError(f"Unsupported stmt: {ast.dump(node)}")

    def _gen_assign(self, node: ast.Assign):
        if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
            raise NotImplementedError(f"smt_rvisa: only single-name assignment supported, got {ast.dump(node)}")
        target = node.targets[0].id
        if isinstance(node.value, ast.Call) and \
                _is_smt_rvisa_attr(node.value.func, "vconfig", self._aliases):
            self._gen_vconfig_assign(target, node.value)
            return
        val, typ = self._gen_expr(node.value)
        self._bind(target, val, typ)

    def _gen_for(self, node: ast.For):
        assert isinstance(node.target, ast.Name)
        loop_var = node.target.id
        assert _is_smt_rvisa_attr(node.iter.func, "range", self._aliases)
        rargs = node.iter.args
        assert len(rargs) in (1, 3)
        if len(rargs) == 1:
            lb = self._const_int(0)
            ub, _ = self._gen_expr(rargs[0])
            step = self._const_int(1)
        else:
            lb, _ = self._gen_expr(rargs[0])
            ub, _ = self._gen_expr(rargs[1])
            step, _ = self._gen_expr(rargs[2])

        outer_vars = set(self._env.keys())
        reassigned = sorted(_find_reassigned(node.body, outer_vars))
        ia_data = [(v, *self._get(v)) for v in reassigned]  # (name, val, ty)
        ia_vals = [d[1] for d in ia_data]

        prev_loop_iter = self._loop_iter_args
        self._loop_iter_args = set(reassigned)
        # vconfig inside the body sets _active_vl/_active_valid; _active_valid can
        # be a region-internal SSA (arith.minsi on N-i). Snapshot and restore so a
        # tail-loop vconfig doesn't leak that value into sibling loops → otherwise
        # a later vload references an SSA from a dead sibling region ('operand does
        # not dominate ... neither in a parent nor in a child region').
        saved_active_vl = self._active_vl
        saved_active_valid = self._active_valid
        # Snapshot full env before body: variables assigned inside the body but
        # not yielded (e.g. loop-local temporaries like `v = vload(...)`) would
        # otherwise leak into the outer scope with SSA values defined in the
        # child region → dominance violations when a later loop picks them up as
        # iter_arg inits.
        saved_env = dict(self._env)

        def for_body(b, iv, region_iter_args):
            # Rebind iter_args to their region block args
            saved = {}
            for d, ria in zip(ia_data, region_iter_args):
                v, _, ts = d
                saved[v] = self._env.get(v)
                self._env[v] = (ria, ts)
            self._env[loop_var] = (iv, INDEX)
            for stmt in node.body:
                self._gen_stmt(stmt)
            yield_vals = [self._get(v)[0] for v in reassigned]
            # Restore env to pre-body state: drop loop-local temporaries, restore
            # iter_arg names to their pre-loop values (re-bound below to results).
            self._env.clear()
            self._env.update(saved_env)
            return yield_vals

        result_vals = self._b.create_scf_for(lb, ub, step, ia_vals, for_body)
        self._loop_iter_args = prev_loop_iter
        # Restore active vconfig state clobbered inside the body.
        self._active_vl = saved_active_vl
        self._active_valid = saved_active_valid

        # Bind results back to the iter_arg names
        for d, rv in zip(ia_data, result_vals):
            v, _, ts = d
            self._env[v] = (rv, ts)

    def _gen_call_stmt(self, node: ast.Call):
        if _is_smt_rvisa_attr(node.func, "proton_mark", self._aliases):
            pass  # skip profiling marks in builder path
        elif _is_smt_rvisa_attr(node.func, "vconfig", self._aliases):
            # Bare `sr.vconfig(avl, lmul)` statement: set active VL/valid
            # without binding a name (same semantics as the assignment form).
            self._apply_vconfig(node)
        elif _is_smt_rvisa_attr(node.func, "vstore", self._aliases):
            self._gen_vstore(node)
        elif _is_smt_rvisa_attr(node.func, "sstore", self._aliases):
            self._gen_sstore(node)
        elif _is_smt_rvisa_attr(node.func, "pack", self._aliases):
            self._gen_pack(node)
        elif _is_smt_rvisa_attr(node.func, "call_intrinsic", self._aliases):
            self._gen_call_intrinsic(node)  # void form (e.g. llvm.riscv.vse)
        else:
            raise NotImplementedError(f"Unsupported call stmt: {ast.dump(node.func)}")

    # ------------------------------------------------------------------
    # Expression generators
    # ------------------------------------------------------------------

    def _gen_expr(self, node, hint: str = "") -> tuple:
        if isinstance(node, ast.Name):
            # Kernel params and locals take precedence over the host-module
            # global constexpr capture. Params are bound to block args in _env;
            # locals (e.g. nvl after `nvl = sr.vconfig(...)`) are recorded in
            # _constexpr_ints by _gen_vconfig_assign. A test-script global that
            # happens to share a name (`N = 64` loop variable in __main__) must
            # never shadow these — otherwise N is baked to the first value and
            # scf.for bounds fold away (N=64 partial-sum bug).
            if node.id in self._env:
                return self._env[node.id]
            if node.id in self._constexpr_ints:
                return self._const_int(self._constexpr_ints[node.id]), INDEX
            if node.id in self._constexpr_floats:
                return self._const_float(self._constexpr_floats[node.id]), F32
            return self._get(node.id)
        if isinstance(node, ast.Constant):
            v = node.value
            if isinstance(v, int): return self._const_int(v), INDEX
            if isinstance(v, float): return self._const_float(v), F32
            raise NotImplementedError(f"Unsupported literal: {v!r}")
        if isinstance(node, ast.BinOp):
            return self._gen_binop(node)
        if isinstance(node, ast.UnaryOp):
            return self._gen_unaryop(node)
        if isinstance(node, ast.Compare):
            return self._gen_compare(node)
        if isinstance(node, ast.Call):
            return self._gen_call_expr(node)
        raise NotImplementedError(f"Unsupported expr: {ast.dump(node)}")

    def _gen_binop(self, node: ast.BinOp) -> tuple:
        lv, lt = self._gen_expr(node.left)
        rv, rt = self._gen_expr(node.right)
        op = type(node.op)
        if lt == INDEX and rt == INDEX:
            m = {ast.Add: "addi", ast.Mult: "muli", ast.Sub: "subi", ast.FloorDiv: "divui"}
            opname = m.get(op)
            if opname is None:
                raise NotImplementedError(f"BinOp {op.__name__} on index")
            fn = getattr(self._b, f"create_arith_{opname}")
            return fn(lv, rv), INDEX
        # Scalar arithmetic (neither operand a vector). Covers reduce-then-scale
        # (mean = vreduce_sum(v) / N): promote index→f32 so a f32 scalar and an
        # index (e.g. row count N) can divide/multiply. _match_operands only
        # broadcasts scalars into vectors, so scalar×scalar must be handled here.
        if not isinstance(lt, VecTy) and not isinstance(rt, VecTy):
            lv, rv, st = self._promote_scalar_pair(lv, lt, rv, rt)
            is_f = isinstance(st, ScalarTy) and st.is_float
            arith = _BINOP_ARITH.get(op)
            if arith is None:
                raise NotImplementedError(f"Operator {op.__name__} not in _BINOP_ARITH")
            opname = arith[0] if is_f else arith[1]
            if opname is None:
                raise NotImplementedError(f"Operator {op.__name__} not defined for scalar {'float' if is_f else 'int'}")
            fn = getattr(self._b, f"create_arith_{opname}")
            return fn(lv, rv), st
        lv, rv, vt = self._match_operands(lv, lt, rv, rt)
        if vt is None:
            raise NotImplementedError(f"BinOp between {lt!r} and {rt!r}")
        elem = vt.elem
        is_f = elem.is_float
        arith = _BINOP_ARITH.get(op)
        if arith is None:
            raise NotImplementedError(f"Operator {op.__name__} not in _BINOP_ARITH")
        opname = arith[0] if is_f else arith[1]
        if opname is None:
            raise NotImplementedError(f"Operator {op.__name__} not defined for {'float' if is_f else 'int'}")
        fn = getattr(self._b, f"create_arith_{opname}")
        return fn(lv, rv), vt

    def _gen_unaryop(self, node: ast.UnaryOp) -> tuple:
        vv, vt = self._gen_expr(node.operand)
        # Scalar negation (e.g. -1e38 as fill= argument, or -mean in a formula)
        if not isinstance(vt, VecTy):
            if isinstance(node.op, ast.USub) and isinstance(vt, ScalarTy) and vt.is_float:
                return self._b.create_arith_negf(vv), vt
            raise NotImplementedError(f"unary on non-vector {vt}")
        elem = vt.elem
        if isinstance(node.op, ast.USub):
            if elem.is_float:
                return self._b.create_arith_negf(vv), vt
            zero = self._broadcast_to(self._const_int_typed(0, elem), vt)
            return self._b.create_arith_subi(zero, vv), vt
        if isinstance(node.op, ast.Invert):
            if elem.is_float:
                raise NotImplementedError(f"~a not defined for float {elem}")
            ones = self._broadcast_to(self._const_int_typed(-1, elem), vt)
            return self._b.create_arith_xori(vv, ones), vt
        raise NotImplementedError(f"Unary {type(node.op).__name__}")

    def _gen_compare(self, node: ast.Compare) -> tuple:
        if len(node.ops) != 1:
            raise NotImplementedError("chained comparison not supported")
        lv, lt = self._gen_expr(node.left)
        rv, rt = self._gen_expr(node.comparators[0])
        op = type(node.ops[0])
        lv, rv, vt = self._match_operands(lv, lt, rv, rt)
        if vt is None:
            raise NotImplementedError(f"compare between {lt!r} and {rt!r}")
        pred_pair = _CMP_PRED.get(op)
        if pred_pair is None:
            raise NotImplementedError(f"Compare {op.__name__}")
        elem = vt.elem
        i1_vt = VecTy((vt.first_dim, ), I1)
        if elem.is_float:
            pred = pred_pair[0]
            return self._b.create_arith_cmpf(pred, lv, rv), i1_vt
        pred = pred_pair[1]
        return self._b.create_arith_cmpi(pred, lv, rv), i1_vt

    def _gen_call_expr(self, node: ast.Call) -> tuple:
        b = self._aliases
        if _is_smt_rvisa_attr(node.func, "vzero", b): return self._gen_vzero(node)
        if _is_smt_rvisa_attr(node.func, "vload", b): return self._gen_vload(node)
        if _is_smt_rvisa_attr(node.func, "vmacc", b): return self._gen_vmacc(node)
        if _is_smt_rvisa_attr(node.func, "vreduce_sum", b): return self._gen_vreduce_sum(node)
        if _is_smt_rvisa_attr(node.func, "vreduce_max", b): return self._gen_vreduce_max(node)
        if _is_smt_rvisa_attr(node.func, "vreduce_min", b): return self._gen_vreduce_min(node)
        if _is_smt_rvisa_attr(node.func, "vreduce_mul", b): return self._gen_vreduce_mul(node)
        if _is_smt_rvisa_attr(node.func, "vmadot", b): return self._gen_vmadot(node)
        if _is_smt_rvisa_attr(node.func, "vpack", b): return self._gen_vpack(node)
        if _is_smt_rvisa_attr(node.func, "vshape", b): return self._gen_vshape(node)
        if _is_smt_rvisa_attr(node.func, "vbroadcast", b): return self._gen_vbroadcast(node)
        if _is_smt_rvisa_attr(node.func, "alloc", b): return self._gen_alloc(node)
        if _is_smt_rvisa_attr(node.func, "spread", b): return self._gen_spread(node)
        if _is_smt_rvisa_attr(node.func, "imin", b):
            av, _ = self._gen_expr(node.args[0])
            bv, _ = self._gen_expr(node.args[1])
            return self._b.create_arith_minsi(av, bv), INDEX
        for nm in ("vmin", "vmax"):
            if _is_smt_rvisa_attr(node.func, nm, b):
                return self._gen_vminmax(node, nm)
        if _is_smt_rvisa_attr(node.func, "sqrt", b): return self._gen_unary_math(node, "sqrt")
        if _is_smt_rvisa_attr(node.func, "rsqrt", b): return self._gen_unary_math(node, "rsqrt")
        if _is_smt_rvisa_attr(node.func, "vexp", b): return self._gen_unary_math(node, "exp")
        if _is_smt_rvisa_attr(node.func, "vlog", b): return self._gen_unary_math(node, "log")
        if _is_smt_rvisa_attr(node.func, "sload", b): return self._gen_sload(node)
        if _is_smt_rvisa_attr(node.func, "call_intrinsic", b): return self._gen_call_intrinsic(node)
        if _is_smt_rvisa_attr(node.func, "llvm_poison", b): return self._gen_llvm_poison(node)
        if _is_smt_rvisa_attr(node.func, "llvm_const", b): return self._gen_llvm_const(node)
        if _is_smt_rvisa_attr(node.func, "llvm_base_ptr", b): return self._gen_llvm_base_ptr(node)
        if _is_smt_rvisa_attr(node.func, "llvm_gep", b): return self._gen_llvm_gep(node)
        if _is_smt_rvisa_attr(node.func, "llvm_size", b): return self._gen_llvm_size(node)
        if _is_smt_rvisa_attr(node.func, "viota", b): return self._gen_viota(node)
        if _is_smt_rvisa_attr(node.func, "abs", b): return self._gen_abs(node)
        if _is_smt_rvisa_attr(node.func, "cast", b): return self._gen_cast(node)
        if _is_smt_rvisa_attr(node.func, "select", b): return self._gen_select(node)
        raise NotImplementedError(f"Unsupported call: {ast.dump(node.func)}")

    # ------------------------------------------------------------------
    # vconfig
    # ------------------------------------------------------------------

    def _gen_vconfig_assign(self, target: str, node: ast.Call):
        self._apply_vconfig(node)
        self._constexpr_ints[target] = self._active_vl

    def _apply_vconfig(self, node: ast.Call):
        lmul = ast.literal_eval(node.args[1]) if len(node.args) > 1 else 1
        vl = _vlmax(int(lmul))
        self._active_vl = vl
        try:
            avl_const = ast.literal_eval(node.args[0])
        except Exception:
            avl_const = None
        if avl_const is not None and avl_const >= _vlmax(1):
            self._active_valid = None
        elif avl_const is not None and avl_const < 0:
            self._active_valid = None
        else:
            avl_val, _ = self._gen_expr(node.args[0])
            vlmax_val = self._const_int(vl)
            self._active_valid = self._b.create_arith_minsi(vlmax_val, avl_val)

    # ------------------------------------------------------------------
    # vzero / vmacc / vreduce_sum
    # ------------------------------------------------------------------

    def _gen_vzero(self, node: ast.Call) -> tuple:
        kwargs = {kw.arg: kw.value for kw in node.keywords}
        dt = ScalarTy(_resolve_dtype(node.args[0] if node.args else None, "f32"))
        vl = self._require_vl()
        group = self._try_const_int(kwargs["group"]) if "group" in kwargs else None
        vt = VecTy((group, vl), dt) if group else VecTy((vl, ), dt)
        zero = self._const_float(0.0, dt)
        return self._b.create_vector_broadcast(zero, self._tt(vt)), vt

    def _gen_vmacc(self, node: ast.Call) -> tuple:
        acc_v, acc_t = self._gen_expr(node.args[0])
        x_v, x_t = self._gen_expr(node.args[1])
        y_v, y_t = self._gen_expr(node.args[2])
        if not isinstance(acc_t, VecTy):
            raise ValueError(f"vmacc type error: acc must be a vector, got {acc_t}")
        # Widening fma: x/y are extended to the accumulator's element type when
        # they differ (e.g. f16 operands into an f32 accumulator).
        wide_T = self._tt(acc_t)
        if x_t != acc_t:
            x_v = self._b.create_arith_extf(x_v, wide_T)
        if y_t != acc_t:
            y_v = self._b.create_arith_extf(y_v, wide_T)
        return self._b.create_math_fma(x_v, y_v, acc_v), acc_t

    def _gen_vreduce_sum(self, node: ast.Call) -> tuple:
        v_v, v_t = self._gen_expr(node.args[0])
        elem = self._reduce_elem(v_t, "vreduce_sum")
        return self._b.create_vector_reduction("add", v_v), elem

    def _gen_vreduce_max(self, node: ast.Call) -> tuple:
        v_v, v_t = self._gen_expr(node.args[0])
        elem = self._reduce_elem(v_t, "vreduce_max")
        if not elem.is_float:
            raise NotImplementedError(f"vreduce_max on integer element {elem!r} not yet wired (add maxsi to binding)")
        return self._b.create_vector_reduction("maxf", v_v), elem

    def _gen_vreduce_min(self, node: ast.Call) -> tuple:
        v_v, v_t = self._gen_expr(node.args[0])
        elem = self._reduce_elem(v_t, "vreduce_min")
        if not elem.is_float:
            raise NotImplementedError(f"vreduce_min on integer element {elem!r} not yet wired (add minsi to binding)")
        return self._b.create_vector_reduction("minf", v_v), elem

    def _gen_vreduce_mul(self, node: ast.Call) -> tuple:
        v_v, v_t = self._gen_expr(node.args[0])
        elem = self._reduce_elem(v_t, "vreduce_mul")
        return self._b.create_vector_reduction("mul", v_v), elem

    @staticmethod
    def _reduce_elem(v_t: Ty, which: str) -> ScalarTy:
        if not isinstance(v_t, VecTy):
            raise ValueError(f"{which} expects a vector, got {v_t}")
        return v_t.elem

    # ------------------------------------------------------------------
    # vmadot / vminmax / unary math / abs / cast / select
    # ------------------------------------------------------------------

    def _gen_vmadot(self, node: ast.Call) -> tuple:
        acc_v, acc_t = self._gen_expr(node.args[0])
        x_v, x_t = self._gen_expr(node.args[1])
        y_v, y_t = self._gen_expr(node.args[2])
        if not (_is_vec2d_of(x_t, ("f16", "bf16")) and _is_vec2d_of(y_t,
                                                                    ("f16", "bf16")) and _is_vec2d_of(acc_t,
                                                                                                      ("f32", ))):
            raise ValueError(f"vmadot type error: x={x_t} y={y_t} acc={acc_t}")
        b1, b2, B = x_t.dims[0], y_t.dims[0], acc_t.dims[0]
        if B != b1 * b2:
            raise ValueError(f"vmadot acc rows must be b1·b2={b1*b2}, got {B}")
        m_s, n_s, k_s = _mma_cube(x_t.elem.name)
        result_T = self._tt(acc_t)
        res = self._b.create_generic_op("vector_ext.cross_batch_matmul", [x_v, y_v, acc_v],
                                        {"k": k_s, "m": m_s, "n": n_s}, [result_T])
        return res[0], acc_t

    def _gen_vminmax(self, node: ast.Call, which: str) -> tuple:
        lv, lt = self._gen_expr(node.args[0])
        rv, rt = self._gen_expr(node.args[1])
        lv, rv, vt = self._match_operands(lv, lt, rv, rt)
        if vt is None:
            raise NotImplementedError(f"{which}: needs at least one vector")
        if vt.elem.is_float:
            fn = self._b.create_arith_minimumf if which == "vmin" else self._b.create_arith_maximumf
        else:
            fn = self._b.create_arith_minsi if which == "vmin" else self._b.create_arith_maxsi
        return fn(lv, rv), vt

    def _gen_unary_math(self, node: ast.Call, which: str) -> tuple:
        vv, vt = self._gen_expr(node.args[0])
        fn = getattr(self._b, f"create_math_{which}")
        return fn(vv), vt

    def _gen_abs(self, node: ast.Call) -> tuple:
        vv, vt = self._gen_expr(node.args[0])
        elem = self._reduce_elem(vt, "abs")
        fn = self._b.create_math_absf if elem.is_float else self._b.create_math_absi
        return fn(vv), vt

    def _gen_cast(self, node: ast.Call) -> tuple:
        vv, vt = self._gen_expr(node.args[0])
        if not isinstance(vt, VecTy):
            # Scalar cast (e.g. cast(i, f32) where i is a scalar index) — used by
            # index-tracking reductions to combine loop counters with float lanes.
            dst = vt if node.args[1] is None else ScalarTy(_resolve_dtype(node.args[1]))
            if dst == vt:
                return vv, vt
            if vt == INDEX and dst.is_float:
                return self._scalar_index_to_float(vv, dst), dst
            if isinstance(vt, ScalarTy) and vt.is_float and dst.is_float:
                fn = self._b.create_arith_extf if dst.bits > vt.bits \
                    else self._b.create_arith_truncf
                return fn(vv, self._tt(dst)), dst
            raise NotImplementedError(f"scalar cast {vt!r} → {dst!r} not supported")
        src_elem = vt.elem
        dst_elem = ScalarTy(_resolve_dtype(node.args[1], src_elem.name))
        if dst_elem == src_elem:
            return vv, vt
        dst_ty = VecTy(vt.dims, dst_elem)
        dst_T = self._tt(dst_ty)
        # index-element vector → float: index has no bit width for extf/sitofp
        # directly; go index → i64 → float (mirrors scalar path).
        if src_elem == INDEX and dst_elem.is_float:
            i64_ty = VecTy(vt.dims, I64)
            i64_v = self._b.create_arith_index_cast(vv, self._tt(i64_ty))
            return self._b.create_arith_sitofp(i64_v, dst_T), dst_ty
        sf, df = src_elem.is_float, dst_elem.is_float
        if sf and df:
            fn = self._b.create_arith_extf if dst_elem.bits > src_elem.bits \
                else self._b.create_arith_truncf
        elif sf and not df:
            fn = self._b.create_arith_fptosi
        elif not sf and df:
            fn = self._b.create_arith_sitofp
        else:
            fn = self._b.create_arith_extsi if dst_elem.bits > src_elem.bits \
                else self._b.create_arith_trunci
        return fn(vv, dst_T), dst_ty

    def _gen_select(self, node: ast.Call) -> tuple:
        mv, mt = self._gen_expr(node.args[0])
        av, at = self._gen_expr(node.args[1])
        bv, bt = self._gen_expr(node.args[2])
        av, bv, vt = self._match_operands(av, at, bv, bt)
        if vt is None:
            raise NotImplementedError("select: a/b must include at least one vector")
        return self._b.create_arith_select(mv, av, bv), vt

    # ------------------------------------------------------------------
    # vshape / vbroadcast
    # ------------------------------------------------------------------

    def _gen_vshape(self, node: ast.Call) -> tuple:
        vv, vt = self._gen_expr(node.args[0])
        if not isinstance(vt, VecTy):
            raise ValueError(f"vshape expects a vector, got {vt}")
        dims = [self._try_const_int(e) for e in node.args[1].elts] \
            if isinstance(node.args[1], ast.Tuple) else [self._try_const_int(node.args[1])]
        if any(d is None for d in dims):
            raise ValueError("vshape shape must be compile-time ints")
        out_t = VecTy(tuple(dims), vt.elem)
        return self._b.create_vector_shape_cast(vv, self._tt(out_t)), out_t

    def _gen_vbroadcast(self, node: ast.Call) -> tuple:
        vv, vt = self._gen_expr(node.args[0])
        n = self._try_const_int(node.args[1])
        if n is None:
            raise ValueError("vbroadcast n must be a compile-time int")
        if not isinstance(vt, VecTy):
            raise ValueError(f"vbroadcast expects a vector, got {vt}")
        out_t = VecTy((n, vt.first_dim), vt.elem)
        return self._b.create_vector_broadcast(vv, self._tt(out_t)), out_t

    # ------------------------------------------------------------------
    # alloc
    # ------------------------------------------------------------------

    def _gen_alloc(self, node: ast.Call) -> tuple:
        kwargs = {kw.arg: kw.value for kw in node.keywords}
        shape_node = node.args[0]
        assert isinstance(shape_node, ast.Tuple)
        dt_node = node.args[1] if len(node.args) > 1 else kwargs.get("dtype")
        dtype = ScalarTy(_resolve_dtype(dt_node, "f16"))
        dims: list = []
        dyn_vals = []
        for e in shape_node.elts:
            cv = self._try_const_int(e)
            if cv is not None:
                dims.append(cv)
            else:
                vv, _ = self._gen_expr(e)
                dims.append(None)
                dyn_vals.append(vv)
        mt = MemTy(tuple(dims), dtype)
        return self._b.create_memref_alloc(self._tt(mt), dyn_vals if dyn_vals else None, 64), mt

    # ------------------------------------------------------------------
    # _ranked_cast helper
    # ------------------------------------------------------------------

    def _ranked_cast(self, ptr_v, ptr_t: Ty) -> tuple:
        if isinstance(ptr_t, MemTy) and ptr_t.unranked:
            ranked_t = ptr_t.as_dynamic_ranked()
            return self._b.create_memref_cast(self._tt(ranked_t), ptr_v), ranked_t
        return ptr_v, ptr_t

    # ------------------------------------------------------------------
    # vload
    # ------------------------------------------------------------------

    def _gen_vload(self, node: ast.Call) -> tuple:
        kwargs = {kw.arg: kw.value for kw in node.keywords}
        ptr_v, ptr_t = self._gen_expr(node.args[0])
        idx_node = node.args[1]
        dtype = ScalarTy(_resolve_dtype(kwargs.get("dtype"), "f16"))
        vl = self._require_vl()
        vt = VecTy((vl, ), dtype)
        # fill= kwarg: value for padding of tail tiles (default 0.0).
        # Softmax exp-accumulation needs fill=-1e38 so exp(fill-xmax)≈0.
        fill_node = kwargs.get("fill")
        if fill_node is not None:
            fill_v, fill_t = self._gen_expr(fill_node)
            # Cast fill to match dtype (e.g. f32 literal -1e38 → f16 for f16 vload).
            # linalg.fill requires fill value type to match output element type.
            if fill_t != dtype:
                fill_v = self._b.create_arith_truncf(fill_v, self._tt(dtype))
            pad = fill_v
        else:
            pad = self._const_float(0.0, dtype)

        # Packed cube tensor from vpack(memref)
        group_kw = kwargs.get("group")
        if isinstance(ptr_t, TensorTy) and isinstance(idx_node, ast.Tuple) and group_kw is not None:
            group = self._try_const_int(group_kw)
            assert group is not None
            elem = ptr_t.elem
            idx_vs = [self._gen_expr(e)[0] for e in idx_node.elts]
            c0 = self._const_int(0)
            flat_t = VecTy((group * vl, ), elem)
            flat_v = self._b.create_vector_transfer_read(self._tt(flat_t), ptr_v, idx_vs + [c0], pad, [True])
            out_t = VecTy((group, vl), elem)
            return self._b.create_vector_shape_cast(flat_v, self._tt(out_t)), out_t

        if not (isinstance(ptr_t, MemTy) and ptr_t.unranked):
            # Ranked memref (alloc scratch)
            assert isinstance(idx_node, ast.Tuple)
            idx_vs = [self._gen_expr(e)[0] for e in idx_node.elts]
            return self._b.create_vector_transfer_read(self._tt(vt), ptr_v, idx_vs, pad, [True]), vt

        # External unranked pointer, tail path: _active_valid (set by a narrowing
        # vconfig) bounds the read to valid elements + fill-pads the rest.
        # vconfig is the single source of truth for effective length; vload only
        # reads + fills.
        valid_v = self._active_valid
        if valid_v is not None:
            assert "group" not in kwargs
            off_v, _ = self._gen_expr(idx_node)
            src_mr_t = MemTy((None, ), dtype, StridedLayout((None, ), True), BRIDGE)
            rsrc = self._b.create_memref_reinterpret_cast(self._tt(src_mr_t), ptr_v, [off_v], [valid_v],
                                                          [self._const_int(1)])
            tens_t = TensorTy((None, ), dtype)
            tsrc = self._b.create_bufferization_to_tensor(rsrc, self._tt(tens_t))
            fill_tens_t = TensorTy((vl, ), dtype)
            escr = self._b.create_tensor_empty(self._tt(fill_tens_t))
            fscr = self._b.create_linalg_fill(pad, escr)
            c0 = self._const_int(0)
            filled = self._b.create_tensor_insert_slice(tsrc, fscr, [c0], [valid_v], [self._const_int(1)])
            return self._b.create_vector_transfer_read(self._tt(vt), filled, [c0], pad, [True]), vt

        ranked_v, ranked_t = self._ranked_cast(ptr_v, ptr_t)
        off_v, _ = self._gen_expr(idx_node)
        group = self._try_const_int(group_kw) if group_kw is not None else None
        if group:
            flat_t = VecTy((group * vl, ), dtype)
            flat_v = self._b.create_vector_transfer_read(self._tt(flat_t), ranked_v, [off_v], pad, [True])
            out_t = VecTy((group, vl), dtype)
            return self._b.create_vector_shape_cast(flat_v, self._tt(out_t)), out_t
        return self._b.create_vector_transfer_read(self._tt(vt), ranked_v, [off_v], pad, [True]), vt

    # ------------------------------------------------------------------
    # sload — scalar load from a pointer at a dynamic index
    # ------------------------------------------------------------------

    def _gen_sload(self, node: ast.Call) -> tuple:
        """sload(ptr, idx, dtype=f32) → scalar element load from ptr[idx].

        s-prefix = scalar op (does not touch VL). Useful for gather-like access
        (e.g. cross_entropy: logits[target]). Uses _ranked_cast + memref.load —
        no C++ changes required.
        """
        kwargs = {kw.arg: kw.value for kw in node.keywords}
        ptr_v, ptr_t = self._gen_expr(node.args[0])
        idx_v, _ = self._gen_expr(node.args[1])
        dtype = ScalarTy(_resolve_dtype(kwargs.get("dtype"), "f32"))
        ranked_v, _ = self._ranked_cast(ptr_v, ptr_t)
        return self._b.create_memref_load(ranked_v, [idx_v]), dtype

    # ------------------------------------------------------------------
    # LLVM-dialect llvm-direct primitives (full call_intrinsic kernel)
    # ------------------------------------------------------------------

    def _gen_call_intrinsic(self, node: ast.Call) -> tuple:
        """call_intrinsic("llvm.riscv.vle", [ops...], result_type="vector<[8]xf16>").

        Emits an LLVM-dialect op. op name starting with 'llvm.' whose text is an
        intrinsic (llvm.riscv.*) → llvm.call_intrinsic with intrin= string attr;
        otherwise the op name is used directly (e.g. llvm.intr.vector.extract).
        result_type="()" → void op (e.g. llvm.riscv.vse store).
        """
        if not isinstance(node.args[0], ast.Constant):
            raise ValueError("call_intrinsic: op name must be a string literal")
        op_name = node.args[0].value
        if not isinstance(node.args[1], (ast.List, ast.Tuple)):
            raise ValueError("call_intrinsic: operands must be a list literal")
        operand_vs = [self._gen_expr(e)[0] for e in node.args[1].elts]
        result_t = None
        for kw in node.keywords:
            if kw.arg == "result_type":
                if not isinstance(kw.value, ast.Constant):
                    raise ValueError("call_intrinsic: result_type must be a string literal")
                result_t = kw.value.value
        if result_t is None:
            raise ValueError("call_intrinsic: must specify result_type=")
        # llvm.riscv.* / other bare intrinsic names → llvm.call_intrinsic with
        # the name carried as the `intrin` string attr. Dotted MLIR op names
        # (llvm.intr.*) are emitted directly.
        is_intrinsic = op_name.startswith("llvm.riscv.") or op_name.startswith("llvm.experimental.")
        if is_intrinsic:
            emit_name = "llvm.call_intrinsic"
            # llvm.call_intrinsic has two operand segments (args, op_bundle_operands);
            # all our operands are args, so segment sizes = [len(args), 0]. Without
            # this the generic builder defaults to [0,0] and verify fails.
            attrs = {
                "intrin": f'"{op_name}"',
                "operandSegmentSizes": f"array<i32: {len(operand_vs)}, 0>",
                "op_bundle_sizes": "array<i32>",
            }
        else:
            emit_name = op_name
            attrs = {}
        result_ty = self._user_ty(result_t)
        is_void = result_ty == VOID
        result_types = [] if is_void else [self._tt(result_ty)]
        res = self._b.create_op_textattr(emit_name, operand_vs, attrs, result_types)
        if is_void:
            return None, VOID
        return res[0], result_ty

    def _gen_llvm_poison(self, node: ast.Call) -> tuple:
        """llvm_poison("vector<[8]xf16>") → llvm.mlir.poison : T (vle passthru)."""
        if not isinstance(node.args[0], ast.Constant):
            raise ValueError("llvm_poison: type must be a string literal")
        ty = self._user_ty(node.args[0].value)
        res = self._b.create_op_textattr("llvm.mlir.poison", [], {}, [self._tt(ty)])
        return res[0], ty

    def _gen_llvm_const(self, node: ast.Call) -> tuple:
        """llvm_const(64, "i64") or llvm_const(0.0, "vector<[4]xf32>").

        Scalar int/float → llvm.mlir.constant(N : T). Vector type → splat
        dense<val> : T (dense elements attr, parsed from text).
        """
        val = self._try_const_int(node.args[0])
        fval = None
        if val is None:
            if isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, float):
                fval = node.args[0].value
            elif isinstance(node.args[0], ast.UnaryOp) and isinstance(node.args[0].op, ast.USub) \
                    and isinstance(node.args[0].operand, ast.Constant):
                fval = -node.args[0].operand.value
            else:
                raise ValueError("llvm_const: value must be a compile-time int/float literal")
        if not isinstance(node.args[1], ast.Constant):
            raise ValueError("llvm_const: type must be a string literal")
        ty = self._user_ty(node.args[1].value)
        # Materialise the MLIR Type once; its official printer text (str) is the
        # type reference embedded in the constant attr (create_op_textattr takes
        # attr values as text — the C++ builder-API contract, rule 168).
        T = self._tt(ty)
        t_text = str(T)
        if isinstance(ty, VecTy):
            lit = f"{fval if fval is not None else val}"
            attr = f"dense<{lit}> : {t_text}"
        elif isinstance(ty, ScalarTy) and ty.is_float:
            attr = f"{fval if fval is not None else float(val)} : {t_text}"
        else:
            attr = f"{val} : {t_text}"
        res = self._b.create_op_textattr("llvm.mlir.constant", [], {"value": attr}, [T])
        return res[0], ty

    def _gen_llvm_base_ptr(self, node: ast.Call) -> tuple:
        """llvm_base_ptr(mem) → llvm.extractvalue %desc[1] : !llvm.ptr (aligned base).

        The raw-kernel memref param must be materialised as an LLVM struct
        descriptor; extractvalue[1] is the aligned pointer field.
        """
        ptr_v, ptr_t = self._gen_expr(node.args[0])
        desc = self._b.create_op_textattr("builtin.unrealized_conversion_cast", [ptr_v], {}, [self._tt(LLVM_DESC)])
        res = self._b.create_op_textattr("llvm.extractvalue", [desc[0]], {"position": "array<i64: 1>"},
                                         [self._tt(LLVM_PTR)])
        return res[0], LLVM_PTR

    def _gen_llvm_gep(self, node: ast.Call) -> tuple:
        """llvm_gep(base_ptr, offset, elem="f16") → llvm.getelementptr."""
        kwargs = {kw.arg: kw.value for kw in node.keywords}
        base_v, _ = self._gen_expr(node.args[0])
        off_v, off_t = self._gen_expr(node.args[1])
        # llvm.getelementptr requires i64 offset, not index. Default-path
        # range/arithmetic produces index; cast when needed.
        if off_t == INDEX:
            off_v = self._b.create_arith_index_cast(off_v, self._tt(I64))
        elem = _resolve_dtype(kwargs.get("elem"), "f16")
        res = self._b.create_op_textattr(
            "llvm.getelementptr", [base_v, off_v],
            {"rawConstantIndices": "array<i32: -2147483648>", "elem_type": str(ScalarTy(elem).build(self._b))},
            [self._tt(LLVM_PTR)])
        return res[0], LLVM_PTR

    def _gen_llvm_size(self, node: ast.Call) -> tuple:
        """llvm_size(mem, dim=0) → llvm.extractvalue %desc[3, dim] : i64 (size field)."""
        kwargs = {kw.arg: kw.value for kw in node.keywords}
        ptr_v, ptr_t = self._gen_expr(node.args[0])
        dim = self._try_const_int(kwargs.get("dim")) if "dim" in kwargs else 0
        desc = self._b.create_op_textattr("builtin.unrealized_conversion_cast", [ptr_v], {}, [self._tt(LLVM_DESC)])
        res = self._b.create_op_textattr("llvm.extractvalue", [desc[0]], {"position": f"array<i64: 3, {dim}>"},
                                         [self._tt(I64)])
        return res[0], I64

    def _gen_viota(self, node: ast.Call) -> tuple:
        """viota() → vector<VLxf32> = [0.0, 1.0, .., VL-1.0].

        Index vector for index-tracking reductions (argmax/argmin), as f32 so it
        composes with float lanes and float reductions.

        Built via a scratch memref filled by a scalar loop, then transfer_read —
        NOT vector.step: spine-mlir ConvertToScalableVector doesn't handle StepOp
        (fails 'Fail to convert to scalable vector'), but it does handle
        memref.store / scf.for / transfer_read (the _gen_spread path, K3-proven).
        """
        vl = self._require_vl()
        vt = VecTy((vl, ), F32)
        scr = self._b.create_memref_alloc(self._tt(MemTy((vl, ), F32)), None, 64)
        c0, c1, cvl = self._const_int(0), self._const_int(1), self._const_int(vl)

        def fill_body(b, iv, _):
            i64 = b.create_arith_index_cast(iv, self._tt(I64))
            fv = b.create_arith_sitofp(i64, self._tt(F32))
            b.create_memref_store(fv, scr, [iv])
            return []

        self._b.create_scf_for(c0, cvl, c1, [], fill_body)
        pad = self._const_float(0.0, F32)
        return self._b.create_vector_transfer_read(self._tt(vt), scr, [c0], pad, [True]), vt

    # ------------------------------------------------------------------
    # vstore
    # ------------------------------------------------------------------

    def _gen_vstore(self, node: ast.Call):
        kwargs = {kw.arg: kw.value for kw in node.keywords}
        ptr_v, ptr_t = self._gen_expr(node.args[0])
        idx_node = node.args[1]
        val_v, val_t = self._gen_expr(node.args[2])
        shape_node = kwargs.get("shape")
        if shape_node is not None:
            dims = [self._try_const_int(e) for e in shape_node.elts]
            if any(d is None for d in dims) or len(dims) != 2:
                raise ValueError("vstore shape= must be 2-tuple of compile-time ints")
            R, C = dims
            if not isinstance(val_t, VecTy):
                raise TypeError(f"vstore shape= expects a vector value, got '{val_t}'")
            elem = val_t.elem
            off_v, _ = self._gen_expr(idx_node)
            m2t = MemTy((R, C), elem, StridedLayout((C, 1), True), BRIDGE)
            # Result type fully static RxC in both dims: pass int to mixed,
            # otherwise static_sizes being all-dynamic conflicts.
            r2 = self._b.create_memref_reinterpret_cast_mixed(self._tt(m2t), ptr_v, [off_v], [R, C], [C, 1])
            c0 = self._const_int(0)
            self._b.create_vector_transfer_write(val_v, r2, [c0, c0], [False, False])
            return
        assert not isinstance(idx_node, ast.Tuple)
        idx_v, _ = self._gen_expr(idx_node)
        if not isinstance(val_t, VecTy):
            raise TypeError(f"vstore expects a vector value (width = VL), got scalar '{val_t}'. "
                            f"Use sstore(ptr, idx, scalar) for a single scalar write.")
        vn = val_t.first_dim
        elem = val_t.elem
        c0 = self._const_int(0)
        if self._active_valid is None:
            # Full-tile path: static memref<VLxT, strided<[1], offset:?>>.
            # Dynamic memref<?xT> causes VL to be clamped by descriptor size
            # → only lane0 written. Static size bypasses clamping.
            m1t = MemTy((vn, ), elem, StridedLayout((1, ), True), BRIDGE)
            r1 = self._b.create_memref_reinterpret_cast_mixed(self._tt(m1t), ptr_v, [idx_v], [vn], [1])
            self._b.create_vector_transfer_write(val_v, r1, [c0], [True])
        else:
            # Tail-tile path: only _active_valid < VL elements are valid.
            # Use dynamic memref<?xT> with size=valid + in_bounds=[false]
            # so transfer_write generates a masked store respecting the bound.
            valid_v = self._active_valid
            m1t = MemTy((None, ), elem, StridedLayout((None, ), True), BRIDGE)
            r1 = self._b.create_memref_reinterpret_cast(self._tt(m1t), ptr_v, [idx_v], [valid_v], [self._const_int(1)])
            self._b.create_vector_transfer_write(val_v, r1, [c0], [False])

    # ------------------------------------------------------------------
    # sstore — scalar store to a pointer at a dynamic index
    # ------------------------------------------------------------------

    def _gen_sstore(self, node: ast.Call):
        """sstore(ptr, idx, scalar) → memref.store of a single scalar at ptr[idx].

        s-prefix = scalar op (does not touch VL). The scalar counterpart of
        vstore; used for reduction results (vreduce_*), sload results, and
        scalar constants/arithmetic. Uses _ranked_cast + memref.store.
        """
        ptr_v, ptr_t = self._gen_expr(node.args[0])
        idx_v, _ = self._gen_expr(node.args[1])
        val_v, val_t = self._gen_expr(node.args[2])
        if isinstance(val_t, VecTy):
            raise TypeError(f"sstore expects a scalar value, got vector '{val_t}'. "
                            f"Use vstore(ptr, idx, vec) for a VL-wide vector write.")
        store_v, _ = self._ranked_cast(ptr_v, ptr_t)
        self._b.create_memref_store(val_v, store_v, [idx_v])

    # ------------------------------------------------------------------
    # vpack
    # ------------------------------------------------------------------

    def _gen_vpack(self, node: ast.Call) -> tuple:
        kwargs = {kw.arg: kw.value for kw in node.keywords}
        first_v, first_t = self._gen_expr(node.args[0])
        if isinstance(first_t, MemTy):
            # memref branch: linalg.pack path
            it = kwargs.get("inner_tiles")
            assert it is not None and isinstance(it, ast.Tuple) and len(it.elts) == 2
            rt = self._try_const_int(it.elts[0])
            kt = self._try_const_int(it.elts[1])
            K = self._try_const_int(kwargs["stride"]) if "stride" in kwargs else None
            rows = self._try_const_int(kwargs["rows"]) if "rows" in kwargs else None
            assert None not in (rt, kt, K, rows)
            et = first_t.elem
            vr_node = kwargs.get("valid_rows")
            Mp = ((rows + rt - 1) // rt) * rt
            Kp = ((K + kt - 1) // kt) * kt
            need_pad = (Mp != rows) or (Kp != K) or (vr_node is not None)
            off_node = kwargs.get("offset")
            off_v = self._gen_expr(off_node)[0] if off_node is not None else self._const_int(0)
            cst = self._const_float(0.0, et)
            if vr_node is not None:
                vr_v = self._gen_expr(vr_node)[0]
                mr_t = MemTy((None, K), et, StridedLayout((K, 1), True), BRIDGE)
                # dim0 dynamic (vr_v), dim1 static K: must use mixed, otherwise
                # static_sizes marks K dynamic too → 'expected result type with
                # size = dynamic instead of K'.
                r2 = self._b.create_memref_reinterpret_cast_mixed(self._tt(mr_t), first_v, [off_v], [vr_v, K], [K, 1])
                tsrc = self._b.create_bufferization_to_tensor(r2, self._tt(TensorTy((None, K), et)))
                dyn_rows_v = vr_v
            else:
                # With an offset the layout carries offset: ? (off_v takes
                # effect); otherwise the offset segment is omitted.
                layout = StridedLayout((K, 1), True) if off_node is not None else StridedLayout((K, 1))
                mr_t = MemTy((rows, K), et, layout, BRIDGE)
                # Both dims fully static: pass int to mixed to keep
                # static_sizes=[rows, K] consistent with the result type.
                r2 = self._b.create_memref_reinterpret_cast_mixed(self._tt(mr_t), first_v, [off_v], [rows, K], [K, 1])
                tsrc = self._b.create_bufferization_to_tensor(r2, self._tt(TensorTy((rows, K), et)))
                dyn_rows_v = None
            if need_pad:
                ep = self._b.create_tensor_empty(self._tt(TensorTy((Mp, Kp), et)))
                fp = self._b.create_linalg_fill(cst, ep)
                # source tsrc is tensor<?xKx> or tensor<rowsxKx>: dim1=K is
                # static; insert_slice sizes must be mixed (dim1 passed as int
                # K), otherwise static_sizes are all dynamic and conflict with
                # the source's static dim → 'expected type tensor<?x?xf16>'
                # rank/size mismatch.
                if dyn_rows_v is not None:
                    ins = self._b.create_tensor_insert_slice_mixed(tsrc, fp, [0, 0], [dyn_rows_v, K], [1, 1])
                else:
                    ins = self._b.create_tensor_insert_slice_mixed(tsrc, fp, [0, 0], [rows, K], [1, 1])
                src_v, src_rows, src_K = ins, Mp, Kp
            else:
                src_v, src_rows, src_K = tsrc, rows, K
            oc, kc = src_rows // rt, src_K // kt
            eP = self._b.create_tensor_empty(self._tt(TensorTy((oc, kc, rt, kt), et)))
            pk = self._b.create_linalg_pack(src_v, eP, cst, [rt, kt], [0, 1], [0, 1])
            col_t = TensorTy((oc, kc, rt * kt), et)
            col = self._b.create_tensor_collapse_shape(pk, [[0], [1], [2, 3]])
            return col, col_t

        # vector branch: group_interleave
        group_len = self._try_const_int(node.args[1])
        if group_len is None:
            raise ValueError("vpack(vector) group_len must be compile-time int")
        if not _is_vec2d_of(first_t, ("f16", "bf16", "f32")):
            raise ValueError(f"vpack(vector) needs rank-2 vector, got {first_t}")
        b, ncol, elem = first_t.dims[0], first_t.dims[1], first_t.elem
        out_t = VecTy((b // 2, ncol * 2), elem)
        res = self._b.create_generic_op("vector_ext.group_interleave", [first_v], {"groupLen": group_len},
                                        [self._tt(out_t)])
        return res[0], out_t

    # ------------------------------------------------------------------
    # spread
    # ------------------------------------------------------------------

    def _gen_spread(self, node: ast.Call) -> tuple:
        kwargs = {kw.arg: kw.value for kw in node.keywords}
        cs_node = kwargs.get("cube_shape") or (node.args[1] if len(node.args) > 1 else None)
        assert isinstance(cs_node, ast.Tuple) and len(cs_node.elts) == 3
        kc = self._try_const_int(cs_node.elts[0])
        n = self._try_const_int(cs_node.elts[1])
        k = self._try_const_int(cs_node.elts[2])
        assert None not in (kc, n, k)
        src_v, src_t = self._gen_expr(node.args[0])
        if not isinstance(src_t, MemTy):
            raise ValueError(f"spread expects a memref source, got {src_t}")
        et = src_t.elem
        total = kc * k
        k_real = self._try_const_int(kwargs["k_real"]) if "k_real" in kwargs else total
        assert k_real is not None and k_real <= total
        pad_k = k_real < total
        # src → 1D <k_real>: dim0 is static k_real (mixed passes int), but the
        # stride is strided<[?]> dynamic (earlier strided fix to satisfy
        # to_tensor), so the stride still passes a Value; offset is static 0.
        src1d_t = MemTy((k_real, ), et, StridedLayout((None, ), False), BRIDGE)
        rsrc = self._b.create_memref_reinterpret_cast_mixed(self._tt(src1d_t), src_v, [0], [k_real],
                                                            [self._const_int(1)])
        scr_t = MemTy((kc, n, k), et)
        scr = self._b.create_memref_alloc(self._tt(scr_t), None, 64)
        c0, c1 = self._const_int(0), self._const_int(1)
        ckc, cn, ck = self._const_int(kc), self._const_int(n), self._const_int(k)
        ckreal = self._const_int(k_real) if pad_k else None
        ckrm1 = self._const_int(k_real - 1) if pad_k else None
        zcst = self._const_float(0.0, et) if pad_k else None

        def outer_body(b, li, _):

            def mid_body(b, lni, _):

                def inner_body(b, lki, _):
                    ck8 = b.create_arith_muli(li, ck)
                    idx = b.create_arith_addi(ck8, lki)
                    if pad_k:
                        inb = b.create_arith_cmpi("slt", idx, ckreal)
                        cl = b.create_arith_minsi(idx, ckrm1)
                        ld = b.create_memref_load(rsrc, [cl])
                        av = b.create_arith_select(inb, ld, zcst)
                    else:
                        av = b.create_memref_load(rsrc, [idx])
                    b.create_memref_store(av, scr, [li, lni, lki])
                    return []

                b.create_scf_for(c0, ck, c1, [], inner_body)
                return []

            b.create_scf_for(c0, cn, c1, [], mid_body)
            return []

        self._b.create_scf_for(c0, ckc, c1, [], outer_body)
        col_t = MemTy((kc, n * k), et)
        col = self._b.create_memref_collapse_shape(scr, [[0], [1, 2]])
        return col, col_t

    # ------------------------------------------------------------------
    # pack (statement)
    # ------------------------------------------------------------------

    def _gen_pack(self, node: ast.Call):
        src_node, src_idx, dst_node, dst_shape, stride_node = node.args[:5]
        assert isinstance(src_idx, ast.Tuple) and len(src_idx.elts) == 2
        assert isinstance(dst_shape, ast.Tuple) and len(dst_shape.elts) == 4
        vl = self._require_vl()
        rows = self._try_const_int(dst_shape.elts[2])
        assert rows is not None
        dst_v, dst_t = self._gen_expr(dst_node)
        if not isinstance(dst_t, MemTy):
            raise ValueError(f"pack expects a memref destination, got {dst_t}")
        dtype = dst_t.elem
        vt = VecTy((vl, ), dtype)
        src_v, src_t = self._gen_expr(src_node)
        ranked_v, ranked_t = self._ranked_cast(src_v, src_t)
        row0_v, _ = self._gen_expr(src_idx.elts[0])
        stride_v, _ = self._gen_expr(stride_node)
        pad = self._const_float(0.0, dtype)
        c0 = self._const_int(0)
        cvl = self._const_int(vl)
        src_mr_t = MemTy((None, ), dtype, StridedLayout((None, ), True), BRIDGE)
        tens_dyn = TensorTy((None, ), dtype)
        tens_vl = TensorTy((vl, ), dtype)

        def loop_body(b, loop_v, _):
            kb = b.create_arith_divui(loop_v, cvl)
            for r in range(rows):
                nir = row0_v if r == 0 else b.create_arith_addi(row0_v, self._const_int(r))
                roff = b.create_arith_muli(nir, stride_v)
                off = b.create_arith_addi(roff, loop_v)
                rem = b.create_arith_subi(stride_v, loop_v)
                valid = b.create_arith_minsi(cvl, rem)
                rsrc = b.create_memref_reinterpret_cast(self._tt(src_mr_t), ranked_v, [off], [valid],
                                                        [self._const_int(1)])
                tsrc = b.create_bufferization_to_tensor(rsrc, self._tt(tens_dyn))
                escr = b.create_tensor_empty(self._tt(tens_vl))
                fscr = b.create_linalg_fill(pad, escr)
                filled = b.create_tensor_insert_slice(tsrc, fscr, [c0], [valid], [self._const_int(1)])
                vec = b.create_vector_transfer_read(self._tt(vt), filled, [c0], pad, [True])
                cr_idx = self._const_int(r)
                # 1D vector<VL> written into a rank-4 memref: the
                # permutation_map has only 1 result (d3); in_bounds must match
                # the map results' rank (=1), not the index count (4).
                b.create_vector_transfer_write(vec, dst_v, [c0, kb, cr_idx, c0], [True])
            return []

        self._b.create_scf_for(c0, stride_v, cvl, [], loop_body)
