import ast
import inspect
from triton import knobs
from triton._flagtree_backend import FLAGTREE_BACKEND

# ========
# Analyzer
# ========


class VariableCollector(ast.NodeVisitor):

    def __init__(self):
        self.variables = set[str]()

    def visit_Name(self, node):
        if isinstance(node.ctx, ast.Load):
            self.variables.add(node.id)
        self.generic_visit(node)

    def visit_Subscript(self, node):
        self.generic_visit(node)

    def visit_Call(self, node):
        self.visit(node.func)
        for arg in node.args:
            self.visit(arg)
        for kw in node.keywords:
            self.visit(kw.value)

    def visit_Attribute(self, node):
        # For tl.arange, skip module prefix 'tl'; for a.b collect 'a'
        if isinstance(node.value, ast.Name):
            # Skip module prefixes like tl, np, etc.
            if node.value.id not in ('tl', 'triton', 'language', 'np', 'torch'):
                self.variables.add(node.value.id)
        self.generic_visit(node)

    @staticmethod
    def collect(node) -> set[str]:
        collector = VariableCollector()
        collector.visit(node)
        return collector.variables


class KernelDependencyAnalyzer(ast.NodeVisitor):

    # Parameter order of `tl.make_tensor_descriptor`, mirrored from
    # `python/triton/language/core.py::make_tensor_descriptor`.
    # Both positional and keyword call forms must be handled because user
    # kernels routinely use either (e.g. `tl.make_tensor_descriptor(a_ptr,
    # shape=..., strides=..., block_shape=...)` mixes positional `base` with
    # keyword tail).
    _MAKE_TMA_DESC_PARAM_ORDER = ('base', 'shape', 'strides', 'block_shape')

    def __init__(self, kernel_globals: dict | None = None):
        self.input_params = set[str]()  # input params
        self.constexpr_params = set[str]()  # constexpr params
        self.var_definitions = dict[str, ast.AST]()  # var -> latest definition node
        # for input-constexpr dependencies analyze
        self.load_addresses = list[ast.AST]()  # tl.load address expressions
        # for TMA descriptor load dependencies analyze
        self.tma_load_assignments = list[dict[str, str | list[ast.AST]]]()
        self.transpose_args_nodes = list[ast.AST]()  # tl.trans args
        # for tl.dot K-dim analyze
        self.dot_calls = list[ast.AST]()  # tl.dot call nodes
        # for tl.load BLOCK_X <-> X pairing via tl.cdiv(X, BLOCK_X).
        # Also stores virtual cdiv Calls synthesized from user helpers
        # whose body wraps tl.cdiv (e.g. prev_multiple_of(K, BLOCK_K)).
        self.cdiv_calls = list[ast.Call]()  # tl.cdiv call nodes
        # desc var -> {"shape": [...], "block_shape": [...]} from make_tensor_descriptor or hook
        self.tma_device_desc_defs_map = dict[str, dict[str, list[str]]]()
        # all historical definitions per var (in order); used for arange extraction
        # to avoid losing info when later assignments overwrite earlier ones
        self.var_all_definitions = dict[str, list[ast.AST]]()
        # Kernel's __globals__; used to resolve user-defined helper functions
        # to their source so we can detect tl.cdiv wrappers.
        self._kernel_globals = kernel_globals or {}
        # Cache: helper_fn_name -> (param_idx_X, param_idx_BLOCK_X) | None.
        # `None` means "no cdiv-wrapping pair" (also used as in-flight sentinel
        # to break recursion cycles).
        self._helper_cdiv_pair_cache = dict[str, tuple[int, int] | None]()

    # Collect function parameters and mark constexpr ones
    def visit_FunctionDef(self, node):
        for arg in node.args.args:
            arg_name = arg.arg
            self.input_params.add(arg_name)

            if arg.annotation:
                ann_str = ast.unparse(arg.annotation) if hasattr(ast, 'unparse') else ''
                if not ann_str:
                    try:
                        ann_str = ast.dump(arg.annotation)
                    except Exception:
                        ann_str = ''  # Python 3.8 fallback
                if 'constexpr' in ann_str:
                    self.constexpr_params.add(arg_name)

        self.generic_visit(node)

    # Record variable definitions, capture desc.load and make_tensor_descriptor
    def visit_Assign(self, node):
        targets = node.targets
        if len(targets) == 1 and isinstance(targets[0], ast.Name):
            var_name = targets[0].id

            # Capture desc.load([addr, ...]) assignments
            if (isinstance(node.value, ast.Call) and self._is_tma_load(node.value) and node.value.args
                    and isinstance(node.value.args[0], ast.List)):
                desc_name = node.value.func.value.id
                addr_exprs = node.value.args[0].elts
                self.tma_load_assignments.append(
                    {'var_name': var_name, 'desc_name': desc_name, 'addr_exprs': addr_exprs})

            # TMA device: record LHS name + shape/block_shape from make_tensor_descriptor.
            # Supports both positional (`tl.make_tensor_descriptor(a_ptr, [M,K], [K,1], [BM,BK])`)
            # and keyword (`tl.make_tensor_descriptor(base=a_ptr, shape=..., block_shape=...)`)
            # forms, and any partial mix of the two.
            if (isinstance(node.value, ast.Call) and self._is_tl_make_tensor_descriptor(node.value)):
                shape_node = self._resolve_call_arg(node.value, 'shape', self._MAKE_TMA_DESC_PARAM_ORDER)
                block_shape_node = self._resolve_call_arg(node.value, 'block_shape', self._MAKE_TMA_DESC_PARAM_ORDER)
                shape_names = self._extract_list_name_ids(shape_node)
                block_names = self._extract_list_name_ids(block_shape_node)
                if shape_names or block_names:
                    self.tma_device_desc_defs_map[var_name] = {"shape": shape_names, "block_shape": block_names}

            self.var_all_definitions.setdefault(var_name, list[ast.AST]()) \
                                    .append(node.value)
            self.var_definitions[var_name] = node.value
        self.generic_visit(node)

    # Treat x += expr as x = x <op> expr for dependency tracking
    def visit_AugAssign(self, node):
        if isinstance(node.target, ast.Name):
            var_name = node.target.id
            # Synthesize an equivalent BinOp node
            binop = ast.BinOp(
                left=ast.Name(id=var_name, ctx=ast.Load()),
                op=node.op,
                right=node.value,
            )
            self.var_all_definitions.setdefault(var_name, list[ast.AST]()) \
                                    .append(binop)
            self.var_definitions[var_name] = binop
        self.generic_visit(node)

    # Record annotated assignments, mark constexpr if annotated
    def visit_AnnAssign(self, node):
        if isinstance(node.target, ast.Name) and node.value is not None:
            var_name = node.target.id
            self.var_all_definitions.setdefault(var_name, list[ast.AST]()) \
                                    .append(node.value)
            self.var_definitions[var_name] = node.value

            if node.annotation:
                ann_str = ast.unparse(node.annotation) if hasattr(ast, 'unparse') else ''
                if not ann_str:
                    try:
                        ann_str = ast.dump(node.annotation)
                    except Exception:
                        ann_str = ''
                if 'constexpr' in ann_str:
                    self.constexpr_params.add(var_name)

        self.generic_visit(node)

    # Capture tl.load addresses, tl.trans args, tl.dot
    # (Device-side make_tensor_descriptor is handled in visit_Assign which
    # populates `tma_device_desc_defs_map`; nothing else needs to be collected
    # at the Call level here.)
    def visit_Call(self, node):
        if self._is_tl_load(node) and node.args:
            self.load_addresses.append(node.args[0])
        elif self._is_tl_transpose(node) and node.args:
            self.transpose_args_nodes.append(node.args[0])
        elif self._is_tl_dot(node):
            self.dot_calls.append(node)
        elif self._is_tl_func(node, 'cdiv') and len(node.args) >= 2:
            self.cdiv_calls.append(node)
        else:
            # User-defined helper that wraps tl.cdiv internally
            # (e.g. `prev_multiple_of(a, b) -> tl.cdiv(a, b) * b - b`).
            # Synthesize a virtual `tl.cdiv(args[i], args[j])` Call so the
            # downstream BLOCK_X <-> X matching can treat it like a real cdiv.
            virtual_cdiv = self._build_virtual_cdiv_from_helper(node)
            if virtual_cdiv is not None:
                self.cdiv_calls.append(virtual_cdiv)
        self.generic_visit(node)

    # Treat `expr.T` (inline transpose) as semantically equivalent to
    # `tl.trans(expr)` for the purpose of descriptor transpose analysis.
    # The `.T` form appears commonly inside `tl.dot(a, b.T, ...)` style code
    # (e.g. python/tutorials/09-persistent-matmul.py) where the user never
    # actually invokes `tl.trans`. Without this, the analyzer would mistake
    # the load result for the canonical layout and emit an "inconsistent
    # bshape" warning while computing K.
    def visit_Attribute(self, node):
        if node.attr == 'T':
            self.transpose_args_nodes.append(node.value)
        self.generic_visit(node)

    # If `call_node` invokes a user-defined helper whose body wraps
    # `tl.cdiv(p_X, p_BLOCK_X)` (with p_X, p_BLOCK_X being the helper's formals),
    # build a synthetic `tl.cdiv(call_node.args[i], call_node.args[j])` Call
    # for downstream pairing. Returns None when no such helper / pair exists.
    #
    # Example handled here::
    #
    #     def prev_multiple_of(a, b):
    #         return tl.cdiv(a, b) * b - b
    #     ...
    #     prev_multiple = prev_multiple_of(K, BLOCK_K)  # virtual tl.cdiv(K, BLOCK_K)
    #
    # Recursion (helper -> helper -> tl.cdiv) is supported via
    # `_get_helper_cdiv_pair`'s in-flight-cycle-breaker cache.
    def _build_virtual_cdiv_from_helper(self, call_node) -> ast.Call | None:
        if not isinstance(call_node.func, ast.Name):
            return None
        helper_name = call_node.func.id
        pair = self._get_helper_cdiv_pair(helper_name)
        if pair is None:
            return None
        i, j = pair
        if i >= len(call_node.args) or j >= len(call_node.args):
            return None
        return ast.Call(
            func=ast.Attribute(value=ast.Name(id='tl', ctx=ast.Load()), attr='cdiv', ctx=ast.Load()),
            args=[call_node.args[i], call_node.args[j]],
            keywords=[],
        )

    # Memoized lookup. Sets the cache to None *before* recursing to break
    # cycles (helper A -> helper A directly or transitively).
    def _get_helper_cdiv_pair(self, helper_name: str) -> tuple[int, int] | None:
        if helper_name in self._helper_cdiv_pair_cache:
            return self._helper_cdiv_pair_cache[helper_name]
        self._helper_cdiv_pair_cache[helper_name] = None
        pair = self._inspect_helper_for_cdiv(helper_name)
        self._helper_cdiv_pair_cache[helper_name] = pair
        return pair

    # Walk `helper_name`'s body and find the first Call whose two args are
    # *both* helper formals and which either (a) is `tl.cdiv` directly, or
    # (b) is itself a Call to another helper that wraps `tl.cdiv`. Returns
    # the (i, j) indices of those formals in helper_name's parameter list.
    def _inspect_helper_for_cdiv(self, helper_name: str) -> tuple[int, int] | None:
        helper_obj = self._kernel_globals.get(helper_name)
        if helper_obj is None:
            return None
        fn_def = self._get_helper_fndef(helper_obj, helper_name)
        if fn_def is None:
            return None
        param_names = [a.arg for a in fn_def.args.args]
        if len(param_names) < 2:
            return None

        for sub in ast.walk(fn_def):
            if not isinstance(sub, ast.Call):
                continue
            # Determine which two arg positions of `sub` are the logical
            # (X, BLOCK_X) cdiv pair.
            if self._is_tl_func(sub, 'cdiv') and len(sub.args) >= 2:
                i_in_sub, j_in_sub = 0, 1
            elif isinstance(sub.func, ast.Name):
                inner_pair = self._get_helper_cdiv_pair(sub.func.id)
                if inner_pair is None:
                    continue
                i_in_sub, j_in_sub = inner_pair
            else:
                continue
            if i_in_sub >= len(sub.args) or j_in_sub >= len(sub.args):
                continue
            ai, aj = sub.args[i_in_sub], sub.args[j_in_sub]
            if not (isinstance(ai, ast.Name) and isinstance(aj, ast.Name)):
                continue
            if ai.id in param_names and aj.id in param_names:
                return (param_names.index(ai.id), param_names.index(aj.id))
        return None

    # Get the FunctionDef AST of a helper function object. Tries
    # JITFunction.parse() first (cheap, already cached source), then falls
    # back to inspect.getsource() for plain Python helpers.
    @staticmethod
    def _get_helper_fndef(helper_obj, helper_name: str) -> ast.FunctionDef | None:
        if hasattr(helper_obj, 'parse') and callable(helper_obj.parse):
            try:
                tree = helper_obj.parse()
                if (isinstance(tree, ast.Module) and tree.body and isinstance(tree.body[0], ast.FunctionDef)):
                    return tree.body[0]
            except Exception:
                pass
        try:
            import inspect
            import textwrap
            src = textwrap.dedent(inspect.getsource(helper_obj))
            tree = ast.parse(src)
            for n in ast.walk(tree):
                if isinstance(n, ast.FunctionDef) and n.name == helper_name:
                    return n
        except Exception:
            return None
        return None

    # Check if a Call node refers to triton.language.<func_name>.
    # Covers: tl.f(), language.f(), triton.language.f(), f()
    @staticmethod
    def _is_tl_func(node, func_name: str) -> bool:
        func = node.func
        # tl.f() / language.f()
        if isinstance(func, ast.Attribute) and func.attr == func_name:
            if isinstance(func.value, ast.Name):
                return func.value.id in ('tl', 'language', 'triton')
            # triton.language.f()
            if (isinstance(func.value, ast.Attribute) and func.value.attr == 'language'
                    and isinstance(func.value.value, ast.Name) and func.value.value.id == 'triton'):
                return True
        # f() (bare import)
        if isinstance(func, ast.Name) and func.id == func_name:
            return True
        return False

    def _is_tl_load(self, node) -> bool:
        return self._is_tl_func(node, 'load')

    # desc.load(...) — any .load() that is NOT triton.language.load
    def _is_tma_load(self, node) -> bool:
        if isinstance(node.func, ast.Attribute) and node.func.attr == 'load':
            return not self._is_tl_func(node, 'load')
        return False

    def _is_tl_make_tensor_descriptor(self, node) -> bool:
        return self._is_tl_func(node, 'make_tensor_descriptor')

    def _is_tl_transpose(self, node) -> bool:
        return self._is_tl_func(node, 'trans')

    # Return the AST node bound to parameter `name` in `call_node`, looking
    # up either keyword form (`name=value`) or positional form (by index in
    # `param_order`). `param_order` MUST mirror the callee's signature so
    # positional resolution is correct.
    #
    # Returns None when `name` is neither passed positionally nor as kwarg.
    @staticmethod
    def _resolve_call_arg(call_node: ast.Call, name: str, param_order: tuple[str, ...]) -> ast.AST | None:
        for kw in call_node.keywords:
            if getattr(kw, 'arg', None) == name:
                return kw.value
        if name in param_order:
            idx = param_order.index(name)
            if idx < len(call_node.args):
                return call_node.args[idx]
        return None

    # Given an AST node expected to be an `ast.List`, return its `Name` elements'
    # ids (skipping non-Name elements such as BinOps / Constants).
    @staticmethod
    def _extract_list_name_ids(list_node: ast.AST | None) -> list[str]:
        if not isinstance(list_node, ast.List):
            return list[str]()
        return [elt.id for elt in list_node.elts if isinstance(elt, ast.Name)]

    # Resolve a symbol (e.g. a local TMA desc var) to its underlying tensor input param
    def _resolve_tensor_param(self, symbol: str | None) -> str | None:
        if not symbol:
            return None

        if symbol in self.input_params:
            return symbol

        # Local var from make_tensor_descriptor: use 'base' kwarg directly
        if symbol in self.var_definitions:
            node = self.var_definitions[symbol]
            if isinstance(node, ast.Call) and self._is_tl_make_tensor_descriptor(node):
                for kw in node.keywords:
                    if getattr(kw, "arg", None) == "base" and isinstance(kw.value, ast.Name):
                        base_name = kw.value.id
                        if base_name in self.input_params:
                            return base_name

        # Fallback: find the unique input param via dependency analysis
        input_deps, _ = self.get_dependencies(symbol)
        if len(input_deps) == 1:
            return list[str](input_deps)[0]

        return None

    def _is_tl_dot(self, node) -> bool:
        return self._is_tl_func(node, 'dot')

    def _is_tl_arange(self, node) -> bool:
        return isinstance(node, ast.Call) and self._is_tl_func(node, 'arange')

    def _extract_arange_bs(self, node: ast.AST) -> set[str]:
        out = set[str]()
        for child in ast.walk(node):
            if isinstance(child, ast.Call) and self._is_tl_arange(child) and len(child.args) >= 2:
                if isinstance(child.args[1], ast.Name):
                    out.add(child.args[1].id)
        return out

    def _extract_arange_bs_recursive(self, var_name: str, visited: set[str] | None = None) -> set[str]:
        # Case (mm_kernel_general): a = tl.load(A + (ram[:, None] * stride_am + rk[None, :] * stride_ak))
        # Case1: var_name = 'rk', ret = {'BLOCK_K'}
        # Case2: var_name = 'ram', ret = {'BLOCK_M'}
        if visited is None:
            visited = set[str]()
        if var_name in visited:
            return set[str]()
        visited.add(var_name)

        # Case1:   var_all_definitions['rk'] = [
        #              (start_k + tl.arange(0, BLOCK_K)).to(tl.int64),
        #              (prev_multiple + tl.arange(0, BLOCK_K)).to(tl.int64)
        #          ]
        # Case2:   var_all_definitions['ram'] = [
        #              tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M).to(tl.int64)
        #          ]
        # Case2.1: var_all_definitions['rm'] = [
        #              pid_m * BLOCK_M + tl.arange(0, BLOCK_M),
        #              rm.to(tl.int64)
        #          ]
        ret = set[str]()
        for def_node in self.var_all_definitions.get(var_name, []):
            ret.update(self._extract_arange_bs(def_node))
            # Case1:   found tl.arange(0, BLOCK_K) => 'BLOCK_K'
            #          VariableCollector.collect(def_node) = {'start_k', 'prev_multiple'}
            # Case2:   not found tl.arange
            #          VariableCollector.collect(def_node) = {'rm'}
            # Case2.1: found tl.arange(0, BLOCK_M) => 'BLOCK_M'
            #          VariableCollector.collect(def_node) = {}
            for child_var in VariableCollector.collect(def_node):
                if child_var != var_name and child_var not in self.input_params and child_var not in self.constexpr_params:
                    # Filter out input_params('A', 'M', 'stride_*') and constexpr_params('BLOCK_*')
                    ret.update(self._extract_arange_bs_recursive(child_var, visited.copy()))
        return ret

    def get_dependencies(self, var_name: str) -> tuple[set[str], set[str]]:
        # Layered (BFS) dependency analysis with input short-circuit.
        # For each layer (one level of definitions):
        #   - constexpr params encountered are always accumulated
        #   - if any input param appears in this layer, return immediately
        #     (don't dive deeper through other branches)
        # This avoids polluting input_deps via pid-derived chains that
        # eventually reach grid_m/grid_n -> M/N. Example: ram's definition
        # `tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), ...)` directly
        # contains M, so we stop there and never traverse rm -> pid_m -> ...
        input_deps = set[str]()
        constexpr_deps = set[str]()

        if var_name in self.input_params and var_name not in self.constexpr_params:
            input_deps.add(var_name)
            return input_deps, constexpr_deps
        if var_name in self.constexpr_params:
            constexpr_deps.add(var_name)
            return input_deps, constexpr_deps

        visited = {var_name}
        queue = [var_name]
        while queue:
            next_queue = list[str]()
            layer_inputs = set[str]()
            for cur in queue:
                if cur not in self.var_definitions:
                    continue
                used_vars = VariableCollector.collect(self.var_definitions[cur])
                for v in used_vars:
                    if v in self.input_params and v not in self.constexpr_params:
                        layer_inputs.add(v)
                    elif v in self.constexpr_params:
                        constexpr_deps.add(v)
                    elif v not in visited:
                        visited.add(v)
                        next_queue.append(v)
            if layer_inputs:
                input_deps.update(layer_inputs)
                return input_deps, constexpr_deps
            queue = next_queue
        return input_deps, constexpr_deps

    def _get_dependencies_vars(self, var_name: str, visited: set[str] | None = None) -> set[str]:
        if visited is None:
            visited = set[str]()
        if var_name in visited:
            return set[str]()
        visited.add(var_name)

        var_deps = set[str]()

        # Check if it is an input or constexpr parameter
        if (var_name in self.input_params) or (var_name in self.constexpr_params):
            return var_deps

        # Recursively analyze the dependencies of the variable definition
        if var_name in self.var_definitions:
            definition_node = self.var_definitions[var_name]
            # Skip runtime value program_id
            if True:
                used_vars = VariableCollector.collect(definition_node)
                for used_var in used_vars:
                    var_deps.update(self._get_dependencies_vars(used_var, visited.copy()))
        return var_deps

    # Analyzer 3: tl.dot
    def analyze_tma_dot_dim(
            self, tma_map: dict[str, set[tuple[str, ...]]]) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
        # tma_map already stores the canonical block_shape per desc,
        # representing (M,K) or (K,N) in memory-layout order.
        # Map each dot operand var back to its desc_name (through desc.load
        # or tl.trans(desc.load result)), then read block_shape from tma_map.

        # var -> desc_name: direct desc.load assignments
        var_to_desc = dict[str, str]()
        for tma_load_assignment in self.tma_load_assignments:
            var_to_desc[tma_load_assignment['var_name']] = tma_load_assignment['desc_name']
        # also trace through tl.trans(src) -> same desc
        for var_name, def_node in self.var_definitions.items():
            if isinstance(def_node, ast.Call) and self._is_tl_transpose(def_node) and def_node.args:
                for src_var in VariableCollector.collect(def_node.args[0]):
                    if src_var in var_to_desc:
                        var_to_desc[var_name] = var_to_desc[src_var]
                        break

        def _get_desc_bs(var_node) -> tuple[str, list[str]] | None:
            for v in VariableCollector.collect(var_node):
                if v in var_to_desc:
                    dn = var_to_desc[v]
                    bs_set = tma_map.get(dn)
                    if bs_set:
                        return (dn, list[str](next(iter(bs_set))))
            return None

        # tl.dot(a, b): a (M, K), b (K, N).
        bs_m_map = dict[str, set[str]]()
        tma_k_map = dict[str, set[str]]()
        for dot_node in self.dot_calls:
            args = dot_node.args
            if len(args) < 2:
                continue

            k_from_a = None
            k_from_b = None
            a_desc_name = None
            b_desc_name = None

            # a: shape (M, K) -> block_shape[0]=M, block_shape[-1]=K
            a_info = _get_desc_bs(args[0])
            if a_info:
                a_desc_name, a_bs = a_info
                if a_bs:
                    m_from_a = a_bs[0]
                    a_tensor_param_for_m = self._resolve_tensor_param(a_desc_name)
                    if m_from_a is not None and a_tensor_param_for_m is not None:
                        bs_m_map.setdefault(m_from_a, set[str]()).add(a_tensor_param_for_m)
                    k_from_a = a_bs[-1]

            # b: shape (K, N) -> block_shape[0]=K
            b_info = _get_desc_bs(args[1])
            if b_info:
                b_desc_name, b_bs = b_info
                if b_bs:
                    k_from_b = b_bs[0]

            bs_k_name = k_from_a if k_from_a is not None else k_from_b
            if (k_from_a is not None and k_from_b is not None and k_from_a != k_from_b):
                if knobs.autotuning.print:
                    print(f"[Analyzer] Warning: inconsistent bshape {k_from_a} != {k_from_b}")
                return {}, {}

            a_tensor_param = self._resolve_tensor_param(a_desc_name)
            b_tensor_param = self._resolve_tensor_param(b_desc_name)

            if bs_k_name is not None:
                if a_tensor_param is not None:
                    tma_k_map.setdefault(bs_k_name, set[str]()).add(a_tensor_param)
                if b_tensor_param is not None:
                    tma_k_map.setdefault(bs_k_name, set[str]()).add(b_tensor_param)

        return bs_m_map, tma_k_map

    # Analyzer 4: tl.dot for general (non-TMA) kernels driven by tl.load.
    def analyze_general_dot_dim(
            self, load_map: dict[str, str]) -> tuple[dict[str, set[str]], dict[str, set[str]], dict[str, set[str]]]:
        """Map each `tl.dot` operand to (tensor_param, BLOCK_X set) using
        `tl.load` results. Parallel of `analyze_tma_dot_dim` for non-TMA paths.

        Algorithm:
          1. For every var that has a `tl.load(addr)` definition in its history
             (`var_all_definitions`), derive:
               - tensor_param: leftmost `Name` reached by walking `BinOp.left`
                 chains of `addr`, restricted to input parameters (typically a
                 tensor pointer like `A`);
               - bs_set: BLOCK_X used in `addr` (via `_extract_arange_bs_recursive`),
                 intersected with `load_map.keys()` to filter out spurious BLOCKs.
          2. Chain through `tl.trans`: `var2 = tl.trans(var1)` inherits var1's
             tensor + bs_set. Axis identity is encoded in the bs *set* (not in
             per-axis order), so the swap is irrelevant for set semantics.
          3. For each `tl.dot(a, b)`:
               - Resolve a -> (A, a_bs); b -> (B, b_bs).
               - K-block = a_bs ∩ b_bs (singleton expected -> shared K dim).
               - M-block = a_bs - b_bs (singleton expected -> a's M dim).
               - N-block = b_bs - a_bs (singleton expected -> b's N dim).
               - m_map[M-block].add(A); k_map[K-block].add({A, B}); n_map[N-block].add(B).

        Example (mthreads mm_kernel)::

            a = tl.load(A + (ram[:, None] * stride_am + rk[None, :] * stride_ak))
            b = tl.load(B + (rk[:, None] * stride_bk + rbn[None, :] * stride_bn))
            acc += tl.dot(a, b, ...)
            # -> m_map = {'BLOCK_M': {'A'}}
            # -> k_map = {'BLOCK_K': {'A', 'B'}}
            # -> n_map = {'BLOCK_N': {'B'}}
        """
        valid_bs = set(load_map.keys())

        # var -> tensor_param and var -> set of BLOCK_X, derived from the
        # earliest tl.load definition of `var` (handles later reassignments
        # like `a = a.to(C.dtype.element_ty)` without losing the load info).
        var_to_tensor = dict[str, str]()
        var_to_bs = dict[str, set[str]]()
        for var_name, def_nodes in self.var_all_definitions.items():
            for def_node in def_nodes:
                if not (isinstance(def_node, ast.Call) and self._is_tl_load(def_node) and def_node.args):
                    continue
                addr = def_node.args[0]
                # Base tensor: leftmost atom of nested BinOp(Add, ...) chains.
                # Covers both `tl.load(A + offset)` and `tl.load(A)` (where A
                # was previously rebound to `A + offset`).
                base = addr
                while isinstance(base, ast.BinOp):
                    base = base.left
                if isinstance(base, ast.Name) and base.id in self.input_params:
                    var_to_tensor[var_name] = base.id
                # BLOCK_X via tl.arange chain, filtered by load_map keys.
                used_bs = set[str]()
                for v in VariableCollector.collect(addr):
                    used_bs.update(self._extract_arange_bs_recursive(v))
                var_to_bs[var_name] = used_bs & valid_bs
                break  # earliest tl.load definition wins

        # Chain through tl.trans: `var2 = tl.trans(var1)` inherits var1's info.
        for var_name, def_node in self.var_definitions.items():
            if not (isinstance(def_node, ast.Call) and self._is_tl_transpose(def_node) and def_node.args):
                continue
            for src_var in VariableCollector.collect(def_node.args[0]):
                if src_var in var_to_tensor:
                    var_to_tensor[var_name] = var_to_tensor[src_var]
                    var_to_bs[var_name] = var_to_bs.get(src_var, set[str]())
                    break

        def _resolve(var_node) -> tuple[str | None, set[str]]:
            for v in VariableCollector.collect(var_node):
                if v in var_to_tensor:
                    return var_to_tensor[v], var_to_bs.get(v, set[str]())
            return None, set[str]()

        m_map = dict[str, set[str]]()
        k_map = dict[str, set[str]]()
        n_map = dict[str, set[str]]()
        for dot_node in self.dot_calls:
            args = dot_node.args
            if len(args) < 2:
                continue
            a_param, a_bs = _resolve(args[0])
            b_param, b_bs = _resolve(args[1])
            if a_param is None or b_param is None or not a_bs or not b_bs:
                continue
            common = a_bs & b_bs  # K-block: shared between a and b
            m_blocks = a_bs - common  # M-block: exclusive to a
            n_blocks = b_bs - common  # N-block: exclusive to b
            if len(common) != 1 or len(m_blocks) != 1 or len(n_blocks) != 1:
                if knobs.autotuning.print:
                    print(f"[Analyzer] Warning: ambiguous dim, common={common} "
                          f"m_only={m_blocks} n_only={n_blocks}")
                return {}, {}, {}
            m_block = next(iter(m_blocks))
            k_block = next(iter(common))
            n_block = next(iter(n_blocks))
            m_map.setdefault(m_block, set[str]()).add(a_param)
            k_map.setdefault(k_block, set[str]()).add(a_param)
            k_map.setdefault(k_block, set[str]()).add(b_param)
            n_map.setdefault(n_block, set[str]()).add(b_param)

        return m_map, k_map, n_map

    def _find_cdiv_x(self, bs_name: str) -> str | None:
        for call in self.cdiv_calls:
            arg0, arg1 = call.args[0], call.args[1]
            if not (isinstance(arg0, ast.Name) and isinstance(arg1, ast.Name)):
                continue
            if arg1.id != bs_name:
                continue
            x_name = arg0.id
            if x_name == bs_name:
                continue
            if x_name not in self.input_params and x_name not in self.constexpr_params:
                continue
            return x_name
        return None

    # Analyzer 1: tl.load
    def analyze_tl_load_bs(self) -> dict[str, str]:
        """Map BLOCK_X (used in any tl.load address) to its tensor-dim arg X.

        Algorithm (driven by tl.load):
          1. For each `tl.load(addr)` address expression, find BLOCK_X actually
             used through the `tl.arange(0, BLOCK_X)` chain.
          2. For each such BLOCK_X, look up a `tl.cdiv(X, BLOCK_X)` call somewhere
             in the kernel and adopt its first argument as the paired tensor-dim X.

        Examples in a typical mm kernel (any of these establishes the pair)::

            grid_m = tl.cdiv(M, BLOCK_M)               # BLOCK_M <-> M
            grid_n = tl.cdiv(N, BLOCK_N)               # BLOCK_N <-> N
            for k in range(0, tl.cdiv(K, BLOCK_K)):    # BLOCK_K <-> K

        Limitations:
          - `tl.cdiv` is recognized via `_is_tl_func` (handles `tl.`, `language.`,
            `triton.language.`, and bare `cdiv()`); user-defined import aliases
            (e.g. `import triton.language as L; L.cdiv(...)`) are not recognized.
        """
        load_map = dict[str, str]()  # [bs_name, ts_name]
        for addr_expr in self.load_addresses:
            # BLOCK_X referenced from this load address (via tl.arange chain)
            used_bs = set[str]()
            for var_name in VariableCollector.collect(addr_expr):
                used_bs.update(self._extract_arange_bs_recursive(var_name))
            # Look up X for each BLOCK_X via tl.cdiv(X, BLOCK_X)
            for bs_name in used_bs:
                ts_name = self._find_cdiv_x(bs_name)
                if ts_name is not None:
                    load_map[bs_name] = ts_name
        if knobs.autotuning.print:
            print(f"[Analyzer] load_map={load_map}")
        return load_map

    # Extract KEY (str) from `nargs["KEY"]` or `nargs.get("KEY", ...)`, where
    # `nargs` is whatever the hook function names its first parameter.
    # Returns None if `value_node` does not match either pattern.
    @staticmethod
    def _extract_nargs_key(value_node: ast.AST, nargs_param: str) -> str | None:
        # Form 1: nargs["KEY"]
        if isinstance(value_node, ast.Subscript) and isinstance(value_node.value, ast.Name):
            if value_node.value.id == nargs_param:
                sl = value_node.slice
                if isinstance(sl, ast.Constant) and isinstance(sl.value, str):
                    return sl.value
                # py3.8 backward-compat: ast.Index(Constant(...))
                inner = getattr(sl, "value", None)
                if isinstance(inner, ast.Constant) and isinstance(inner.value, str):
                    return inner.value
        # Form 2: nargs.get("KEY", ...)
        if isinstance(value_node, ast.Call):
            f = value_node.func
            if (isinstance(f, ast.Attribute) and f.attr == "get" and isinstance(f.value, ast.Name)
                    and f.value.id == nargs_param and value_node.args and isinstance(value_node.args[0], ast.Constant)
                    and isinstance(value_node.args[0].value, str)):
                return value_node.args[0].value
        return None

    # Parse nargs["a_desc"].block_shape = [BLOCK_M, BLOCK_K] in a pre_hook
    def _parse_hook_desc_bshapes(self, hook_ast: ast.FunctionDef) -> dict[str, list[list[str]]]:
        # Hook's first parameter name (typically 'nargs').
        nargs_param = hook_ast.args.args[0].arg if hook_ast.args.args else None

        # Build local-name -> constexpr-key substitution from hook-local rebinds:
        #   BLOCK_M = nargs["BLOCK_SIZE_M"]            -> {'BLOCK_M': 'BLOCK_SIZE_M'}
        #   BLOCK_M = nargs.get("BLOCK_SIZE_M", ...)   -> {'BLOCK_M': 'BLOCK_SIZE_M'}
        # Without this, hooks that rename constexprs locally (a common pattern)
        # leak hook-local names into bs_names, causing KeyError downstream when
        # `current[bs_name]` is looked up against the kernel's actual constexpr
        # parameter names.
        local_to_key = dict[str, str]()
        if nargs_param is not None:
            for stmt in ast.walk(hook_ast):
                if not isinstance(stmt, ast.Assign) or len(stmt.targets) != 1:
                    continue
                tgt = stmt.targets[0]
                if not isinstance(tgt, ast.Name):
                    continue
                key = self._extract_nargs_key(stmt.value, nargs_param)
                if key is not None:
                    local_to_key[tgt.id] = key

        ret = dict[str, list[list[str]]]()
        for node in ast.walk(hook_ast):
            if not isinstance(node, ast.Assign) or len(node.targets) != 1:
                continue
            t = node.targets[0]
            if not isinstance(t, ast.Attribute) or t.attr != "block_shape":
                continue
            if not isinstance(t.value, ast.Subscript):
                continue
            sub = t.value  # nargs[desc_name]
            if not isinstance(sub.value, ast.Name):
                continue
            desc_name = None
            if isinstance(sub.slice, ast.Constant):
                desc_name = sub.slice.value
            elif getattr(sub.slice, "value", None) is not None and isinstance(sub.slice.value, ast.Constant):
                desc_name = sub.slice.value.value
            try:
                if desc_name is None:
                    desc_name = ast.literal_eval(sub.slice)
            except Exception:
                pass
            if not isinstance(desc_name, str):
                continue
            if not isinstance(node.value, ast.List):
                continue
            desc_bshape = list[str]()
            for elt in node.value.elts:
                if isinstance(elt, ast.Name):
                    # Rewrite hook-local rebinds back to the actual constexpr name.
                    desc_bshape.append(local_to_key.get(elt.id, elt.id))
            if desc_bshape:  # ['BLOCK_M', 'BLOCK_K']
                ret.setdefault(desc_name, list[list[str]]()) \
                   .append(desc_bshape)
        if knobs.autotuning.print:
            # {'a_desc': [['BLOCK_M', 'BLOCK_K'], ['BLOCK_K', 'BLOCK_M']], ...}
            print(f"[Analyzer] hook_desc_bshapes_map = {ret}")
        return ret  # dict[desc_name, list[desc_bshape]]

    # Analyzer 2: desc.load
    def analyze_desc_load_bs(self, pre_hook_fn: object | None = None) -> dict[str, set[tuple[str, ...]]]:
        # 2.0) desc_bshapes_map: dict[desc_name, list[desc_bshape]]
        desc_bshapes_map = dict[str, list[list[str]]]()

        # 2.1) Build desc_bshapes_map from tma_device
        for desc_name, defn in self.tma_device_desc_defs_map.items():
            blist = defn.get("block_shape")
            if blist:
                desc_bshapes_map[desc_name] = [list[str](blist)]

        # 2.2) Extend desc_bshapes_map from tma_host pre_hook
        if pre_hook_fn is not None and hasattr(pre_hook_fn, "__code__"):
            try:
                hook_src = inspect.getsource(pre_hook_fn)
                hook_ast = ast.parse(hook_src)
                for n in ast.walk(hook_ast):
                    if not isinstance(n, ast.FunctionDef):
                        continue
                    hook_desc_bshapes_map = self._parse_hook_desc_bshapes(n)
                    for desc_name, hook_desc_bshapes in hook_desc_bshapes_map.items():
                        desc_bshapes_map.setdefault(desc_name, list[list[str]]()) \
                                             .extend(hook_desc_bshapes)
                    break
            except Exception as e:
                if knobs.autotuning.print:
                    print(f"[AABS] Warning: Parse tma_host desc_bshapes_map failed: {e}")
                pass

        # 2.3) Build transpose_used_vars
        transpose_used_vars = set[str]()  # set[trans_var_name]
        for arg_node in self.transpose_args_nodes:
            for v in VariableCollector.collect(arg_node):
                transpose_used_vars.add(v)
                transpose_used_vars.update(self._get_dependencies_vars(v))
        if knobs.autotuning.print:
            # {'a_t', 'b_t'}
            print(f"[Analyzer] transpose_used_vars = {transpose_used_vars}")

        # 2.4) Build desc_load_info_map: dict[desc_name, list[tuple[is_trans, bshape]]]
        desc_load_info_map = dict[str, list[tuple[bool, list[str]]]]()
        if knobs.autotuning.print and self.tma_load_assignments:
            # [{'var_name': 'a', 'desc_name': 'a_desc', 'addr_exprs': [offset_am, offset_ak]},
            #  {'var_name': 'a_t', 'desc_name': 'a_desc', 'addr_exprs': [offset_ak, offset_am]}, ...]
            print(f"[Analyzer] tma_load_assignments = {self.tma_load_assignments}")
        for tma_load_assignment in self.tma_load_assignments:
            desc_name = tma_load_assignment["desc_name"]
            var_name = tma_load_assignment["var_name"]
            is_trans = var_name in transpose_used_vars
            candidate_bshapes = desc_bshapes_map.get(desc_name) or []
            # candidate_bshapes: [['BLOCK_M', 'BLOCK_K'], ['BLOCK_K', 'BLOCK_M']]
            candidate_bs_names = set[str]()  # {'BLOCK_M', 'BLOCK_K'}
            for candidate_bshape in candidate_bshapes:
                candidate_bs_names.update(candidate_bshape)
            if len(candidate_bshapes) == 1:
                matched_bshape = list[str](candidate_bshapes[0])
            elif len(candidate_bshapes) > 1:
                # var_name='a',   desc_name='a_desc', is_trans=False, matched_bshape=['BLOCK_M', 'BLOCK_K']
                # var_name='a_t', desc_name='a_desc', is_trans=True,  matched_bshape=['BLOCK_K', 'BLOCK_M']
                matched_bshape = self._match_bshape_by_addr(
                    tma_load_assignment.get("addr_exprs") or [], candidate_bshapes, candidate_bs_names)
            else:
                matched_bshape = list[str]()
            if matched_bshape:
                desc_load_info_map.setdefault(desc_name, list[tuple[bool, list[str]]]()) \
                                  .append((is_trans, matched_bshape))
        if knobs.autotuning.print:
            # {'a_desc': [(False, ['BLOCK_M', 'BLOCK_K']), (True, ['BLOCK_K', 'BLOCK_M'])], ...}
            print(f"[Analyzer] desc_load_info_map = {desc_load_info_map}")

        # 2.5) Build tma_map: prefer canonical shape; if only transposed, swap dims
        tma_map = dict[str, set[tuple[str, ...]]]()  # dict[desc_name, set[tuple[bshapes]]]
        for desc_name, load_list in desc_load_info_map.items():
            canonical_shape = None
            trans_shape = None
            for is_trans, shape in load_list:
                if not is_trans:
                    canonical_shape = shape
                else:
                    trans_shape = shape
            if canonical_shape is not None:
                tma_map[desc_name] = {tuple(canonical_shape)}
            elif trans_shape is not None:
                swapped = list[str](trans_shape)
                if len(swapped) >= 2:
                    swapped[-1], swapped[-2] = swapped[-2], swapped[-1]
                tma_map[desc_name] = {tuple(swapped)}
        # {'a_desc', {('BLOCK_M', 'BLOCK_K')}, ...}
        return tma_map

    def _match_bshape_by_addr(self, addr_exprs: list, candidate_bshapes: list[list[str]],
                              candidate_bs_names: set[str]) -> list[str]:
        # candidate_bshapes: [['BLOCK_M', 'BLOCK_K'], ['BLOCK_K', 'BLOCK_M']]
        # candidate_bs_names: {'BLOCK_K', 'BLOCK_M'}
        matched_bshape = list[str | set[str]]()
        for addr_expr in addr_exprs:
            var_names = VariableCollector.collect(addr_expr)
            if isinstance(addr_expr, ast.Name) and addr_expr.id in self.var_definitions:
                var_names |= VariableCollector.collect(self.var_definitions[addr_expr.id])
            # var_names: {'pid_m', 'tl', 'BLOCK_M', 'offset_am'} => matched: {'BLOCK_M'}
            # var_names: {'BLOCK_K', 'tl', 'offset_ak', 'k'} => matched: {'BLOCK_K'}
            matched_bs_set = var_names & candidate_bs_names
            if len(matched_bs_set) == 1:
                # matched_bs_set: {'BLOCK_M'}
                # matched_bs_set: {'BLOCK_K'}
                matched_bshape.append(matched_bs_set.pop())
            else:
                matched_bshape.append(matched_bs_set)
        # matched_bshape = ['BLOCK_M', 'BLOCK_K']
        for candidate_bshape in candidate_bshapes:
            if len(candidate_bshape) == 0:
                if knobs.autotuning.print:
                    print(f"[AABS] Warning: candidate_bshape={candidate_bshape}")
                continue
            if candidate_bshape == matched_bshape:
                return list[str](candidate_bshape)
            if len(candidate_bshape) == len(matched_bshape):
                matched = True
                for i in range(len(candidate_bshape)):
                    candidate_bs = candidate_bshape[i]
                    matched_bs_or_set = matched_bshape[i]
                    if isinstance(matched_bs_or_set, set):
                        if candidate_bs not in matched_bs_or_set:
                            matched = False
                            break
                    elif candidate_bs != matched_bs_or_set:
                        matched = False
                        break
                if matched:
                    return list[str](candidate_bshape)
        if knobs.autotuning.print:
            print("[AABS] Warning: _match_bshape_by_addr failed")
        return list[str](candidate_bshapes[0]) if candidate_bshapes else list[str]()


_analysis_cache = dict[int, tuple]()


# Analyzer
def analyze_kernel_dependencies(jit_fn, pre_hook_fn: object | None = None) -> tuple:
    cache_key = (id(jit_fn), id(pre_hook_fn) if pre_hook_fn is not None else None)
    if cache_key in _analysis_cache:
        return _analysis_cache[cache_key]

    try:
        fn_ast = jit_fn.parse()
        # Pass __globals__ so the analyzer can resolve user-defined helpers
        # whose body wraps tl.cdiv (e.g. prev_multiple_of, next_multiple_of).
        kernel_globals = getattr(jit_fn, '__globals__', None)
        analyzer = KernelDependencyAnalyzer(kernel_globals=kernel_globals)
        analyzer.visit(fn_ast)

        # Analyzer 1: tl.load - tl.arange
        load_map = analyzer.analyze_tl_load_bs()  # BS_M: M
        # Analyzer 2: desc.load - desc.block_shape
        tma_map = analyzer.analyze_desc_load_bs(pre_hook_fn)  # a_desc: (BS_M, BS_K)
        # Analyzer 3: tl.dot M/N/K via tma_map
        tma_m_map, tma_k_map = analyzer.analyze_tma_dot_dim(tma_map)  # BS_M: {A}, BS_K: {A, B}
        # Analyzer 4: tl.dot M/N/K via load_map
        ge_m_map, ge_k_map, ge_n_map = analyzer.analyze_general_dot_dim(load_map)  # BS_M: {A}, BS_K: {A, B}, BS_N: {B}
        # cache
        _analysis_cache[cache_key] = (load_map, tma_map, tma_m_map, tma_k_map, ge_m_map, ge_k_map, ge_n_map)

        if knobs.autotuning.print:
            jit_fn_name = getattr(jit_fn, '__name__', 'unknown')
            if load_map:
                print(f"\n=== [Analyzer] tl.load (by tl.arange): {jit_fn_name} ===")
                for bs_name, ts_name in load_map.items():
                    print(f"  load_map[bs_name, ts_name] = '{bs_name}' -> '{ts_name}'")
            if tma_map:
                print(f"\n=== [Analyzer] desc.load (by block_shape): {jit_fn_name} ===")
                for desc_name, bs_names_set in tma_map.items():
                    print(f"  tma_map[desc_name, set[bshapes]] = '{desc_name}' -> {bs_names_set}")
            if tma_m_map or tma_k_map:
                print(f"\n=== [Analyzer] tma tl.dot: {jit_fn_name} ===")
                print(f"  tma_m_map[bs_name, set[param_name]] = {tma_m_map}")
                print(f"  tma_k_map[bs_name, set[param_name]] = {tma_k_map}")
            if ge_m_map or ge_k_map or ge_n_map:
                print(f"\n=== [Analyzer] general tl.dot: {jit_fn_name} ===")
                print(f"  ge_m_map[bs_name, set[param_name]] = {ge_m_map}")
                print(f"  ge_k_map[bs_name, set[param_name]] = {ge_k_map}")
                print(f"  ge_n_map[bs_name, set[param_name]] = {ge_n_map}")
            print("==============================================================\n")

        return (load_map, tma_map, tma_m_map, tma_k_map, ge_m_map, ge_k_map, ge_n_map)

    except Exception as e:
        print(f"Warning: adjust_kernel_param failed: {e}")
        return (None, None, None, None, None, None, None)


def clear_analysis_cache():
    _analysis_cache.clear()


# ========
# Adjuster
# ========


def update_bs(nargs, current, config, bs_name, bs, title, reason):
    if knobs.autotuning.print:
        print(f'[AABS] {title}: adjust {bs_name} {current[bs_name]} => {bs} because {bs_name} {reason}')
    current[bs_name] = bs
    config.kwargs[bs_name] = bs


def adjust_block_size_tl_load(nargs, current, config, bs_name, ts_name):
    if bs_name not in current or ts_name not in nargs:
        return
    bs = current[bs_name]
    ts = nargs[ts_name]
    if not isinstance(bs, int) or not isinstance(ts, int):
        return
    if bs > ts:  # block_size > tensor_size
        from triton import next_power_of_2
        update_bs(nargs, current, config, bs_name, next_power_of_2(ts), "tl.load", f"> {ts}")


def adjust_block_size_tma(nargs, current, config, desc_name, bs_names):
    import torch
    from triton.tools.tensor_descriptor import TensorDescriptor
    from triton import next_power_of_2
    if desc_name not in nargs:
        return
    if not isinstance(nargs[desc_name], TensorDescriptor):
        return
    desc_base: torch.Tensor = nargs[desc_name].base
    if not isinstance(desc_base, torch.Tensor):
        return
    if len(desc_base.shape) != len(bs_names):
        if knobs.autotuning.print:
            print(
                f"[AABS] Warning: len(desc_base.shape)={len(desc_base.shape)} != {len(bs_names)}=len(bs_names), bs_names={bs_names}"
            )
        return
    for shape_size, bs_name in zip(desc_base.shape, bs_names):
        bs = current[bs_name]
        if not isinstance(shape_size, int) or not isinstance(bs, int):
            continue
        if bs > shape_size:
            update_bs(nargs, current, config, bs_name, next_power_of_2(shape_size), "TMA", f"> {shape_size}")


def adjust_block_size_dot_k_dim(nargs, current, config, tma_k_map, limit):
    for bs_name in tma_k_map.keys():
        if bs_name not in current:
            continue
        bs = current[bs_name]
        if not isinstance(bs, int):
            continue
        if bs < limit:
            update_bs(nargs, current, config, bs_name, limit, "tma tl.dot", f"< {limit}=limit_k")


def adjust_block_size_dot_m_dim(nargs, current, config, tma_k_map, tma_m_map, limit_bytes):
    import torch
    from triton.tools.tensor_descriptor import TensorDescriptor

    bs_k = 1
    for bs_k_name in tma_k_map.keys():
        if bs_k_name in current and isinstance(current[bs_k_name], int):
            bs_k = current[bs_k_name]
            break

    for bs_name, param_names in tma_m_map.items():
        if bs_name not in current:
            continue
        bs = current[bs_name]
        if not isinstance(bs, int):
            continue
        elem_type_size = None
        for pname in param_names:
            if pname not in nargs:
                continue
            narg = nargs[pname]
            if isinstance(narg, TensorDescriptor):
                narg = narg.base
            if isinstance(narg, torch.Tensor):
                elem_type_size = narg.element_size()
                break
        if elem_type_size is None:
            continue
        # SWIZZLE_NONE: bs_k * elem_type_size = 16B, limit = 8
        # SWIZZLE_16B:  bs_k * elem_type_size = 16B, limit = 8
        # SWIZZLE_32B:  bs_k * elem_type_size = 32B, limit = 4
        # SWIZZLE_64B:  bs_k * elem_type_size = 64B, limit = 2
        # SWIZZLE_128B: bs_k * elem_type_size = 128B, limit = 1
        limit = max(int(limit_bytes / bs_k / elem_type_size), 1)
        if bs < limit:
            update_bs(nargs, current, config, bs_name, limit, "tma tl.dot", f"< {limit}=limit_m")


def adjust_block_size_general_dot_mn_dim(nargs, current, config, ge_mn_map, limit):
    for bs_name in ge_mn_map.keys():
        if bs_name not in current:
            continue
        bs = current[bs_name]
        if not isinstance(bs, int):
            continue
        if bs < limit:
            update_bs(nargs, current, config, bs_name, limit, "general tl.dot", f"< {limit}=limit_m/n")


# AABS
def auto_adjust_block_sizes(nargs, fn, configs, current, config):
    pre_hook_fn = getattr(config, "pre_hook", None) or (configs[0].pre_hook if configs else None)
    load_map, tma_map, tma_m_map, tma_k_map, ge_m_map, ge_k_map, ge_n_map = analyze_kernel_dependencies(
        fn, pre_hook_fn=pre_hook_fn)

    if load_map:  # tl.load or tma_device.load
        if knobs.autotuning.print:
            print("[AABS] 1. adjust bs in tl.load or tma_device.load")
        for bs_name, ts_name in load_map.items():
            adjust_block_size_tl_load(nargs, current, config, bs_name, ts_name)

    if tma_map:  # tma_host.load
        if knobs.autotuning.print:
            print("[AABS] 2. adjust bs in tma_host.load")
        for desc_name, bs_names_set in tma_map.items():
            for bs_names in bs_names_set:
                adjust_block_size_tma(nargs, current, config, desc_name, bs_names)

    if tma_k_map or tma_m_map:  # tl.dot with tma_device or tma_host
        if knobs.autotuning.print:
            print("[AABS] 3. adjust bs in tl.dot with tma_device or tma_host")
        adjust_block_size_dot_k_dim(nargs, current, config, tma_k_map, 16)
        adjust_block_size_dot_m_dim(nargs, current, config, tma_k_map, tma_m_map, 128)

    if ge_m_map or ge_n_map:  # tl.dot with general tl.load
        if FLAGTREE_BACKEND == "hcu":
            if knobs.autotuning.print:
                print("[AABS] 4. adjust bs in tl.dot with general tl.load")
            adjust_block_size_general_dot_mn_dim(nargs, current, config, ge_m_map, 16)
            adjust_block_size_general_dot_mn_dim(nargs, current, config, ge_n_map, 16)

    if knobs.autotuning.print:
        nargs_str = ''
        if nargs:
            import torch
            from triton.tools.tensor_descriptor import TensorDescriptor
            nargs_parts = list[str]()
            for k, v in nargs.items():
                if isinstance(v, (torch.Tensor, TensorDescriptor)):
                    nargs_parts.append(k)
                else:
                    nargs_parts.append(f'{k}={v}')
            nargs_str = ', '.join(nargs_parts)
        base_fn = fn
        while not inspect.isfunction(base_fn):
            base_fn = base_fn.fn
        print(f'[AABS] ==== Finish: {base_fn.__name__}({nargs_str})')
        print(f'[AABS] ====         adjusted_config={config}')
        print("==============================================================\n")
