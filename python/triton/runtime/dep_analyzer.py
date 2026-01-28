import ast
from typing import Dict, Set, Optional, Tuple, List
from functools import lru_cache


class VariableCollector(ast.NodeVisitor):

    def __init__(self):
        self.variables: Set[str] = set()

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
        # 对于 tl.arange 这样的形式，不收集 tl
        # 对于 a.b 形式，收集 a
        if isinstance(node.value, ast.Name):
            # 跳过模块前缀如 tl, np 等
            if node.value.id not in ('tl', 'triton', 'np', 'torch'):
                self.variables.add(node.value.id)
        self.generic_visit(node)

    @staticmethod
    def collect(node) -> Set[str]:
        """收集 AST 节点中的所有变量"""
        collector = VariableCollector()
        collector.visit(node)
        return collector.variables


class KernelDependencyAnalyzer(ast.NodeVisitor):

    def __init__(self):
        self.input_params: Set[str] = set()      # 输入参数名集合
        self.constexpr_params: Set[str] = set()  # constexpr 参数名集合
        self.var_definitions: Dict[str, ast.AST] = {}  # 变量名 -> AST 定义节点
        # for input-constexpr dependencies analyze
        self.load_addresses: list = []  # 存储 tl.load 的地址表达式
        # for make_tensor_descriptor dependencies analyze
        self.tma_args = {} # 存储 tl.make_tensor_descriptor 的 base 及其对应的 stride 和 block shape
        # for TMA descriptor load dependencies analyze
        self.tma_load_assignments = []  # 存储 tma_desc.load 赋值的目标变量和相关信息
        self.transpose_args_nodes = []  # 存储 tl.trans 参数

    def visit_FunctionDef(self, node):
        """分析函数定义，收集参数信息"""
        # 收集所有参数
        for arg in node.args.args:
            arg_name = arg.arg
            self.input_params.add(arg_name)

            # 检查是否是 constexpr
            if arg.annotation:
                ann_str = ast.unparse(arg.annotation) if hasattr(ast, 'unparse') else ''
                if not ann_str:
                    # Python 3.8 fallback
                    try:
                        ann_str = ast.dump(arg.annotation)
                    except:
                        ann_str = ''
                if 'constexpr' in ann_str:
                    self.constexpr_params.add(arg_name)

        # 继续分析函数体
        self.generic_visit(node)

    def visit_Assign(self, node):
        """分析赋值语句，记录变量定义"""
        targets = node.targets
        if len(targets) == 1 and isinstance(targets[0], ast.Name):
            var_name = targets[0].id

            # 检查右侧是否是TMA load调用
            if (isinstance(node.value, ast.Call) and
                self._is_tma_load(node.value) and
                node.value.args and
                isinstance(node.value.args[0], ast.List)):
                # 记录TMA load的赋值目标和相关信息
                tma_desc_name = node.value.func.value.id
                addr_exprs = node.value.args[0].elts
                self.tma_load_assignments.append({
                    'var_name': var_name,
                    'tma_desc_name': tma_desc_name,
                    'addr_exprs': addr_exprs
                })

            self.var_definitions[var_name] = node.value
        self.generic_visit(node)

    def visit_AnnAssign(self, node):
        """分析带注解的赋值语句"""
        if isinstance(node.target, ast.Name):
            var_name = node.target.id
            self.var_definitions[var_name] = node.value

            # 检查是否是 constexpr
            if node.annotation:
                ann_str = ast.unparse(node.annotation) if hasattr(ast, 'unparse') else ''
                if not ann_str:
                    try:
                        ann_str = ast.dump(node.annotation)
                    except:
                        ann_str = ''
                if 'constexpr' in ann_str:
                    self.constexpr_params.add(var_name)

        self.generic_visit(node)

    def visit_Call(self, node):
        """分析函数调用，捕获 tl.load"""
        # 检查是否是 tl.load 调用
        if self._is_tl_load(node) and node.args:
            self.load_addresses.append(node.args[0])
        elif self._is_tl_transpose(node) and node.args:
            # 获取 transpose 的参数
            self.transpose_args_nodes.append(node.args[0])
        elif self._is_tl_make_tensor_descriptor(node):
            base = None
            # 收集 make_tensor_descriptor 中的 base 的节点
            for kw in node.keywords:
                if hasattr(kw, 'arg') and kw.arg == 'base':
                    if kw.value not in self.tma_args:
                        base = kw.value
                        self.tma_args[base] = {'strides': [], 'block_shape': []}
            # 收集 make_tensor_descriptor 中的 stride 和 block_shape 元素的节点
            for kw in node.keywords:
                if hasattr(kw, 'arg') and (kw.arg in ['strides', 'block_shape']):
                    if hasattr(kw, 'value') and isinstance(kw.value, ast.List):
                        for elt in kw.value.elts:
                            self.tma_args[base][kw.arg].append(elt)
        self.generic_visit(node)

    def _is_tl_load(self, node) -> bool:
        """检查是否是 tl.load 调用"""
        if isinstance(node.func, ast.Attribute):
            if node.func.attr == 'load':
                if isinstance(node.func.value, ast.Name):
                    return node.func.value.id in ('tl', 'triton')
        return False

    def _is_tma_load(self, node) -> bool:
        """检查是否是 tma_desc.load 调用"""
        if isinstance(node.func, ast.Attribute):
            if node.func.attr == 'load':
                if isinstance(node.func.value, ast.Name):
                    return node.func.value.id not in ('tl', 'triton')
        return False

    def _is_tl_make_tensor_descriptor(self, node) -> bool:
        """检查是否是 tl.make_tensor_descriptor 调用"""
        if isinstance(node.func, ast.Attribute):
            if node.func.attr == 'make_tensor_descriptor':
                if isinstance(node.func.value, ast.Name):
                    return node.func.value.id in ('tl', 'triton')
        return False

    def _is_tl_transpose(self, node) -> bool:
        """检查是否是 tl.trans 或 triton.trans 调用"""
        if isinstance(node.func, ast.Attribute):
            if node.func.attr == 'trans':
                if isinstance(node.func.value, ast.Name):
                    return node.func.value.id in ('tl', 'triton')
        return False

    def get_dependencies(self, var_name: str, visited: Optional[Set[str]] = None) -> tuple:
        """
        递归获取变量依赖的输入参数和 constexpr

        Returns:
            (input_deps, constexpr_deps): 两个集合
        """
        if visited is None:
            visited = set()

        if var_name in visited:
            return set(), set()
        visited.add(var_name)

        input_deps = set()
        constexpr_deps = set()

        # 检查是否是非 constexpr 的输入参数
        if var_name in self.input_params and var_name not in self.constexpr_params:
            input_deps.add(var_name)
            return input_deps, constexpr_deps

        # 检查是否是 constexpr
        if var_name in self.constexpr_params:
            constexpr_deps.add(var_name)
            return input_deps, constexpr_deps

        # 递归分析变量定义
        if var_name in self.var_definitions and not var_name.startswith('pid'):
            definition_node = self.var_definitions[var_name]
            used_vars = VariableCollector.collect(definition_node)
            for used_var in used_vars:
                sub_inputs, sub_constexprs = self.get_dependencies(used_var, visited.copy())
                input_deps.update(sub_inputs)
                constexpr_deps.update(sub_constexprs)

        return input_deps, constexpr_deps

    def _get_dependencies_vars(self, var_name: str, visited: Optional[Set[str]] = None) -> bool:
        """
        返回变量 var_name 依赖的所有变量
        """
        if visited is None:
            visited = set()

        if var_name in visited:
            return set()
        visited.add(var_name)

        var_deps = set()

        # 检查是否是 输入 或 constexpr 参数
        if (var_name in self.input_params) or (var_name in self.constexpr_params):
            return var_deps

        # 递归分析变量定义
        if var_name in self.var_definitions and not var_name.startswith('pid'):
            definition_node = self.var_definitions[var_name]
            used_vars = VariableCollector.collect(definition_node)
            for used_var in used_vars:
                used_var_deps = self._get_dependencies_vars(used_var, visited.copy())
                var_deps.update(used_var_deps)

        return var_deps

    def analyze_tma_make_tensor_descriptor(self) -> Dict[str, List[str]]:
        tma_relationships = {}

        for arg_base in self.tma_args.keys():
            base_name = None
            blck_shape_name = None

            # 收集 base 中使用的变量
            base_used_vars = VariableCollector.collect(arg_base)

            # 分析每个变量的依赖
            for base_used_var_name in base_used_vars:
                input_deps, _ = self.get_dependencies(base_used_var_name)
                if len(input_deps) == 1:
                    base_name = list(input_deps)[0]

            # 找到最后一维 stride 对应的 block shape
            last_stride = self.tma_args[arg_base]['strides'][-1]
            last_blck_shape = self.tma_args[arg_base]['block_shape'][-1]

            if isinstance(last_stride, ast.Constant) and last_stride.value == 1:
                # 收集最后一维 block shape 中使用的变量
                blck_shape_used_vars = VariableCollector.collect(last_blck_shape)

                # 分析每个变量的依赖
                for blck_shape_used_var_name in blck_shape_used_vars:
                    _, constexpr_deps = self.get_dependencies(blck_shape_used_var_name)
                    if len(constexpr_deps) == 1:
                        blck_shape_name = list(constexpr_deps)[0]

            if blck_shape_name not in tma_relationships:
                tma_relationships[blck_shape_name] = [base_name]
            else:
                tma_relationships[blck_shape_name].append(base_name)

        return tma_relationships

    def analyze_tma_desc_load_with_trans_check(self):
        """
        分析TMA描述符load操作及其后续是否有trans操作
        返回: [(tma_desc, block_names, has_trans), ...]
        """
        tma_desc_relationships = {}

        # 收集 transpose 参数直接使用的变量名
        transpose_used_vars = set()
        for arg_node in self.transpose_args_nodes:
            used_vars = VariableCollector.collect(arg_node)
            # 收集地址表达式中使用的变量
            for var_name in used_vars:
                transpose_used_vars.add(var_name)
                transpose_used_vars.update(self._get_dependencies_vars(var_name))

        # 检查每个TMA load后面是否紧跟trans调用
        for tma_info in self.tma_load_assignments:
            target_var = tma_info['var_name']
            tma_desc_name = tma_info['tma_desc_name']
            addr_exprs = tma_info['addr_exprs']

            # 分析TMA load地址表达式中的block names
            block_names = []
            for addr_expr in addr_exprs:
                used_vars = VariableCollector.collect(addr_expr)
                for var_name in used_vars:
                    _, constexpr_deps = self.get_dependencies(var_name)
                    if len(constexpr_deps) == 1:
                        block_names.append(list(constexpr_deps)[0])
                        break

            if target_var in transpose_used_vars:
                block_names[-1], block_names[-2] = block_names[-2], block_names[-1]

            # 添加每个 TMA Descriptor 对应的 block names（可能有多对）
            if tma_desc_name in tma_desc_relationships.keys():
                tma_desc_relationships[tma_desc_name].add(tuple(block_names))
            else:
                tma_desc_relationships[tma_desc_name] = set()
                tma_desc_relationships[tma_desc_name].add(tuple(block_names))

        return tma_desc_relationships

    def analyze(self) -> Dict[str, Set[str]]:
        """
        分析所有 tl.load 或 TMA相关的地址表达式，返回参数-constexpr 依赖关系

        Returns:
            dict: {input_param: set(related_constexpr_params)}
        """
        relationships = {}

        for addr_expr in self.load_addresses:
            # 收集地址表达式中使用的变量
            used_vars = VariableCollector.collect(addr_expr)

            # 分析每个变量的依赖
            for var_name in used_vars:
                input_deps, constexpr_deps = self.get_dependencies(var_name)
                # 如果同时依赖输入参数和 constexpr，且都分别只依赖一个，则记录关系
                if len(input_deps) == 1 and len(constexpr_deps) == 1:
                    input_dep = list(input_deps)[0]
                    constexpr_dep = list(constexpr_deps)[0]
                    relationships[input_dep] = constexpr_dep

        return relationships


# 缓存分析结果，避免重复分析
_analysis_cache: Dict[int, Tuple[Dict[str, Set[str]]]] = {}


def analyze_kernel_dependencies(jit_fn) -> Dict[str, Set[str]]:
    # 检查缓存
    fn_id = id(jit_fn)
    if fn_id in _analysis_cache:
        return _analysis_cache[fn_id]

    try:
        # 获取函数的 AST
        fn_ast = jit_fn.parse()

        # 创建分析器并分析
        analyzer = KernelDependencyAnalyzer()
        analyzer.visit(fn_ast)
        relationships = analyzer.analyze()
        tma_make_desc_relationships = analyzer.analyze_tma_make_tensor_descriptor()
        tma_desc_load_relationships = analyzer.analyze_tma_desc_load_with_trans_check()

        # 缓存结果
        _analysis_cache[fn_id] = (relationships, tma_make_desc_relationships, tma_desc_load_relationships)

        # 可选：打印分析结果
        import os
        if os.environ.get('PRINT_FLAGTREE_DEPENDENCY_ANALYZER', '0') == '1':
            if relationships:
                print(f"\n=== Kernel 依赖分析: {getattr(jit_fn, '__name__', 'unknown')} ===")
                for param, constexprs in relationships.items():
                    print(f"  输入参数 '{param}' 与常量 {constexprs} 相关")
            if tma_make_desc_relationships:
                print(f"\n=== TMA 块形状依赖分析: {getattr(jit_fn, '__name__', 'unknown')} ===")
                for blck_shape, bases in tma_make_desc_relationships.items():
                    print(f"  最低维度块形状 '{blck_shape}' 与基 {bases} 相关")
            if tma_desc_load_relationships:
                print(f"\n=== TMA 输入描述符块形状依赖分析: {getattr(jit_fn, '__name__', 'unknown')} ===")
                for tma_desc, block_names in tma_desc_load_relationships.items():
                    print(f"  TMA 输入描述符 '{tma_desc}' 与块形状 {block_names} 相关")

        return (relationships, tma_make_desc_relationships, tma_desc_load_relationships)

    except Exception as e:
        # 分析失败时返回空字典
        import os
        print(f"Warning: Dependency analysis failed: {e}")
        return {}


def clear_analysis_cache():
    global _analysis_cache
    _analysis_cache.clear()
