import ast
from typing import Dict, Set, Optional
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
        self.load_addresses: list = []  # 存储 tl.load 的地址表达式

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
        self.generic_visit(node)

    def _is_tl_load(self, node) -> bool:
        """检查是否是 tl.load 调用"""
        if isinstance(node.func, ast.Attribute):
            if node.func.attr == 'load':
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

    def analyze(self) -> Dict[str, Set[str]]:
        """
        分析所有 tl.load 的地址表达式，返回参数-constexpr 依赖关系

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
_analysis_cache: Dict[int, Dict[str, Set[str]]] = {}


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

        # 缓存结果
        _analysis_cache[fn_id] = relationships

        # 可选：打印分析结果
        import os
        if os.environ.get('PRINT_FLAGTREE_DEPENDENCY_ANALYZER', '0') == '1':
            if relationships:
                print(f"\n=== Kernel 依赖分析: {getattr(jit_fn, '__name__', 'unknown')} ===")
                for param, constexprs in relationships.items():
                    print(f"  输入参数 '{param}' 与常量 {constexprs} 相关")

        return relationships

    except Exception as e:
        # 分析失败时返回空字典
        import os
        print(f"Warning: Dependency analysis failed: {e}")
        return {}


def clear_analysis_cache():
    global _analysis_cache
    _analysis_cache.clear()
