# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

import ast
from typing import Dict, List, Union


class AutoParser(ast.NodeVisitor):
    """
    Base class for parsing triton dsl kernel code using AST analysis.

    Provides common functionality for traversing the AST (abstract syntax tree)
    of a triton dsl kernel function and identifying specific elements in the code.

    Subclassed should implement specific parsing logic by overriding the relevant
    node visit methods.
    """

    def __init__(self, func_ast: ast.AST):
        self.func_ast = func_ast

    def parse(self):
        self.visit(self.func_ast)

    def contains_target_var(self, node, var):
        """
        Recursively checks if a given AST node or its children contain a reference
        to the specified variable.

        :param node: the AST node to check
        :type node: ast.AST
        :param var: the variable name to search for
        :type var: str
        :return: True if the variable is found, False otherwise
        """
        if isinstance(node, ast.Name) and node.id == var:
            return True
        for _, value in ast.iter_fields(node):
            if isinstance(value, list):
                for item in value:
                    if self.contains_target_var(item, var):
                        return True
            elif isinstance(value, ast.AST):
                if self.contains_target_var(value, var):
                    return True
        return False


class AxesKeyParser(AutoParser):
    """
    A parser for extracting axis information from a given function's AST.
    This class is designed to handle specific patterns in the function's code to
    determine the axis associated with a given variable. It is particularly useful
    for parsing triton DSL kernel code and identifying axis information.
    It recursively processes assignment nodes and lessthan nodes to obtain the axes
    corresponding to the specified var in the given function.
    """

    def __init__(self, func_ast: ast.AST, keys: Dict[str, str]):
        super().__init__(func_ast)
        self.keys = keys
        self.checked_vars = list()

    def get_axis(self, var: str, node=None):
        """
        Traverse the AST using the provided variable name and mask-based less-than
        operations to obtain the corresponding axis name.

        :param var: the variable name to get the corresponding axis.
        :type var: str
        """
        if var in self.checked_vars:
            return None
        axis = None
        if not node:
            node = self.func_ast
        for child_node in ast.walk(node):
            # handle compare node
            if isinstance(child_node, ast.Compare):
                axis = self.handle_lt_node(var, child_node)
            elif isinstance(child_node, ast.Assign):
                axis = self.handle_assign_node(var, child_node)

            elif isinstance(child_node, ast.BinOp) and \
                 isinstance(child_node.op, ast.BitAnd):

                axis = self.handle_lt_node(var, child_node.left)
                if axis is None:
                    axis = self.handle_lt_node(var, child_node.right)

            if axis is not None:
                return axis
        self.checked_vars.append(var)
        return None

    def handle_assign_node(self, var, node):
        if not isinstance(node, ast.Assign) or not isinstance(node.targets, list):
            return None
        if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
            return None

        target = node.targets[0].id
        if target in self.checked_vars:
            return None
        # Prevent cyclic assignment.
        if var == target or not self.contains_target_var(node.value, var):
            return None

        axis = self.get_axis(var, node.value)
        if axis:
            return axis

        axis = self.get_axis(target)
        return axis

    def handle_lt_node(self, var, node):
        if not isinstance(node, ast.Compare) or not isinstance(node.ops, list):
            return None
        if len(node.ops) != 1 or not isinstance(node.ops[0], ast.Lt):
            return None
        if not isinstance(node.comparators, list) or len(node.comparators) != 1:
            return None
        if not isinstance(node.left, ast.Name) or var != node.left.id:
            return None

        comparator = node.comparators[0]
        if not isinstance(comparator, ast.Name) and \
           not (isinstance(comparator, ast.Call) and \
           isinstance(comparator.func, ast.Name) and \
           comparator.func.id == 'min'):
            return None

        for k, v in self.keys.items():
            if self.contains_target_var(comparator, v):
                return k
        return None


class SplitAxesParser(AxesKeyParser):
    """
    Extracts the split axis parameters from triton kernel code. The parsing is based on the
    `tl.program_id` statement. This class identifies potential split axes by analyzing the usage
    of the `tl.program_id` variable in the program and its multiplication operations with other
    variables(currently supporting scenarios where multiplication is either direct or indirect via
    intermediate variables). It then filters these candidates based on a list of candidate parameters
    (parameters not provided by the user). After that, it confirms the split axis corresponding to
    the current parameter using mask comparison and the `keys` passed in `autotune`.

    Note:
    1. Split axis parameters must be multiplied with `tl.program_id`.
    2. Without mask comparision, it is impossible to confirm the exact split axis, which would lead
       to parameter parsing failure. (eg. mask = offsets < n_elements)
    3. The identified split axes are limited to the list of candidated parameters, ensuring that
       only those parameters that can be dynamically adjusted through the autotune process are considered.
    """

    def __init__(self, func_ast: ast.AST, keys: Dict[str, str], candidates_params: List[str]):
        """
        :param func_ast: Abstract syntax tree of the triton kernel function
        :type func_ast: ast.AST
        :param keys: a dict of axis name: argument name, used to confirm the split axis corresponding to
            the split axis parameters.
        :type keys: Dict[str, str]
        :param candidates_params: a list of parameters names that were not provided by the user when calling
            triton kernel function. The parser will only consider these parameters as potential split axis
            parameters.
        :type candidates_params: List[str]
        """
        super().__init__(func_ast, keys)
        self.split_axes = dict()
        self.program_id_vars = list()
        self.program_id_var_dims = dict()
        self.num_programs_var_dims = dict()
        self.grid_stride_tiling_only = dict()
        # axis_name -> program_id axis dim
        self.split_axis_pid_dims = dict()
        # axis_name -> program_id axis dim (includes axes inferred without split params)
        self.axis_pid_dims = dict()
        self.candidates_params = candidates_params

    def parse(self) -> Dict[str, str]:
        super().parse()
        return self.split_axes

    def visit_Assign(self, node):
        pid_dim = self._get_program_id_dim(node.value)
        if pid_dim is not None:
            if (len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
                    and node.targets[0].id not in self.program_id_vars):
                self.program_id_vars.append(node.targets[0].id)
                self.program_id_var_dims[node.targets[0].id] = pid_dim
        num_programs_dim = self._get_num_programs_dim(node.value)
        if num_programs_dim is not None:
            if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                self.num_programs_var_dims[node.targets[0].id] = num_programs_dim
        self.generic_visit(node)

    def visit_BinOp(self, node):
        if isinstance(node.op, ast.Mult):
            split_axes_val = None
            split_axis_pid_dim = None
            if isinstance(node.left, ast.Name) and node.left.id in self.program_id_vars:
                if isinstance(node.right, ast.Name):
                    split_axes_val = node.right.id
                    split_axis_pid_dim = self.program_id_var_dims.get(node.left.id)
            elif isinstance(node.left, ast.Call) and isinstance(node.left.func, ast.Attribute):
                if (isinstance(node.left.func.value, ast.Name) and node.left.func.value.id == "tl"
                        and node.left.func.attr == "program_id"):
                    if isinstance(node.right, ast.Name):
                        split_axes_val = node.right.id
                        split_axis_pid_dim = self._get_program_id_dim(node.left)

            if isinstance(node.right, ast.Name) and node.right.id in self.program_id_vars:
                if isinstance(node.left, ast.Name):
                    split_axes_val = node.left.id
                    split_axis_pid_dim = self.program_id_var_dims.get(node.right.id)
            elif isinstance(node.right, ast.Call) and isinstance(node.right.func, ast.Attribute):
                if (isinstance(node.right.func.value, ast.Name) and node.right.func.value.id == "tl"
                        and node.right.func.attr == "program_id"):
                    if isinstance(node.left, ast.Name):
                        split_axes_val = node.left.id
                        split_axis_pid_dim = self._get_program_id_dim(node.right)

            if split_axes_val in self.candidates_params and \
               split_axes_val not in self.split_axes.values():
                split_axes_key = self.get_axis(split_axes_val)
                if split_axes_key and not self._is_tiling_only_split(split_axes_key, split_axes_val):
                    self.split_axes[split_axes_key] = split_axes_val
                    if split_axis_pid_dim is not None:
                        self._record_axis_pid_dim(split_axes_key, split_axis_pid_dim)
        self.generic_visit(node)

    def visit_For(self, node):
        if not isinstance(node.iter, ast.Call):
            self.generic_visit(node)
            return

        iter_fn = node.iter.func
        is_range = isinstance(iter_fn, ast.Name) and iter_fn.id == "range"
        is_tl_range = (isinstance(iter_fn, ast.Attribute) and isinstance(iter_fn.value, ast.Name)
                       and iter_fn.value.id == "tl" and iter_fn.attr == "range")
        if not (is_range or is_tl_range):
            self.generic_visit(node)
            return

        if len(node.iter.args) == 0:
            self.generic_visit(node)
            return

        start = node.iter.args[0] if len(node.iter.args) >= 2 else None
        stop = node.iter.args[1] if len(node.iter.args) >= 2 else node.iter.args[0]
        pid_dim = self._extract_pid_dim_from_expr(start)
        axis = self._axis_from_expr(stop)
        if axis is not None and pid_dim is not None:
            self._record_axis_pid_dim(axis, pid_dim)
            if len(node.iter.args) >= 3:
                step = node.iter.args[2]
                loop_tiling_only_param = self._extract_grid_stride_split_param(start, step, pid_dim)
                if loop_tiling_only_param is not None:
                    self._mark_tiling_only_param(axis, loop_tiling_only_param)

        self.generic_visit(node)

    def _get_program_id_dim(self, node):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and isinstance(
                node.func.value, ast.Name) and node.func.value.id == "tl" and node.func.attr == "program_id"):
            return None

        axis_dim = 0
        if len(node.args) > 0:
            if isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, int):
                axis_dim = node.args[0].value
            else:
                return None

        for kw in node.keywords:
            if kw.arg == "axis":
                if isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, int):
                    axis_dim = kw.value.value
                else:
                    return None
                break
        return axis_dim

    def _get_num_programs_dim(self, node):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and isinstance(
                node.func.value, ast.Name) and node.func.value.id == "tl" and node.func.attr == "num_programs"):
            return None

        axis_dim = 0
        if len(node.args) > 0:
            if isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, int):
                axis_dim = node.args[0].value
            else:
                return None

        for kw in node.keywords:
            if kw.arg == "axis":
                if isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, int):
                    axis_dim = kw.value.value
                else:
                    return None
                break
        return axis_dim

    def _extract_pid_dim_from_expr(self, node):
        if node is None:
            return None
        for child in ast.walk(node):
            if isinstance(child, ast.Name) and child.id in self.program_id_var_dims:
                return self.program_id_var_dims[child.id]
            pid_dim = self._get_program_id_dim(child)
            if pid_dim is not None:
                return pid_dim
        return None

    def _contains_pid_dim(self, node, pid_dim):
        if node is None:
            return False
        for child in ast.walk(node):
            if isinstance(child, ast.Name):
                if self.program_id_var_dims.get(child.id, None) == pid_dim:
                    return True
            if self._get_program_id_dim(child) == pid_dim:
                return True
        return False

    def _contains_num_programs_dim(self, node, pid_dim):
        if node is None:
            return False
        for child in ast.walk(node):
            if isinstance(child, ast.Name):
                if self.num_programs_var_dims.get(child.id, None) == pid_dim:
                    return True
            if self._get_num_programs_dim(child) == pid_dim:
                return True
        return False

    def _is_candidate_name(self, node, candidate_name):
        return (isinstance(node, ast.Name) and node.id == candidate_name and candidate_name in self.candidates_params)

    def _extract_pid_multiplied_candidate(self, node, pid_dim):
        if node is None:
            return None
        candidates = set()
        for child in ast.walk(node):
            if not isinstance(child, ast.BinOp) or not isinstance(child.op, ast.Mult):
                continue
            left = child.left
            right = child.right
            if isinstance(left, ast.Name) and left.id in self.candidates_params and \
               self._contains_pid_dim(right, pid_dim):
                candidates.add(left.id)
            if isinstance(right, ast.Name) and right.id in self.candidates_params and \
               self._contains_pid_dim(left, pid_dim):
                candidates.add(right.id)
        if len(candidates) == 1:
            return next(iter(candidates))
        return None

    def _contains_num_programs_multiplied_candidate(self, node, candidate_name, pid_dim):
        if node is None:
            return False
        for child in ast.walk(node):
            if not isinstance(child, ast.BinOp) or not isinstance(child.op, ast.Mult):
                continue
            if self._is_candidate_name(child.left, candidate_name):
                if self._contains_num_programs_dim(child.right, pid_dim):
                    return True
            if self._is_candidate_name(child.right, candidate_name):
                if self._contains_num_programs_dim(child.left, pid_dim):
                    return True
        return False

    def _extract_grid_stride_split_param(self, start, step, pid_dim):
        if start is None or step is None:
            return None
        candidate_name = self._extract_pid_multiplied_candidate(start, pid_dim)
        if candidate_name is None:
            return None
        if self._contains_num_programs_multiplied_candidate(step, candidate_name, pid_dim):
            return candidate_name
        return None

    def _mark_tiling_only_param(self, axis, candidate_name):
        self.grid_stride_tiling_only.setdefault(axis, set()).add(candidate_name)
        if self.split_axes.get(axis, None) == candidate_name:
            del self.split_axes[axis]
            self.split_axis_pid_dims.pop(axis, None)

    def _is_tiling_only_split(self, axis, candidate_name):
        return candidate_name in self.grid_stride_tiling_only.get(axis, set())

    def _axis_from_expr(self, node):
        if node is None:
            return None
        for k, v in self.keys.items():
            if self.contains_target_var(node, v):
                return k
        return None

    def _record_axis_pid_dim(self, axis, pid_dim):
        self.axis_pid_dims[axis] = pid_dim
        if axis in self.split_axes:
            self.split_axis_pid_dims[axis] = pid_dim


class TilingAxesParser(AxesKeyParser):
    """
    Extracts the tiling axis parameters from triton kernel code. The parsing is based on the
    `tl.arange`, `tl.range` and `range()` statement. This class identifies potential tiling axes by analyzing
    the usage of the `range` and `tl.range` within `for` loop in the program. Common parameters
    between `range()` or `tl.range` and `tl.arange` are extracted. It then filters these candidates based on a
    list of candidate parameters (parameters not provided by the user). After that, it confirms the
    tiling axis corresponding to the current parameter using mask comparison and the `keys` passed
    in `autotune`.

    Note:
    1. Tiling axis parameters must be calculated within the `tl.arange` function and the `for` loop
       using `tl.range`.
    2. Without mask comparision, it is impossible to confirm the exact tiling axis, which would lead
       to parameter parsing failure. (eg. mask = offsets < n_elements).
    3. The identified tiling axes are limited to the list of candidated parameters, ensuring that
       only those parameters that can be dynamically adjusted through the autotune process are considered.
    """

    def __init__(self, func_ast: ast.AST, keys: Dict[str, str], candidates_params: List[str]):
        """
        :param func_ast: Abstract syntax tree of the triton kernel function
        :type func_ast: ast.AST
        :param keys: a dict of axis name: argument name, used to confirm the tiling axis corresponding to
            the tiling axis parameters.
        :type keys: Dict[str, str]
        :param candidates_params: a list of parameters names that were not provided by the user when calling
            triton kernel function. The parser will only consider these parameters as potential tiling axis
            parameters.
        :type candidates_params: List[str]
        """
        super().__init__(func_ast, keys)
        self.tiling_axes = dict()
        self.candidates_params = candidates_params
        self.candidates_params_for_loop = list()

    def parse(self) -> Dict[str, str]:
        super().parse()
        return self.tiling_axes

    def visit_For(self, node):
        if isinstance(node.iter, ast.Call) and len(node.iter.args) == 3:
            step_expr = node.iter.args[2]
            for_loop_param = self._extract_unique_candidate(step_expr)
            if (for_loop_param is not None and for_loop_param not in self.candidates_params_for_loop):
                self.candidates_params_for_loop.append(for_loop_param)
        self.generic_visit(node)

    def visit_Assign(self, node):
        if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            # handle FloorDiv
            if isinstance(node.value, ast.BinOp) and isinstance(node.value.op, ast.FloorDiv):
                denominator = node.value.right
                denominator_param = self._extract_unique_candidate(denominator)
                if denominator_param is not None and \
                   denominator_param not in self.candidates_params_for_loop:
                    self.candidates_params_for_loop.append(denominator_param)
                    self.visit(self.func_ast)

            tiling_axes_val = self.get_tiling_axes_val(node.value)
            if tiling_axes_val is not None and \
               tiling_axes_val in self.candidates_params_for_loop:
                tiling_axes_key = self.get_axis(tiling_axes_val)
                if tiling_axes_key:
                    self.tiling_axes[tiling_axes_key] = tiling_axes_val
        self.generic_visit(node)

    def get_tiling_axes_val(self, node):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr == 'arange' and \
               isinstance(node.func.value, ast.Name) and \
               node.func.value.id == 'tl':
                if isinstance(node.args, list) and len(node.args) == 2:
                    for param in self.candidates_params_for_loop:
                        if self.contains_target_var(node.args[1], param):
                            return param

        for _, value in ast.iter_fields(node):
            if isinstance(value, list):
                for item in value:
                    val = self.get_tiling_axes_val(item)
                    if val:
                        return val
            elif isinstance(value, ast.AST):
                val = self.get_tiling_axes_val(value)
                if val:
                    return val
        return None

    def _extract_unique_candidate(self, expr):
        """
        Extract a unique tiling candidate from an expression.
        Return None when no candidate or ambiguous (more than one candidate) appears.
        """
        if expr is None:
            return None
        candidates = [param for param in self.candidates_params if self.contains_target_var(expr, param)]
        if len(candidates) == 1:
            return candidates[0]
        return None


class ReductionAxesParser(AxesKeyParser):
    """
    Extracts the reduction axis from triton kernel code. The parsing is based on the
    reduction function (eg. tl.max, tl.min, tl.sum, ...). This class identifies the
    dimensions of reduction operations by analyzing the reduction function calls in
    the program. After that, It confirms the reduction axis corresponding to the current
    parameter using mask comparison and the keys passed in autotune.

    Note:
    1. The call to the reduction function must start with 'tl', meaning the function must
       be a function from triton.language
    2. It's preferable to specify the reduction axis dimension in the reduction function
       using keyword arguments(eg. axis=xxx). Otherwise, specifying it via positional
       arguments may lead to errors.
    3. Mask comparison must be performed on the potential reduction axis length parameters,
       and the comparison parameters or target parameters of the comparison expression
       must be sliced. Otherwise, the correspondence between dimensions and axes cannot
       be confirmed, which will lead to failure in parsing the reduction axis.
    4. The identified reduction axes are limited to the candidate list provided in the keys.
    """

    def __init__(self, func_ast: ast.AST, keys: Dict[str, str]):
        """
        :param func_ast: Abstract syntax tree of the triton kernel function
        :type func_ast: ast.AST
        :param keys: a dict of axis name: argument name, used to confirm the reduction axis.
        :type keys: Dict[str, str]
        """
        super().__init__(func_ast, keys)
        self.reduction_axes = list()
        self.reduction_func = ('sum', 'xor_sum', 'max', 'min', 'argmax', 'argmin')  # tl.xxx
        self.ndim = 1

    def parse(self) -> List[str]:
        super().parse()
        return self.reduction_axes

    def visit_Assign(self, node):
        self._scan_subscripts(node.value)
        self.generic_visit(node)

    def _scan_subscripts(self, node):
        if isinstance(node, ast.Subscript):
            ndim = self._get_subscripts_ndim(node)
            if ndim > self.ndim:
                self.ndim = ndim

        for child in ast.iter_child_nodes(node):
            self._scan_subscripts(child)

    def _get_subscripts_ndim(self, subscript_node):
        slice_node = subscript_node.slice

        if isinstance(slice_node, ast.Tuple):
            # e.g. [:, None] -> Tuple(elts=[Slice(), Constant(None)])
            return len(slice_node.elts)
        elif isinstance(slice_node, (ast.Slice, ast.Constant, ast.Name, ast.UnaryOp, ast.BinOp)):
            # e.g. [0], [:], [i], [-1], [i+1]
            return 1
        else:
            # Fallback: treat as 1D
            return 1

    def visit_Call(self, node):
        if not isinstance(node.func, ast.Attribute):
            return
        func = node.func
        if not isinstance(func.value, ast.Name) or func.value.id != 'tl':
            self.generic_visit(node)
            return
        if func.attr not in self.reduction_func:
            return

        axis_dim = None
        args = node.args
        if len(args) == 1:
            # Axis passed as keyword argument
            for keyword in node.keywords:
                if keyword.arg == 'axis':
                    axis_dim = self.get_axis_dim(keyword.value)
                    break

        elif len(args) == 2:
            # Axis passed as positional argument. Check the second param
            axis_dim = self.get_axis_dim(args[1])

        else:
            raise ValueError("Reduction funtions args error")

        if axis_dim is not None:
            reduction_axis = self.get_axis(axis_dim)
            if reduction_axis and reduction_axis not in self.reduction_axes:
                self.reduction_axes.append(reduction_axis)

    def get_axis_dim(self, node):
        if isinstance(node, ast.Constant):
            axis_dim = node.value
        elif isinstance(node, ast.UnaryOp) and \
            isinstance(node.op, ast.USub):
            operand = node.operand
            if isinstance(operand, ast.Constant):
                axis_dim = self.ndim - operand.value
        else:
            raise ValueError(f"Reduction function axis error, got: {ast.dump(node)}")

        if not isinstance(axis_dim, int):
            raise ValueError("Reduction function axis must be an integer, "
                             f"got {type(node.value).__name__}: {node.value}")
        return axis_dim

    def get_axis(self, axis_dim: int):
        """
        Override the parent class method to accept an integer axis dimension
        instead of a string.

        :param axis_dim:
        :type axis_dim: int
        """
        if axis_dim in self.checked_vars:
            return None
        self.checked_vars.append(axis_dim)
        for node in ast.walk(self.func_ast):
            if not isinstance(node, ast.Assign):
                continue
            reduction_axis = self.handle_assign_node(axis_dim, node)
            if reduction_axis:
                return reduction_axis
        return None

    def handle_assign_node(self, axis_dim: int, node):
        if not isinstance(node.value, ast.Compare):
            return None

        # only support less than
        if len(node.value.ops) != 1 or not isinstance(node.value.ops[0], ast.Lt):
            return None

        target_axis_len = None
        for axis_len in self.keys.values():
            if self.contains_target_var(node.value, axis_len):
                target_axis_len = axis_len
                break
        if not target_axis_len:
            return None

        # handel compare left var
        if isinstance(node.value.left, ast.Name):
            if self.check_compare_left(node.value.left.id, axis_dim):
                reduction_axis = next((k for k, v in self.keys.items() if target_axis_len == v), None)
                if reduction_axis and reduction_axis not in self.reduction_axes:
                    return reduction_axis
        # handel compare target var
        if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            if self.check_compare_target(node.targets[0].id, axis_dim):
                reduction_axis = next((k for k, v in self.keys.items() if target_axis_len == v), None)
                if reduction_axis and reduction_axis not in self.reduction_axes:
                    return reduction_axis
        return None

    def check_compare_left(self, var, axis_dim):
        for node in ast.walk(self.func_ast):
            if not isinstance(node, ast.Assign):
                continue
            if len(node.targets) != 1 or \
               not isinstance(node.targets[0], ast.Name) or \
               node.targets[0].id != var:
                continue
            if self.is_current_dim_slice(node.value, axis_dim):
                return True
        return False

    def check_compare_target(self, var, axis_dim, node=None):
        if not node:
            node = self.func_ast
        for _, value in ast.iter_fields(node):
            if isinstance(value, list):
                for item in value:
                    if self.check_compare_target(var, axis_dim, item):
                        return True
            elif isinstance(value, ast.AST):
                if isinstance(value, ast.Subscript):
                    if not isinstance(value.value, ast.Name) or value.value.id != var:
                        continue
                    if self.is_current_dim_slice(value, axis_dim):
                        return True
                else:
                    if self.check_compare_target(var, axis_dim, value):
                        return True
        return False

    def is_current_dim_slice(self, node, dim):
        for node in ast.walk(node):
            if not isinstance(node, ast.Subscript) or not isinstance(node.slice, ast.Tuple):
                continue
            elts = node.slice.elts
            if len(elts) != 0 and isinstance(elts[dim], ast.Slice):
                return True
        return False


class LowDimsAxesParser(AxesKeyParser):
    """
    Extracts the low dimensions axis from triton kernel code. The parsing is based on the
    `tl.arange` statement. This class identifies low dimensions axis by analyzing the usage
    of the `tl.arange` in the program and extracts the variables computed by `tl.arange` and
    their associated operations. Then it checks if these variables are involved in slicing
    operations to determine dimension expansion and filters out variables that are expanded
    in non-lowest dimensions. After that, it compares the extracted variables with the provided
    `keys` to map them to specific low-dimensional axis.

    Note:
    1. low dimensions axis must be calculated within the `tl.arange` function and involved in
       slicing operations to be identified.
    2. Without mask comparision, it is impossible to confirm the exact low dimensions axis, which
       would lead to parameter parsing failure. (eg. mask = offsets < n_elements).
    """

    def __init__(self, func_ast: ast.AST, keys: Dict[str, str]):
        """
        :param func_ast: Abstract syntax tree of the triton kernel function
        :type func_ast: ast.AST
        :param keys: a dict of axis name: argument name, used to confirm the low-dimensional axis.
        :type keys: Dict[str, str]
        """
        super().__init__(func_ast, keys)
        self.low_dims_axis = list()
        self.keys = keys
        self.checked_slice_vars = list()

    def parse(self) -> List[str]:
        super().parse()
        return self.low_dims_axis

    def visit_Assign(self, node):
        if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            tl_arange_node = self.get_tl_arange_node(node)
            low_dims_axis = None
            if isinstance(tl_arange_node, ast.Call):
                partin_other_slice = [False]
                if self.is_partin_low_dim_slice(node.targets[0].id, partin_other_slice):
                    low_dims_axis = self.get_axis(node.targets[0].id)
                elif not partin_other_slice[0]:
                    low_dims_axis = self.get_axis(node.targets[0].id)
            elif isinstance(tl_arange_node, ast.Subscript) and \
                 self.is_low_dim_slice(tl_arange_node, [False]):
                low_dims_axis = self.get_axis(node.targets[0].id)

            if low_dims_axis and low_dims_axis not in self.low_dims_axis:
                self.low_dims_axis.append(low_dims_axis)
        self.generic_visit(node)

    def get_tl_arange_node(self, node):
        for _, value in ast.iter_fields(node):
            if isinstance(value, list):
                for item in value:
                    if self.is_tl_arange_call(item):
                        return item
                    node = self.get_tl_arange_node(item)
                    if node:
                        return node
            elif isinstance(value, ast.AST):
                if self.is_tl_arange_call(value):
                    return value
                node = self.get_tl_arange_node(value)
                if node:
                    return node
        return None

    def is_tl_arange_call(self, node):
        """
        Checks if the given AST node is a call to `tl.arange` or a subscript of `tl.arange`.
        It supports direct calls to `tl.arange` and subscripts of `tl.arange`, such as
        `tl.arange()[None, :]`
        """
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr == 'arange' and \
               isinstance(node.func.value, ast.Name) and \
               node.func.value.id == 'tl':
                return True
        elif isinstance(node, ast.Subscript):
            return self.is_tl_arange_call(node.value)
        return False

    def is_low_dim_slice(self, node: ast.Subscript, partin_other_slice):
        if not isinstance(node.slice, ast.Tuple) or not isinstance(node.slice.elts, list):
            return False
        elts = node.slice.elts
        if len(elts) != 0 and not isinstance(elts[-1], ast.Slice):
            partin_other_slice[0] = True
            return False
        return True

    def is_partin_low_dim_slice(self, var, partin_other_slice, node=None):
        if not node:
            node = self.func_ast
        for child_node in ast.walk(node):
            if isinstance(child_node, ast.Subscript) and isinstance(child_node.value, ast.Name):
                if var == child_node.value.id and self.is_low_dim_slice(child_node, partin_other_slice):
                    return True
            elif isinstance(child_node, ast.Assign):
                if len(child_node.targets) == 1 and \
                   isinstance(child_node.targets[0], ast.Name) and \
                   var != child_node.targets[0].id: # Prevent cyclic assignment.
                    if not self.contains_target_var(child_node.value, var):
                        continue
                    target_var = child_node.targets[0].id
                    if target_var in self.checked_slice_vars:
                        continue

                    if self.is_partin_low_dim_slice(var, partin_other_slice, child_node.value):
                        return True
                    if self.is_partin_low_dim_slice(target_var, partin_other_slice):
                        return True

        self.checked_slice_vars.append(var)
        return False


class PtrNumsParser(AutoParser):
    """
    Counts the number of pointer parameters from triton kernel code. The parsing of pointer-type
    parameters is determined based on whether these parameters participate in memory access
    statements such as `tl.load` and `tl.store`.
    First, all input parameters in the kernel function are parsed, and then recursively, all variables
    involved in the computation of each input parameter are identified.
    If an input parameter directly participates in the computation of the first argument of `tl.load`
    or `tl.store`, or if an intermediate variable computed from this input parameter indirectly
    participates in the computation of the first argument of `tl.load` or `tl.store`, then this
    parameter is considered a pointer-type parameter.

    Note:
    1. Variables modified with `tl.constexpr` are not pointer-type variables and will not be
       further parsed.
    2. Only memory access statementes where the input parameter is directly involved or indirectly
       involved through one level of computation are counted. Intermediate variables computed from
       the input parameter through two or more levels of computation are not counted.
    """

    def __init__(self, func_ast: ast.AST, keys: Dict[str, str], miss_params: List[str]):
        """
        :param func_ast: Abstract syntax tree of the triton kernel function
        :type func_ast: ast.AST
        :param keys: a dict of axis name: argument name, used to exclude potential ptr params.
        :type keys: Dict[str, str]
        :param miss_params: a list of parameters names that were not provided by the user when calling triton
            kernel function.
        :type miss_params: List[str]
        """
        super().__init__(func_ast)
        self.checked_vars = list()
        self.ptr_nums = 0
        self.ptr_params = list()
        self.keys = keys
        self.miss_params = miss_params
        self.constexpr_params = list()

    def parse(self):
        super().parse()
        return self.ptr_nums, self.ptr_params

    def visit_FunctionDef(self, node):
        if isinstance(node.args, ast.arguments):
            for arg in node.args.args:
                if not isinstance(arg, ast.arg):
                    continue

                if isinstance(arg.annotation, ast.Attribute):
                    # var modified by tl.constexpr are not pointer type var, passed
                    is_tl = isinstance(arg.annotation.value, ast.Name) and \
                            arg.annotation.value.id == 'tl'
                    if is_tl and arg.annotation.attr == 'constexpr':
                        if arg.arg not in self.constexpr_params:
                            self.constexpr_params.append(arg.arg)
                        continue

                if self.is_in_addr_calc(arg.arg) and arg.arg not in self.keys.values():
                    self.ptr_params.append(arg.arg)
                    self.ptr_nums += 1

        for miss_param in self.miss_params:
            if miss_param not in self.constexpr_params:
                print(f"[WARNING] The parameter '{miss_param}' needs to be declared as tl.constexpr!")
        self.generic_visit(node)

    def is_in_addr_calc(self, var):
        for node in ast.walk(self.func_ast):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Attribute) and \
                   isinstance(node.func.value, ast.Name):
                    if node.func.value.id == "tl" and \
                       (node.func.attr == "load" or node.func.attr == "store"):
                        if [arg for arg in node.args if self.contains_target_var(arg, var)]:
                            return True

            elif isinstance(node, ast.Assign):
                if len(node.targets) == 1 and \
                   isinstance(node.targets[0], ast.Name) and \
                   var != node.targets[0].id: # Prevent cyclic assignment.
                    target_var = node.targets[0].id
                    if target_var in self.checked_vars:
                        continue
                    if isinstance(node.value, ast.BinOp) and \
                       isinstance(node.value.op, ast.Add):
                        if isinstance(node.value.left, ast.Name) and \
                           node.value.left.id == var:
                            if self.is_in_addr_calc(node.targets[0].id):
                                return True
                        elif isinstance(node.value.right, ast.Name) and \
                             node.value.right.id == var:
                            if self.is_in_addr_calc(node.targets[0].id):
                                return True
        self.checked_vars.append(var)
        return False
