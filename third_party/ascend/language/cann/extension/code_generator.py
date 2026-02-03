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
"""
Ascend-specific code generation handlers for 'with' statement context managers.
"""

__all__ = ["handle_scope_with", "mangle_ty"]
import ast


def mangle_ty(ty):
    """
    Replacement implementation for triton.compiler.code_generator.mangle_ty.

    This is registered via ASCEND_WITH_DISPATCH["mangle_ty"] and picked up by
    triton.compiler.code_generator through its global WITH_DISPATCH table.
    """
    # Lazy imports to avoid circular dependencies at module import time.
    from triton import language
    from triton.extension.buffer.language import core as bl

    # Buffer types are Python-side dtypes; handle them first.
    if isinstance(ty, bl.buffer_type):
        elt = mangle_ty(ty.element_ty)
        shape = "_".join(map(str, ty.shape))
        return f"B{elt}S{shape}S"

    if ty.is_ptr():
        return "P" + mangle_ty(ty.element_ty)
    if ty.is_int():
        SIGNED = language.dtype.SIGNEDNESS.SIGNED
        prefix = "i" if ty.int_signedness == SIGNED else "u"
        return prefix + str(ty.int_bitwidth)
    if ty.is_floating():
        return str(ty)
    if ty.is_block():
        elt = mangle_ty(ty.scalar)
        shape = "_".join(map(str, ty.shape))
        return f"{elt}S{shape}S"
    if ty.is_void():
        return "V"
    raise TypeError(f"Unsupported type {ty}")


def _extract_scope_attributes(context_expr):
    """Extract attributes from scope(...) call."""
    scope_attrs = {}
    for keyword in context_expr.keywords:
        if isinstance(keyword.value, ast.Constant):
            scope_attrs[keyword.arg] = keyword.value.value
    return scope_attrs


def _py_value_to_mlir_attr(builder, value):
    """Convert Python value to MLIR attribute."""
    attr_creators = {
        str: lambda v: builder.get_str_attr(v),
        bool: lambda v: builder.get_bool_attr(v),
        int: lambda v: builder.get_int32_attr(v),
        list: lambda v: builder.get_i64_array_attr(v),
    }
    creator = attr_creators.get(type(value))
    return creator(value) if creator else value


def _handle_core_mode_attr(builder, core_mode):
    """Handle core_mode attribute conversion."""
    if core_mode not in ("cube", "vector"):
        return {}
    return {
        builder.get_t_core_type_attr_name():
        (builder.get_t_core_type_cube_attr() if core_mode == "cube" else builder.get_t_core_type_vector_attr())
    }


def _build_mlir_attrs_from_scope_attrs(builder, scope_attrs):
    """Convert Python scope attributes to MLIR attributes.

    Args:
        builder: The IR builder
        scope_attrs: Dict of scope attributes (e.g., {'core_mode': 'vector', 'noinline': True})

    Returns:
        Dict of MLIR attributes
    """
    mlir_attrs = {"noinline": builder.get_unit_attr()}
    for k, v in scope_attrs.items():
        if k == "core_mode":
            mlir_attrs.update(_handle_core_mode_attr(builder, v))
        elif k == "noinline":
            if not v:
                mlir_attrs.pop("noinline")
        elif k == "disable_auto_sync":
            if v:
                mlir_attrs["hivm.disable_auto_sync"] = _py_value_to_mlir_attr(builder, v)
        else:
            mlir_attrs[k] = _py_value_to_mlir_attr(builder, v)
    return mlir_attrs


def _verify_loop_carried_variable(_is_triton_value, _is_triton_tensor, name, loop_val, live_val):
    """Verify that loop-carried variable types are consistent."""
    assert _is_triton_value(loop_val), f'cannot reassign constxpr {name} in the loop'
    assert _is_triton_value(live_val), f'cannot reasign constexpr {name} in the loop'
    assert type(loop_val) == type(live_val), f'Loop carried variable {name} changed type'
    assert not _is_triton_tensor(loop_val) or loop_val.type == live_val.type, \
        f'Loop-carried variable {name} has initial type {live_val.type} '\
        f'but is re-assigned to {loop_val.type} in loop! '\
        f'Please make sure that the type stays consistent.'


def _reconstruct_value_from_ir(language, entry_block_arg, ret_type):
    """Reconstruct a tensor value from IR."""
    return language.core.tensor(entry_block_arg, ret_type)


def handle_scope_with(generator, node):
    """
    Handle 'with scope(...)' statements by creating a scope.scope operation.

    This creates a scope.scope operation with a region for the scope block.
    Uses SSA threading to properly handle variables modified inside the scope.

    Args:
        generator: The CodeGenerator instance
        node: AST node for the with statement
    """
    # Lazy imports to avoid circular dependency
    from triton import language
    from triton.compiler.code_generator import enter_sub_region, _is_triton_value, _is_triton_tensor

    context_expr = node.items[0].context_expr
    scope_attrs = _extract_scope_attributes(context_expr)

    with enter_sub_region(generator) as sr:
        liveins, _ = sr
        ip, last_loc = generator._get_insertion_point_and_loc()

        # This implementation is similar to visit_while
        dummy = generator.builder.create_block()
        generator.builder.set_insertion_point_to_start(dummy)
        generator.visit_compound_statement(node.body)
        scope_defs = generator.local_defs
        dummy.erase()

        # Verify and get return type of the scope.scope
        # (variables that exist in parent scope AND are modified in scope)
        names = []
        ret_types = []
        for name in scope_defs:
            scope_val = scope_defs[name]
            ret_types.append(scope_val.type)
            names.append(name)
            if name in liveins:
                live_val = liveins[name]
                _verify_loop_carried_variable(_is_triton_value, _is_triton_tensor, name, scope_val, live_val)

        # Convert Python primitives to MLIR attributes
        mlir_attrs = _build_mlir_attrs_from_scope_attrs(generator.builder, scope_attrs)

        # Create scope operation with operands (values from outside)
        generator._set_insertion_point_and_loc(ip, last_loc)
        scope_op = generator.builder.create_scope_op(mlir_attrs, [ty.to_ir(generator.builder) for ty in ret_types])

        # Create the entry block with arguments matching the operands
        entry_block = generator.builder.create_block_with_parent(scope_op.get_region(0), [])
        generator.builder.set_insertion_point_to_start(entry_block)

        # Initialize the scope's symbol table with liveins
        generator.lscope = liveins.copy()
        generator.visit_compound_statement(node.body)
        generator.builder.set_insertion_point_to_end(entry_block)
        reconstructed_values = []

        for i in range(len(names)):
            # generator.lscope[names[i]] is already a tensor, just get its IR handle
            reconstructed_values.append(generator.lscope[names[i]].handle)
        generator.builder.scope_return(reconstructed_values)

    # After exiting enter_sub_region, update symbol table with results
    # Convert IR values back to tensor objects
    for i, name in enumerate(names):
        generator.set_value(name, _reconstruct_value_from_ir(language, scope_op.get_result(i), ret_types[i]))
    return None
