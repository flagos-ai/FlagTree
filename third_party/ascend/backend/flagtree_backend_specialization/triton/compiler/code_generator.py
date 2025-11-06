def anno_CodeGenerator_visit_Assign():
    # flagtree: First, do normal assignment processing
    return

def ext_CodeGenerator_visit_Assign_hint_anno(code_generator, node, names, values):
    import ast
    from triton.compiler.code_generator import _is_triton_value
    # flagtree: After normal processing, check if we need to add hint annotation
    if hasattr(node, 'lineno') and hasattr(code_generator, 'jit_fn'):
        line_num = node.lineno
        # TODO: reparse needed in case we need to deal with complex cases, will be redesigned later
        function_def = code_generator.jit_fn.parse()
        line_flagtree_hints = getattr(function_def.body[0], 'line_flagtree_hints', {})
        flagtree_hints = line_flagtree_hints.get(line_num)

        # Check if this is a tl.load call with dot_pad_only_k hint
        if (flagtree_hints and 'dot_pad_only_k' in flagtree_hints and
            isinstance(node.value, ast.Call) and
            isinstance(node.value.func, ast.Attribute) and
            isinstance(node.value.func.value, ast.Name) and
            node.value.func.value.id == 'tl' and
            node.value.func.attr == 'load'):

            # Add hint annotation to the loaded tensor(s)
            for name, value in zip(names, values):
                if _is_triton_value(value):
                    # print(f"[FLAGTREE] Creating hint annotation for tensor: {flagtree_hints}")
                    # Create hint annotation
                    hint_val = code_generator.builder.get_unit_attr()
                    code_generator.builder.create_annotation(value.handle, 'dot_pad_only_k', hint_val)

def init_bind_sub_block():
    return None

def is_visit_For_support_parallel():
    return True

def set_bind_sub_block_when_parallel(IteratorClass, iterator, bind_sub_block):
    import triton.language as language
    if (IteratorClass is language.parallel):
        return iterator.bind_sub_block
    return bind_sub_block

def check_override_bind_sub_block(code_generator, node, bind_sub_block):
    # flagtree: After normal processing, check if we need to override bind_sub_block
    if hasattr(node, 'lineno') and hasattr(self, 'jit_fn'):
        line_num = node.lineno
        # TODO: reparse needed in case we need to deal with complex cases, will be redesigned later
        function_def = code_generator.jit_fn.parse()
        line_flagtree_hints = getattr(function_def.body[0], 'line_flagtree_hints', {})
        flagtree_hints = line_flagtree_hints.get(line_num)

        # Check if this is a range/for loop with bind_sub_block hint
        if flagtree_hints and 'bind_sub_block' in flagtree_hints:
            return True
            # print(f"[FLAGTREE] Found bind_sub_block hint at line {line_num}")
    return bind_sub_block

def forop_setattr_for_bind_sub_block(code_generator, for_op, bind_sub_block):
    for_op.set_attr("bind_sub_block", code_generator.builder.get_bool_attr(bind_sub_block))

def need_repr_in_CodeGenerator_CompilationError():
    return True
