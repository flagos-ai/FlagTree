# should store at thrid_party/???/backend/
from triton.compiler.hint_manager import BaseHintHandler
import triton.language as language 
import ast
from triton.compiler.code_generator import _is_triton_value

class AscendHintHandler(BaseHintHandler):

    @staticmethod
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

    @staticmethod
    def check_override_bind_sub_block(code_generator, node, bind_sub_block):
        # flagtree: After normal processing, check if we need to override bind_sub_block
        if hasattr(node, 'lineno') and hasattr(code_generator, 'jit_fn'):
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

    @staticmethod
    def forop_setattr_for_bind_sub_block(code_generator, for_op, bind_sub_block):
        for_op.set_attr("bind_sub_block", code_generator.builder.get_bool_attr(bind_sub_block))


    @staticmethod
    def maps_line_numbers_to_comment_hints(jit_fn):
    import tokenize
    from io import StringIO
    # Maps line numbers to comment hints
    line_flagtree_hints = {}
    code_str = jit_fn.src
    g = tokenize.generate_tokens(StringIO(code_str).readline)
    for tok_type, tok_text, start, end, _ in g:
        if tok_type == tokenize.COMMENT:
            comment = tok_text.replace(" ", "").strip()
            if comment.startswith('#@hint:'):
                flagtree_hints = comment[len('#@hint:'):].strip()
                # Record the line number of the comment
                line_num = start[0]
                line_flagtree_hints[line_num] = flagtree_hints

                # print(f"[FLAGTREE] Parsed hint at line {line_num}: {flagtree_hints}")

    return line_flagtree_hints

    @staticmethod
    def attach_line_number_to_comment_mapping(tree, line_flagtree_hints):
    # Attach the line number to comment mapping to the function definition node
    tree.body[0].line_flagtree_hints = line_flagtree_hints