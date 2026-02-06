# should store at third_party/aipu/backend/
from triton.compiler.hint_manager import BaseHintHandler
import triton.language as language
import ast
from triton.compiler.code_generator import _is_triton_value


class AipuHintHandler(BaseHintHandler):
    # because aipu is diff from ascend in 2 aspects
    # 1. not backend_spec, modify triton src violently
    # 2. modify builder, semantic, core, and so on. pollute the src, which cant be involved in hint manager
    # for this, we just move changes in codegen & jit into hintmanager.

    @staticmethod
    def get_node_hints(code_generator, node):
        line_num = node.lineno
        function_def = code_generator.jit_fn.parse()
        line_flagtree_hints = getattr(function_def.body[0], 'line_flagtree_hints', {})
        flagtree_hints = line_flagtree_hints.get(line_num)
        return flagtree_hints

    @staticmethod
    def inject_kwargs_with_hints(fn, flagtree_hints, line_num, kws):
        if fn.__name__ == "load" and flagtree_hints is not None:
            print(f"[FLAGTREE] tl.load at line {line_num} has annotation {flagtree_hints}")
            if 'flagtree_hints' not in kws:
                kws['flagtree_hints'] = ""
            if flagtree_hints not in kws['flagtree_hints']:
                kws['flagtree_hints'] = flagtree_hints

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

        return line_flagtree_hints

    @staticmethod
    def attach_line_number_to_comment_mapping(tree, line_flagtree_hints):
        if tree.body:
            tree.body[0].line_flagtree_hints = line_flagtree_hints
