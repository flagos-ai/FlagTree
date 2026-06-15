# should store at third_party/nvidia/backend/
import functools
import tokenize
from io import StringIO

from triton.compiler.hint_manager import BaseHintHandler


class NvidiaHintHandler(BaseHintHandler):

    # TODO : below can be reused by other backend which need to implement "hint";
    @staticmethod
    def get_node_hints(code_generator, node):
        line_flagtree_hints = getattr(code_generator, 'flagtree_line_hints', {})
        return line_flagtree_hints.get(node.lineno)

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
        return dict(NvidiaHintHandler._maps_line_numbers_to_comment_hints_from_source(jit_fn.src))

    @staticmethod
    @functools.lru_cache(maxsize=256)
    def _maps_line_numbers_to_comment_hints_from_source(code_str):
        # Maps line numbers to comment hints
        line_flagtree_hints = {}
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
