# should store at third_party/aipu/backend/
from triton.compiler.hint_manager import BaseHintHandler
import triton.language as language
import ast
from triton.compiler.code_generator import _is_triton_value

class AipuHintHandler(BaseHintHandler):


    @staticmethod
    def func1(code_generator, node, names, values):
        line_num = node.lineno
        function_def = self.jit_fn.parse()
        line_flagtree_hints = getattr(function_def.body[0], 'line_flagtree_hints', {})
        flagtree_hints = line_flagtree_hints.get(line_num)

    def func2():
        if fn.__name__ == "load" and flagtree_hints is not None:
                    print(f"[FLAGTREE] tl.load at line {line_num} has annotation {flagtree_hints}")
                    if 'flagtree_hints' not in kws:
                        kws['flagtree_hints'] = ""
                    if flagtree_hints not in kws['flagtree_hints']:
                        kws['flagtree_hints'] = flagtree_hints

    @staticmethod
    def ext_CodeGenerator_visit_Assign_hint_anno(code_generator, node, names, values):
        if not (hasattr(node, 'lineno') and hasattr(code_generator, 'jit_fn')):
            return

        if not flagtree_hints:
            return

        # 3. AIPU 特有的 Hint 处理逻辑
        # [请对照 PR 修改此处字符串] 假设 AIPU 有一个 hint 叫 'global_memory_cache' 或者类似的
        target_hint_key = 'aipu_specific_hint_key' # <--- 请替换为 PR 里的真实字符串
        
        if target_hint_key in flagtree_hints:
            # 检查是否作用于 tl.load / tl.store
            if (isinstance(node.value, ast.Call) and
                isinstance(node.value.func, ast.Attribute) and
                isinstance(node.value.func.value, ast.Name) and
                node.value.func.value.id == 'tl'):
                
                # 例如：如果是 load 操作
                if node.value.func.attr == 'load':
                    for name, value in zip(names, values):
                        if _is_triton_value(value):
                            # 创建 AIPU 特有的 Annotation
                            # print(f"[FLAGTREE][AIPU] Applying {target_hint_key} to {name}")
                            hint_val = code_generator.builder.get_unit_attr()
                            # 'aipu.hint_name' 需要与后端 LLVM IR 处理逻辑对应
                            code_generator.builder.create_annotation(value.handle, target_hint_key, hint_val)

    @staticmethod
    def check_override_bind_sub_block(code_generator, node, bind_sub_block):
        """
        对应 CodeGenerator.visit_For 中决定是否开启 bind_sub_block 的逻辑
        """
        if not (hasattr(node, 'lineno') and hasattr(code_generator, 'jit_fn')):
            return bind_sub_block

        line_num = node.lineno
        function_def = code_generator.jit_fn.parse()
        line_flagtree_hints = getattr(function_def.body[0], 'line_flagtree_hints', {})
        flagtree_hints = line_flagtree_hints.get(line_num)

        # 检查 AIPU 是否也支持通过 hint 强制开启/关闭 sub_block 绑定
        if flagtree_hints and 'bind_sub_block' in flagtree_hints:
             # 如果 AIPU 后端支持此特性，则返回 True
            return True
        
        return bind_sub_block

    @staticmethod
    def maps_line_numbers_to_comment_hints(jit_fn):
        # Maps line numbers to comment hints
        line_flagtree_hints = {}
        code_str = self.src
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