def enable_stream_in_kwargs(kwargs):
    return True

def ignore_params_in_JITFunction_run(kwargs, excess_kwargs):
    ignor_params = ["debug", "sanitize_overflow", "llvm_version", "kernel_name", \
                "allowed_dot_input_precisions", "multibuffer", "stream"]
    not_work_params = []
    for k in kwargs:
        if k in ignor_params:
            continue
        elif k in excess_kwargs:
            not_work_params.append(k)
    if len(not_work_params) != 0:
        print("[WARNING] Please DO NOT tune args {}!".format(not_work_params))

def check_grid_size(grid_0, grid_1, grid_2):
    import os
    grid_all_size = grid_0 * grid_1 * grid_2
    if os.getenv("TRITON_ALL_BLOCKS_PARALLEL", "0") == "0":
        if grid_all_size > 65535:
            raise RuntimeError("grid should be less than 65536! You can try \"export TRITON_ALL_BLOCKS_PARALLEL=1\" to avoid this problem.")

def explicit_load_kernel_library(kernel):
    # explicitly define run method and load kernel binary
    kernel._init_handles()

def get_JITFunction_spec_attr(deserialized_obj):
    from triton.backends.ascend.compiler import AscendAttrsDescriptor
    return AscendAttrsDescriptor.from_dict(deserialized_obj['attrs'])

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

def attach_line_number_to_comment_mapping(tree, line_flagtree_hints):
    # Attach the line number to comment mapping to the function definition node
    tree.body[0].line_flagtree_hints = line_flagtree_hints

def enable_extra_option():
    return True
