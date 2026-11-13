"""Read supported Mojo A5 pytest specializations without running tests.

Shared helpers handle AST reading, argument types and specialization validation.
The four family readers retain their wrapper-specific shape and mask rules;
this is not a general Python interpreter or a reader for arbitrary kernels.
"""

import ast
import math
import operator
from pathlib import Path

_BINARY_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.FloorDiv: operator.floordiv,
}


def _static_value(node, values=None):
    """Evaluate only simple expressions used by the SWA test and launch code."""
    values = values or {}
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name) and node.id in values:
        return values[node.id]
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
        key = f"{node.value.id}.{node.attr}"
        if key in values:
            return values[key]
        if node.value.id == "torch" and node.attr in {"bfloat16", "float16", "float32"}:
            return key
    if isinstance(node, (ast.List, ast.Tuple)):
        return [_static_value(item, values) for item in node.elts]
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY_OPS:
        return _BINARY_OPS[type(node.op)](_static_value(node.left, values), _static_value(node.right, values))
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -_static_value(node.operand, values)
    if isinstance(node, ast.Compare) and len(node.ops) == len(node.comparators) == 1:
        if isinstance(node.ops[0], ast.Eq):
            return _static_value(node.left, values) == _static_value(node.comparators[0], values)
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        functions = {"max": max, "min": min}
        if node.func.id in functions and not node.keywords:
            return functions[node.func.id](*[_static_value(arg, values) for arg in node.args])
    raise ValueError(f"cannot statically read expression: {ast.unparse(node)}")


def _parse(path):
    return ast.parse(Path(path).read_text(encoding="utf-8"), filename=str(path))


def _definition(tree, name, kind):
    for node in tree.body:
        if isinstance(node, kind) and node.name == name:
            return node
    raise ValueError(f"{name} not found in source")


def _assignment(tree, name):
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id == name for target in node.targets):
            return node.value
    raise ValueError(f"{name} not found in source")


def _call(function, name):
    for node in ast.walk(function):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == name:
            return node
    raise ValueError(f"call to {name} not found in source")


def _launch(function, kernel_name):
    for node in ast.walk(function):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Subscript)
                and isinstance(node.func.value, ast.Name) and node.func.value.id == kernel_name):
            return node
    raise ValueError(f"launch of {kernel_name} not found in source")


def _gqa_interleave(root, layout):
    operator_tree = _parse(root / "mojo_opset/core/operators/attention.py")
    mojo_swa = _definition(operator_tree, "MojoSWA", ast.ClassDef)
    constructor = next(node for node in mojo_swa.body if isinstance(node, ast.FunctionDef) and node.name == "__init__")
    for node in ast.walk(constructor):
        if isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Attribute) and target.attr == "gqa_interleave" for target in node.targets):
            return _static_value(node.value, {"gqa_layout": layout})
    raise ValueError("MojoSWA.gqa_interleave assignment not found")


def _block_sizes(wrapper, dtype):
    for node in wrapper.body:
        if isinstance(node, ast.If):
            try:
                selected = node.body if _static_value(node.test, {"q.dtype": dtype}) else node.orelse
            except ValueError:
                continue
            sizes = {}
            for statement in selected:
                if isinstance(statement, ast.Assign) and len(statement.targets) == 1:
                    target = statement.targets[0]
                    if isinstance(target, ast.Name) and target.id in {"BLOCK_M", "BLOCK_N"}:
                        sizes[target.id] = _static_value(statement.value)
            if set(sizes) == {"BLOCK_M", "BLOCK_N"}:
                return sizes["BLOCK_M"], sizes["BLOCK_N"]
    raise ValueError("BLOCK_M/BLOCK_N selection not found in SWA wrapper")


def _block_d(wrapper, head_dim):
    for node in wrapper.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name) and target.id == "BLOCK_D":
                return _static_value(node.value, {"head_dim": head_dim})
    raise ValueError("BLOCK_D assignment not found in SWA wrapper")


def _mask_shape(kernel_tree, wrapper, global_window, local_window, block_values=None):
    mask_call = _call(wrapper, "get_mask_causal_with_window")
    if len(mask_call.args) < 2:
        raise ValueError("SWA mask block dimensions not found")
    block_m, block_n = (_static_value(node, block_values) for node in mask_call.args[:2])
    aux_size = _static_value(_assignment(kernel_tree, "AUX_MASK_SIZE"))
    try:
        mask_function = _definition(kernel_tree, "get_mask_causal_with_window", ast.FunctionDef)
        if not any(
                isinstance(node, ast.Assign) and any(
                    isinstance(target, ast.Name) and target.id == "M_boundary"
                    for target in node.targets)
                for node in mask_function.body):
            raise ValueError("mask calculation is delegated")
    except ValueError:
        mask_function = _definition(kernel_tree, "_get_mask_causal_with_window_cached", ast.FunctionDef)
    values = {
        "BLOCK_M": block_m, "BLOCK_N": block_n, "local_window_size": local_window, "global_window_size": global_window,
        "AUX_MASK_SIZE": aux_size
    }
    for node in mask_function.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name) and target.id in {"M", "N", "M_boundary", "N_boundary"}:
                values[target.id] = _static_value(node.value, values)
    if "M_boundary" not in values or "N_boundary" not in values:
        raise ValueError("SWA mask shape calculation not found")
    return values["M_boundary"], values["N_boundary"]


def _test_row(test_tree, assignment_name, config_id):
    rows = _static_value(_assignment(test_tree, assignment_name))
    matching = [row for row in rows if row[-1] == config_id]
    if len(matching) != 1:
        raise ValueError(f"expected one {assignment_name} row with id {config_id!r}")
    return matching[0]


def _master_test_row(test_tree, assignment_name, config_id):
    """Read a named pytest.param from the refactored master test lists."""
    value = _assignment(test_tree, assignment_name)
    if isinstance(value, ast.ListComp):
        value = value.generators[0].iter
    if not isinstance(value, (ast.List, ast.Tuple)):
        raise ValueError(f"unsupported {assignment_name} parametrization")
    for entry in value.elts:
        if isinstance(entry, ast.Call) and isinstance(entry.func, ast.Attribute) and entry.func.attr == "param":
            name = next((_static_value(kw.value) for kw in entry.keywords if kw.arg == "id"), None)
            if name == config_id:
                return [_static_value(arg) for arg in entry.args] + [config_id]
        elif isinstance(entry, ast.Tuple):
            row = _static_value(entry)
            if row[-1] == config_id:
                return row
    raise ValueError(f"{assignment_name} row {config_id!r} not found")


def _master_test_window(test_tree, test_name, layout):
    if test_name == "test_swa_infer":
        rows = _static_value(_assignment(test_tree, "SWA_INFER_WINDOWS"))
        return next((row[1], row[2]) for row in rows if row[0] == layout)
    if test_name == "test_swa":
        function = _definition(test_tree, test_name, ast.FunctionDef)
        for decorator in function.decorator_list:
            if isinstance(decorator, ast.Call) and len(decorator.args) >= 2:
                try:
                    if _static_value(decorator.args[0]) == "interleave,global_window,local_window":
                        rows = _static_value(decorator.args[1])
                        return next((row[1], row[2]) for row in rows if row[0] == (layout == "ABAB"))
                except ValueError:
                    continue
    if test_name == "test_paged":
        cases = _assignment(test_tree, "PAGED_CASES")
        for node in ast.walk(cases):
            if not (isinstance(node, ast.ListComp) and isinstance(node.elt, ast.Tuple) and len(node.elt.elts) >= 6
                    and isinstance(node.elt.elts[5], ast.Name) and node.elt.elts[5].id == "window"):
                continue
            for generator in node.generators:
                if isinstance(generator.target, ast.Tuple):
                    names = [item.id for item in generator.target.elts if isinstance(item, ast.Name)]
                    if names == ["layout", "window"]:
                        rows = _static_value(generator.iter)
                        return (_static_value(node.elt.elts[4]), next(row[1] for row in rows if row[0] == layout))
    raise ValueError(f"master {test_name} window for {layout!r} not found")


def _test_window(test_tree, test_name, layout):
    function = _definition(test_tree, test_name, ast.FunctionDef)
    for decorator in function.decorator_list:
        if not (isinstance(decorator, ast.Call) and len(decorator.args) >= 2
                and isinstance(decorator.args[0], ast.Constant)):
            continue
        names = decorator.args[0].value
        if names not in {"gqa_layout, global_window, local_window", "gqa_interleave, global_window, local_window"}:
            continue
        rows = _static_value(decorator.args[1])
        selector = layout if names.startswith("gqa_layout") else layout == "ABAB"
        matching = [row for row in rows if row[0] == selector]
        if len(matching) != 1:
            raise ValueError(f"expected one window row for layout {layout!r}")
        return matching[0][1], matching[0][2]
    raise ValueError("SWA window parametrization not found")


_ARG_TYPES = {
    "swa": {
        "*i32":
        "cu_q_lens_ptr cu_total_seq_lens_ptr kv_lens_ptr kvlens_ptr seqlens_ptr block_table_ptr block_tables_ptr",
        "*fp32": "o_f32_ptr lse_ptr delta_ptr",
        "*i1": "causal_mask_ptr",
        "fp32": "scale softmax_scale",
    },
    "paged": {
        "*i32":
        "task_b_ptr task_q_block_ptr task_q_head_ptr core_task_offsets_ptr cu_q_lens_ptr seqlens_kv_ptr block_tables_ptr seqlens_ptr",
        "*fp32": "acc_ws_ptr lse_ws_ptr",
        "*i1": "aux_mask_ptr",
        "fp32": "softmax_scale",
    },
    "sdpa": {"*fp32": "d lse M dk dv", "*i1": "mask", "fp32": "scale"},
    "flex": {
        "*bf16": "Q K V OUT DO DQ",
        "*fp32": "LSE DELTA",
        "*i1": "DENSE_MASK PARTIAL_MASK_PACKED",
        "fp32": "SM_SCALE",
        "i32": "NUM_TASKS NUM_Q_BLOCKS Q_HEAD Q_LEN KV_LEN GQA_SHARED_HEADS",
    },
}


def _runtime_type(name, family, dtype):
    for arg_type, names in _ARG_TYPES[family].items():
        if name in names.split():
            return arg_type
    if family == "flex":
        if name.startswith("stride_"):
            return "i32"
        if name.isupper():
            return "*i32"
        raise ValueError(f"cannot infer FlexAttention argument type: {name}")
    if family == "sdpa" or name.endswith("_ptr"):
        return {"torch.bfloat16": "*bf16", "torch.float16": "*fp16", "torch.float32": "*fp32"}[dtype]
    return "i32"


def _specialization(kernel, wrapper, constants, dtype, family, *, values=None, options=None):
    """Build and validate the common (signature, constants, options) result."""
    if dtype not in {"torch.bfloat16", "torch.float16", "torch.float32"}:
        raise ValueError(f"unsupported {family} test dtype: {dtype}")
    args = kernel.args.args
    constexpr = {arg.arg for arg in args if arg.annotation and ast.unparse(arg.annotation) == "tl.constexpr"}
    if family == "sdpa" or (family == "flex" and "USE_Q_TASK_LIST" not in constexpr):
        constants = {name: value for name, value in constants.items() if name in constexpr}
    # SDPA leaves optional constexpr arguments at their declared defaults.
    if family == "sdpa":
        optional = {"LOW_TYPE", "HIGH_TYPE"}
        if kernel.args.defaults:
            optional |= {arg.arg for arg in args[-len(kernel.args.defaults):]}
        required = constexpr - optional
        constant_names = set(constants)
        if not required <= constant_names or not constant_names <= constexpr:
            raise ValueError(f"{kernel.name} constexpr parameters changed: "
                             f"missing={required - constant_names}, extra={constant_names - constexpr}")
    else:
        required = constexpr
        if required != set(constants):
            raise ValueError(f"{kernel.name} constexpr parameters changed: {required ^ set(constants)}")
    signature = {arg.arg: _runtime_type(arg.arg, family, dtype) for arg in args if arg.arg not in constexpr}
    aliases = {"intra_cache_num": "buf_slot_num_of_veccore", "inter_cache_num": "buf_slot_num_of_crosscore"}
    arg_names = {arg.arg for arg in args}
    launch_options = {
        aliases.get(kw.arg, kw.arg): _static_value(kw.value, values)
        for kw in _launch(wrapper, kernel.name).keywords
        if kw.arg is not None and kw.arg not in arg_names
    }
    return signature, constants, {**launch_options, **(options or {})}


def _first_autotune_config(kernel):
    for decorator in kernel.decorator_list:
        if not isinstance(decorator, ast.Call):
            continue
        for keyword in decorator.keywords:
            if keyword.arg != "configs":
                continue
            comprehension = keyword.value
            if isinstance(comprehension, ast.List):
                comprehension = comprehension.elts[0] if comprehension.elts else None
            if isinstance(comprehension, ast.Call):
                if comprehension.args and isinstance(comprehension.args[0], ast.Dict):
                    return {
                        _static_value(key): _static_value(value)
                        for key, value in zip(comprehension.args[0].keys, comprehension.args[0].values)
                    }
            if not isinstance(comprehension, ast.ListComp):
                continue
            names = {}
            for generator in comprehension.generators:
                source = generator.iter
                if isinstance(source, ast.IfExp):
                    # The selected source file is npu/a5, so is_910() is false.
                    source = source.body
                candidates = _static_value(source)
                names[generator.target.id] = candidates[0]
            config_call = comprehension.elt
            if not (isinstance(config_call, ast.Call) and config_call.args
                    and isinstance(config_call.args[0], ast.Dict)):
                continue
            values = {
                _static_value(key): _static_value(value, names)
                for key, value in zip(config_call.args[0].keys, config_call.args[0].values)
            }
            return values
    return {}


def infer_mojo_swa_family(root, test_path, test_name, kernel_path, kernel_name, config_id, layout):
    """Read one specialization for the seven Ascend 950 SWA JIT kernels."""
    master = Path(kernel_path).parent.name == "npu_a5_triton"
    families = {
        "_sdpa_infer_kernel": ("test_swa_infer", "SWA_INFER_CASES" if master else "test_configs_swa_infer", "infer"),
        "_swa_paged_prefill_kernel":
        ("test_paged" if master else "test_paged_prefill_swa", "test_configs_swa_prefill", "prefill"),
        "_swa_paged_prefill_aggregation_kernel":
        ("test_paged" if master else "test_paged_prefill_swa", "test_configs_swa_prefill", "prefill"),
        "_swa_paged_decode_kernel":
        ("test_paged" if master else "test_paged_decode_swa", "test_configs_swa_decode", "decode"),
        "_swa_fwd_kernel":
        ("test_swa" if master else "test_swa_function", "SWA_CASES" if master else "test_configs_swa", "training"),
        "_swa_bwd_dkdv_kernel": ("test_swa" if master else "test_swa_function",
                                 "SWA_CASES" if master else "test_configs_swa", "training"),
        "_swa_bwd_dq_kernel": ("test_swa" if master else "test_swa_function",
                               "SWA_CASES" if master else "test_configs_swa", "training"),
    }
    if kernel_name not in families:
        raise ValueError(f"no SWA source reader for {kernel_name}")
    expected_test, config_assignment, family = families[kernel_name]
    if test_name != expected_test:
        raise ValueError(f"{kernel_name} requires {expected_test}")
    root = Path(root)
    test_tree = _parse(test_path)
    kernel_tree = _parse(kernel_path)
    row = (_master_test_row if master else _test_row)(test_tree, config_assignment, config_id)
    global_window, local_window = (_master_test_window if master else _test_window)(test_tree, test_name, layout)
    if family == "infer":
        batch, q_heads, kv_heads, head_dim, _, _, dtype, _ = row
        page_size = None
    elif family == "prefill":
        batch, q_heads, kv_heads, head_dim, _, _, page_size, dtype, _ = row
    elif family == "decode":
        batch, _, q_heads, kv_heads, head_dim, _, page_size, dtype, _ = row
    else:
        batch, q_heads, kv_heads, head_dim, _, _, dtype, _ = row
        page_size = None
    if batch <= 0 or layout not in {"ABAB", "AABB"}:
        raise ValueError("unsupported SWA test batch or layout")

    if family == "training":
        interleave = layout == "ABAB"
        test_function = _definition(test_tree, test_name, ast.FunctionDef)
        if master:
            options_call = _call(test_function, "dict")
            is_causal = next(_static_value(kw.value) for kw in options_call.keywords if kw.arg == "is_causal")
            output_f32 = next(
                _static_value(kw.value) for node in ast.walk(test_function)
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "swa"
                for kw in node.keywords if kw.arg == "output_f32")
        else:
            test_call = _call(test_function, "swa_func")
            is_causal = _static_value(test_call.args[5])
            output_f32 = _static_value(test_call.args[10])
        if not isinstance(is_causal, bool) or not isinstance(output_f32, bool):
            raise ValueError("SWA training test flags could not be read")
    else:
        if master:
            interleave = layout == "ABAB"
            is_causal = True
        else:
            interleave = _gqa_interleave(root, layout)
            test_function = _definition(test_tree, test_name, ast.FunctionDef)
            constructor = _call(test_function,
                                {"infer": "MojoSWA", "prefill": "MojoPagedPrefillSWA", "decode":
                                 "MojoPagedDecodeSWA"}[family])
            is_causal = next((_static_value(k.value) for k in constructor.keywords if k.arg == "is_causal"), None)
        if not isinstance(is_causal, bool):
            raise ValueError("SWA test is_causal flag could not be read")
        output_f32 = False

    wrapper_name = {
        "infer": "swa_infer_impl", "prefill": "swa_paged_prefill_impl", "decode": "swa_paged_decode_impl", "training":
        "swa_fwd_impl"
    }[family]
    if family == "training" and kernel_name != "_swa_fwd_kernel":
        wrapper_name = ({"_swa_bwd_dkdv_kernel": "swa_dkdv", "_swa_bwd_dq_kernel": "swa_dq"}[kernel_name]
                        if master else "swa_bwd_impl")
    wrapper = _definition(kernel_tree, wrapper_name, ast.FunctionDef)
    kernel = _definition(kernel_tree, kernel_name, ast.FunctionDef)

    if family == "infer":
        block_m, block_n = _block_sizes(wrapper, dtype)
        block_d = _block_d(wrapper, head_dim)
    elif family == "prefill":
        block_m = 128 if dtype != "torch.float32" else 64
        block_n = min(block_m, 1 << (page_size - 1).bit_length())
        block_d = _block_d(wrapper, head_dim)
    elif family == "decode":
        block_m = None
        block_n = min(128, 1 << (page_size - 1).bit_length())
        block_d = 1 << (head_dim - 1).bit_length()
    else:
        autotune = _first_autotune_config(kernel)
        block_m = autotune["BLOCK_M"]
        block_n = autotune["BLOCK_N"]
        block_d = _block_d(wrapper, head_dim)

    constants = {}
    if family != "decode":
        mask_m, mask_n = _mask_shape(kernel_tree, wrapper, global_window, local_window,
                                     {"BLOCK_M": block_m, "BLOCK_N": block_n})
        constants.update(causal_mask_m_size=mask_m, causal_mask_n_size=mask_n, IS_CAUSAL=is_causal)
    if family in {"infer", "prefill", "training"}:
        constants.update(GLOBAL_WINDOW=global_window, LOCAL_WINDOW=local_window, NUM_Q_HEADS=q_heads,
                         NUM_KV_HEADS=kv_heads, GQA_INTERLEAVE=interleave, HEAD_DIM=head_dim, BLOCK_M=block_m,
                         BLOCK_N=block_n, BLOCK_D=block_d)
    else:
        constants.update(GLOBAL_WINDOW=global_window, LOCAL_WINDOW=local_window, NUM_Q_HEADS=q_heads,
                         NUM_KV_HEADS=kv_heads, GQA_INTERLEAVE=interleave, HEAD_DIM=head_dim, PAGE_SIZE=page_size,
                         BLOCK_SIZE_D=block_d, BLOCK_SIZE_N=block_n)
    if family == "prefill":
        constants["PAGE_SIZE"] = page_size
        if kernel_name == "_swa_paged_prefill_aggregation_kernel":
            if not (page_size < 128 and 128 % page_size == 0):
                raise ValueError("selected test case does not launch page aggregation")
            constants["PAGE_AGGREGATION_NUM"] = 128 // page_size
        elif page_size < 128 and 128 % page_size == 0:
            raise ValueError("selected test case launches page aggregation instead")
    if family == "training":
        if kernel_name == "_swa_fwd_kernel":
            constants["OUTPUT_F32"] = output_f32
        if block_m is None or block_n is None:
            raise ValueError("SWA autotune block sizes unavailable")

    options = {"multibuffer": autotune["multibuffer"]} if family == "training" else {}
    return _specialization(kernel, wrapper, constants, dtype, "swa", values={"unit_flag": True}, options=options)


def infer_mojo_paged_attention(root, test_path, test_name, kernel_path, kernel_name, config_id, layout,
                               cube_cores=None):
    """Read one Ascend 950 paged attention specialization from source."""
    master = Path(kernel_path).parent.name == "npu_a5_triton"
    prefill_names = {"paged_prefill_kernel", "paged_prefill_page_aggregation_kernel"}
    decode_names = {"paged_decode_kernel", "paged_decode_fd_kernel"}
    if kernel_name in prefill_names:
        expected_test = "test_paged_prefill_gqa"
        row_name = "test_configs_prefill"
    elif kernel_name in decode_names:
        expected_test = "test_paged_decode_gqa"
        row_name = "test_configs_decode"
    else:
        raise ValueError(f"no paged attention source reader for {kernel_name}")
    if test_name != ("test_paged" if master else expected_test) or layout not in {"ABAB", "AABB"}:
        raise ValueError(f"{kernel_name} requires {expected_test} and an ABAB/AABB layout")

    test_tree = _parse(test_path)
    test_function = _definition(test_tree, test_name, ast.FunctionDef)
    expected_call = ("paged_case"
                     if master else "MojoPagedPrefillGQA" if kernel_name in prefill_names else "MojoPagedDecodeGQA")
    if not any(
            isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == expected_call
            for node in ast.walk(test_function)):
        raise ValueError("test no longer calls the expected paged attention operator")
    row = _test_row(test_tree, row_name, config_id)
    if kernel_name in prefill_names:
        batch, q_heads, kv_heads, head_dim, _, _, page_size, dtype, _ = row
    else:
        batch, q_heads, kv_heads, head_dim, max_kv_len, page_size, dtype, _ = row
    if dtype not in {"torch.bfloat16", "torch.float16", "torch.float32"}:
        raise ValueError(f"unsupported paged attention dtype: {dtype}")
    kernel_tree = _parse(kernel_path)
    kernel = _definition(kernel_tree, kernel_name, ast.FunctionDef)
    wrapper_name = "paged_attention_prefill_impl" if kernel_name in prefill_names else "paged_attention_decode_impl"
    wrapper = _definition(kernel_tree, wrapper_name, ast.FunctionDef)

    block_n = min(128, 1 << (page_size - 1).bit_length())
    constants = {
        "NUM_Q_HEADS": q_heads, "NUM_KV_HEADS": kv_heads, "GQA_INTERLEAVE": layout == "ABAB", "HEAD_DIM": head_dim,
        "PAGE_SIZE": page_size
    }
    if kernel_name in prefill_names:
        if kernel_name == "paged_prefill_page_aggregation_kernel":
            if not (page_size < 128 and 128 % page_size == 0):
                raise ValueError("selected test case does not launch page aggregation")
            constants["PAGE_AGGREGATION_NUM"] = 128 // page_size
        elif page_size < 128 and 128 % page_size == 0:
            raise ValueError("selected test case launches page aggregation instead")
        for node in ast.walk(wrapper):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "ones"
                    and node.args):
                constants["AUX_MASK_SIZE"] = _static_value(node.args[0])
                break
        constants.update(BLOCK_SIZE_M=128, BLOCK_SIZE_N=block_n, BLOCK_SIZE_D=head_dim)
    else:
        constants.update(BLOCK_SIZE_D=1 << (head_dim - 1).bit_length(), BLOCK_SIZE_N=block_n)
        if kernel_name == "paged_decode_fd_kernel":
            # The test draws sequence lengths at random. Use its declared
            # maximum as a deterministic representative specialization.
            if cube_cores is None:
                import triton
                cube_cores = triton.runtime.driver.active.utils.get_device_properties("npu")["num_aicore"]
            loop_times = batch * kv_heads
            constants["KV_SPLIT_PARTS"] = max(1, min(cube_cores // loop_times, max_kv_len // 256))
    return _specialization(kernel, wrapper, constants, dtype, "paged")


def infer_mojo_sdpa(root, test_path, test_name, kernel_path, kernel_name):
    """Use one SDPA accuracy test shape for the four Ascend 950 SDPA kernels."""
    master = Path(kernel_path).parent.name == "npu_a5_triton"
    supported = {"_sdpa_infer_kernel", "kernel_sdpa_fwd", "kernel_sdpa_bwd_qkv", "kernel_sdpa_bwd_q"}
    if kernel_name not in supported or test_name != "test_sdpa":
        raise ValueError("SDPA source reader requires test_sdpa and an Ascend 950 SDPA kernel")
    test_tree = _parse(test_path)
    test_function = _definition(test_tree, test_name, ast.FunctionDef)
    row = _static_value(_assignment(test_tree, "SDPA_CASES"))[0] if master else None
    if not master:
        for decorator in test_function.decorator_list:
            if (isinstance(decorator, ast.Call) and len(decorator.args) >= 2
                    and isinstance(decorator.args[0], ast.Constant)
                    and decorator.args[0].value == "bsz, q_head_num, kv_head_num, head_dim, seq_length, block_size"):
                rows = _static_value(decorator.args[1])
                if len(rows) != 1:
                    raise ValueError("test_sdpa must have exactly one source-readable test shape")
                row = rows[0]
                break
    if row is None:
        raise ValueError("test_sdpa shape parametrization not found")
    batch, q_heads, kv_heads, head_dim, test_seq_len, _ = row
    generator = _definition(test_tree, "make_sdpa_case" if master else "generate_diffusion_attn_test_data",
                            ast.FunctionDef)
    for node in ast.walk(generator):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "randn"
                and len(node.args) >= 4):
            sequence = test_seq_len if master else _static_value(node.args[2], {"seq_length": test_seq_len})
            dtype = next((_static_value(k.value) for k in node.keywords if k.arg == "dtype"), None)
            if master and dtype is None:
                dtype = "torch.bfloat16" if any(
                    isinstance(k.value, ast.Attribute) and k.value.attr == "bfloat16"
                    for k in node.keywords
                    if k.arg == "dtype") else None
            break
    else:
        raise ValueError("test_sdpa tensor creation not found")
    if dtype != "torch.bfloat16":
        raise ValueError(f"unsupported test_sdpa dtype: {dtype}")
    scale = 1.0 / math.sqrt(head_dim)

    kernel_tree = _parse(kernel_path)
    kernel = _definition(kernel_tree, kernel_name, ast.FunctionDef)
    wrapper_name = {
        "_sdpa_infer_kernel": "sdpa_infer_impl", "kernel_sdpa_fwd": "sdpa_fwd_impl", "kernel_sdpa_bwd_qkv":
        "sdpa_bwd_impl", "kernel_sdpa_bwd_q": "sdpa_bwd_impl"
    }[kernel_name]
    wrapper = _definition(kernel_tree, wrapper_name, ast.FunctionDef)
    tune = _first_autotune_config(kernel)
    q_strides = (q_heads * sequence * head_dim, sequence * head_dim, head_dim, 1)
    kv_strides = (kv_heads * sequence * head_dim, sequence * head_dim, head_dim, 1)
    d_strides = (q_heads * sequence, sequence, 1)

    constants = {}
    if kernel_name == "_sdpa_infer_kernel":
        prefixes = {"q": q_strides, "k": kv_strides, "v": kv_strides, "o": q_strides}
        suffixes = {
            "q": ("z", "h", "m", "k"), "k": ("z", "h", "n", "k"), "v": ("z", "h", "n", "k"), "o": ("z", "h", "m", "n")
        }
        for prefix, strides in prefixes.items():
            constants.update({f"stride_{prefix}{axis}": value for axis, value in zip(suffixes[prefix], strides)})
        constants.update(BSZ=batch, Q_HEAD_NUM=q_heads, KV_HEAD_NUM=kv_heads, SEQ=sequence, HEAD_DIM=head_dim,
                         HAS_MASK=True)
    else:
        constants.update(scale=scale, num_group=kv_heads, B=batch, N=q_heads, S=sequence, H=head_dim,
                         LOW_TYPE="tl.bfloat16", HIGH_TYPE="tl.float32")
        for prefix, strides in (("Q", q_strides), ("K", kv_strides), ("V", kv_strides)):
            constants.update({f"STRIDE_{prefix}_{axis}": value for axis, value in zip(("B", "N", "S", "H"), strides)})
        constants.update({f"STRIDE_D_{axis}": value for axis, value in zip(("B", "N", "S"), d_strides)})
    constants.update({name: value for name, value in tune.items() if name.startswith("BLOCK_")})

    options = {"multibuffer": tune["multibuffer"]} if "multibuffer" in tune else {}
    return _specialization(kernel, wrapper, constants, dtype, "sdpa", options=options)


def infer_mojo_flex(root, test_path, test_name, kernel_path, kernel_name, config_id):
    """Read one FlexAttention test shape and a source-backed JIT specialization."""
    master = Path(kernel_path).parent.name == "npu_a5_triton"
    if test_name != ("test_business" if master else "test_flex_attention") or kernel_name not in {
            "flex_attention_kernel", "flex_attention_backward_dq_kernel"
    }:
        raise ValueError("FlexAttention reader requires test_flex_attention and a supported JIT kernel")
    test_tree = _parse(test_path)
    function = _definition(test_tree, test_name, ast.FunctionDef)
    row = None
    nodes = (ast.walk(_assignment(test_tree, "BUSINESS_FLEX_CASES")) if master else
             (node for decorator in function.decorator_list for node in ast.walk(decorator)))
    for node in nodes:
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "param"):
            continue
        case_id = next((_static_value(k.value) for k in node.keywords if k.arg == "id"), None)
        if case_id == config_id:
            if row is not None:
                raise ValueError(f"duplicate FlexAttention test id {config_id!r}")
            row = node
    if row is None:
        raise ValueError(f"FlexAttention test id {config_id!r} not found")
    _, q_heads, kv_heads, head_dim = [_static_value(node) for node in row.args[:4]]
    dtype = _static_value(row.args[8])
    if dtype != "torch.bfloat16":
        raise ValueError(f"unsupported FlexAttention dtype: {dtype}")

    kernel_tree = _parse(kernel_path)
    kernel = _definition(kernel_tree, kernel_name, ast.FunctionDef)
    wrapper_name = (
        ("_flex_attention_forward_leaf" if kernel_name == "flex_attention_kernel" else "_flex_attention_dq_leaf")
        if master else
        ("flex_attention_fwd_impl" if kernel_name == "flex_attention_kernel" else "flex_attention_bwd_impl"))
    wrapper = _definition(kernel_tree, wrapper_name, ast.FunctionDef)
    tile = _static_value(_assignment(kernel_tree, "TILE_BLOCK_SIZE"))
    if master:
        mask_call = next(node for node in ast.walk(_definition(test_tree, "_business_inputs", ast.FunctionDef))
                         if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                         and node.func.id == "create_flex_block_mask")
        sparse_q = sparse_kv = next(_static_value(kw.value) for kw in mask_call.keywords if kw.arg == "BLOCK_SIZE")
    else:
        sparse_q = _static_value(_assignment(test_tree, "Q_BLOCK_SIZE"))
        sparse_kv = _static_value(_assignment(test_tree, "KV_BLOCK_SIZE"))
    if sparse_q % tile or sparse_kv % tile:
        raise ValueError("FlexAttention test block sizes do not match kernel tile")

    # The test builds a packed BlockMask, while the Q-task list is calculated
    # from mask contents at runtime. Choose the direct scheduling variant.
    constants = {
        "QK_HEAD_DIM": head_dim, "V_HEAD_DIM": head_dim, "BLOCK_M": tile, "BLOCK_N": tile, "SPARSE_Q_BLOCK_SIZE":
        sparse_q, "SPARSE_KV_BLOCK_SIZE": sparse_kv, "HAS_FULL_BLOCKS": True, "USE_PACKED_PARTIAL_MASK": True,
        "USE_Q_TASK_LIST": False
    }
    if kernel_name == "flex_attention_backward_dq_kernel":
        constants["SM_SCALE"] = 1.0 / math.sqrt(head_dim)
        constants["NUM_KV_SUB_BLOCKS"] = sparse_kv // tile
        constants["GQA_SHARED_HEADS"] = q_heads // kv_heads

    return _specialization(kernel, wrapper, constants, dtype, "flex")


def infer_mojo_case(root, test_path, test_name, kernel_path, kernel_name, config_id=None, layout=None, cube_cores=None):
    """Dispatch an A5 kernel to its source reader, with no test execution."""
    kernel_path = Path(kernel_path)
    old_layout = kernel_path.parent.name == "a5" and kernel_path.parent.parent.name == "npu"
    master_layout = kernel_path.parent.name == "npu_a5_triton" and kernel_path.parent.parent.name == "kernels"
    if not (old_layout or master_layout):
        raise ValueError(f"source reader supports only A5 Triton kernels: {kernel_path}")
    readers = {
        "swa.py":
        lambda: infer_mojo_swa_family(root, test_path, test_name, kernel_path, kernel_name, config_id, layout),
        "flash_attention.py":
        lambda: infer_mojo_paged_attention(root, test_path, test_name, kernel_path, kernel_name, config_id, layout,
                                           cube_cores),
        "sdpa.py":
        lambda: infer_mojo_sdpa(root, test_path, test_name, kernel_path, kernel_name),
        "flex_attention.py":
        lambda: infer_mojo_flex(root, test_path, test_name, kernel_path, kernel_name, config_id),
    }
    if kernel_path.name not in readers:
        raise ValueError(f"no A5 source reader for {kernel_path.name}")
    return readers[kernel_path.name]()


def infer_mojo_swa_infer(root, test_path, test_name, kernel_path, kernel_name, config_id, layout):
    """Compatibility entry point for the original SWA inference reader."""
    if test_name != "test_swa_infer" or kernel_name != "_sdpa_infer_kernel":
        raise ValueError("source extraction currently supports test_swa_infer -> _sdpa_infer_kernel")
    return infer_mojo_swa_family(root, test_path, test_name, kernel_path, kernel_name, config_id, layout)
