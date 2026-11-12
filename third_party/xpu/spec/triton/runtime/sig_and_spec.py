import inspect
import itertools
import os
import re
import torch
import torch
from packaging import version

# 预先检测一次PyTorch版本
_current_version = version.parse(torch.__version__)
_min_version = version.parse("2.5.0")  # 设置最低版本，决定是否包含 uints

# 构建字典
type_map_dict = {
    torch.bool: "i1",
    torch.float16: "fp16",
    torch.bfloat16: "bf16",
    torch.float32: "fp32",
    torch.float64: "fp64",
    torch.int8: "i8",
    torch.int16: "i16",
    torch.int32: "i32",
    torch.int64: "i64",
    torch.uint8: "u8",
}

if _current_version >= _min_version:
    type_map_dict.update({
        torch.uint16: "u16",
        torch.uint32: "u32",
        torch.uint64: "u64",
    })


def fast_compute_spec_key(v):
    if isinstance(v, int):
        if v % 16 == 0:
            return "D"
        elif v == 1:
            return "1"
        else:
            return "N"
    # if callable(v.data_ptr):
    #     return "D"
    return "D"


def fast_mangle_type(arg, is_const=False):

    if arg is None:
        return "none"
    elif isinstance(arg, bool):
        return "i1"
    elif isinstance(arg, int):
        if -(2**31) <= arg and arg <= 2**31 - 1:
            return "i32"
        elif 2**63 <= arg and arg <= 2**64 - 1:
            return "u64"
        else:
            return "i64"
    elif isinstance(arg, float):
        return "fp32"
    else:
        # dtypes are hashable so we can memoize this mapping:
        dtype = arg.dtype
        suffix = type_map_dict[dtype]
        res = f"{'*k' if is_const else '*'}{suffix}"

        return res


def fast_create_function_from_signature(sig, kparams):
    """生成可缓存的高效参数处理函数，优化内核启动开销"""

    assert len(sig.parameters) == len(kparams)

    func_args = []
    dict_entries = []
    constexpr_names = []
    non_constexpr_names = []
    signature_type_exprs = []
    specialisation_exprs = []

    for (param_name, param), kparam in zip(sig.parameters.items(), kparams):
        if param.default is inspect.Parameter.empty:
            func_args.append(param_name)
        else:
            func_args.append(f"{param_name}=default_{param_name}")
        dict_entries.append(f"'{param_name}': {param_name}")
        if kparam.is_constexpr:
            constexpr_names.append(param_name)
        else:
            non_constexpr_names.append(param_name)
            if not kparam.do_not_specialize:
                specialisation_exprs.append(f'compute_spec_key({param_name})')

            type_expr = (f'"{kparam.annotation_type}"'
                         if kparam.annotation_type else f'mangle_type({param_name}, {kparam.is_const})')
            signature_type_exprs.append(type_expr)

    cache_key = ', '.join(signature_type_exprs + specialisation_exprs)
    constexpr_tuple = ', '.join(constexpr_names) or ' '
    non_constexpr_tuple = ', '.join(non_constexpr_names) or ' '

    func_args.append('**excess_kwargs')
    args_str = ', '.join(func_args)
    dict_str = ', '.join(dict_entries)

    func_body = (f"def dynamic_func({args_str}):\n"
                 f"    return {{{dict_str}}}, ({cache_key}), ({constexpr_tuple}), "
                 f"({non_constexpr_tuple}), excess_kwargs")

    func_namespace = {
        f"default_{name}": p.default
        for name, p in sig.parameters.items()
        if p.default is not inspect.Parameter.empty
    }
    func_namespace.update({'mangle_type': fast_mangle_type, 'compute_spec_key': fast_compute_spec_key})

    exec(func_body, func_namespace)
    return func_namespace['dynamic_func']
