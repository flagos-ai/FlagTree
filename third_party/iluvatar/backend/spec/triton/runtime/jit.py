import ast
from triton.runtime.jit import JITFunction


def get_disable_sme():
    import torch
    import os
    disable_sme = os.getenv("TRITON_DISABLE_SME", default="0") == "1"
    cc = torch.cuda.get_device_capability()
    cc = cc[0] * 10 + cc[1]
    if cc == 70:  # for ivcore10
        disable_sme = True

    return disable_sme


def get_corex_sme(enable_sme=True):
    import torch
    if not enable_sme:
        return 0
    if not (hasattr(torch, "corex") and torch.corex):
        return 0
    close_sme = get_disable_sme()
    if close_sme:
        return 0
    return 1


class CallVisitor(ast.NodeVisitor):

    def __init__(self, globals):
        super().__init__()
        self.use_sme = False
        self.globals = globals

    def visit_Call(self, node):
        if isinstance(node.func, ast.Name):
            val = self.globals.get(node.func.id, None)
            if isinstance(val, JITFunction):
                self.use_sme = val.use_corex_load_inc


def device_of(arg):
    if hasattr(arg, "device") and hasattr(arg.device, "type"):
        return arg.device.type
    else:
        return ""


def pinned_memory_of(arg):
    if hasattr(arg, "is_pinned") and callable(arg.is_pinned):
        return arg.is_pinned()
    else:
        return False


def ext_JITFunction_spec_of(arg):
    return (arg % 16 == 0, arg % JITFunction.divisibility_8 == 0, arg == 1)


def is_corex_param(x, enable_sme):
    if enable_sme:
        if hasattr(x, "data_ptr"):
            return x.data_ptr() % JITFunction.corex_divisibility == 0
        elif isinstance(x, int):
            return True
    return False


def get_corex_param(arg):
    import torch
    res_stride = (1 << 31) - 1  # max int32_t
    if hasattr(arg, "data_ptr") and torch.is_tensor(arg) and arg.dtype in [
            torch.float16, torch.float32, torch.bfloat16, torch.int8
    ]:
        if arg.dim() >= 2:
            # Remove dimension of 1
            squeezed_arg = arg.squeeze()
            if squeezed_arg.dim() >= 2 and (squeezed_arg.stride()[-1] == 1 or squeezed_arg.stride()[-2] == 1):
                res_stride = squeezed_arg.stride()[-1] * squeezed_arg.stride()[-2]
        else:
            return 1
    elif isinstance(arg, int):
        if arg < res_stride:
            res_stride = arg
    return res_stride


def ext_JITFunction_get_config(jitFunction, divisible_by_16, equal_to_1, *args):
    from triton.compiler import AttrsDescriptor
    enable_sme = get_corex_sme(jitFunction.use_corex_load_inc or jitFunction.visitor.use_sme)
    corex_param = {
        param.num: get_corex_param(arg)
        for param, arg in zip(jitFunction.params, args)
        if is_corex_param(arg, enable_sme) and not param.do_not_specialize and not param.is_constexpr
    }
    # folded equal_to_1 and None
    # TODO: method to collect all folded args
    return AttrsDescriptor(tuple(divisible_by_16), tuple(equal_to_1), corex_param)


def get_JITFunction_key(jitFunction, bound_args, sig_and_spec, constexpr_vals, excess_kwargs, *args, **kwargs):
    import os
    import torch
    from triton.runtime.driver import driver
    target = driver.active.get_current_target()
    backend = jitFunction.make_backend(target)
    options = backend.parse_options(kwargs)
    only_save_best_config_cache = os.environ.get("TRITON_ONLY_SAVE_BEST_CONFIG_CACHE", "0") == "1"
    options.use_sme = get_corex_sme(jitFunction.use_corex_load_inc or jitFunction.visitor.use_sme)
    #need get sme_param
    configs = None
    if options.use_sme:
        bound_vals = tuple(bound_args.values())
        configs = (jitFunction._get_config(*bound_vals), )
        options.hash_corex = configs[0].hash()
    shape_info = ''
    if only_save_best_config_cache:
        if not shape_info:
            for arg in args:
                if torch.is_tensor(arg):
                    shape_info += '_' + '_'.join(str(_) for _ in list(arg.shape))
        key = ''.join(sig_and_spec) + str((constexpr_vals, excess_kwargs)) + str((options.hash_corex, shape_info))
    else:
        key = ''.join(sig_and_spec) + str((constexpr_vals, excess_kwargs)) + str(options.hash_corex)
    return key


def is_JITFunction_support_cpu(*args):
    pinned_memory_flags = [pinned_memory_of(arg) for arg in args]
    device_types = [device_of(arg) for arg in args]
    device_types = [_device_type for _device_type in device_types if _device_type != ""]
    is_cpu = device_types and all(device_type == "cpu" for device_type in device_types)
    is_pinned_memory = any(pinned_memory_flag for pinned_memory_flag in pinned_memory_flags)
    if is_cpu and not is_pinned_memory:
        raise ValueError("Cannot find backend for cpu")


def get_JITFunction_options(jitFunction, target, backend, options, bound_args):
    options.use_sme = get_corex_sme(jitFunction.use_corex_load_inc or jitFunction.visitor.use_sme)
    #need get sme_param
    configs = None
    if options.use_sme:
        bound_vals = tuple(bound_args.values())
        configs = (jitFunction._get_config(*bound_vals), )
        options.hash_corex = configs[0].hash()
    return options


def ext_JITFunction_init(jitFunction):
    # use to record fn cache files
    jitFunction.hash_cache_file = None
    jitFunction.so_path = None
    jitFunction.use_corex_load_inc = 'dot' in jitFunction.src
    jitFunction.visitor = CallVisitor(jitFunction.__globals__)
    jitFunction.visitor.visit(ast.parse(jitFunction.src))
