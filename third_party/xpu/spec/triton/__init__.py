from ._utils import validate_block_shape


def knobs_extend_globals(globals_dict):
    # Keep the shared knobs and their scope/reset behaviour; only XPU defaults
    # and the XPU launch fast-path switch differ.
    env_bool = globals_dict["env_bool"]
    compilation_knobs = globals_dict["compilation_knobs"]
    store_binary_only = env_bool("TRITON_STORE_BINARY_ONLY", default=True)
    store_binary_only.__set_name__(compilation_knobs, "store_binary_only")
    compilation_knobs.store_binary_only = store_binary_only
    runtime_knobs = globals_dict["runtime_knobs"]
    runtime_knobs.disable_launch_fast_path = env_bool("TRITON_DISABLE_LAUNCH_FAST_PATH").get()


def do_bench(*args, **kwargs):
    from .testing import do_bench as xpu_do_bench
    return xpu_do_bench(*args, **kwargs)
