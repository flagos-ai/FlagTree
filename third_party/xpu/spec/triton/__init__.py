from ._utils import validate_block_shape


def _utils_extend_globals(globals_dict):
    # XPU's prebuilt Python modules import these helpers from ``triton._utils``.
    # Keep their implementation in the XPU spec tree and publish them only for
    # an active XPU build.
    from ._utils import get_xpu_arch, parse_xpu_arch
    globals_dict["get_xpu_arch"] = get_xpu_arch
    globals_dict["parse_xpu_arch"] = parse_xpu_arch


def language_extend_globals(globals_dict):
    # These XPU-only language builtins were exported by XPU's former package
    # overlay.  Restore them through the shared language specialization hook.
    from triton.language.core import atomic_mul
    from triton.language.math import gelu
    globals_dict["atomic_mul"] = atomic_mul
    globals_dict["gelu"] = gelu


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
