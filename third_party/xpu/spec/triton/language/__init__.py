def language_extend_globals(globals_dict):
    # These XPU-only language builtins were exported by XPU's former package
    # overlay.  Restore them through the shared language specialization hook.
    from triton.language.core import atomic_mul
    from triton.language.math import gelu
    globals_dict["atomic_mul"] = atomic_mul
    globals_dict["gelu"] = gelu
