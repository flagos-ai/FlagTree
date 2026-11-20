def layout_tune():
    """Tune layout candidates for the wrapped @triton.jit kernel (MetaX CommonIR)."""
    from triton.runtime._commonir_autolayout import LayoutTuner
    return LayoutTuner


def triton_extend_globals(globals_dict):
    globals_dict["layout_tune"] = layout_tune
