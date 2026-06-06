def language_extend_globals(globals_dict):
    from triton.tools.get_ascend_devices import is_compile_on_910_95
    globals_dict["is_compile_on_910_95"] = is_compile_on_910_95