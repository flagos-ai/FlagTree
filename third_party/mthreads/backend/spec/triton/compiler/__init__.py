def compiler_extend_globals(globals_dict):
    # NOTE: Must use absolute path import.
    from triton.compiler.compiler import max_shared_mem
    globals_dict["max_shared_mem"] = max_shared_mem
