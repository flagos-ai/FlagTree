def language_extend_globals(globals_dict):
    from triton.language.core import fp_to_fp4, fp4_to_fp_scaled

    globals_dict["fp_to_fp4"] = fp_to_fp4
    globals_dict["fp4_to_fp_scaled"] = fp4_to_fp_scaled
