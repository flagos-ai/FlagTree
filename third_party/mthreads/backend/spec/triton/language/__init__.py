def language_extend_globals(globals_dict):
    # NOTE: Must use absolute path import.
    from triton.language.standard import squeeze, unsqueeze  # Triton 3.7
    from triton.language.core import to_tensor  # Triton 3.7
    from triton.language.core import _experimental_descriptor_load, _experimental_descriptor_store
    globals_dict["squeeze"] = squeeze  # Triton 3.7
    globals_dict["unsqueeze"] = unsqueeze  # Triton 3.7
    globals_dict["to_tensor"] = to_tensor  # Triton 3.7
    globals_dict["_experimental_descriptor_load"] = _experimental_descriptor_load
    globals_dict["_experimental_descriptor_store"] = _experimental_descriptor_store
