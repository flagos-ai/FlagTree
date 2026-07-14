def language_extend_globals(globals_dict):
    # NOTE: Must use absolute path import.
    from triton.language.standard import squeeze, unsqueeze
    from triton.language.core import _experimental_descriptor_load, _experimental_descriptor_store
    from triton.language.core import to_tensor
    globals_dict["squeeze"] = squeeze
    globals_dict["unsqueeze"] = unsqueeze
    globals_dict["_experimental_descriptor_load"] = _experimental_descriptor_load
    globals_dict["_experimental_descriptor_store"] = _experimental_descriptor_store
    globals_dict["to_tensor"] = to_tensor
