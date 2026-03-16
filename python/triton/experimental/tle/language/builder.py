# Copyright 2026- Xcoresigma Technology Co., Ltd


def create_dsa_method_wrapper_with_tle_builder(main_builder, delegate_builder, method_name):
    delegate_method = getattr(delegate_builder, method_name)

    def wrapper(*args, **kwargs):
        saved_ip = main_builder.get_insertion_point()
        saved_loc = main_builder.get_loc()
        delegate_builder.restore_insertion_point(saved_ip)
        if saved_loc:
            delegate_builder.set_loc(saved_loc)
        result = delegate_method(*args, **kwargs)
        main_builder.restore_insertion_point(saved_ip)
        if saved_loc:
            main_builder.set_loc(saved_loc)
        return result

    wrapper.__name__ = method_name
    wrapper.__doc__ = getattr(delegate_method, '__doc__', None)
    return wrapper


def attach_builder_methods_with_tle_builder(main_builder, delegate_builder, method_names):
    """Attach multiple methods from a delegate builder to the main builder."""
    for method_name in method_names:
        wrapper = create_dsa_method_wrapper_with_tle_builder(main_builder, delegate_builder, method_name)

        if hasattr(main_builder, method_name):
            raise AttributeError(f"Method '{method_name}' already exists in the main builder.")
        setattr(main_builder, method_name, wrapper)


def setup_unified_builder_with_tle_builder(main_builder, buffer_builder):
    """Set up a unified builder interface by attaching methods from specialized builders."""
    main_builder._buffer_builder = buffer_builder
    buffer_methods = [
        'create_dsa_alloc',
        'create_dsa_copy',
        'create_dsa_add',
        'create_dsa_sub',
        'create_dsa_mul',
        'create_dsa_div',
        'create_dsa_max',
        'create_dsa_min',
        # 'create_dsa_dot',
        'dsa_to_buffer',
        'dsa_to_tensor',
        'dsa_get_null_attr',
        'dsa_get_buffer_type',
        'dsa_get_buffer_type_with_strides',
        "create_dsa_extract_scalar",
        "create_dsa_extract_slice",
        "create_dsa_insert_slice",
        "create_dsa_subview",
    ]
    attach_builder_methods_with_tle_builder(main_builder, buffer_builder, buffer_methods)
