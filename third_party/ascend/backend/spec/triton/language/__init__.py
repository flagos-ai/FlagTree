def language_modify_all(all_array):
    try:
        import acl
        is_compile_on_910_95 = acl.get_soc_name().startswith("Ascend910_95")
    except Exception as e:
        is_compile_on_910_95 = False

    from .standard import topk
    from .core import (
        make_tensor_descriptor,
        load_tensor_descriptor,
        store_tensor_descriptor,
        gather,
    )
    all_array.append("topk")
    all_array.append("make_tensor_descriptor")
    all_array.append("load_tensor_descriptor")
    all_array.append("store_tensor_descriptor")
    all_array.append("gather")
