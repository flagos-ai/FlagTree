def ops_modify_all(all_array):
    from .bmm_matmul import _bmm, bmm
    all_array.append("bmm")
    all_array.append("_bmm")
    return all_array
