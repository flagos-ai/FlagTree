def language_modify_all(all_array):
    import triton.language as tl
    from .core import corex_sme

    # 把后端实现挂到公共 namespace 上
    tl.corex_sme = corex_sme

    # 同时加入导出列表
    all_array.append("corex_sme")
    return all_array
