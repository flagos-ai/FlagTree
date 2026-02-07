def _get_divisible_by_4(attrs):
    div4 = getattr(attrs, "divisible_by_4", None)
    if div4:
        return set(div4)
    return set()


def kernel_suffix_by_divisibility(specialization, i, suffix: str) -> str:
    if i in specialization.divisible_by_8:
        suffix += 'e'
    if i in _get_divisible_by_4(specialization):
        suffix += 'f'
    return suffix


def generate_new_attrs_in_ast_to_ttir(attrs):
    corex = getattr(attrs, "corexLoad", {}) or {}
    new_attrs = {k: [("tt.corex_stride", v)] for k, v in corex.items() if isinstance(k, int) and k >= 0}
    div4 = _get_divisible_by_4(attrs)
    divisible_func = div4 if div4 else attrs.divisible_by_16
    divisible_bytes = 4 if div4 else 16
    for k in divisible_func:
        attr = new_attrs[k] if k in new_attrs else []
        attr.append(("tt.divisibility", divisible_bytes))
        new_attrs[k] = attr
    return new_attrs
