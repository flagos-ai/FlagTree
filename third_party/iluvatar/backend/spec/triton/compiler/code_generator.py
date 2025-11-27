def kernel_suffix_by_divisibility(specialization, i, suffix: str) -> str:
    if i in specialization.divisible_by_8:
        suffix += 'e'
    return suffix


def generate_new_attrs_in_ast_to_ttir(attrs):
    new_attrs = {k[0]: [("tt.corex_stride", k[1])] for k in attrs.corexLoad.items()}
    for k in attrs.divisible_by_16:
        attr = new_attrs[k] if k in new_attrs else []
        attr.append(("tt.divisibility", 16))
        new_attrs[k] = attr
    return new_attrs
