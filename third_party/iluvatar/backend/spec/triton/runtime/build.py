def is_corex():
    import torch
    return hasattr(torch, "corex") and torch.corex == True


def get_cc(clang, gcc):
    if is_corex():
        cc = clang if clang is not None else gcc
        return cc
    else:
        return None
