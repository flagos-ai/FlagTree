import triton
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource


def iluvatar_target():
    import torch
    device = torch.cuda.current_device()
    capability = torch.cuda.get_device_capability(device)
    capability = capability[0] * 10 + capability[1]
    warp_size = 64
    return GPUTarget("corex", capability, warp_size)


def compile_iluvatar(fn, signature, constexprs=None):
    src = ASTSource(fn=fn, signature=signature, constexprs=constexprs or {})
    return triton.compile(src, target=iluvatar_target())
