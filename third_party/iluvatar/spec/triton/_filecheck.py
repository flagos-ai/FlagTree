from triton.backends.compiler import GPUTarget


def filecheck_make_ir(src, target, options, codegen_fns, module_map, context):
    return src.make_ir(target, options, codegen_fns, module_map, context)


def spec_get_stub_target() -> GPUTarget:
    return GPUTarget("corex", 71, 64)
