def init_AttrsDescriptor_corexLoad(corexLoad):
    return dict() if corexLoad is None else corexLoad


def ext_AttrsDescriptor_to_dict(corexLoad):
    return {'corexLoad': list(corexLoad.items())}


def ext_AttrsDescriptor_from_dict(data):
    from triton.compiler.compiler import AttrsDescriptor
    return AttrsDescriptor(divisible_by_16=set(data.get('divisible_by_16', [])),
                           equal_to_1=set(data.get('equal_to_1', [])), corexLoad=dict(data.get('corexLoad', [])))


def ext_AttrsDescriptor_hash_key(attrsDescriptor):
    key = str([
        sorted(x) if isinstance(x, tuple) or isinstance(x, set) else x.values()
        for x in attrsDescriptor.__dict__.values()
    ])
    return key


def set_src_fn_hash_cache_file(ir_source, src, hash):
    from triton.runtime.jit import JITFunction
    if not ir_source and isinstance(src.fn, JITFunction):
        src.fn.hash_cache_file = hash


def set_src_fn_so_path(ir_source, src):
    from triton.runtime.driver import driver
    if not ir_source:
        src.fn.so_path = driver.active.get_cache_path()


def handle_n_threads_in_CompiledKernel_init(compiledKernel, *n_threads):
    from triton.runtime.autotuner import OutOfResources
    if compiledKernel.metadata.num_warps * 64 > n_threads[0]:
        compiledKernel.module = None
        raise OutOfResources(compiledKernel.metadata.num_warps * 64, compiledKernel.n_threads, "threads")
