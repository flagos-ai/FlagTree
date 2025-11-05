def ext_ASTSource_attrs(ast_source):
    from triton.backends.ascend.compiler import AscendAttrsDescriptor
    if ast_source.attrs is None:
        ast_source.attrs = AscendAttrsDescriptor()

def opt_ascend_compile_speed(file_name, metadata_path, fn_cache_manager):
    import os
    compile_speed_opt = os.getenv("TRITON_ASCEND_COMPILE_SPEED_OPT", 'false').lower() in ('true', '1')
    if (compile_speed_opt):
        ttir_path = f"{file_name}.ttir"
        if (metadata_path is None) and (fn_cache_manager.has_file(ttir_path)):
            # Already compile once but failed. So directly return
            raise Exception("already failed once")

def set_CompiledKernel_metadata_stream(compiled_kernel, stream):
    if stream is None:
        return stream
    return compiled_kernel.metadata.stream

def handle_compile_error(e, ext):
    from triton.compiler.errors import MLIRCompilationError
    if (ext == "ttadapter"):
        stage_name = "ConvertTritonIRToLinalgIR"
    elif (ext == "npubin"):
        stage_name = "ConvertLinalgRToBinary"
    else:
        stage_name = "MLIRCompile"
    error_detail = e.stderr.decode('utf-8') if hasattr(e, 'stderr') and e.stderr else str(e)
    raise MLIRCompilationError(stage_name, error_detail)

def is_CompiledKernel_getattribute_need_init_handles():
    return False