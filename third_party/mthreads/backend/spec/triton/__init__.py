from ._utils import apply_with_path, _tuple_create
from ._filecheck import spec_get_stub_target
from .compiler import compiler_extend_globals
from .language import language_extend_globals


def triton_extend_globals(globals_dict):
    # NOTE: Must use absolute path import.
    from triton.compiler.compiler import max_shared_mem
    globals_dict["max_shared_mem"] = max_shared_mem
