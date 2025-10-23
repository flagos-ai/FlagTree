from triton.runtime.cache import get_cache_manager, get_dump_manager
from pathlib import Path
import tempfile
import os
import sysconfig
import subprocess
import importlib
from triton.backends.ascend.utils import _get_llvm_path

# TODO: temporarily fake CPUUtils class
class CPUUtils(object):
    def __new__(cls):
        if not hasattr(cls, 'instance'):
            cls.instance = super(CPUUtils, cls).__new__(cls)
        return cls.instance

    def __init__(self):
        pass

    def get_device_properties(self, device):
        # temperoarily added properties to avoid triton-compiler complain
        # fetch available memory at runtime
        return {"max_shared_mem": 1}
    
    def load_binary(self, name, kernel, shared, device):
        # TODO (temperoarily fake function) load a binary from binary object to device
        # return value are: (mod, funcptr/handle, n_regs, n_spills)
        return None, kernel, 0, 0
    
class CPULauncher(object):
    def __init__(self, src, metadata):
        kernel_name = metadata.name.split()[0]
        signature = src.signature
        constants = src.constants
        launcher_src = generate_cpu_wrapper_src(constants, signature, kernel_name)
        self.launch = compile_module(launcher_src)

    def __call__(self, *args, **kwargs):
        self.launch(*args, **kwargs)


class CPUDriver:
    def __init__(self):
        self.utils = CPUUtils()
        self.launcher_cls = CPULauncher
        super().__init__()

    def get_current_target(self):
        # TODO: do we rely on CPU arch?
        return ("cpu", "arm-64")

    def get_current_device(self):
        """
        Get current device
        """
        # TODO: dummy device-getter for cpu backend
        return 0

    def set_current_device(self, device):
        """
        Set current device as the given device
        """
        # TODO: dummy device-setter for cpu backend
        return

    def get_current_stream(self, device):
        """
        Get stream for current device
        """
        # TODO: dummy stream api for cpu backend.
        return 0

# the template is from triton-adapter HEAD. Wrapping the generated kernel assembly into a python module
def generate_cpu_wrapper_src(constants, signature, kernel_name):
    def _ty_to_cpp(ty):
        if ty[0] == '*':
            return "void*"
        return {
            "i1": "int32_t",
            "i8": "int8_t",
            "i16": "int16_t",
            "i32": "int32_t",
            "i64": "int64_t",
            "u32": "uint32_t",
            "u64": "uint64_t",
            "fp16": "float",
            "bf16": "float",
            "fp32": "float",
            "f32": "float",
            "fp64": "double",
        }[ty]
    
    def _extracted_ty(ty):
        if ty[0] == '*':
            return "PyObject*"
        return {
            'i1': 'int32_t',
            'i32': 'int32_t',
            'i64': 'int64_t',
            'u32': 'uint32_t',
            'u64': 'uint64_t',
            'fp16': 'float',
            'bf16': 'float',
            'fp32': 'float',
            'f32': 'float',
            'fp64': 'double',
        }[ty]

    def _format_of(ty):
        return {
            "PyObject*": "O",
            "float": "f",
            "double": "d",
            "long": "l",
            "uint32_t": "I",
            "int32_t": "i",
            "uint64_t": "K",
            "int64_t": "L",
        }[ty]
    
    def _generate_launcher(constants, signature, kernel_name):
        arg_decls = ', '.join(f"{_ty_to_cpp(ty)} arg{i}" for i, ty in signature.items())
        format = "iiiOOO" + ''.join([_format_of(_extracted_ty(ty)) for ty in signature.values()])
        # to be filled
        return f"""
        """
    launcher_src = _generate_launcher(constants, signature, kernel_name)
    return launcher_src

def compile_module(launcher_src):
    # This function was renamed and made public in Python 3.10
    if hasattr(sysconfig, 'get_default_scheme'):
        scheme = sysconfig.get_default_scheme()
    else:
        scheme = sysconfig._get_default_scheme()
    # 'posix_local' is a custom scheme on Debian. However, starting Python 3.10, the default install
    # path changes to include 'local'. This change is required to use triton with system-wide python.
    if scheme == 'posix_local':
        scheme = 'posix_prefix'
    py_include_dir = sysconfig.get_paths(scheme=scheme)["include"]

    def launch(gridX, gridY, gridZ, stream, cu_function,
               packed_metadata, launch_metadata,
               launch_enter_hook, launch_exit_hook,
               *args):
        kernel_name = packed_metadata["kernel_name"]
        cache = get_cache_manager(packed_metadata["hash"])
        filename = f"{kernel_name}_cpu_launcher.so"
        cache_path = cache.get_file(filename)
        if cache_path is None:
            asm_src = cu_function
            with tempfile.TemporaryDirectory() as tmpdir:
                asm_src_path = os.path.join(tmpdir, "kernel.s")
                launcher_src_path = os.path.join(tmpdir, "main.cxx")
                if packed_metadata["debug"]:
                    dump_manager = get_dump_manager(packed_metadata["hash"])
                    dump_manager.put(launcher_src, "kernel_cpu_launcher.cxx", binary=False)
                so_path = os.path.join(tmpdir, "kernel.so")
                Path(asm_src_path).write_bytes(asm_src)
                Path(launcher_src_path).write_text(launcher_src)
                # Compile it together.
                subprocess.check_call([_get_llvm_path("bin", "clang++"), launcher_src_path, asm_src_path, f"-I{py_include_dir}", f"-I{Path(__file__).resolve().parent}", "-shared", "-fPIC", "-o", so_path])

                with open(so_path, "rb") as f:
                    cache_path = cache.put(f.read(), filename, binary=True)
    
        # Load and launch the compiled kernel.
        spec = importlib.util.spec_from_file_location("__triton_adapter_ref_cpu_kernel_launcher", cache_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod.launch(gridX, gridY, gridZ, launch_enter_hook, launch_exit_hook, packed_metadata, *args)
    
    return launch