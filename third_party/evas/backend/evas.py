##### Add EVBackend here
import hashlib
from tabnanny import check
import os, tempfile
import sysconfig
import numpy as np
import importlib.util
from pathlib import Path
import site
import sys
import logging
import re
import subprocess
import functools

from triton.runtime.cache import get_dump_manager, get_cache_manager
from triton.backends.driver import DriverBase
from triton.backends.compiler import BaseBackend, GPUTarget


from .utils import run_command, _cache_key

from triton._C import libtriton
from triton._C.libtriton import ir, passes
try:
    from triton._C.libtriton import triton_shared
except ImportError:
    triton_shared = None
from dataclasses import dataclass
from typing import Any, Tuple, Dict
from types import ModuleType

CLUSTER_NUM = 4
CORE_NUM = 4
# EVAS Driver
# -------------------- Launcher ----------------------------
def _ty_to_cpp(ty):
    return {
        "i1": "int32_t",
        "i8": "int8_t",
        "i16": "int16_t",
        "i32": "int32_t",
        "i64": "int64_t",
        "u1": "uint32_t",
        "u8": "uint8_t",
        "u16": "uint16_t",
        "u32": "uint32_t",
        "u64": "uint64_t",
        "fp16": "float",
        "bf16": "float",
        "fp32": "float",
        "f32": "float",
        "fp64": "double",
        "*fp32": "float*",
        "*fp16": "_Float16*",
        "*i1": "int8_t*",
        "*i8": "int8_t*",
        "*i16": "int16_t*",
        "*i32": "int32_t*",
        "*i64": "int64_t*",
        "*u1" : "int8_t*",
        "*u8" : "int8_t*",
        "*bf16": "bfloat16_t*"
    }[ty]


def compile_module(signature, constants, launcher_src, arg_names, launch_call_placeholder):
    params = list(signature.items())

    def gen_args(signature, args):
        # kernel_arg_decls = [_ty_to_cpp(ty) for i, ty in signature.items() if i not in constants and ty[0] == "*"]
        args_desc = []
        for index, (arg_idx, ty) in enumerate(params):
            arg = args[index]
            is_output = arg_names[arg_idx].startswith("out")
            is_pointer = ty[0] == "*"
            is_tensor = hasattr(arg, 'numel') and hasattr(arg, 'element_size')
            args_desc.append(
                {
                    "tensor": arg,
                    "size": arg.numel() * arg.element_size() if is_pointer and is_tensor else None,
                    "arg_idx": arg_idx,
                    "dtype": arg.dtype if is_pointer and is_tensor else None,
                    "arg_type": _ty_to_cpp(ty),
                    "is_pointer": is_pointer,
                    "addr": arg.data_ptr() if is_pointer and is_tensor else None,
                    "is_output": is_output,
                    "is_cpu": arg.device.type == "cpu" if is_pointer and is_tensor else None
                }
            )
        return args_desc

    def normalize_args(args):
        if len(args) == len(params):
            return list(args)
        return [arg for i, arg in enumerate(args) if i in signature.keys()]

    def pack_launcher(args_desc, launcher_src, kernel_name):
        packed = launcher_src
        host_to_device_code = ' '.join(f'void* host_ptr{i} = ptr_info{i}.dev_ptr; {desc["arg_type"]} device_ptr{i}; evMalloc((void **)&device_ptr{i}, {desc["size"]}, mem_affinitymap); evMemcpy(device_ptr{i}, host_ptr{i}, {desc["size"]}, evMemcpyHostToDevice, mem_affinitymap); printf("[LOG] Copy from host %p to device %p\\n", host_ptr{i}, device_ptr{i}); ptr_info{i}.dev_ptr = (void*)device_ptr{i};' for i, desc in enumerate(args_desc) if desc['is_pointer'] and desc['is_cpu'])
        device_to_host_code = ' '.join(f'evMemcpy(host_ptr{i}, device_ptr{i}, {desc["size"]}, evMemcpyDeviceToHost, mem_affinitymap); printf("[LOG] Copy from device %p to host %p\\n", device_ptr{i}, host_ptr{i});' for i, desc in enumerate(args_desc) if desc['is_pointer'] and desc['is_output'] and desc['is_cpu'])
        return  packed.replace("HOST_TO_DEVICE_PLACE_HOLDER", host_to_device_code).replace("DEVICE_TO_HOST_PLACE_HOLDER", device_to_host_code).replace(launch_call_placeholder, kernel_name)


    def launch(
        gridX,
        gridY,
        gridZ,
        stream,
        cu_function,
        kernel_metadata,
        launch_metadata,
        launch_enter_hook,
        launch_exit_hook,
        *args,
    ):
        asm_src = cu_function
        kernel_name = kernel_metadata[6]
        args = normalize_args(args)
        args_desc = gen_args(signature, args)
        src = pack_launcher(args_desc, launcher_src, kernel_name)
        module_key = hashlib.sha256(src.encode("utf-8") + asm_src).hexdigest()
        name = f"__triton_ref_epu_kernel_launcher_{module_key[:16]}"
        src = src.replace("MODULE_NAME_PLACEHOLDER", name)
        key = hashlib.sha256(src.encode("utf-8") + asm_src).hexdigest()
        cache = get_cache_manager(key)
        filename = f"{name}.so"
        cache_path = cache.get_file(filename)

        if cache_path is None:
            module = compile_module_from_src(src.encode("utf-8"))
            cache.put(src, f"{name}.cc")
            cache_path = cache.put(module, filename, binary=True)
        
        # Load and launch the compiled kernel.
        spec = importlib.util.spec_from_file_location(name, cache_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        with tempfile.TemporaryDirectory() as tmpdir:
            asm_src_path = os.path.join(tmpdir, "kernel.elf")
            Path(asm_src_path).write_bytes(asm_src)
            return mod.launch(gridX, gridY, gridZ, asm_src_path,
                            kernel_metadata, launch_metadata,
                            launch_enter_hook, launch_exit_hook,
                            *args)

    return launch

def _extracted_type(ty):
    if ty[0] == '*':
        return "PyObject*"
    return _ty_to_cpp(ty)

def _format_of(ty):
    return {
      "PyObject*": "O",
      "float": "f",
      "double": "d",
      "long": "l",
      "int8_t": "b",
      "int16_t": "h",
      "int32_t": "i",
      "int64_t": "l",
      "uint8_t": "B",
      "uint16_t": "H",
      "uint32_t": "I",
      "uint64_t": "K",
    }[ty]

def _generate_launcher(constants, signature, kernel_name):
    args_format = ''.join([_format_of(_extracted_type(ty)) for ty in signature.values()])
    format = "iiisOOOO" + args_format

    args_list = ', '.join(f"&_arg{i}" for i, ty in signature.items()) if len(signature) > 0 else ''

    
    params = [(i, ty) for i, ty in signature.items() if i not in constants]
    # kernel_arg_decls = ', '.join(_ty_to_cpp(ty) if ty[0] != "*" else f"void*" for i, ty in params)
    # kernel_arg_decls += ', ' if kernel_arg_decls else ''

    # kernel_parameters = ', '.join(f"static_cast<{_ty_to_cpp(ty)}>(arg{i})" if ty[0] != "*" else f"arg{i}" for i, ty in params)
    # kernel_parameters += ', ' if kernel_parameters else ''

    return f"""
#include <assert.h>
#include <stdbool.h>
#include <hpe.h>
#include <stdint.h>
#include <stdio.h>
#include <unistd.h>
#include <Python.h>
#include <helper_hpe.h>

typedef struct _DevicePtrInfo {{
  void *dev_ptr;
  bool valid;
}} DevicePtrInfo;

static inline DevicePtrInfo getPointer(PyObject *obj, int idx) {{
  DevicePtrInfo ptr_info;
  ptr_info.dev_ptr = 0;
  ptr_info.valid = true;
  if (PyLong_Check(obj)) {{
    ptr_info.dev_ptr = reinterpret_cast<void *>(PyLong_AsUnsignedLongLong(obj));
    return ptr_info;
  }}
  if (obj == Py_None) {{
    // valid nullptr
    return ptr_info;
  }}
  PyObject *ptr = PyObject_GetAttrString(obj, "data_ptr");
  
  if(ptr){{
    PyObject *empty_tuple = PyTuple_New(0);
    PyObject *ret = PyObject_Call(ptr, empty_tuple, NULL);
    Py_DECREF(empty_tuple);
    Py_DECREF(ptr);
    if (!PyLong_Check(ret)) {{
      PyErr_SetString(PyExc_TypeError, "data_ptr method of Pointer object must return 64-bit int");
      ptr_info.valid = false;
      return ptr_info;
    }}
    ptr_info.dev_ptr = reinterpret_cast<void *>(PyLong_AsUnsignedLongLong(ret));
    if(!ptr_info.dev_ptr)
      return ptr_info;
    Py_DECREF(ret);  // Thanks ChatGPT!
    return ptr_info;
  }}
  PyErr_SetString(PyExc_TypeError, "Pointer argument must be either uint64 or have data_ptr method");
  return ptr_info;
}}

static PyObject* launch(PyObject* self, PyObject* args) {{
  int gridX, gridY, gridZ;
  PyObject *launch_enter_hook = NULL;
  PyObject *launch_exit_hook = NULL;
  PyObject *kernel_metadata = NULL;
  PyObject *launch_metadata = NULL;
  const char *kernel_elf_path = NULL;
  {' '.join([f"{_extracted_type(ty)} _arg{i}; " for i, ty in signature.items()])}
  if(!PyArg_ParseTuple(args, \"{format}\", &gridX, &gridY, &gridZ, &kernel_elf_path,
                                           &kernel_metadata, &launch_metadata,
                                           &launch_enter_hook, &launch_exit_hook, {args_list})) {{
    return NULL;
  }}

  // [CPULauncher-specific]: We don't need the metadata below but just put them
  // here anyway to be consistent with others.
  // This will make updating the driver easier in the future.

  //  int num_warps, num_ctas, shared_memory, clusterDimX, clusterDimY, clusterDimZ;
  //  if (!PyArg_ParseTuple(kernel_metadata, \"iiiiii\", &num_warps, &num_ctas, &shared_memory, &clusterDimX, &clusterDimY, &clusterDimZ)) {{
  //    PyErr_SetString(PyExc_TypeError, "kernel_metadata must be a tuple");
  //    return NULL;
  //  }}

  // extract launch metadata
  if (launch_enter_hook != Py_None){{
    PyObject* args = Py_BuildValue("(O)", launch_metadata);
    PyObject* ret = PyObject_CallObject(launch_enter_hook, args);
    Py_DECREF(args);
    if (!ret)
      return NULL;
  }}
  
  void *input_buf;
  unsigned int file_size;
  evModule_t module;
  const char *kernel_name = "{kernel_name}";
  const uint64_t mem_affinitymap = evAffinityMapDefault;
  const uint64_t kernel_affinitymap = evAffinityMapDefault;
  checkErrors(evSetDevice(0));
  ReadBinFile(kernel_elf_path, &input_buf, &file_size);
  // Logging before evModuleLoadData
  printf("[LOG] Calling evModuleLoadData\\n");
  checkErrors(evModuleLoadData(&module, input_buf, file_size));
  printf("[LOG] Finished evModuleLoadData\\n");

  evFunc hfunc;

  // Logging before evModuleGetFunction
  printf("[LOG] Calling evModuleGetFunction\\n");
  checkErrors(evModuleGetFunction(&hfunc, module, kernel_name));
  printf("[LOG] Finished evModuleGetFunction\\n");

  // Logging before checkNotEqual
  printf("[LOG] Checking if hfunc is not NULL\\n");
  checkNotEqual(hfunc, (evFunc)NULL);
  printf("[LOG] hfunc is valid\\n");

  
  // raise exception asap
  {"; ".join([f"DevicePtrInfo ptr_info{i} = getPointer(_arg{i}, {i}); if (!ptr_info{i}.valid) return NULL;" if ty[0] == "*" else "" for i, ty in params])};
  
  HOST_TO_DEVICE_PLACE_HOLDER
  int grid_args[3] = {{gridX, gridY, gridZ}};
  printf("[LOG] grid_args: %d, %d, %d\\n", grid_args[0], grid_args[1], grid_args[2]);
  void *kernel_args[{(len(params) + 3) * 2 + 1}] = {{ {", ".join([f"(void *)&(ptr_info{i}.dev_ptr), (void*)8" if ty[0]=="*" else f"(void*)(&_arg{i}), (void*)8" for i, ty in params])}, {", ".join([f"(void*)(&(grid_args[{i}])), (void*)8" for i in range(3)])}}};
  kernel_args[{(len(params) + 3) * 2}] = NULL;
  // Logging before evConfigureCall
  printf("[LOG] Calling evConfigureCall\\n");
  checkErrors(evConfigureCall({CLUSTER_NUM}, {CORE_NUM}, 0, kernel_affinitymap, evKernelFlushCache));
  printf("[LOG] Finished evConfigureCall with ClusterNum: %d, CoreNum: %d\\n", {CLUSTER_NUM}, {CORE_NUM});

  // Logging before evLaunchKernel
  printf("[LOG] Calling evLaunchKernel\\n");
  checkErrors(evLaunchKernel(&module, kernel_name, kernel_args));
  printf("[LOG] Finished evLaunchKernel\\n");

  DEVICE_TO_HOST_PLACE_HOLDER

  // Logging before evModuleUnload
  printf("[LOG] Calling evModuleUnload\\n");
  checkErrors(evModuleUnload(module));
  printf("[LOG] Finished evModuleUnload\\n");
  // Logging before evDeviceSynchronize
  printf("[LOG] Calling evDeviceSynchronize\\n");
  evDeviceSynchronize();
  printf("[LOG] Finished evDeviceSynchronize\\n");
  if (PyErr_Occurred()) {{
    return NULL;
  }}
  if(launch_exit_hook != Py_None){{
    PyObject* args = Py_BuildValue("(O)", launch_metadata);
    PyObject* ret = PyObject_CallObject(launch_exit_hook, args);
    Py_DECREF(args);
    if (!ret)
      return NULL;
  }}

  // return None
  Py_INCREF(Py_None);
  return Py_None;
}}

static PyMethodDef ModuleMethods[] = {{
  {{"launch", launch, METH_VARARGS, "Entry point for all kernels with this signature"}},
  {{NULL, NULL, 0, NULL}} // sentinel
}};

static struct PyModuleDef ModuleDef = {{
  PyModuleDef_HEAD_INIT,
  \"MODULE_NAME_PLACEHOLDER\",
  NULL, //documentation
  -1, //size
  ModuleMethods
}};

PyMODINIT_FUNC PyInit_MODULE_NAME_PLACEHOLDER(void) {{
  PyObject *m = PyModule_Create(&ModuleDef);
  if(m == NULL) {{
    return NULL;
  }}
  PyModule_AddFunctions(m, ModuleMethods);
  return m;
}}
"""

def get_sys_path(bin_name: str):
    # Get the path to the system clang++
    path = shutil.which(bin_name)
    if path is None:
        raise RuntimeError(bin_name + " not found")
    return path

class EvasLauncher(object):

    def __init__(self, src, metadata):
        kernel_name_placeholder = "KERNEL_NAME_PLACEHOLDER"
        constants = src.constants if hasattr(src, "constants") else dict()
        cst_key = lambda i: src.fn.arg_names.index(i) if isinstance(i, str) else i
        constants = {cst_key(key): value for key, value in constants.items()}
        signature = {cst_key(key): value for key, value in src.signature.items() if value != 'constexpr'}
        # constants has no use here and can be removed
        launcher_src = _generate_launcher(constants, signature, kernel_name_placeholder)
        # Later KERNEL_NAME_PLACEHOLDER will be used to assign the kernel name
        # in the following launch function.
        self.launch = compile_module(signature, constants, launcher_src, src.fn.arg_names, kernel_name_placeholder)

    def __call__(self, *args, **kwargs):
        self.launch(*args, **kwargs)


class EvasUtils(object):
    def __new__(cls):
        if not hasattr(cls, "instance"):
            cls.instance = super(EvasUtils, cls).__new__(cls)
        return cls.instance

    # Note:
    # nvidia and amd backends have their corresponding driver.c file that exposes
    # get_device_properties and load_binary using python bindings.
    # (see third_party/nvidia/backend/driver.c)
    # These methods are then used in compiler.py to initialize handles before running
    # the triton kernels.
    # Since we recompile the kernel every time (see compile_module above),
    # and the metadata generated by these functions aren't applicable to the cpu
    # backend, just define the same functions with dummy implementation.
    @staticmethod
    def get_device_properties(device):
        return {
            "max_shared_mem": 2**20,
            "multiprocessor_count": None,
            "sm_clock_rate": None,
            "mem_clock_rate": None,
            "mem_bus_width": None,
        }

    # Important note:
    # Since we cannot easy pass function pointers around, we pass along the
    # assembly source code so that compile_module above can recompile the
    # module every time.
    @staticmethod
    def load_binary(name, kernel_obj, shared, device):
        return (
            None,  # module
            kernel_obj,  # function
            None,  # n_regs
            4,  # n_spills
            sys.maxsize, # n_max_threads
        )


class EvasDriver(DriverBase):

    def __init__(self):
        super().__init__()
        self.utils = EvasUtils()
        self.launcher_cls = EvasLauncher
        self.binary_ext = "elf"  ## cc or elf??

    # CPU driver won't be automatically chosen unless explicitly set through
    # triton.runtime.driver.set_active(CPUDriver())
    @staticmethod
    def is_active():
        return True

    def map_python_to_cpp_type(self, ty: str) -> str:
        """
        Converts a Triton type string to its corresponding C++ type string for EV backend.
        """
        type_mapping = {
            "i1": "int8_t",
            "i8": "int8_t",
            "i16": "int16_t",
            "i32": "int32_t",
            "i64": "int64_t",
            "u1": "uint8_t",
            "u8": "uint8_t",
            "u16": "uint16_t",
            "u32": "uint32_t",
            "u64": "uint64_t",
            "fp16": "float",
            "bf16": "float",
            "fp32": "float",
            "f32": "float",
            "fp64": "double",
            "*fp32": "float*",
            "*fp16": "float*",
            "*i1": "int8_t*",
            "*i8": "int8_t*",
            "*i16": "int16_t*",
            "*i32": "int32_t*",
            "*i64": "int64_t*",
            "*u8": "uint8_t*"
        }
        return type_mapping.get(ty, "void*")

    def get_device_capability(self):
        return ("epu", 0)

    def get_active_torch_device(self):
        """
        Return the active torch device for this backend.
        For EV backend, we don't use torch, so return None.
        """
        return None

    def get_benchmarker(self):
        """
        Return the benchmarking function that this backend should use by default.
        For EV backend, we provide a simple benchmarker.
        """
        def simple_benchmarker(kernel_call, *, quantiles=None, **kwargs):
            # Simple benchmarker that just calls the kernel once
            # In a real implementation, you might want to call it multiple times
            # and measure performance
            result = kernel_call(**kwargs)
            if quantiles is None:
                quantiles = [0.5]  # Default to median
            return [0.0] * len(quantiles)  # Placeholder timing
        return simple_benchmarker

    def get_current_stream(self, device):
        return None

    def get_current_device(self):
        # CPU doesn't have a device to return. Return something.
        return "epu"

    def set_current_device(self, device):
        # CPU doesn't have a device to set
        assert device == "epu"
        return

    def get_current_target(self):
        return GPUTarget("epu", 0, 0)

    def assemble_tensormap_to_arg(self, tensormaps_info, args):
        return args


# EVAS Compiler


def _make_ttir(mod, metadata, opt):
    ttir_code = str(mod)
    pattern = r"tt\.func public @(.*?)(?=\()"
    matches = re.findall(pattern, ttir_code)
    metadata["name"] = matches[0]
    pm = ir.pass_manager(mod.context)
    pm.enable_debug()
    passes.common.add_inliner(pm)
    passes.ttir.add_rewrite_tensor_pointer(pm) ## triton 3.7 deleted
    passes.ttir.add_rewrite_tensor_descriptor_to_pointer(pm)
    passes.common.add_canonicalizer(pm)
    passes.ttir.add_combine(pm)
    passes.ttir.add_reorder_broadcast(pm)
    passes.common.add_cse(pm)
    passes.ttir.add_triton_licm(pm)
    passes.common.add_symbol_dce(pm)
    passes.ttir.add_loop_unroll(pm)
    pm.run(mod, "make_ttir")
    return mod.str()


def _get_evas_triton_opt_path() -> str:
    path = os.getenv("EVAS_TRITON_OPT_PATH", None)
    if path == None:
        raise Exception(
            "EVAS_TRITON_OPT_PATH is not set. Build FlagTree with the EVAS backend or set EVAS_TRITON_OPT_PATH."
        )
    return path



def ttshared_opt_command(src, dst, metadata):
    evas_triton_opt_path = _get_evas_triton_opt_path()
    ret = [
        evas_triton_opt_path,
        src,
        "--pass-pipeline=builtin.module(triton-to-evas)",
        "-o",
        dst,
    ]
    enable_ir_dump = os.environ.get("TRITON_KERNEL_DUMP", "0") == "1"
    if enable_ir_dump:
        fn_dump_manager = get_dump_manager(metadata["hash"])
        ret.extend(
            [
                "--mlir-print-ir-after-all",
                "--mlir-print-ir-tree-dir=" + fn_dump_manager.cache_dir + "/ttshared_ir",
            ]
        )
    return ret


def _ttir_to_ttsharedir(ttir_code, metadata):
    with tempfile.TemporaryDirectory() as tmpdir:
        src_path = os.path.join(tmpdir, "tt.mlir")
        dst_path = os.path.join(tmpdir, "ttshared.mlir")
        Path(src_path).write_text(ttir_code)
        cmd = ttshared_opt_command(src_path, dst_path, metadata)
        run_command(cmd)
        return Path(dst_path).read_text()


def _optimize_ttsharedir(ttsharedir: str):
    # We don't apply any optimizations now, but we can add passes if needed.
    return ttsharedir

def _rename_entry_func_to_kernel(ttsharedir: str, metadata) -> str:
    original_name = metadata.get("name")
    if original_name == "kernel":
        return ttsharedir
    if original_name:
        ttsharedir = re.sub(
            rf'(\bsym_name\s*=\s*)"{re.escape(original_name)}"',
            r'\1"kernel"',
            ttsharedir,
            count=1,
        )
        ttsharedir = re.sub(
            rf'(\bfunc\.func\s+(?:public\s+|private\s+)?@){re.escape(original_name)}\b',
            r'\1kernel',
            ttsharedir,
            count=1,
        )
    if 'sym_name = "kernel"' not in ttsharedir and "func.func @kernel" not in ttsharedir:
        ttsharedir = re.sub(r'(\bsym_name\s*=\s*)"[^"]+"', r'\1"kernel"', ttsharedir, count=1)
    return ttsharedir


def pack_header(ccsrc: str, metadata): 
    def _add_extern_device_to_kernel(ccsrc: str) -> str:
        # Find the line containing "MCU void kernel(" and add extern "C" __global__ prefix
        # Replace "MCU void kernel(" with "extern \"C\" __global__ MCU void kernel(" in the entire ccsrc
        ccsrc = re.sub(r'MCU\s+void\s+kernel\(', '__device__ MCU void kernel(', ccsrc)
        return ccsrc

    def _add_device_to_func_def(ccsrc: str) -> str:
        # Find all function definitions that are not "kernel" and add __device__ prefix
        # This regex matches function definitions like "void func_name(" or "int func_name(" etc.
        # but excludes the kernel function
        # Add __device__ only to function definitions that start with "MCU ALWAYS_INLINE" and are not kernel
        pattern = r'(?!.*__device__)(\bMCU.+\([^;{]*\)\s*{)'
        # matches = list(re.finditer(pattern, ccsrc, flags=re.MULTILINE))
        # for match in matches:
        #     print(match.group(0))
        ccsrc = re.sub(
            pattern,  # Match MCU ALWAYS_INLINE function def but not kernel
            r'__device__ \1',
            ccsrc,
            flags=re.MULTILINE
        )
        return ccsrc

    def _extract_kernel_function_args(ccsrc: str):
        """
        Extract the kernel function arguments (names and types) from C/C++ code.
        
        Args:
            cc_code (str): The C/C++ code string
            
        Returns:
            dict: A dictionary where keys are argument names and values are their types
                Returns None if no kernel function is found
        """
        # Pattern to match the kernel function declaration
        # Looks for "void kernel(" or "MCU void kernel(" or similar patterns
        kernel_pattern = r'(?:__global__\s+)?(?:extern\s+"C"\s+)?(?:MCU\s+)?(?:__device__\s+)?void\s+kernel\s*\((.*?)\)'
        
        # Find the kernel function
        kernel_match = re.search(kernel_pattern, ccsrc, re.DOTALL)
        
        if not kernel_match:
            raise RuntimeError("No kernel function found in the provided C/C++ code.")
        
        # Extract the arguments section
        args_section = kernel_match.group(1)
        
        # Handle empty arguments case
        if args_section.strip() == "":
            raise RuntimeError("Kernel function has no arguments.")
        
        # Split multiple arguments
        args = []
        
        # Handle complex arguments with nested commas (like template parameters)
        depth = 0
        current_arg = ""
        
        for char in args_section:
            if char == ',' and depth == 0:
                args.append(current_arg.strip())
                current_arg = ""
            else:
                if char == '<':
                    depth += 1
                elif char == '>':
                    depth -= 1
                current_arg += char
        
        # Add the last argument
        if current_arg.strip():
            args.append(current_arg.strip())
        
        # Process each argument to extract type and name
        arg_dict = []
        
        for arg in args:
            # Pattern to match type and name: everything before the last word is the type
            arg = arg.strip()
            
            # Skip empty arguments
            if not arg:
                continue
                
            # Handle pointers and references in names
            name_pattern = r'(\w+)(?:\s*\[\s*\w*\s*\])*\s*$'
            name_match = re.search(name_pattern, arg)
            
            if name_match:
                name = name_match.group(1)
                # Extract the type by removing the name from the end
                type_part = arg[:arg.rfind(name)].strip()
                
                # Clean up any trailing spaces, asterisks should stick with the type
                if type_part.endswith('*'):
                    while type_part.endswith(' *'):
                        type_part = type_part[:-2] + '*'
                
                arg_dict.append({"name": name, "type": type_part})
            else:
                # For arguments that don't match the expected pattern
                raise ValueError(f"Could not parse argument: '{arg}'")
    
        return arg_dict

    def _delete_MCU(ccsrc: str) -> str:
        # Replace all occurrences of "MCU" with ""
        return re.sub(r'\bMCU\s*', '', ccsrc)

    def _replace_wt_data_setction(ccsrc: str) -> str:
        # Replace all occurrences of ".wtdata" with ".data"
        return ccsrc.replace('.wtdata', '.data')

    def _delete_always_inline(ccsrc: str) -> str:
        # Replace all occurrences of "ALWAYS_INLINE" with ""
        return re.sub(r'\bALWAYS_INLINE\s*', '', ccsrc)

    kernel_name = metadata["name"]
    ccsrc = _add_device_to_func_def(ccsrc)
    ccsrc = _delete_MCU(ccsrc)
    #ccsrc = _delete_always_inline(ccsrc)
    ccsrc = _replace_wt_data_setction(ccsrc)
    args = _extract_kernel_function_args(ccsrc)
    outer_args = args[:-3]
    gridx, gridy, gridz = outer_args[-3:]
    return f"""
#ifdef __AC_DEVICE_COMPILE__
#include "visa_defs_v1.h"
#include "visa.h"
#include "operator/te/te.h"
#include "target/evamind/evamind-v1/visa_evamind_me.h"
#include "target/evamind/visa_evamind_te.h"
#include "target/evamind/visa_evamind_vcall.h"
#include "util/tuple.h"
#include "visa_matrix.h"

#include <npurt.h>
using namespace visa::te;
using namespace visa::matrix;
using namespace visa;
#endif

#include <stdint.h>
#include <helper_hpe.h>
{ccsrc}

__device__ int get_hw_core_id(int dim_x, int dim_y, int dim_z, int pid_x, int pid_y, int pid_z){{
    int total_size = dim_x * dim_y * dim_z;
    int total_core_num = {CORE_NUM} * {CLUSTER_NUM};
    // Calculate elements per core (rounded up)
    int elements_per_core = (total_size + total_core_num - 1) / total_core_num;
    // Calculate linear index
    int linear_index = pid_x + pid_y * dim_x + pid_z * dim_x * dim_y;
    // Calculate target core
    int hw_id = linear_index / elements_per_core;
    // Ensure we don't exceed available cores
    hw_id = hw_id % total_core_num;
    return hw_id;
}}


extern "C" __global__ void {kernel_name}({', '.join([f"{arg['type']} {arg['name']}" for arg in outer_args])}) {{
    int core_id = 0;
    int cluster_id = 0;
    for(int z = 0; z < {gridz['name']}; z++) {{
      for(int y = 0; y < {gridy['name']}; y++) {{
        for(int x = 0; x < {gridx['name']}; x++) {{
            int hw_core_id = get_hw_core_id({gridx['name']}, {gridy['name']}, {gridz['name']}, x, y, z);
            cluster_id = hw_core_id / {CORE_NUM};
            core_id = hw_core_id % {CORE_NUM};
            if (cluster_id == ClusterID && core_id == CoreID) {{
                printf("[KERNEL LOG]: run program with core_id: %d, cluster_id: %d\\n", core_id, cluster_id);
                kernel({', '.join([f"{arg['name']}" for arg in outer_args])}, x, y, z);
            }}
        }}
      }}
    }}
}} 
"""

def get_py_include_dir():
    # This function was renamed and made public in Python 3.10
    if hasattr(sysconfig, 'get_default_scheme'):
        scheme = sysconfig.get_default_scheme()
    else:
        scheme = sysconfig._get_default_scheme()
    # 'posix_local' is a custom scheme on Debian. However, starting Python 3.10, the default install
    # path changes to include 'local'. This change is required to use triton with system-wide python.
    if scheme == 'posix_local':
        scheme = 'posix_prefix'
    return sysconfig.get_paths(scheme=scheme)["include"]

def compile_mod_command(src_path: str, dst_path: str):
    return  [
        "evcc",
        "-O2",
        "-Wall",
        "-Werror",
        "-std=c++17",
        "-fno-omit-frame-pointer",
        "-Wno-unused-command-line-argument",
        "-Wno-implicitly-unsigned-literal",
        "-Wno-unused-variable",
        "-ftls-model=local-exec",
        "-fno-common",
        "-ffast-math",
        "-ffunction-sections",
        "-fdata-sections",
        "-Wl,--gc-sections",
        "-D__riscv_v_vlen=1024",
        # "--gcc-toolchain=/usr",
        "-shared", "-fPIC", "-o",
        dst_path,
        src_path,
        "-I",
        get_py_include_dir(), 
        "-I",
        os.path.join(os.path.dirname(__file__), "include/evas")
    ]
    
    
def cc_to_kernel_elf_command(src_path: str, dst_path: str):
    visa_dir = os.environ.get("VISA_PATH")
    if visa_dir is None:
        sdk_root = os.environ.get("EVAS_SDK_ROOT")
        if sdk_root:
            visa_dir = os.path.join(sdk_root, "application", "visa")
    if visa_dir is None:
        raise EnvironmentError("Please set VISA_PATH or EVAS_SDK_ROOT.")
    hgss_root = os.environ.get("RISCV_HGSS")
    return  [
        "evcc",
        "-O2",
        "-Wall",
        "-Werror",
        "-std=c++17",
        "-fno-omit-frame-pointer",
        "-Wno-unused-command-line-argument",
        "-Wno-implicitly-unsigned-literal",
        "-Wno-unused-variable",
        "-ftls-model=local-exec",
        "-fno-common",
        "-ffast-math",
        "-ffunction-sections",
        "-fdata-sections",
        "-Wl,--gc-sections",
        "-D__riscv_v_vlen=1024",
        "-D__evamind_v1",
        "-D__EVAMIND_DEVICE_V1",
        # "--gcc-toolchain=/usr",
        "--ac-device-only",
        "-emit-device-elf",
        "-x",
        "ac",
        "-o",
        dst_path,
        src_path,
        "-I",
        visa_dir + "/include",
        "-I",
        visa_dir + "/include/target/evamind/evamind-v1",
        *(
            ["-I", os.path.join(hgss_root, "misc", "include")]
            if hgss_root
            else []
        ),
        "-I",
        os.path.join(os.path.dirname(__file__), "include/evas")
    ]
   

def compile_module_from_src(cc: bytes):
    dst_name = "kernel"
    with tempfile.TemporaryDirectory() as tmpdir:
        src_path = os.path.join(tmpdir, dst_name + ".cc")
        Path(src_path).write_bytes(cc)
        dst_path = os.path.join(tmpdir, dst_name + ".so")
        command = compile_mod_command(src_path, dst_path)
        run_command(command)
        return Path(dst_path).read_bytes()

def compile_elf(cc: bytes, metadata):
    dst_name = "kernel"
    with tempfile.TemporaryDirectory() as tmpdir:
        src_path = os.path.join(tmpdir, dst_name + ".cc")
        Path(src_path).write_bytes(cc)
        dst_path = os.path.join(tmpdir, dst_name + ".elf")
        command = cc_to_kernel_elf_command(src_path, dst_path)
        run_command(command)
        return Path(dst_path).read_bytes()

def evofc_lib_dir():
    evofc_dir = os.environ.get("EVOFC_OPT_PATH")
    if evofc_dir is not None:
        evofc_lib_dir = os.environ.get("EVOFC_OPT_PATH")
    else:
        site_packages_path = site.getsitepackages()[0]
        evofc_lib_dir = os.path.join(site_packages_path, 'jax_plugins','evas_epu')
    try:
        if evofc_lib_dir == None or not os.path.exists(evofc_lib_dir):
            raise FileNotFoundError(
                f"Error: '{evofc_lib_dir}' does not exist, please set EVOFC_OPT_PATH."
            )
    except FileNotFoundError as e:
        print(e)
    return evofc_lib_dir


def ev_opt_command(src: str, dst: str, metadata):
    # visa-debug-print=1 visa-debug-core=0
    ev_opt = os.path.join(evofc_lib_dir(), "evofc-opt")
    hash_suffix = metadata["hash"][:5]
    incbin_dir = get_cache_manager(_cache_key(f"incbin/incbin_{hash_suffix}")).cache_dir
    passes = [
        "--operands-memory-reuse-setter",
        "--rewrite-reduce-copy",
        "--preprocess-copy-elimination",
        "--lower-affine",
        "--canonicalize",
        "--cse",
        "--rewrite-memref-cast",
        "--copy-elimination",
        "--cse",
        "--mc-op-collapse",
        "--ev-canonicalize",
        "--canonicalize",
        "--cse",
        "--rewrite-memref-copy",
        "--rewrite-slice-deslice-to-copy",
        "--ev-decompose-linalg-ops",
        "--ev-collapse-dimensions",
        "--rewrite-multi-dims-linalg",
        "--canonicalize",
        "--cse",
        "--linalg-named-to-visa=simt-level=simt",
        "--kernel-to-visa",
        "--canonicalize",
        "--cse",
        "--visa-te-canonicalizer",
        "--canonicalize",
        "--cse",
        "--rewrite-matmul-with-batch",
        "--ev-collapse-dimensions-visa",
        "--rewrite-visa-shape-and-stride",
        "--rewrite-multi-dims-visa",
        "--rewrite-visa-deform-conv2d",
        "--rewrite-visa-conv2d-layout",
        "--rewrite-visa-conv2d-to-matmul",
        "--rewrite-visa-conv2d-kernel-hw",
        "--canonicalize",
        "--cse",
        "--rewrite-strided-visa",
        "--canonicalize",
        "--cse",
        "--inject-sync=simt-level=simt",
        f"--visa-to-emitc=simt-level=simt incbin-dir={incbin_dir}",
        "--inject-memref-transfer-func",
        "--canonicalize",
        "--cse",
        "--func-to-emitc",
        "--canonicalize",
        "--cse",
        "--ev-scf-to-emitc=simplify-line-break=true",
        "--form-expressions",
    ]
    ev_opt_command = [
        ev_opt,
        *passes,
        src,
        "-o",
        dst,
    ]
    enable_ir_dump = os.environ.get("TRITON_KERNEL_DUMP", "0") == "1"
    if enable_ir_dump:
        fn_dump_manager = get_dump_manager(metadata["hash"])
        ev_opt_command.extend(
            [
                "--mlir-print-ir-after-all",
                "--mlir-print-ir-tree-dir=" + fn_dump_manager.cache_dir + "/evopt_ir",
            ]
        )
    return ev_opt_command

def ev_trans_command(src: str, dst: str):
    ev_trans = os.path.join(evofc_lib_dir(), "evofc-translate")
    ev_trans_command = [ev_trans, "-mlir-to-cpp", src, "-o", dst]
    return ev_trans_command


   

def _ttshared_to_cc(ttsharedir: str, metadata):
    with tempfile.TemporaryDirectory() as tmpdir:
        ### stub code for debugging
        src_path = os.path.join(tmpdir, "ttshared.mlir")
        Path(src_path).write_text(_rename_entry_func_to_kernel(ttsharedir, metadata))
        emitc_path = os.path.join(tmpdir, "emitc.mlir")
        cc_path = os.path.join(tmpdir, "dst.cc")

        run_command(ev_opt_command(src_path, emitc_path, metadata))
        run_command(ev_trans_command(emitc_path, cc_path))
        return pack_header(Path(cc_path).read_text(), metadata)

def _cc_to_evbin(ccsrc: str, metadata):
    return compile_elf(ccsrc.encode("utf-8"), metadata)


#### Add EVBackend here
@dataclass(frozen=True)
class EvasOptions:
    debug: bool = False
    arch: str = None
    num_warps: int = 0
    num_ctas: int = 0
    num_stages: int = 1
    instrumentation_mode: str = ""
    one_tile_per_cta: bool = False
    enable_warp_specialization: bool = False
    enable_fp_fusion: bool = False
    extern_libs = None
    cluster_dims: tuple = (1, 1, 1)
    shared: bool = False
    allow_fp8e4nv: bool = False
    allowed_dot_input_precisions: Tuple[str] = ("ieee",)
    # TODO: just for lowering to extern elementwise op (libdevice.py)
    # some operations like libdevice.rsqrt and so on
    backend_name = "cuda"
    to_ttsharedir: bool = False
    sanitize_overflow: bool = True
    # Disable FP8 here since this is a sample CPU backend.
    # Target specific backends can eanble it with supported types.
    supported_fp8_dtypes: Tuple[str] = ()
    ir_override: str = None
    def __post_init__(self):
        pass

    def hash(self):
        key = "_".join([f"{name}-{val}" for name, val in self.__dict__.items()])
        return hashlib.sha256(key.encode("utf-8")).hexdigest()


class EvasBackend(BaseBackend):
    binary_ext = "elf"

    @staticmethod
    def supports_target(target: GPUTarget):
        return target.backend == "epu"

    def __init__(self, target: GPUTarget) -> None:
        super().__init__(target)

    def parse_options(self, opts) -> Any:
        args = {"arch": self.target.arch}
        args.update(
            {k: opts[k] for k in EvasOptions.__dataclass_fields__.keys() if k in opts}
        )
        return EvasOptions(**args)

    def get_codegen_implementation(self, options):
        codegen_fns = {"min_dot_size": lambda lhsType, rhsType: (1, 1, 1)}
        return codegen_fns

    def pack_metadata(self, metadata):
        # Note: We actually don't need any of these except for the name which is
        # used in the launch function in driver.py. Putting these in so we're
        # consistent with other backends
        return (
            metadata.num_warps,
            metadata.num_ctas,
            metadata.shared,
            metadata.cluster_dims[0],
            metadata.cluster_dims[1],
            metadata.cluster_dims[2],
            metadata.name,
        )

    # Our compilation pipeline isn't in python like nvidia or amd, no need to load
    # dialects. See `triton_shared.cc`
    def load_dialects(self, ctx):
        if triton_shared is not None and hasattr(triton_shared, "load_dialects"):
            triton_shared.load_dialects(ctx)
            return
        libtriton.evas.load_dialects(ctx)
        return

    def add_stages(self, stages, options, language=None):
        stages["ttir"] = lambda src, metadata: _make_ttir(src, metadata, options)
        if options.to_ttsharedir:
            stages["elf"] = lambda src, metadata: _optimize_ttsharedir(
                _ttir_to_ttsharedir(src, metadata)
            )    
        else:
            stages["ttsharedir"] = lambda src, metadata: _optimize_ttsharedir(
                _ttir_to_ttsharedir(src, metadata)
            )
            stages["cc"] = lambda src, metadata: _ttshared_to_cc(src, metadata)
            stages["elf"] = lambda src, metadata: _cc_to_evbin(src, metadata)

    def get_module_map(self) -> Dict[str, ModuleType]:
        """
        Return a map of interface modules to their device-specific implementations
        """
        # TODO: maybe no need to add module mapping for EV backend
        return {}


    @functools.lru_cache()
    def hash(self):
        return self.target
