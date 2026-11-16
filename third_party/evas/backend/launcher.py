# Copyright 2025-     FlagOS Contributors
# SPDX-License-Identifier: MIT
"""SDK launcher generation, CPU/device staging and module lifetime management."""

import hashlib
import os
import tempfile
import sysconfig
import importlib.util
from pathlib import Path
import logging
import re
import threading
import operator
from collections import OrderedDict

from triton.runtime.cache import get_cache_manager
from .utils import run_command
from .local_shards import local_signature, make_buffer_plan

CLUSTER_NUM = 4
CORE_NUM = 4
# Bound retained SDK modules independently of Triton's disk cache.
MODULE_CACHE_SIZE = 128
_module_cache = OrderedDict()
_active_modules = {}
_module_cache_lock = threading.RLock()


def _evict_module():
    # Callers hold the lock. Hooks may launch kernels recursively, so do not
    # evict a module until its outermost launch (including hooks) has returned.
    for key, mod in _module_cache.items():
        if key not in _active_modules:
            mod.unload()
            del _module_cache[key]
            return True
    return False


def _launch_cached_module(key, name, cache_path, elf, args, device_index=0):
    # Bound device modules, independently of Triton's on-disk compile cache.
    # TODO: extend retention to other devices; keep their load/launch/unload path.
    use_cache = device_index == 0
    limit = operator.index(MODULE_CACHE_SIZE) if use_cache else 0
    if limit < 0:
        raise ValueError("MODULE_CACHE_SIZE must be non-negative")
    with _module_cache_lock:
        mod = _module_cache.get(key) if use_cache else None
        if mod is None:
            while use_cache and len(_module_cache) >= max(1, limit):
                if not _evict_module():
                    break
            spec = importlib.util.spec_from_file_location(name, cache_path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            with tempfile.TemporaryDirectory() as tmpdir:
                elf_path = os.path.join(tmpdir, "kernel.elf")
                Path(elf_path).write_bytes(elf)
                # TODO: account for module sizes and per-die capacity; validate
                # SDK OOM recovery before adding load retries.
                mod.load(elf_path)
            if use_cache:
                _module_cache[key] = mod
        if use_cache:
            _module_cache.move_to_end(key)
        _active_modules[key] = _active_modules.get(key, 0) + 1
        try:
            return mod.launch(*args)
        finally:
            _active_modules[key] -= 1
            if not _active_modules[key]:
                del _active_modules[key]
            if not use_cache and key not in _active_modules:
                mod.unload()
            # Zero disables retention; also trim after nested launches finish.
            while use_cache and len(_module_cache) > limit:
                if not _evict_module():
                    break


def _ty_to_cpp(ty):
    # IRSource uses MLIR's f32 spelling; ASTSource uses Triton's fp32.
    ty = re.sub(r"^(\*?)f(16|32|64)$", r"\1fp\2", ty)
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
        # Match EvoFC's byte ABI for the FN payload; decoding stays in the kernel.
        "*fp8e4nv": "uint8_t*",
        "*i1": "int8_t*",
        "*i8": "int8_t*",
        "*i16": "int16_t*",
        "*i32": "int32_t*",
        "*i64": "int64_t*",
        "*u1": "int8_t*",
        "*u8": "int8_t*",
        "*bf16": "bfloat16_t*",
    }[ty]


def compile_module(signature, constants, launcher_src, arg_names, launch_call_placeholder, use_local_abi=False):
    params = list(signature.items())
    launch_signature = local_signature(signature) if use_local_abi else signature

    def gen_args(signature, args):
        args_desc = []
        for index, (arg_idx, ty) in enumerate(signature.items()):
            arg = args[index]
            is_pointer = ty[0] == "*"
            is_tensor = hasattr(arg, 'numel') and hasattr(arg, 'element_size')
            # IRSource has positional types but no Python argument names. Its
            # caller must supply device pointers: guessing output roles would
            # silently omit CPU copy-back for an output buffer.
            if arg_names is None and is_pointer and is_tensor and arg.device.type == "cpu":
                raise ValueError("IRSource launches require device pointers; CPU tensor output roles are unknown")
            is_output = arg_names is not None and arg_idx < len(arg_names) and arg_names[arg_idx].startswith("out")
            args_desc.append({
                "tensor": arg, "size": arg.numel() * arg.element_size() if is_pointer and is_tensor else None,
                "arg_idx": arg_idx, "dtype": arg.dtype if is_pointer and is_tensor else None, "arg_type":
                _ty_to_cpp(ty), "is_pointer": is_pointer, "addr": arg.data_ptr() if is_pointer and is_tensor else None,
                "is_output": is_output, "is_cpu": arg.device.type == "cpu" if is_pointer and is_tensor else None
            })
        return args_desc

    def normalize_args(args):
        if len(args) == len(params):
            return list(args)
        return [arg for i, arg in enumerate(args) if i in signature.keys()]

    def pack_launcher(args_desc, launcher_src, kernel_name):
        packed = launcher_src
        host_to_device_code = ' '.join(
            f'void* host_ptr{i} = ptr_info{i}.dev_ptr; {desc["arg_type"]} device_ptr{i}; evMalloc((void **)&device_ptr{i}, {desc["size"]}, mem_affinitymap); evMemcpy(device_ptr{i}, host_ptr{i}, {desc["size"]}, evMemcpyHostToDevice, mem_affinitymap); printf("[LOG] Copy from host %p to device %p\\n", host_ptr{i}, device_ptr{i}); ptr_info{i}.dev_ptr = (void*)device_ptr{i};'
            for i, desc in enumerate(args_desc)
            if desc['is_pointer'] and desc['is_cpu'])
        device_to_host_code = ' '.join(
            f'evMemcpy(host_ptr{i}, device_ptr{i}, {desc["size"]}, evMemcpyDeviceToHost, mem_affinitymap); printf("[LOG] Copy from device %p to host %p\\n", device_ptr{i}, host_ptr{i});'
            for i, desc in enumerate(args_desc)
            if desc['is_pointer'] and desc['is_output'] and desc['is_cpu'])
        return packed.replace("HOST_TO_DEVICE_PLACE_HOLDER", host_to_device_code).replace(
            "DEVICE_TO_HOST_PLACE_HOLDER", device_to_host_code).replace(launch_call_placeholder, kernel_name)

    def launch_bound(
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
        args = list(args)
        if len(args) != len(launch_signature):
            raise ValueError("EVAS bound launcher argument count disagrees with its ABI")
        args_desc = gen_args(launch_signature, args)
        device_indices = {
            desc["tensor"].device.index
            for desc in args_desc
            if desc["is_pointer"] and desc["is_cpu"] is False and desc["tensor"].device.index is not None
        }
        if len(device_indices) > 1:
            raise RuntimeError(f"kernel arguments use multiple devices: {sorted(device_indices)}")
        device_index = next(iter(device_indices), 0)
        src = pack_launcher(args_desc, launcher_src, kernel_name)
        # Separate extension state so uncached devices cannot overwrite a
        # cached module belonging to device 0.
        src = src.replace("DEVICE_INDEX_PLACEHOLDER", str(device_index))
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

        return _launch_cached_module(key, name, cache_path, asm_src,
                                     (gridX, gridY, gridZ, kernel_metadata, launch_metadata, launch_enter_hook,
                                      launch_exit_hook, device_index, *args), device_index=device_index)

    def launch(gridX, gridY, gridZ, stream, cu_function, kernel_metadata, launch_metadata, launch_enter_hook,
               launch_exit_hook, *args):
        automatic = len(kernel_metadata) > 8 and kernel_metadata[8]
        if not automatic:
            return launch_bound(gridX, gridY, gridZ, stream, cu_function, kernel_metadata, launch_metadata,
                                launch_enter_hook, launch_exit_hook, *normalize_args(args))
        # Validate before allocation, copying or hooks; a zero grid is a no-op.
        dimensions = tuple(operator.index(n) for n in (gridX, gridY, gridZ))
        if any(n < 0 for n in dimensions):
            raise ValueError("EVAS grid dimensions must be non-negative")
        if any(n > 2147483647 for n in dimensions):
            raise OverflowError("EVAS grid dimension exceeds i32")
        if not all(dimensions):
            return None
        if dimensions[0] * dimensions[1] * dimensions[2] > 2147483647 - 16:
            raise OverflowError("EVAS grid exceeds the i32 scheduling range")
        if not kernel_metadata[7]:
            raise ValueError("automatic dual-die code requires a dual-die launch")
        contract = kernel_metadata[9] if len(kernel_metadata) > 9 else None
        local_access = kernel_metadata[10] if len(kernel_metadata) > 10 else None
        if bool(local_access and local_access.get("abi")) != use_local_abi:
            raise ValueError("EVAS local-window launcher ABI mismatch; recompile the kernel")
        if launch_enter_hook is not None:
            launch_enter_hook(launch_metadata)
        # Hooks observe the caller's tensors, before gather and after scatter.
        with make_buffer_plan(signature, normalize_args(args), contract, local_access, dimensions) as plan:
            if hasattr(plan, "local_report"):
                logging.getLogger(__name__).debug("EVAS buffer plan: %s", plan.local_report)
            result = launch_bound(*dimensions, stream, cu_function, kernel_metadata, launch_metadata, None, None,
                                  *plan.bindings)
        if launch_exit_hook is not None:
            launch_exit_hook(launch_metadata)
        return result

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
    format = "iiiOOOO" + "i" + args_format

    args_list = ', '.join(f"&_arg{i}" for i, ty in signature.items()) if len(signature) > 0 else ''

    params = [(i, ty) for i, ty in signature.items() if i not in constants]
    slot_copies = []
    for slot, (i, ty) in enumerate(params):
        value = f"ptr_info{i}.dev_ptr" if ty[0] == "*" else f"_arg{i}"
        slot_copies.append(f'  static_assert(sizeof({value}) <= sizeof(kernel_slots[{slot}]), '
                           '"kernel argument exceeds SDK slot");\n'
                           f'  memcpy(&kernel_slots[{slot}], &({value}), sizeof({value}));')
    slot_setup = "\n".join(slot_copies)
    return f"""
// Python defines feature-test macros and must precede system/runtime headers.
#include <Python.h>
#include <assert.h>
#include <stdbool.h>
#if __has_include(<hpe.h>)
#include <hpe.h>
#else
#include <ev_runtime.h>
#endif
#include <stdint.h>
#include <stdio.h>
#include <unistd.h>
#include <helper_hpe.h>

static bool module_loaded = false;
static bool module_ready = false;
static evModule_t module;
static void *module_buffer = NULL;
static const int module_device = DEVICE_INDEX_PLACEHOLDER;
static const char *kernel_name = "{kernel_name}";

static PyObject* runtime_error(evError_t status, const char *operation) {{
  PyErr_Format(PyExc_RuntimeError, "%s failed: %d (%s)", operation,
               (int)status, evGetErrorName(status));
  return NULL;
}}

static PyObject* load(PyObject* self, PyObject* path) {{
  if (module_loaded && module_ready)
    Py_RETURN_NONE;
  if (module_loaded) {{
    PyErr_SetString(PyExc_RuntimeError, "Previous module load failed; unload before retrying");
    return NULL;
  }}
  const char *kernel_elf_path = PyUnicode_AsUTF8(path);
  if (!kernel_elf_path)
    return NULL;
  evError_t status = evSetDevice(module_device);
  if (status != evSuccess)
    return runtime_error(status, "evSetDevice");
  void *input_buf = NULL;
  unsigned int file_size = 0;
  if (ReadBinFile(kernel_elf_path, &input_buf, &file_size) != 0) {{
    free(input_buf);
    PyErr_SetString(PyExc_RuntimeError, "Failed to read kernel ELF");
    return NULL;
  }}
  printf("[LOG] Calling evModuleLoadData\\n");
  status = evModuleLoadData(&module, input_buf, file_size);
  if (status != evSuccess) {{
    free(input_buf);
    module = NULL;
    if (status == evErrorDeviceOutOfMemory) {{
      PyErr_SetString(PyExc_MemoryError, "evModuleLoadData: device module memory exhausted");
      return NULL;
    }}
    return runtime_error(status, "evModuleLoadData");
  }}
  printf("[LOG] Finished evModuleLoadData\\n");
  evFunc hfunc;
  status = evModuleGetFunction(&hfunc, module, kernel_name);
  if (status != evSuccess || hfunc == (evFunc)NULL) {{
    evError_t cleanup_status = evModuleUnload(module);
    if (cleanup_status != evSuccess) {{
      module_buffer = input_buf;
      module_loaded = true;
      return runtime_error(cleanup_status, "evModuleUnload after failed entry lookup");
    }}
    free(input_buf);
    module = NULL;
    if (status != evSuccess)
      return runtime_error(status, "evModuleGetFunction");
    PyErr_SetString(PyExc_RuntimeError, "Kernel entry point is NULL");
    return NULL;
  }}
  // Retain the ELF buffer until unload, including runtimes that borrow it.
  module_buffer = input_buf;
  module_loaded = true;
  module_ready = true;
  Py_RETURN_NONE;
}}

static PyObject* unload(PyObject* self, PyObject* unused) {{
  if (module_loaded) {{
    int previous_device;
    evError_t status = evGetDevice(&previous_device);
    if (status != evSuccess)
      return runtime_error(status, "evGetDevice");
    status = evSetDevice(module_device);
    if (status != evSuccess)
      return runtime_error(status, "evSetDevice");
    status = evDeviceSynchronize();
    if (status != evSuccess) {{
      evSetDevice(previous_device);
      return runtime_error(status, "evDeviceSynchronize");
    }}
    printf("[LOG] Calling evModuleUnload\\n");
    status = evModuleUnload(module);
    evError_t restore_status = evSetDevice(previous_device);
    if (status != evSuccess)
      return runtime_error(status, "evModuleUnload");
    module_loaded = false;
    module_ready = false;
    module = NULL;
    free(module_buffer);
    module_buffer = NULL;
    if (restore_status != evSuccess)
      return runtime_error(restore_status, "restore previous device");
  }}
  Py_RETURN_NONE;
}}

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
    ptr_info.valid = !PyErr_Occurred();
    return ptr_info;
  }}
  if (obj == Py_None) {{
    // valid nullptr
    return ptr_info;
  }}
  PyObject *ptr = PyObject_GetAttrString(obj, "data_ptr");

  if(ptr){{
    PyObject *ret = PyObject_CallNoArgs(ptr);
    Py_DECREF(ptr);
    if (!ret) {{
      ptr_info.valid = false;
      return ptr_info;
    }}
    if (!PyLong_Check(ret)) {{
      Py_DECREF(ret);
      PyErr_SetString(PyExc_TypeError, "data_ptr method of Pointer object must return 64-bit int");
      ptr_info.valid = false;
      return ptr_info;
    }}
    ptr_info.dev_ptr = reinterpret_cast<void *>(PyLong_AsUnsignedLongLong(ret));
    ptr_info.valid = !PyErr_Occurred();
    Py_DECREF(ret);
    return ptr_info;
  }}
  PyErr_SetString(PyExc_TypeError, "Pointer argument must be either uint64 or have data_ptr method");
  ptr_info.valid = false;
  return ptr_info;
}}

static PyObject* launch(PyObject* self, PyObject* args) {{
  int gridX, gridY, gridZ;
  PyObject *launch_enter_hook = NULL;
  PyObject *launch_exit_hook = NULL;
  PyObject *kernel_metadata = NULL;
  PyObject *launch_metadata = NULL;
  int device_index;
  {' '.join([f"{_extracted_type(ty)} _arg{i}; " for i, ty in signature.items()])}
  if(!PyArg_ParseTuple(args, \"{format}\", &gridX, &gridY, &gridZ,
                                           &kernel_metadata, &launch_metadata,
                                           &launch_enter_hook, &launch_exit_hook,
                                           &device_index, {args_list})) {{
    return NULL;
  }}
  if (gridX < 0 || gridY < 0 || gridZ < 0) {{
    PyErr_SetString(PyExc_ValueError, "EVAS grid dimensions must be non-negative");
    return NULL;
  }}
  if (gridX == 0 || gridY == 0 || gridZ == 0)
    Py_RETURN_NONE;
  // AutoBlockify uses i32 induction variables; leave room for the last step.
  int64_t programs = 1;
  const int dimensions[3] = {{gridX, gridY, gridZ}};
  for (int dim : dimensions) {{
    if (programs > (INT32_MAX - 16) / dim) {{
      PyErr_SetString(PyExc_OverflowError, "EVAS grid exceeds the i32 scheduling range");
      return NULL;
    }}
    programs *= dim;
  }}
  if (!module_loaded || !module_ready) {{
    PyErr_SetString(PyExc_RuntimeError, "Kernel module is not loaded");
    return NULL;
  }}

  int enable_dual_die = 0;
  if (PyTuple_Check(kernel_metadata) && PyTuple_Size(kernel_metadata) > 7) {{
    enable_dual_die = PyObject_IsTrue(PyTuple_GetItem(kernel_metadata, 7));
    if (enable_dual_die < 0)
      return NULL;
  }}
  int cluster_num = enable_dual_die ? {CLUSTER_NUM * 2} : {CLUSTER_NUM};
  if (PyTuple_Check(kernel_metadata) && PyTuple_Size(kernel_metadata) > 8) {{
    int automatic = PyObject_IsTrue(PyTuple_GetItem(kernel_metadata, 8));
    if (automatic < 0)
      return NULL;
    if (automatic && !enable_dual_die) {{
      PyErr_SetString(PyExc_ValueError, "automatic dual-die code requires a dual-die launch");
      return NULL;
    }}
  }}
  // extract launch metadata
  if (launch_enter_hook != Py_None){{
    PyObject* args = Py_BuildValue("(O)", launch_metadata);
    PyObject* ret = PyObject_CallObject(launch_enter_hook, args);
    Py_DECREF(args);
    if (!ret)
      return NULL;
    Py_DECREF(ret);
  }}

  const uint64_t mem_affinitymap = evAffinityMapDefault;
  const uint64_t kernel_affinitymap = evAffinityMapDefault;
  printf("[LOG] device_index: %d\\n", device_index);
  evError_t launch_status = evSetDevice(device_index);
  if (launch_status != evSuccess)
    return runtime_error(launch_status, "evSetDevice");


  // raise exception asap
  {"; ".join([f"DevicePtrInfo ptr_info{i} = getPointer(_arg{i}, {i}); if (!ptr_info{i}.valid) return NULL;" if ty[0] == "*" else "" for i, ty in params])};

  HOST_TO_DEVICE_PLACE_HOLDER
  int grid_args[3] = {{gridX, gridY, gridZ}};
  printf("[LOG] grid_args: %d, %d, %d\\n", grid_args[0], grid_args[1], grid_args[2]);
  // The SDK copies complete 8-byte words, including narrow scalar/grid
  // arguments. Copy their object representation into owned, zero-filled
  // slots; never advertise eight readable bytes at a smaller object's address.
  uint64_t kernel_slots[{len(params) + 3}] = {{0}};
{slot_setup}
  for (int i = 0; i < 3; ++i)
    memcpy(&kernel_slots[{len(params)} + i], &grid_args[i], sizeof(grid_args[i]));
  void *kernel_args[{(len(params) + 3) * 2 + 1}] = {{ {", ".join([f"(void *)&kernel_slots[{i}], (void *)sizeof(kernel_slots[{i}])" for i in range(len(params) + 3)])} }};
  kernel_args[{(len(params) + 3) * 2}] = NULL;
  // Logging before evConfigureCall
  printf("[LOG] Calling evConfigureCall\\n");
  launch_status = evConfigureCall(cluster_num, {CORE_NUM}, 0, kernel_affinitymap, evKernelFlushCache);
  if (launch_status != evSuccess)
    return runtime_error(launch_status, "evConfigureCall");
  printf("[LOG] Finished evConfigureCall with ClusterNum: %d, CoreNum: %d\\n", cluster_num, {CORE_NUM});

  // Logging before evLaunchKernel
  printf("[LOG] Calling evLaunchKernel\\n");
  launch_status = evLaunchKernel(&module, kernel_name, kernel_args);
  if (launch_status != evSuccess)
    return runtime_error(launch_status, "evLaunchKernel");
  printf("[LOG] Finished evLaunchKernel\\n");

  // Logging before evDeviceSynchronize
  printf("[LOG] Calling evDeviceSynchronize\\n");
  launch_status = evDeviceSynchronize();
  if (launch_status != evSuccess)
    return runtime_error(launch_status, "evDeviceSynchronize");
  printf("[LOG] Finished evDeviceSynchronize\\n");
  DEVICE_TO_HOST_PLACE_HOLDER
  // The Python cache decides when this module can be evicted.
  if (PyErr_Occurred()) {{
    return NULL;
  }}
  if(launch_exit_hook != Py_None){{
    PyObject* args = Py_BuildValue("(O)", launch_metadata);
    PyObject* ret = PyObject_CallObject(launch_exit_hook, args);
    Py_DECREF(args);
    if (!ret)
      return NULL;
    Py_DECREF(ret);
  }}

  // return None
  Py_INCREF(Py_None);
  return Py_None;
}}

static PyMethodDef ModuleMethods[] = {{
  {{"load", load, METH_O, "Load the device module"}},
  {{"unload", unload, METH_NOARGS, "Unload the device module"}},
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


class EvasLauncher(object):

    def __init__(self, src, metadata):
        kernel_name_placeholder = "KERNEL_NAME_PLACEHOLDER"
        constants = src.constants if hasattr(src, "constants") else dict()
        arg_names = src.fn.arg_names if hasattr(src, "fn") else None
        cst_key = lambda i: arg_names.index(i) if isinstance(i, str) else i
        constants = {cst_key(key): value for key, value in constants.items()}
        signature = {cst_key(key): value for key, value in src.signature.items() if value != 'constexpr'}
        local_access = getattr(metadata, "local_access", None)
        use_local_abi = bool(local_access and local_access.get("abi"))
        # Hidden argument indices can overlap removed trailing constexprs.
        # The extended signature already excludes those original constants.
        launcher_src = _generate_launcher({} if use_local_abi else constants,
                                          local_signature(signature) if use_local_abi else signature,
                                          kernel_name_placeholder)
        # Later KERNEL_NAME_PLACEHOLDER will be used to assign the kernel name
        # in the following launch function.
        self.launch = compile_module(signature, constants, launcher_src, arg_names, kernel_name_placeholder,
                                     use_local_abi)

    def __call__(self, *args, **kwargs):
        self.launch(*args, **kwargs)


def get_py_include_dir():
    scheme = sysconfig.get_default_scheme()
    # 'posix_local' is a custom scheme on Debian. However, starting Python 3.10, the default install
    # path changes to include 'local'. This change is required to use triton with system-wide python.
    if scheme == 'posix_local':
        scheme = 'posix_prefix'
    return sysconfig.get_paths(scheme=scheme)["include"]


def compile_mod_command(src_path: str, dst_path: str):
    # CGSL SDK clang may need an explicit host GCC installation directory.
    gcc_install_dir = os.environ.get("EVAS_GCC_INSTALL_DIR")
    hgss_root = os.environ.get("RISCV_HGSS")
    return [
        "evcc", *([f"--gcc-install-dir={gcc_install_dir}"] if gcc_install_dir else []), "-O2", "-Wall", "-Werror",
        "-std=c++17", "-fno-omit-frame-pointer", "-Wno-unused-command-line-argument",
        "-Wno-implicitly-unsigned-literal", "-Wno-unused-variable", "-ftls-model=local-exec", "-fno-common",
        "-ffast-math", "-ffunction-sections", "-fdata-sections", "-Wl,--gc-sections", "-D__riscv_v_vlen=1024",
        "-shared", "-fPIC", "-o", dst_path, src_path, "-I",
        get_py_include_dir(), *(["-I", os.path.join(hgss_root, "include")] if hgss_root else []), "-I",
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
