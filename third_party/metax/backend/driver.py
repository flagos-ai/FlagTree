import functools
import os
import hashlib
import subprocess
import tempfile
from pathlib import Path
from triton.runtime.build import compile_module_from_src
from triton.runtime import _allocation
from triton.backends.compiler import GPUTarget
from triton.backends.driver import GPUDriver

dirname = os.path.dirname(os.path.realpath(__file__))
include_dir = [os.path.join(dirname, "include")]
libdevice_dir = os.path.join(dirname, "lib")
libraries = []


@functools.lru_cache()
def maca_home_dirs():
    return os.getenv("MACA_PATH")


@functools.lru_cache()
def libmaca_dirs():
    maca_path = maca_home_dirs()
    return ["{}/lib/".format(maca_path)]


@functools.lru_cache()
def hdrmaca_dirs():
    maca_path = maca_home_dirs()
    return [os.path.join(maca_path, "include")]


@functools.lru_cache()
def library_dirs():
    return [libdevice_dir, *libmaca_dirs()]


# ------------------------
# Utils
# ------------------------
class MacaUtils(object):

    def __new__(cls):
        if not hasattr(cls, "instance"):
            cls.instance = super(MacaUtils, cls).__new__(cls)
        return cls.instance

    def __init__(self):
        mod = compile_module_from_src(src=Path(os.path.join(dirname, "driver.c")).read_text(), name="maca_utils",
                                      library_dirs=library_dirs(), include_dirs=hdrmaca_dirs(), libraries=libraries,
                                      ccflags=["-D__MACA__", "-lmcruntime"])

        self.load_binary = mod.load_binary
        self.get_device_properties = mod.get_device_properties
        self.set_printf_fifo_size = mod.set_printf_fifo_size


# ------------------------
# Launcher
# ------------------------
def ty_to_cpp(ty):
    if ty[0] == '*':
        return "mcDeviceptr_t"
    return {
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
        "fp16": "double",
        "bf16": "double",
        "fp32": "double",
        "f32": "double",
        "fp64": "double",
        "nvTmaDesc": "CUtensorMap",
    }[ty]


FLOAT_STORAGE_TYPE = {
    "fp16": "uint16_t",
    "bf16": "uint16_t",
    "fp32": "uint32_t",
    "f32": "uint32_t",
    "fp64": "uint64_t",
}
FLOAT_PACK_FUNCTION = {
    "fp16": "pack_fp16",
    "bf16": "pack_bf16",
    "fp32": "pack_fp32",
    "f32": "pack_fp32",
    "fp64": "pack_fp64",
}

_BASE_ARGS_FORMAT = "iiiKKOOOOOO"
_BASE_ARGS_FORMAT_LEN = len(_BASE_ARGS_FORMAT)


def make_launcher(constants, signature, ids):

    def _expand_signature(signature):
        output = []
        tensordesc_idx = 0
        # Expand tensor descriptor arguments into either nvTmaDesc, shape and
        # strides, or base pointer, shape and strides depending on whether the
        # kernel was lowered to use the nvTmaDesc or not.
        for sig in signature:
            if isinstance(sig, str) and sig.startswith("tensordesc"):
                meta = tensordesc_meta[tensordesc_idx] if tensordesc_meta else None
                tensordesc_idx += 1

                match = re.match("tensordesc<([^[>]*)\\[([^]]*)\\]", sig)
                dtype = match.group(1)
                shape = match.group(2)
                ndim = shape.count(",") + 1

                if meta is None:
                    output.append("*" + dtype)
                    # Currently the host side tensor descriptors get passed in as a
                    # tensor desc, shape, and strides. We have no way to use these
                    # shape and strides when processing tensor descriptors which is
                    # why we provide our own decomposition above. Sadly this means
                    # we have to pass the shape and strides twice.
                    for _ in range(2 * ndim):
                        output.append("i64")
                    output.append("i1")
                else:
                    output.append("nvTmaDesc")

                for _ in range(ndim):
                    output.append("i32")
                for _ in range(ndim):
                    output.append("i64")
            else:
                output.append(sig)

        # assert not tensordesc_meta or tensordesc_idx == len(tensordesc_meta)
        return output

    def _flatten_signature(sig, output):
        # Flatten tuples
        if isinstance(sig, tuple):
            for x in sig:
                _flatten_signature(x, output)
        else:
            output.append(sig)

    def _extracted_type(ty):
        if isinstance(ty, tuple):
            val = ','.join(map(_extracted_type, ty))
            return f"[{val}]"
        if ty[0] == '*':
            return "PyObject*"
        if ty in ("constexpr", "nvTmaDesc"):
            return "PyObject*"
        return ty_to_cpp(ty)

    def format_of(ty):
        if isinstance(ty, tuple):
            val = ''.join(map(format_of, ty))
            return f"({val})"
        if ty[0] == '*':
            return "O"
        if ty in ("constexpr", "nvTmaDesc"):
            return "O"
        if ty.startswith("tensordesc"):
            return "O"
        return {
            "double": "d",
            "long": "l",
            "int8_t": "b",
            "int16_t": "h",
            "int32_t": "i",
            "int64_t": "L",
            "uint8_t": "B",
            "uint16_t": "H",
            "uint32_t": "I",
            "uint64_t": "K",
        }[ty_to_cpp(ty)]

    expand_signature = _expand_signature(signature.values())
    signature = {i: s for i, s in enumerate(expand_signature)}

    args_format = ''.join([format_of(ty) for ty in signature.values()])
    format = _BASE_ARGS_FORMAT + args_format

    flat_signature = []
    for sig in signature.values():
        _flatten_signature(sig, flat_signature)
    signature = {i: s for i, s in enumerate(flat_signature)}
    args_list = ', ' + ', '.join(f"&_arg{i}" for i, ty in signature.items()) if len(signature) > 0 else ''
    arg_decl_list = []
    for i, ty in signature.items():
        if ty == "constexpr":
            continue
        if ty in FLOAT_STORAGE_TYPE:
            arg_decl_list.append(f"{FLOAT_STORAGE_TYPE[ty]} arg{i}")
        else:
            arg_decl_list.append(f"{ty_to_cpp(ty)} arg{i}")
    arg_decls = ', '.join(arg_decl_list)
    internal_args_list = []
    for i, ty in signature.items():
        if ty[0] == "*":
            internal_args_list.append(f"ptr_info{i}.dev_ptr")
        elif ty in FLOAT_STORAGE_TYPE:
            internal_args_list.append(f"_arg{i}_storage")
        elif ty == "nvTmaDesc":
            # Note: we have to dereference the pointer
            internal_args_list.append(f"*tma_ptr{i}")
        elif ty != "constexpr":
            internal_args_list.append(f"_arg{i}")

    # generate glue code
    newline = '\n  '
    ptr_decls = [
        f"DevicePtrInfo ptr_info{i} = getPointer(_arg{i}, {i}); if (!ptr_info{i}.valid) return NULL;"
        for i, ty in signature.items()
        if ty[0] == "*"
    ]
    tma_decls = [
        f"CUtensorMap* tma_ptr{i} = getTmaDesc(_arg{i}); if (!tma_ptr{i}) return NULL;" for i, ty in signature.items()
        if ty == "nvTmaDesc"
    ]
    float_storage_decls = [
        f"{FLOAT_STORAGE_TYPE[ty]} _arg{i}_storage = {FLOAT_PACK_FUNCTION[ty]}(_arg{i});"
        for i, ty in signature.items()
        if ty in FLOAT_STORAGE_TYPE
    ]
    params = [f"&arg{i}" for i, ty in signature.items() if ty != "constexpr"]
    params.append("&global_scratch")
    params.append("&profile_scratch")
    src = f"""
#include <mcr/mc_runtime.h>
#include <stdbool.h>
#include <Python.h>
#include <dlfcn.h>

static inline void gpuAssert(mcError_t code, const char *file, int line)
{{
   if (code != mcSuccess)
   {{
      const char* prefix = "Triton Error [MACA]: ";
      const char* str = mcGetErrorString(code);
      char err[1024] = {{0}};
      strcat(err, prefix);
      strcat(err, str);
      PyGILState_STATE gil_state;
      gil_state = PyGILState_Ensure();
      PyErr_SetString(PyExc_RuntimeError, err);
      PyGILState_Release(gil_state);
   }}
}}

#define MACA_CHECK(ans) {{ gpuAssert((ans), __FILE__, __LINE__); }}

static void _launch(int gridX, int gridY, int gridZ, int num_warps, int num_ctas, int clusterDimX, int clusterDimY, int clusterDimZ, int shared_memory, mcStream_t stream, mcFunction_t function, mcDeviceptr_t global_scratch, mcDeviceptr_t profile_scratch{', ' + arg_decls if len(arg_decls) > 0 else ''}) {{
  void *params[] = {{ {', '.join(params)} }};
  // TODO: support launchConfig
  if (gridX*gridY*gridZ > 0) {{
    assert(num_ctas == 1);
    MACA_CHECK(mcModuleLaunchKernel(function, gridX, gridY, gridZ, 64*num_warps, 1, 1, shared_memory, stream, params, 0));
  }}
}}

typedef struct _DevicePtrInfo {{
    mcDeviceptr_t dev_ptr;
    bool valid;
}} DevicePtrInfo;

static PyObject* data_ptr_str = NULL;
static PyObject* py_tensor_map_type = NULL;

static inline DevicePtrInfo getPointer(PyObject *obj, int idx) {{
  DevicePtrInfo ptr_info;
  ptr_info.dev_ptr = 0;
  ptr_info.valid = true;
  if (PyLong_Check(obj)) {{
    ptr_info.dev_ptr = (mcDeviceptr_t)PyLong_AsUnsignedLongLong(obj);
    return ptr_info;
  }}
  if (obj == Py_None) {{
    // valid nullptr
    return ptr_info;
  }}
  PyObject *ret = PyObject_CallMethodNoArgs(obj, data_ptr_str);
  if (!ret) {{
    PyErr_SetString(PyExc_TypeError, "Pointer argument must be either uint64 or have data_ptr method");
    ptr_info.valid = false;
    goto cleanup;
  }}
  if (!PyLong_Check(ret)) {{
    PyErr_SetString(PyExc_TypeError, "data_ptr method of Pointer object must return 64-bit int");
    ptr_info.valid = false;
    goto cleanup;
  }}
  ptr_info.dev_ptr = (mcDeviceptr_t)(PyLong_AsUnsignedLongLong(ret));
  if(!ptr_info.dev_ptr)
    return ptr_info;
  uint64_t dev_ptr;
  int status = mcPointerGetAttribute(&dev_ptr, mcPointerAttributeDevicePointer, ptr_info.dev_ptr);
  if (status == mcErrorInvalidValue) {{
      PyErr_Format(PyExc_ValueError,
                  "Pointer argument (at %d) cannot be accessed from Triton (cpu tensor?)", idx);
      ptr_info.valid = false;
  }} else if (status != mcSuccess) {{
      MACA_CHECK(status);
      ptr_info.valid = false;
  }}
  ptr_info.dev_ptr = (mcDeviceptr_t)dev_ptr;
cleanup:
  Py_XDECREF(ret);  // Thanks ChatGPT!
  return ptr_info;
}}


static uint16_t pack_fp16(double f) {{
    uint16_t result;
    // from https://github.com/python/pythoncapi-compat
#if 0x030600B1 <= PY_VERSION_HEX && PY_VERSION_HEX <= 0x030B00A1 && !defined(PYPY_VERSION)
    _PyFloat_Pack2(f, (unsigned char*)&result, 1);
#else
    PyFloat_Pack2(f, (unsigned char*)&result, 1);
#endif
    return result;
}}

static uint16_t pack_bf16(double f) {{
    float f32 = (float)f;
    uint32_t u32 = *(uint32_t*)&f32;
    return (uint16_t)(u32 >> 16);
}}

static uint32_t pack_fp32(double f) {{
    float f32 = (float)f;
    return *(uint32_t*)&f32;
}}

static uint64_t pack_fp64(double f) {{
    return *(uint64_t*)&f;
}}

static PyObject* launch(PyObject* self, PyObject* args) {{
  int gridX, gridY, gridZ;
  uint64_t _stream;
  uint64_t _function;
  PyObject *launch_enter_hook = NULL;
  PyObject *launch_exit_hook = NULL;
  PyObject *kernel_metadata = NULL;
  PyObject *launch_metadata = NULL;
  PyObject *global_scratch_obj = NULL;
  PyObject *profile_scratch_obj = NULL;
  {newline.join([f"{_extracted_type(ty)} _arg{i};" for i, ty in signature.items()])}
  if(!PyArg_ParseTuple(args, \"{format}\", &gridX, &gridY, &gridZ,
                                           &_stream, &_function, &global_scratch_obj, &profile_scratch_obj,
                                           &kernel_metadata, &launch_metadata,
                                           &launch_enter_hook, &launch_exit_hook {args_list})) {{
    return NULL;
  }}

  int num_warps, num_ctas, shared_memory, clusterDimX, clusterDimY, clusterDimZ;
  if (!PyArg_ParseTuple(kernel_metadata, \"iiiiii\", &num_warps, &num_ctas, &shared_memory, &clusterDimX, &clusterDimY, &clusterDimZ)) {{
    PyErr_SetString(PyExc_TypeError, "kernel_metadata must be a tuple");
    return NULL;
  }}

  // extract launch metadata
  if (launch_enter_hook != Py_None){{
    PyObject* args = Py_BuildValue("(O)", launch_metadata);
    PyObject* ret = PyObject_CallObject(launch_enter_hook, args);
    Py_DECREF(args);
    if (!ret)
      return NULL;
  }}

  mcDeviceptr_t global_scratch = 0;
  if (global_scratch_obj != Py_None) {{
    DevicePtrInfo global_scratch_info = getPointer(global_scratch_obj, -1);
    if (!global_scratch_info.valid) {{
      return NULL;
    }}
    global_scratch = global_scratch_info.dev_ptr;
  }}


  mcDeviceptr_t profile_scratch = 0;
  if (profile_scratch_obj != Py_None) {{
    DevicePtrInfo profile_scratch_info = getPointer(profile_scratch_obj, -1);
    if (!profile_scratch_info.valid) {{
      return NULL;
    }}
    profile_scratch = profile_scratch_info.dev_ptr;
  }}

  // raise exception asap
  {newline.join(ptr_decls)}
  {newline.join(tma_decls)}
  {newline.join(float_storage_decls)}
  Py_BEGIN_ALLOW_THREADS;
  _launch(gridX, gridY, gridZ, num_warps, num_ctas, clusterDimX, clusterDimY, clusterDimZ, shared_memory, (mcStream_t)_stream, (mcFunction_t)_function, global_scratch, profile_scratch{', ' + ', '.join(internal_args_list) if len(internal_args_list) > 0 else ''});
  Py_END_ALLOW_THREADS;
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
  \"__triton_launcher\",
  NULL, //documentation
  -1, //size
  ModuleMethods
}};

PyMODINIT_FUNC PyInit___triton_launcher(void) {{
  data_ptr_str = PyUnicode_InternFromString("data_ptr");
  if(data_ptr_str == NULL) {{
    return NULL;
  }}
  PyObject *m = PyModule_Create(&ModuleDef);
  if(m == NULL) {{
    return NULL;
  }}
  PyModule_AddFunctions(m, ModuleMethods);
  return m;
}}
"""
    return src


class MacaLauncher(object):

    def __init__(self, src, metadata):
        ids = {"ids_of_const_exprs": src.fn.constexprs if hasattr(src, "fn") else tuple()}
        constants = src.constants if hasattr(src, "constants") else dict()
        cst_key = lambda i: src.fn.arg_names.index(i) if isinstance(i, str) else i
        constants = {cst_key(key): value for key, value in constants.items()}
        signature = {cst_key(key): value for key, value in src.signature.items()}
        src = make_launcher(constants, signature, ids)
        mod = compile_module_from_src(
            src=src,
            name="__triton_launcher",
            library_dirs=library_dirs(),
            include_dirs=hdrmaca_dirs(),
            libraries=libraries,
        )
        self.launch = mod.launch
        self.global_scratch_size = metadata.global_scratch_size
        self.global_scratch_align = metadata.global_scratch_align
        self.profile_scratch_size = metadata.profile_scratch_size
        self.profile_scratch_align = metadata.profile_scratch_align

    def __call__(self, gridX, gridY, gridZ, stream, function, *args):

        def allocate_scratch(size, align, allocator):
            if size > 0:
                grid_size = gridX * gridY * gridZ
                alloc_size = grid_size * self.num_ctas * size
                alloc_fn = allocator.get()
                return alloc_fn(alloc_size, align, stream)
            return None

        global_scratch = allocate_scratch(self.global_scratch_size, self.global_scratch_align, _allocation._allocator)
        profile_scratch = allocate_scratch(self.profile_scratch_size, self.profile_scratch_align,
                                           _allocation._profile_allocator)
        self.launch(gridX, gridY, gridZ, stream, function, global_scratch, profile_scratch, *args)


class MacaDriver(GPUDriver):

    def __init__(self):
        self.utils = MacaUtils()  # TODO: make static
        self.launcher_cls = MacaLauncher
        super().__init__()

    def get_active_torch_device(self):
        import torch
        return torch.device("cuda", self.get_current_device())

    def get_device_interface(self):
        import torch
        return torch.cuda

    def get_current_target(self):
        device = self.get_current_device()
        capability = self.get_device_capability(device)
        capability = capability[0] * 10 + capability[1]
        warp_size = 64
        return GPUTarget("maca", capability, warp_size)

    @staticmethod
    def is_active():
        import torch
        return torch.cuda.is_available() and (torch.version.hip is None)

    def map_python_to_cpp_type(self, ty: str) -> str:
        return ty_to_cpp(ty)

    def get_benchmarker(self):
        from triton.testing import do_bench
        return do_bench

    def get_empty_cache_for_benchmark(self):
        import torch

        # We maintain a buffer of 256 MB that we clear
        # before each kernel call to make sure that the L2 cache
        # doesn't contain any input data before the run
        cache_size = 256 * 1024 * 1024
        return torch.empty(int(cache_size // 4), dtype=torch.int, device='cuda')

    def clear_cache(self, cache):
        cache.zero_()
