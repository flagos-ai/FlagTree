import functools
import os
import platform
import subprocess
import re
import triton
from pathlib import Path
from triton import knobs
from triton.runtime.build import compile_module_from_src
from triton.runtime import _allocation
from triton.backends.compiler import GPUTarget
from triton.backends.driver import GPUDriver

dirname = os.path.dirname(os.path.realpath(__file__))
include_dirs = [os.path.join(dirname, "include")]
libdevice_dir = os.path.join(dirname, "lib")
libraries = ['tang', 'tangrt_shared']
arch = platform.machine()


@functools.lru_cache()
def libtang_dirs():
    if env_libtang_path := knobs.env_opt_str("TRITON_LIBTANG_PATH"):
        return [env_libtang_path]

    libs = subprocess.check_output(["/sbin/ldconfig", "-p"]).decode(errors="ignore")
    # each line looks like the following:
    # libtang.so.1 (libc6,x86-64) => /lib/x86_64-linux-gnu/libtang.so.1
    locs = [line.split()[-1] for line in libs.splitlines() if "libtang.so" in line]
    dirs = [os.path.dirname(loc) for loc in locs]
    env_ld_library_path = os.getenv("LD_LIBRARY_PATH")
    if env_ld_library_path and not dirs:
        dirs = [dir for dir in env_ld_library_path.split(":") if os.path.exists(os.path.join(dir, "libtang.so"))]
    if not dirs:
        dirs = [f'/usr/local/tangrt/lib/linux-{arch}/stub/']
    msg = 'libtang.so cannot found!\n'
    if locs:
        msg += 'Possible files are located at %s.' % str(locs)
        msg += 'Please create a symlink of libtang.so to any of the files.'
    else:
        msg += 'Please make sure GPU is set up and then run "/sbin/ldconfig"'
        msg += ' (requires sudo) to refresh the linker cache.'
    assert any(os.path.exists(os.path.join(path, 'libtang.so')) for path in dirs), msg
    return dirs


@functools.lru_cache()
def library_dirs():
    return [libdevice_dir, *libtang_dirs(), f"/usr/local/tangrt/lib/linux-{arch}"]


# ------------------------
# Utils
# ------------------------


class SunriseUtils(object):

    def __new__(cls):
        if not hasattr(cls, "instance"):
            cls.instance = super(SunriseUtils, cls).__new__(cls)
        return cls.instance

    def __init__(self):
        mod = compile_module_from_src(
            src=Path(os.path.join(dirname, "driver.c")).read_text(),
            name="tang_utils",
            library_dirs=library_dirs(),
            include_dirs=include_dirs,
            libraries=libraries,
        )
        self.load_binary = mod.load_binary
        self.get_device_properties = mod.get_device_properties


# ------------------------
# Launcher
# ------------------------


def ty_to_cpp(ty):
    if ty[0] == '*':
        return "TAdeviceptr"
    return {
        "i1": "int_t",
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

_BASE_ARGS_FORMAT = "iiiKKOOOOO"


def make_launcher(constants, signature, warp_size, tensordesc_meta):

    def _expand_signature(signature):
        output = []
        # Expand tensor descriptor arguments into base pointer, shape, and
        # strides
        for sig in signature:
            if isinstance(sig, str) and sig.startswith("tensordesc"):
                ndim = sig.count(",") + 1
                dtype = re.match("tensordesc<([^[>]*)", sig).group()

                output.append("*" + dtype)
                # Currently the host side tensor descriptors get passed in as a
                # tensor desc, shape, and strides. We have no way to use these
                # shape and strides when processing tensor descriptors which is
                # why we provide our own decomposition above. Sadly this means
                # we have to pass the shape and strides twice.
                for _ in range(2 * ndim):
                    output.append("i64")
                output.append("i1")

                for _ in range(ndim):
                    output.append("i32")
                for _ in range(ndim):
                    output.append("i64")
            else:
                output.append(sig)

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
        if ty == "constexpr":
            return "PyObject*"
        return ty_to_cpp(ty)

    def format_of(ty):
        if isinstance(ty, tuple):
            val = ''.join(map(format_of, ty))
            return f"({val})"
        if ty[0] == '*':
            return "O"
        if ty == "constexpr":
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
    # Record the end of regular arguments;
    # subsequent arguments are architecture-specific descriptors, such as tensor descriptors for CUDA.
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
        elif ty != "constexpr":
            internal_args_list.append(f"_arg{i}")

    # generate glue code
    newline = '\n  '
    ptr_decls = [
        f"DevicePtrInfo ptr_info{i} = getPointer(_arg{i}, {i}); if (!ptr_info{i}.valid) return NULL;"
        for i, ty in signature.items()
        if ty[0] == "*"
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
#include \"tang.h\"
#include \"tang_runtime.h\"
#include <stdbool.h>
#include <Python.h>
#include <dlfcn.h>

static inline void gpuAssert(TAresult code, const char *file, int line)
{{
   if (code != TANG_SUCCESS)
   {{
      const char* prefix = "Triton Error [TANG]: ";
      const char* str;
      taGetErrorString(code, &str);
      char err[1024] = {{0}};
      strcat(err, prefix);
      strcat(err, str);
      PyGILState_STATE gil_state;
      gil_state = PyGILState_Ensure();
      PyErr_SetString(PyExc_RuntimeError, err);
      PyGILState_Release(gil_state);
   }}
}}

#define TANG_CHECK(ans) {{ gpuAssert((ans), __FILE__, __LINE__); }}

static void _launch(int gridX, int gridY, int gridZ, int num_warps, int num_ctas, int shared_memory, TAstream stream, TAfunction function, TAdeviceptr profile_scratch{', ' + arg_decls if len(arg_decls) > 0 else ''}) {{
  TAdeviceptr global_scratch = 0;
  void *params[] = {{ {', '.join(params)} }};
  if (gridX*gridY*gridZ > 0) {{
    TANG_CHECK(taLaunchKernel(function, gridX, gridY, gridZ, {warp_size}*num_warps, 1, 1, shared_memory, stream, params, 0));
  }}
}}

typedef struct _DevicePtrInfo {{
    TAdeviceptr dev_ptr;
    bool valid;
}} DevicePtrInfo;

static inline DevicePtrInfo getPointer(PyObject *obj, int idx) {{
  DevicePtrInfo ptr_info;
  ptr_info.dev_ptr = 0;
  ptr_info.valid = true;

  if (PyLong_Check(obj)) {{
    ptr_info.dev_ptr = PyLong_AsUnsignedLongLong(obj);
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

    ptr_info.dev_ptr = PyLong_AsUnsignedLongLong(ret);

    if(!ptr_info.dev_ptr) {{
      return ptr_info;
    }}

    // 暂时使用PyTorch接口的方案, 后续taPointerGetAttribute支持使用指针切换后，还是使用它
    // 获取 device 属性
    PyObject* device_obj = PyObject_GetAttrString(obj, "device");
    if (device_obj && device_obj != Py_None) {{
        // 获取 device.index
        PyObject* index_obj = PyObject_GetAttrString(device_obj, "index");
        if (index_obj && PyLong_Check(index_obj)) {{
            int dev = PyLong_AsLong(index_obj);
            // printf("[DEBUG] Switching to tensor device (device.index): %d\\n", dev);
            tangSetDevice(dev);
        }}
        Py_XDECREF(index_obj);
    }}
    Py_XDECREF(device_obj);

    uint64_t dev_ptr;
    int status = taPointerGetAttribute(&dev_ptr, TA_POINTER_ATTRIBUTE_DEVICE_POINTER, ptr_info.dev_ptr);
    if (status == TANG_ERROR_INVALID_VALUE) {{
        PyErr_Format(PyExc_ValueError,
                     "Pointer argument (at %d) cannot be accessed from Triton (cpu tensor?)", idx);
        ptr_info.valid = false;
    }} else if (status != TANG_SUCCESS) {{
        TANG_CHECK(status);  // Catch any other TANG API errors
        ptr_info.valid = false;
    }}
    ptr_info.dev_ptr = dev_ptr;
    Py_DECREF(ret);  // Thanks ChatGPT!
    return ptr_info;
  }}
  PyErr_SetString(PyExc_TypeError, "Pointer argument must be either uint64 or have data_ptr method");
  ptr_info.valid = false;
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
  PyObject *profile_scratch_obj = NULL;
  {' '.join([f"{_extracted_type(ty)} _arg{i}; " for i, ty in signature.items()])}
  if(!PyArg_ParseTuple(args, \"{format}\", &gridX, &gridY, &gridZ, &_stream, &_function, &profile_scratch_obj,
                                           &kernel_metadata, &launch_metadata,
                                           &launch_enter_hook, &launch_exit_hook {args_list})) {{
    PyErr_SetString(PyExc_TypeError, "get input data error");
    return NULL;
  }}

  {' '.join(float_storage_decls)}

  int num_warps, num_ctas, shared_memory;
  if (!PyArg_ParseTuple(kernel_metadata, \"iii\", &num_warps, &num_ctas, &shared_memory)) {{
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
    Py_DECREF(ret);
  }}

  TAdeviceptr profile_scratch = 0;
  if (profile_scratch_obj != Py_None) {{
    DevicePtrInfo profile_scratch_info = getPointer(profile_scratch_obj, -1);
    if (!profile_scratch_info.valid) {{
      return NULL;
    }}
    profile_scratch = profile_scratch_info.dev_ptr;
  }}

  // raise exception asap
  {"".join([f"DevicePtrInfo ptr_info{i} = getPointer(_arg{i}, {i}); if (!ptr_info{i}.valid) return NULL;" if ty[0] == "*" else "" for i, ty in signature.items()])};
  _launch(gridX, gridY, gridZ, num_warps, num_ctas, shared_memory, (TAstream)_stream, (TAfunction)_function, profile_scratch{', ' + ', '.join(internal_args_list) if len(internal_args_list) > 0 else ''});
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

  Py_RETURN_NONE;
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
  PyObject *m = PyModule_Create(&ModuleDef);
  if(m == NULL) {{
    return NULL;
  }}
  PyModule_AddFunctions(m, ModuleMethods);
  return m;
}}
"""
    return src


def wrap_handle_tensordesc(launcher, signature, tensordesc_meta):
    has_tensor_desc_arg = any(isinstance(sig, str) and sig.startswith("tensordesc") for sig in signature.values())
    if not has_tensor_desc_arg:
        return launcher

    from triton.tools.tensor_descriptor import TensorDescriptor

    def inner(*args):
        meta_args = args[:len(_BASE_ARGS_FORMAT)]
        raw_kernel_args = args[len(_BASE_ARGS_FORMAT):]
        final_args = []
        for arg in raw_kernel_args:
            if isinstance(arg, TensorDescriptor):
                # Currently the host side tensor descriptors get decomposed in
                # the frontend to tensor desc, shape, and strides. We have no
                # way to use these shape and strides when processing tensor
                # descriptors which is why we provide our own decomposition
                # above. Sadly this means we have to pass the shape and strides
                # twice.
                final_args.extend([arg.base, *arg.shape, *arg.strides, *arg.shape, *arg.strides])
            else:
                final_args.append(arg)
        return launcher(*meta_args, *final_args)

    return inner


class SunriseLauncher(object):

    def __init__(self, src, metadata):
        constants = src.constants if hasattr(src, "constants") else dict()
        arg_idx = lambda x: (src.fn.arg_names.index(x), ) if isinstance(x, str) else x
        constants = {arg_idx(idx): value for idx, value in constants.items()}
        signature = {idx: value for idx, value in src.signature.items()}
        tensordesc_meta = getattr(metadata, "tensordesc_meta", None)
        src = make_launcher(constants, signature, metadata.warp_size, tensordesc_meta)
        mod = compile_module_from_src(
            src=src,
            name="__triton_launcher",
            library_dirs=library_dirs(),
            include_dirs=include_dirs,
            libraries=libraries,
        )
        has_tensor_desc_arg = any(isinstance(sig, str) and sig.startswith("tensordesc") for sig in signature.values())
        self.launch = wrap_handle_tensordesc(mod.launch) if has_tensor_desc_arg else mod.launch
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

        profile_scratch = allocate_scratch(self.profile_scratch_size, self.profile_scratch_align,
                                           _allocation._profile_allocator)
        self.launch(gridX, gridY, gridZ, stream, function, profile_scratch, *args)


class SunriseDriver(GPUDriver):

    def __init__(self):
        self.utils = SunriseUtils()  # TODO: make static
        self.launcher_cls = SunriseLauncher
        import torch
        if not hasattr(torch, "ptpu"):
            raise RuntimeError("torch.ptpu is not available")
        self.get_device_capability = torch.ptpu.get_device_capability
        self.get_current_stream = lambda dev_idx: torch.ptpu.current_stream(dev_idx).ptpu_stream
        self.get_current_device = torch.ptpu.current_device
        self.set_current_device = torch.ptpu.set_device

    def get_current_target(self):
        arch = knobs.runtime.override_arch
        if not arch:
            arch = "S2"
        warp_size = 32
        return GPUTarget("tang", arch, warp_size)

    def get_active_torch_device(self):
        import torch
        return torch.device("ptpu", self.get_current_device())

    def get_device_interface(self):
        import torch
        return torch.ptpu

    @staticmethod
    def is_active():
        # 默认用pts后端
        import torch
        return True
        return torch.cuda.is_available() and (torch.version.hip is None) and (torch.version.cuda is None)

    def map_python_to_cpp_type(self, ty: str) -> str:
        return ty_to_cpp(ty)

    def get_benchmarker(self):
        from triton.testing import do_bench
        return do_bench

    def get_empty_cache_for_benchmark(self):
        import torch
        import torch_ptpu

        # We maintain a buffer of 256 MB that we clear
        # before each kernel call to make sure that the L2 cache
        # doesn't contain any input data before the run
        cache_size = 256 * 1024 * 1024
        return torch.empty(int(cache_size // 4), dtype=torch.int, device='ptpu')

    def clear_cache(self, cache):
        cache.zero_()
