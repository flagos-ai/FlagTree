#include <xpu/runtime.h>
#define PY_SSIZE_T_CLEAN // control type size
#include <Python.h>

static inline void xpuAssert(int code, const char *file, int line,
                             const char *call) {
  if (code != XPU_SUCCESS) {
    const char *err_msg = xpu_strerror(code);
    char buf[1024] = {0};
    sprintf(buf, "%s:%d: %s -> %s(err_code: %d)", file, line, call, err_msg,
            code);
    PyGILState_STATE gil_state;
    gil_state = PyGILState_Ensure();
    PyErr_SetString(PyExc_RuntimeError, buf);
    PyGILState_Release(gil_state);
  }
}

#define XPU_CHECK(ans)                                                         \
  {                                                                            \
    xpuAssert((ans), __FILE__, __LINE__, #ans);                                \
    if (PyErr_Occurred())                                                      \
      return NULL;                                                             \
  }

uint32_t checksum(const char *data, size_t length) {
  uint32_t crc32 = 0;
  for (size_t i = 0; i < length; ++i) {
    crc32 += static_cast<uint32_t>(data[i]);
  }
  return crc32;
}

// XPU3 runtime export this function no more, copy it from runtime repo.
// XPU Kernel type
enum kernel_type {
  /// XPU Cluster kernel
  KT_CLUSTER = 0,
  /// XPU SDNN kernel
  KT_SDCDNN = 1,
};

// Place of XPU kernel binary
enum kernel_place {
  /// XPU kernel binary locates on CPU memory
  KP_CPU = 0,
  /// XPU kernel binary locates on XPU memory
  KP_XPU = 1,
};

// XPU Kernel
struct xpu_kernel {
  /// Combination of kernel place and type:
  /// [31:16] kernel place, KP_CPU or KP_XPU
  /// [15:0]  kernel type, KT_CLUSTER or KT_SDCDNN
  uint32_t type : 16;
  uint32_t place : 16;
  /// kernel code address on CPU Memory
  uint64_t code_addr;
  /// kernel code size in bytes
  uint32_t code_byte_size;
  /// initial program counter
  uint32_t code_pc;
  /// dword size kernel needed to transfer params
  /// essentially, this is the count of param registers needed
  uint32_t param_dword_size;
  /// kernel code hash, for cache indexing
  uint64_t hash;
  /// (maybe mangled) function name
  const char *name;
  /// private data structure used by xpu runtime
  void *rt_private;
  uint64_t printf_buffer_offset;
};

static int __xpu_create_func(XPUFunc *pfunc, int type, uint64_t code_addr,
                             uint32_t code_bsz, uint32_t code_pc,
                             uint32_t param_dsz, uint64_t hash,
                             const char *name, bool on_xpu,
                             uint64_t printf_buf_offset) {
  if (pfunc == NULL) {
    return -XPUERR_INVALID_PARAM;
  }

  struct xpu_kernel *kern = new struct xpu_kernel();
  //   printf("create func @0x%" PRIx64 " hash=0x%" PRIx64 " name='%s'(%p)\n",
  //          code_addr, hash, (name == NULL) ? "NULL" : name, name);

  kern->type = type;
  kern->place = (on_xpu) ? KP_XPU : KP_CPU;
  kern->code_addr = code_addr;
  kern->code_byte_size = code_bsz;
  kern->code_pc = code_pc;
  kern->param_dword_size = param_dsz;
  kern->hash = hash;
  kern->name = name;
  // printf("printf_buf_offset = 0x%08lx\n", printf_buf_offset);
  kern->printf_buffer_offset = printf_buf_offset;

  *pfunc = kern;

  return 0;
}

static PyObject *loadBinary(PyObject *self, PyObject *args) {
  // XPU uses an XPU-specific load_binary ABI, restored in
  // triton/compiler/compiler.py via an `is_xpu_backend` branch:
  //   load_binary(name, kernel_bytes, is_sdnn, printf_buf_offset)
  // Both fields come from backend/compiler.py metadata and must be honored:
  // - is_sdnn distinguishes KT_CLUSTER vs KT_SDCDNN at __xpu_create_func
  // - printf_buf_offset is required for kernels using device-side printf
  const char *name;
  const char *data;
  Py_ssize_t data_size;
  int is_sdnn_kernel;
  uint64_t printf_buf_offset;
  if (!PyArg_ParseTuple(args, "ss#iK", &name, &data, &data_size,
                        &is_sdnn_kernel, &printf_buf_offset)) {
    return NULL;
  }

  // Create XPUFunc
  XPUFunc pfunc;
  int type = is_sdnn_kernel ? KT_SDCDNN : KT_CLUSTER;
  uint64_t code_addr = reinterpret_cast<uint64_t>(data);
  uint32_t code_byte_size = static_cast<uint32_t>(data_size);
  uint32_t code_pc = 0;
  uint32_t param_dword_size = 0;
  uint32_t hash = checksum(data, data_size);
  bool on_xpu = false;

  XPU_CHECK(__xpu_create_func(&pfunc, type, code_addr, code_byte_size, code_pc,
                              param_dword_size, hash, name, on_xpu,
                              printf_buf_offset));

  // Build Output Value
  const void *mod = static_cast<const void *>(data);
  int32_t n_regs = 0;
  int32_t n_spills = 0;
  // Triton 3.6 expects (mod, func, n_regs, n_spills, n_max_threads).
  // XPU has no fine-grained thread limit; use a large sentinel so the
  // num_warps*warp_size check always passes.
  int32_t n_max_threads = 1 << 30;
  return Py_BuildValue("(KKiii)", (uint64_t)mod, (uint64_t)pfunc, n_regs,
                       n_spills, n_max_threads);
}

static PyObject *getDeviceProperties(PyObject *self, PyObject *args) {
  int device_id;
  if (!PyArg_ParseTuple(args, "i", &device_id))
    return NULL;

  // create a struct to hold device properties
  int max_shared_mem = 256 * 1024; // 256K for XPU2
  int max_num_regs = 0;
  int warp_size = 1;
  int sm_clock_rate = 0;
  int mem_clock_rate = 0;
  int mem_bus_width = 0;

  int multiprocessor_count = 0;
  uint64_t num_cluster = 0;
  XPU_CHECK(xpu_device_get_attr(&num_cluster, XPUATTR_NUM_CLUSTER, device_id));
  multiprocessor_count = num_cluster;

  uint64_t model = (uint64_t)KL3;
  int device_model_int = -1;
  XPU_CHECK(xpu_device_get_attr((uint64_t *)&model, XPUATTR_MODEL, device_id));

  // ==================== FLAGTREE XPU SYNC MARK ====================
  // This FlagTree XPU path currently supports the XPU3/KL3 runtime bundle.
  // Keep probing constrained to KL3 instead of referencing newer SDK model
  // macros that may not exist in the local headers.
  // ==================== FLAGTREE XPU SYNC MARK ====================
  assert(model <= KL3_END && "model version must be less than or equal to KL3");
  assert(model >= KL3_BEGIN &&
         "model version must be more than or equal to KL3");
  device_model_int = 3;

  return Py_BuildValue(
      "{s:i, s:i, s:i, s:i, s:i, s:i, s:i, s:i}", "max_shared_mem",
      max_shared_mem, "max_num_regs", max_num_regs, "multiprocessor_count",
      multiprocessor_count, "warpSize", warp_size, "sm_clock_rate",
      sm_clock_rate, "mem_clock_rate", mem_clock_rate, "mem_bus_width",
      mem_bus_width, "device_model", device_model_int);
}

static PyMethodDef ModuleMethods[] = {
    {"load_binary", loadBinary, METH_VARARGS,
     "Load provided xpubin into XPU driver"},
    {"get_device_properties", getDeviceProperties, METH_VARARGS,
     "Get the properties for a given device"},
    {NULL, NULL, 0, NULL} // sentinel
};

static struct PyModuleDef ModuleDef = {PyModuleDef_HEAD_INIT, "xpu_utils",
                                       NULL, // documentation
                                       -1,   // size global
                                       ModuleMethods};

// Python C API Binding
PyMODINIT_FUNC PyInit_xpu_utils(void) {
  PyObject *m = PyModule_Create(&ModuleDef);
  if (m == NULL) {
    return NULL;
  }

  PyModule_AddFunctions(m, ModuleMethods);

  return m;
}
