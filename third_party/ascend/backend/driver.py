from pathlib import Path
import tempfile
import os
import os.path
import re
import subprocess
import sysconfig
from typing import Optional
import functools
import hashlib
from triton.runtime.cache import get_cache_manager, get_dump_manager
from triton.backends.driver import DriverBase
from triton.backends.compiler import GPUTarget
from triton.backends.ascend.utils import (
    _build_npu_ext,
    _check_cxx11_abi,
    convert_sigtype_to_int,
    _is_auto_map_parallel_blocks_enabled,
)

class NPUUtils(object):
    def __new__(cls):
        if not hasattr(cls, 'instance'):
            cls.instance = super(NPUUtils, cls).__new__(cls)
        return cls.instance

    def __init__(self):
        dirname = os.path.dirname(os.path.realpath(__file__))
        src = Path(os.path.join(dirname, "npu_utils.cpp")).read_text()
        key = hashlib.sha256(src.encode("utf-8")).hexdigest()
        cache = get_cache_manager(key)
        fname = "npu_utils.so"
        cache_path = cache.get_file(fname)
        if cache_path is None:
            with tempfile.TemporaryDirectory() as tmpdir:
                src_path = os.path.join(tmpdir, "npu_utils.cpp")
                with open(src_path, "w") as f:
                    f.write(src)
                so = _build_npu_ext("npu_utils", src_path, tmpdir)
                with open(so, "rb") as f:
                    cache_path = cache.put(f.read(), fname, binary=True)
        import importlib.util
        spec = importlib.util.spec_from_file_location("npu_utils", cache_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        self.npu_utils_mod = mod

    def load_binary(self, name, kernel, shared, device):
        fnname, mix_mode = name.split()
        return self.npu_utils_mod.load_kernel_binary(fnname, kernel, shared, device, mix_mode)

    @functools.lru_cache()
    def get_device_properties(self, device):
        # temperoarily added "max_shared_mem" properties to avoid triton-compiler complain
        # fetch available memory at runtime
        num_aic = self.get_aicore_num()
        num_aiv = num_aic * 2
        return {"max_shared_mem": 1, "num_aicore": num_aic, "num_vectorcore": num_aiv}

    @functools.lru_cache()
    def get_arch(self):
        # temporarily return empty arch descriptor
        return self.npu_utils_mod.get_arch()

    @functools.lru_cache()
    def get_aicore_num(self):
        # temporarily return empty arch descriptor
        return self.npu_utils_mod.get_aicore_num()

    @functools.lru_cache()
    def get_aivector_core_num(self):
        return self.get_device_properties("npu")["num_vectorcore"]


class NPULauncher(object):
    def __init__(self, src, metadata):
        debug_mode = metadata.debug
        workspace_size = int(metadata.workspace_size) \
                              if hasattr(metadata, 'workspace_size') else -1
        lock_init_value = int(metadata.lock_init_value) \
                              if hasattr(metadata, 'lock_init_value') else 0
        lock_num = int(metadata.lock_num) \
                              if hasattr(metadata, 'lock_num') else -1
        constants = src.constants if hasattr(src, "constants") else dict()
        cst_key = lambda i: src.fn.arg_names.index(i) if isinstance(i, str) else i
        constants = {cst_key(key): value for key, value in constants.items()}
        signature = {cst_key(key): value for key, value in src.signature.items()}
        mix_mode = metadata.mix_mode
        wrapper_src = generate_npu_wrapper_src(constants, signature, \
                                               workspace_size, mix_mode, \
                                               lock_num, lock_init_value)
        so_launcher_path = make_npu_launcher_stub(wrapper_src, debug_mode)
        # initialize launcher
        import importlib.util
        spec = importlib.util.spec_from_file_location("__triton_launcher", so_launcher_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        self.launch = getattr(mod, "launch")

    def __call__(self, *args, **kwargs):
        self.launch(*args, **kwargs)


class NPUDriver(DriverBase):
    def __init__(self):
        self.utils = NPUUtils()
        self.launcher_cls = NPULauncher
        super().__init__()

    @classmethod
    def is_active(cls):
        def test_npucompiler():
            from triton.backends.ascend.utils import _get_bisheng_path
            npucompiler = _get_bisheng_path()
            targets = subprocess.check_output([npucompiler, "-print-targets"]).decode().strip().split()
            return "hiipu64" in targets
        try:
            return test_npucompiler()
        except Exception as e_npucompiler:
            import warnings
            red = "\x1b[31;20m"
            reset = "\x1b[0m"
            warnings.warn(red + str(e_npucompiler) + reset)
            return False

    def get_current_target(self):
        backend = "npu"
        arch = self.utils.get_arch()
        warp_size = 0
        return GPUTarget(backend, arch, warp_size)

    def get_current_device(self):
        """
        Get current device
        """
        import torch
        import torch_npu
        return torch.npu.current_device()

    def set_current_device(self, device):
        """
        Set current device as the given device
        """
        import torch
        import torch_npu
        return torch.npu.set_device(device)

    def get_current_stream(self, device: Optional[int] = None) -> int:
        """
        Get stream for current device
        """
        # According to torch_npu, the content of a torch.npu.Stream is essentilly an rtStream_t
        # TODO: use CANN API instead of torchnpu
        import torch
        import torch_npu
        if device is None:
            device = self.get_current_device()
        return torch.npu.current_stream(device).npu_stream

    def get_benchmarker(self):
        from triton.testing import do_bench
        return do_bench

    def get_device_interface(self):
        import torch
        return torch.npu

    def get_empty_cache_for_benchmark(self):
        import torch
        cache_size = 192 * 1024 * 1024
        return torch.empty(cache_size // 4, dtype=torch.int, device='npu')


def make_npu_launcher_stub(src, debug=False):
    """
    Generate the launcher stub to launch the kernel
    """
    # try to get cached file
    so_cache_key = hashlib.sha256(src.encode("utf-8")).hexdigest()
    so_cache_manager = get_cache_manager(so_cache_key)
    # append the cxx11_abi value to the launcher name to avoid
    # linking to a launcher with wrong cxx11_abi.
    use_cxx11_abi = _check_cxx11_abi()
    name = f"launcher_cxx11abi{use_cxx11_abi}"
    suffix = sysconfig.get_config_var('EXT_SUFFIX')
    so_name = f"{name}{suffix}"

    if debug:
        dump_manager = get_dump_manager(so_cache_key)
        print(f"Dumping {name}.cxx to {dump_manager.cache_dir}")
        dump_manager.put(src, f"{name}.cxx", binary = False)

    cache_path = so_cache_manager.get_file(so_name)
    if cache_path is not None:
        return cache_path

    with tempfile.TemporaryDirectory() as tmpdir:
        if debug:
            so_cache_manager.put(src, f"{name}.cxx", binary=False)
        src_path = os.path.join(tmpdir, f"{name}.cxx")
        with open(src_path, "w") as f:
            f.write(src)
        enable_taskqueue = os.getenv("TRITON_ENABLE_TASKQUEUE", 'true').lower() in ('true', '1')
        if (enable_taskqueue):
            kernel_launcher_type = "torch"
        else:
            kernel_launcher_type = None
        so = _build_npu_ext(name, src_path, tmpdir, kernel_launcher=kernel_launcher_type)
        if debug:
            with open(so, "rb") as f:
                return dump_manager.put(f.read(), so_name, binary=True)
        with open(so, "rb") as f:
            return so_cache_manager.put(f.read(), so_name, binary=True)


def extract_device_print_code_from_cann():
    from triton.backends.ascend.utils import _get_bisheng_path
    ccec_compiler_bin_folder, _ = os.path.split(os.path.realpath(_get_bisheng_path()))
    ccec_compiler_folder, _ = os.path.split(ccec_compiler_bin_folder)
    clang_version = os.listdir(os.path.join(ccec_compiler_folder, "lib/clang/"))[0]
    ccelib_path = os.path.join(ccec_compiler_folder, f"lib/clang/{clang_version}/include/ccelib")

    def read_header(header_path):
        with open(os.path.join(ccelib_path, header_path), 'r') as f:
            code = f.read()

        # remove all #include "..."
        lines = code.splitlines()
        purged_lines = []
        for line in lines:
            normalized_line = ' '.join(line.split())
            if not normalized_line.startswith('#include "'):
                purged_lines.append(line)
        code = '\n'.join(purged_lines)

        # remove [aicore] functions
        aicore_positions = []
        for m in re.finditer('\[aicore\]', code):
            aicore_positions.append(m.start())

        def find_aicore_function_span(src, pos):
            for i in range(pos - 1, -1, -1):
                if src[i] == '}':  # this relies on that all [aicore] functions come after normal functions
                    left = i + 1
                    break
            n = len(src)
            brace_nest = 0
            for j in range(pos, n, 1):
                if src[j] == '{':
                    brace_nest += 1
                elif src[j] == '}':
                    brace_nest -= 1
                    if brace_nest == 0:
                        right = j
                        break
            return left, right

        new_code = ''
        segment_start = 0
        for pos in aicore_positions:
            left, right = find_aicore_function_span(code, pos)
            new_code += code[segment_start:left]
            segment_start = right + 1
        new_code += code[segment_start:]

        # remove __gm__ and rename macros
        new_code = new_code.replace('__gm__', ' ')
        new_code = new_code.replace('__CCELIB_RT_ERROR_NONE', 'RT_ERROR_NONE')
        new_code = new_code.replace('__CCELIB_RT_MEMORY_HBM', 'RT_MEMORY_HBM')
        new_code = new_code.replace('__CCELIB_RT_MEMCPY_HOST_TO_DEVICE', 'RT_MEMCPY_HOST_TO_DEVICE')
        new_code = new_code.replace('__CCELIB_RT_MEMCPY_DEVICE_TO_HOST', 'RT_MEMCPY_DEVICE_TO_HOST')
        return new_code

    # the following headers should be included in this order 
    return '\n'.join([
        read_header('common/common_impl.h'),
        read_header('internal/debug_tunnel/payload.h'),
        read_header('internal/debug_tunnel/payload_impl.h'),
        read_header('internal/debug_tunnel/tunnel.h'),
        read_header('internal/debug_tunnel/tunnel_impl.h')
    ])


# the template is from triton-adapter HEAD. Wrapping the generated kernel binary into a python module
def generate_npu_wrapper_src(constants, signature, workspace_size, mix_mode, lock_num, lock_ini_val):
    import os

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

    arg_decls = ', '.join(f"{_ty_to_cpp(ty)} arg{i}" for i, ty in signature.items())
    """
    args:
        int gridX, gridY, gridZ;
        rtStream_t stream;
        const void *functon;
        PyObject* packed_metadata, *launch_metadata;
        PyObject* launch_enter_hook, *launch_exit_hook;
        *args_expand
    """
    format = "iiiKKOOOO" + ''.join([_format_of(_extracted_ty(ty)) for ty in signature.values()])

    grid_info = {'X': 'i32', 'Y': 'i32', 'Z': 'i32'}

    enable_device_print = os.getenv(
        "TRITON_DEVICE_PRINT", 'false').lower() in ('true', '1')
    enable_taskqueue = os.getenv(
        "TRITON_ENABLE_TASKQUEUE", 'true').lower() in ('true', '1')
    enable_auto_map_parallel_blocks = _is_auto_map_parallel_blocks_enabled()
    npu_utils = NPUUtils()
    num_physical_blocks = npu_utils.get_aivector_core_num(
    ) if mix_mode == "aiv" else npu_utils.get_aicore_num()
    task_type = "MSPROF_GE_TASK_TYPE_AIV" if mix_mode == "aiv" else "MSPROF_GE_TASK_TYPE_AI_CORE"
    LINE_CHANGE_CHAR = chr(10)  # it is \n
    alloc_success_code = 'return 1;'
    sync_lock_fail_code = 'fprintf(stderr, "Error: syncBlockLock allocation failed\\n"); return;'
    workspace_fail_code = 'fprintf(stderr, "Error: workspace allocation failed\\n"); return;'

    cpp_device_pointer = """
typedef struct _DevicePtrInfo {
  void *dev_ptr;
  bool valid;
} DevicePtrInfo;

static inline DevicePtrInfo getPointer(PyObject *obj, int idx) {
  DevicePtrInfo ptr_info;
  ptr_info.dev_ptr = 0;
  ptr_info.valid = true;
  if (PyLong_Check(obj)) {
    ptr_info.dev_ptr = reinterpret_cast<void *>(PyLong_AsUnsignedLongLong(obj));
    return ptr_info;
  }
  if (obj == Py_None) {
    // valid nullptr
    return ptr_info;
  }
  PyObject *ptr = PyObject_GetAttrString(obj, "data_ptr");
  if(ptr){
    PyObject *empty_tuple = PyTuple_New(0);
    PyObject *ret = PyObject_Call(ptr, empty_tuple, NULL);
    Py_DECREF(empty_tuple);
    Py_DECREF(ptr);
    if (!PyLong_Check(ret)) {
      PyErr_SetString(PyExc_TypeError, "data_ptr method of Pointer object must return 64-bit int");
      ptr_info.valid = false;
      return ptr_info;
    }
    ptr_info.dev_ptr = reinterpret_cast<void *>(PyLong_AsUnsignedLongLong(ret));
    if(!ptr_info.dev_ptr)
      return ptr_info;
    Py_DECREF(ret);  // Thanks ChatGPT!
    return ptr_info;
  }
  PyErr_SetString(PyExc_TypeError, "Pointer argument must be either uint64 or have data_ptr method");
  return ptr_info;
}
"""

    cpp_msprof_extern = """
extern "C" {
  typedef int (* callback)(unsigned int type, void* data, unsigned int len);
  extern int MsprofReportApi(unsigned int  agingFlag, const MsprofApi *api);
  extern unsigned long int  MsprofSysCycleTime();
  extern int MsprofRegisterCallback(unsigned int moduleId, callback handle);
  static unsigned int __MsprofFlagL0  = 0;
  static unsigned int __MsprofFlagL1  = 0;

  int ProfCtrlHandle(unsigned int CtrlType, void* CtrlData, unsigned int DataLen) {
    if ((CtrlData == nullptr) || (DataLen == 0U)) {
      return 1;
    }

    if (CtrlType == 1) {
      MsprofCommandHandle* handle = (MsprofCommandHandle *)(CtrlData);
      if (handle->type >= 6)  // 6 is not used here
        return 1;
      if (handle->type == 1) {  // init - 0  , start - 1
        __MsprofFlagL0 = ((0x00000800ULL & handle->profSwitch) == 0x00000800ULL) ? 1 : 0;
        __MsprofFlagL1 = ((0x00000002ULL & handle->profSwitch) == 0x00000002ULL) ? 1 : 0;
      }
    }
    return 0;
  }
}
"""

    cpp_msprof_callback = """
  MsprofRegisterCallback(8, ProfCtrlHandle);      // 8 - CCE defined in msprof headerfile slog.h
"""

    cpp_msprof_call_before_launch = """
    unsigned long int beginTime = 0;
    unsigned long int endTime = 0;
    unsigned long int opNameHashID = 0;
    unsigned int threadId = 0;
    char* _kernelName = const_cast<char*>(name.c_str());
    size_t length = name.length();
    if (__MsprofFlagL0 || __MsprofFlagL1)
    {
      beginTime = MsprofSysCycleTime();
    }
"""

    cpp_msprof_call_after_launch = f"""
    if (__MsprofFlagL0 || __MsprofFlagL1)
    {{
      endTime = MsprofSysCycleTime();
      opNameHashID = MsprofGetHashId(_kernelName, length);
      threadId = (unsigned int)(syscall(SYS_gettid));
      MsprofApi info;
      info.level = MSPROF_REPORT_NODE_LEVEL;
      info.magicNumber = 0x5a5a;      //MSPROF_REPORT_DATA_MAGIC_NUM
      info.type = MSPROF_REPORT_NODE_LAUNCH_TYPE;
      info.threadId = threadId;
      info.reserve = 0;
      info.beginTime = beginTime;
      info.endTime = endTime;
      info.itemId = opNameHashID;
      MsprofReportApi(false, &info);
    }}
    if (__MsprofFlagL1)
    {{
      MsprofCompactInfo nodeBasicInfo;
      nodeBasicInfo.level = MSPROF_REPORT_NODE_LEVEL;
      nodeBasicInfo.magicNumber = 0x5a5a;      //MSPROF_REPORT_DATA_MAGIC_NUM
      nodeBasicInfo.type = MSPROF_REPORT_NODE_BASIC_INFO_TYPE;
      nodeBasicInfo.threadId = threadId;
      nodeBasicInfo.timeStamp = endTime;
      nodeBasicInfo.data.nodeBasicInfo.opName = opNameHashID;
      nodeBasicInfo.data.nodeBasicInfo.opType = opNameHashID;
      nodeBasicInfo.data.nodeBasicInfo.taskType = {task_type};
      nodeBasicInfo.data.nodeBasicInfo.blockDim = blockNum;
      MsprofReportCompactInfo(0, static_cast<void *>(&nodeBasicInfo), sizeof(MsprofCompactInfo));

      // Report tensor info
      int max_tensors_num = tensorShapes.size() < MSPROF_GE_TENSOR_DATA_NUM ? tensorShapes.size() : MSPROF_GE_TENSOR_DATA_NUM;
      MsprofAdditionalInfo tensorInfo;
      tensorInfo.level = MSPROF_REPORT_NODE_LEVEL;
      tensorInfo.type = MSPROF_REPORT_NODE_TENSOR_INFO_TYPE;
      tensorInfo.threadId = threadId;
      tensorInfo.timeStamp = endTime;
      auto profTensorData = reinterpret_cast<MsprofTensorInfo *>(tensorInfo.data);
      profTensorData->opName = opNameHashID;
      int tensorCount = 0;
      int dataTypes[MSPROF_GE_TENSOR_DATA_NUM];
      if (tensorShapes.size() > 0) {{
        {LINE_CHANGE_CHAR.join(
          f'dataTypes[{i}] = {convert_sigtype_to_int(ty[1:])};'
          for i, ty in signature.items()
          if ty.startswith("*") and i < 5
        )}
      }}
      for (int i = 0; i < tensorShapes.size() && tensorCount < MSPROF_GE_TENSOR_DATA_NUM; i++) {{
        auto fillTensorData = [&](int index, int tensorType) {{
          profTensorData->tensorData[index].tensorType = tensorType;
          profTensorData->tensorData[index].format = 2; // GeDataFormat: ND = 2
          profTensorData->tensorData[index].dataType = dataTypes[i];
          int nDim = tensorShapes[i].size();
          nDim = nDim < MSPROF_GE_TENSOR_DATA_SHAPE_LEN ? nDim : MSPROF_GE_TENSOR_DATA_SHAPE_LEN;
          for (int j = 0; j < nDim; j++) {{
            profTensorData->tensorData[index].shape[j] = tensorShapes[i][j];
          }}
          for (int j = nDim; j < MSPROF_GE_TENSOR_DATA_SHAPE_LEN; j++) {{
            profTensorData->tensorData[index].shape[j] = 0;
          }}
        }};
        int tensorType = (i < tensorKinds.size()) ? tensorKinds[i] : 0;  // DeFault tensor type is input
        if (tensorType == TENSOR_KIND_INPUT || tensorType == TENSOR_KIND_INPUT_OUTPUT) {{
          fillTensorData(tensorCount, MSPROF_GE_TENSOR_TYPE_INPUT);
          tensorCount++;
        }}
        if ((tensorType == TENSOR_KIND_OUTPUT || tensorType == TENSOR_KIND_INPUT_OUTPUT) && tensorCount < MSPROF_GE_TENSOR_DATA_NUM){{
          fillTensorData(tensorCount, MSPROF_GE_TENSOR_TYPE_OUTPUT);
          tensorCount++;
        }}
      }}
      profTensorData->tensorNum = tensorCount;
      MsprofReportAdditionalInfo(false, static_cast<void *>(&tensorInfo), sizeof(MsprofAdditionalInfo));
    }}
"""

    return f"""
#include <assert.h>
#include <stdbool.h>
#include <string>
#include <sys/syscall.h>
#include <vector>
#define PY_SSIZE_T_CLEAN
#include <Python.h>
{'#include <torch_npu/csrc/framework/OpCommand.h>' if enable_taskqueue else ''}
#include "experiment/runtime/runtime/rt.h"
#include <torch_npu/csrc/core/npu/NPUCachingAllocator.h>
{extract_device_print_code_from_cann() if enable_device_print else ''}

#define TENSOR_KIND_INPUT 0
#define TENSOR_KIND_OUTPUT 1
#define TENSOR_KIND_INPUT_OUTPUT 2

{cpp_msprof_extern}

{cpp_device_pointer}

static void _launch(const char* kernelName, const void* func, rtStream_t stream, int gridX, int gridY, int gridZ, std::vector<std::vector<int64_t>> &tensorShapes, std::vector<int> &tensorKinds{', ' + arg_decls if len(signature) > 0 else ''}) {{
  // only 1D parallelization is supported for NPU
  // Pointer type becomes flattend 1-D Memref tuple: base_ptr, data_ptr, offset, shape, stride
  // base_ptr offset shape and stride are not used, arbitrarily set for now
  std::string name = "";
  name.append(kernelName);
  {'auto launch_call = [=]() -> rtError_t' if enable_taskqueue else ''} {{
    uint32_t blockNum = gridX * gridY * gridZ;
    {'blockNum = std::min(blockNum, (uint32_t)' + str(num_physical_blocks) + ');' if enable_auto_map_parallel_blocks else ''}
    {'cce::internal::DebugTunnelData *DTData = cce::internal::DebugTunnel::Open(blockNum);' if enable_device_print else ''}
    rtError_t ret;
    void *ffts_addr = NULL;
    uint32_t ffts_len; ret = rtGetC2cCtrlAddr((uint64_t*)&ffts_addr, &ffts_len);
    if (ret != RT_ERROR_NONE) {{
      return {'ret' if enable_taskqueue else ''};
    }}
    c10::DataPtr syncBlockLock_ptr;
    c10::DataPtr workspace_addr_ptr;
    uint16_t ModuleId = 0;
    auto* npu_allocator = c10_npu::NPUCachingAllocator::get();
    {f'''
    uint64_t syncBlockLockSize = {lock_num} * sizeof(int64_t);
    syncBlockLock_ptr = npu_allocator->allocate(syncBlockLockSize);
    if (!syncBlockLock_ptr) {{
      {alloc_success_code if enable_taskqueue else sync_lock_fail_code}
    }}
    std::vector<int64_t> lockInitData({lock_num}, {lock_ini_val});
    ret = rtMemcpy(
        syncBlockLock_ptr.get(), syncBlockLockSize,
        reinterpret_cast<void *>(lockInitData.data()), syncBlockLockSize,
        RT_MEMCPY_HOST_TO_DEVICE
    );
    if (ret != RT_ERROR_NONE) {{
      return {'ret' if enable_taskqueue else ''};
    }}
    ''' if lock_num > 0 else ''}
    {f'''
    uint64_t totalWorkSpaceSize = {workspace_size} * blockNum;
    workspace_addr_ptr = npu_allocator->allocate(totalWorkSpaceSize);
    if (!workspace_addr_ptr) {{
      {alloc_success_code if enable_taskqueue else workspace_fail_code}
    }}
    ''' if workspace_size > 0 else ''}
    struct __attribute__((packed)) {{
      void* ffts_addr __attribute__((aligned(8)));
      void* syncBlockLock __attribute__((aligned(8)));
      void* workspace_addr __attribute__((aligned(8)));
      {' '.join(f'{_ty_to_cpp(ty)} arg{i} __attribute__((aligned({4 if ty[0] != "*" and ty[-2:] != "64" else 8})));' for i, ty in signature.items() if i not in constants)}
      {' '.join(f'{_ty_to_cpp(ty)} grid{mark} __attribute__((aligned(4)));' for mark, ty in grid_info.items())}
      {'void* DTData __attribute__((aligned(8)));' if enable_device_print else ''}
    }} args = {{
      static_cast<void*>(ffts_addr),
      {f'syncBlockLock_ptr.get()' if lock_num > 0 else 'nullptr'},
      {f'workspace_addr_ptr.get()' if workspace_size > 0 else 'nullptr'},
      {(', '.join(f'static_cast<{_ty_to_cpp(ty)}>(arg{i})' for i, ty in signature.items() if i not in constants) + ',') if len(signature) > 0 else ''}
      {', '.join(f'static_cast<{_ty_to_cpp(ty)}>(grid{mark})' for mark, ty in grid_info.items())}
      {', static_cast<void*>(DTData)' if enable_device_print else ''}
    }};
    {cpp_msprof_call_before_launch}
    ret = rtKernelLaunch(func, blockNum, static_cast<void*>(&args), sizeof(args), NULL, stream);
    {'void *&stream_ref = const_cast<void*&>(stream);' if enable_device_print else ''}
    {'cce::internal::DebugTunnel::Close(DTData, stream_ref);' if enable_device_print else ''}
    {cpp_msprof_call_after_launch}
    {'return ret;' if enable_taskqueue else ''}
   }};
   {'at_npu::native::OpCommand cmd; cmd.Name(name.c_str()).SetCustomHandler(launch_call).Run();' if enable_taskqueue else ''}
  return;
}}

// Extract tensor shape from PyObject
static std::vector<int64_t> _get_tensor_shape(PyObject *tensor) {{
  std::vector<int64_t> shape;

  // Early return if tensor is None or null
  if (!tensor || tensor == Py_None) {{
    return shape;
  }}

  // Calling tensor.size()
  PyObject* size_result = PyObject_CallMethod(tensor, "size", NULL);
  if (!size_result) {{
    return shape;
  }}
  // Using PySequence_Fast to improve access efficiency
  PyObject* seq = PySequence_Fast(size_result, "Expected a sequence from tensor.size()");
  if (seq) {{
    Py_ssize_t len = PySequence_Fast_GET_SIZE(seq);
    PyObject** items = PySequence_Fast_ITEMS(seq);
    for (Py_ssize_t i = 0; i < len; ++i) {{
      PyObject* dim = items[i];
      if (PyLong_Check(dim)) {{
        shape.push_back(PyLong_AsLong(dim));
      }}
    }}
  }}
  Py_DECREF(seq);
  Py_DECREF(size_result);
  return shape;
}}

static PyObject* launch(PyObject* self, PyObject* args) {{
  int gridX, gridY, gridZ;
  rtStream_t stream;
  const void *function;
  PyObject *packedMetadata = NULL;
  PyObject *launch_metadata = NULL;
  PyObject *launch_enter_hook = NULL;
  PyObject *launch_exit_hook = NULL;
  std::vector<std::vector<int64_t>> tensorShapes;
  {' '.join([f"{_extracted_ty(ty)} _arg{i}; " for i, ty in signature.items()])}
  if(!PyArg_ParseTuple(
      args, \"{format}\",
      &gridX, &gridY, &gridZ, &stream, &function,
      &packedMetadata, &launch_metadata,
      &launch_enter_hook, &launch_exit_hook
      {', ' + ', '.join(f"&_arg{i}" for i, ty in signature.items()) if len(signature) > 0 else ''}
      )
    ) {{
    return NULL;
  }}
  if (__MsprofFlagL1)
  {{
    {
      LINE_CHANGE_CHAR.join(
        f"{{ auto tmp = _get_tensor_shape(_arg{i}); if (!tmp.empty()) tensorShapes.push_back(tmp); }}"
        for i, ty in signature.items() if ty[0] == "*"
      )
    }
  }}

  if (launch_enter_hook != Py_None && !PyObject_CallObject(launch_enter_hook, args)) {{
    return NULL;
  }}

  // get kernel_name
  PyObject *kernelNameObj = PyDict_GetItemString(packedMetadata, "kernel_name");
  const char *kernelName = PyUnicode_AsUTF8(kernelNameObj);
  // get tensor_kinds
  std::vector<int> tensorKinds;
  PyObject *tensorKindList = PyDict_GetItemString(packedMetadata, "tensor_kinds");
  if (tensorKindList) {{
    int size = PyObject_Size(tensorKindList);
    for (int i = 0; i < size; i++) {{
      PyObject *kind = PySequence_GetItem(tensorKindList, i);
      tensorKinds.push_back(PyLong_AsLong(kind));
    }}
  }}

  // raise exception asap
  {"; ".join([f"DevicePtrInfo ptr_info{i} = getPointer(_arg{i}, {i}); if (!ptr_info{i}.valid) return NULL;" if ty[0]=="*" else "" for i, ty in signature.items()])};
  _launch(kernelName, function, stream, gridX, gridY, gridZ, tensorShapes, tensorKinds{', ' + ', '.join(f"ptr_info{i}.dev_ptr" if ty[0]=="*" else f"_arg{i}" for i, ty in signature.items()) if len(signature) > 0 else ''});
  if (PyErr_Occurred()) {{
    return NULL;
  }}
  if (launch_exit_hook != Py_None && !PyObject_CallObject(launch_exit_hook, args)) {{
    return NULL;
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
  {cpp_msprof_callback}
  return m;
}}
"""
