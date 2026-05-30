# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

import os
from typing import Callable, Dict


class BackendStrategyRegistry:

    def __init__(self):
        self.strategies: Dict[str:Dict[str, Callable]] = {}

    def register(self, category: str, method: str):

        def decorator(func: Callable):
            if category not in self.strategies:
                self.strategies[category] = {}
            if method in self.strategies[category]:
                raise ValueError(f"Strategy {method} already registered")
            self.strategies[category][method] = func
            return func

        return decorator

    def execute_func(self, category, method, *args, **kwargs):
        if category not in self.strategies:
            raise ValueError(f"Strategy {category} not registered")
        if method not in self.strategies[category]:
            raise ValueError(f"Strategy {method} not registered")
        return self.strategies[category][method](*args, **kwargs)

    def list_categories(self):
        return list(self.strategies.keys())

    def list_methods(self, category):
        if category not in self.strategies:
            raise ValueError(f"Strategy {category} not registered")
        return list(self.strategies[category].keys())


class _LazyBackendStrategyRegister:

    def __init__(self):
        self._instance = None

    def _get_instance(self):
        if self._instance is None:
            self._instance = BackendStrategyRegistry()
        return self._instance

    def register(self, *args, **kwargs):
        return self._get_instance().register(*args, **kwargs)

    def execute_func(self, *args, **kwargs):
        return self._get_instance().execute_func(*args, **kwargs)


backend_strategy_registry = _LazyBackendStrategyRegister()


@backend_strategy_registry.register("mindspore", "version_hash")
def version_hash():
    import mindspore
    return [str(mindspore.version)]


@backend_strategy_registry.register("torch_npu", "version_hash")
def version_hash():
    import torch
    import torch_npu
    return [torch.version.git_version, torch_npu.version.git_version]


@backend_strategy_registry.register("mindspore", "cxx_abi")
def get_mindspore_cxx_abi():
    return 0


@backend_strategy_registry.register("torch_npu", "cxx_abi")
def get_torch_cxx_abi():
    import torch
    return 1 if torch._C._GLIBCXX_USE_CXX11_ABI else 0


@backend_strategy_registry.register("mindspore", "type_convert")
def type_convert():
    import mindspore
    import numpy as np
    MINDSPORE_TO_NUMPY_DTYPE = {
        mindspore.float32: np.float32,
        mindspore.float64: np.float64,
        mindspore.float16: np.float16,
        mindspore.int8: np.int8,
        mindspore.uint8: np.uint8,
        mindspore.int16: np.int16,
        mindspore.int32: np.int32,
        mindspore.int64: np.int64,
        mindspore.bool: np.bool_,
        mindspore.complex64: np.complex64,
        mindspore.complex128: np.complex128,
    }
    return MINDSPORE_TO_NUMPY_DTYPE


@backend_strategy_registry.register("torch_npu", "type_convert")
def type_convert():
    import torch
    import numpy as np
    TORCH_TO_NUMPY_DTYPE = {
        torch.float32: np.float32,
        torch.float64: np.float64,
        torch.float16: np.float16,
        torch.int8: np.int8,
        torch.uint8: np.uint8,
        torch.int16: np.int16,
        torch.int32: np.int32,
        torch.int64: np.int64,
        torch.bool: np.bool_,
        torch.complex64: np.complex64,
        torch.complex128: np.complex128,
    }
    return TORCH_TO_NUMPY_DTYPE


@backend_strategy_registry.register("mindspore", "get_device_interface")
def get_device_interface():
    import mindspore
    return mindspore


@backend_strategy_registry.register("torch_npu", "get_device_interface")
def get_device_interface():
    import torch
    return torch.npu


@backend_strategy_registry.register("mindspore", "get_empty_tensor")
def get_empty_tensor(size):
    import mindspore
    return mindspore.mint.empty(size, dtype=mindspore.int32)


@backend_strategy_registry.register("torch_npu", "get_empty_tensor")
def get_empty_tensor(size):
    import torch
    return torch.empty(size, dtype=torch.int32, device='npu')


@backend_strategy_registry.register("mindspore", "get_tensor_params_shape")
def get_tensor_params_shape(*args):
    import mindspore
    tensor_params = [arg for arg in args if isinstance(arg, mindspore.Tensor)]
    tensor_params_shape = []
    for t in tensor_params:
        tensor_params_shape.append([s for s in t.shape])
    return tensor_params_shape


@backend_strategy_registry.register("torch_npu", "get_tensor_params_shape")
def get_tensor_params_shape(*args):
    import torch
    tensor_params = [arg for arg in args if isinstance(arg, torch.Tensor)]
    tensor_params_shape = []
    for t in tensor_params:
        tensor_params_shape.append([s for s in t.shape])
    return tensor_params_shape


@backend_strategy_registry.register("mindspore", "get_cc_cmd")
def get_cc_cmd(build_pch):
    import mindspore
    mindspore_path = os.path.dirname(os.path.realpath(mindspore.__file__))
    cc_cmd = [
        f"-I{mindspore_path}",
        f"-I{os.path.join(mindspore_path, 'include/')}",
        f"-I{os.path.join(mindspore_path, 'include/third_party')}",
        f"-I{os.path.join(mindspore_path, 'include/third_party/robin_hood_hashing')}",
        f"-I{os.path.join(mindspore_path, 'include/mindspore/core')}",
        f"-I{os.path.join(mindspore_path, 'include/mindspore/core/include')}",
        f"-I{os.path.join(mindspore_path, 'include/mindspore/core/mindrt/include')}",
        f"-I{os.path.join(mindspore_path, 'include/mindspore/ccsrc')}",
        f"-I{os.path.join(mindspore_path, 'include/mindspore/ccsrc/include')}",
        f"-I{os.path.join(mindspore_path, 'include/mindspore/ops')}",
        f"-I{os.path.join(mindspore_path, 'include/mindspore/ops/include')}",
        f"-D_GLIBCXX_USE_CXX11_ABI={get_mindspore_cxx_abi()}",
        "-DENABLE_FAST_HASH_TABLE=1",
    ]
    if not build_pch:
        cc_cmd += [
            f"-L{os.path.join(mindspore_path, 'lib')}",
            f"-lmindspore_pynative_utils",
        ]
    return cc_cmd


@backend_strategy_registry.register("torch_npu", "get_cc_cmd")
def get_cc_cmd(build_pch):
    import torch
    import torch_npu
    torch_path = os.path.dirname(os.path.realpath(torch.__file__))
    torch_npu_path = os.path.dirname(os.path.realpath(torch_npu.__file__))
    cc_cmd = [
        f"-I{os.path.join(torch_path, 'include')}",
        f"-I{os.path.join(torch_npu_path, 'include')}",
        f"-D_GLIBCXX_USE_CXX11_ABI={get_torch_cxx_abi()}",
    ]
    if not build_pch:
        cc_cmd += [
            f"-L{os.path.join(torch_npu_path, 'lib')}",
            f"-ltorch_npu",
        ]
    return cc_cmd


@backend_strategy_registry.register("mindspore", "get_current_device")
def get_current_device():
    import mindspore
    return mindspore.get_current_device().device_id


@backend_strategy_registry.register("torch_npu", "get_current_device")
def get_current_device():
    import torch
    import torch_npu
    return torch.npu.current_device()


@backend_strategy_registry.register("mindspore", "set_current_device")
def set_current_device(device_id):
    import mindspore
    return mindspore.set_device("Ascend", device_id)


@backend_strategy_registry.register("torch_npu", "set_current_device")
def set_current_device(device_id):
    import torch
    import torch_npu
    return torch.npu.set_device(device_id)


@backend_strategy_registry.register("mindspore", "get_current_stream")
def get_current_stream(device):
    import mindspore
    try:
        return mindspore.current_stream().stream_ptr()
    except Exception:
        return mindspore.current_stream().id


@backend_strategy_registry.register("torch_npu", "get_current_stream")
def get_current_stream(device):
    import torch
    import torch_npu
    if device is None:
        device = torch.npu.current_device()
    if hasattr(torch_npu._C, "_npu_getCurrentRawStreamNoWait"):
        from torch_npu._C import _npu_getCurrentRawStreamNoWait
        return _npu_getCurrentRawStreamNoWait(device)
    else:
        from torch_npu._C import _npu_getCurrentRawStream
        return _npu_getCurrentRawStream(device)


@backend_strategy_registry.register("mindspore", "header_file")
def header_file(enable_taskqueue):
    return f'''#include "include/utils/device_manager_conf.h"
#include "include/runtime/hardware_abstract/device_context/device_context_manager.h"
#include "include/mindspore/ops/kernel/ascend/aclnn/pyboost_impl/aclnn_utils.h"
{'#include "include/pynative/utils/runtime/op_executor.h"' if {enable_taskqueue} else ''}
{'#include "include/runtime/pipeline/pipeline.h"' if {enable_taskqueue} else ''}'''


@backend_strategy_registry.register("torch_npu", "header_file")
def header_file(enable_taskqueue):
    return f'''#include <ATen/ATen.h>
#include <torch_npu/csrc/core/npu/NPUWorkspaceAllocator.h>
{'#include <torch_npu/csrc/framework/OpCommand.h>' if {enable_taskqueue} else ''}'''


@backend_strategy_registry.register("mindspore", "allocate_memory")
def allocate_memory(size, stream):
    return f'''auto work_ptr = std::make_shared<mindspore::kernel::pyboost::MemBlock>(device_context, {size}, reinterpret_cast<uint64_t>({stream}));
    workspace_addr_ptr = work_ptr->ptr_;'''


@backend_strategy_registry.register("torch_npu", "allocate_memory")
def allocate_memory(size, stream):
    return f"workspace_addr_ptr = const_cast<void *>(at::empty({size}, at::TensorOptions().device(at::kPrivateUse1).dtype(at::kByte)).storage().data());"


@backend_strategy_registry.register("mindspore", "allocate_sync_block_lock")
def allocate_sync_block_lock(size, stream):
    return f'''auto sync_ptr = std::make_shared<mindspore::kernel::pyboost::MemBlock>(device_context, {size}, reinterpret_cast<uint64_t>({stream}));
    syncBlockLock_ptr = work_ptr->ptr_;'''


@backend_strategy_registry.register("torch_npu", "allocate_sync_block_lock")
def allocate_sync_block_lock(size, stream):
    return f"syncBlockLock_ptr = const_cast<void *>(at_npu::native::allocate_workspace({size}, {stream}).storage().data());"


@backend_strategy_registry.register("mindspore", "pre_launch")
def pre_launch(first_call):
    if first_call:
        return '''static auto device_context = mindspore::device::DeviceContextManager::GetInstance().GetOrCreateDeviceContext({mindspore::device::DeviceType::kAscend, mindspore::DeviceManagerConf::GetInstance()->device_id()});
        device_context->device_res_manager_->BindDeviceToCurrentThread(false);'''
    else:
        return '''device_context->device_res_manager_->BindDeviceToCurrentThread(false);'''


@backend_strategy_registry.register("torch_npu", "pre_launch")
def pre_launch(first_call):
    return ""


@backend_strategy_registry.register("mindspore", "async_launch")
def async_launch(func):
    return f'''mindspore::runtime::OpExecutor::DispatchLaunchTask({func});'''


@backend_strategy_registry.register("torch_npu", "async_launch")
def async_launch(func):
    return f'''at_npu::native::OpCommand cmd;
    cmd.Name(name.c_str()).SetCustomHandler({func}).Run();'''
