# FlagTree Backend Specialization 统一设计（Python）

FlagTree 设计的后端统一特化，目的是整合后端接入范式，对后端的特化实现清晰化管理，为后端适配 Triton 版本升级迁移提供工程基础。具体实施方案是将各后端对 Triton 的特化，从以往的 fork 仓库直接修改并单独维护，标准化为定义接口并在后端目录下给出差异化实现。

## 1. 原则与规范

主干代码在保证缺省逻辑不变的基础上，允许调用接口，然后在后端目录中（third_party/backendxxx/）实现特化。主干代码原则上不允许直接出现某后端的特化实现，也不允许对后端做选择判断后特化实现。<br>
得益于 Python 的语法能力，通过统一的接口 spec、spec_func 接入特化函数字符串，特化函数由后端按需添加。当多后端对同一段主干代码有特化需求时，应协调保障多方特化功能。<br>

## 2. 接口

FlagTree 为 Python 代码的后端特化提供两种接口：spec 接口特化函数实现，spec_func 接口特化函数定义。由于调用了当前活动驱动类中的成员，只能在活动后端发现并激活后使用，因此一般来说只能用于一个局部作用域内。如果用在 py 文件的全局作用域且该文件在启动初期被 import，则会报错。

- python/triton/runtime/driver.py
```python
# flagtree backend specialization
def spec(function_name: str, *args, **kwargs):
    if hasattr(driver.active, "spec"):
        spec = driver.active.spec
        if hasattr(spec, function_name):
            func = getattr(spec, function_name)
            return func(*args, **kwargs)
    return None
```

```python
# flagtree backend func specialization
def spec_func(function_name: str):
    if hasattr(driver.active, "spec"):
        spec = driver.active.spec
        if hasattr(spec, function_name):
            func = getattr(spec, function_name)
            return func
    return None
```

## 3. 后端入口注册

后端驱动类下需添加 spec 成员，注册该后端目录下的特化实现入口（以 iluvatar 后端为例）。注意原有的 utils 成员需改成 property，否则会循环注册。

- third_party/iluvatar/backend/driver.py
```python
class BackendDriver(GPUDriver):
    def __init__(self):
        # self.utils = CudaUtils()  # 对于 Triton 3.1 需改为 property
        self.launcher_cls = CudaLauncher
        # flagtree backend specialization
        from triton.backends.iluvatar import spec
        self.spec = spec
        super().__init__()
    @property
    def utils(self):
        return CudaUtils()
```

## 4. 使用实例

### 4.1 情形一：特化函数实现（spec）

#### 4.1.1 第一步：调用统一特化

本例中，缺省实现是 return tl.tensor(...)，特化函数起名为 atomic_add_int64。

- python/triton/language/semantic.py
```python
def atomic_add(ptr: tl.tensor, val: tl.tensor, mask: tl.tensor, sem: str, scope: str, builder: ir.builder) -> tl.tensor:
    ...
    rett = tl.tensor(builder.create_atomic_rmw(op, ptr.handle, val.handle, mask.handle, sem, scope), val.type)
    # flagtree backend specialization
    from triton.runtime.driver import spec
    return spec("atomic_add_int64", sca_ty, builder, val, ptr, mask, sem, scope) or rett
```

#### 4.1.2 第二步：注册特化方法

- <strong>third_party/iluvatar/backend/spec/</strong>\_\_init\_\_.py
```python
from .triton.language.semantic import *
__all__ = [
    ..., "atomic_add_int64", ...
]
```

#### 4.1.3 第三步：实现特化函数

- <strong>third_party/iluvatar/backend/spec/</strong>triton/language/semantic.py
```python
def atomic_add_int64(sca_ty, builder, val, ptr, mask, sem, scope):
    from triton.language.semantic import full, and_, cast, lshr, bitcast, add, _bool_like, where, shl, or_
    ...
```

需要注意的是，如果需要特化一个判断条件（即特化函数返回布尔类型），那么应设计为后端特化时返回 True（缺省返回 False）。这是为了与 spec 方法当后端未做相应函数的特化时缺省返回 None 保持判断结果一致，保证缺省实现不变。

### 4.2 情形二：特化函数定义（spec_func）

#### 4.2.1 第一步：调用统一特化

- python/triton/ops/matmul.py
```python
@jit
def _kernel(A, B, C, M, N, K, ...):
    ...

class _matmul(torch.autograd.Function):
    # flagtree backend specialization
    from triton.runtime.driver import spec_func
    kernel = spec_func("matmul_kernel") or _kernel
    ...
```

#### 4.2.2 第二步：注册特化方法

- <strong>third_party/iluvatar/backend/spec/</strong>\_\_init\_\_.py
```python
from .triton.ops.matmul import *
__all__ = [
    ..., "matmul_kernel", ...
]
```

#### 4.2.3 第三步：实现特化函数

- <strong>third_party/iluvatar/backend/spec/</strong>triton/ops/matmul.py
```python
def matmul_kernel(grid, a, b, c, M, N, K, ...):
    from triton.ops.matmul import get_configs_io_bound
    ...

    @jit
    def _kernel(A, B, C, M, N, K, ...):
        ...
    return _kernel[grid](a, b, c, M, N, K, ...)
```

### 4.3 情形三：添加新的方法（例如 spec_semantic_func）

在 python/triton/language/ 目录下常有后端需要添加新的 tl 原语。上文介绍过，spec_func 在例如 semantic.py 的全局 scope 下是不能调用的，因此添加方法需要使用本节介绍的方案。

#### 4.3.1 第一步：调用统一特化

自动遍历后端定义在 core_ext_spec_func_list 列表中的方法，加入到本模块（tl.core）。当然，也可以按需加入到其他模块（例如 tl）。注意对于 semantic.py 方法名需加上 ext_semantic_ 前缀，与 core.py 的重名函数区分开。

- python/triton/language/core.py
```python
def spec_core_func(spec):
    import sys
    current_module_name = __name__
    parent_module_name = '.'.join(current_module_name.split('.')[:-1])

    for spec_func_name in spec.core_ext_spec_func_list:
        if hasattr(spec, spec_func_name):
            spec_func = getattr(spec, spec_func_name)
            # triton.language
            setattr(sys.modules[parent_module_name], spec_func.__name__, spec_func)
            # triton.language.core
            setattr(sys.modules[__name__], spec_func.__name__, spec_func)
```

#### 4.3.2 第二步：注册后端入口

- third_party/ascend/backend/driver.py
```python
class NPUDriver(DriverBase):
    def __init__(self):
        self.utils = NPUUtils()
        self.launcher_cls = NPULauncher
        # flagtree backend specialization
        from triton.backends.ascend import spec
        self.spec = spec  # 4.1 情形一
        from triton.language.core import spec_core_func
        spec_core_func(spec)  # 4.3 情形三
        from triton.language.semantic import spec_semantic_func
        spec_semantic_func(spec)  # 4.3 情形三
        from triton.language.standard import spec_standard_func
        spec_standard_func(spec)  # 4.3 情形三
        from triton.language.math import spec_math_func
        spec_math_func(spec)  # 4.4 情形四
        super().__init__()
```

- <strong>third_party/ascend/backend/spec/</strong>\_\_init\_\_.py
```python
from .triton.language.semantic import *
__all__  = [
    "core_ext_spec_func_list",
]
```

#### 4.3.3 第三步：实现特化函数

- <strong>third_party/ascend/backend/spec/</strong>triton/language/core.py
```python
@_tensor_member_fn
@builtin
def gather(src, index, axis, _builder=None):
    ...

core_ext_spec_func_list = [
    "gather", ...
]
```

### 4.4 情形四：修改或新增 tl.math 方法

第一、第二步与 4.3 大体一致，第三步的区别在于应按 Triton 规范实现于 libdevice.py。

#### 4.4.1 第一步：调用统一特化

```python
def spec_math_func(spec):
    import sys
    current_module_name = __name__
    parent_module_name = '.'.join(current_module_name.split('.')[:-1])

    for spec_func_name in spec.math_ext_base_func_list:
        if hasattr(spec, spec_func_name):
            spec_func = getattr(spec, spec_func_name)
            # triton.language
            setattr(sys.modules[parent_module_name], spec_func.__name__, spec_func)
            # triton.language.math
            setattr(sys.modules[__name__], spec_func.__name__, spec_func)

    for spec_func_name in spec.math_ext_spec_func_list:
        if hasattr(spec, spec_func_name):
            spec_func = getattr(spec, spec_func_name)
            # triton.language.math
            setattr(sys.modules[__name__], spec_func.__name__, spec_func)
```

#### 4.4.2 第二步：注册后端入口

- third_party/ascend/backend/driver.py
```python
class NPUDriver(DriverBase):
    def __init__(self):
        self.utils = NPUUtils()
        self.launcher_cls = NPULauncher
        # flagtree backend specialization
        from triton.backends.ascend import spec
        self.spec = spec  # 4.1 情形一
        from triton.language.math import spec_math_func
        spec_math_func(spec)  # 4.4 情形四
        super().__init__()
```

#### 4.4.3 第三步：实现特化函数

- <strong>third_party/ascend/backend/spec/</strong>triton/language/math.py
```python
import triton.language as language
exp = language.extra.ascend.libdevice.exp
math_ext_base_func_list = [
    "exp", ...  # tl.math 原有方法，但实现有特化，例如支持的 dtype 不同
]
math_ext_spec_func_list = [
    "isnan", ...  # 后端向 tl.math 新增的方法
]
```

- third_party/ascend/language/ascend/libdevice.py
```python
from triton.language import core, semantic

@core.extern
@_check_dtype(dtypes=["bf16", "fp16", "fp32"])
@_add_math_1arg_docstr("exponential")
@core._tensor_member_fn
def exp(x, _builder=None):
    x = semantic.to_tensor(x, _builder)
    return core.tensor(_builder.create_exp(x.handle), x.type)

@core.extern
def isnan(arg0, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"),): ("__hmf_isnan", core.dtype("int1")),
            (core.dtype("fp16"),): ("__hmf_isnan", core.dtype("int1")),
            (core.dtype("bf16"),): ("__hmf_isnan", core.dtype("int1")),
        }, is_pure=True, _builder=_builder)
```
