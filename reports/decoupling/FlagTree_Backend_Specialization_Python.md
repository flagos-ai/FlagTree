# FlagTree Backend Specialization 统一设计（Python）

## 1. 接口
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

## 2. 后端入口注册
后端驱动类下需添加 spec 成员，注册该后端目录下的特化实现入口（本文以 iluvatar 后端为例）。注意原有的 utils 成员需改成 property，否则会循环注册。
- third_party/iluvatar/backend/driver.py
```python
class BackendDriver(GPUDriver):
    def __init__(self):
        # self.utils = CudaUtils()  # 改为 property
        self.launcher_cls = CudaLauncher
        # flagtree backend specialization
        from triton.backends.iluvatar import spec
        self.spec = spec
        super().__init__()
    @property
    def utils(self):
        return CudaUtils()
```

## 3. 使用实例

### 3.1 情形一：特化函数实现（spec）

#### 调用统一特化
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

#### 注册特化方法
- <strong>third_party/iluvatar/backend/spec/</strong>\_\_init\_\_.py
```python
__all__ = [
    ..., "atomic_add_int64", ...
]
```

#### 实现特化函数
- <strong>third_party/iluvatar/backend/spec/</strong>triton/language/semantic.py
```python
def atomic_add_int64(sca_ty, builder, val, ptr, mask, sem, scope):
    from triton.language.semantic import full, and_, cast, lshr, bitcast, add, _bool_like, where, shl, or_
    ...
```

### 3.2 情形二：特化函数定义（spec_func）

#### 调用统一特化
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

#### 注册特化方法
- <strong>third_party/iluvatar/backend/spec/</strong>\_\_init\_\_.py
```python
__all__ = [
    ..., "matmul_kernel", ...
]
```

#### 实现特化函数
```python
def matmul_kernel(grid, a, b, c, M, N, K, ...):
    from triton.ops.matmul import get_configs_io_bound
    ...

    @jit
    def _kernel(A, B, C, M, N, K, ...):
        ...

    return _kernel[grid](a, b, c, M, N, K, ...)
```
