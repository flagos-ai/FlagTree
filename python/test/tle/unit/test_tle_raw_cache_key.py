import ast
import textwrap

import triton.language as tl
from triton.experimental.tle.raw import dialect
from triton.experimental.tle.raw.cache_key import (
    TLE_RAW_SOURCE_CACHE_KEY_ATTR,
    compute_tle_raw_source_cache_key,
)
from triton.runtime.jit import DependenciesFinder


def test_compute_tle_raw_source_cache_key_reads_file_content(tmp_path):
    cu_file = tmp_path / "kernel.cu"
    cu_file.write_text("__device__ void foo() {}\n")
    kwargs = {"name": "cuda", "file": cu_file}
    key1 = compute_tle_raw_source_cache_key(kwargs)

    cu_file.write_text("__device__ void bar() {}\n")
    key2 = compute_tle_raw_source_cache_key(kwargs)

    assert key1 != key2


def test_compute_tle_raw_source_cache_key_includes_extern_file(tmp_path):
    main = tmp_path / "main.cu"
    extern = tmp_path / "extern.py"
    main.write_text("__device__ void foo() {}\n")
    extern.write_text("helper = 1\n")

    with_extern = compute_tle_raw_source_cache_key({
        "name": "cuda",
        "file": main,
        "extern_file": extern,
    })
    without_extern = compute_tle_raw_source_cache_key({
        "name": "cuda",
        "file": main,
    })
    assert with_extern != without_extern


def test_compute_tle_raw_source_cache_key_includes_dialect_kwargs(tmp_path):
    cu_file = tmp_path / "kernel.cu"
    cu_file.write_text("__device__ void foo() {}\n")

    bc_key = compute_tle_raw_source_cache_key({
        "name": "cuda",
        "compiler": "clang",
        "target": "bc",
        "file": cu_file,
        "extern_func_name": "foo",
    })
    llvm_key = compute_tle_raw_source_cache_key({
        "name": "cuda",
        "compiler": "clang",
        "target": "llvm",
        "file": cu_file,
        "extern_func_name": "foo",
    })
    assert bc_key != llvm_key


def test_bind_tle_raw_source_cache_key_rereads_source_file(tmp_path):
    cu_file = tmp_path / "kernel.cu"
    cu_file.write_text("__device__ void foo() {}\n")

    @dialect(name="cuda", file=cu_file)
    def edsl(*args, **kwargs):
        ...

    key1 = getattr(edsl, TLE_RAW_SOURCE_CACHE_KEY_ATTR)()

    cu_file.write_text("__device__ void bar() {}\n")
    key2 = getattr(edsl, TLE_RAW_SOURCE_CACHE_KEY_ATTR)()

    assert key1 != key2


def test_jit_cache_key_includes_raw_source(tmp_path):
    cu_file = tmp_path / "vector-add.cu"
    cu_file.write_text(
        textwrap.dedent("""\
        __device__ void VectorAdd(float *C, const float *A, const float *B, const int N) {
          const int idx = blockIdx.x * blockDim.x + threadIdx.x;
          for (int i = idx; i < N; i += blockDim.x * gridDim.x) {
            C[i] = A[i] + B[i];
          }
        }
    """))

    @dialect(name="cuda", file=cu_file)
    def edsl(*args, **kwargs):
        ...

    kernel_src = textwrap.dedent("""\
        def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
            tle_raw.call(edsl, [output_ptr, x_ptr, y_ptr, n_elements])
    """)

    def make_cache_key() -> str:
        finder = DependenciesFinder(
            name="add_kernel",
            globals={
                "edsl": edsl, "tl": tl, "tle_raw": __import__("triton.experimental.tle.language.raw", fromlist=["raw"])
            },
            nonlocals={},
            src=kernel_src,
        )
        finder.visit(ast.parse(kernel_src))
        return finder.ret

    key1 = make_cache_key()

    cu_file.write_text(
        textwrap.dedent("""\
        __device__ void VectorAdd(float *C, const float *A, const float *B, const int N) {
          const int idx = blockIdx.x * blockDim.x + threadIdx.x;
          for (int i = idx; i < N; i += blockDim.x * gridDim.x) {
            C[i] = A[i] + B[i] + 1.0f;
          }
        }
    """))
    key2 = make_cache_key()

    assert key1 != key2


def test_mlir_dialect_cache_key_changes_with_edsl_source():

    @dialect(name="mlir")
    def edsl_v1():
        pass

    @dialect(name="mlir")
    def edsl_v2():
        return 1

    assert getattr(edsl_v1, TLE_RAW_SOURCE_CACHE_KEY_ATTR)() != getattr(edsl_v2, TLE_RAW_SOURCE_CACHE_KEY_ATTR)()
