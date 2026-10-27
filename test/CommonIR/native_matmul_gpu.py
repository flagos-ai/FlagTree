import argparse
from pathlib import Path

import torch
import triton
import triton.language as tl
import triton.experimental.tle.language as tle


@triton.jit
def _matmul_gpu_kernel(a_ptr, b_ptr, c_ptr, M, N, K, stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
                       BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    a_buf = tle.gpu.alloc([BLOCK_M, BLOCK_K], dtype=tl.float16, layout=None, scope=tle.gpu.smem,
                          nv_mma_shared_layout=False)
    b_buf = tle.gpu.alloc([BLOCK_K, BLOCK_N], dtype=tl.float16, layout=None, scope=tle.gpu.smem,
                          nv_mma_shared_layout=False)

    for k0 in range(0, K, BLOCK_K):
        a_tile = a_ptr + offs_m[:, None] * stride_am + (k0 + offs_k)[None, :] * stride_ak
        b_tile = b_ptr + (k0 + offs_k)[:, None] * stride_bk + offs_n[None, :] * stride_bn
        tle.gpu.copy(a_tile, a_buf, [BLOCK_M, BLOCK_K])
        tle.gpu.copy(b_tile, b_buf, [BLOCK_K, BLOCK_N])
        a_vals = tle.gpu.to_tensor(a_buf, writable=False)
        b_vals = tle.gpu.to_tensor(b_buf, writable=False)
        acc += tl.dot(a_vals, b_vals, input_precision="ieee")

    c_buf = tle.gpu.alloc([BLOCK_M, BLOCK_N], dtype=tl.float32, layout=None, scope=tle.gpu.smem,
                          nv_mma_shared_layout=False)
    tle.gpu.store_tensor(acc, c_buf)
    c_vals = tle.gpu.to_tensor(c_buf, writable=False)
    c_tile = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_tile, c_vals, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


def _compile_options():
    from triton.compiler.compiler import make_backend
    from triton.runtime.driver import driver

    target = driver.active.get_current_target()
    backend = make_backend(target)
    options = backend.parse_options({
        "num_warps": 4,
        "num_stages": 3,
    })
    return target, backend, options


def _frontend_module(kernel, signature, constants):
    from triton.compiler.compiler import ASTSource
    from triton._C.libtriton import ir

    target, backend, options = _compile_options()
    context = ir.context()
    ir.load_dialects(context)
    backend.load_dialects(context)
    source = ASTSource(kernel, signature, constants)
    module = source.make_ir(target, options, backend.get_codegen_implementation(options), backend.get_module_map(),
                            context)
    if not module.verify():
        raise RuntimeError("frontend module.verify() failed")
    return module


def dump_tileir(path):
    constants = {
        "BLOCK_M": 16,
        "BLOCK_N": 16,
        "BLOCK_K": 16,
    }
    signature = {
        "a_ptr": "*fp16",
        "b_ptr": "*fp16",
        "c_ptr": "*fp32",
        "M": "i32",
        "N": "i32",
        "K": "i32",
        "stride_am": "i32",
        "stride_ak": "i32",
        "stride_bk": "i32",
        "stride_bn": "i32",
        "stride_cm": "i32",
        "stride_cn": "i32",
    }
    module = _frontend_module(_matmul_gpu_kernel, signature, constants)
    text = str(module)
    for needle in ("tile.alloc", "tile.copy", "tile.to_tensor", "tile.store_tensor"):
        if needle not in text:
            raise RuntimeError(f"expected {needle} in TileIR dump")
    Path(path).write_text(text, encoding="utf-8")
    print(f"[dump-tileir] wrote {path}")


def _add_commonir_to_ttgir(pm, enable_async=False):
    lib = __import__("triton._C.libtriton", fromlist=["tle", "iluvatar", "nvidia"])
    for ns in ("tle", "iluvatar", "nvidia"):
        add = getattr(getattr(getattr(getattr(lib, ns, None), "passes", None), "commonir", None), "add_to_ttgir", None)
        if add is not None:
            add(pm, enable_async)
            return ns
    raise RuntimeError("converted TTIR dump requires CommonIRToTTGIR (add_to_ttgir)")


def dump_ttir(path):
    from triton._C.libtriton import ir
    from triton._C.libtriton import passes

    constants = {
        "BLOCK_M": 16,
        "BLOCK_N": 16,
        "BLOCK_K": 16,
    }
    signature = {
        "a_ptr": "*fp16",
        "b_ptr": "*fp16",
        "c_ptr": "*fp32",
        "M": "i32",
        "N": "i32",
        "K": "i32",
        "stride_am": "i32",
        "stride_ak": "i32",
        "stride_bk": "i32",
        "stride_bn": "i32",
        "stride_cm": "i32",
        "stride_cn": "i32",
    }
    module = _frontend_module(_matmul_gpu_kernel, signature, constants)
    pm = ir.pass_manager(module.context)
    passes.common.add_inliner(pm)
    pm.run(module, "native_matmul_gpu.inliner")
    pm = ir.pass_manager(module.context)
    _add_commonir_to_ttgir(pm, False)
    pm.run(module, "native_matmul_gpu.tileir_to_ttgir")
    text = str(module)
    for needle in ("tile.alloc", "tile.copy", "tile.to_tensor", "tile.store_tensor"):
        if needle in text:
            raise RuntimeError(f"converted TTIR dump still contains {needle}")
    for needle in ("ttg.local_alloc", "ttg.local_load", "ttg.local_store"):
        if needle not in text:
            raise RuntimeError(f"converted TTIR dump expected {needle}")
    Path(path).write_text(text, encoding="utf-8")
    print(f"[dump-ttir] wrote {path}")


def run_check(M=64, N=64, K=64):
    torch.manual_seed(0)
    _musa = False
    try:
        import torch_musa  # noqa: F401
        _musa = torch.musa.is_available()
    except Exception:
        _musa = False
    device = "cuda" if torch.cuda.is_available() else ("musa" if _musa else "gcu")
    a = torch.randn((M, K), device=device, dtype=torch.float16)
    b = torch.randn((K, N), device=device, dtype=torch.float16)
    c = torch.empty((M, N), device=device, dtype=torch.float32)
    grid = (triton.cdiv(M, 16), triton.cdiv(N, 16))
    _matmul_gpu_kernel[grid](a, b, c, M, N, K, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0),
                             c.stride(1), BLOCK_M=16, BLOCK_N=16, BLOCK_K=16)
    ref = torch.matmul(a.float(), b.float())
    torch.testing.assert_close(c, ref, atol=2e-2, rtol=2e-2)
    print("[check] native_matmul_gpu precision PASS")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dump-tileir", nargs="?", const="native_matmul_gpu_tileir.mlir")
    parser.add_argument("--dump-ttir", nargs="?", const="native_matmul_gpu_ttir.mlir")
    parser.add_argument("--no-check", action="store_true")
    parser.add_argument("--M", type=int, default=64)
    parser.add_argument("--N", type=int, default=64)
    parser.add_argument("--K", type=int, default=64)
    args = parser.parse_args()
    if args.dump_tileir:
        dump_tileir(args.dump_tileir)
    if args.dump_ttir:
        dump_ttir(args.dump_ttir)
    if not args.no_check:
        run_check(args.M, args.N, args.K)


if __name__ == "__main__":
    main()