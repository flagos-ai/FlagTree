#!/usr/bin/env python3
"""Standalone launch_kernel (LK) board smoke test for the RPU backend.

This validates ONLY the on-board ``launch_kernel`` dispatch path and nothing
else. It is meant to be invoked directly by CI on an RPU board runner:

    python3 third_party/rpu/python/test/board/lk_board_smoke.py

What it does, end to end, for each case in the op suite:

  1. compile the kernel with ``RPU_OUTPUT_ELF=1`` so the backend emits the
     ``.ref`` ELF the launch_kernel runtime consumes;
  2. stage one raw little-endian SPM buffer per kernel argument, laid out the
     way the device kernel expects to read it (see per-op notes below);
  3. dispatch the kernel on the real device through the ``launch_kernel_runner``
     CLI (a front-end over the ``rhino-launch-kernel`` runtime library);
  4. read the output buffer back and compare it to the numpy golden.

Op coverage (all run on ``/dev/rpu``):
  * elementwise (byte-exact): add, mul, fused mul-add / add-mul, relu, maximum,
    broadcast_add (rank-2 [N,C] + rank-1 [N]);
  * fp32-accumulating (fp16 tolerance): gemm, softmax, reduce_sum over axis 0/1.

Per-op SPM layout contracts that the device imposes (mirrored when staging):
  * gemm: ``ctx.mma`` consumes the RHS in column-major, so RHS is staged as
    ``rhs.T`` (plain row-major fails numerically);
  * softmax: ``ctx.load_contig<NVEC>`` reads a full V256 (256 fp16), so the
    unused input lanes are padded with fp16 -inf — ``exp(-inf - max) = 0`` keeps
    the denominator equal to the sum over the valid elements;
  * reduce_sum_axis{0,1}: the emit materialises the result as the full [M,N]
    matrix (the DSL rejects a 1xN / Mx1 store), with every row (axis 0) or every
    column (axis 1) holding the singleton output, so the golden is tiled/repeated
    to [M,N] and the whole matrix is compared;
  * broadcast_add: the rank-1 operand is staged in the V256 broadcast layout the
    rank-2 ``+`` reads (lane block ``16*tid .. 16*tid+16`` holds ``b[tid]``).

Prerequisites (auto-detected). When any is missing the test SKIPS with a
clear reason and exits 0, so it is harmless on non-board CI runners. Pass
``--require-board`` (or set ``RPU_LK_REQUIRE=1``) to turn a skip into a
failure — use that on the dedicated board runner so a misconfigured
environment is caught instead of silently skipped.

  * triton with the RPU backend importable
  * a usable RPU clang under ``RPU_LLVM_ROOT`` (and ``RPU_ASM_PATH`` for the
    assembler), to compile the kernel and emit the ``.ref``
  * the ``launch_kernel_runner`` CLI: ``RPU_LK_RUNNER``, or on PATH
  * the ``/dev/rpu`` device node

Environment knobs:
  RPU_LLVM_ROOT       install prefix containing bin/clang (or build/bin/clang)
  RPU_ASM_PATH        path to the RPU assembler executable
  RPU_LK_RUNNER       path to launch_kernel_runner (else searched on PATH)
  RPU_LK_SUDO=0       do not prefix the runner with ``sudo -n``
  RPU_LK_REQUIRE=1    treat skips as failures

Exit codes: 0 = pass (or skip when not required); 1 = failure.
"""

import argparse
import os
import shutil
import signal
import subprocess
import sys
import tempfile
from pathlib import Path

# V256 is 256 fp16 = 512 bytes. load_contig/store_contig always touch at least
# one full V256, so every LK argument buffer is padded up to a 512-byte
# multiple to keep each kernel access inside its own SPM allocation.
V256_BYTES = 512

# fp16 tolerance for ops that accumulate in fp32 and round to fp16 (gemm /
# softmax / reduce_sum): |out - gold| <= atol + rtol*|gold| per element.
FP16_ATOL = 1.5e-3
FP16_RTOL = 2e-3


class Skip(Exception):
    """Raised when a prerequisite for the LK path is unavailable."""


def _v256_size(used_bytes):
    n = (used_bytes + V256_BYTES - 1) // V256_BYTES
    return max(n, 1) * V256_BYTES


def _resolve_clang():
    root = os.environ.get("RPU_LLVM_ROOT")
    if not root:
        raise Skip("RPU_LLVM_ROOT is not set")
    for rel in ("bin/clang", "build/bin/clang"):
        if (Path(root) / rel).exists():
            return
    raise Skip(f"no clang under RPU_LLVM_ROOT={root} (tried bin/clang, build/bin/clang)")


def _resolve_runner():
    """Return the launch_kernel_runner CLI path, or raise Skip."""
    runner = os.environ.get("RPU_LK_RUNNER")
    if not runner:
        runner = shutil.which("launch_kernel_runner")
    if not runner or not Path(runner).exists():
        raise Skip("launch_kernel_runner not found (set RPU_LK_RUNNER to the "
                   "built CLI, or put launch_kernel_runner on PATH)")
    return runner


def _set_compile_env(work_dir):
    os.environ.setdefault("TRITON_RPU_ACTIVE", "1")
    os.environ.setdefault("RPU_ARCH", "rpu-v1")
    os.environ.setdefault("RPU_LOGICAL_LANES", "16")
    os.environ.setdefault("TRITON_CODEGEN_BACKENDS", "rpu")
    os.environ["RPU_OUTPUT_ELF"] = "1"
    os.environ["TRITON_ALWAYS_COMPILE"] = "1"
    os.environ["TRITON_CACHE_DIR"] = str(Path(work_dir) / "cache")


def _v256_pad(arr, fill=0.0):
    """Return arr flattened and padded up to a whole V256 (256 fp16) with fill.

    fill="-inf" pads with fp16 negative infinity (0xFC00)."""
    import numpy as np

    arr = np.asarray(arr, dtype=np.float16).flatten()
    region = ((arr.size + 255) // 256) * 256
    if fill == "-inf":
        buf = np.full(region, 0xFC00, dtype=np.uint16).view(np.float16).copy()
    else:
        buf = np.full(region, np.float16(fill), dtype=np.float16)
    buf[:arr.size] = arr
    return buf


def _build_cases():
    """Define every LK case after triton is importable.

    Each case dict has: name, kernel (@triton.jit fn), n_ptr_args, constants
    {arg_index: value}, args (ndarrays in signature order; None = output slot),
    out_index, out_elems, golden (flat fp16), compare ("exact" or "tol").
    """
    import numpy as np
    import triton
    import triton.language as tl

    rng = np.random.default_rng(42)

    def rand(*shape, lo=-1.0, hi=1.0):
        return rng.uniform(lo, hi, shape).astype(np.float16)

    # --- elementwise (byte-exact) ---------------------------------------
    @triton.jit
    def rpu_add_kernel(out, a, b, n: tl.constexpr):
        o = tl.arange(0, n)
        tl.store(out + o, tl.load(a + o) + tl.load(b + o))

    @triton.jit
    def rpu_mul_kernel(out, a, b, n: tl.constexpr):
        o = tl.arange(0, n)
        tl.store(out + o, tl.load(a + o) * tl.load(b + o))

    @triton.jit
    def rpu_mul_add_kernel(out, a, b, c, n: tl.constexpr):
        o = tl.arange(0, n)
        tl.store(out + o, tl.load(a + o) * tl.load(b + o) + tl.load(c + o))

    @triton.jit
    def rpu_add_mul_kernel(out, a, b, c, n: tl.constexpr):
        o = tl.arange(0, n)
        tl.store(out + o, (tl.load(a + o) + tl.load(b + o)) * tl.load(c + o))

    @triton.jit
    def rpu_relu_kernel(out, x, n: tl.constexpr):
        o = tl.arange(0, n)
        tl.store(out + o, tl.maximum(tl.load(x + o), 0.0).to(tl.float16))

    @triton.jit
    def rpu_maximum_kernel(out, a, b, n: tl.constexpr):
        o = tl.arange(0, n)
        tl.store(out + o, tl.maximum(tl.load(a + o), tl.load(b + o)).to(tl.float16))

    @triton.jit
    def rpu_bcast_kernel(out, a, b, N: tl.constexpr, C: tl.constexpr):
        on = tl.arange(0, N)[:, None]
        oc = tl.arange(0, C)[None, :]
        av = tl.load(a + on * C + oc)
        bv = tl.load(b + tl.arange(0, N))
        tl.store(out + on * C + oc, av + bv[:, None])

    # --- fp32-accumulating (tolerance) ----------------------------------
    @triton.jit
    def rpu_gemm_kernel(out, lhs, rhs, M: tl.constexpr, K: tl.constexpr, N: tl.constexpr):
        a = tl.load(lhs + tl.arange(0, M)[:, None] * K + tl.arange(0, K)[None, :])
        b = tl.load(rhs + tl.arange(0, K)[:, None] * N + tl.arange(0, N)[None, :])
        tl.store(out + tl.arange(0, M)[:, None] * N + tl.arange(0, N)[None, :], tl.dot(a, b).to(tl.float16))

    @triton.jit
    def rpu_softmax_kernel(out, x, n: tl.constexpr):
        o = tl.arange(0, n)
        v = tl.load(x + o)
        e = tl.exp(v - tl.max(v, axis=0))
        tl.store(out + o, e / tl.sum(e, axis=0))

    @triton.jit
    def rpu_reduce_sum_axis0_kernel(out, x, M: tl.constexpr, N: tl.constexpr):
        om = tl.arange(0, M)[:, None]
        on = tl.arange(0, N)[None, :]
        tl.store(out + tl.arange(0, N), tl.sum(tl.load(x + om * N + on), axis=0))

    @triton.jit
    def rpu_reduce_sum_axis1_kernel(out, x, M: tl.constexpr, N: tl.constexpr):
        om = tl.arange(0, M)[:, None]
        on = tl.arange(0, N)[None, :]
        tl.store(out + tl.arange(0, M), tl.sum(tl.load(x + om * N + on), axis=1))

    cases = []

    def add_case(name, kernel, n_ptr_args, constants, args, out_index, out_elems, golden, compare):
        cases.append(
            dict(name=name, kernel=kernel, n_ptr_args=n_ptr_args, constants=constants, args=args, out_index=out_index,
                 out_elems=out_elems, golden=np.asarray(golden, np.float16).flatten(), compare=compare))

    # elementwise n=16 (output reads only n, so plain inputs are fine)
    n = 16
    a, b, c = rand(n), rand(n), rand(n)
    add_case("add_n16", rpu_add_kernel, 3, {3: n}, [None, a, b], 0, n, a + b, "exact")
    add_case("mul_n16", rpu_mul_kernel, 3, {3: n}, [None, a, b], 0, n, a * b, "exact")
    add_case("mul_add_n16", rpu_mul_add_kernel, 4, {4: n}, [None, a, b, c], 0, n,
             (a.astype(np.float32) * b + c).astype(np.float16), "tol")
    add_case("add_mul_n16", rpu_add_mul_kernel, 4, {4: n}, [None, a, b, c], 0, n,
             ((a.astype(np.float32) + b) * c).astype(np.float16), "tol")
    add_case("relu_n16", rpu_relu_kernel, 2, {2: n}, [None, a], 0, n, np.maximum(a, np.float16(0.0)), "exact")
    add_case("maximum_n16", rpu_maximum_kernel, 3, {3: n}, [None, a, b], 0, n, np.maximum(a, b), "exact")

    # gemm 16x16x16: RHS must be column-major; small range keeps fp16 accurate.
    M = K = N = 16
    ga = rand(M, K, lo=-0.25, hi=0.25)
    gb = rand(K, N, lo=-0.25, hi=0.25)
    gemm_gold = (ga.astype(np.float32) @ gb.astype(np.float32)).astype(np.float16)
    add_case("gemm_16x16x16", rpu_gemm_kernel, 3, {3: M, 4: K, 5: N},
             [None, ga.flatten(), gb.T.copy().flatten()], 0, M * N, gemm_gold, "tol")

    # softmax n=16: pad input V256 with -inf so exp(pad-max)=0.
    sx = rand(n, lo=-2.0, hi=2.0)
    sxf = sx.astype(np.float32)
    se = np.exp(sxf - sxf.max())
    add_case("softmax_n16", rpu_softmax_kernel, 2, {2: n}, [None, _v256_pad(sx, "-inf")], 0, n,
             (se / se.sum()).astype(np.float16), "tol")

    # reduce_sum over each axis on a 16x16: the emit writes the full [M,N]
    # matrix, so golden is tiled (axis0: every row = col sums) / repeated
    # (axis1: every col = row sums) and the whole matrix is compared.
    rx = rand(M, N, lo=-0.5, hi=0.5)
    col_sums = rx.astype(np.float32).sum(axis=0).astype(np.float16)
    row_sums = rx.astype(np.float32).sum(axis=1).astype(np.float16)
    add_case("reduce_sum_axis0_16x16", rpu_reduce_sum_axis0_kernel, 2, {2: M, 3: N}, [None, rx.flatten()], 0, M * N,
             np.tile(col_sums, M), "tol")
    add_case("reduce_sum_axis1_16x16", rpu_reduce_sum_axis1_kernel, 2, {2: M, 3: N}, [None, rx.flatten()], 0, M * N,
             np.repeat(row_sums, N), "tol")

    # broadcast_add [N,C] + [N]: stage b in the V256 broadcast layout the
    # rank-2 '+' reads (lane block 16*tid..16*tid+16 == b[tid]).
    ba = rand(N, K)  # [N, C] with C=16
    bb = rand(N)
    bcast_gold = (ba + bb[:, None]).astype(np.float16)
    m_tiles = (N + 15) // 16
    b_buf = np.zeros(max(m_tiles * 256, 256), dtype=np.float16)
    for v in range(m_tiles):
        for tid in range(16):
            row = v * 16 + tid
            if row < N:
                b_buf[v * 256 + tid * 16:v * 256 + tid * 16 + 16] = bb[row]
    add_case("broadcast_add_16x16", rpu_bcast_kernel, 3, {3: N, 4: K}, [None, ba.flatten(), b_buf], 0, N * K,
             bcast_gold, "exact")

    return cases


def _compile_case(case):
    """Compile one case's kernel with ELF emission; return elf bytes."""
    import triton
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource

    fn = case["kernel"]
    arg_names = fn.arg_names
    signature = {arg_names[i]: "*fp16" for i in range(case["n_ptr_args"])}
    constexprs = {(idx, ): val for idx, val in case["constants"].items()}
    src = ASTSource(fn=fn, constexprs=constexprs, signature=signature)
    kernel = triton.compile(src=src, target=GPUTarget("rpu", "rpu-v1", 1), options={"num_warps": 1})
    elf_hex = kernel.asm.get("rpuelf")
    if not elf_hex:
        raise RuntimeError(f"{case['name']}: kernel.asm has no 'rpuelf' "
                           "(RPU_OUTPUT_ELF=1 did not take effect)")
    # The backend surfaces the device ELF as hex text in kernel.asm["rpuelf"];
    # decode it back to the raw ELF bytes the launch runner expects.
    elf = bytes.fromhex(elf_hex) if isinstance(elf_hex, str) else bytes(elf_hex)
    if not elf.startswith(b"\x7fELF"):
        raise RuntimeError(f"{case['name']}: rpuelf is not an ELF (head={elf[:8]!r})")
    return elf


def _stage_lk_case(case_dir, elf, args, out_index, out_elems):
    """Write kernel.ref and one raw-LE V256-padded buffer per arg.

    Each input array is staged verbatim (already laid out for the device) then
    zero-padded up to a whole V256; the output slot is zero-filled.
    """
    import numpy as np

    case = Path(case_dir)
    case.mkdir(parents=True, exist_ok=True)
    (case / "kernel.ref").write_bytes(elf)
    specs = []  # (role, size, path)
    for i, arr in enumerate(args):
        path = case / f"lk_arg{i}.bin"
        if i == out_index:
            size = _v256_size(out_elems * 2)
            path.write_bytes(b"\x00" * size)
            specs.append(("out", size, path))
        else:
            buf = np.asarray(arr, dtype=np.float16).flatten().tobytes()
            size = _v256_size(len(buf))
            buf = (buf + b"\x00" * size)[:size]
            path.write_bytes(buf)
            specs.append(("in", size, path))
    return specs


def _kill_dispatch(proc, case_dir, use_sudo):
    """Tear down a hung launch_kernel dispatch so it cannot wedge the device.

    A plain ``subprocess`` timeout only SIGKILLs the direct child (``sudo``);
    sudo runs the real dispatcher (``launch_kernel_runner``) in its own pty
    session, so it survives as an orphan that keeps ``/dev/rpu`` open and
    leaves ``run_work`` stuck in the RPU driver queue -- every later case then
    times out too. Kill the whole spawned process group, then reap any runner
    still matching this case's unique work dir (its argv carries the
    ``kernel.ref`` path, so the match is surgical and never touches a
    concurrent job on a shared board).
    """
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass
    if use_sudo:
        subprocess.run(
            ["sudo", "-n", "pkill", "-9", "-f", str(case_dir)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        pass


def _run_cli(runner, case_dir, elf_name, kernel_name, specs):
    use_sudo = os.environ.get("RPU_LK_SUDO", "1") not in ("0", "OFF", "FALSE", "NO")
    cmd = (["sudo", "-n"] if use_sudo else []) + [runner, str(Path(case_dir) / elf_name), kernel_name]
    cmd += [f"{role}:{size}:{path}" for role, size, path in specs]
    # start_new_session so the dispatch is its own process group and a hang can
    # be torn down as a group (see _kill_dispatch) instead of leaking an orphan.
    proc = subprocess.Popen(
        cmd,
        cwd=case_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        out, err = proc.communicate(timeout=120)
    except subprocess.TimeoutExpired:
        _kill_dispatch(proc, case_dir, use_sudo)
        raise
    return subprocess.CompletedProcess(cmd, proc.returncode, out, err)


def _compare(out_path, golden, n_elements, compare):
    import numpy as np

    out = np.frombuffer(Path(out_path).read_bytes(), dtype="<f2")[:n_elements].astype(np.float16)
    gold = np.asarray(golden, dtype=np.float16).flatten()[:n_elements]
    if out.shape != gold.shape:
        return False, f"shape mismatch out={out.shape} gold={gold.shape}"
    diff = np.abs(out.astype(np.float32) - gold.astype(np.float32))
    if compare == "exact":
        if np.array_equal(out, gold):
            return True, "byte_exact_ok"
    else:
        if np.allclose(out.astype(np.float32), gold.astype(np.float32), rtol=FP16_RTOL, atol=FP16_ATOL):
            return True, f"allclose_ok max_diff={diff.max():.3e}"
    return False, f"mismatch max_diff={diff.max():.3e} out[:4]={out[:4]} gold[:4]={gold[:4]}"


def _run_one(case, runner, work_dir, verbose):
    elf = _compile_case(case)
    case_dir = Path(work_dir) / f"case_{case['name']}"
    specs = _stage_lk_case(case_dir, elf, case["args"], case["out_index"], case["out_elems"])
    r = _run_cli(runner, case_dir, "kernel.ref", case["kernel"].__name__, specs)
    if r.returncode != 0:
        if verbose:
            print(r.stdout)
            print(r.stderr)
        return f"FAIL:lk_dispatch_exit={r.returncode} stderr={r.stderr.strip()[:200]}"
    ok, info = _compare(specs[case["out_index"]][2], case["golden"], case["out_elems"], case["compare"])
    return "PASS" if ok else f"FAIL:{info}"


def run(only=None, verbose=False):
    """Run the LK smoke suite. Returns (overall, results) where overall is
    'PASS'/'FAIL'/'SKIP:<reason>' and results is a list of (name, status)."""
    try:
        import numpy  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        raise Skip(f"numpy unavailable: {exc}")
    try:
        import triton  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        raise Skip(f"triton unavailable: {exc}")

    _resolve_clang()
    runner = _resolve_runner()
    if not Path("/dev/rpu").exists():
        raise Skip("/dev/rpu not present")

    work_dir = Path(tempfile.mkdtemp(prefix="rpu_lk_smoke_"))
    try:
        _set_compile_env(work_dir)
        cases = _build_cases()
        if only:
            cases = [c for c in cases if c["name"] in only]
            if not cases:
                raise Skip(f"no cases match {only}")
        results = []
        for case in cases:
            try:
                status = _run_one(case, runner, work_dir, verbose)
            except Exception as exc:  # noqa: BLE001
                status = f"FAIL:exception={type(exc).__name__}:{exc}"
            print(f"  {case['name']:<26} {status}")
            results.append((case["name"], status))
        overall = "PASS" if all(s == "PASS" for _, s in results) else "FAIL"
        return overall, results
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser(description="RPU launch_kernel board smoke test")
    parser.add_argument("-v", "--verbose", action="store_true", help="print dispatcher stdout/stderr on failure")
    parser.add_argument("--only", nargs="+", metavar="CASE", help="run only the named case(s)")
    parser.add_argument(
        "--require-board",
        action="store_true",
        default=os.environ.get("RPU_LK_REQUIRE") in ("1", "ON", "TRUE", "YES"),
        help="treat a skipped prerequisite as a failure (use on the board runner)",
    )
    args = parser.parse_args()

    print("=== RPU launch_kernel board smoke test ===")
    try:
        overall, results = run(only=args.only, verbose=args.verbose)
    except Skip as skip:
        if args.require_board:
            print(f"FAIL: prerequisite missing but --require-board set: {skip}")
            return 1
        print(f"SKIP: {skip}")
        return 0

    n_pass = sum(1 for _, s in results if s == "PASS")
    print(f"RESULT: {overall} ({n_pass}/{len(results)} cases passed)")
    return 0 if overall == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
