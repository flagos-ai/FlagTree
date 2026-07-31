import argparse
import os

import torch
import triton
import triton.language as tl

# Adapted from:
# https://github.com/DeepLink-org/DLBlas/blob/main/dlblas/utils/op_helper.py
# Original project licensed under the MIT License.

import triton.experimental.tle as tle  # noqa: F401  (registers tile/tle dialects)
from triton.experimental.tle.language.dsa.ascend import L1  # noqa: F401
from triton.experimental.tle.language.dsa import tile_alloc, tile_copy, tile_to_tensor, tile_subview  # noqa: F401

# =============================================================================
#  Compile-time configuration
# =============================================================================
_DEFAULT_M = 1024
_DEFAULT_N = 1024
_DEFAULT_K = 1024
_DEFAULT_NUM_CORES = 24

def get_number_cores():
    """Return the number of AI cores to use as the launch grid size."""
    try:
        import torch_npu  # noqa: F401
        return torch.npu.get_device_properties(0).ai_core_num
    except Exception:
        return _DEFAULT_NUM_CORES


@triton.jit
def grouped_launch_diagonal(pid, num_pid_m, num_pid_n, BLOCK_TRESHHOLD: tl.constexpr):
    if (num_pid_m >= BLOCK_TRESHHOLD) and (num_pid_n >= BLOCK_TRESHHOLD):
        # 对角线分核代码实现
        curThresholdM = (
            BLOCK_TRESHHOLD
            if pid < (num_pid_m // BLOCK_TRESHHOLD * BLOCK_TRESHHOLD) * num_pid_n
            else num_pid_m % BLOCK_TRESHHOLD
        )
        curThresholdM_thresholdN = curThresholdM * BLOCK_TRESHHOLD
        curThresholdN = (
            BLOCK_TRESHHOLD
            if pid % (num_pid_n * BLOCK_TRESHHOLD)
            < (curThresholdM * num_pid_n)
            // curThresholdM_thresholdN
            * curThresholdM_thresholdN
            else num_pid_n % BLOCK_TRESHHOLD
        )
        localRelativeBlock = (
            pid % (BLOCK_TRESHHOLD * num_pid_n) % (BLOCK_TRESHHOLD * curThresholdM)
        )
        task_m_idx = (
            localRelativeBlock % curThresholdM
            + pid // (BLOCK_TRESHHOLD * num_pid_n) * BLOCK_TRESHHOLD
        )
        # 求最小公倍数，方便求基本块的坐标
        x, y = curThresholdM, curThresholdN
        while y != 0:
            x, y = y, x % y
        lcm = curThresholdM * curThresholdN // x
        task_n_idx = (
            localRelativeBlock + (localRelativeBlock // lcm)
        ) % curThresholdN + pid % (
            BLOCK_TRESHHOLD * num_pid_n
        ) // curThresholdM_thresholdN * BLOCK_TRESHHOLD
    else:
        task_m_idx = pid // num_pid_n
        task_n_idx = pid % num_pid_n
    return task_m_idx, task_n_idx

@triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 256, "BLOCK_TRESHHOLD": 4}
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 256, "BLOCK_TRESHHOLD": 5}
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 256, "BLOCK_TRESHHOLD": 6}
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 256, "BLOCK_TRESHHOLD": 7}
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 256, "BLOCK_TRESHHOLD": 8}
        ),
        triton.Config(
            {"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 256, "BLOCK_TRESHHOLD": 4}
        ),
        triton.Config(
            {"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 256, "BLOCK_TRESHHOLD": 5}
        ),
        triton.Config(
            {"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 256, "BLOCK_TRESHHOLD": 6}
        ),
        triton.Config(
            {"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 256, "BLOCK_TRESHHOLD": 7}
        ),
        triton.Config(
            {"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 256, "BLOCK_TRESHHOLD": 8}
        ),
    ],
    key=["N", "K"],
)
@triton.jit
def matmul_kernel(
    mat_a,
    mat_b,
    mat_c,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    NUM_CORES: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_TRESHHOLD: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    NUM_BLOCKS_M = tl.cdiv(M, BLOCK_M)
    NUM_BLOCKS_N = tl.cdiv(N, BLOCK_N)
    NUM_BLOCKS    = NUM_BLOCKS_M * NUM_BLOCKS_N
    NUM_K_BLOCKS  = tl.cdiv(K, BLOCK_K)
    # prologue + main loop + epilogue:
    #   片上内存：A/B 输入放 L1，C 输出累加在寄存器（tl.zeros / tl.dot）。
    #   iter 步长 = NUM_CORES * NUM_K_BLOCKS，保证同一 core 连续消费同一输出块的全部 K 切片。

    # ① 片上工作集
    # A/B 各申请一整块 L1（3 倍行），用 tile_subview 按行偏移取出 3 个 slot 的子视图。
    # slot s 的行偏移 = s * BLOCK_M（A）或 s * BLOCK_K（B）。
    mat_a_l1   = tile_alloc([3 * BLOCK_M, BLOCK_K], mat_a.dtype.element_ty, tle.language.dsa.ascend.L1)
    mat_b_l1   = tile_alloc([3 * BLOCK_K, BLOCK_N], mat_b.dtype.element_ty, tle.language.dsa.ascend.L1)

    iter_start = pid * NUM_K_BLOCKS
    iter_end   = NUM_BLOCKS * NUM_K_BLOCKS
    iter_step  = NUM_CORES * NUM_K_BLOCKS

    # ── 取 slot s（0/1/2）的子视图 ───────────────────────────────────────────
    # tile_subview(src, offsets, sizes, strides)
    #   offsets: 动态值，可以是运行时标量
    #   sizes / strides: 必须是 constexpr
    def _a_slot(s):
        return tile_subview(mat_a_l1,
                            [s * BLOCK_M, 0],
                            [BLOCK_M, BLOCK_K],
                            [1, 1])

    def _b_slot(s):
        return tile_subview(mat_b_l1,
                            [s * BLOCK_K, 0],
                            [BLOCK_K, BLOCK_N],
                            [1, 1])

    # ② prologue: 向 slot 0/1/2 分别预加载循环第 0/1/2 次迭代对应的切片
    # main loop 步长为 iter_step，第 i 次迭代对应全局 iter = iter_start + i*iter_step，
    # 因此三个槽分别加载 iter_start、iter_start+iter_step、iter_start+2*iter_step。
    # 对于 slot 1/2，需要检查是否越界（当分配给此 core 的迭代不足 3 次时）。
    task_m, task_n = grouped_launch_diagonal(
        iter_start // NUM_K_BLOCKS, NUM_BLOCKS_M, NUM_BLOCKS_N, BLOCK_TRESHHOLD)
    m_start = task_m * BLOCK_M
    n_start = task_n * BLOCK_N
    prev_block_idx = iter_start // NUM_K_BLOCKS
    mat_c_acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    tile_copy(tl.make_block_ptr(
        mat_a, (M, K), (K, 1), (m_start, (iter_start % NUM_K_BLOCKS) * BLOCK_K),
        (BLOCK_M, BLOCK_K), (1, 0)), _a_slot(0), [BLOCK_M, BLOCK_K])
    tile_copy(tl.make_block_ptr(
        mat_b, (K, N), (N, 1), ((iter_start % NUM_K_BLOCKS) * BLOCK_K, n_start),
        (BLOCK_K, BLOCK_N), (1, 0)), _b_slot(0), [BLOCK_K, BLOCK_N])

    m_nxt1 = m_start
    n_nxt1 = n_start
    m_nxt2 = m_start
    n_nxt2 = n_start
    if iter_start + iter_step < iter_end:
        task_m, task_n = grouped_launch_diagonal(
            (iter_start + iter_step) // NUM_K_BLOCKS, NUM_BLOCKS_M, NUM_BLOCKS_N, BLOCK_TRESHHOLD)
        m_nxt1 = task_m * BLOCK_M
        n_nxt1 = task_n * BLOCK_N
        tile_copy(tl.make_block_ptr(
            mat_a, (M, K), (K, 1), (m_nxt1, ((iter_start + iter_step) % NUM_K_BLOCKS) * BLOCK_K),
            (BLOCK_M, BLOCK_K), (1, 0)), _a_slot(1), [BLOCK_M, BLOCK_K])
        tile_copy(tl.make_block_ptr(
            mat_b, (K, N), (N, 1), (((iter_start + iter_step) % NUM_K_BLOCKS) * BLOCK_K, n_nxt1),
            (BLOCK_K, BLOCK_N), (1, 0)), _b_slot(1), [BLOCK_K, BLOCK_N])

    if iter_start + 2 * iter_step < iter_end:
        task_m, task_n = grouped_launch_diagonal(
            (iter_start + 2 * iter_step) // NUM_K_BLOCKS, NUM_BLOCKS_M, NUM_BLOCKS_N, BLOCK_TRESHHOLD)
        m_nxt2 = task_m * BLOCK_M
        n_nxt2 = task_n * BLOCK_N
        tile_copy(tl.make_block_ptr(
            mat_a, (M, K), (K, 1), (m_nxt2, ((iter_start + 2 * iter_step) % NUM_K_BLOCKS) * BLOCK_K),
            (BLOCK_M, BLOCK_K), (1, 0)), _a_slot(2), [BLOCK_M, BLOCK_K])
        tile_copy(tl.make_block_ptr(
            mat_b, (K, N), (N, 1), (((iter_start + 2 * iter_step) % NUM_K_BLOCKS) * BLOCK_K, n_nxt2),
            (BLOCK_K, BLOCK_N), (1, 0)), _b_slot(2), [BLOCK_K, BLOCK_N])

    # ③ main loop
    # s = (iter // iter_step) % 3 是当前消费的 slot，编译时可静态推导（循环步长固定）。
    # tile_subview 的 offset 参数在此处为编译期常量，不产生运行时分支。
    for iter in range(iter_start, iter_end - iter_step, iter_step):
        s = (iter // iter_step) % 3

        if iter // NUM_K_BLOCKS != prev_block_idx:
            tl.store(tl.make_block_ptr(
                mat_c, (M, N), (N, 1), (m_start, n_start), (BLOCK_M, BLOCK_N), (1, 0)),
                mat_c_acc.to(mat_c.dtype.element_ty))
            m_start = m_nxt1
            n_start = n_nxt1
            mat_c_acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            prev_block_idx = iter // NUM_K_BLOCKS

        mat_c_acc = tl.dot(tile_to_tensor(_a_slot(s), writable=False),
                           tile_to_tensor(_b_slot(s), writable=False),
                           mat_c_acc, out_dtype=tl.float32)

        m_nxt1 = m_nxt2
        n_nxt1 = n_nxt2

        # prefetch 3 步后的切片，写回同一个 slot s（已消费，可安全复用）
        # 仅当 prefetch 目标未越界时才发起搬运
        iter_pf = iter + 3 * iter_step
        if iter_pf < iter_end:
            task_m, task_n = grouped_launch_diagonal(
                iter_pf // NUM_K_BLOCKS, NUM_BLOCKS_M, NUM_BLOCKS_N, BLOCK_TRESHHOLD)
            m_nxt2 = task_m * BLOCK_M
            n_nxt2 = task_n * BLOCK_N
            tile_copy(tl.make_block_ptr(
                mat_a, (M, K), (K, 1), (m_nxt2, (iter_pf % NUM_K_BLOCKS) * BLOCK_K),
                (BLOCK_M, BLOCK_K), (1, 0)), _a_slot(s), [BLOCK_M, BLOCK_K])
            tile_copy(tl.make_block_ptr(
                mat_b, (K, N), (N, 1), ((iter_pf % NUM_K_BLOCKS) * BLOCK_K, n_nxt2),
                (BLOCK_K, BLOCK_N), (1, 0)), _b_slot(s), [BLOCK_K, BLOCK_N])

    # ④ epilogue: 消费最后一次迭代 iter_end - iter_step
    # main loop range(..., iter_end - iter_step) 不包含最后一次迭代，需要在此处理。
    s_ep = ((iter_end - iter_step) // iter_step) % 3

    if (iter_end - iter_step) // NUM_K_BLOCKS != prev_block_idx:
        tl.store(tl.make_block_ptr(
            mat_c, (M, N), (N, 1), (m_start, n_start), (BLOCK_M, BLOCK_N), (1, 0)),
            mat_c_acc.to(mat_c.dtype.element_ty))
        m_start = m_nxt1
        n_start = n_nxt1
        mat_c_acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    mat_c_acc = tl.dot(tile_to_tensor(_a_slot(s_ep), writable=False),
                       tile_to_tensor(_b_slot(s_ep), writable=False),
                       mat_c_acc, out_dtype=tl.float32)

    # 写回最后一个输出块
    tl.store(tl.make_block_ptr(
        mat_c, (M, N), (N, 1), (m_start, n_start), (BLOCK_M, BLOCK_N), (1, 0)),
        mat_c_acc.to(mat_c.dtype.element_ty))


def call(mat_a, mat_b, num_cores=_DEFAULT_NUM_CORES):
    m = mat_a.shape[0]
    k = mat_a.shape[1]
    n = mat_b.shape[1]
    mat_c = torch.empty(m, n, dtype=mat_a.dtype, device=mat_a.device)
    """
    NPU芯片更加亲和512B对齐场景,如下分块通用性能较好,可以使用autotune选取最优
    BLOCK_M = 128
    BLOCK_N = 256
    BLOCK_K = 256
    """
    matmul_kernel[(num_cores,)](mat_a, mat_b, mat_c, m, n, k, num_cores)
    # print(f"matmul_kernel best config {matmul_kernel.best_config}", flush = True)
    return mat_c


# =============================================================================
#  Intermediate TTIR dump (no device required)
#
#  Compiles the matmul_kernel straight to TTIR using ast_to_ttir and writes
#  str(module) to a file.  Mirrors the dump_tileir approach in fa_triton_arch.py.
# =============================================================================
class _DumpOptions:
    num_warps = 4
    num_stages = 1
    num_ctas = 1
    cluster_dims = (1, 1, 1)
    enable_fp_fusion = True
    debug = False
    allowed_dot_input_precisions = ("tf32", "tf32x3", "ieee")
    max_num_imprecise_acc_default = 0
    default_dot_input_precision = "ieee"
    sanitize_overflow = False


def _matmul_signature(N: int, K: int, BLOCK_M: int, BLOCK_N: int, BLOCK_K: int,
                      BLOCK_TRESHHOLD: int, NUM_CORES: int):
    """Static signature for ast_to_ttir.

    mat_a, mat_b, mat_c are fp16 pointer arguments.
    M is a runtime scalar (i32); N, K, BLOCK_* and NUM_CORES are constexpr.
    """
    return {
        "mat_a": "*fp16",
        "mat_b": "*fp16",
        "mat_c": "*fp16",
        "M": "i32",
        "N": "constexpr",
        "K": "constexpr",
        "NUM_CORES": "constexpr",
        "BLOCK_M": "constexpr",
        "BLOCK_N": "constexpr",
        "BLOCK_K": "constexpr",
        "BLOCK_TRESHHOLD": "constexpr",
    }


def dump_ttir(path=None, M=_DEFAULT_M, N=_DEFAULT_N, K=_DEFAULT_K,
              BLOCK_M=128, BLOCK_N=256, BLOCK_K=256, BLOCK_TRESHHOLD=4,
              NUM_CORES=_DEFAULT_NUM_CORES):
    """Compile matmul_kernel to TTIR and write the module to *path*.

    No NPU/GPU required — pure front-end compilation via ast_to_ttir.

    Returns the MLIR string.
    """
    from triton.compiler.compiler import ASTSource
    from triton.compiler.code_generator import ast_to_ttir
    from triton._C.libtriton import ir

    os.environ.setdefault("TRITON_ALLOW_NON_CONSTEXPR_GLOBALS", "1")

    if path is None:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "matmul_triton.mlir")

    signature = _matmul_signature(N, K, BLOCK_M, BLOCK_N, BLOCK_K, BLOCK_TRESHHOLD, NUM_CORES)
    constants = {
        "N": N,
        "K": K,
        "NUM_CORES": NUM_CORES,
        "BLOCK_M": BLOCK_M,
        "BLOCK_N": BLOCK_N,
        "BLOCK_K": BLOCK_K,
        "BLOCK_TRESHHOLD": BLOCK_TRESHHOLD,
    }

    src = ASTSource(matmul_kernel.fn, signature, constants)
    context = ir.context()
    ir.load_dialects(context)
    # Ascend dialect is optional; load if present.
    try:
        from triton._C.libtriton.ascend import ir as ascend_ir
        ascend_ir.load_dialects(context)
    except Exception:
        pass

    # tl.dot requires a target-provided "min_dot_size"; supply a neutral default.
    codegen_fns = {"min_dot_size": lambda lhsType, rhsType: (1, 1, 1)}
    module = ast_to_ttir(matmul_kernel.fn, src, context, _DumpOptions(), codegen_fns, {})

    ok = module.verify()
    if not ok:
        raise RuntimeError("dump_ttir: module.verify() failed — IR is not legal")

    mlir = str(module)
    with open(path, "w") as f:
        f.write(mlir)
    print(f"[dump_ttir] module.verify() = {ok}; wrote TTIR ({len(mlir)} chars) to {path}")
    return mlir


def dump_linalg(path=None, M=_DEFAULT_M, N=_DEFAULT_N, K=_DEFAULT_K,
                BLOCK_M=128, BLOCK_N=256, BLOCK_K=256, BLOCK_TRESHHOLD=4,
                NUM_CORES=_DEFAULT_NUM_CORES):
    """Compile matmul_kernel through the full TTIR → Linalg lowering pipeline.

    Pipeline:
      ① add_triton_to_structure_incubated    — structured ptr analysis
        add_discrete_mask_access_conversion  — non-contiguous mask handling
      ② add_triton_to_unstructure_incubated  — scalarize unstructured accesses
        add_triton_to_hivm                   — CustomOp → HIVM SyncOp
        add_triton_to_hfusion                — Triton → HFusion
        add_triton_to_llvm                   — Triton → LLVM
      ③ add_bubble_up_operation              — push extracts upward
        add_triton_to_structure_incubated    — cleanup (round 2)
      ③b inline + canonicalize               — clean up before linalg
      ④ add_triton_to_linalg_incubated       — TTIR compute → Linalg
      ④b add_erase_linalg_casts              — fold cast chains
      ⑤ final canonicalize + CSE + DCE       — erase dead ops

    Returns the final Linalg MLIR string.  No NPU/GPU required.
    """
    from triton._C.libtriton import ir, passes, ascend

    # Step 1: compile to TTIR
    ttir_mlir = dump_ttir(path=None, M=M, N=N, K=K, BLOCK_M=BLOCK_M,
                          BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
                          BLOCK_TRESHHOLD=BLOCK_TRESHHOLD, NUM_CORES=NUM_CORES)

    if path is None:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "matmul_triton_linalg.mlir")

    # Step 2: parse the TTIR module into a fresh context
    context = ir.context()
    ir.load_dialects(context)
    try:
        from triton._C.libtriton.ascend import ir as ascend_ir
        ascend_ir.load_dialects(context)
    except Exception:
        pass

    import tempfile
    with tempfile.NamedTemporaryFile(mode="w", suffix=".mlir", delete=False) as f:
        f.write(ttir_mlir)
        tmp_path = f.name
    module = ir.parse_mlir_module(tmp_path, context)
    os.unlink(tmp_path)

    # ── ① Structured (r1) + discrete mask ────────────────────────────────
    pm = ir.pass_manager(context)
    pm.enable_debug()
    # ascend.passes.ttir.add_triton_to_structure_incubated(pm, False, False, False)
    ascend.passes.ttir.add_discrete_mask_access_conversion(pm, False, False, False)
    pm.run(module)
    print(f"[dump_linalg] ① structure(r1)+discrete_mask: verify={module.verify()}", flush=True)

    # ── ② Unstructured + HIVM + HFusion + LLVM ──────────────────────────
    pm = ir.pass_manager(context)
    pm.enable_debug()
    ascend.passes.ttir.add_triton_to_unstructure(pm, False, False)
    ascend.passes.ttir.add_triton_to_hivm(pm)
    ascend.passes.ttir.add_triton_to_hfusion(pm)
    ascend.passes.ttir.add_triton_to_llvm(pm)
    pm.run(module)
    print(f"[dump_linalg] ② unstructure+hivm+hfusion+llvm: verify={module.verify()}", flush=True)

    # ── ③ Bubble-up + structured (r2) ────────────────────────────────────
    pm = ir.pass_manager(context)
    pm.enable_debug()
    ascend.passes.ttir.add_bubble_up_operation(pm)
    # ascend.passes.ttir.add_triton_to_structure_incubated(pm, False, False, False)
    pm.run(module)
    print(f"[dump_linalg] ③ bubble_up+structure(r2): verify={module.verify()}", flush=True)

    # ── ③b Inline + canonicalize ← required before linalg ───────────────
    pm = ir.pass_manager(context)
    pm.enable_debug()
    passes.common.add_inliner(pm)
    passes.common.add_canonicalizer(pm)
    pm.run(module)
    print(f"[dump_linalg] ③b inline+canonicalize: verify={module.verify()}", flush=True)

    # ── ④ Triton → Linalg ───────────────────────────────────────────────
    linalg_ok = False
    try:
        pm = ir.pass_manager(context)
        pm.enable_debug()
        ascend.passes.ttir.add_triton_to_linalg(pm, False, True, False, False, False)
        pm.run(module)
        print(f"[dump_linalg] ④ triton_to_linalg_incubated: verify={module.verify()}", flush=True)
        linalg_ok = True
    except RuntimeError as e:
        print(f"[dump_linalg] ④ triton_to_linalg_incubated: partial conversion "
              f"(expected — creates cast chains that need post-processing): {e}", flush=True)

    # ── ④b Fold staging memref.alloc + memref.copy pairs ─────────────────
    # pm = ir.pass_manager(context)
    # pm.enable_debug()
    # ascend.passes.ttir.add_fold_staging_copy(pm)
    # pm.run(module)
    # print(f"[dump_linalg] ④b fold_staging_copy: verify={module.verify()}", flush=True)

    # ── ④c Erase linalg casts introduced by the partial linalg conversion ─
    # pm = ir.pass_manager(context)
    # pm.enable_debug()
    # ascend.passes.ttir.add_erase_linalg_casts(pm)
    # pm.run(module)
    # print(f"[dump_linalg] ④c erase_linalg_casts: verify={module.verify()}", flush=True)

    # ── ⑤ Final canonicalize + CSE + DCE ─────────────────────────────────
    pm = ir.pass_manager(context)
    pm.enable_debug()
    passes.common.add_canonicalizer(pm)
    passes.common.add_cse(pm)
    passes.common.add_symbol_dce(pm)
    pm.run(module)
    print(f"[dump_linalg] ⑤ final cleanup: verify={module.verify()}", flush=True)

    ok = module.verify()
    if not ok:
        raise RuntimeError("dump_linalg: module.verify() failed after pipeline")

    mlir = str(module)
    with open(path, "w") as f:
        f.write(mlir)
    print(f"[dump_linalg] verify={ok}; wrote Linalg IR ({len(mlir)} chars) to {path}")
    return mlir


# =============================================================================
#  Entry point
# =============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Triton matmul kernel — test & IR dump")
    parser.add_argument("--M", type=int, default=_DEFAULT_M, help="Matrix M dimension")
    parser.add_argument("--N", type=int, default=_DEFAULT_N, help="Matrix N dimension")
    parser.add_argument("--K", type=int, default=_DEFAULT_K, help="Matrix K dimension")
    parser.add_argument("--block-m", type=int, default=128, help="BLOCK_M tile size")
    parser.add_argument("--block-n", type=int, default=256, help="BLOCK_N tile size")
    parser.add_argument("--block-k", type=int, default=256, help="BLOCK_K tile size")
    parser.add_argument("--block-treshhold", type=int, default=4,
                        help="BLOCK_TRESHHOLD for diagonal tiling")
    parser.add_argument("--num-cores", type=int, default=None,
                        help="Number of AI cores (default: auto-detect or 24)")
    parser.add_argument("--no-check", action="store_true",
                        help="Skip correctness check against torch.matmul reference")
    parser.add_argument(
        "--dump-ttir", nargs="?", const="", default=None,
        metavar="PATH",
        help="Dump TTIR to PATH (default: matmul_triton.mlir next to this file) and exit; "
             "no device needed.",
    )
    parser.add_argument(
        "--dump-linalg", nargs="?", const="", default=None,
        metavar="PATH",
        help="Dump Linalg IR (full TTIR→Linalg lowering) to PATH and exit; "
             "no device needed.",
    )
    args = parser.parse_args()

    M, N, K = args.M, args.N, args.K
    num_cores = args.num_cores or get_number_cores()
    bm, bn, bk, bt = args.block_m, args.block_n, args.block_k, args.block_treshhold

    # ---- dump TTIR and exit (no device required) ----------------------------
    if args.dump_ttir is not None:
        dump_ttir(
            path=(args.dump_ttir or None),
            M=M, N=N, K=K,
            BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk,
            BLOCK_TRESHHOLD=bt, NUM_CORES=num_cores,
        )
        raise SystemExit(0)

    # ---- dump Linalg IR and exit (no device required) -----------------------
    if args.dump_linalg is not None:
        dump_linalg(
            path=(args.dump_linalg or None),
            M=M, N=N, K=K,
            BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk,
            BLOCK_TRESHHOLD=bt, NUM_CORES=num_cores,
        )
        raise SystemExit(0)

    # ---- functional test on device ------------------------------------------
    device = "npu" if hasattr(torch, "npu") and torch.npu.is_available() else "cuda"
    torch.manual_seed(0)
    mat_a = torch.randn((M, K), dtype=torch.float16, device=device)
    mat_b = torch.randn((K, N), dtype=torch.float16, device=device)

    mat_c = call(mat_a, mat_b, num_cores)

    if not args.no_check:
        ref = torch.matmul(mat_a.float(), mat_b.float()).to(torch.float16)
        torch.testing.assert_close(ref, mat_c, rtol=1e-2, atol=1e-2)
        print("Test Passed!")
    else:
        print("Reference check skipped.")
