// =============================================================================
// Flash Attention Forward 3-Task Pipeline — TileOp 形态
//
// 对应 Python：skill/op/fa_triton_3_task.py::flash_attention_fwd_3task_kernel
// Common IR 规约：skill/design/unified_ir.md §2.1
// 下降准则：skill/mlir/skill.md §1（Pass Pipeline / HIVM Op 定义）
//
// 本示例为 Ascend NPU 3-task 软件流水（MIX 模式）：
//   每拍 (tick) 内同时执行：
//     Cube  : Bmm1(k)        ||  Bmm2(k-1)
//     Vector: Vec1(k-1)      ||  Vec2(k-2)
//   三拍完成一个子任务的完整流程 Bmm1(g) → [Vec1(g), Bmm2(g)] → Vec2(g)
//
// 关键抽象：
//   - GM ping-pong 工作区：mm1Res[2, BM, BN] / stage1Res[2, BM, BN] / mm2Res[2, BM, Dv]
//   - Task 元数据环：taskId % 3 索引，每槽持有软 max rescale + rowsum carry
//   - 跨引擎同步：set_flag/wait_flag 显式化 Cube↔Vector 数据依赖边界
//   - Cube 异步发起：cube_launch → !cube.token → cube_wait，实现引擎重叠
//
// 内存层级（对应 HIVM address_space <gm/ub/l1/l0a/l0b/l0c>）：
//   #tile.global — GM HBM （输入张量 + mm1Res/stage1Res/mm2Res 工作区）
//   #tile.l1     — L1 数据暂存（GM→L0 中转）
//   #tile.l0a    — Cube A-side 矩阵 L0 缓冲
//   #tile.l0b    — Cube B-side 矩阵 L0 缓冲
//   #tile.l0c    — Cube 累加器 L0 缓冲
//   #tile.ub     — Unified Buffer（Vector 计算 + MTE3 写 GM）
// =============================================================================
//
// ----- 类型缩写（仅用于阅读；MLIR 中写完整 !tile.type<...>）--------------------------
//   !l1_t     = !tile.type<[BM,  D ],  f16, #tile.l1>
//   !l0a_t    = !tile.type<[BM,  D ],  f16, #tile.l0a>
//   !l0b_t    = !tile.type<[D,   BN],  f16, #tile.l0b>
//   !l0c_t    = !tile.type<[BM,  BN],  f32, #tile.l0c>
//   !ub_f32   = !tile.type<[BM,  BN],  f32, #tile.ub>
//   !ub_f16   = !tile.type<[BM,  BN],  f16, #tile.ub>
//   !ub_acc   = !tile.type<[BM,  Dv],  f32, #tile.ub>
//   !ub_sm1   = !tile.type<[BM,  1],   f32, #tile.ub>
//   !tok      = !tile.cube_token
//   !gm_f16   = !tt.ptr<f16, #tile.global>
//   !gm_f32   = !tt.ptr<f32, #tile.global>
// =============================================================================

tile.func @flash_attention_fwd_3task
    // ---- 函数属性（skill.md §1：core_type = MIX → 单指令流驱动 Cube+Vector）---
    {hivm.func_core_type = #hivm.core_type<MIX>}
(
    // ---- 输入张量（GM 指针）-----------------------------------------------
    %Q_ptr    : !tt.ptr<f16, #tile.global>,
    %K_ptr    : !tt.ptr<f16, #tile.global>,
    %V_ptr    : !tt.ptr<f16, #tile.global>,
    %Out_ptr  : !tt.ptr<f16, #tile.global>,
    // ---- GM ping-pong 工作区（host 预分配）---------------------------------
    %mm1Res   : !tt.ptr<f32, #tile.global>,    // [NUM_CORES, 2, BM, BN]
    %stage1Res: !tt.ptr<f16, #tile.global>,    // [NUM_CORES, 2, BM, BN]
    %mm2Res   : !tt.ptr<f32, #tile.global>,    // [NUM_CORES, 2, BM, Dv]
    // ---- 标量参数 ----------------------------------------------------------
    %sm_scale   : f32,
    %B    : index, %Hq : index, %Hkv : index, %S : index,
    %sQb : index, %sQh : index, %sQs : index, %sQd : index,
    %sKb : index, %sKh : index, %sKs : index, %sKd : index,
    %sOb : index, %sOh : index, %sOs : index, %sOd : index,
    %num_seq_blocks : index, %heads_q : index, %gqa_group : index,
    %n_iters : index, %q_tasks : index, %r_tasks : index,
    NUM_ITERS  : i64,
    IS_CAUSAL  : i1)
{
    // =========================================================================
    // ⓪ 常量与程序坐标
    // =========================================================================
    %c0   = arith.constant 0 : index
    %c1   = arith.constant 1 : index
    %c2   = arith.constant 2 : index
    %c3   = arith.constant 3 : index
    %cBM  = arith.constant 128 : index
    %cBN  = arith.constant 128 : index
    %cD   = arith.constant 128 : index
    %cDv  = arith.constant 128 : index
    %c0_f = arith.constant 0.0 : f32
    %neg_inf_f = arith.constant 0xFF800000 : f32  // -inf

    %cid = tt.get_program_id 0 : index

    // 静态任务分配：my_start = cid * q_tasks + min(cid, r_tasks)
    %cid_lt_r   = arith.cmpi slt, %cid, %r_tasks : index
    %cid_clamp  = arith.select %cid_lt_r, %cid, %r_tasks : index
    %base_off   = arith.muli %cid, %q_tasks : index
    %my_start   = arith.addi %base_off, %cid_clamp : index
    // my_count = q_tasks + (cid < r_tasks ? 1 : 0)
    %extra      = arith.select %cid_lt_r, %c1, %c0 : index
    %my_count   = arith.addi %q_tasks, %extra : index

    %n_sub = arith.muli %my_count, %n_iters : index
    %n_loop = arith.addi %n_sub, %c2 : index   // PIPE_DEPTH = 2（3 拍完成一轮）

    // =========================================================================
    // ① 片上 Tile 分配
    //
    // Cube 侧：L1 暂存 → L0A/L0B → L0C 累加器
    // Vector 侧：UB 工作区 + 在线 softmax 状态 + task 环 carry
    //
    // 双缓冲：L0 slot 0 = Bmm1, slot 1 = Bmm2
    // task 环：[RING=3] 槽 = taskId % 3，存 Vec1→Vec2 的 rescale/rowsum carry
    // =========================================================================

    // ----- Cube L1 staging --------------------------------------------------
    %q_l1 = tile.alloc<[128, 128], f16, #tile.l1> {role = "cube_a_staging"}
    %k_l1 = tile.alloc<[128, 128], f16, #tile.l1> {role = "cube_b_staging"}
    %v_l1 = tile.alloc<[128, 128], f16, #tile.l1> {role = "cube_b_staging"}
    %p_l1 = tile.alloc<[128, 128], f16, #tile.l1> {role = "cube_a_staging"}

    // ----- Cube L0 double buffer (slot 0 = Bmm1, slot 1 = Bmm2) -----------
    %l0a = tile.alloc<[2, 128, 128], f16, #tile.l0a> {policy = #tile.double_buffer}
    %l0b = tile.alloc<[2, 128, 128], f16, #tile.l0b> {policy = #tile.double_buffer}
    %l0c = tile.alloc<[2, 128, 128], f32, #tile.l0c> {policy = #tile.double_buffer}

    // ----- Vector UB 工作区 -------------------------------------------------
    // 在线 softmax 累加器（整段 Q tile 循环常驻）
    %acc_o  = tile.alloc<[128, 128], f32, #tile.ub>
        {role = "output_accum", lifetime = "loop_carried"}
    %sumexp = tile.alloc<[128, 1], f32, #tile.ub>
        {role = "softmax_denom", lifetime = "loop_carried"}

    // neg_sm ping-pong [2, BM, 1]（per-sub-task running max, negated & scaled）
    %neg_sm_buf = tile.alloc<[2, 128, 1], f32, #tile.ub> {policy = #tile.double_buffer}

    // Vec1 → Vec2 carry 环 [RING=3, BM, 1]（taskId % 3 槽）
    %r_factors = tile.alloc<[3, 128, 1], f32, #tile.ub> {role = "softmax_carry"}
    %sumexp_is = tile.alloc<[3, 128, 1], f32, #tile.ub> {role = "softmax_carry"}

    // 通用临时 UB 缓冲
    %s_tmp = tile.alloc<[128, 128], f32, #tile.ub> {role = "softmax_workspace"}

    // =========================================================================
    // ② 初始化：累加器归零 + 跨引擎 flag 预置
    //
    //   set_flag(MTE3_MTE2)：预置 P 已消费信号，使首次进入 Bmm2 的 wait_flag 通过
    //   （第 0 拍没有 Vec1 产生 P，因此需 pre-arm）
    // =========================================================================
    %acc_init  = tile.splat %c0_f : f32 -> !tile.type<[128, 128], f32, #tile.ub>
    %se_init   = tile.splat %c0_f : f32 -> !tile.type<[128, 1], f32, #tile.ub>
    %neg_inf_v = tile.splat %neg_inf_f : f32 -> !tile.type<[128, 1], f32, #tile.ub>
    %tok_none  = tile.cube_token.create  // 空 token，首次 cube_wait 前不会被使用

    tile.alloc.store %acc_o, %acc_init : !tile.type<[128, 128], f32, #tile.ub>
    tile.alloc.store %sumexp, %se_init : !tile.type<[128, 1], f32, #tile.ub>

    // pre-arm MTE3_MTE2 信号（首次 Vec1 还不存在，但 Bmm2 在第 3 拍要 wait 它）
    tile.set_flag {event = #tile.event<MTE3_MTE2>}

    // =========================================================================
    // ③ 主循环 —— 3-task 流水
    //
    //   迭代 g ∈ [0, n_sub + PIPE_DEPTH)：
    //     g=0          : Bmm1(0)
    //     g=1          : Bmm1(1), Vec1(0)+Bmm2(0)
    //     g∈[2,n_sub-1]: Bmm1(g), Vec1(g-1)+Bmm2(g-1), Vec2(g-2)
    //     g=n_sub      : Vec1(n_sub-1)+Bmm2(n_sub-1), Vec2(n_sub-2)
    //     g=n_sub+1    : Vec2(n_sub-1)
    //
    //   循环状态（iter_args）：
    //     %tok_b1 — Bmm1(g-1) 的 async token（上拍 cube_launch 产出，本拍 cube_wait 消费）
    //     %tok_b2 — Bmm2(g-1) 的 async token
    //
    //   acc_o/sumexp/r_factors/sumexp_is/neg_sm_buf 持存在 UB 内存中，
    //   通过 tile.alloc.store / tile.alloc.load 访问，不参与 iter_args。
    // =========================================================================

    %loop:2 = scf.for %g = %c0 to %n_loop step %c1
        iter_args(
            %tok_b1 = %tok_none,     // Bmm1(g-1) token
            %tok_b2 = %tok_none      // Bmm2(g-1) token
        )
        -> (!tile.cube_token, !tile.cube_token)
    {
        // ---- 辅助索引 ---------------------------------------------------------
        %g_ge_1   = arith.cmpi sge, %g, %c1 : index
        %g_ge_2   = arith.cmpi sge, %g, %c2 : index
        %g1       = arith.subi %g, %c1 : index
        %g2       = arith.subi %g, %c2 : index
        %g_lt_ns  = arith.cmpi slt, %g, %n_sub : index
        %g1_lt_ns = arith.cmpi slt, %g1, %n_sub : index
        %g2_lt_ns = arith.cmpi slt, %g2, %n_sub : index

        %cur_pp   = arith.remui %g,  %c2 : index     // taskId % 2
        %prev_pp  = arith.remui %g1, %c2 : index     // (taskId-1) % 2
        %prev2_pp = arith.remui %g2, %c2 : index     // (taskId-2) % 2

        !slot_ty = !tile.type<[128, 128], f16, #tile.l0a>  // l0a 单槽视图

        // =====================================================================
        // ─── 1) WaitBmm1Result(g-1)：等上拍 Bmm1 GM 落盘 ──────────────────────
        //     只有 g≥1 且 g-1 在有效范围内才执行。
        //     释放 l0c[0]（可被本轮 Bmm1(g) 复用）并保证 mm1Res[prev_pp] 完整。
        // =====================================================================
        %do_wait_b1 = and %g_ge_1, %g1_lt_ns : i1
        %tok_b1_done = scf.if %do_wait_b1 -> (!tile.cube_token) {
            tile.cube_wait %tok_b1 : !tile.cube_token
            scf.yield %tok_b1 : !tile.cube_token
        } else {
            scf.yield %tok_b1 : !tile.cube_token
        }

        // =====================================================================
        // ─── 2) IterateBmm1(g)：异步 Q(g)·K(g)^T → mm1Res[g%2] ──────────────
        //     只在 g < n_sub 时执行。K 从 GM→L1→L0B[0]，Q 从 L1→L0A[0]，
        //     Cube mma → L0C[0] → FIX → mm1Res[cur_pp]。
        //     返回 token（下一拍 cube_wait 消费）。
        // =====================================================================
        %tok_b1_new = scf.if %g_lt_ns -> (!tile.cube_token) {
            // ---- 任务坐标解码 ------------------------------------------------
            %t   = arith.divui %g, %n_iters : index
            %j   = arith.remui %g, %n_iters : index
            %tid = arith.addi %my_start, %t : index
            %bx  = arith.remui %tid, %num_seq_blocks : index
            %by  = arith.remui (arith.divui %tid, %num_seq_blocks), %heads_q : index
            %bz  = arith.divui %tid, (arith.muli %num_seq_blocks, %heads_q) : index
            %kv_by = arith.divui %by, %gqa_group : index

            // ---- 首子块加载 Q（常驻 L1）--------------------------------------
            %is_first_j = arith.cmpi eq, %j, %c0 : index
            scf.if %is_first_j {
                %q_gm = tt.make_block_ptr %Q_ptr
                    shape=(%S, %cD), strides=(%sQs, %sQd)
                    offsets=(%bx * %cBM, %c0), sizes=(%cBM, %cD), order=(1, 0)
                    : !tt.ptr<f16, #tile.global> -> !tt.block_ptr<memref<?x?xf16>, #tile.global>
                %q_load = tt.load %q_gm
                    : !tt.block_ptr<memref<?x?xf16>, #tile.global> -> tensor<128x128xf16>
                tile.prefetch src(%q_load) dst(%q_l1)
                    {kind = #tile.async_copy<g2s>, pipe = mte2, ticket = "q"}
                    : tensor<128x128xf16> -> !tile.type<[128, 128], f16, #tile.l1>
            }

            // K[j] → GM → L1（ticket = "k"）
            %k_gm = tt.make_block_ptr %K_ptr
                shape=(%S, %cD), strides=(%sKs, %sKd)
                offsets=(%j * %cBN, %c0), sizes=(%cBN, %cD), order=(1, 0)
                : !tt.ptr<f16, #tile.global> -> !tt.block_ptr<memref<?x?xf16>, #tile.global>
            %k_load = tt.load %k_gm
                : !tt.block_ptr<memref<?x?xf16>, #tile.global> -> tensor<128x128xf16>
            tile.prefetch src(%k_load) dst(%k_l1)
                {kind = #tile.async_copy<g2s>, pipe = mte2, ticket = "k"}
                : tensor<128x128xf16> -> !tile.type<[128, 128], f16, #tile.l1>

            // K L1 → L0B[0]（转置，Cube 需 B 列主序）
            %k_slot0 = tile.subview %l0b[%c0, 0, 0]
                [128, 128] [1, 1] : !tile.type<[2, 128, 128], f16, #tile.l0b>
                -> !tile.type<[128, 128], f16, #tile.l0b>
            tile.prefetch src(%k_l1) dst(%k_slot0)
                {kind = #tile.dma_copy<l12l0b>, transpose = true,
                 pipe = mte2, ticket = "k_l0"}
                : !tile.type<[128, 128], f16, #tile.l1> -> !tile.type<[128, 128], f16, #tile.l0b>

            // Q L1 → L0A[0]
            %q_slot0 = tile.subview %l0a[%c0, 0, 0]
                [128, 128] [1, 1] : !tile.type<[2, 128, 128], f16, #tile.l0a>
                -> !tile.type<[128, 128], f16, #tile.l0a>
            tile.prefetch src(%q_l1) dst(%q_slot0)
                {kind = #tile.dma_copy<l12l0a>,
                 pipe = mte2, ticket = "q_l0"}
                : !tile.type<[128, 128], f16, #tile.l1> -> !tile.type<[128, 128], f16, #tile.l0a>

            // ── 异步 Cube Bmm1：Q·K^T → mm1Res[cur_pp] ──────────────────────
            // 等效 AscendC IterateBmm1<async=false>：
            //   L0A[0]×L0B[0]→L0C[0]→FIX→GM mm1Res[cur_pp]
            %mm1_off     = arith.muli %cid, (%c2 * %cBM * %cBN) : index
            %mm1_pp_off  = arith.muli %cur_pp, (%cBM * %cBN) : index
            %mm1_base    = arith.addi %mm1_off, %mm1_pp_off : index
            %mm1_slot_bp = tt.make_block_ptr %mm1Res
                shape=(%cBM, %cBN), strides=(%cBN, 1)
                offsets=(%mm1_base, 0), sizes=(%cBM, %cBN), order=(1, 0)
                : !tt.ptr<f32, #tile.global> -> !tt.block_ptr<memref<?x?xf32>, #tile.global>

            %tok = tile.cube_launch %q_slot0, %k_slot0, %l0c[%c0], %mm1_slot_bp
                {init = true, cube_op = "bmm1", schema = "async"}
                : !tile.type<[128, 128], f16, #tile.l0a>,
                  !tile.type<[128, 128], f16, #tile.l0b>,
                  !tile.type<[128, 128], f32, #tile.l0c>,
                  !tt.block_ptr<memref<?x?xf32>, #tile.global>
                -> !tile.cube_token
            scf.yield %tok : !tile.cube_token

        } else {
            scf.yield %tok_b1_done : !tile.cube_token
        }

        // =====================================================================
        // ─── 3) ProcessVec1(g-1)：softmax(mm1Res[prev]) → stage1Res[prev] ────
        //     Vector 引擎，与 Cube Bmm1(g) 并行。
        //     流程：MTE2 取 mm1Res → Vector softmax → MTE3 写 stage1Res
        //     跨引擎事件：MTE2_V(取完→算) → V_MTE3(算完→写) → MTE3_MTE2(写完→Bmm2)
        // =====================================================================
        %do_vec1 = and %g_ge_1, %g1_lt_ns : i1
        scf.if %do_vec1 {
            %j1   = arith.remui %g1, %n_iters : index
            %pp1  = arith.remui %g1, %c2 : index
            %slot = arith.remui %g1, %c3 : index

            // neg_sm ping-pong 槽（子块内双缓冲）
            %neg_cur = arith.remui %j1, %c2 : index
            %neg_prv = arith.subi %c1, %neg_cur : index

            // ---- PIPE_V 内屏障（清空 Vector 流水线）--------------------------
            tile.pipe_barrier {pipe = #tile.pipe<V>}

            // ---- 首子块：初始化 neg_sm（running max = -inf）-------------------
            %is_first_v1 = arith.cmpi eq, %j1, %c0 : index
            scf.if %is_first_v1 {
                tile.alloc.store %neg_sm_buf, %neg_inf_v
                    : !tile.type<[2, 128, 1], f32, #tile.ub>
            }

            // ---- GM mm1Res[pp1] → UB s_tmp（MTE2）---------------------------
            %mm1_off1    = arith.muli %cid, (%c2 * %cBM * %cBN) : index
            %mm1_pp_off1 = arith.muli %pp1, (%cBM * %cBN) : index
            %mm1_base1   = arith.addi %mm1_off1, %mm1_pp_off1 : index
            %mm1_load_bp = tt.make_block_ptr %mm1Res
                shape=(%cBM, %cBN), strides=(%cBN, 1)
                offsets=(%mm1_base1, 0), sizes=(%cBM, %cBN), order=(1, 0)
                : !tt.ptr<f32, #tile.global> -> !tt.block_ptr<memref<?x?xf32>, #tile.global>
            %s_raw = tt.load %mm1_load_bp
                : !tt.block_ptr<memref<?x?xf32>, #tile.global> -> tensor<128x128xf32>
            tile.prefetch src(%s_raw) dst(%s_tmp)
                {kind = #tile.dma_copy<g2s>, pipe = mte2}
                : tensor<128x128xf32> -> !tile.type<[128, 128], f32, #tile.ub>

            tile.set_flag {event = #tile.event<MTE2_V>}
            tile.wait_flag {event = #tile.event<MTE2_V>}

            // ---- Vector：在线 softmax ----------------------------------------
            // s_tmp = load(m1) * sm_scale
            %scale_bc = tile.splat %sm_scale : f32 -> !tile.type<[128, 128], f32, #tile.ub>
            %s_scaled = tile.elemwise @arith.mulf(%s_tmp, %scale_bc)
                : !tile.type<[128, 128], f32, #tile.ub>,
                  !tile.type<[128, 128], f32, #tile.ub>
                -> !tile.type<[128, 128], f32, #tile.ub>

            // causal mask（TileMaskSplit pass 将 Phase 2 版本 DCE）
            scf.if %IS_CAUSAL {
                // tile.masked_region {phase = "with_mask"}
                //   期望 TileMaskSplit pass 处理：非 causal 时此段 DCE
            }

            // row_max = max(s, axis=1)
            %row_max = tile.reduce %s_scaled
                {axis = 1 : i64, op = @arith.maximumf, scope = #tile.warp}
                : !tile.type<[128, 128], f32, #tile.ub> -> !tile.type<[128, 1], f32, #tile.ub>

            // neg_cur = -sm_scale * row_max
            %neg_scale = arith.negf %sm_scale : f32
            %neg_bcast = tile.splat %neg_scale : f32 -> !tile.type<[128, 1], f32, #tile.ub>
            %neg_calc  = tile.elemwise @arith.mulf(%neg_bcast, %row_max)
                : !tile.type<[128, 1], f32, #tile.ub>,
                  !tile.type<[128, 1], f32, #tile.ub>
                -> !tile.type<[128, 1], f32, #tile.ub>

            // neg_new = min(neg_calc, neg_sm[prev]) — most-negative = largest m
            %neg_slot_prv = tile.subview %neg_sm_buf[%neg_prv, 0, 0]
                [128, 1] [1, 1] : !tile.type<[2, 128, 1], f32, #tile.ub>
                -> !tile.type<[128, 1], f32, #tile.ub>
            %neg_prv_v    = tile.alloc.load %neg_slot_prv
                : !tile.type<[128, 1], f32, #tile.ub>
            %neg_new      = tile.elemwise @arith.minimumf(%neg_calc, %neg_prv_v)
                : !tile.type<[128, 1], f32, #tile.ub>,
                  !tile.type<[128, 1], f32, #tile.ub>
                -> !tile.type<[128, 1], f32, #tile.ub>

            // 写回 neg_sm[cur] 槽
            %neg_slot_cur = tile.subview %neg_sm_buf[%neg_cur, 0, 0]
                [128, 1] [1, 1] : !tile.type<[2, 128, 1], f32, #tile.ub>
                -> !tile.type<[128, 1], f32, #tile.ub>
            tile.alloc.store %neg_slot_cur, %neg_new
                : !tile.type<[128, 1], f32, #tile.ub>

            // P = exp(sm_scale*s + neg_new_bc)
            %neg_bc = tile.broadcast %neg_new {axis = 1 : i64, shape = [128, 128]}
                : !tile.type<[128, 1], f32, #tile.ub> -> !tile.type<[128, 128], f32, #tile.ub>
            %s_shift = tile.elemwise @arith.addf(%s_scaled, %neg_bc)
                : !tile.type<[128, 128], f32, #tile.ub>,
                  !tile.type<[128, 128], f32, #tile.ub>
                -> !tile.type<[128, 128], f32, #tile.ub>
            %p_f32 = tile.elemwise @math.exp(%s_shift)
                : !tile.type<[128, 128], f32, #tile.ub> -> !tile.type<[128, 128], f32, #tile.ub>

            // ---- PIPE_V 屏障（Vector 引擎内 RAW）-----------------------------
            tile.pipe_barrier {pipe = #tile.pipe<V>}

            // ---- P UB → GM stage1Res[pp1]（MTE3）----------------------------
            tile.set_flag {event = #tile.event<V_MTE3>}
            tile.wait_flag {event = #tile.event<V_MTE3>}

            %p_f16   = tile.cast %p_f32
                : !tile.type<[128, 128], f32, #tile.ub> -> !tile.type<[128, 128], f16, #tile.ub>
            %s1_off  = arith.muli %cid, (%c2 * %cBM * %cBN) : index
            %s1_pp   = arith.muli %pp1, (%cBM * %cBN) : index
            %s1_base = arith.addi %s1_off, %s1_pp : index
            %s1_bp   = tt.make_block_ptr %stage1Res
                shape=(%cBM, %cBN), strides=(%cBN, 1)
                offsets=(%s1_base, 0), sizes=(%cBM, %cBN), order=(1, 0)
                : !tt.ptr<f16, #tile.global> -> !tt.block_ptr<memref<?x?xf16>, #tile.global>
            tt.store %s1_bp, %p_f16
                : !tt.block_ptr<memref<?x?xf16>, #tile.global>, !tile.type<[128, 128], f16, #tile.ub>

            tile.set_flag {event = #tile.event<MTE3_MTE2>}   // P[pp1] → Bmm2 可读

            // ---- 记录 Vec1 → Vec2 carry（per ring slot）---------------------
            // sumexp_is[slot] = rowsum(P)
            %p_rowsum = tile.reduce %p_f32
                {axis = 1 : i64, op = @arith.addf, scope = #tile.warp}
                : !tile.type<[128, 128], f32, #tile.ub> -> !tile.type<[128, 1], f32, #tile.ub>
            %se_slot = tile.subview %sumexp_is[%slot, 0, 0]
                [128, 1] [1, 1] : !tile.type<[3, 128, 1], f32, #tile.ub>
                -> !tile.type<[128, 1], f32, #tile.ub>
            tile.alloc.store %se_slot, %p_rowsum
                : !tile.type<[128, 1], f32, #tile.ub>

            // r_factors[slot] = neg_new - neg_prv_v (= -(m_new - m_prev) = m_prev - m_new)
            %r_factor = tile.elemwise @arith.subf(%neg_new, %neg_prv_v)
                : !tile.type<[128, 1], f32, #tile.ub>,
                  !tile.type<[128, 1], f32, #tile.ub>
                -> !tile.type<[128, 1], f32, #tile.ub>
            %rf_slot = tile.subview %r_factors[%slot, 0, 0]
                [128, 1] [1, 1] : !tile.type<[3, 128, 1], f32, #tile.ub>
                -> !tile.type<[128, 1], f32, #tile.ub>
            tile.alloc.store %rf_slot, %r_factor
                : !tile.type<[128, 1], f32, #tile.ub>
        }

        // =====================================================================
        // ─── 4) WaitBmm2Result(g-2)：等上上拍 Bmm2 GM 落盘 ───────────────────
        //     释放 l0c[1]，保证 mm2Res[prev2_pp] 可被 Vec2(g-2) 安全读取。
        // =====================================================================
        %do_wait_b2 = and %g_ge_2, %g2_lt_ns : i1
        %tok_b2_done = scf.if %do_wait_b2 -> (!tile.cube_token) {
            tile.cube_wait %tok_b2 : !tile.cube_token
            scf.yield %tok_b2 : !tile.cube_token
        } else {
            scf.yield %tok_b2 : !tile.cube_token
        }

        // =====================================================================
        // ─── 5) IterateBmm2(g-1)：异步 stage1Res[prev]·V → mm2Res[prev] ────
        //     等 MTE3_MTE2（Vec1 写 P 完成）后，P 从 GM→L1→L0A[1]，
        //     V 从 GM→L1→L0B[1]，Cube mma → L0C[1] → FIX → mm2Res[prev_pp]。
        //     与 Bmm1(g)（L0 slot 0）共用 Cube 但槽位不同，可流水重叠。
        // =====================================================================
        %tok_b2_new = scf.if %do_vec1 -> (!tile.cube_token) {
            tile.wait_flag {event = #tile.event<MTE3_MTE2>}

            %pp1  = arith.remui %g1, %c2 : index

            // stage1Res[pp1] → L1 p_l1
            %s1r_off  = arith.muli %cid, (%c2 * %cBM * %cBN) : index
            %s1r_pp   = arith.muli %pp1, (%cBM * %cBN) : index
            %s1r_base = arith.addi %s1r_off, %s1r_pp : index
            %s1r_bp   = tt.make_block_ptr %stage1Res
                shape=(%cBM, %cBN), strides=(%cBN, 1)
                offsets=(%s1r_base, 0), sizes=(%cBM, %cBN), order=(1, 0)
                : !tt.ptr<f16, #tile.global> -> !tt.block_ptr<memref<?x?xf16>, #tile.global>
            %s1r_data = tt.load %s1r_bp
                : !tt.block_ptr<memref<?x?xf16>, #tile.global> -> tensor<128x128xf16>
            tile.prefetch src(%s1r_data) dst(%p_l1)
                {kind = #tile.dma_copy<g2s>, pipe = mte2, ticket = "p"}
                : tensor<128x128xf16> -> !tile.type<[128, 128], f16, #tile.l1>

            // V[j1] → L1 v_l1
            %t1   = arith.divui %g1, %n_iters : index
            %j1   = arith.remui %g1, %n_iters : index
            %tid1 = arith.addi %my_start, %t1 : index
            %by1  = arith.remui (arith.divui %tid1, %num_seq_blocks), %heads_q : index
            %bz1  = arith.divui %tid1, (arith.muli %num_seq_blocks, %heads_q) : index
            %kv_by1 = arith.divui %by1, %gqa_group : index

            %v_gm = tt.make_block_ptr %V_ptr
                shape=(%S, %cD), strides=(%sKs, %sKd)
                offsets=(%j1 * %cBN, %c0), sizes=(%cBN, %cD), order=(1, 0)
                : !tt.ptr<f16, #tile.global> -> !tt.block_ptr<memref<?x?xf16>, #tile.global>
            %v_data = tt.load %v_gm
                : !tt.block_ptr<memref<?x?xf16>, #tile.global> -> tensor<128x128xf16>
            tile.prefetch src(%v_data) dst(%v_l1)
                {kind = #tile.dma_copy<g2s>, pipe = mte2, ticket = "v"}
                : tensor<128x128xf16> -> !tile.type<[128, 128], f16, #tile.l1>

            // L1 → L0 slot 1
            %p_slot1 = tile.subview %l0a[%c1, 0, 0]
                [128, 128] [1, 1] : !tile.type<[2, 128, 128], f16, #tile.l0a>
                -> !tile.type<[128, 128], f16, #tile.l0a>
            tile.prefetch src(%p_l1) dst(%p_slot1)
                {kind = #tile.dma_copy<l12l0a>, pipe = mte2, ticket = "p_l0"}
                : !tile.type<[128, 128], f16, #tile.l1> -> !tile.type<[128, 128], f16, #tile.l0a>

            %v_slot1 = tile.subview %l0b[%c1, 0, 0]
                [128, 128] [1, 1] : !tile.type<[2, 128, 128], f16, #tile.l0b>
                -> !tile.type<[128, 128], f16, #tile.l0b>
            tile.prefetch src(%v_l1) dst(%v_slot1)
                {kind = #tile.dma_copy<l12l0b>, pipe = mte2, ticket = "v_l0"}
                : !tile.type<[128, 128], f16, #tile.l1> -> !tile.type<[128, 128], f16, #tile.l0b>

            // ── 异步 Cube Bmm2：P·V → mm2Res[pp1] ──────────────────────────
            // 等效 AscendC IterateBmm2<async=false>，返回 token 供下下拍 cube_wait
            %mm2_off    = arith.muli %cid, (%c2 * %cBM * %cDv) : index
            %mm2_pp_off = arith.muli %pp1, (%cBM * %cDv) : index
            %mm2_base   = arith.addi %mm2_off, %mm2_pp_off : index
            %mm2_slot_bp = tt.make_block_ptr %mm2Res
                shape=(%cBM, %cDv), strides=(%cDv, 1)
                offsets=(%mm2_base, 0), sizes=(%cBM, %cDv), order=(1, 0)
                : !tt.ptr<f32, #tile.global> -> !tt.block_ptr<memref<?x?xf32>, #tile.global>

            %tok2 = tile.cube_launch %p_slot1, %v_slot1, %l0c[%c1], %mm2_slot_bp
                {init = true, cube_op = "bmm2", schema = "async"}
                : !tile.type<[128, 128], f16, #tile.l0a>,
                  !tile.type<[128, 128], f16, #tile.l0b>,
                  !tile.type<[128, 128], f32, #tile.l0c>,
                  !tt.block_ptr<memref<?x?xf32>, #tile.global>
                -> !tile.cube_token
            scf.yield %tok2 : !tile.cube_token

        } else {
            scf.yield %tok_b2_done : !tile.cube_token
        }

        // =====================================================================
        // ─── 6) ProcessVec2(g-2)：rescale 在线 softmax + add mm2Res + 写 O ──
        //     Vector 引擎，与 Bmm1(g) + Bmm2(g-1) 在 Cube 引擎上并行。
        //     从 task 环 [slot2] 读 Vec1 记录的 rescale_factor + rowsum_is，
        //     更新累加器 acc_o / sumexp。
        // =====================================================================
        %do_vec2 = and %g_ge_2, %g2_lt_ns : i1
        scf.if %do_vec2 {
            %j2     = arith.remui %g2, %n_iters : index
            %pp2    = arith.remui %g2, %c2 : index
            %slot2  = arith.remui %g2, %c3 : index
            %is_last  = arith.cmpi eq, %j2, %c2 : index   // == NUM_ITERS-1

            // ---- 首子块：累加器归零 -----------------------------------------
            %is_first_v2 = arith.cmpi eq, %j2, %c0 : index
            scf.if %is_first_v2 {
                %z_acc = tile.splat %c0_f : f32 -> !tile.type<[128, 128], f32, #tile.ub>
                %z_se  = tile.splat %c0_f : f32 -> !tile.type<[128, 1], f32, #tile.ub>
                tile.alloc.store %acc_o, %z_acc
                    : !tile.type<[128, 128], f32, #tile.ub>
                tile.alloc.store %sumexp, %z_se
                    : !tile.type<[128, 1], f32, #tile.ub>
            }

            // ---- 应用 Vec1 记录的 rescale factor ------------------------------
            // rf = exp(r_factors[slot2]) = exp(m_prev - m_new) ∈ (0, 1]
            %rf_slot2 = tile.subview %r_factors[%slot2, 0, 0]
                [128, 1] [1, 1] : !tile.type<[3, 128, 1], f32, #tile.ub>
                -> !tile.type<[128, 1], f32, #tile.ub>
            %rf       = tile.alloc.load %rf_slot2
                : !tile.type<[128, 1], f32, #tile.ub>
            %rf_exp   = tile.elemwise @math.exp(%rf)
                : !tile.type<[128, 1], f32, #tile.ub> -> !tile.type<[128, 1], f32, #tile.ub>

            // sumexp = sumexp * rf + sumexp_is[slot2]
            %se     = tile.alloc.load %sumexp
                : !tile.type<[128, 1], f32, #tile.ub>
            %se_is_slot2 = tile.subview %sumexp_is[%slot2, 0, 0]
                [128, 1] [1, 1] : !tile.type<[3, 128, 1], f32, #tile.ub>
                -> !tile.type<[128, 1], f32, #tile.ub>
            %se_is  = tile.alloc.load %se_is_slot2
                : !tile.type<[128, 1], f32, #tile.ub>
            %se_sc  = tile.elemwise @arith.mulf(%se, %rf_exp)
                : !tile.type<[128, 1], f32, #tile.ub>,
                  !tile.type<[128, 1], f32, #tile.ub>
                -> !tile.type<[128, 1], f32, #tile.ub>
            %se_new = tile.elemwise @arith.addf(%se_sc, %se_is)
                : !tile.type<[128, 1], f32, #tile.ub>,
                  !tile.type<[128, 1], f32, #tile.ub>
                -> !tile.type<[128, 1], f32, #tile.ub>
            tile.alloc.store %sumexp, %se_new
                : !tile.type<[128, 1], f32, #tile.ub>

            // acc_o = acc_o * rf
            %acc    = tile.alloc.load %acc_o
                : !tile.type<[128, 128], f32, #tile.ub>
            %rf_bc  = tile.broadcast %rf_exp {axis = 1 : i64, shape = [128, 128]}
                : !tile.type<[128, 1], f32, #tile.ub> -> !tile.type<[128, 128], f32, #tile.ub>
            %acc_sc = tile.elemwise @arith.mulf(%acc, %rf_bc)
                : !tile.type<[128, 128], f32, #tile.ub>,
                  !tile.type<[128, 128], f32, #tile.ub>
                -> !tile.type<[128, 128], f32, #tile.ub>

            // ---- mm2Res[pp2] → UB（MTE2）→ acc_o += o_part ----------------
            %o_base    = arith.muli %cid, (%c2 * %cBM * %cDv) : index
            %o_pp      = arith.muli %pp2, (%cBM * %cDv) : index
            %o_base_pp = arith.addi %o_base, %o_pp : index
            %o_part_bp = tt.make_block_ptr %mm2Res
                shape=(%cBM, %cDv), strides=(%cDv, 1)
                offsets=(%o_base_pp, 0), sizes=(%cBM, %cDv), order=(1, 0)
                : !tt.ptr<f32, #tile.global> -> !tt.block_ptr<memref<?x?xf32>, #tile.global>
            %o_part_t = tt.load %o_part_bp
                : !tt.block_ptr<memref<?x?xf32>, #tile.global> -> tensor<128x128xf32>
            tile.prefetch src(%o_part_t) dst(%s_tmp)
                {kind = #tile.dma_copy<g2s>, pipe = mte2}
                : tensor<128x128xf32> -> !tile.type<[128, 128], f32, #tile.ub>

            tile.set_flag {event = #tile.event<MTE2_V>}
            tile.wait_flag {event = #tile.event<MTE2_V>}

            %o_part_f32 = tile.alloc.load %s_tmp
                : !tile.type<[128, 128], f32, #tile.ub>
            %acc_new = tile.elemwise @arith.addf(%acc_sc, %o_part_f32)
                : !tile.type<[128, 128], f32, #tile.ub>,
                  !tile.type<[128, 128], f32, #tile.ub>
                -> !tile.type<[128, 128], f32, #tile.ub>
            tile.alloc.store %acc_o, %acc_new
                : !tile.type<[128, 128], f32, #tile.ub>

            // ---- 末子块：归一化 + 写 O ---------------------------------------
            scf.if %is_last {
                // O = acc / sumexp
                %se_bc = tile.broadcast %se_new {axis = 1 : i64, shape = [128, 128]}
                    : !tile.type<[128, 1], f32, #tile.ub> -> !tile.type<[128, 128], f32, #tile.ub>
                %o_div  = tile.elemwise @arith.divf(%acc_new, %se_bc)
                    : !tile.type<[128, 128], f32, #tile.ub>,
                      !tile.type<[128, 128], f32, #tile.ub>
                    -> !tile.type<[128, 128], f32, #tile.ub>
                %o_f16  = tile.cast %o_div
                    : !tile.type<[128, 128], f32, #tile.ub> -> !tile.type<[128, 128], f16, #tile.ub>

                // 计算 GM 偏移
                %t2   = arith.divui %g2, %n_iters : index
                %tid2 = arith.addi %my_start, %t2 : index
                %bx2  = arith.remui %tid2, %num_seq_blocks : index
                %by2  = arith.remui (arith.divui %tid2, %num_seq_blocks), %heads_q : index
                %bz2  = arith.divui %tid2, (arith.muli %num_seq_blocks, %heads_q) : index

                %o_bp = tt.make_block_ptr %Out_ptr
                    shape=(%S, %cDv), strides=(%sOs, %sOd)
                    offsets=(%bx2 * %cBM, %c0), sizes=(%cBM, %cDv), order=(1, 0)
                    : !tt.ptr<f16, #tile.global> -> !tt.block_ptr<memref<?x?xf16>, #tile.global>

                tile.set_flag {event = #tile.event<V_MTE3>}
                tile.wait_flag {event = #tile.event<V_MTE3>}
                tt.store %o_bp, %o_f16
                    : !tt.block_ptr<memref<?x?xf16>, #tile.global>,
                      !tile.type<[128, 128], f16, #tile.ub>
            }
        }

        // =====================================================================
        // 更新 iter_args 并进入下一拍
        // =====================================================================
        scf.yield %tok_b1_new, %tok_b2_new
            : !tile.cube_token, !tile.cube_token
    }

    tile.return
}

// =============================================================================
// 下降映射说明（skill.md §3 输出要求）
//
// TileOp → HIVM Op 映射表：
//
// | TileOp                                | HIVM Op                             | 映射类型 |
// |---------------------------------------|-------------------------------------|---------|
// | tile.alloc<*, #tile.l1>               | hivm.hir.alloc {mem_space = l1}     | 直接    |
// | tile.alloc<*, #tile.l0a/l0b/l0c>      | hivm.hir.alloc {mem_space = l0x}    | 直接    |
// | tile.alloc<*, #tile.ub>               | hivm.hir.alloc {mem_space = ub}     | 直接    |
// | tile.prefetch {kind = g2s, pipe=mte2} | hivm.hir.dma_load_gm_to_ub          | 直接    |
// | tile.prefetch {kind = l12l0a}         | hivm.hir.dma_load_l1_to_l0a         | 直接    |
// | tile.prefetch {kind = l12l0b}         | hivm.hir.dma_load_l1_to_l0b         | 直接    |
// | tile.cube_launch {cube_op=bmm1}       | hivm.hir.cube (Bmm1 async)          | 直接    |
// | tile.cube_launch {cube_op=bmm2}       | hivm.hir.cube (Bmm2 async)          | 直接    |
// | tile.cube_wait                        | hivm.hir.cube_wait                  | 直接    |
// | tile.set_flag {event=MTE3_MTE2}       | hivm.hir.set_flag                   | 直接    |
// | tile.wait_flag {event=MTE3_MTE2}      | hivm.hir.wait_flag                  | 直接    |
// | tile.pipe_barrier {pipe=V}            | hivm.hir.pipe_barrier               | 直接    |
// | tile.subview (multi-buffer slot)      | hivm.hir.subview                    | 直接    |
// | tile.elemwise / tile.reduce           | hivm.hir.vec_*                      | 组合    |
// | tile.broadcast                        | hivm.hir.broadcast                  | 直接    |
// | tt.dot                                | hivm.hir.cube_mmad                  | 组合    |
// | tt.load / tt.store (block_ptr)        | hivm.hir.dma_load / dma_store       | 直接    |
//
// HIVM 表达力缺口清单：
//   - tile.cube_launch {schema="async"} + !tile.cube_token：
//     当前 HIVM dialect 无显式异步 Cube token 类型。
//     建议新增 hivm.hir.cube_launch_op / hivm.hir.cube_wait_op，
//     或通过 hivm.hir.set_flag + hivm.hir.cube 组合表达。
//   - tile.alloc.store / tile.alloc.load：
//     当前 HIVM 无独立的 "buffer store/load" 指令，
//     建议通过 hivm.hir.dma_copy 或直接内存访问表达。
// =============================================================================