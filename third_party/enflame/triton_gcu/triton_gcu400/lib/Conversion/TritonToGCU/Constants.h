/**
 * Copyright 2024-2026 Enflame. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *  http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef KURAMA_TRITONGPU_TO_GCU_CONSTANTS_H
#define KURAMA_TRITONGPU_TO_GCU_CONSTANTS_H

#include <climits>
#include <cstdint>

namespace mlir {

// -2147483647 (INT32_MIN + 1), used as a sentinel to indicate dynamic stride
constexpr static int32_t Dynamic_stride_symbol = INT32_MIN + 1;

constexpr static int64_t INVALID_ALIGNMENT = -1;
constexpr static char kAlignment[] = "alignment";

constexpr static unsigned kOaccSizeInBytes = 512;
constexpr static unsigned kMinVecSizeInBytes = 128;
constexpr static unsigned kLoopUnrollTimes = 16;
const char *const kIsContinual = "IsContinual";
const char *const kConstancy = "Constancy";

constexpr static int64_t OACC_F32_LENGTH = 128; // 128 elements for float32
constexpr static int64_t OACC_MAX_NUM = 192;
constexpr static int64_t GEMM_MIN_M = 32; // minimum M for gemm instructions

constexpr static char kGcuSinkImplicitDef[] = "gcu.sink_implicit_def";
constexpr static char kTotalNumWarps[] = "ttg.total-num-warps";
constexpr static char kNumWarps[] = "ttg.num-warps";
constexpr static char kAccReuseCandidate[] = "acc_reuse_candidate";
constexpr static char kAccReuseLocal[] = "acc_reuse_local";
constexpr static char kAccReuseOacc[] = "acc_reuse_oacc";
constexpr static char kAccLoad[] = "acc_load";
constexpr static char kAccLoadNone[] = "none";
constexpr static char kAccLoadConstant[] = "constant";
constexpr static char kAccLoadGlobal[] = "global";
constexpr static char kAccLoadLocal[] = "local";
constexpr static char kAccLoadOacc[] = "oacc";
constexpr static char kAccStore[] = "acc_store";
constexpr static char kAccStoreNone[] = "none";
constexpr static char kAccStoreGlobal[] = "global";
constexpr static char kAccStoreCvtGlobal[] = "cvt_global";
constexpr static char kAccStoreLocal[] = "local";
constexpr static char kAccStoreCvtLocal[] = "cvt_local";
constexpr static char kLoadAsync[] = "tt.load.async";
// ---------------------------------------------------------------------------
// TLE memory-space hint: tle.gpu.memory_space(value, space).
//
// Two groups, one per side of GCUTleAssignMemorySpace (Transforms/
// GCUTleAssignMemorySpace.cpp):
//
//   1. kTleHint*      - the frontend side. The hint reaches the backend as a
//                       `tt.memory_space` StringAttr on the op that defines the
//                       value; the pass reads it and removes it. No other pass
//                       may look at these.
//   2. kTle*          - the markers the pass leaves behind, consumed by the
//                       TritonToGCU lowerings. The same marker name is carried
//                       either as a UnitAttr on the hinted op, or as a
//                       DenseI64ArrayAttr of result indices on the
//                       triton_gcu.elementwise_fusion_region that contains it
//                       (there the value consumers see is the region result).
// ---------------------------------------------------------------------------
// Frontend: the hint attribute and the two space names it accepts.
constexpr static char kTleHint[] = "tt.memory_space";
constexpr static char kTleHintShared[] = "shared_memory";
constexpr static char kTleHintRegister[] = "register";
// "This result must be computed directly in the shared-memory tile": the
// producer writes the tile through a per-warp view instead of a private buffer
// that is desliced into it afterwards. The same marker is also put on the
// ttg.local_alloc(src) created for the hint, so the producer lowering can find
// the staging pair that consumes its value.
constexpr static char kTleResultInSmemTile[] = "tle_use_smem_tile";
// "This result must live in a memref.alloca": vector.load/vector.store path
// instead of the memref.alloc tar_load/tar_store path.
constexpr static char kTleResultInAlloca[] = "tle_use_alloca";

// Set (DenseI32ArrayAttr) on the shared-tag memref alloc: the tag indices whose
// DTE notifies its pipeline on completion, so the backend binds those DTE slots
// to the pipe instead of waiting on them.
constexpr static char kNotifyPipelineTags[] = "ttgcu.notify_pipeline_tags";

// Set (UnitAttr) on a triton_gcu.store whose value should be written by the
// backend's gcu.matrix_store rather than copied element-wise.
constexpr static char kMaxtrixStore[] = "matrix_store";

// Set on a triton_gcu.elementwise_fusion_region op by AnnotateDotFusion when
// the fused accumulator can be kept in OACC across a loop. Its integer value
// is the operand index that the fusion should write to in place (the loop
// iter-arg / OACC buffer), instead of allocating a fresh output buffer.
constexpr static char kAccReuseInplaceOperand[] = "acc_reuse_inplace_operand";
constexpr static char kAccReuseInplaceResult[] = "acc_reuse_inplace_result";
constexpr static char kDirectStore[] = "direct_store";

// Set on a triton::DotOp by AnnotateDotAllocaReuse to indicate that the
// dot's oacc alloca can be shared with a previous dot. Dots with the same
// integer group ID share the same alloca buffer.
constexpr static char kAllocaReuseGroup[] = "alloca_reuse_group";

// Set on a triton::DotOp by OptimizeDotOperands pass to indicate the right-hand
// operand has been transposed and should use column-major layout in the
// gemm builtin call.
constexpr static char kRhsColumnMajor[] = "rhs_column_major";

} // namespace mlir

#endif // KURAMA_TRITONGPU_TO_GCU_CONSTANTS_H
