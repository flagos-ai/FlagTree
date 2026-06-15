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

namespace mlir {

constexpr static int64_t INVALID_ALIGNMENT = -1;
constexpr static char kAlignment[] = "alignment";

constexpr static int64_t VECTOR_BYTES = 512;
constexpr static int64_t OACC_F32_LENGTH = 128; // 128 elements for float32
constexpr static int64_t OACC_MAX_NUM = 128;
constexpr static int64_t GEMM_MIN_M = 32; // minimum M for gemm instructions

constexpr static char kTotalNumWarps[] = "ttg.total-num-warps";
constexpr static char kNumWarps[] = "ttg.num-warps";
constexpr static char kAccReuseCandidate[] = "acc_reuse_candidate";
constexpr static char kAccReuseLocal[] = "acc_reuse_local";
constexpr static char kAccReuseOacc[] = "acc_reuse_oacc";
constexpr static char kAccStore[] = "acc_store";
constexpr static char kAccStoreGlobal[] = "global";
constexpr static char kAccStoreCvtGlobal[] = "cvt_global";
constexpr static char kAccStoreLocal[] = "local";
constexpr static char kAccStoreCvtLocal[] = "cvt_local";
constexpr static char kLoadAsync[] = "tt.load.async";

} // namespace mlir

#endif // KURAMA_TRITONGPU_TO_GCU_CONSTANTS_H
