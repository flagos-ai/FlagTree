/**
 * Copyright 2025-2026 Enflame. All Rights Reserved.
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
#ifndef TRITON_GCU_DIALECT_WS_TRANSFORMS_ATTRNAME_H_
#define TRITON_GCU_DIALECT_WS_TRANSFORMS_ATTRNAME_H_

namespace mlir {
namespace triton {

#ifdef NUM_STAGE
static const char *kNumStagesAttrName = "tt.num_stages";
#undef NUM_STAGE
#endif

#ifdef WARP_SPECIALIZE
static const char *kWarpSpecializeAttrName = "tt.warp_specialize";
#undef WARP_SPECIALIZE
#endif

} // namespace triton
} // namespace mlir

#endif // TRITON_GCU_DIALECT_WS_TRANSFORMS_ATTRNAME_H_
