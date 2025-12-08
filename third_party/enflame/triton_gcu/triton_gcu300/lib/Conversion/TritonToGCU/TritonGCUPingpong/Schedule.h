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

#ifndef TRITON_TRIRONTOGCU_PINGPONG_SCHEDULE_H_
#define TRITON_TRIRONTOGCU_PINGPONG_SCHEDULE_H_

#include "PipelineExpander.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/ArrayRef.h"
#include <vector>

namespace mlir {
namespace triton {
namespace gcu {

/// This fill out the pipelining options including schedule and annotations
/// for wait ops. This also does pre-processing by converting some of the
/// loads into async loads so that the IR is ready to be pipelined.
bool preProcessLoopAndGetSchedule(scf::ForOp &forOp, int numStages,
                                  mlir::triton::gcu::PipeliningOption &options);

/// Fills out pipelining options for an outer loop pipelining case. This
/// schedules async copies to overlap with the epilogue of a loop.
bool getOuterLoopSchedule(scf::ForOp &forOp, int numStages,
                          mlir::triton::gcu::PipeliningOption &options);

} // namespace gcu
} // namespace triton
} // namespace mlir
#endif // TRITON_TRITONGPU_TRANSFORM_PIPELINE_SCHEDULE_H_
