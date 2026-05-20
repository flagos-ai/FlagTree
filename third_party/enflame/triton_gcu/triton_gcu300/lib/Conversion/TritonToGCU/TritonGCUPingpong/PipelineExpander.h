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

#ifndef TRITON_DIALECT_TRIRONTOGCU_PINGPONG_PIPELINE_H_
#define TRITON_DIALECT_TRIRONTOGCU_PINGPONG_PIPELINE_H_

// This is a fork of upstream pipeline transformation. This will be merged back
// upstream once we have a stable solution.
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/ArrayRef.h"
#include <utility>
#include <vector>
namespace mlir {

class RewriterBase;
class Operation;
class Value;

namespace scf {
class ForOp;
}

namespace triton {
namespace gcu {
/// Options to dictate how loops should be pipelined.
struct PipeliningOption {
  /// Lambda returning all the operation in the forOp, with their stage, in the
  /// order picked for the pipelined loop.
  using GetScheduleFnType = std::function<void(
      scf::ForOp, std::vector<std::pair<Operation *, unsigned>> &)>;
  GetScheduleFnType getScheduleFn = nullptr;
  enum class PipelinerPart {
    Prologue,
    Kernel,
    Epilogue,
  };
  /// Lambda called by the pipeliner to allow the user to annotate the IR while
  /// it is generated.
  /// The callback passes the operation created along with the part of the
  /// pipeline and the iteration index. The iteration index is always 0 for the
  /// kernel. For the prologue and epilogue, it corresponds to the iteration
  /// peeled out of the loop in the range [0, maxStage[.
  using AnnotationlFnType =
      std::function<void(Operation *, PipelinerPart, unsigned)>;
  AnnotationlFnType annotateFn = nullptr;

  /// Control whether the epilogue should be peeled out of the loop or
  /// operations should be predicated to skip the early stages in the last loop
  /// iterations. If the epilogue is predicated; the user needs to provide a
  /// lambda to generate the predicated version of operations.
  bool peelEpilogue = true;

  /// Control whether the transformation checks that the number of iterations is
  /// greater or equal to the number of stages and skip the transformation if
  /// this is not the case. If the loop is dynamic and this is set to true the
  /// pipeliner will have to predicate operations in the the prologue/epilogue.
  bool supportDynamicLoops = false;

  // Callback to predicate operations when the prologue or epilogue are not
  // peeled. This takes the original operation, an i1 predicate value and the
  // pattern rewriter. It is expected to replace the given operation with
  // the predicated equivalent and return it, or return nullptr if the
  // predication is impossible. In the latter case, pipelining will fail and
  // may leave IR in a partially transformed state.
  using PredicateOpFnType =
      std::function<Operation *(RewriterBase &, Operation *, Value)>;
  PredicateOpFnType predicateFn = nullptr;

  // TODO(triton): add option to decide if the prologue should be peeled.
};

/// Generate a pipelined version of the scf.for loop based on the schedule given
/// as option. This applies the mechanical transformation of changing the loop
/// and generating the prologue/epilogue for the pipelining and doesn't make any
/// decision regarding the schedule.
/// Based on the options the loop is split into several stages.
/// The transformation assumes that the scheduling given by user is valid.
/// For example if we break a loop into 3 stages named S0, S1, S2 we would
/// generate the following code with the number in parenthesis as the iteration
/// index:
///
///   S0(0)                        // Prologue
///   S0(1) S1(0)                  // Prologue
///   scf.for %I = %C0 to %N - 2 {
///     S0(I+2) S1(I+1) S2(I)       // Pipelined kernel
///   }
///   S1(N) S2(N-1)                // Epilogue
///   S2(N)                        // Epilogue
///
/// If `modifiedIR` is provided, it will be set to a value that indicates
/// whether pipelining modified the IR before failing, signaling to the caller
/// whether they can proceed with different transformations.
FailureOr<scf::ForOp> pipelineForLoop(RewriterBase &rewriter, scf::ForOp forOp,
                                      const PipeliningOption &options,
                                      bool *modifiedIR = nullptr);

} // namespace gcu
} // namespace triton
} // namespace mlir
#endif // TRITON_DIALECT_TRIRONTOGCU_PINGPONG_PIPELINE_H_
