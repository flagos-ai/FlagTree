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
#ifndef GCUWS_OPS
#define GCUWS_OPS

include "GCUWSDialect.td"
include "GCUWSTypes.td"
include "GCUWSAttrDefs.td"
include "GCUWSOpInterfaces.td"

include "mlir/IR/OpBase.td"
include "mlir/IR/EnumAttr.td"
include "mlir/Interfaces/InferTypeOpInterface.td" // SameOperandsAndResultType
include "mlir/Interfaces/ControlFlowInterfaces.td" // RegionBranchOpInterface
include "mlir/Interfaces/DestinationStyleOpInterface.td"
include "mlir/Interfaces/InferTypeOpInterface.td"  // SameOperandsAndResultType
include "mlir/Interfaces/SideEffectInterfaces.td"  // Pure
include "mlir/Interfaces/ViewLikeInterface.td"
include "triton/Dialect/Triton/IR/TritonTypes.td"
include "triton/Dialect/Triton/IR/TritonAttrDefs.td"
include "triton/Dialect/TritonGPU/IR/TritonGPUTypes.td"

class GCUWS_Op<string mnemonic, list<Trait> traits = []> :
    Op<GCUWS_Dialect, mnemonic, traits>;

def GCUWS_InitPipelineOp : GCUWS_Op<"init_pipeline", [Pure]> {
  let summary = "Initialize pipeline.";

  let description = [{
    Initialize pipeline.
  }];

  let results = (outs Res<GCUWS_PipelineType, "",
                          [MemAlloc<DefaultResource, 0, FullEffect>]>:$pipeline);

  let assemblyFormat = [{
     `(` `)` attr-dict `:` type(results)
  }];
}

def GCUWS_ProducerAcquireOp : GCUWS_Op<"producer_acquire"> {
  let summary = "Producer acquires a token to fill buffers";

  let description = [{
    The producer will try to acquire the token prior to filling the buffers.
    If the buffers are not ready to be filled, the producer will wait to be
    signalled by the consumer which finishes consuming the buffers and
    releases the token.
  }];

  let arguments = (ins GCUWS_PipelineType:$pipeline);

  let assemblyFormat = "`(` operands `)` attr-dict `:` type(operands)";
}

def GCUWS_ProducerCommitOp : GCUWS_Op<"producer_commit"> {
  let summary = "Producer commits the buffer changes";

  let description = [{ The producer will release the token and signal the consumer
    that the buffers are ready to be consumed.
  }];

  let arguments = (ins GCUWS_PipelineType:$pipeline);

  let assemblyFormat = "`(` operands `)` attr-dict `:` type(operands)";
}

def GCUWS_ConsumerWaitOp : GCUWS_Op<"consumer_wait"> {
  let summary = "Consumer awaits buffer readiness";

  let description = [{
    The consumer will wait for the buffer to be ready to be consumed.
    If the buffers are not ready, the consumer will wait to be
    signalled by the producer which finishes filling the buffers and
    releases the token.
  }];

  let arguments = (ins GCUWS_PipelineType:$pipeline);

  let assemblyFormat = "`(` operands `)` attr-dict `:` type(operands)";
}

def GCUWS_ConsumerReleaseOp : GCUWS_Op<"consumer_release"> {
  let summary = "Consumer releases the token";
  let description = [{
    The consumer will release the token and signal the producer that
    the buffers are ready to be filled.
  }];

  let arguments = (ins GCUWS_PipelineType:$pipeline);

  let assemblyFormat = "`(` operands `)` attr-dict `:` type(operands)";
}

#endif
