
#ifndef XSMT_OPS
#define XSMT_OPS

include "mlir/IR/OpBase.td"
include "triton/Dialect/Triton/IR/TritonTypes.td"
include "mlir/IR/EnumAttr.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "triton-shared/Dialect/XSMT/IR/XSMTDialect.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/DestinationStyleOpInterface.td"
include "mlir/Interfaces/ViewLikeInterface.td"
include "triton/Dialect/Triton/IR/TritonOpInterfaces.td"
include "triton/Dialect/Triton/IR/TritonAttrDefs.td"
include "mlir/Dialect/Arith/IR/ArithBase.td"
include "triton-shared/Dialect/XSMT/IR/XSMTTypes.td"


class XSMT_Op<string mnemonic, list<Trait> traits = []> :
    Op<XSMT_Dialect, mnemonic, traits>;

def AnnotationOp : XSMT_Op<"annotation", [Pure, MemoryEffects<[MemWrite]>]> {
  let summary = "Annotate a tensor with key-value attribute pairs";
  let description = [{
    `xsmt.annotation` operation can be used to annotate a tensor with
    key-value attribute pairs.

    Example:
    ```mlir
    xsmt.annotation %target {key : val}
    ```
  }];
  let arguments = (ins TT_Tensor:$src);
  let assemblyFormat = [{
    $src attr-dict `:` type($src)
  }];
}


// ===----------------------------------------------------------------------===//
// New ops: split from ViewOp / ViewPtrOp (Spec 11)
// ===----------------------------------------------------------------------===//

def PackOp : XSMT_Op<"pack", [Pure, AttrSizedOperandSegments]> {
  let summary = "Pack a 2D tensor into 4D packed layout";
  let description = [{
    Packs a 2D tensor into a 4D tensor with packed tile layout.
    Result shape: [shape[0]/packed_size[0], shape[1]/packed_size[1], packed_size[0], packed_size[1]].

    If `destination` is provided, the pack result is written into the destination
    tensor (destination-passing style). This avoids generating intermediate
    alloc + materialize_in_destination roundtrips during lowering.
  }];

  let arguments = (ins
    TT_Tensor:$base,
    Variadic<I32>:$offsets,
    Optional<TT_Tensor>:$destination,
    DenseI32ArrayAttr:$shape,
    DenseI32ArrayAttr:$packed_size
  );

  let results = (outs TT_Tensor:$result);

  let hasCustomAssemblyFormat = 1;

  let builders = [
    // Builder without destination (backward compatible)
    OpBuilder<(ins
      "Value":$base,
      "ValueRange":$offsets,
      "ArrayRef<int32_t>":$shape,
      "ArrayRef<int32_t>":$packed_size
    )>,
    // Builder with destination (destination-passing style)
    OpBuilder<(ins
      "Value":$base,
      "ValueRange":$offsets,
      "Value":$destination,
      "ArrayRef<int32_t>":$shape,
      "ArrayRef<int32_t>":$packed_size
    )>
  ];

  let hasVerifier = 1;
}

def UnpackOp : XSMT_Op<"unpack", [Pure, AttrSizedOperandSegments]> {
  let summary = "Unpack a 4D packed tensor back to 2D";
  let description = [{
    Unpacks a 4D packed tensor back to a 2D tensor.
    The packed tile size is inferred from the input 4D shape (dim[2], dim[3]).

    If `destination` is provided, the unpack result is written into the
    destination tensor (destination-passing style). This enables direct
    unpack into a target subview without intermediate full-size tensors.
  }];

  let arguments = (ins
    TT_Tensor:$base,
    Variadic<I32>:$offsets,
    Optional<TT_Tensor>:$destination,
    DenseI32ArrayAttr:$shape
  );

  let results = (outs TT_Tensor:$result);

  let hasCustomAssemblyFormat = 1;

  let builders = [
    // Builder without destination (backward compatible)
    OpBuilder<(ins
      "Value":$base,
      "ValueRange":$offsets,
      "ArrayRef<int32_t>":$shape
    )>,
    // Builder with destination (destination-passing style)
    OpBuilder<(ins
      "Value":$base,
      "ValueRange":$offsets,
      "Value":$destination,
      "ArrayRef<int32_t>":$shape
    )>
  ];

  let hasVerifier = 1;
}

def RepackOp : XSMT_Op<"repack", [Pure, AttrSizedOperandSegments]> {
  let summary = "Repack a 4D tensor with different packed tile size";
  let description = [{
    Repacks a 4D packed tensor into a new 4D layout with different packed tile sizes.
    Equivalent to unpack + slice + pack with new packed tiles.

    If `destination` is provided, the repack result is written into the
    destination tensor (destination-passing style).
  }];

  let arguments = (ins
    TT_Tensor:$base,
    Variadic<I32>:$offsets,
    Optional<TT_Tensor>:$destination,
    DenseI32ArrayAttr:$shape,
    DenseI32ArrayAttr:$packed_size
  );

  let results = (outs TT_Tensor:$result);

  let hasCustomAssemblyFormat = 1;

  let builders = [
    // Builder without destination (backward compatible)
    OpBuilder<(ins
      "Value":$base,
      "ValueRange":$offsets,
      "ArrayRef<int32_t>":$shape,
      "ArrayRef<int32_t>":$packed_size
    )>,
    // Builder with destination (destination-passing style)
    OpBuilder<(ins
      "Value":$base,
      "ValueRange":$offsets,
      "Value":$destination,
      "ArrayRef<int32_t>":$shape,
      "ArrayRef<int32_t>":$packed_size
    )>
  ];

  let hasVerifier = 1;
}

def SubviewOp : XSMT_Op<"subview", [Pure]> {
  let summary = "Create a pointer subview without changing packing";
  let description = [{
    Creates a subview of a tensor pointer, preserving the existing packed tile layout.
    No packed_size attribute — the packing is inherited from the base.
  }];

  let arguments = (ins
    TT_TensorPtr:$base,
    Variadic<I32>:$offsets,
    DenseI32ArrayAttr:$shape
  );

  let results = (outs TT_TensorPtr:$result);

  let assemblyFormat = [{
    $base `,` `[` $offsets `]` `,` `[` $shape `]` attr-dict `:`
    type($base) `->` type($result)
  }];

  let builders = [
    OpBuilder<(ins
      "Value":$base,
      "ValueRange":$offsets,
      "ArrayRef<int32_t>":$shape
    )>
  ];

  let hasVerifier = 1;
}

def SubviewPackOp : XSMT_Op<"subview_pack", [Pure]> {
  let summary = "Create a pointer subview with packed tile packing";
  let description = [{
    Creates a subview of a tensor pointer and applies packed tile packing.
    Combines subview + expand_shape + transpose in lowering.
  }];

  let arguments = (ins
    TT_TensorPtr:$base,
    Variadic<I32>:$offsets,
    DenseI32ArrayAttr:$shape,
    DenseI32ArrayAttr:$packed_size
  );

  let results = (outs TT_TensorPtr:$result);

  let assemblyFormat = [{
    $base `,` `[` $offsets `]` `,` `[` $shape `]` `,` `[` $packed_size `]` attr-dict `:`
    type($base) `->` type($result)
  }];

  let builders = [
    OpBuilder<(ins
      "Value":$base,
      "ValueRange":$offsets,
      "ArrayRef<int32_t>":$shape,
      "ArrayRef<int32_t>":$packed_size
    )>
  ];

  let hasVerifier = 1;
}

def DescriptorLoadViewOp : XSMT_Op<"descriptor_load_view", [Pure]> {
  let summary = "fuse DescriptorLoadOp and ViewOp";
  let description = [{
  }];

  let arguments = (ins
    TT_Ptr:$base,
    Variadic<I64>:$offsets,
    DenseI32ArrayAttr:$shape,
    DenseI32ArrayAttr:$packed_size
  );

  let results = (outs TT_Tensor:$result);

  let assemblyFormat = [{
    $base `,` `[` $offsets `]` `,` `[` $shape `]` `,` `[` $packed_size `]` attr-dict `:`
    type($base) `->` type($result)
  }];

  let builders = [
    OpBuilder<(ins
      "Value":$base,
      "ValueRange":$offsets,
      "ArrayRef<int32_t>":$shape,
      "ArrayRef<int32_t>":$packed_size
    )>
  ];
}

def AllocOp : XSMT_Op<"alloc", [MemoryEffects<[MemAlloc]>]> {
  let summary = "Allocate a tensor in specified memory storage";
  let description = [{
    Allocates memory for a tensor with given shape in specified storage (e.g., l2, shared).
  }];

  let arguments = (ins
    DenseI32ArrayAttr:$shape,
    StrAttr:$storage
  );

  let results = (outs TT_TensorPtr:$result);

  let assemblyFormat = [{
    `[` $shape `]` `storage` `(` $storage `)` attr-dict `:` type($result)
  }];
}

def AllocCopiesOp : XSMT_Op<"alloc_copies", [MemoryEffects<[MemAlloc]>]> {
  let summary = "Allocate a buffer tensor in specified memory storage";
  let description = [{
    Allocate a multi-copy tensor. The first dimension of `shape` is the number of copies.
    i.e. shape = [num, M, N].
  }];

  let arguments = (ins
    DenseI64ArrayAttr:$shape,   // shape already includes num: [num, M, N]
    StrAttr:$storage            // e.g. "l2", "smem"
  );

  let results = (outs XSMT_BufferType:$result);

  let builders = [
    OpBuilder<(ins
      "llvm::ArrayRef<int64_t>":$shape,
      "mlir::Type":$elementType,
      "llvm::StringRef":$storage
    ), [{
      auto *ctx = $_builder.getContext();
      if (shape.empty()) {
        llvm_unreachable("alloc_copies shape cannot be empty");
      }

      int copies = static_cast<int>(shape.front());
      auto shapeAttr = $_builder.getDenseI64ArrayAttr(shape);
      auto storageAttr = ::mlir::StringAttr::get(ctx, storage);

      auto resultTy = ::mlir::xsmt::BufferType::get(
          ctx, shape, elementType, copies, storageAttr);

      build($_builder, $_state, resultTy, shapeAttr, storageAttr);
    }]>
  ];

  let assemblyFormat = [{
  `[` $shape `]` `storage` `(` $storage `)` attr-dict `:` qualified(type($result))
  }];

  // let hasVerifier = 1;
}

def MMT4DOp : XSMT_Op<"mmt4d", [Pure, MemoryEffects<[MemWrite]>]> {
  let summary = "Matrix multiplication with 4D packed tensors";
  let description = [{
    The `xsmt.mmt4d` operation performs matrix multiplication on two 4D packed tensors.
    The operation expects the two inputs to be packed with the same inner_tile on the K dimension.

    If `c` is provided, performs `result = a @ b + c` (accumulation).
    If `c` is not provided, performs `result = a @ b` (no accumulation).

    Example:
    ```mlir
    // With accumulator
    %result = xsmt.mmt4d %a, %b, %c :
        (tensor<4x8x8x16xf32>, tensor<8x8x8x16xf32>, tensor<4x8x8x8xf32>) -> tensor<4x8x8x8xf32>

    // Without accumulator
    %result += xsmt.mmt4d %a, %b :
        (tensor<4x8x8x16xf32>, tensor<8x8x8x16xf32>) -> tensor<4x8x8x8xf32>
    ```
  }];

  let arguments = (ins
    TT_Tensor:$a,
    TT_Tensor:$b,
    Optional<TT_Tensor>:$c
  );

  let results = (outs TT_Tensor:$result);

  let assemblyFormat = [{
    $a `,` $b (`,` $c^)? attr-dict `:` type($a) `,` type($b) (`,` type($c)^)? `->` type($result)
  }];
}

def XSMT_MBarrierCreateOp : XSMT_Op<"mbarrier_create"> {
  let summary = "Create a software barrier for thread synchronization.";

  let description = [{
    Creates a software barrier handle with specified configuration:
    - `flag`: Barrier mode (0=normal, 1=async, 2=with fence)
    - `atc`: Initial arrival thread count
    - `tc`: Total threads that must arrive before release
    - `exp`: Expected version value after barrier release

    Returns an opaque barrier handle for use with arrive/wait operations.
  }];

  let arguments = (ins
    I16:$flag,
    I16:$atc,
    I16:$tc,
    I16:$exp
  );

  let results = (outs TT_Ptr:$barrier);

  let assemblyFormat = [{
    $flag `,` $atc `,` $tc `,` $exp attr-dict `:` type($barrier)
  }];

}

def XSMT_BarrierArriveOp : XSMT_Op<"barrier_arrive"> {
  let summary = "Signal thread arrival at software barrier.";

  let description = [{
    Signals that the calling thread has arrived at the barrier.
    Decrements the pending arrival count. When count reaches zero,
    waiting threads are released.
  }];

  let arguments = (ins TT_Ptr:$barrier);

  let assemblyFormat = "$barrier attr-dict `:` type($barrier)";

}

def XSMT_BarrierWaitOp : XSMT_Op<"barrier_wait"> {
  let summary = "Wait for barrier to reach expected version.";

  let description = [{
    Blocks the calling thread until the barrier reaches the expected
    version value, indicating all required threads have arrived.

    - `flag`: Wait mode (0=normal, 1=async, 2=with fence)
    - `exp`: Expected version to wait for
  }];

  let arguments = (ins
    TT_Ptr:$barrier,
    I16:$flag,
    I16:$exp
  );

  let assemblyFormat = [{
    $barrier `,` $flag `,` $exp attr-dict `:` type($barrier)
  }];

}

def GetThreadOp : XSMT_Op<"get_num_of_thread", [Pure, MemoryEffects<[MemWrite]>]> {

  let summary = "Get the current thread";

  let description = [{
    Returns the thread ID of the current context.
    Example:
      %0 = xsmt.get_num_of_thread
  }];

  let arguments = (ins);

  let results = (outs I32:$result);

  let assemblyFormat = "attr-dict";
}


def XSMT_GlobalMBarrierInitOp : XSMT_Op<"global_mbarrier_init", [MemoryEffects<[MemWrite]>]> {
  let summary = "Create a global barrier.";

  let description = [{

  }];

  let arguments = (ins I16:$id);

  let results = (outs I64:$barrier);

  let assemblyFormat = [{
    $id attr-dict `:` type($barrier)
  }];

}

def XSMT_BarrierSetEepectOp : XSMT_Op<"barrier_set_expect"> {
  let summary = "Wait for barrier to reach expected version.";

  let description = [{
  }];

  let arguments = (ins
    I64:$barrier,
    I16:$exp
  );

  let assemblyFormat = [{
    $barrier `,` $exp attr-dict `:` type($barrier)
  }];

}

def XSMT_BufferTensorViewOp : XSMT_Op<"buffer_tensor_view", [Pure]> {
    let summary = "create a subview of a buffer tensor";
    let description = [{
        Creates a subview of a buffer tensor at the specified buffer index.
        The result type is a tensor pointer to the subview.
    }];

    let arguments = (ins
        XSMT_BufferType:$buffer,
        I32:$bufferIdx
    );

    let results = (outs
        TT_TensorPtr:$result
    );

    let assemblyFormat = [{
        $buffer `[` $bufferIdx `]` attr-dict `:` type($buffer) `->` type($result)
    }];

    let builders = [
        OpBuilder<(ins "Value":$buffer, "Value":$bufferIdx)>
    ];
}

def MBarrierCopiesOp : XSMT_Op<"mbarrier_copies", [MemoryEffects<[MemAlloc]>]> {
  let summary = "Allocate multiple mbarrier copies";
  let description = [{
    Allocates multiple mbarrier instances for use with async operations.

    Parameters:
    - numCopies: number of barrier copies (e.g., 2 for double buffering)
    - flag: barrier flag (typically 0 or 1)
    - arriveCount: initial arrive count
    - transactionCount: expected transaction bytes (for TMA)
    - expectCount: number of arrivals before barrier can proceed
  }];

  let arguments = (ins
    I64Attr:$numCopies,
    I64Attr:$flag,
    I64Attr:$arriveCount,
    I64Attr:$transactionCount,
    I64Attr:$expectCount
  );

  let results = (outs XSMT_MBarrierType:$result);

  let builders = [
    OpBuilder<(ins
      "mlir::Type":$resultType,
      "int":$numCopies,
      "int":$flag,
      "int":$arriveCount,
      "int":$transactionCount,
      "int":$expectCount
    ), [{
      build($_builder, $_state, resultType,
            $_builder.getI64IntegerAttr(numCopies),
            $_builder.getI64IntegerAttr(flag),
            $_builder.getI64IntegerAttr(arriveCount),
            $_builder.getI64IntegerAttr(transactionCount),
            $_builder.getI64IntegerAttr(expectCount));
    }]>
  ];

  let assemblyFormat = [{
    `copies` `(` $numCopies `)`
    `flag` `(` $flag `)`
    `arrive` `(` $arriveCount `)`
    `txn` `(` $transactionCount `)`
    `expect` `(` $expectCount `)`
    attr-dict `:` type($result)
  }];
}

def MBarrierSubviewOp : XSMT_Op<"mbarrier_subview", [Pure]> {
  let summary = "Get barrier slot from mbarrier at given index";

  let arguments = (ins
    XSMT_MBarrierType:$mbarrier,
    AnyTypeOf<[I32, I64, Index]>:$index  // 支持动态索引
  );

  let results = (outs I64:$result);

  let assemblyFormat = [{
    $mbarrier `[` $index `]` attr-dict `:` `(` type($mbarrier) `,` type($index) `)` `->` type($result)
  }];
}

#endif
