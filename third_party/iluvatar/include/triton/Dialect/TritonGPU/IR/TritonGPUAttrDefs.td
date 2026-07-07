#ifndef TRITONGPU_ATTRDEFS
#define TRITONGPU_ATTRDEFS

include "triton/Dialect/TritonGPU/IR/TritonGPUAttrBase.td"

//===----------------------------------------------------------------------===//
// Traits, Interfaces and shared Parameters
//===----------------------------------------------------------------------===//

def LayoutEncodingTrait : AttrInterface<"LayoutEncodingTrait"> {
  let cppNamespace = "::mlir::triton::gpu";
  let description = [{
    Common trait for all TTGIR layouts.
  }];
  let methods = [
    InterfaceMethod<"Get the CTA layout backing this encoding.",
                    "CTAEncodingAttr", "getCTALayout">,
    InterfaceMethod<"Get the rank of the layout.", "unsigned", "getRank",
                    (ins), [{}], [{
      return $_attr.getCTALayout().getRank();
    }]>
  ];
}
def DeclareLayoutEncodingMethods : DeclareAttrInterfaceMethods<
  LayoutEncodingTrait, ["getCTALayout"]>;

def SharedEncodingTrait : AttrInterface<"SharedEncodingTrait"> {
  let cppNamespace = "::mlir::triton::gpu";

  let description = [{
    Common trait describing shared memory.
  }];
  let methods = [
    InterfaceMethod<"Return the default alignment for the layout.",
                    "int32_t", "getAlignment", (ins), [{}], [{ return 16; }]>,
  ];
}
def DeclareSharedEncodingMethods : DeclareAttrInterfaceMethods<
  SharedEncodingTrait, ["getAlignment"]>;

//===----------------------------------------------------------------------===//
// Shared Layout Encoding
//===----------------------------------------------------------------------===//

def SwizzledSharedEncodingAttr
    : TritonGPU_Attr<"SwizzledSharedEncoding", "swizzled_shared_encoding",
                     [SharedEncodingTrait, LayoutEncodingTrait,
                      DeclareLayoutEncodingMethods]> {
  let mnemonic = "swizzled_shared";

  let description = [{
An encoding for tensors whose elements may be simultaneously accessed by
different GPU threads in the programs, via shared memory. In other words,
for all indices i \in Z^d, \mathcal{L}(i) = {0, 1, ..., 32*num_warps - 1}.

In order to avoid shared memory bank conflicts, elements may be swizzled.
Here are some examples.  In all cases, the input tensor is [0, 1, ..., n-1].

1. Basic swizzling

  #ttg.swizzled_shared<{vec=1, perPhase=1, maxPhase=4, order=[1,0]}>
  [ 0,  1,  2,  3],  // xor with 0
  [ 5,  4,  7,  6],  // xor with 1
  [10, 11,  8,  9],  // xor with 2
  [15, 14, 13, 12]   // xor with 3

Here elements of row r are xor'ed with r (or more properly, in[r][c] ->
out[r][c^r]).

2. Multiple rows per phase

  #ttg.swizzled_shared<{vec=1, perPhase=2, maxPhase=4, order=[1,0]}>
  [ 0,  1,  2,  3],  // phase 0 (xor with 0)
  [ 4,  5,  6,  7],
  [ 9,  8, 11, 10],  // phase 1 (xor with 1)
  [13, 12, 15, 14]

Elements of row r are xor'ed with r/2.  In other words, perPhase=2
means that pairs of 2 rows get the same swizzling.

3. Max-phase applied

  #ttg.swizzled_shared<{vec=1, perPhase=1, maxPhase=2, order=[1,0]}>
  [ 0,  1,  2,  3],  // phase 0 (xor with 0)
  [ 5,  4,  7,  6],  // phase 1 (xor with 1)
  [ 8,  9, 10, 11],  // phase 0
  [13, 12, 15, 14],  // phase 1
  [16, 17, 18, 19],  // ...
  [21, 20, 23, 22],
  [24, 25, 26, 27],
  [29, 28, 31, 30]

Elements of row r are xor'ed with (r/2) % 2.  In other words, maxPhase=m has the
effect of limiting the maximum value of the xor to m-1.

4. Max-phase and per-phase

  #ttg.swizzled_shared<{vec=1, perPhase=2, maxPhase=2, order=[1,0]}>
  [ 0,  1,  2,  3],  // phase 0 (xor with 0)
  [ 4,  5,  6,  7],  // phase 0
  [ 9,  8, 11, 10],  // phase 1 (xor with 1)
  [13, 12, 15, 14],  // phase 1
  [16, 17, 18, 19],  // phase 0
  [20, 21, 22, 23],  // phase 0
  [25, 24, 27, 26],  // phase 1
  [29, 28, 31, 30]]  // phase 1

Here the xor value (the "phase", I guess?) changes every perPhase rows, up to a
maximum value of maxPhase-1.  In other words, elements of row r are xor'ed with
(r/2) % 2.

5. Adding vec

  #ttg.swizzled_shared<{vec=2, perPhase=1, maxPhase=4, order=[1,0]}>
  [ 0,  1,  2,  3,  4,  5,  6,  7],
  [10, 11,  8,  9, 14, 15, 12, 13],
  [20, 21, 22, 23, 16, 17, 18, 19],
  [30, 31, 28, 29, 26, 27, 24, 25]

When vec=2, elements are swizzled in pairs of 2.  In other words, the element at
(r,c) has value

  ((c / 2) ^ r) * 2 + (c % 2).
  }];

  // swizzle info: vec, perPhase, maxPhase
  // order: the fastest-changing axis first
  let parameters = (
    ins
    "unsigned":$vec,
    "unsigned":$perPhase,
    "unsigned":$maxPhase,
    ArrayRefParameter<"unsigned">:$order,
    "CTAEncodingAttr":$CTALayout,
    "bool":$useTcu
  );

  let builders = [
    AttrBuilder<(ins "unsigned":$vec,
                     "unsigned":$perPhase,
                     "unsigned":$maxPhase,
                     "ArrayRef<unsigned>":$order,
                     "CTAEncodingAttr":$CTALayout), [{
        return $_get(context, vec, perPhase, maxPhase, order, CTALayout,
                     /*useTcu=*/false);
    }]>,

    AttrBuilder<(ins "DotOperandEncodingAttr":$dotOpEnc,
                     "ArrayRef<int64_t>":$shape,
                     "ArrayRef<unsigned>":$order,
                     "CTAEncodingAttr":$CTALayout,
                     "unsigned":$typeWidthInBit), [{
        bool needTrans = false; // default value
        return get(context, dotOpEnc, shape, order, CTALayout, typeWidthInBit, needTrans);
    }]>,

    // TODO(jlebar): This should not be an overload of
    // SwizzledSharedEncodingAttr::get().  It's misleading, because it does a bunch of
    // nontrivial work based on the given dotOpEnc.
    AttrBuilder<(ins "DotOperandEncodingAttr":$dotOpEnc,
                     "ArrayRef<int64_t>":$shape,
                     "ArrayRef<unsigned>":$order,
                     "CTAEncodingAttr":$CTALayout,
                     "unsigned":$typeWidthInBit,
                     "bool":$needTrans), [{

#ifdef __ILUVATAR__
        if (auto iluMmaEnc = mlir::dyn_cast<IluvatarMmaEncodingAttr>(dotOpEnc.getParent())) {
          return iluMmaEnc.composeSharedLayoutForOperand(
              CTALayout, dotOpEnc.getOpIdx(), shape, order, dotOpEnc.getKWidth(),
              typeWidthInBit, needTrans, dotOpEnc.getUseSme());
        }
#endif

        auto mmaEnc = mlir::dyn_cast<NvidiaMmaEncodingAttr>(dotOpEnc.getParent());

        if(!mmaEnc)
          return get(context, 1, 1, 1, order, CTALayout);

        // ---- begin Ampere & Hopper ----
        if (mmaEnc.isAmpere() || mmaEnc.isHopper()) {
          return get(context, dotOpEnc.getOpIdx(), dotOpEnc.getKWidth(), shape, order, CTALayout, typeWidthInBit, needTrans);
        }

        // ---- not implemented ----
        llvm_unreachable("unsupported swizzling for provided MMA version");
    }]>,

    // NVIDIA constructor!
    // TODO(lezcano): We should totally get rid of all these constructors...
    AttrBuilder<(ins "int":$opIdx,
                     "unsigned":$kWidth,
                     "ArrayRef<int64_t>":$shape,
                     "ArrayRef<unsigned>":$order,
                     "CTAEncodingAttr":$CTALayout,
                     "unsigned":$bitwidth,
                     "bool":$needTrans), [{
        int K =  getShapePerCTA(CTALayout.getCTASplitNum(), shape)[order[0]];
        // Elems necessary to cover all the banks divided by the inner dimension
        // This packs a few rows together for small K
        int perPhase = std::max<int>(1024 / (bitwidth * K), 1);

        int mmaStride = 8;
        int vec = 4 * kWidth;
        // needsTrans is equiv. to flipping the opIdx
        if (needTrans)
          std::swap(vec, mmaStride);
        assert(opIdx == 0 || opIdx == 1);
        int rank = order.size();
        int kDim = opIdx == 0 ? rank-1 : rank-2;
        if (order[0] != kDim)
          std::swap(vec, mmaStride);
        // Count how many vec elements are needed to cover all the banks
        int maxPhase = std::max(std::min<int>(mmaStride, 1024 / (vec * bitwidth)), 1);
        // Account for the row packing from perPhase: mmaStride / perPhase
        maxPhase = std::max(maxPhase / perPhase, 1);
        return get(context, vec, perPhase, maxPhase, order, CTALayout);
    }]>,

    AttrBuilder<(ins "DotOperandEncodingAttr":$dotOpEnc,
                     "ArrayRef<int64_t>":$shape,
                     "ArrayRef<unsigned>":$order,
                     "CTAEncodingAttr":$CTALayout,
                     "Type":$eltTy), [{
      unsigned bitwidth = eltTy.getIntOrFloatBitWidth();
      return get(context, dotOpEnc, shape, order, CTALayout, bitwidth);
    }]>,

    AttrBuilder<(ins "DotOperandEncodingAttr":$dotOpEnc,
                     "ArrayRef<int64_t>":$shape,
                     "ArrayRef<unsigned>":$order,
                     "CTAEncodingAttr":$CTALayout,
                     "Type":$eltTy,
                     "bool":$needTrans), [{
      unsigned bitwidth = eltTy.getIntOrFloatBitWidth();
      return get(context, dotOpEnc, shape, order, CTALayout, bitwidth, needTrans);
    }]>,
  ];

  let hasCustomAssemblyFormat = 1;
  let genVerifyDecl = 1;
}

def PaddedSharedEncodingAttr
    : TritonGPU_Attr<"PaddedSharedEncoding", "padded_shared_encoding",
                     [SharedEncodingTrait, DeclareLayoutEncodingMethods]> {
  let mnemonic = "padded_shared";

  let description = [{
An encoding for tensors whose elements may be simultaneously accessed by
different GPU threads in the programs, via shared memory. In other words,
for all indices i \in Z^d, \mathcal{L}(i) = {0, 1, ..., 32*num_warps - 1}.
Compared to SwizzledSharedEncodingAttr, this encoding combines padding with
element reordering via linear transformation (e.g. row permutation) to avoid
shared memory bank conflicts.

Formally, given a layout:
    padded_shared<[<interval_0>:+<pad_0>, <interval_1>:+<pad_1>, ...]>
We insert a padding of `<pad_i>` elements after every `<interval_i>` elements.
Multi interval-padding pairs are supported for flexibility of multi tiered
padding schemes; they compose in an additive manner. So for a 1-D tensor element
at index i, the corresponding shared memory location index is
    i + \sum_{k} (i / interval_k) * pad_k = 1
`<interval_i>` and `<pad_i>` all need to be power of two.

Some concrete examples ignoring the linear component, using `eM` to mean tensor
elements and `pN` to mean padding:

1. Single interval-padding pair:

   #ttg.padded_shared<[2:+2], {...}>
   [e0, e1, p0, p1,
    e2, e3, p2, p3,
    ...]

2. Double interval-padding pairs:

   #ttg.padded_shared<[2:+1, 4:+2], {...}>
   [e0, e1, p0,
    e2, e3, p1, p2, p3,
    e4, e5, p4,
    e6, e7, p5, p6, p7,
    ...]

Furthermore this encoding allows for a linear remapping from the 1-D shared
memory offset to logical n-D tensor elements. The remapping is given in the form
of linear bases mapping from offset to [dim0, dim1...dimN-1].
See LinearLayout.h for more details how linear layouts are applied to remap
elements.
Some concrete examples using `xN` and `yN` to mean the logical n-D tensor elements
and `pN` to mean padding:

1. 1D Single interval-padding with strided elements

    #ttg.padded_shared<[2:+2] {offset = [[2], [1]], block = []}>
    [x0, x2, p0 p1,
     x1, x3, p2, p3
     ...]

2. 2D single interval-padding with rearranged rows.

    #ttg.padded_shared<[16:+1] {offset = [[0, 1], [0, 2], /*gap, stride by 2 rows*/[2, 0], [4, 0], [1, 0]]], block = []}>
    [
      x0y0, x0y1, x0y2, x0y3,
      x2y0, x2y1, x2y2, x2y3,
      x4y0, x4y1, x4y2, x4y3,
      x6y0, x6y1, x6y2, x6y3,
      p0,
      x1y0, x1y1, x1y2, x1y3,
      x3y0, x3y1, x3y2, x3y3,
      x5y0, x5y1, x5y2, x5y3,
      x7y0, x7y1, x7y2, x7y3,
      p1,
    ]

For identity mappings a short form based on order and shape is used to increase readability. The following two encodings are the same:

    #ttg.padded_shared<[2:+2] {order = [1, 0], shape = [16, 32]}>
    #ttg.padded_shared<[2:+2] {offset = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [1, 0], [2, 0], [4, 0], [8, 0]], block = []}>


  }];

  let parameters = (ins
      ArrayRefParameter<"unsigned">:$intervals,
      ArrayRefParameter<"unsigned">:$paddings,
      LinearLayoutParam:$linearComponent
  );

  let builders = [
      AttrBuilder<(ins "ArrayRef<std::pair<unsigned, unsigned>>":$intervalPads,
                       "LinearLayout":$linearComponent)>,

      // Builder to create an identity mapping as the linear component
      AttrBuilder<(ins "ArrayRef<std::pair<unsigned, unsigned>>":$intervalPads,
                       "ArrayRef<unsigned>":$order, "ArrayRef<int64_t>":$shape,
                       "CTAEncodingAttr":$ctaLayout)>,
  ];

  let extraClassDeclaration = extraBaseClassDeclaration # [{
    // Returns the order of the dimensions `dimName` of the layout.
    // If more than dimension is of size one, it uses defaultOrder to determine
    // the order of the dimensions of size one.
    SmallVector<unsigned> orderPerDim(StringAttr dimName,
                                      ArrayRef<unsigned> defaultOrder) const;
    SmallVector<unsigned> getOrder() const;

    // Returns the bases of the dimensions `dimName` of the linear_component.
    // If skipBroadcast is false, we count a base zero
    SmallVector<unsigned> basesPerDim(StringAttr dimName,
                                      bool skipBroadcast = true) const;

    unsigned getMinInterval() const {
      return *llvm::min_element(getIntervals());
    }

    // Returns the total number of elements including padding given the input
    // tensor shape.
    int64_t getPaddedSize(ArrayRef<int64_t> shape) const;
  }];
  let hasCustomAssemblyFormat = 1;
  let genVerifyDecl = 1;
}

def SharedLinearEncodingAttr
    : TritonGPU_Attr<"SharedLinearEncoding", "shared_linear_encoding",
                     [SharedEncodingTrait, LayoutEncodingTrait,
                      DeclareLayoutEncodingMethods]> {
  let mnemonic = "shared_linear";

  let description = [{
    Linear shared encodings mirror LinearEncodingAttr but operate on shared
    memory layouts. The LinearLayout parameter captures how shared memory
    offsets (and optionally blocks) map to logical tensor indices.
  }];

  let parameters = (ins LinearLayoutParam:$linearLayout, "unsigned":$layoutAlignment);

  let extraClassDeclaration = [{
    SmallVector<unsigned> basesPerDim(StringAttr dimName,
                                      bool skipBroadcast = true) const;
    SmallVector<unsigned> orderPerDim(StringAttr dimName,
                                      ArrayRef<unsigned> defaultOrder) const;

    SmallVector<unsigned> getOrder() const;

    LinearLayout toLinearLayout(ArrayRef<int64_t> shape) const;

    int32_t getAlignment() const { return static_cast<int32_t>(getLayoutAlignment()); }
  }];

  let genVerifyDecl = 1;
  let hasCustomAssemblyFormat = 1;
}

def NVMMASharedEncodingAttr : TritonGPU_Attr<"NVMMASharedEncoding", "nvmma_shared_encoding",
                     [DeclareSharedEncodingMethods, LayoutEncodingTrait,
                      DeclareLayoutEncodingMethods]> {
  let mnemonic = "nvmma_shared";

  let description = [{
    Represent blocked shared memory matching MMAv3/MMAv5 shared memory input.
    This is meant to represent 2d tiled blocked layout.
    The full layout representation is described here:
    https://docs.nvidia.com/cuda/parallel-thread-execution/#asynchronous-warpgroup-level-matrix-shared-memory-layout
    When the memdesc has more than 2 dimensions the tiling is applied to 8 rows even if the first outer dimension is smaller than 8.
    In this case `transposed` means that the contiguous dimension is the most outer dimension of the memdesc.
  }];


  // fp4Padded: Indicates that this encoding represents a mixed-precision fp4 operand in MMAv5 scaled dot, which needs
  // to be in the special padded layout as described in https://docs.nvidia.com/cuda/parallel-thread-execution/#packing-format-used-for-matrix-a-and-b-by-kind-mxf8f6f4-in-shared-memory
  let parameters = (
    ins
    "unsigned":$swizzlingByteWidth,
    "bool":$transposed,
    "unsigned":$elementBitWidth,
    "bool":$fp4Padded,
    "CTAEncodingAttr":$CTALayout
  );

  let builders = [
    AttrBuilder<(ins "ArrayRef<int64_t>":$shape,
                     "ArrayRef<unsigned>":$order,
                     "CTAEncodingAttr":$CTALayout,
                     "Type":$eltTy,
                     "bool": $fp4Padded), [{
        auto shapePerCTA = getShapePerCTA(CTALayout.getCTASplitNum(), shape);
        int32_t swizzlingByteWidth = 0;
        unsigned eleBitWidth = eltTy.getIntOrFloatBitWidth();
        int packingFactor = fp4Padded ? 2 : 1;

        // get proper shared memory swizzling mode from the contiguous dimension
        // size of the origin blocked layout.
        auto contigDimSizeInByte = shapePerCTA[order[0]] * packingFactor * eleBitWidth / 8;
        if (contigDimSizeInByte >= 128 && contigDimSizeInByte % 128 == 0) {
          swizzlingByteWidth = 128;
        } else if (contigDimSizeInByte >= 64 && contigDimSizeInByte % 64 == 0) {
          swizzlingByteWidth = 64;
        } else if (contigDimSizeInByte >= 32 && contigDimSizeInByte % 32 == 0) {
          swizzlingByteWidth = 32;
        } else {
          swizzlingByteWidth = 0;
        }
        int flattenOutterDim = 1;
        for (int i = 1; i < shapePerCTA.size(); i++) {
          flattenOutterDim *= shapePerCTA[order[i]];
        }
        if (shapePerCTA.size() < 2 || flattenOutterDim < 8) {
          swizzlingByteWidth = 0;
        }
        bool transposed = order[0] == 0;
        return $_get(context, swizzlingByteWidth, transposed, eleBitWidth, fp4Padded, CTALayout);
    }]>
  ];

  let extraClassDeclaration = extraBaseClassDeclaration # [{
    int getPerPhase() const;
    int getMaxPhase() const;
    int getVec() const;
  }];
  let hasCustomAssemblyFormat = 1;
}

//===----------------------------------------------------------------------===//
// Distributed Layout Encoding
//===----------------------------------------------------------------------===//

def DistributedEncodingTrait : AttrInterface<"DistributedEncodingTrait"> {
  let cppNamespace = "::mlir::triton::gpu";

  let description = [{
The Distributed encoding describes the layout L with the 4-level compute hierarchy on GPU.
It is abstracted from the top to the bottom as CTAs Per CGA->Warps Per CTA->Threads Per Warp->Values Per Thread.

For CTAs Per CGA and Warps Per CTA level, the linear id is distributed contiguously with the shape and order.
For example, for a shape/order pair defines a distribution layout
shape = [4, 4]
order = [0, 1] // The fastest-changing axis first
->
layout = [0  4  8  12]
         [1  5  9  13]
         [2  6  10 14]
         [3  7  11 15]

For the Threads Per Warp and Values Per Thread level, the linear id distribution is variant for each sub-class encoding.

If the layout does not completely cover the tensor, we tile it until we cover the entire tensor.
We call each individual tile "rep".
  }];

  let methods = [
    InterfaceMethod<"Get the order of reps (tiles of this layout that tile the whole tensor). The fastest-changing axis first",
                    "SmallVector<unsigned>",
                    "getRepOrder">,
    InterfaceMethod<"Return total element size per thread.",
                    "unsigned",
                    "getTotalElemsPerThread",
                     (ins "ArrayRef<int64_t>":$shape),
                     /*defaultImplementation=*/[{
                         return toLinearEncoding($_self, shape).getTotalElemsPerThread(shape);
                     }]>,
    InterfaceMethod<"Return element size per thread in each dimension.",
                    "SmallVector<unsigned>",
                    "getElemsPerThread",
                     (ins "ArrayRef<int64_t>":$shape),
                     /*defaultImplementation=*/[{
                         return toLinearEncoding($_self, shape).getElemsPerThread(shape);
                     }]>,
    InterfaceMethod<"Convert to LinearLayout.",
                    "LinearLayout",
                    "toLinearLayout",
                    (ins "ArrayRef<int64_t>":$shape)>,
  ];
}

class DistributedEncoding<string name, string attrMnemonic, list<Trait> traits = []>
  : TritonGPU_Attr<name, attrMnemonic,
                   !listconcat([DistributedEncodingTrait, LayoutEncodingTrait,
                                DeclareLayoutEncodingMethods],
                               traits)> {

  let description = [{
Distributed encodings have a layout function L that is entirely characterized
by a d-dimensional tensor T. Note that L doesn't need to have the same shape
(or even the same rank) as the tensor it is encoding.

The layout function \mathcal{L} of this layout is then defined, for an
index `i` \in Z^d, as follows:

\mathcal{L}(T)[i_d] = L[(i_d + k_d*T.shape[d]) % L.shape[d]] \forall k_d such as i_d + k_d*T.shape[d] < L.shape[d]

Intuitively, when the tensor dim size T.shape[d] is larger than the layout
dim size L.shape[d], on that particular dim, we distribute values from the
tensor to threads mapped in the layout in a "wrapped around" manner, with
each thread owning multiple values.

OTOH, when the tensor dim size T.shape[d] is smaller than the layout
dim size L.shape[d], on that particular dim, we distribute values from the
tensor to threads mapped in the layout in a "broadcasted" manner, with
each value owned by multiple threads.

For example, for a tensor/layout pair
T = [x  x  x  x  x  x  x  x]
    [x  x  x  x  x  x  x  x]
L = [0  1  2  3 ]
    [4  5  6  7 ]
    [8  9  10 11]
    [12 13 14 15]

Then the data of T would be distributed as follow between the 16 CUDA threads:
L(T) = [ {0,8} , {1,9} , {2,10}, {3,11}, {0,8} , {1, 9} , {2, 10}, {3, 11},
         {4,12}, {5,13}, {6,14}, {7,15}, {4,12}, {5, 13}, {6, 14}, {7, 15} ]
  }];

  code extraDistributedDeclaration  = extraBaseClassDeclaration # [{
    // Implemented in subclasses
    SmallVector<unsigned> getRepOrder() const;

    LinearLayout toLinearLayout(ArrayRef<int64_t> shape) const;
  }];
}

//===----------------------------------------------------------------------===//
// Linear Layout Encoding
//===----------------------------------------------------------------------===//

def LinearEncodingAttr : DistributedEncoding<"LinearEncoding", "linear_encoding"> {
  let mnemonic = "linear";

  let description = [{
    See the docs in LinearLayout.h for the definition of linear layouts.
  }];

  let parameters = (ins LinearLayoutParam:$linearLayout);

  let extraClassDeclaration = extraDistributedDeclaration # [{
    // Generic distributed encoding methods
    unsigned getTotalElemsPerThread(ArrayRef<int64_t> shape) const;
    SmallVector<unsigned> getElemsPerThread(ArrayRef<int64_t> shape) const;

    SmallVector<unsigned int> getContig(const char *, SmallVector<unsigned int>) const;
    SmallVector<unsigned> getContigPerThread() const;
    SmallVector<unsigned> getContigPerWarp() const;
    SmallVector<unsigned> getOrder() const;
    SmallVector<unsigned> getWarpOrder() const;
    SmallVector<unsigned> getThreadOrder() const;


    // Generalizes get{Warp,Thread,CTA}Order to linear layouts.
    // Returns the order of the dimensions `dimName` of the layout.
    // If more than dimension is of size one, it uses defaultOrder to determine
    // the order of the dimensions of size one.
    SmallVector<unsigned> orderPerDim(StringAttr dimName,
                                      ArrayRef<unsigned> defaultOrder) const;

    // Generalizes getThreadsPerWarp, getWarpsPerCTA, getCTAsPerCGA to linear layouts.
    // Returns the bases of the dimensions `dimName` of the layout.
    // If skipBroadcast is false, we count a base zero
    SmallVector<unsigned> basesPerDim(StringAttr dimName,
                                      bool skipBroadcast = true) const;
    SmallVector<unsigned> getThreadsPerWarp() const;
    SmallVector<unsigned> getWarpsPerCTA() const;

    // [FIXME LL] Supports legacy behaviour. We should remove these functions
    SmallVector<unsigned> getSizePerThread() const;
  }];

  let genVerifyDecl = 1;
  // Example of assembly format:
  // <{register = [[0, 1], [8, 0], [0, 8], [64, 0]],
  //   lane = [[0, 2], [0, 4], [1, 0], [2, 0], [4, 0]],
  //   warp = [[16, 0], [32, 0]],
  //   block = []}>
  let hasCustomAssemblyFormat = 1;
}


//===----------------------------------------------------------------------===//
// Blocked Layout Encoding
//===----------------------------------------------------------------------===//

def BlockedEncodingAttr : DistributedEncoding<"BlockedEncoding", "blocked_encoding"> {
  let mnemonic = "blocked";

  let description = [{
An encoding where each warp owns a contiguous portion of the target tensor. This is typically the kind of data layout
used to promote memory coalescing in LoadInst and StoreInst.
It is characterized by three tuples -- thread tile size, warp tile size, and block tile size -- which
specify the amount of elements owned by each CUDA thread, warp and CTA respectively.

Example 1, a row-major coalesced layout may partition a 16x16 tensor over 2 warps (i.e. 64 threads) as follows:

[ 0  0  1  1  2  2  3  3  ; 32 32 33 33 34 34 35 35 ]
[ 0  0  1  1  2  2  3  3  ; 32 32 33 33 34 34 35 35 ]
[ 4  4  5  5  6  6  7  7  ; 36 36 37 37 38 38 39 39 ]
[ 4  4  5  5  6  6  7  7  ; 36 36 37 37 38 38 39 39 ]
...
[ 28 28 29 29 30 30 31 31 ; 60 60 61 61 62 62 63 63 ]
[ 28 28 29 29 30 30 31 31 ; 60 60 61 61 62 62 63 63 ]

for

#ttg.blocked_layout<{
  sizePerThread = {2, 2}
  threadsPerWarp = {8, 4}
  blocked = {{0, 1}}
}>

Example 2, a row-major coalesced layout may partition a 32x32 tensor over 2 warps (i.e. 64 threads) as follows:

[ 0  0  1  1  2  2  3  3  ; 32 32 33 33 34 34 35 35  0  0  1  1  2  2  3  3  ; 32 32 33 33 34 34 35 35 ]
[ 0  0  1  1  2  2  3  3  ; 32 32 33 33 34 34 35 35  0  0  1  1  2  2  3  3  ; 32 32 33 33 34 34 35 35 ]
[ 4  4  5  5  6  6  7  7  ; 36 36 37 37 38 38 39 39  4  4  5  5  6  6  7  7  ; 36 36 37 37 38 38 39 39 ]
[ 4  4  5  5  6  6  7  7  ; 36 36 37 37 38 38 39 39  4  4  5  5  6  6  7  7  ; 36 36 37 37 38 38 39 39 ]
...                                                 ...
[ 28 28 29 29 30 30 31 31 ; 60 60 61 61 62 62 63 63  28 28 29 29 30 30 31 31 ; 60 60 61 61 62 62 63 63 ]
[ 28 28 29 29 30 30 31 31 ; 60 60 61 61 62 62 63 63  28 28 29 29 30 30 31 31 ; 60 60 61 61 62 62 63 63 ]
[ 0  0  1  1  2  2  3  3  ; 32 32 33 33 34 34 35 35  0  0  1  1  2  2  3  3  ; 32 32 33 33 34 34 35 35 ]
[ 0  0  1  1  2  2  3  3  ; 32 32 33 33 34 34 35 35  0  0  1  1  2  2  3  3  ; 32 32 33 33 34 34 35 35 ]
[ 4  4  5  5  6  6  7  7  ; 36 36 37 37 38 38 39 39  4  4  5  5  6  6  7  7  ; 36 36 37 37 38 38 39 39 ]
[ 4  4  5  5  6  6  7  7  ; 36 36 37 37 38 38 39 39  4  4  5  5  6  6  7  7  ; 36 36 37 37 38 38 39 39 ]
...                                                 ...
[ 28 28 29 29 30 30 31 31 ; 60 60 61 61 62 62 63 63  28 28 29 29 30 30 31 31 ; 60 60 61 61 62 62 63 63 ]
[ 28 28 29 29 30 30 31 31 ; 60 60 61 61 62 62 63 63  28 28 29 29 30 30 31 31 ; 60 60 61 61 62 62 63 63 ]
for

#ttg.blocked_layout<{
  sizePerThread = {2, 2}
  threadsPerWarp = {8, 4}
  blocked = {{0, 1}}
}>

Example 3, A row-major coalesced layout may partition a 32x32 tensor over 2 warps (i.e. 64 threads) and
4 CTAs (taking 2x2 for example) as follows:

CTA [0,0]                                              CTA [0,1]
[ 0  0  1  1  2  2  3  3  ; 32 32 33 33 34 34 35 35 ]  [ 0  0  1  1  2  2  3  3  ; 32 32 33 33 34 34 35 35 ]
[ 0  0  1  1  2  2  3  3  ; 32 32 33 33 34 34 35 35 ]  [ 0  0  1  1  2  2  3  3  ; 32 32 33 33 34 34 35 35 ]
[ 4  4  5  5  6  6  7  7  ; 36 36 37 37 38 38 39 39 ]  [ 4  4  5  5  6  6  7  7  ; 36 36 37 37 38 38 39 39 ]
[ 4  4  5  5  6  6  7  7  ; 36 36 37 37 38 38 39 39 ]  [ 4  4  5  5  6  6  7  7  ; 36 36 37 37 38 38 39 39 ]
...                                                    ...
[ 28 28 29 29 30 30 31 31 ; 60 60 61 61 62 62 63 63 ]  [ 28 28 29 29 30 30 31 31 ; 60 60 61 61 62 62 63 63 ]
[ 28 28 29 29 30 30 31 31 ; 60 60 61 61 62 62 63 63 ]  [ 28 28 29 29 30 30 31 31 ; 60 60 61 61 62 62 63 63 ]

CTA [1,0]                                              CTA [1,1]
[ 0  0  1  1  2  2  3  3  ; 32 32 33 33 34 34 35 35 ]  [ 0  0  1  1  2  2  3  3  ; 32 32 33 33 34 34 35 35 ]
[ 0  0  1  1  2  2  3  3  ; 32 32 33 33 34 34 35 35 ]  [ 0  0  1  1  2  2  3  3  ; 32 32 33 33 34 34 35 35 ]
[ 4  4  5  5  6  6  7  7  ; 36 36 37 37 38 38 39 39 ]  [ 4  4  5  5  6  6  7  7  ; 36 36 37 37 38 38 39 39 ]
[ 4  4  5  5  6  6  7  7  ; 36 36 37 37 38 38 39 39 ]  [ 4  4  5  5  6  6  7  7  ; 36 36 37 37 38 38 39 39 ]
...                                                    ...
[ 28 28 29 29 30 30 31 31 ; 60 60 61 61 62 62 63 63 ]  [ 28 28 29 29 30 30 31 31 ; 60 60 61 61 62 62 63 63 ]
[ 28 28 29 29 30 30 31 31 ; 60 60 61 61 62 62 63 63 ]  [ 28 28 29 29 30 30 31 31 ; 60 60 61 61 62 62 63 63 ]
for

#ttg.blocked_layout<{
  sizePerThread = {2, 2}
  threadsPerWarp = {8, 4}
  blocked = {{0, 1}, {1, 0}}
}>
}];

  let parameters = (
    ins
    ArrayRefParameter<"unsigned">:$sizePerThread,
    ArrayRefParameter<"unsigned">:$threadsPerWarp,
    ArrayRefParameter<"unsigned">:$warpsPerCTA,
    ArrayRefParameter<"unsigned">:$order, // the fastest-changing axis first

    // CTALayout is optional in the textual IR.  If omitted, we infer it to be a
    // single CTA (i.e. the trivial map onto dim0..dimn-1)
    "CTAEncodingAttr":$CTALayout,
    "bool":$isSme,
    ArrayRefParameter<"unsigned">:$smeWarpsPerCTA
  );
  let genVerifyDecl = 1;

  let builders = [
    AttrBuilder<(ins "ArrayRef<int64_t>":$shape,
                     "ArrayRef<unsigned>":$sizePerThread,
                     "ArrayRef<unsigned>":$order,
                     "unsigned":$numWarps,
                     "unsigned":$numThreadsPerWarp,
                     "CTAEncodingAttr":$CTALayout), [{
      unsigned rank = sizePerThread.size();
      SmallVector<unsigned, 4> threadsPerWarp(rank);
      SmallVector<unsigned, 4> warpsPerCTA(rank);
      SmallVector<unsigned, 4> smeWpt(rank);
      SmallVector<int64_t> shapePerCTA = getShapePerCTA(CTALayout.getCTASplitNum(), shape);

      unsigned remainingLanes = numThreadsPerWarp;
      unsigned remainingThreads = numWarps * numThreadsPerWarp;
      unsigned remainingWarps = numWarps;
      unsigned prevLanes = 1;
      unsigned prevWarps = 1;

      // starting from the contiguous dimension
      for (unsigned d = 0; d < rank - 1; ++d) {
        unsigned i = order[d];
        unsigned threadsPerCTA = std::clamp<unsigned>(remainingThreads, 1, std::max<unsigned>(1, shapePerCTA[i] / sizePerThread[i]));
        threadsPerWarp[i] = std::clamp<unsigned>(threadsPerCTA, 1, remainingLanes);
        warpsPerCTA[i] = std::clamp<unsigned>(threadsPerCTA / threadsPerWarp[i], 1, remainingWarps);
        remainingWarps /= warpsPerCTA[i];
        remainingLanes /= threadsPerWarp[i];
        remainingThreads /= threadsPerCTA;
        prevLanes *= threadsPerWarp[i];
        prevWarps *= warpsPerCTA[i];
      }

      // Expand the last dimension to fill the remaining lanes and warps
      threadsPerWarp[order[rank - 1]] = numThreadsPerWarp / prevLanes;
      warpsPerCTA[order[rank - 1]] = numWarps / prevWarps;

      return $_get(context, sizePerThread, threadsPerWarp, warpsPerCTA, order, CTALayout, false, smeWpt);
    }]>,

    AttrBuilder<(ins "ArrayRef<int64_t>":$shape,
                     "ArrayRef<unsigned>":$sizePerThread,
                     "ArrayRef<unsigned>":$order,
                     "unsigned":$numWarps,
                     "unsigned":$numThreadsPerWarp,
                     "unsigned":$numCTAs), [{
      unsigned rank = sizePerThread.size();
      SmallVector<unsigned, 4> CTAsPerCGA(rank);
      SmallVector<unsigned, 4> CTASplitNum(rank);
      ArrayRef<unsigned> CTAOrder = order;

      unsigned remainingCTAs = numCTAs;

      // starting from the most strided dimension
      for (int d = rank - 1; d >= 0; --d) {
        unsigned i = order[d];
        CTAsPerCGA[i] = std::clamp<unsigned>(remainingCTAs, 1, std::max<unsigned>(1, shape[i] / sizePerThread[i]));
        CTASplitNum[i] = CTAsPerCGA[i];
        remainingCTAs /= CTAsPerCGA[i];
      }

      CTAsPerCGA[rank - 1] *= remainingCTAs; // wrap at CTA level

      CTAEncodingAttr CTALayout = CTAEncodingAttr::fromSplitParams(context, CTAsPerCGA, CTASplitNum, CTAOrder);
      return get(context, shape, sizePerThread, order, numWarps, numThreadsPerWarp, CTALayout);
    }]>,

    // Backward-compatible 5-param builder without isSme/smeWarpsPerCTA
    AttrBuilder<(ins "ArrayRef<unsigned>":$sizePerThread,
                     "ArrayRef<unsigned>":$threadsPerWarp,
                     "ArrayRef<unsigned>":$warpsPerCTA,
                     "ArrayRef<unsigned>":$order,
                     "CTAEncodingAttr":$CTALayout), [{
      return $_get(context, sizePerThread, threadsPerWarp, warpsPerCTA, order,
                   CTALayout, /*isSme=*/false,
                   /*smeWarpsPerCTA=*/ArrayRef<unsigned>());
    }]>,

    AttrBuilder<(ins "bool":$isSme,
                     "unsigned":$numWarps,
                     "Type":$eltTy,
                     "ArrayRef<int64_t>":$shape,
                     "ArrayRef<unsigned>":$order,
                     "ArrayRef<unsigned>":$sizePerThread,
                     "ArrayRef<unsigned>":$threadsPerWarp,
                     "ArrayRef<unsigned>":$warpsPerCTA,
                     "unsigned":$numCTAs), [{
      assert(isSme && "only sme inc can use this interface");
      SmallVector<unsigned> wpt({1, 1});
      SmallVector<unsigned> wpt_nm1;

      // The SME hardware tile is 16 rows x 64 contiguous bytes. The number of
      // elements along the contiguous dimension is therefore 64B / elemBytes =
      // 512 / bitwidth (fp16/bf16 -> 32, fp32 -> 16, int8 -> 64). The strided
      // dimension is always 16 rows.
      unsigned bitwidth = eltTy.getIntOrFloatBitWidth();
      unsigned contigElems = 512 / bitwidth;
      SmallVector<int, 2> spw(2);
      if (order[0] == 1) {
        spw[0] = 16;
        spw[1] = contigElems;
      } else {
        spw[0] = contigElems;
        spw[1] = 16;
      }

      do {
        wpt_nm1 = wpt;
        if (wpt[0] * wpt[1] < numWarps && shape[0] >= spw[0])
          wpt[0] = std::clamp<int>(wpt[0] * 2, 1, shape[0] / spw[0]);
        if (wpt[0] * wpt[1] < numWarps && shape[1] >= spw[1])
          wpt[1] = std::clamp<int>(wpt[1] * 2, 1, shape[1] / spw[1]);
      } while (wpt_nm1 != wpt);

      unsigned rank = sizePerThread.size();
      SmallVector<unsigned, 4> CTAsPerCGA(rank);
      SmallVector<unsigned, 4> CTASplitNum(rank);
      ArrayRef<unsigned> CTAOrder = order;

      unsigned remainingCTAs = numCTAs;

      // starting from the most strided dimension
      for (int d = rank - 1; d >= 0; --d) {
        unsigned i = order[d];
        CTAsPerCGA[i] = std::clamp<unsigned>(remainingCTAs, 1, shape[i] / sizePerThread[i]);
        CTASplitNum[i] = CTAsPerCGA[i];
        remainingCTAs /= CTAsPerCGA[i];
      }

      CTAsPerCGA[rank - 1] *= remainingCTAs; // wrap at CTA level

      CTAEncodingAttr CTALayout = CTAEncodingAttr::fromSplitParams(context, CTAsPerCGA, CTASplitNum, CTAOrder);

      return $_get(context, sizePerThread, threadsPerWarp, warpsPerCTA, order, CTALayout, isSme, wpt);
    }]>
  ];

  let extraClassDeclaration = extraDistributedDeclaration;

  let hasCustomAssemblyFormat = 1;
}

//===----------------------------------------------------------------------===//
// MMA Layout Encoding
//===----------------------------------------------------------------------===//

def MmaEncodingTrait : AttrInterface<"MmaEncodingTrait"> {
  let cppNamespace = "::mlir::triton::gpu";
  let methods = [
    InterfaceMethod<"Get the order of reps (tiles of this layout that tile the whole tensor). The fastest-changing axis first",
                    "SmallVector<unsigned>",
                    "getRepOrderForOperand",
                    (ins "int":$opIdx)>,
  ];
}

def NvidiaMmaEncodingAttr : DistributedEncoding<"NvidiaMmaEncoding", "nvidia_mma_encoding", [MmaEncodingTrait]> {
  let mnemonic = "nvidia_mma";

  let description = [{
An encoding for tensors that have been produced by tensor cores.

It is characterized by two parameters:
- A 'versionMajor' which specifies the generation the tensor cores
  whose output is being partitioned:
  - 1 for first-gen tensor cores (Volta), and
  - 2 for second-gen tensor cores (Turing/Ampere).
- A 'versionMinor' which indicates the specific layout of a tensor core
  generation, e.g. for Volta, there might be multiple kinds of layouts
  annotated by 0,1,2 and so on.
- A `blockTileSize` to indicate how data should be partitioned between warps.

// -------------------------------- version = 1 --------------------------- //

For first-gen tensor cores, the implicit warpTileSize is [16, 16].
Note: the layout is different from the recommended in PTX ISA
https://docs.nvidia.com/cuda/parallel-thread-execution/index.html
(mma.884 section, FP32 accumulator).

For example, when versionMinor=1, the matrix L corresponding to
blockTileSize=[32,16] is:

                               warp 0
--------------------------------/\-------------------------------
[ 0   0   2   2   8   8   10  10   0   0   2   2   8   8   10  10 ]
[ 1   1   3   3   9   9   11  11   1   1   3   3   9   9   11  11 ]
[ 0   0   2   2   8   8   10  10   0   0   2   2   8   8   10  10 ]
[ 1   1   3   3   9   9   11  11   1   1   3   3   9   9   11  11 ]
[ 4   4   6   6   12  12  14  14   4   4   6   6   12  12  14  14 ]
[ 5   5   7   7   13  13  15  15   5   5   7   7   13  13  15  15 ]
[ 4   4   6   6   12  12  14  14   4   4   6   6   12  12  14  14 ]
[ 5   5   7   7   13  13  15  15   5   5   7   7   13  13  15  15 ]
[ 16  16  18  18  20  20  22  22   16  16  18  18  20  20  22  22 ]
[ 17  17  19  19  21  21  23  23   17  17  19  19  21  21  23  23 ]
[ 16  16  18  18  20  20  22  22   16  16  18  18  20  20  22  22 ]
[ 17  17  19  19  21  21  23  23   17  17  19  19  21  21  23  23 ]
[ 24  24  26  26  28  28  30  30   24  24  26  26  28  28  30  30 ]
[ 25  25  27  27  29  29  31  31   25  25  27  27  29  29  31  31 ]
[ 24  24  26  26  28  28  30  30   24  24  26  26  28  28  30  30 ]
[ 25  25  27  27  29  29  31  31   25  25  27  27  29  29  31  31 ]

                          warp 1 = warp0 + 32
--------------------------------/\-------------------------------
[ 32  32  34  34  40  40  42  42   32  32  34  34  40  40  42  42 ]
[ 33  33  35  35  41  41  43  43   33  33  35  35  41  41  43  43 ]
[ ............................................................... ]


// -------------------------------- version = 2 --------------------------- //

For second-gen tensor cores, the implicit warpTileSize is [16, 8].
Information about this layout can be found in the official PTX documentation
https://docs.nvidia.com/cuda/parallel-thread-execution/index.html
(mma.16816 section, FP32 accumulator).

For example, the matrix L corresponding to blockTileSize=[32,16] is:
                warp 0                          warp 2
-----------------/\-------------  ----------------/\-------------
[ 0   0   1   1   2   2   3   3   32  32  33  33  34  34  35  35
[ 4   4   5   5   6   6   7   7   36  36  37  37  38  38  39  39
[ ..............................  ..............................
[ 28  28  29  29  30  30  31  31  60  60  61  61  62  62  63  63
[ 0   0   1   1   2   2   3   3   32  32  33  33  34  34  35  35
[ 4   4   5   5   6   6   7   7   36  36  37  37  38  38  39  39
[ ..............................  ..............................
[ 28  28  29  29  30  30  31  31  60  60  61  61  62  62  63  63

              warp 1                           warp 3
----------------/\-------------   ----------------/\-------------
[ 64  64  65  65  66  66  67  67  96  96  97  97  98  98  99  99
[ 68  68  69  69  70  70  71  71  100 100 101 101 102 102 103 103
[ ..............................  ...............................
[ 92  92  93  93  94  94  95  95  124 124 125 125 126 126 127 127
[ 64  64  65  65  66  66  67  67  96  96  97  97  98  98  99  99
[ 68  68  69  69  70  70  71  71  100 100 101 101 102 102 103 103
[ ..............................  ...............................
[ 92  92  93  93  94  94  95  95  124 124 125 125 126 126 127 127

}];

  let parameters = (
    ins
    "unsigned":$versionMajor,
    "unsigned":$versionMinor,
    ArrayRefParameter<"unsigned">:$warpsPerCTA,
    "CTAEncodingAttr":$CTALayout,
    ArrayRefParameter<"unsigned">:$instrShape
  );


  let extraClassDeclaration = extraDistributedDeclaration # [{
    bool isVolta() const;
    bool isTuring() const;
    bool isAmpere() const;
    bool isHopper() const;

    SmallVector<int64_t> getRepForOperand(ArrayRef<int64_t> shape,
                                          int bitwidth, int kWidth,
                                          int opIdx) const;
    SmallVector<unsigned> getRepOrderForOperand(int opIdx) const;
  }];

  let hasCustomAssemblyFormat = 1;
}

def IluvatarMmaEncodingAttr : DistributedEncoding<"IluvatarMmaEncoding", "iluvatar_mma_encoding", [MmaEncodingTrait]> {
  let mnemonic = "iluvatar_mma";

  let description = [{
An encoding for tensors that have been produced by tensor cores.

It is characterized by the standard MMA encoding fields used by the common
TritonGPU layout pipeline:
- A 'versionMajor' which specifies the generation the tensor cores
  whose output is being partitioned:
  - 1 for first-gen tensor cores (BI/MR)
- A 'versionMinor' which indicates the specific layout of a tensor core
  generation
- A `warpsPerCTA` field to indicate how data should be partitioned between
  warps.
- An `instrShape` field to describe the Iluvatar TCU instruction shape.  The
  initial BI/MR path uses `{16, 16, 16}`.

============================ Iluvatar GPU ============================
For Iluvatar GPU, we also use versionMinor to distinguish MMA Layout,
specifically, when versionMinor =
0: Default MMA Layout after dotOp;
1: Starting from the default MMA Layout, but swap each warp's each
   consecutive 16x64B(2 TCU) elements to make it scanline-style.
2: Row-swizzled MMA Layout after dotOp when Operand A/B's row is
   swizzled.
3: Starting from row-swizzled MMA Layout, but swap each warp's each
   consecutive 16x64B(2 TCU) elements to make it scanline-style.

For BI and MR series GPU (versionMajor = 1), warpTileSize is [16, 16] and
the instruction shape is [16, 16, 16].

// --------------------------- versionMinor = 0 --------------------------- //
For example, the matrix L corresponding to blockTileSize=[16, 32]
with num_warps = 1 is:

       warp 0 rep0                    warp 0 rep1
-----------/\-----------       -----------/\-----------
0   1   2  ... 13  14  15      0   1   2  ... 13  14  15
16  17  18 ... 29  30  31      16  17  18 ... 29  30  31
32  33  34 ... 45  46  47      32  33  34 ... 45  46  47
48  49  50 ... 61  62  63      48  49  50 ... 61  62  63
0   1   2  ... 13  14  15      0   1   2  ... 13  14  15
16  17  18 ... 29  30  31      16  17  18 ... 29  30  31
32  33  34 ... 45  46  47      32  33  34 ... 45  46  47
48  49  50 ... 61  62  63      48  49  50 ... 61  62  63
0   1   2  ... 13  14  15      0   1   2  ... 13  14  15

16  17  18 ... 29  30  31      16  17  18 ... 29  30  31
32  33  34 ... 45  46  47      32  33  34 ... 45  46  47
48  49  50 ... 61  62  63      48  49  50 ... 61  62  63
0   1   2  ... 13  14  15      0   1   2  ... 13  14  15
16  17  18 ... 29  30  31      16  17  18 ... 29  30  31
32  33  34 ... 45  46  47      32  33  34 ... 45  46  47
48  49  50 ... 61  62  63      48  49  50 ... 61  62  63


// --------------------------- versionMinor = 1 --------------------------- //

For example, the matrix L corresponding to blockTileSize=[16, 32]
with num_warps = 1 is( when dtype is bf16/f16):

                     warp 0
-----------------------/\-----------------------
0   0   1   1   2   2  ... 13  13  14  14  15  15
16  16  17  17  18  18 ... 29  29  30  30  31  31
32  32  33  33  34  34 ... 45  45  46  46  47  47
48  48  49  49  50  50 ... 61  61  62  62  63  63
0   0   1   1   2   2  ... 13  13  14  14  15  15
16  16  17  17  18  18 ... 29  29  30  30  31  31
32  32  33  33  34  34 ... 45  45  46  46  47  47
48  48  49  49  50  50 ... 61  61  62  62  63  63
0   0   1   1   2   2  ... 13  13  14  14  15  15
16  16  17  17  18  18 ... 29  29  30  30  31  31
32  32  33  33  34  34 ... 45  45  46  46  47  47
48  48  49  49  50  50 ... 61  61  62  62  63  63
0   0   1   1   2   2  ... 13  13  14  14  15  15
16  16  17  17  18  18 ... 29  29  30  30  31  31
32  32  33  33  34  34 ... 45  45  46  46  47  47
48  48  49  49  50  50 ... 61  61  62  62  63  63


// --------------------------- versionMinor = 2 --------------------------- //

For example, the matrix L corresponding to blockTileSize=[16, 32]
with num_warps = 1 is( when dtype is bf16/f16):

       warp 0 rep0                    warp 0 rep1
-----------/\-----------       -----------/\-----------
0   1   2  ... 13  14  15      0   1   2  ... 13  14  15
32  33  34 ... 45  46  47      32  33  34 ... 45  46  47
0   1   2  ... 13  14  15      0   1   2  ... 13  14  15
32  33  34 ... 45  46  47      32  33  34 ... 45  46  47
16  17  18 ... 29  30  31      16  17  18 ... 29  30  31
48  49  50 ... 61  62  63      48  49  50 ... 61  62  63
16  17  18 ... 29  30  31      16  17  18 ... 29  30  31
48  49  50 ... 61  62  63      48  49  50 ... 61  62  63
0   1   2  ... 13  14  15      0   1   2  ... 13  14  15
32  33  34 ... 45  46  47      32  33  34 ... 45  46  47
0   1   2  ... 13  14  15      0   1   2  ... 13  14  15
32  33  34 ... 45  46  47      32  33  34 ... 45  46  47
16  17  18 ... 29  30  31      16  17  18 ... 29  30  31
48  49  50 ... 61  62  63      48  49  50 ... 61  62  63
16  17  18 ... 29  30  31      16  17  18 ... 29  30  31
48  49  50 ... 61  62  63      48  49  50 ... 61  62  63


// --------------------------- versionMinor = 3 --------------------------- //

For example, the matrix L corresponding to blockTileSize=[16, 32]
with num_warps = 1 is( when dtype is bf16/f16):

                     warp 0
-----------------------/\-----------------------
0   0   1   1   2   2  ... 13  13  14  14  15  15
32  32  33  33  34  34 ... 45  45  46  46  47  47
0   0   1   1   2   2  ... 13  13  14  14  15  15
32  32  33  33  34  34 ... 45  45  46  46  47  47
16  16  17  17  18  18 ... 29  29  30  30  31  31
48  48  49  49  50  50 ... 61  61  62  62  63  63
16  16  17  17  18  18 ... 29  29  30  30  31  31
48  48  49  49  50  50 ... 61  61  62  62  63  63
0   0   1   1   2   2  ... 13  13  14  14  15  15
32  32  33  33  34  34 ... 45  45  46  46  47  47
0   0   1   1   2   2  ... 13  13  14  14  15  15
32  32  33  33  34  34 ... 45  45  46  46  47  47
16  16  17  17  18  18 ... 29  29  30  30  31  31
48  48  49  49  50  50 ... 61  61  62  62  63  63
16  16  17  17  18  18 ... 29  29  30  30  31  31
48  48  49  49  50  50 ... 61  61  62  62  63  63

}];

  let parameters = (
    ins
    "unsigned":$versionMajor,
    "unsigned":$versionMinor,
    ArrayRefParameter<"unsigned">:$warpsPerCTA,
    "CTAEncodingAttr":$CTALayout,
    ArrayRefParameter<"unsigned">:$instrShape
  );


  let extraClassDeclaration = extraDistributedDeclaration # [{
    bool isVolta() const;

    SmallVector<int64_t> getRepForOperand(ArrayRef<int64_t> shape,
                                          int bitwidth, int kWidth,
                                          int opIdx) const;
    SmallVector<unsigned> getRepOrderForOperand(int opIdx) const;

    // Returns the Iluvatar shared layout for a dot operand. useTcu marks
    // shared memory used by the TCU path, separating it from ordinary shared
    // layouts that may also use vec/perPhase/maxPhase = 1/1/1.
    SwizzledSharedEncodingAttr composeSharedLayoutForOperand(
        CTAEncodingAttr ctaLayout, int operandIdx, ArrayRef<int64_t> operandShape,
        ArrayRef<unsigned> sharedOrder, unsigned kWidth, unsigned elemBitWidth,
        bool needTrans, unsigned useSme) const;
  }];

  let hasCustomAssemblyFormat = 1;
}

def SliceEncodingAttr : DistributedEncoding<"SliceEncoding", "slice_encoding"> {
  let mnemonic = "slice";

  let description = [{
    Given a `parent` layout and a `dim`, squeezes the given `dim` in the `parent`
    layout and distributes values in a tensor T according to the new layout.

    For example, given

    T = [x  x  x  x  x  x  x  x]
    L_parent = [0  1  2  3 ]
               [4  5  6  7 ]
               [8  9  10 11]
               [12 13 14 15] (with 16 CUDA threads)

    With dim = 0, squeezing out dim 0, we have
    L = [{0,4,8,12},  {1,5,9,13}, {2,6,10,14},  {3,7,11,15} ]

    Then the data of T would be distributed as follow between the 16 CUDA threads:
    L(T) = [ {0,4,8,12} , {1,5,9,13} , ... {3,7,11,15}, {0,4,8,12} , ..., {3,7,11,15} ]

    With dim = 1, squeezing out dim 1, we have
    L = [ {0,1,2,3}, {4,5,6,7}, {8,9,10,11}, {12,13,14,15} ]

    Then the data of T would be distributed as follow between the 16 CUDA threads:
    L = [ {0,1,2,3}, {4,5,6,7}, ..., {12,13,14,15}, {0,1,2,3}, ..., {12,13,14,15} ]

    This is useful for constructing the inverse layout of an expand_dims operation
    during some optimization passes.
  }];

  let parameters = (
    ins
    "unsigned":$dim,
    "DistributedEncodingTrait":$parent
  );

  let extraClassDeclaration = extraDistributedDeclaration # [{
    template<class T>
    SmallVector<T> paddedShape(ArrayRef<T> shape) const;
  }];

  let hasCustomAssemblyFormat = 1;
  let genVerifyDecl = 1;
}

def DotOperandEncodingAttr : DistributedEncoding<"DotOperandEncoding", "dot_operand_encoding"> {
  let mnemonic = "dot_op";

  let description = [{
In the TritonGPU dialect, given `d = tt.dot a, b, c` tt.dot's operands a and b
must be of DotOperandEncodingAttr layout, if the dot is MMA v1 or v2 (i.e.
pre-Hopper).  For MMA v3, the operands are *almost always* in a regular shared
encoding, but sometimes the LHS is also a dot-operand encoding.

a's opIdx is 0, b's opIdx is 1.

The parent field is the layout of d.

kWidth defines number of consecutive elements stored by one thread along k dimension.
Some layouts do not use this parameter, either because they have a fixed number of
elements along the K dim, or they use all elements of the tensor along the K dim.

# WGMMA Notes
We require kWidth to be provided for Hopper because the dtype at loading might be
different from the dtype at WGMMA, due to casting. The kWidth is determined by the
dtype at WGMMA.

The encoded tensor consists of operand A for possibly multiple wgmma instructions.
For each wgmma, each warp in a warp group feeds a single "warp matrix"
Each warp matrix consists of 2x2 "quads".
Each thread holds several elements in each quad. Right before a wgmma,
the sum of bitwidth of
the elements in each quad should add up to 32.

These values are stored unrolled in `elements`.
The ordering of dimensions is as follows by convention:
batch (only 1 batch for Hopper currently)
matM (m-index of the "warp matrix")
matK (k-index of the "warp matrix")
quadK (k-index of the "quad" in the core matrix)
quadM (m-index of the "quad" in the core matrix)
vecIdx (index of the element in the quad; this is always along the k-dim)
  }];

  let parameters = (
    ins
    "unsigned":$opIdx,
    "Attribute":$parent,
    DefaultValuedParameter<"unsigned", "0">:$kWidth,
    "unsigned":$useSme
  );

  let builders = [
    AttrBuilder<(ins "unsigned":$opIdx,
                     "Attribute":$parent,
                     "Type":$eltTy), [{
      NvidiaMmaEncodingAttr parentAttr = mlir::dyn_cast<NvidiaMmaEncodingAttr>(parent);
#ifdef __ILUVATAR__
      IluvatarMmaEncodingAttr iluvatarParentAttr =
          mlir::dyn_cast<IluvatarMmaEncodingAttr>(parent);
#endif
      if ((!parentAttr || (!parentAttr.isAmpere() && !parentAttr.isHopper()))
#ifdef __ILUVATAR__
          && !iluvatarParentAttr
#endif
      )
        return $_get(context, opIdx, parent, 0, 0);
      // For MMAV2 and V3
      unsigned bitwidth = eltTy.getIntOrFloatBitWidth();
      unsigned kWidth = std::max(32 / bitwidth, 1u);
      return $_get(context, opIdx, parent, kWidth, 0);
    }]>,

    // Backward-compatible builder (opIdx, parent, kWidth) without useSme
    AttrBuilder<(ins "unsigned":$opIdx,
                     "Attribute":$parent,
                     "unsigned":$kWidth), [{
      return $_get(context, opIdx, parent, kWidth, 0);
    }]>,

    // Specially for MR/BI150
    AttrBuilder<(ins "unsigned":$opIdx,
                     "Attribute":$parent,
                     "Type":$eltTy,
                     "unsigned":$useSme), [{
      IluvatarMmaEncodingAttr parentAttr = mlir::dyn_cast<IluvatarMmaEncodingAttr>(parent);
      unsigned bitwidth = eltTy.getIntOrFloatBitWidth();
      unsigned kWidth = 32 / bitwidth;
      return $_get(context, opIdx, parent, kWidth, useSme);
    }]>
  ];

  let assemblyFormat = "`<` `{` struct(params) `}` `>`";
  let genVerifyDecl = 1;
  let extraClassDeclaration = extraDistributedDeclaration;
}

def TTG_SharedMemorySpace : AttrDef<TritonGPU_Dialect, "SharedMemorySpace"> {
  let mnemonic = "shared_memory";
  let description = [{
    Attribute to indicate that the memory descriptor points to shared memory.
  }];
}

#endif
