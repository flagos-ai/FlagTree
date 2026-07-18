#ifndef TRITON_DIALECT_TRITONGPU_TRANSFORMS_MACACOMMON_H_
#define TRITON_DIALECT_TRITONGPU_TRANSFORMS_MACACOMMON_H_

#include "triton/Analysis/AxisInfo.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

using namespace mlir;
using ::mlir::triton::gpu::BlockedEncodingAttr;
using ::mlir::triton::gpu::ConvertLayoutOp;
using ::mlir::triton::gpu::SwizzledSharedEncodingAttr;
namespace ttg = triton::gpu;
namespace tt = triton;
using namespace triton;
namespace maca {

// Linearize supposing order is [0, 1, .. , n]
template <typename T>
T getSubLinearIndexImpl(llvm::ArrayRef<T> multiDimIndex,
                        llvm::ArrayRef<T> shape) {
  assert(multiDimIndex.size() == shape.size());
  // shape: {a, b, c, d}  ->  accMul: {1, a, a*b, a*b*c}
  size_t rank = shape.size();
  T accMul = product(shape.drop_back());
  T linearIndex = 0;
  for (int i = rank - 1; i >= 0; --i) {
    linearIndex += multiDimIndex[i] * accMul;
    if (i != 0) {
      accMul = accMul / shape[i - 1];
    }
  }
  return linearIndex;
}

template <typename T>
T getSubLinearIndex(llvm::ArrayRef<T> multiDimIndex, llvm::ArrayRef<T> shape,
                    llvm::ArrayRef<unsigned> order) {
  assert(shape.size() == order.size());
  return getSubLinearIndexImpl<T>(applyPermutation(multiDimIndex, order),
                                  applyPermutation(shape, order));
}
namespace debug {

#define DECLARE_MACA_ENV_API(name) bool name()

#define DEFINE_MACA_ENV_API_NE(api_name, env_name)                             \
  bool api_name() {                                                            \
    static char const *temp = std::getenv(env_name);                           \
    return temp != nullptr;                                                    \
  }

DECLARE_MACA_ENV_API(get_enable_debug_internal);

/*
Add an item in DebugOption must also update its debugString in
getDebugOptionString(...), See [getDebugOptionString].
*/
enum class DebugOption : uint16_t {
  kCheckCollectValidOp = 0,
  // *****************************
  kTotalcount
};

/*
DebugResult will be static_casted into int when using in attribute value in
ParentFunctionOp. See function setParentFunctionOpAttrDebug(...).
*/
enum class DebugResult : uint16_t {
  kSuccess = 0,
  kFail = 1,
  // *****************************
  kTotalcount
};

void setParentFunctionOpAttrDebug(Operation *op, DebugOption option,
                                  DebugResult result);

} // namespace debug
} // namespace maca

enum class Layout : uint8_t {
  NT = 0, // default
  TN,
  TT,
  NN,
};

// lds(shared->dotoperand op)'s size
// {m, n, k, numWarps} -> {elemsM, elemsN, elemsK}, {warpM, warpN}
using TileTable =
    std::map<std::pair<llvm::SmallVector<int, 4>, Layout>,
             llvm::SmallVector<std::tuple<llvm::SmallVector<unsigned, 3>,
                                          llvm::SmallVector<unsigned, 2>>>>;

// lds(shared->dotoperand op)'s size
// {m, n, k, numWarps} -> {elemsM, elemsN, elemsK}, {warpM, warpN}, {version}
// version is for different pattern.
using TileTablePattern = std::map<
    std::pair<llvm::SmallVector<int, 4>, Layout>,
    llvm::SmallVector<std::tuple<llvm::SmallVector<unsigned, 3>,
                                 llvm::SmallVector<unsigned, 2>, int>>>;
using LayoutTable = std::map<
    std::pair<llvm::SmallVector<unsigned, 2>, llvm::SmallVector<unsigned, 2>>,
    Layout>;

// lds(shared->dotoperand op)'s size
// {m, n, k, numWarps} -> {stageM, stageN}
using StageMNTable = std::map<std::pair<llvm::SmallVector<int, 4>, Layout>,
                              llvm::SmallVector<unsigned, 2>>;

static TileTable fp32table = {
    {{{128, 128, 16, 8}, Layout::TT}, {{{1, 4, 4}, {4, 2}}}},
    {{{128, 128, 16, 8}, Layout::TN}, {{{1, 1, 4}, {4, 2}}}},
    {{{128, 128, 16, 8}, Layout::NT}, {{{2, 2, 1}, {2, 4}}}},
    {{{128, 128, 16, 8}, Layout::NN}, {{{2, 1, 4}, {2, 4}}}},
    {{{128, 128, 32, 4}, Layout::TN}, {{{1, 1, 4}, {2, 2}}}},
};

static StageMNTable fp32StageMNTable = {
    {{{128, 128, 16, 8}, Layout::TN}, {4, 4}},
};

static TileTable fp16table = {
    {{{256, 256, 16, 8}, Layout::TT},
     {{{1, 8, 4},
       {4,
        2}}}}, // {m, n, k, numWarps}, {elemsM, elemsN, elemsK}, {warpM, warpN}
    {{{256, 256, 16, 8}, Layout::TN}, {{{1, 1, 4}, {4, 2}}}},
    {{{256, 256, 16, 8}, Layout::NN}, {{{4, 4, 4}, {2, 4}}}},
    {{{256, 256, 16, 8}, Layout::NT}, {{{4, 4, 4}, {2, 4}}}},
    {{{256, 256, 32, 8}, Layout::TN}, {{{1, 1, 8}, {2, 4}}}},
    {{{256, 256, 64, 8}, Layout::TN}, {{{1, 1, 8}, {2, 4}}}},
    {{{128, 128, 32, 8}, Layout::TN}, {{{1, 1, 8}, {4, 2}}}},
    {{{128, 128, 256, 4}, Layout::TN}, {{{1, 1, 8}, {2, 2}}}},
    {{{128, 128, 128, 4}, Layout::TN},
     {{{1, 1, 8}, {2, 2}}, {{1, 4, 8}, {2, 2}}}},
    {{{128, 128, 64, 4}, Layout::TN}, {{{1, 1, 8}, {2, 2}}}},
    {{{128, 128, 32, 4}, Layout::TN}, {{{1, 1, 8}, {2, 2}}}},
    {{{128, 128, 128, 4}, Layout::TT}, {{{1, 4, 8}, {2, 2}}}},
    {{{128, 128, 64, 4}, Layout::TT}, {{{1, 1, 8}, {2, 2}}}},
    {{{128, 128, 32, 4}, Layout::NT}, {{{1, 1, 8}, {2, 2}}}},
    {{{64, 128, 64, 4}, Layout::TN},
     {{{1, 1, 8}, {2, 2}}, {{1, 4, 8}, {2, 2}}}},
    {{{64, 128, 128, 4}, Layout::TN},
     {{{1, 1, 8}, {2, 2}}, {{1, 4, 8}, {2, 2}}}},
    {{{64, 64, 64, 4}, Layout::TN}, {{{1, 2, 8}, {2, 2}}}},
    {{{64, 64, 64, 4}, Layout::TT}, {{{1, 1, 8}, {4, 1}}}},
    {{{64, 64, 32, 4}, Layout::TT}, {{{1, 4, 8}, {4, 1}}}},
    {{{64, 64, 64, 4}, Layout::NT}, {{{1, 2, 8}, {4, 1}}}},
    {{{64, 64, 32, 4}, Layout::TN}, {{{1, 1, 8}, {4, 1}}}},
    {{{32, 128, 32, 8}, Layout::TN}, {{{1, 1, 8}, {2, 4}}}},
    {{{32, 64, 64, 8}, Layout::TN}, {{{1, 1, 8}, {2, 4}}}},
    {{{16, 128, 32, 8}, Layout::TN}, {{{1, 1, 8}, {1, 8}}}},
    {{{16, 128, 32, 4}, Layout::TN}, {{{1, 1, 8}, {1, 4}}}},
    {{{16, 256, 16, 4}, Layout::TN}, {{{1, 1, 8}, {1, 4}}}},
    {{{16, 256, 16, 8}, Layout::TN}, {{{1, 1, 8}, {1, 8}}}},
    {{{16, 64, 32, 4}, Layout::TN}, {{{1, 1, 8}, {1, 4}}}},
    {{{16, 64, 64, 4}, Layout::TN}, {{{1, 1, 8}, {1, 4}}}},
    {{{16, 64, 128, 4}, Layout::TN}, {{{1, 1, 8}, {1, 4}}}},
    {{{64, 128, 32, 8}, Layout::TN}, {{{1, 1, 8}, {2, 4}}}},
    {{{32, 128, 64, 4}, Layout::TN}, {{{1, 1, 8}, {2, 2}}}},
    {{{32, 128, 128, 4}, Layout::TN}, {{{1, 1, 8}, {2, 2}}}},
    {{{128, 32, 64, 4}, Layout::TN}, {{{1, 1, 8}, {2, 2}}}},
    {{{128, 64, 64, 4}, Layout::TN}, {{{1, 1, 8}, {2, 2}}}},
    {{{128, 256, 64, 4}, Layout::TN}, {{{1, 1, 8}, {2, 2}}}},
    {{{128, 32, 128, 4}, Layout::TN}, {{{1, 1, 8}, {2, 2}}}},
    {{{128, 64, 128, 4}, Layout::TN}, {{{1, 1, 8}, {2, 2}}}},
};

static StageMNTable fp16StageMNTable = {
    {{{128, 128, 64, 4}, Layout::TN}, {4, 4}},
    {{{128, 256, 64, 4}, Layout::TN}, {2, 4}},
    {{{128, 128, 128, 4}, Layout::TN}, {4, 4}},
    {{{64, 128, 128, 4}, Layout::TN}, {2, 4}},
    {{{128, 128, 128, 4}, Layout::TN}, {2, 4}},
    {{{32, 128, 64, 4}, Layout::TN}, {1, 4}},
    {{{32, 128, 128, 4}, Layout::TN}, {1, 4}},
    {{{64, 128, 64, 4}, Layout::TN}, {2, 4}},
    {{{128, 32, 64, 4}, Layout::TN}, {4, 1}},
    {{{128, 64, 64, 4}, Layout::TN}, {4, 2}},
    {{{128, 32, 128, 4}, Layout::TN}, {4, 1}},
    {{{128, 64, 128, 4}, Layout::TN}, {4, 2}},
};

static TileTable fp16table_86 = {
    {{{256, 256, 16, 8}, Layout::TT},
     {{{1, 8, 4},
       {4,
        2}}}}, // {m, n, k, numWarps}, {elemsM, elemsN, elemsK}, {warpM, warpN}
    {{{256, 256, 16, 8}, Layout::TN}, {{{1, 1, 4}, {4, 2}}}},
    {{{256, 256, 16, 8}, Layout::NN}, {{{4, 4, 4}, {2, 4}}}},
    {{{256, 256, 16, 8}, Layout::NT}, {{{4, 4, 4}, {2, 4}}}},
    {{{256, 256, 32, 8}, Layout::TN}, {{{1, 1, 8}, {2, 4}}}},
    {{{256, 256, 64, 8}, Layout::TN}, {{{1, 1, 8}, {2, 4}}}},
    {{{128, 128, 32, 8}, Layout::TN}, {{{1, 1, 8}, {4, 2}}}},
    {{{128, 128, 256, 4}, Layout::TN}, {{{1, 1, 8}, {2, 2}}}},
    {{{128, 128, 128, 4}, Layout::TN},
     {{{1, 1, 8}, {2, 2}}, {{1, 4, 8}, {2, 2}}}},
    {{{128, 128, 64, 4}, Layout::TN}, {{{1, 1, 8}, {2, 2}}}},
    {{{128, 128, 32, 4}, Layout::TN}, {{{1, 1, 8}, {2, 2}}}},
    {{{128, 128, 128, 4}, Layout::TT}, {{{1, 4, 8}, {2, 2}}}},
    {{{128, 128, 64, 4}, Layout::TT}, {{{2, 2, 8}, {2, 2}}}},
    {{{128, 128, 64, 4}, Layout::NN}, {{{2, 2, 8}, {2, 2}}}},
    {{{128, 128, 64, 4}, Layout::NT}, {{{2, 2, 8}, {2, 2}}}},
    {{{128, 128, 32, 4}, Layout::NT}, {{{2, 2, 4}, {2, 2}}}},
    {{{128, 128, 32, 4}, Layout::TT}, {{{1, 2, 4}, {2, 2}}}},
    {{{128, 128, 32, 4}, Layout::NN}, {{{2, 1, 4}, {2, 2}}}},
    {{{128, 128, 16, 4}, Layout::NT}, {{{1, 1, 4}, {2, 2}}}},
    {{{128, 128, 16, 4}, Layout::TT}, {{{2, 1, 4}, {2, 2}}}},
    {{{128, 128, 16, 4}, Layout::NN}, {{{1, 2, 4}, {2, 2}}}},
    {{{64, 128, 64, 4}, Layout::TN},
     {{{1, 1, 8}, {2, 2}}, {{1, 4, 8}, {2, 2}}}},
    {{{64, 128, 128, 4}, Layout::TN},
     {{{1, 1, 8}, {2, 2}}, {{1, 4, 8}, {2, 2}}}},
    {{{64, 64, 64, 4}, Layout::TN}, {{{1, 2, 8}, {2, 2}}}},
    {{{64, 64, 64, 4}, Layout::TT}, {{{1, 1, 8}, {2, 2}}}},
    {{{64, 64, 64, 4}, Layout::NT}, {{{1, 1, 8}, {2, 2}}}},
    {{{64, 64, 64, 4}, Layout::NN}, {{{1, 1, 8}, {2, 2}}}},
    {{{64, 32, 16, 4}, Layout::NT}, {{{2, 1, 4}, {2, 2}}}},
    {{{64, 32, 32, 4}, Layout::NT}, {{{1, 1, 8}, {2, 2}}}},
    {{{64, 32, 32, 4}, Layout::TT}, {{{1, 1, 4}, {2, 2}}}},
    {{{64, 32, 16, 4}, Layout::TT}, {{{1, 1, 4}, {2, 2}}}},
    {{{64, 32, 16, 4}, Layout::TT}, {{{1, 1, 4}, {2, 2}}}},
    {{{32, 64, 32, 4}, Layout::TT}, {{{1, 2, 4}, {2, 2}}}},
    {{{32, 64, 16, 4}, Layout::TT}, {{{1, 2, 4}, {2, 2}}}},
    {{{64, 32, 16, 4}, Layout::NN}, {{{2, 1, 4}, {2, 2}}}},
    {{{64, 64, 32, 4}, Layout::TN}, {{{1, 1, 8}, {4, 1}}}},
    {{{32, 32, 32, 4}, Layout::NN}, {{{1, 1, 4}, {2, 2}}}},
    {{{32, 32, 32, 4}, Layout::TT}, {{{1, 1, 4}, {2, 2}}}},
    {{{32, 32, 32, 4}, Layout::NT}, {{{1, 1, 8}, {2, 2}}}},
    {{{32, 32, 32, 4}, Layout::TN}, {{{1, 1, 8}, {2, 2}}}},
    {{{32, 128, 32, 8}, Layout::TN}, {{{1, 1, 8}, {2, 4}}}},
    {{{32, 64, 64, 8}, Layout::TN}, {{{1, 1, 8}, {2, 4}}}},
    {{{16, 128, 32, 8}, Layout::TN}, {{{1, 1, 8}, {1, 8}}}},
    {{{16, 128, 32, 4}, Layout::TN}, {{{1, 1, 8}, {1, 4}}}},
    {{{16, 256, 16, 4}, Layout::TN}, {{{1, 1, 8}, {1, 4}}}},
    {{{16, 256, 16, 8}, Layout::TN}, {{{1, 1, 8}, {1, 8}}}},
    {{{16, 64, 32, 4}, Layout::TN}, {{{1, 1, 8}, {1, 4}}}},
    {{{16, 64, 64, 4}, Layout::TN}, {{{1, 1, 8}, {1, 4}}}},
    {{{16, 64, 128, 4}, Layout::TN}, {{{1, 1, 8}, {1, 4}}}},
    {{{64, 128, 32, 8}, Layout::TN}, {{{1, 1, 8}, {2, 4}}}},
};

static TileTablePattern fp16ChainTable = {
    // NOTICE: if chain dot match different config will cause mma2mma in the
    // graph
    {{{128, 128, 64, 8}, Layout::TN},
     {{{1, 2, 8},
       {8, 1},
       4}}}, // {m, n, k, numWarps}, {elemsM, elemsN, elemsK}, {warpM, warpN}
    {{{128, 64, 128, 8}, Layout::TT}, {{{1, 2, 8}, {8, 1}, 4}}},
    {{{128, 64, 128, 8}, Layout::NT}, {{{1, 2, 8}, {8, 1}, 4}}},
    {{{128, 128, 32, 8}, Layout::TT}, {{{1, 2, 8}, {8, 1}, 4}}},
    {{{128, 32, 128, 8}, Layout::TN}, {{{1, 2, 8}, {8, 1}, 4}}},
    {{{64, 32, 128, 4}, Layout::TN}, {{{1, 2, 8}, {4, 1}, 4}}},
    {{{64, 128, 32, 4}, Layout::TT}, {{{1, 2, 8}, {4, 1}, 4}}},
    {{{64, 32, 32, 4}, Layout::TN}, {{{1, 2, 8}, {4, 1}, 4}}},
    {{{64, 32, 32, 4}, Layout::TT}, {{{1, 2, 8}, {4, 1}, 4}}},
    {{{64, 32, 32, 4}, Layout::NT}, {{{1, 2, 8}, {4, 1}, 4}}},
    {{{64, 32, 64, 4}, Layout::TN}, {{{1, 2, 8}, {4, 1}, 4}}},
    {{{64, 64, 64, 4}, Layout::TN}, {{{1, 2, 8}, {4, 1}, 5}}},
    {{{64, 64, 64, 4}, Layout::TT}, {{{1, 2, 8}, {4, 1}, 5}}},
    {{{64, 64, 64, 4}, Layout::NT}, {{{1, 2, 8}, {4, 1}, 5}}},
    {{{128, 64, 64, 8}, Layout::TN}, {{{1, 2, 8}, {8, 1}, 6}}},
    {{{128, 64, 64, 8}, Layout::TT}, {{{1, 2, 8}, {8, 1}, 6}}},
    {{{64, 64, 128, 4}, Layout::TN}, {{{1, 2, 8}, {4, 1}, 7}}},
    {{{64, 128, 64, 4}, Layout::TT}, {{{1, 2, 8}, {4, 1}, 7}}},
    {{{16, 16, 512, 1}, Layout::TT}, {{{1, 1, 4}, {1, 1}, 8}}},
    {{{16, 512, 16, 1}, Layout::TT}, {{{1, 1, 4}, {1, 1}, 8}}},
    {{{16, 16, 512, 1}, Layout::TN}, {{{1, 1, 4}, {1, 1}, 8}}},
    {{{16, 16, 64, 1}, Layout::TN}, {{{1, 1, 4}, {1, 1}, 8}}},
    {{{16, 16, 512, 4}, Layout::TT}, {{{1, 1, 4}, {1, 4}, 9}}},
    {{{16, 512, 16, 4}, Layout::TT}, {{{1, 1, 4}, {1, 4}, 9}}},
    {{{16, 512, 16, 4}, Layout::TN}, {{{1, 1, 4}, {1, 4}, 9}}},
    {{{16, 16, 512, 4}, Layout::TN}, {{{1, 1, 4}, {1, 4}, 9}}},
    {{{16, 16, 64, 4}, Layout::TN}, {{{1, 1, 4}, {1, 4}, 9}}},
    {{{16, 16, 32, 4}, Layout::TN}, {{{1, 1, 4}, {1, 4}, 9}}},
    {{{16, 16, 256, 4}, Layout::TN}, {{{1, 1, 4}, {1, 4}, 9}}},
    {{{16, 16, 256, 4}, Layout::TT}, {{{1, 1, 4}, {1, 4}, 9}}},
    {{{16, 256, 16, 4}, Layout::TT}, {{{1, 1, 4}, {1, 4}, 9}}},
    {{{16, 16, 128, 4}, Layout::TN}, {{{1, 1, 4}, {1, 4}, 9}}},
    {{{16, 16, 128, 4}, Layout::TT}, {{{1, 1, 4}, {1, 4}, 9}}},
    {{{16, 128, 16, 4}, Layout::TT}, {{{1, 1, 4}, {1, 4}, 9}}},
    {{{32, 16, 512, 8}, Layout::TN}, {{{1, 1, 4}, {2, 4}, 10}}},
    {{{32, 16, 64, 8}, Layout::TN}, {{{1, 1, 4}, {2, 4}, 10}}},
    {{{32, 16, 512, 8}, Layout::TT}, {{{1, 1, 4}, {2, 4}, 10}}},
    {{{32, 512, 16, 8}, Layout::TT}, {{{1, 1, 4}, {2, 4}, 10}}},
    {{{32, 512, 16, 8}, Layout::TN}, {{{1, 1, 4}, {2, 4}, 10}}},
    {{{32, 16, 512, 4}, Layout::TN}, {{{1, 1, 4}, {2, 2}, 11}}},
    {{{32, 16, 64, 4}, Layout::TN}, {{{1, 1, 4}, {2, 2}, 11}}},
    {{{32, 16, 512, 4}, Layout::TT}, {{{1, 1, 4}, {2, 2}, 11}}},
    {{{32, 512, 16, 4}, Layout::TT}, {{{1, 1, 4}, {2, 2}, 11}}},
    {{{32, 512, 16, 4}, Layout::TN}, {{{1, 1, 4}, {2, 2}, 11}}},
    {{{64, 16, 512, 8}, Layout::TN}, {{{1, 1, 4}, {4, 2}, 12}}},
    {{{64, 16, 64, 8}, Layout::TN}, {{{1, 1, 4}, {4, 2}, 12}}},
    {{{64, 16, 512, 8}, Layout::TT}, {{{1, 1, 4}, {4, 2}, 12}}},
    {{{64, 512, 16, 8}, Layout::TT}, {{{1, 1, 4}, {4, 2}, 12}}},
    {{{16, 32, 512, 4}, Layout::TN}, {{{1, 2, 4}, {1, 4}, 13}}},
    {{{16, 32, 64, 4}, Layout::TN}, {{{1, 2, 4}, {1, 4}, 13}}},
    {{{16, 32, 512, 4}, Layout::TT}, {{{1, 2, 4}, {1, 4}, 13}}},
    {{{16, 512, 32, 4}, Layout::TT}, {{{1, 2, 4}, {1, 4}, 13}}},
};

static TileTable tf32table = {
    {{{128, 128, 16, 8}, Layout::TT}, {{{1, 4, 4}, {4, 2}}}},
    {{{128, 128, 16, 8}, Layout::TN}, {{{1, 1, 4}, {4, 2}}}},
    {{{128, 128, 16, 8}, Layout::NT}, {{{2, 2, 2}, {2, 4}}}},
    {{{128, 128, 16, 8}, Layout::NN}, {{{2, 1, 4}, {2, 4}}}},
};

static StageMNTable tf32StageMNTable = {
    {{{128, 128, 16, 8}, Layout::TN}, {4, 4}},
};

static TileTable i8table = {
    {{{256, 256, 64, 8}, Layout::TN}, {{{1, 1, 16}, {4, 2}}}},
    {{{256, 256, 32, 8}, Layout::TN}, {{{1, 1, 8}, {4, 2}}}},
    {{{128, 128, 64, 8}, Layout::TN}, {{{1, 1, 16}, {4, 2}}}},
    {{{256, 128, 64, 8}, Layout::TN}, {{{1, 1, 16}, {4, 2}}}},
    {{{128, 128, 128, 4}, Layout::TN},
     {{{1, 1, 16}, {2, 2}}, {{1, 4, 16}, {2, 2}}}},
    {{{128, 128, 64, 4}, Layout::TN}, {{{1, 1, 16}, {2, 2}}}},
    {{{256, 256, 128, 8}, Layout::TN}, {{{1, 1, 16}, {4, 2}}}},
    {{{128, 128, 256, 4}, Layout::TN},
     {{{1, 1, 16}, {2, 2}}, {{1, 4, 16}, {2, 2}}}},
    {{{128, 128, 32, 4}, Layout::TN},
     {{{1, 1, 8}, {2, 2}}, {{1, 4, 8}, {2, 2}}}},
    {{{64, 64, 128, 4}, Layout::TN}, {{{1, 1, 16}, {2, 2}}}},
    {{{64, 128, 64, 4}, Layout::TN},
     {{{1, 1, 16}, {2, 2}}, {{1, 4, 16}, {2, 2}}}},
    {{{64, 128, 128, 4}, Layout::TN},
     {{{1, 1, 16}, {2, 2}}, {{1, 4, 16}, {2, 2}}}},
    {{{64, 256, 64, 4}, Layout::TN},
     {{{1, 1, 16}, {1, 4}}, {{1, 4, 16}, {1, 4}}}},
    {{{16, 64, 256, 4}, Layout::TN}, {{{1, 1, 16}, {2, 2}}}},
    {{{16, 64, 128, 4}, Layout::TN}, {{{1, 1, 16}, {2, 2}}}},
    {{{16, 256, 64, 4}, Layout::TN},
     {{{1, 1, 16}, {1, 4}}, {{1, 4, 16}, {1, 4}}}},
    {{{16, 256, 32, 4}, Layout::TN}, {{{1, 1, 8}, {1, 4}}}},
    {{{32, 256, 32, 4}, Layout::TN},
     {{{1, 1, 8}, {2, 2}}, {{1, 4, 8}, {2, 2}}}},
    {{{32, 128, 64, 4}, Layout::TN}, {{{1, 1, 16}, {2, 2}}}},
    {{{32, 128, 128, 4}, Layout::TN}, {{{1, 1, 16}, {2, 2}}}},
    {{{32, 128, 256, 4}, Layout::TN}, {{{1, 1, 16}, {2, 2}}}},
    {{{64, 128, 256, 4}, Layout::TN}, {{{1, 1, 16}, {2, 2}}}},
    {{{128, 32, 128, 4}, Layout::TN}, {{{1, 1, 16}, {2, 2}}}},
    {{{128, 64, 128, 4}, Layout::TN}, {{{1, 1, 16}, {2, 2}}}},
    {{{256, 64, 256, 8}, Layout::TN}, {{{1, 1, 16}, {4, 2}}}},
    {{{256, 128, 256, 8}, Layout::TN}, {{{1, 1, 16}, {4, 2}}}},
    {{{64, 256, 128, 8}, Layout::TN}, {{{1, 1, 16}, {4, 2}}}},
    {{{128, 128, 128, 8}, Layout::TN}, {{{1, 1, 16}, {4, 2}}}},
};

static StageMNTable i8StageMNTable = {
    {{{128, 128, 128, 4}, Layout::TN}, {4, 4}},
    {{{256, 256, 128, 8}, Layout::TN}, {4, 4}},
    {{{128, 128, 256, 4}, Layout::TN}, {4, 4}},
    {{{64, 128, 64, 4}, Layout::TN}, {2, 4}},
    {{{64, 256, 64, 4}, Layout::TN}, {1, 4}},
    {{{64, 128, 128, 4}, Layout::TN}, {2, 4}},
    {{{32, 128, 64, 4}, Layout::TN}, {1, 4}},
    {{{32, 128, 128, 4}, Layout::TN}, {1, 4}},
    {{{32, 128, 256, 4}, Layout::TN}, {1, 4}},
    {{{64, 128, 256, 4}, Layout::TN}, {2, 4}},
    {{{128, 32, 128, 4}, Layout::TN}, {4, 1}},
    {{{128, 64, 128, 4}, Layout::TN}, {4, 2}},
    {{{256, 64, 128, 8}, Layout::TN}, {4, 1}},
    {{{256, 128, 128, 8}, Layout::TN}, {4, 2}},
    {{{256, 64, 256, 8}, Layout::TN}, {4, 1}},
    {{{256, 128, 256, 8}, Layout::TN}, {4, 2}},
    {{{64, 256, 128, 8}, Layout::TN}, {1, 4}},
    {{{128, 256, 128, 8}, Layout::TN}, {2, 4}},
};

static TileTable i8table_86 = {
    {{{256, 256, 64, 8}, Layout::TN}, {{{1, 1, 16}, {4, 2}}}},
    {{{256, 256, 32, 8}, Layout::TN}, {{{1, 1, 8}, {4, 2}}}},
    {{{128, 128, 64, 8}, Layout::TN}, {{{1, 1, 16}, {4, 2}}}},
    {{{256, 128, 64, 8}, Layout::TN}, {{{1, 1, 16}, {4, 2}}}},
    {{{128, 128, 128, 4}, Layout::TN},
     {{{1, 1, 16}, {2, 2}}, {{1, 4, 16}, {2, 2}}}},
    {{{128, 128, 64, 4}, Layout::TN}, {{{1, 1, 16}, {2, 2}}}},
    {{{128, 128, 64, 4}, Layout::NN}, {{{1, 1, 8}, {2, 2}}}},
    {{{128, 128, 64, 4}, Layout::NT}, {{{2, 2, 16}, {2, 2}}}},
    {{{128, 128, 64, 4}, Layout::TT}, {{{1, 2, 8}, {2, 2}}}},
    {{{256, 256, 128, 8}, Layout::TN}, {{{1, 1, 16}, {4, 2}}}},
    {{{256, 256, 64, 8}, Layout::TN}, {{{1, 1, 16}, {4, 2}}}},
    {{{128, 128, 256, 4}, Layout::TN},
     {{{1, 1, 16}, {2, 2}}, {{1, 4, 16}, {2, 2}}}},
    {{{128, 128, 32, 4}, Layout::TN},
     {{{1, 1, 8}, {2, 2}}, {{1, 4, 8}, {2, 2}}}},
    {{{64, 64, 128, 4}, Layout::TN}, {{{1, 1, 16}, {2, 2}}}},
    {{{64, 128, 64, 4}, Layout::TN},
     {{{1, 1, 16}, {2, 2}}, {{1, 4, 16}, {2, 2}}}},
    {{{64, 256, 64, 4}, Layout::TN},
     {{{1, 1, 16}, {1, 4}}, {{1, 4, 16}, {1, 4}}}},
    {{{64, 128, 128, 4}, Layout::TN},
     {{{1, 1, 16}, {2, 2}}, {{1, 4, 16}, {2, 2}}}},
    {{{16, 64, 256, 4}, Layout::TN}, {{{1, 1, 16}, {2, 2}}}},
    {{{16, 64, 128, 4}, Layout::TN}, {{{1, 1, 16}, {2, 2}}}},
    {{{16, 256, 64, 4}, Layout::TN},
     {{{1, 1, 16}, {1, 4}}, {{1, 4, 16}, {1, 4}}}},
    {{{16, 256, 32, 4}, Layout::TN}, {{{1, 1, 8}, {2, 2}}}},
    {{{32, 256, 32, 4}, Layout::TN},
     {{{1, 1, 8}, {2, 2}}, {{1, 4, 8}, {2, 2}}}},
};

static LayoutTable layouttable = {
    {{{0, 1}, {1, 0}}, Layout::NT},
    {{{1, 0}, {0, 1}}, Layout::TN},
    {{{1, 0}, {1, 0}}, Layout::TT},
    {{{0, 1}, {0, 1}}, Layout::NN},
};

bool updateLayout(llvm::SmallVector<unsigned, 3> &elemsPerThread,
                  llvm::SmallVector<unsigned, 2> &warpsPerTile,
                  llvm::SmallVector<int, 4> tile, int &version, int numWarps,
                  bool enableTf32, mlir::Type dtype, ArrayRef<unsigned> aorder,
                  ArrayRef<unsigned> border, bool disablePrefetch,
                  bool chainDot = false, bool storeCoalesce = false,
                  int computeCapability = 80);

bool matchTable(const std::pair<llvm::SmallVector<int, 4>, Layout> &pattern,
                const TileTable &table, const int &numWarps,
                const bool enableTf32, const mlir::Type dtype,
                const ArrayRef<unsigned> aorder,
                const ArrayRef<unsigned> border,
                llvm::SmallVector<unsigned, 3> &elemsPerThread,
                llvm::SmallVector<unsigned, 2> &warpsPerTile,
                bool storeCoalesce = false);

llvm::SmallVector<unsigned, 2>
matchMNStage(const std::pair<llvm::SmallVector<int, 4>, Layout> &pattern,
             mlir::Type dtype, bool enableTf32);

llvm::SmallVector<unsigned, 3>
getDefaultElemsPerThread(Type type, bool enableTf32 = false,
                         int computeCapability = 80);

int getGVMNumberPerOp(BlockedEncodingAttr &enc, RankedTensorType &tensor);

unsigned getGVMNumberPerOp(triton::LoadOp &load);

SmallVector<unsigned, 2> getOrder(triton::gpu::ConvertLayoutOp &op);

SmallVector<unsigned, 2> getOrder(Value &src);

SmallVector<int64_t, 2> getShape(Value &src);

mlir::Type getType(const Value &src);

bool isNeedMerge(SwizzledSharedEncodingAttr srcSharedEnc,
                 SwizzledSharedEncodingAttr dstSharedEnc,
                 ArrayRef<int64_t> shape);

std::pair<SmallVector<unsigned>, SmallVector<unsigned>>
extractSubCtaAndAllElem(unsigned subTensorSize, unsigned shapePerCTA,
                        unsigned totalSizePerThread, unsigned subTensorIdx);

std::pair<SmallVector<unsigned>, SmallVector<unsigned>>
extractAllCtaAndSubElem(unsigned numCtas, unsigned subSizePerThread,
                        unsigned subTensorIdx);

std::pair<SmallVector<int64_t>, SmallVector<int64_t>>
calExtractTensorIdx(RankedTensorType tensorType, RankedTensorType subTensorType,
                    ArrayRef<int64_t> subTensorIdx);

/// generate swizzled mask by forward iteration.
void genSwiMask(Value dep, unsigned inVec,
                triton::ModuleAxisInfoAnalysis &axisInfoAnalysis,
                RankedTensorType srcTy, triton::gpu::MemDescType resTy,
                OpBuilder &builder, ArrayRef<int64_t> subSizes,
                ArrayRef<int64_t> tileIdx, int numStages,
                ArrayRef<int64_t> elemTileNum, IRMapping &mapping, Location loc,
                Value cstValue = Value(), int allStage = 0);

/// check valid mask to be swizzled and collect deps by backward recursion.
/// return_val > 1: valid and dependent ops.
/// return_val == 1: valid but not dependent ops.
/// return_val == 0 or < 1: not valid nor dependent ops.
float checkMaskDepsRecursion(Value src, unsigned inVec,
                             triton::ModuleAxisInfoAnalysis &axisInfoAnalysis,
                             SmallVector<Operation *> &deps,
                             triton::gpu::MemDescType resTy);

///  1. collect deps of old mask by checkMaskDepsRecursion.
///  2. generate new deps and mask by genSwiMask.
Value genSwiSubMask(Value mask, triton::LoadOp &op, RankedTensorType srcTy,
                    triton::gpu::MemDescType resTy, OpBuilder &builder,
                    ArrayRef<int64_t> subSizes,
                    ArrayRef<int64_t> tileIdx = {0, 0}, int numStages = 0,
                    ArrayRef<int64_t> elemTileNum = {1, 1},
                    Value cstValue = Value(), int allStage = 0);

/// check if LoadOp can be converted into AsyncCopyGlobalToLocalOp.
bool checkUseCpAsync(triton::LoadOp &op, mlir::Type elemTy,
                     triton::ModuleAxisInfoAnalysis &axisInfoAnalysis,
                     triton::gpu::SwizzledSharedEncodingAttr &resSharedLayout);

/// Check if mask of ops and its dependencies that are not pipelinable
LogicalResult
checkMaskDeps(scf::ForOp &forOp, SetVector<Operation *> &ops,
              DenseMap<Value, bool> &loadGenMask,
              DenseMap<Value, triton::gpu::MemDescType> &loadsBufferType);

/// Same as checkMaskDeps but there is one more arg loadCanCpAsync
void checkMatchCpAsync(
    scf::ForOp &forOp, SetVector<Operation *> &ops,
    DenseMap<Value, bool> &loadGenMask,
    DenseMap<Value, triton::gpu::MemDescType> &loadsBufferType,
    DenseMap<Operation *, bool> &loadCanCpAsync,
    DenseMap<Operation *, bool> &loadUseRegPipeline);

/// collect addptrs which accumulates ptr of valid addptrs recursively.
void collectAddPtrDepsRecursion(Operation *op, scf::ForOp &forOp,
                                SetVector<Value> &validAddPtrs,
                                SmallVector<triton::AddPtrOp> &addptrs);

// collect deps of valid addptr to optimize offset calculation.
// The deps hit the pattern will make optimization otherwise will not.
// supported pattern:
//   addptr -> blockarg -> addptr(valid)
//   addptr -> addptr(valid)
// unsupported pattern:
//   addptr(valid) -> addptr(valid)
//   addptr -> blockarg -> addptr(valid) -> addptr(valid)
//   other(e.g. splat/broadcast) -> blockarg -> addptr(valid)
//   other -> addptr(valid)
void collectAddPtrDeps(scf::ForOp &forOp, bool enableSaddrOpt,
                       SmallVector<Operation *> &orderedDeps,
                       SetVector<Value> &validAddPtrs,
                       DenseMap<triton::AddPtrOp, SmallVector<triton::AddPtrOp>>
                           &validPtrDepMapping);

/// add op with automatic upgrade from int32_t to int64_t.
Value addInt32OrInt64(OpBuilder &builder, Location loc, Value lhs, Value rhs,
                      Operation *anchor = nullptr);

// optimize C Store, directly store to global in MMALayout
void OptimizeCStore(OpBuilder &builder, triton::StoreOp &storeOp);

// caiculate arrive counts for pipelineTN/TT.
int calWaitArriveCounts(int stage_m, int stage_n, int i, int numStagesInner);

bool getIfLdsTrans(const SmallVector<unsigned, 3> &elemsMNK, unsigned major,
                   unsigned minor, ArrayRef<unsigned> order, bool isA,
                   Type elemTy);

/// collect all valid loads and deps of dot op.
LogicalResult
collectValidOp(scf::ForOp &forOp, scf::YieldOp &yieldOp, int numStages,
               DenseMap<Value, Value> &loadsMapping,
               DenseMap<Value, Value> &elemsMapping,
               SetVector<Value> &validLoads, SetVector<Operation *> &dotsDeps,
               SetVector<Operation *> &dotsDepsOutOfLoop,
               SetVector<Operation *> &ops, bool enableDotStage = false);

// generate the deps op which between load op and dot op.
SmallVector<Operation *> genDotDeps(
    OpBuilder &builder, scf::ForOp &newForOp, SetVector<Value> &validLoads,
    IRMapping &nextMapping, IRMapping &mapping,
    DenseMap<Value, Value> &loadsMapping, DenseMap<Value, Value> &elemsMapping,
    SetVector<Operation *> &dotsDeps, DenseMap<Value, Value> &nextDotOperands,
    DenseMap<Value, Value> &cvtStageBuffer, DenseMap<Value, Value> &cvtsExtract,
    Value extractSliceIdx, bool enableDotStage = false, int loadIdx = -1,
    bool multiDot = false);

/// [EXPERIMENTAL] get number of instruction number between mma.
int getMmaInstNum(Type type);

int getNumWarps(ModuleOp mod);

mlir::Attribute getDotOperandsAttr(Value v);

#endif // TRITON_DIALECT_TRITONGPU_TRANSFORMS_MACACOMMON_H_
