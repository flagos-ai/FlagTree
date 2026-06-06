#include "TritonAffinityOpt/Passes.h"

#include "bishengir/Dialect/Annotation/IR/Annotation.h"
#include "bishengir/Dialect/Scope/IR/Scope.h"
#include "TritonAffinityOpt/Utils.hpp"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/IR/HIVMImpl.h"
#include "bishengir/Dialect/HIVM/IR/HIVMInterfaces.h"
#include "bishengir/Dialect/HIVM/Transforms/Passes.h"
#include "bishengir/Dialect/HIVM/Utils/Utils.h"
#include "bishengir/Dialect/Scope/IR/Scope.h"
#include "mlir/Analysis/AliasAnalysis.h"
#include "mlir/Analysis/DataFlow/DeadCodeAnalysis.h"
#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallBitVector.h"
#include "llvm/Support/Casting.h"

#include "Utils/Utils.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FormatVariadic.h"
#include <cstddef>
#include <memory>
#include <optional>

#include "TritonAffinityOpt/DAG.h"
#include "triton/Dialect/Triton/IR/Types.h"

namespace mlir {
namespace triton {
#define GEN_PASS_DEF_DAGSYNC
#include "ascend/include/TritonAffinityOpt/Passes.h.inc"
}  // namespace triton
}  // namespace mlir

#define DEBUG_TYPE "triton-affinity-dag-sync"

// 使用 DAG 命名空间
using namespace mlir;
using namespace hivm;
using namespace AffinityDAG;

constexpr size_t MAX_FLAG_ID = 14;
llvm::DenseMap<Value, CoreType>* valueTypes;
// 修改类声明，将数据搬运逻辑集成到同步插入中
namespace {
struct DAGSyncPass : public mlir::triton::impl::DAGSyncBase<DAGSyncPass> {
    void runOnOperation() override;

private:
    // 原有的辅助函数
    CoreType getNodeDeviceType(OpNode *node, llvm::DenseMap<mlir::Value, CoreType> *valueTypes);
    bool needVectorCubeSync(CoreType src, CoreType dst);

    // 修改后的同步插入函数，包含数据搬运
    void insertSyncAndMovement(mlir::Operation *srcOp, mlir::Operation *dstOp,
                               CoreType srcType, CoreType dstType,
                               mlir::OpBuilder &builder, int flag, llvm::DenseMap<Value, CoreType>* valueMap, Graph &mainGraph);

    // 新增：处理跨 block 的同步和数据搬运
    void insertSyncAndMovementForCrossBlock(mlir::Operation *srcOp, mlir::Operation *dstOp,
                                           CoreType srcType, CoreType dstType,
                                           mlir::OpBuilder &builder, int flag,
                                           bool dstIsInnerBlock, llvm::DenseMap<Value, CoreType>* valueMap, Graph &mainGraph);

    // 新增：处理 scf.for 循环迭代参数的同步
    void processScfForSync(mlir::scf::ForOp forOp,
                          Node* forNode,
                          llvm::DenseMap<mlir::Value, CoreType> *valueTypes,
                          mlir::OpBuilder &builder,
                          int &flag);

    // 数据搬运相关的辅助函数
    void insertCubeToVectorDataMovement(mlir::Operation *srcOp, mlir::Operation *dstOp,
                                       mlir::Value srcResult, mlir::OpBuilder &builder,
                                       mlir::Location loc, mlir::Value iterArgs);

    void insertVectorToCubeDataMovement(mlir::Operation *srcOp, mlir::Operation *dstOp, Operation * posOp,
                                       mlir::Value srcResult, mlir::OpBuilder &builder,
                                       mlir::Location loc, llvm::DenseMap<Value, CoreType>* valueMap);

    // 获取或创建合适的 memref.alloc
    mlir::Value getOrCreateAllocation(mlir::Operation *op, mlir::Type tensorType,
                                      hivm::AddressSpace addressSpace,
                                      mlir::OpBuilder &builder, mlir::Location loc);

    // 获取 tensor 的形状和元素类型
    mlir::RankedTensorType getTensorType(mlir::Value tensorValue);

    // 替换 dstOp 中使用 srcResult 的操作数
    void replaceOperandWithNewValue(mlir::Operation *dstOp, mlir::Value oldValue,
                                    mlir::Value newValue);

    // Find sync position
    Operation* FindLastestPosition(Operation* srcOp, Graph &mainGraph, OpBuilder &builder);
    Operation* FindEarliestPosition(Operation* dstOp, Graph &mainGraph, OpBuilder &builder);
};
}  // namespace

void DAGSyncPass::processScfForSync(mlir::scf::ForOp forOp,
                                   Node* forNode,
                                   llvm::DenseMap<mlir::Value, CoreType> *valueTypes,
                                   mlir::OpBuilder &builder,
                                   int &flag) {

    mlir::Block* loopBody = forOp.getBody();
    mlir::scf::YieldOp yieldOp = nullptr;
    for (mlir::Operation &op : *loopBody) {
        if (auto yield = mlir::dyn_cast<mlir::scf::YieldOp>(&op)) {
            yieldOp = yield;
            break;
        }
    }
    Location loc = forOp.getLoc();

    for (int i = 0; i < forOp.getInitArgs().size(); i++) {
        mlir::BlockArgument iterArg = loopBody->getArgument(i+1);
        // 找到首次使用
        mlir::Operation* firstUser = nullptr;

        for (mlir::Operation &op : *loopBody) {
            // 跳过 yield 操作
            if (mlir::isa<mlir::scf::YieldOp>(&op)) {
                continue;
            }

            // 检查是否使用该迭代参数
            bool usesIterArg = false;
            for (mlir::Value operand : op.getOperands()) {
                if (operand == iterArg) {
                    usesIterArg = true;
                    break;
                }
            }

            if (usesIterArg) {
                firstUser = &op;
                break;
            }
        }
        // map 内找到对应的iterType，iterType由首次在loop内使用到的op定义
        if (!firstUser) {
            continue;
        }
        CoreType iterType = CoreType::CUBE_AND_VECTOR;
        if (valueTypes->find(firstUser->getResult(0)) != valueTypes->end()) {
            iterType = valueTypes->find(firstUser->getResult(0))->second;
        }

        // 获取对应yield
        mlir::Value yieldOperand = yieldOp->getOperand(i);
        CoreType yieldType = CoreType::CUBE_AND_VECTOR;
        if (valueTypes->find(yieldOperand) != valueTypes->end()) {
            yieldType = valueTypes->find(yieldOperand)->second;
        }
        mlir::Operation* yieldDefiningOp = yieldOperand.getDefiningOp();

    if (yieldType == CoreType::CUBE_ONLY && iterType == CoreType::VECTOR_ONLY) {

        // 2. 插入同步指令
        auto coreAttr = hivm::TCoreTypeAttr::get(builder.getContext(), hivm::TCoreType::CUBE);
        auto setPipe = PipeAttr::get(builder.getContext(), hivm::PIPE::PIPE_FIX);
        auto waitPipe = PipeAttr::get(builder.getContext(), hivm::PIPE::PIPE_V);
        auto flagId = builder.getIntegerAttr(builder.getI64Type(), flag);

        // set 在 yieldDefiningOp 后
        builder.setInsertionPointAfter(yieldDefiningOp);
        builder.create<SyncBlockSetOp>(loc, coreAttr, setPipe, waitPipe, flagId);

        mlir::Value srcResult = yieldDefiningOp->getResult(0);

        // // 1. 插入数据搬运
        insertCubeToVectorDataMovement(yieldDefiningOp, firstUser, srcResult, builder, loc, iterArg);

        // wait 在 firstUser 前
        builder.setInsertionPoint(firstUser);
        coreAttr = hivm::TCoreTypeAttr::get(builder.getContext(), hivm::TCoreType::VECTOR);
        builder.create<SyncBlockWaitOp>(loc, coreAttr, setPipe, waitPipe, flagId);
        // llvm::outs() << "yieldOp" << yieldDefiningOp << "iterargs" << firstUser << "\n";
        // llvm::outs() << "Inserted CUBE->VECTOR sync and data movement (flag=" << flag << ")\n";
    }
    // VECTOR -> CUBE
    else if (yieldType == CoreType::VECTOR_ONLY && iterType == CoreType::CUBE_ONLY) {

        // 2. 插入同步指令
        auto coreAttr = hivm::TCoreTypeAttr::get(builder.getContext(), hivm::TCoreType::VECTOR);
        auto setPipe = PipeAttr::get(builder.getContext(), hivm::PIPE::PIPE_MTE3);
        auto waitPipe = PipeAttr::get(builder.getContext(), hivm::PIPE::PIPE_MTE1);
        auto flagId = builder.getIntegerAttr(builder.getI64Type(), flag);

        // set 在 yieldDefiningOp 后
        builder.setInsertionPointAfter(yieldDefiningOp);
        builder.create<SyncBlockSetOp>(loc, coreAttr, setPipe, waitPipe, flagId);

        // 1. 插入数据搬运
        // insertVectorToCubeDataMovement(yieldDefiningOp, firstUser, srcResult, builder, loc, iterArg);

        // wait 在 firstUser 前
        builder.setInsertionPoint(firstUser);
        coreAttr = hivm::TCoreTypeAttr::get(builder.getContext(), hivm::TCoreType::CUBE);
        builder.create<SyncBlockWaitOp>(loc, coreAttr, setPipe, waitPipe, flagId);
        // llvm::outs() << "yieldOp" << yieldDefiningOp << "iterargs" << firstUser << "\n";
        // llvm::outs() << "Inserted VECTOR->CUBE sync and data movement (flag=" << flag << ")\n";
    }
    }
}

// 获取节点的设备类型
CoreType DAGSyncPass::getNodeDeviceType(OpNode *node, llvm::DenseMap<mlir::Value, CoreType> *valueTypes)
{
    if (!node || !node->op) {
        return CoreType::CUBE_AND_VECTOR;
    }

    // 尝试从节点的结果中获取设备类型
    // 通常使用第一个结果来代表节点的设备类型
    if (node->op->getNumResults() > 0) {
        mlir::Value result = node->op->getResult(0);
        auto it = valueTypes->find(result);
        if (it != valueTypes->end()) {
            return it->second;
        }
    }

    // 如果没有找到，检查操作数
    // for (mlir::Value operand : node->op->getOperands()) {
    //     auto it = valueTypes->find(operand);
    //     if (it != valueTypes->end()) {
    //         return it->second;
    //     }
    // }

    return CoreType::CUBE_AND_VECTOR;  // 默认
}

// 判断是否需要vector<->cube同步
bool DAGSyncPass::needVectorCubeSync(CoreType src, CoreType dst)
{
    return (src == CoreType::VECTOR_ONLY && dst == CoreType::CUBE_ONLY) ||
           (src == CoreType::CUBE_ONLY && dst == CoreType::VECTOR_ONLY);
}

// 获取 tensor 类型
mlir::RankedTensorType DAGSyncPass::getTensorType(mlir::Value tensorValue) {
    if (auto tensorType = dyn_cast<mlir::RankedTensorType>(tensorValue.getType())) {
        return tensorType;
    }
    return nullptr;
}

// 替换操作数
void DAGSyncPass::replaceOperandWithNewValue(mlir::Operation *dstOp, mlir::Value oldValue,
                                            mlir::Value newValue) {
    for (unsigned i = 0; i < dstOp->getNumOperands(); ++i) {
        if (dstOp->getOperand(i) == oldValue) {
            dstOp->setOperand(i, newValue);
            // llvm::outs() << "Replaced operand " << i << " of " << dstOp->getName().getStringRef()
            //             << " with new value\n";
        }
    }
}

// 修改 getOrCreateAllocation 函数，将 alloc 提到函数最外层
mlir::Value DAGSyncPass::getOrCreateAllocation(mlir::Operation *op, mlir::Type tensorType,
                                               hivm::AddressSpace addressSpace,
                                               mlir::OpBuilder &builder, mlir::Location loc) {
    auto rankedTensorType = cast<mlir::RankedTensorType>(tensorType);
    auto elementType = rankedTensorType.getElementType();
    auto shape = rankedTensorType.getShape();

    auto addressSpaceAttr = hivm::AddressSpaceAttr::get(builder.getContext(), addressSpace);
    auto memrefType = mlir::MemRefType::get(shape, elementType, /*layout=*/nullptr, addressSpaceAttr);

    // 查找是否已经存在相同类型的 allocation（在函数的 entry block 中）
    mlir::Operation* funcOp = op;
    while (funcOp && !mlir::isa<mlir::triton::FuncOp>(funcOp)) {
        funcOp = funcOp->getParentOp();
    }

    if (auto func = mlir::dyn_cast<mlir::triton::FuncOp>(funcOp)) {
        // 在函数的 entry block 中查找现有的 allocation
        mlir::Block& entryBlock = func.getBody().front();
        // for (auto& blockOp : entryBlock) {
        //     if (auto allocOp = mlir::dyn_cast<memref::AllocOp>(&blockOp)) {
        //         if (allocOp.getType() == memrefType) {
        //             // 找到匹配的 allocation，直接复用
        //             llvm::outs() << "Reusing existing allocation: " << allocOp << "\n";
        //             return allocOp.getResult();
        //         }
        //     }
        // }

        // 没有找到现有的 allocation，在函数开头创建新的
        builder.setInsertionPointToStart(&entryBlock);
        return builder.create<memref::AllocOp>(loc, memrefType);
    }

    // 如果没有找到函数，回退到原逻辑
    builder.setInsertionPoint(op);
    return builder.create<memref::AllocOp>(loc, memrefType);
}

// 插入 CUBE -> VECTOR 数据搬运
void DAGSyncPass::insertCubeToVectorDataMovement(mlir::Operation *srcOp, mlir::Operation *dstOp,
                                                mlir::Value srcResult, mlir::OpBuilder &builder,
                                                mlir::Location loc, mlir::Value iterArgs) {
    auto srcTensorType = getTensorType(srcResult);
    if (!srcTensorType) {
        return;
    }

    // 1. 在 srcOp 之后创建 UB 空间的 memref.alloc
    builder.setInsertionPointAfter(srcOp);
    mlir::Value ubAlloc = getOrCreateAllocation(srcOp, srcTensorType,
                                                hivm::AddressSpace::UB, builder, loc);

    // 2. 创建 fixpipe 指令
    builder.setInsertionPointAfter(srcOp);
    FixpipeDMAModeAttr dmaModeAttr = FixpipeDMAModeAttr::get(builder.getContext(), FixpipeDMAMode::NZ2ND);

    auto fixpipeOp = builder.create<hivm::FixpipeOp>(
        loc,
        mlir::TypeRange{}, // 没有返回值
        srcResult,         // src
        ubAlloc,           // dst
        /*unit_flag_cond=*/mlir::ValueRange{},
        /*dma_mode=*/dmaModeAttr,
        /*dual_dst_mode=*/nullptr,
        /*pre_quant=*/nullptr,
        /*pre_relu=*/nullptr,
        /*channel_split=*/nullptr,
        /*unit_flag_mode=*/mlir::ArrayAttr{});

    llvm::outs() << "Inserted fixpipe after " << srcOp->getName().getStringRef()
                 << " for CUBE->VECTOR data movement\n";

    // 3. 在 dstOp 前创建 memory_space_cast 和 to_tensor
    builder.setInsertionPoint(dstOp);

    // memory_space_cast（如果需要）
    mlir::Value plainMemref = ubAlloc;
    auto memrefType = cast<mlir::MemRefType>(ubAlloc.getType());
    if (memrefType.getMemorySpace()) {
        auto plainMemrefType = mlir::MemRefType::get(memrefType.getShape(),
                                                     memrefType.getElementType());
        plainMemref = builder.create<memref::MemorySpaceCastOp>(loc, plainMemrefType, ubAlloc);
        (*valueTypes)[plainMemref] = CoreType::VECTOR_ONLY;
    }

    // 4. 创建 to_tensor
    auto toTensorOp = builder.create<bufferization::ToTensorOp>(
        loc,
        srcTensorType, // 原始的 tensor 类型
        plainMemref,
        /*restrict=*/true,
        /*writable=*/true
    );
    (*valueTypes)[toTensorOp.getResult()] = CoreType::VECTOR_ONLY;

    // 5. 替换 dstOp 的操作数
    if (!iterArgs) {
        replaceOperandWithNewValue(dstOp, srcResult, toTensorOp.getResult());
    } else {
        replaceOperandWithNewValue(dstOp, iterArgs, toTensorOp.getResult());
    }
}

static uint64_t getElemBytesForAlign(Type t) {
  if (auto ft = dyn_cast<FloatType>(t))
    return (uint64_t)((ft.getWidth() + 7) / 8);
  if (auto it = dyn_cast<IntegerType>(t))
    return (uint64_t)((it.getWidth() + 7) / 8);
  if (isa<IndexType>(t))
    return 8ULL;
  if (auto ct = dyn_cast<ComplexType>(t))
    return 2ULL * getElemBytesForAlign(ct.getElementType());
  return 0ULL;
}

static FailureOr<uint64_t> getBlockElemsFor32BAlign(Type elemType) {
  constexpr uint64_t kAlignBytes = 32;
  uint64_t elemBytes = getElemBytesForAlign(elemType);
  if (elemBytes <= 0)
    return failure();
  if (elemBytes >= kAlignBytes)
    return 1;
  if (kAlignBytes % elemBytes != 0)
    return failure();
  return kAlignBytes / elemBytes;
}

static std::optional<SmallVector<int64_t, 4>> newCbubAllocShape(memref::AllocOp allocOp) {
  auto type = dyn_cast<MemRefType>(allocOp.getType());
  // 仅支持静态 2D MemRef
  if (!type || type.getRank() != 2)
    return std::nullopt;

  auto shape = type.getShape();
  int64_t M = shape[0];
  int64_t N = shape[1];
  auto elemType = type.getElementType();
  auto blkOr = getBlockElemsFor32BAlign(elemType);
  int64_t blk = (int64_t)*blkOr;
  // 必须是静态且 16 对齐
  if (ShapedType::isDynamic(M) || ShapedType::isDynamic(N))
    return std::nullopt;
  if (M % 16 != 0)
    return std::nullopt;

  // 新 shape: (N/16, M/16, 16, 16)
  SmallVector<int64_t, 4> newShape = {N / blk, M / 16, 16, blk};

  return newShape;
}

// 修改 VECTOR->CUBE 数据搬运函数
void DAGSyncPass::insertVectorToCubeDataMovement(mlir::Operation *srcOp, mlir::Operation *dstOp, Operation* posOp,
                                                mlir::Value srcResult, mlir::OpBuilder &builder,
                                                mlir::Location loc, llvm::DenseMap<Value, CoreType>* valueMap) {
    auto srcTensorType = getTensorType(srcResult);
    if (!srcTensorType) {
        return;
    }
    if (isa<scf::ForOp>(srcOp) && isa<scf::ForOp>(dstOp)) {
        return;
    }

    // 1. 在 srcOp 之后创建 UB 空间的 memref.alloc（用于 to_memref）
    builder.setInsertionPointAfter(srcOp);

    // 首先创建 UB 空间的 memref type
    auto ubSpaceAttr = hivm::AddressSpaceAttr::get(builder.getContext(), hivm::AddressSpace::UB);
    auto ubMemrefType = mlir::MemRefType::get(srcTensorType.getShape(),
                                              srcTensorType.getElementType(),
                                              /*layout=*/nullptr,
                                              ubSpaceAttr);

    // 创建 bufferization.to_memref
    if (srcOp->getBlock() == dstOp->getBlock()) {
        builder.setInsertionPoint(posOp);
    }
    auto toBufferOp = builder.create<bufferization::ToBufferOp>(
        loc,
        ubMemrefType,
        srcResult
    );

    // 2. 创建 CBUF 空间的 memref.alloc（用于 copy 的目标）
    mlir::Value cbufAllocOld = getOrCreateAllocation(srcOp, srcTensorType,
                                                    hivm::AddressSpace::L1, builder, loc);
    auto cbufShape = *newCbubAllocShape(dyn_cast<memref::AllocOp>(cbufAllocOld.getDefiningOp()));
    // 获取旧的memref类型并创建新的类型
    auto oldType = dyn_cast<MemRefType>(cbufAllocOld.getType());

    // 获取新的维度数量
    unsigned newRank = cbufShape.size();

    // 方法1：创建新的恒等布局映射
    AffineMap identityMap = builder.getMultiDimIdentityMap(newRank);
    MemRefLayoutAttrInterface layout = AffineMapAttr::get(identityMap);

    // 方法2：如果旧类型有布局，尝试调整它（更安全的选择）
    // 先检查旧类型是否有布局
    if (auto oldLayout = oldType.getLayout()) {
        if (auto affineMapAttr = dyn_cast<AffineMapAttr>(oldLayout)) {
            // 如果旧布局是AffineMap，尝试创建新的恒等映射
            // 因为维度改变，旧的affine map可能不再有效
            layout = AffineMapAttr::get(identityMap);
        } else {
            // 对于其他类型的布局，可能需要特殊处理
            layout = oldLayout;
        }
    }

    // 创建新的alloc类型
    auto newAllocType = MemRefType::get(
        cbufShape,
        oldType.getElementType(),
        layout,  // 使用新创建的布局
        oldType.getMemorySpace()
    );

    builder.setInsertionPoint(cbufAllocOld.getDefiningOp());
    // 创建新的alloc操作
    auto cbufAlloc = builder.create<memref::AllocOp>(
        cbufAllocOld.getDefiningOp()->getLoc(),
        newAllocType
    );

    builder.setInsertionPointAfter(toBufferOp);
    // 3. 创建 copy 指令（src 是 ub memref，dst 是 cbuf memref）
    auto copyOp = builder.create<hivm::CopyOp>(
        loc,
        mlir::TypeRange{}, // 没有返回值
        toBufferOp.getResult(),  // src (memref in UB)
        cbufAlloc                // dst (memref in CBUF)
    );

    // llvm::outs() << "Inserted copy after " << srcOp->getName().getStringRef()
    //              << " for VECTOR->CUBE data movement\n";

    // 4. 在 dstOp 前创建 convert_layout
    builder.setInsertionPoint(dstOp);
    auto ndLayout = hivm::DataLayoutAttr::get(builder.getContext(), hivm::DataLayout::ND);
    // 创建 convert_layout
    auto convertLayoutOp = builder.create<hivm::ConvertLayoutOp>(
        loc,
        cbufAllocOld.getType(), // 输出类型与输入相同
        cbufAlloc,
        ndLayout,  // srcLayout
        ndLayout   // dstLayout
    );
    (*valueTypes)[convertLayoutOp.getResult()] = CoreType::CUBE_ONLY;

    // 5. 创建 memory_space_cast
    auto cbufMemrefType = cast<mlir::MemRefType>(convertLayoutOp.getType());
    auto plainMemrefType = mlir::MemRefType::get(cbufMemrefType.getShape(),
                                                 cbufMemrefType.getElementType());

    auto memspaceCastOp = builder.create<memref::MemorySpaceCastOp>(
        loc,
        plainMemrefType,
        convertLayoutOp.getResult()
    );
    (*valueTypes)[memspaceCastOp.getResult()] = CoreType::CUBE_ONLY;

    // 6. 创建 to_tensor
    auto toTensorOp = builder.create<bufferization::ToTensorOp>(
        loc,
        srcTensorType, // 原始的 tensor 类型
        memspaceCastOp.getResult(),
        /*restrict=*/true,
        /*writable=*/true
    );
    (*valueTypes)[toTensorOp.getResult()] = CoreType::CUBE_ONLY;

    // 7. 替换 dstOp 的操作数
    replaceOperandWithNewValue(dstOp, srcResult, toTensorOp.getResult());
}

Operation* DAGSyncPass::FindLastestPosition(Operation* srcOp, Graph &mainGraph, OpBuilder &builder) {
    Operation* insertPos = nullptr;
    auto opMap = mainGraph.getOpMapLegacy();
    auto valueTypes = &mainGraph.getValueTypes();
    // Find the first cube-dependent vector core operation.
    for(auto nextOp = srcOp->getNextNode();nextOp!=nullptr; nextOp=nextOp->getNextNode()) {
        auto nextType = getNodeDeviceType(opMap[nextOp], valueTypes);
        if(nextType == CoreType::CUBE_ONLY) continue;
        // No memref ops in IR yet; directly tracing operands
        for(auto operand: nextOp->getOperands()) {
            auto defOp = operand.getDefiningOp();
            auto defType = getNodeDeviceType(opMap[defOp], valueTypes);
            if(defType == CoreType::CUBE_ONLY) {
                //To prevent UB overflow, we need to break the dependency at the point where the result shape is minimized
                // — i.e., trace upward to find the first broadcast.
                for(auto prevOp = nextOp->getPrevNode(); prevOp != nullptr && prevOp != srcOp; prevOp = prevOp->getPrevNode()) {
                    if(isa<triton::BroadcastOp>(prevOp)) {
                        if(prevOp->getPrevNode() && isa<triton::ExpandDimsOp>(prevOp->getPrevNode())) {
                            return prevOp->getPrevNode();
                        }
                        return prevOp;
                    }
                }
                // Can't find the result shape is minimized
                return nextOp;
            }
        }

        // Once meet SyncBlockWaitOp, return now!
        if(auto waitOp = dyn_cast<hivm::SyncBlockWaitOp>(nextOp)) {
            if(waitOp.getTcoreType() == hivm::TCoreTypeAttr::get(builder.getContext(), hivm::TCoreType::VECTOR)) {
                return nextOp;
            }
        }
        insertPos = nextOp;
    }
    return insertPos;
}

Operation* DAGSyncPass::FindEarliestPosition(Operation* dstOp, Graph &mainGraph, OpBuilder &builder)
{
    auto insertPos = dstOp;
    auto opMap = mainGraph.getOpMapLegacy();
    auto valueTypes = &mainGraph.getValueTypes();
    for (auto prevOp = dstOp->getPrevNode(); prevOp != nullptr; prevOp = prevOp->getPrevNode()) {
        if (dstOp->getBlock() != prevOp->getBlock()) continue;
        // Once meet SyncBlockSetOp, return now!
        if (auto waitOp = dyn_cast<hivm::SyncBlockSetOp>(prevOp)) {
            if (waitOp.getTcoreType() == hivm::TCoreTypeAttr::get(builder.getContext(), hivm::TCoreType::VECTOR)) {
                return insertPos;
            }
        }
        insertPos = prevOp;
    }
    return insertPos;
}

// 主要的同步和数据搬运插入函数
void DAGSyncPass::insertSyncAndMovement(mlir::Operation *srcOp, mlir::Operation *dstOp,
                                       CoreType srcType, CoreType dstType,
                                       mlir::OpBuilder &builder, int flag, llvm::DenseMap<Value, CoreType>* valueMap, Graph &mainGraph) {
    mlir::Location loc = srcOp->getLoc();
    // 保存当前的插入点
    mlir::OpBuilder::InsertionGuard guard(builder);

    // 检查是否是跨 block
    mlir::Block *srcBlock = srcOp->getBlock();
    mlir::Block *dstBlock = dstOp->getBlock();
    bool sameBlock = (srcBlock == dstBlock);

    if (!sameBlock) {
        // 检查是否是外层到内层的依赖
        bool dstIsInnerBlock = false;
        mlir::Operation *dstParentOp = dstBlock->getParentOp();
        while (dstParentOp) {
            if (dstParentOp->getBlock() == srcBlock) {
                dstIsInnerBlock = true;
                break;
            }
            if (dstParentOp->getBlock()) {
                dstParentOp = dstParentOp->getBlock()->getParentOp();
            } else {
                break;
            }
        }

        if (dstIsInnerBlock) {
            insertSyncAndMovementForCrossBlock(srcOp, dstOp, srcType, dstType, builder, flag, true, valueMap, mainGraph);
            return;
        }
    }

    // 同一 block 内的处理
    // 获取 srcOp 的输出（假设第一个结果）
    if (srcOp->getNumResults() == 0) {
        return;
    }
    mlir::Value srcResult = srcOp->getResult(0);

    // CUBE -> VECTOR
    if (srcType == CoreType::CUBE_ONLY && dstType == CoreType::VECTOR_ONLY) {

        // 2. 插入同步指令
        auto coreAttr = hivm::TCoreTypeAttr::get(builder.getContext(), hivm::TCoreType::CUBE);
        auto setPipe = PipeAttr::get(builder.getContext(), hivm::PIPE::PIPE_FIX);
        auto waitPipe = PipeAttr::get(builder.getContext(), hivm::PIPE::PIPE_V);
        auto lastSetPipe = PipeAttr::get(builder.getContext(), hivm::PIPE::PIPE_MTE3);
        auto lastWaitPipe = PipeAttr::get(builder.getContext(), hivm::PIPE::PIPE_MTE1);
        auto flagId = builder.getIntegerAttr(builder.getI64Type(), flag);
        auto flagAddId = builder.getIntegerAttr(builder.getI64Type(), flag * 2);
        auto lastFlagAddId = builder.getIntegerAttr(builder.getI64Type(), (flag - 1) * 2);

        // set 在 srcOp 后
        builder.setInsertionPointAfter(srcOp);
        builder.create<SyncBlockSetOp>(loc, coreAttr, setPipe, waitPipe, flagId);

        // wait 在 dstOp 前

        auto posOp = FindEarliestPosition(dstOp, mainGraph, builder);
        builder.setInsertionPoint(posOp);
        coreAttr = hivm::TCoreTypeAttr::get(builder.getContext(), hivm::TCoreType::VECTOR);
        builder.create<SyncBlockWaitOp>(loc, coreAttr, setPipe, waitPipe, flagId);

        // 1. 插入数据搬运
        insertCubeToVectorDataMovement(srcOp, dstOp, srcResult, builder, loc, nullptr);

        // llvm::outs() << "Inserted CUBE->VECTOR sync and data movement (flag=" << flag << ")\n";
    }
    // VECTOR -> CUBE
    else if (srcType == CoreType::VECTOR_ONLY && dstType == CoreType::CUBE_ONLY) {

        // 2. 插入同步指令
        auto coreAttr = hivm::TCoreTypeAttr::get(builder.getContext(), hivm::TCoreType::VECTOR);
        auto setPipe = PipeAttr::get(builder.getContext(), hivm::PIPE::PIPE_MTE3);
        auto waitPipe = PipeAttr::get(builder.getContext(), hivm::PIPE::PIPE_MTE1);
        auto lastSetPipe = PipeAttr::get(builder.getContext(), hivm::PIPE::PIPE_FIX);
        auto lastWaitPipe = PipeAttr::get(builder.getContext(), hivm::PIPE::PIPE_V);
        auto flagId = builder.getIntegerAttr(builder.getI64Type(), flag);
        auto flagAddId = builder.getIntegerAttr(builder.getI64Type(), flag * 2);
        auto lastFlagAddId = builder.getIntegerAttr(builder.getI64Type(), (flag - 1) * 2);

        // set 在 srcOp 后
        // builder.setInsertionPointAfter(srcOp);
        auto posOp = FindLastestPosition(srcOp, mainGraph, builder);
        if (posOp) {
            builder.setInsertionPoint(posOp);
        } else {
            builder.setInsertionPointAfter(srcOp);
        }
        auto setOp = builder.create<SyncBlockSetOp>(loc, coreAttr, setPipe, waitPipe, flagId);

        // wait 在 dstOp 前
        builder.setInsertionPoint(dstOp);
        coreAttr = hivm::TCoreTypeAttr::get(builder.getContext(), hivm::TCoreType::CUBE);
        builder.create<SyncBlockWaitOp>(loc, coreAttr, setPipe, waitPipe, flagId);

        // 1. 插入数据搬运
        insertVectorToCubeDataMovement(srcOp, dstOp, setOp, srcResult, builder, loc, valueMap);

        // llvm::outs() << "Inserted VECTOR->CUBE sync and data movement (flag=" << flag << ")\n";
    }
}

// 跨 block 的同步和数据搬运
void DAGSyncPass::insertSyncAndMovementForCrossBlock(mlir::Operation *srcOp, mlir::Operation *dstOp,
                                                    CoreType srcType, CoreType dstType,
                                                    mlir::OpBuilder &builder, int flag,
                                                    bool dstIsInnerBlock, llvm::DenseMap<Value, CoreType>* valueMap, Graph &mainGraph) {
    if (!dstIsInnerBlock) {
        insertSyncAndMovement(srcOp, dstOp, srcType, dstType, builder, flag, valueMap, mainGraph);
        return;
    }

    mlir::Location loc = srcOp->getLoc();
    mlir::Block *dstBlock = dstOp->getBlock();

    // 获取 srcOp 的输出
    if (srcOp->getNumResults() == 0) {
        return;
    }
    mlir::Value srcResult = srcOp->getResult(0);

    // CUBE -> VECTOR
    if (srcType == CoreType::CUBE_ONLY && dstType == CoreType::VECTOR_ONLY) {

        // 2. 插入同步指令（跨 block 特殊处理）
        auto coreAttr = hivm::TCoreTypeAttr::get(builder.getContext(), hivm::TCoreType::CUBE);
        auto setPipe = PipeAttr::get(builder.getContext(), hivm::PIPE::PIPE_FIX);
        auto waitPipe = PipeAttr::get(builder.getContext(), hivm::PIPE::PIPE_V);
        auto flagId = builder.getIntegerAttr(builder.getI64Type(), flag);

        // set 在 srcOp 后（外层）
        builder.setInsertionPointAfter(srcOp);
        builder.create<SyncBlockSetOp>(loc, coreAttr, setPipe, waitPipe, flagId);

        // 1. 插入数据搬运（同 block 内逻辑）
        insertCubeToVectorDataMovement(srcOp, dstOp, srcResult, builder, loc, nullptr);

        // wait 在内层 block 入口前
        mlir::Operation *parentOp = dstBlock->getParentOp();
        while (srcOp->getBlock() != parentOp->getBlock()) {
            parentOp = parentOp->getBlock()->getParentOp();
        }
        if (parentOp) {
            builder.setInsertionPoint(parentOp);
            coreAttr = hivm::TCoreTypeAttr::get(builder.getContext(), hivm::TCoreType::VECTOR);
            builder.create<SyncBlockWaitOp>(loc, coreAttr, setPipe, waitPipe, flagId);
        } else {
            builder.setInsertionPoint(dstOp);
            coreAttr = hivm::TCoreTypeAttr::get(builder.getContext(), hivm::TCoreType::VECTOR);
            builder.create<SyncBlockWaitOp>(loc, coreAttr, setPipe, waitPipe, flagId);
        }

    }
    // VECTOR -> CUBE
    else if (srcType == CoreType::VECTOR_ONLY && dstType == CoreType::CUBE_ONLY) {

        // 2. 插入同步指令（跨 block 特殊处理）
        auto coreAttr = hivm::TCoreTypeAttr::get(builder.getContext(), hivm::TCoreType::VECTOR);
        auto setPipe = PipeAttr::get(builder.getContext(), hivm::PIPE::PIPE_MTE3);
        auto waitPipe = PipeAttr::get(builder.getContext(), hivm::PIPE::PIPE_MTE1);
        auto flagId = builder.getIntegerAttr(builder.getI64Type(), flag);

        // set 在 srcOp 后（外层）
        builder.setInsertionPointAfter(srcOp);
        builder.create<SyncBlockSetOp>(loc, coreAttr, setPipe, waitPipe, flagId);

        // 1. 插入数据搬运（同 block 内逻辑）
        insertVectorToCubeDataMovement(srcOp, dstOp, srcOp, srcResult, builder, loc, valueMap);

        // wait 在内层 block 入口前
        mlir::Operation *parentOp = dstBlock->getParentOp();
        if (parentOp) {
            builder.setInsertionPoint(parentOp);
            coreAttr = hivm::TCoreTypeAttr::get(builder.getContext(), hivm::TCoreType::CUBE);
            builder.create<SyncBlockWaitOp>(loc, coreAttr, setPipe, waitPipe, flagId);
        } else {
            builder.setInsertionPoint(dstOp);
            coreAttr = hivm::TCoreTypeAttr::get(builder.getContext(), hivm::TCoreType::CUBE);
            builder.create<SyncBlockWaitOp>(loc, coreAttr, setPipe, waitPipe, flagId);
        }

        // llvm::outs() << "Inserted cross-block VECTOR->CUBE sync and data movement (flag=" << flag << ")\n";
    }
}

void LegalizeDot(triton::FuncOp funcOp) {
    mlir::OpBuilder builder(funcOp);
    funcOp.walk([&](triton::DotOp dotOp) {
        // 获取dot操作的输入
      Value a = dotOp.getOperands()[0];
      Value b = dotOp.getOperands()[1];
      Value c = dotOp.getOperands()[2];  // 累加器参数

      // 检查累加器是否为全零常量
      bool isZeroAccumulator = false;

      // 检查是否直接是arith.constant 0
      if (auto constantOp = c.getDefiningOp<arith::ConstantOp>()) {
        if (auto denseAttr = dyn_cast<DenseElementsAttr>(constantOp.getValue())) {
          if (denseAttr.isSplat() && denseAttr.getSplatValue<FloatAttr>().getValueAsDouble() == 0.0) {
            isZeroAccumulator = true;
          }
        }
      }

      if (!isZeroAccumulator) {
        // 创建新的零累加器
        Location loc = dotOp.getLoc();
        auto resultType = dotOp.getResult().getType();

        Value originalResult = dotOp.getResult();
        builder.setInsertionPoint(dotOp);
        // 创建全零张量
        auto zeroAttr = DenseElementsAttr::get(
            dyn_cast<RankedTensorType>(resultType),
            APFloat(0.0f));
        auto zeroConstant = builder.create<arith::ConstantOp>(loc, zeroAttr);

        // 创建新的dot操作，使用零作为累加器
        auto newDot = builder.create<triton::DotOp>(
            loc, resultType, a, b, zeroConstant);

        // 创建加法操作，将新的dot结果与原来的累加器c相加
        auto addOp = builder.create<arith::AddFOp>(loc, newDot, c);

        // 用addOp替换原来的dotOp
        originalResult.replaceAllUsesWith(addOp.getResult());

        // 删除原dotOp（如果它没有其他用途）
        if (dotOp.use_empty()) {
            dotOp.erase();
        }
      }

    });
}

static void rewriteCopyChainForCbub(
    hivm::CopyOp copyOp,
    ArrayRef<int64_t> newShape,
    OpBuilder &builder) {

  // 获取 copy 的输入（ins），应为 to_memref 的结果
  Value insVal = copyOp.getOperands()[0];
  auto toBufferOp = insVal.getDefiningOp<bufferization::ToBufferOp>();
  if (!toBufferOp)
    return;

  Value inputTensor = toBufferOp.getTensor();
  auto inputTensorType = dyn_cast<RankedTensorType>(inputTensor.getType());
  if (!inputTensorType || inputTensorType.getRank() != 2)
    return;

  // blk = 32/位宽
  // 中间 reshape 形状：[M/16, 16, N/ blk, blk]
  int64_t M = inputTensorType.getShape()[0];
  int64_t N = inputTensorType.getShape()[1];
  auto elemType = inputTensorType.getElementType();
  auto blkOr = getBlockElemsFor32BAlign(elemType);
  int64_t blk = (int64_t)*blkOr;
  SmallVector<int64_t, 3> intermediateShape3D = {M, N / blk, blk};
  SmallVector<int64_t, 3> intermediateShapetrans = {N / blk, M, blk};
  auto elementType = inputTensorType.getElementType();
  auto interTensor3DType = RankedTensorType::get(intermediateShape3D, elementType);
  auto interTensortransType = RankedTensorType::get(intermediateShapetrans, elementType);

  auto finalTensorType = RankedTensorType::get(newShape, elementType);

  auto loc = inputTensor.getLoc();

  // Set insertion point before copyOp (or toBufferOp)
  auto tensorOp = inputTensor.getDefiningOp();
  builder.setInsertionPointAfter(tensorOp);

  // 插入 triton.reshape 将 2D tensor 展开为 3D
  auto reshape3DOp = builder.create<triton::ReshapeOp>(
      loc, interTensor3DType, inputTensor);
  (*valueTypes)[reshape3DOp.getResult()] = CoreType::VECTOR_ONLY;

  // nark tiling dim for reshapeop
  auto markOp3d = builder.create<annotation::MarkOp>(loc, reshape3DOp);
  auto tilingDimAttr3d = builder.getDictionaryAttr(SmallVector<NamedAttribute>{
    NamedAttribute(builder.getStringAttr("1"), builder.getIndexAttr(1))});
  markOp3d->setAttr("tiling_dim_mapping", tilingDimAttr3d);

  // 插入 triton.trans 调整维度顺序 Insert tt.trans {order = [1, 0, 2]}
  SmallVector<int32_t, 4> order = {1, 0, 2};
  auto orderAttr = builder.getDenseI32ArrayAttr(order);  // OpBuilder supports this
  auto transOp = builder.create<triton::TransOp>(
      loc, interTensortransType, reshape3DOp.getResult(), orderAttr);
  (*valueTypes)[transOp.getResult()] = CoreType::VECTOR_ONLY;

  // 插入 triton.reshape 将 3D tensor 展开为 4D
  auto reshape4DOp = builder.create<triton::ReshapeOp>(
      loc, finalTensorType, transOp.getResult());
  (*valueTypes)[reshape4DOp.getResult()] = CoreType::VECTOR_ONLY;

  // nark tiling dim for reshapeop
  auto markOp4d = builder.create<annotation::MarkOp>(loc, reshape4DOp);
  auto tilingDimAttr4d = builder.getDictionaryAttr(SmallVector<NamedAttribute>{
    NamedAttribute(builder.getStringAttr("1"), builder.getIndexAttr(1))});
  markOp4d->setAttr("tiling_dim_mapping", tilingDimAttr4d);

  // Create new to_memref
  builder.setInsertionPoint(toBufferOp);
  auto oldBufferType = cast<BaseMemRefType>(toBufferOp.getResult().getType());
  auto newMemRefType = MemRefType::get(
      newShape,
      elementType,
      mlir::AffineMap{},
      oldBufferType.getMemorySpace());
  auto newtoBufferOp = builder.create<bufferization::ToBufferOp>(
      toBufferOp.getLoc(),
      newMemRefType,
      reshape4DOp.getResult());
  (*valueTypes)[newtoBufferOp.getResult()] = CoreType::VECTOR_ONLY;

  // Create NEW copyOp (replacing the old one)
  builder.setInsertionPoint(copyOp);
  auto resultTypes = copyOp->getResultTypes();
  auto newCopyOp = builder.create<hivm::CopyOp>(
      copyOp.getLoc(),
      resultTypes,                      // TypeRange
      reshape4DOp.getResult(),        // src (ins)
      copyOp.getOperands()[1]           // dst (outs)
  );

  // 替换 uses 并清理旧 op
  copyOp.replaceAllUsesWith(newCopyOp);
  copyOp.erase();
  toBufferOp.erase();

  return;
}

bool valueIsPtrOrShapedPtr(Value value)
{
    auto type = value.getType();
    if (llvm::isa<triton::PointerType>(type)) {
        return true;
    }

    if (auto tensorType = llvm::dyn_cast<ShapedType>(type)) {
        return llvm::isa<triton::PointerType>(tensorType.getElementType());
    }

    return false;
}

using AliasSet = llvm::SmallBitVector;
using AliasInfo = llvm::DenseMap<Value, AliasSet>;
constexpr size_t EXPECTED_MAX_ROOT_PTR_COUNT = 16;

/**
 * Reworked alias analysis inplace of triton::SharedMemoryAliasAnalysis or mlir::AliasAnalysis
 * where both failed to establish aliases from function arguments (i.e. global memory):
 * 1. triton::SharedMemoryAliasAnalysis - does not reach global memory
 * 2. mlir::AliasAnalysis - marks all function arguments as MayAlias or PartialAlias;
 *      we may be able to get around by marking all arguments as noalias, but it may affect other analyses...
 */
AliasInfo getAlias(triton::FuncOp funcOp, Graph::ValueMapRaw& valueMap)
{
    AliasInfo aliasInfo;

    struct WorkNode {
        Value value;
        Value cause;
    };

    llvm::SmallVector<WorkNode, 0> worklist;

    auto addDownstreamToWorklist = [&worklist](ValueNode* node) {
        if (node == nullptr) {
            return;
        }
        for (auto *output : node->getOutputs()) {
            if (auto *opNode = dyn_cast<OpNode>(output)) {
                for (auto result : opNode->op->getResults()) {
                    if (valueIsPtrOrShapedPtr(result)) {
                        worklist.push_back({
                            result,
                            node->value
                        });
                    }
                }
            } else if (auto *valueNode = dyn_cast<ValueNode>(output)) {
                worklist.push_back({
                    valueNode->value,
                    node->value
                });
            }
        }
    };

    SmallVector<Value, EXPECTED_MAX_ROOT_PTR_COUNT> rootPtrs;

    for (auto arg : funcOp.getBody().getArguments()) {
        if (valueIsPtrOrShapedPtr(arg)) {
            rootPtrs.push_back(arg);
        }
    }
    funcOp.walk([&](Operation* op) {
        // If an op generates ptr from nowhere (e.g. tt.empty()), then we consider it as a root
        bool generatesRootPtr = !llvm::any_of(op->getOperands(), valueIsPtrOrShapedPtr);
        if (generatesRootPtr) {
            for (auto result : op->getResults()) {
                if (valueIsPtrOrShapedPtr(result)) {
                    rootPtrs.push_back(result);
                }
            }
        }
    });

    size_t numRootPtrs = rootPtrs.size();

    for (auto [index, value] : llvm::enumerate(rootPtrs)) {
        auto& bitset = aliasInfo[value];
        bitset.resize(numRootPtrs);
        bitset[index] = true;

        addDownstreamToWorklist(getFromSmartPtr(valueMap, value));
    }

    while (!worklist.empty()) {
        auto [value, upstream] = worklist.pop_back_val();
        if (!valueIsPtrOrShapedPtr(value)) {
            continue;
        }
        auto node = getFromSmartPtr(valueMap, value);
        if (!node) {
            continue;
        }

        /**
         * Safety:
         * 1. aliasInfo[value] : may insert into densemap and cause resize
         * 2. aliasInfo[upstream] : steady reference without any insertion into the densemap, will not cause resize, since upstream is in the map
         * 3. after aliasInfo[upstream] : currAlias is still valid
         */
        auto& currAlias = aliasInfo[value];
        auto& causeAlias = aliasInfo[upstream];

        if (!causeAlias.test(currAlias)) {
            continue;
        }

        currAlias |= causeAlias;
        addDownstreamToWorklist(node);
    }
    return aliasInfo;
}

/**
 * @returns whether two values share any root ptr
 */
bool mayAlias(AliasInfo &aliasInfo, Value valueA, Value valueB)
{
    auto aliasesA = getPtr(aliasInfo, valueA);
    auto aliasesB = getPtr(aliasInfo, valueB);
    if (
        !valueIsPtrOrShapedPtr(valueA) ||
        !valueIsPtrOrShapedPtr(valueB) ||
        aliasesA == nullptr ||
        aliasesB == nullptr
    ) {
        return false;
    }

    return aliasesA->anyCommon(*aliasesB);
}

template <typename OpTy>
OpTy createBlockSync(OpBuilder builder,
                     hivm::TCoreType coreType,
                     hivm::PIPE srcPipe,
                     hivm::PIPE dstPipe,
                     int flag,
                     Operation *cause)
{
    auto flagId = builder.getIntegerAttr(builder.getI64Type(), flag);
    auto coreAttr = hivm::TCoreTypeAttr::get(builder.getContext(), coreType);
    auto setPipe = PipeAttr::get(builder.getContext(), srcPipe);
    auto waitPipe = PipeAttr::get(builder.getContext(), dstPipe);
    return builder.create<OpTy>(cause->getLoc(), coreAttr, setPipe, waitPipe, flagId);
}

const size_t MAX_EXPECTED_PARENTS_COUNT = 8;
std::optional<std::pair<Operation *, Operation *>> findAncestorCommonBlock(mlir::Operation *opA, mlir::Operation *opB)
{
    if (opA->getBlock() == opB->getBlock()) {
        return std::make_pair(opA, opB);
    }

    // record all ancestors of opA
    llvm::SmallPtrSet<mlir::Operation *, MAX_EXPECTED_PARENTS_COUNT> ancestorsA;
    mlir::Operation *curr = opA;
    while (curr) {
        ancestorsA.insert(curr);
        curr = curr->getParentOp();
    }

    // find the last ancestor of opB which is also the ancestor of opA
    mlir::Operation *commonAncOp = nullptr;
    curr = opB;
    while (curr) {
        if (ancestorsA.count(curr)) {
            commonAncOp = curr;
            break;
        }
        curr = curr->getParentOp();
    }

    if (!commonAncOp) {
        return std::nullopt;
    }

    // find the ancestors in the given region
    for (mlir::Region &region : commonAncOp->getRegions()) {
        for (mlir::Block &block : region) {
            auto *ancA = block.findAncestorOpInBlock(*opA);
            auto *ancB = block.findAncestorOpInBlock(*opB);
            if (ancA && ancB) {
                return std::make_pair(ancA, ancB);
            }
        }
    }
    return std::nullopt;
}

struct SyncCandidate {
    CoreType srcCoreType;
    Operation *setCause;
    Operation *setAfter;
    Operation *waitCause;
    Operation *waitBefore;
};

// setOp, waitOp
std::pair<SyncBlockSetOp, SyncBlockWaitOp> createBlockSyncBetween(OpBuilder builder,
                                                                  hivm::PIPE srcPipe,
                                                                  hivm::PIPE dstPipe,
                                                                  SyncCandidate candidate,
                                                                  int flag)
{
    auto srcCoreType = toHivm(candidate.srcCoreType);
    auto dstCoreType = toHivm(!candidate.srcCoreType);

    builder.setInsertionPointAfter(candidate.setAfter);
    auto setOp = createBlockSync<SyncBlockSetOp>(builder, srcCoreType, srcPipe, dstPipe, flag, candidate.setCause);
    builder.setInsertionPoint(candidate.waitBefore);
    auto waitOp = createBlockSync<SyncBlockWaitOp>(builder, dstCoreType, srcPipe, dstPipe, flag, candidate.waitCause);
    return {setOp, waitOp};
};

/**
 * @returns the source/destination of a memory op
 *
 * Strangely, triton memory ops do not register their pointers to memory effects, so we need some special cases as fallback here
 */
Value getMemoryAddress(Operation *op, MemoryEffects::EffectInstance effect)
{
    if (mlir::Value v = effect.getValue())
        return v;

    return TypeSwitch<Operation*, Value>(op)
        .Case<
            triton::StoreOp,
            triton::LoadOp,
            triton::AtomicRMWOp,
            triton::AtomicCASOp
        >([](auto op) {
            return op.getPtr();
        })
        .Default([](auto) {
            return nullptr;
        });
}

void addMemEffectsSync(triton::FuncOp funcOp, Graph *graph, OpBuilder &builder, int &syncFlag)
{
    DominanceInfo domInfo(funcOp);
    PostDominanceInfo postDomInfo(funcOp);

    // [(node, EffectInstance, LinearisationPt)]
    llvm::SmallVector<std::tuple<OpNode *, MemoryEffects::EffectInstance>> memOps;

    // [(setAfter, waitBefore, srcOP, dstOp)][CoreType]
    llvm::SmallVector<SyncCandidate> candidates;
    llvm::SmallVector<SyncCandidate> backwardCandidates;

    auto aliasInfo = getAlias(funcOp, graph->getValueMap());

    funcOp.walk([&](MemoryEffectOpInterface memIface) {
        auto *op = memIface.getOperation();
        if (llvm::isa<triton::AssertOp>(op)) {
            return;
        }

        auto *currNode = graph->getOpMap()[op].get();
        SmallVector<MemoryEffects::EffectInstance> effects;

        memIface.getEffects(effects);

        for (auto &effect : effects) {
            if (!isa<MemoryEffects::Write, MemoryEffects::Read>(effect.getEffect())) {
                continue;
            }
            auto currVal = getMemoryAddress(op, effect);
            if (!currVal) {
                op->emitWarning("Mem effect src/dst is unknown!");
                continue;
            }
            memOps.emplace_back(currNode, effect);
            bool isWrite = isa<MemoryEffects::Write>(effect.getEffect());
            for (auto &[prevNode, prevEffect] : memOps) {
                auto prevVal = getMemoryAddress(prevNode->op, prevEffect);
                if ((isa<MemoryEffects::Write>(prevEffect.getEffect()) || isWrite) &&
                    mayAlias(aliasInfo, prevVal, currVal) &&
                    prevNode->isOn() != currNode->isOn() // write is forced on single core type, so we are safe to judge
                                                         // based on whether the core types are different
                ) {
                    CoreType srcCoreType = isWrite ? !currNode->isOn() : prevNode->isOn();
                    CoreType dstCoreType = !srcCoreType;
                    auto opPair = findAncestorCommonBlock(prevNode->op, currNode->op);
                    if (!opPair.has_value()) {
                        op->emitWarning(
                            llvm::formatv("Unable to find ancestors in common block with {0}\n", *prevNode->op));
                        continue;
                    }
                    auto [setAfter, waitBefore] = opPair.value();
                    if (setAfter == waitBefore) {
                        continue;
                    }
                    candidates.push_back(SyncCandidate {srcCoreType, prevNode->op, setAfter, op, waitBefore});

                    if (op->getParentOfType<LoopLikeOpInterface>()) { // need to insert backward sync
                        backwardCandidates.push_back(SyncCandidate {
                            dstCoreType,
                            op,
                            waitBefore,
                            prevNode->op,
                            setAfter
                        });
                    }
                }
            }
        }
    });

    /**
     * Adds block sync operations for the candidate
     * Specially, for backward syncs, we also perform a kick-start for first wait and clean-up to avoid affecting possible future syncs
     */
    auto addBlockSyncCommon = [&builder, &syncFlag, &aliasInfo](SyncCandidate cand) {
        LLVM_DEBUG(llvm::dbgs() << "\n\n=== Insert sync between ===\n"
                     << *cand.setAfter << "\n"
                     << "Cause: " << *cand.setCause << "\n");

        LLVM_DEBUG(llvm::dbgs() << "----------" << "\n"
                     << *cand.waitBefore << "\n"
                     << "Cause: " << *cand.waitCause << "\n");

        LLVM_DEBUG(llvm::dbgs() << "=== Insert Sync End ===\n\n");

        auto srcPipe = cand.srcCoreType == CoreType::CUBE_ONLY ? hivm::PIPE::PIPE_FIX : hivm::PIPE::PIPE_MTE2;
        auto dstPipe = hivm::PIPE::PIPE_S;
        auto [forwardSet, forwardWait] = createBlockSyncBetween(builder, srcPipe, dstPipe, cand, syncFlag % MAX_FLAG_ID);
        forwardSet->setAttr("ssbuf.flagid", builder.getUI32IntegerAttr(syncFlag));
        forwardWait->setAttr("ssbuf.flagid", builder.getUI32IntegerAttr(syncFlag));

        auto loopAncestor = cand.waitBefore->getParentOfType<LoopLikeOpInterface>();
        if (loopAncestor) { // backward sync, need to kick-start and clean-up; only need to ensure
                                                  // the first and last one, because we ensure once a flag is consumed,
                                                  // it will be set by construction (wait-set in the same block)
            while (auto newAncestor = loopAncestor->getParentOfType<LoopLikeOpInterface>()) {
                loopAncestor = newAncestor;
            }

            auto srcPipe = cand.srcCoreType != CoreType::CUBE_ONLY ? hivm::PIPE::PIPE_FIX : hivm::PIPE::PIPE_MTE2;
            auto backendFlag = syncFlag;
            syncFlag++;
            auto backwardCandidate = SyncCandidate {
                !cand.srcCoreType,
                cand.waitCause,
                cand.waitBefore,
                cand.setCause,
                cand.setAfter
            };
            auto [backwardSet, backwardWait] = createBlockSyncBetween(builder, srcPipe, dstPipe,
                backwardCandidate, syncFlag % MAX_FLAG_ID);

            backwardSet->setAttr("ssbuf.backward", builder.getUnitAttr());
            backwardWait->setAttr("ssbuf.backward", builder.getUnitAttr());
            backwardSet->setAttr("ssbuf.flagid", builder.getUI32IntegerAttr(backendFlag));
            backwardWait->setAttr("ssbuf.flagid", builder.getUI32IntegerAttr(backendFlag));

            builder.setInsertionPoint(loopAncestor);
            auto kickstart = createBlockSync<SyncBlockSetOp>(
                builder, toHivm(backwardCandidate.srcCoreType), srcPipe, dstPipe, syncFlag, backwardCandidate.setCause);

            builder.setInsertionPointAfter(loopAncestor);
            auto cleanup = createBlockSync<SyncBlockWaitOp>(
                builder, toHivm(!backwardCandidate.srcCoreType), srcPipe, dstPipe, syncFlag, backwardCandidate.waitCause);
        }

        syncFlag++;
    };

    if (candidates.empty()) {
        return;
    }

    auto setAfterDominate = [&domInfo](Operation *a, Operation *b) {
        if (domInfo.dominates(a, b)) {
            return true;
        }
        if (domInfo.dominates(b, a)) {
            return false;
        }
        if (a->isAncestor(b)) {
            return false;
        }
        if (b->isAncestor(a)) {
            return true;
        }
        return false;
    };

    auto waitBeforePostDominate = [&postDomInfo](Operation *a, Operation *b) {
        if (postDomInfo.postDominates(a, b)) {
            return true;
        }
        if (postDomInfo.postDominates(b, a)) {
            return false;
        }
        if (a->isAncestor(b)) {
            return true;
        }
        if (b->isAncestor(a)) {
            return false;
        }
        return false;
    };

    /**
     * Sorts the candidates by (setAfter, waitBefore), and then removes redundant sync candidates that fully contains another
     */
    auto selectCandidatesToInsertSync = [&setAfterDominate, &waitBeforePostDominate, &addBlockSyncCommon](
                                            llvm::SmallVectorImpl<SyncCandidate>& candidates) {
        llvm::sort(candidates, [&](const SyncCandidate &a, const SyncCandidate &b) {
            if (a.setAfter != b.setAfter) {
                return setAfterDominate(a.setAfter, b.setAfter);
            }

            if (a.waitBefore != b.waitBefore) {
                return waitBeforePostDominate(a.waitBefore, b.waitBefore);
            }

            return false;
        });

        for (auto [i, cand] : llvm::enumerate(candidates)) {
            bool shouldInsert = true;
            for (auto otherCand : ArrayRef(candidates).drop_front(i + 1)) {
                bool duplicated = (cand.waitBefore == otherCand.waitBefore && cand.setAfter == otherCand.setAfter &&
                                   cand.srcCoreType == otherCand.srcCoreType);
                bool containsOther =
                    (cand.srcCoreType == otherCand.srcCoreType && setAfterDominate(cand.setAfter, otherCand.setAfter) &&
                     waitBeforePostDominate(cand.waitBefore, otherCand.waitBefore));
                if (duplicated || containsOther) {
                    shouldInsert = false;
                    break;
                }
            }

            if (shouldInsert) {
                addBlockSyncCommon(cand);
            }
        }
    };

    selectCandidatesToInsertSync(candidates);
}

void DAGSyncPass::runOnOperation()
{
    auto module = getOperation();
    mlir::OpBuilder builder(&getContext());

    // 遍历所有函数
    for (auto funcOp : llvm::make_early_inc_range(module.getOps<triton::FuncOp>())) {
        // 跳过无效函数
        LegalizeDot(funcOp);
        if (funcOp.getBody().empty()) {
            continue;
        }

        // llvm::outs() << "\n====================================\n";
        // llvm::outs() << "处理函数: " << funcOp.getName() << "\n";
        // llvm::outs() << "====================================\n";

        auto unique_graph = Graph::fromMultiBlockFunc(funcOp);
        std::shared_ptr<Graph> shared_graph = std::move(unique_graph);
        auto& main_graph = *shared_graph;

        auto funcName = funcOp.getName();

        // 获取 DAG 图的映射
        auto opMapRaw = main_graph.getOpMapLegacy();
        valueTypes = &main_graph.getValueTypes();
        auto *opMap = &opMapRaw;
        for (const auto& pair : *opMap) {
            Operation* op = pair.first;  // 键：Operation 指针
            Node* node = pair.second;    // 值：Node 指针

            // 打印指针地址（最直接的方式）
            // llvm::outs() << "Operation*: " << *op
            //              << "\n";
            for (auto res : op->getResults()) {
                 llvm::outs() << "Value: " << (*valueTypes)[res]
                         << "\n";
            }

        }

        if (!opMap) {
            llvm::errs() << "Warning: Failed to create DAG graph for function " << funcOp.getName() << "\n";
            continue;
        }

        // 用于避免重复插入同步
        llvm::DenseSet<std::pair<mlir::Operation *, mlir::Operation *>> processedPairs;
        int syncFlag = 1;
        addMemEffectsSync(funcOp, shared_graph.get(), builder, syncFlag);

        // 3. 使用 walk 遍历函数中的所有操作
        funcOp.walk([&](mlir::Operation *op) {
            // 查找当前操作对应的 Node
            auto nodeIt = opMap->find(op);
            if (nodeIt == opMap->end()) {
                // 这个操作不在 entry block 的 DAG 图中
                // 可能是嵌套在控制流内部的操作
                return;
            }

            OpNode *currentNode = nodeIt->second;

            // 检查是否是 scf.for 操作
            if (auto forOp = mlir::dyn_cast<mlir::scf::ForOp>(op)) {
                // 处理 scf.for 循环的特殊同步逻辑
                int temp = syncFlag % MAX_FLAG_ID;
                processScfForSync(forOp, currentNode, valueTypes, builder, temp);
            }

            // 获取当前节点的设备类型
            CoreType currentType = getNodeDeviceType(currentNode, valueTypes);

            // 打印操作信息（可选）
            // if (!llvm::isa<scf::SCFDialect>(op->getDialect())) {
            //     llvm::outs() << "操作: " << *op
            //              << " 设备类型: "
            //              << (currentType == CoreType::VECTOR_ONLY ? "VECTOR" :
            //                  currentType == CoreType::CUBE_ONLY ? "CUBE" : "SCALAR")
            //              << "\n";
            // }

            // 4. 遍历当前节点的所有输入节点
            for (ValueNode *inputValNode : currentNode->getInputs()) {
                auto inputOp = inputValNode->value.getDefiningOp();
                if (!inputOp && opMap->contains(inputOp)) {
                    continue;
                }

                auto inputNode = (*opMap)[inputOp];

                // 获取输入节点的设备类型
                CoreType inputType = getNodeDeviceType(inputNode, valueTypes);

                // 5. 判断是否需要插入同步和数据搬运
                if (needVectorCubeSync(inputType, currentType)) {
                    // 检查是否已经处理过这对操作
                    auto opPair = std::make_pair(inputOp, op);
                    if (processedPairs.insert(opPair).second) {
                        // 插入同步和数据搬运指令
                        // 检查是否是跨 block 的依赖
                        mlir::Block *srcBlock = inputOp->getBlock();
                        mlir::Block *dstBlock = op->getBlock();

                        if (srcBlock == dstBlock) {
                            // 同一 block 内
                            insertSyncAndMovement(inputOp, op, inputType, currentType, builder, syncFlag % 14, valueTypes, main_graph);
                            syncFlag ++;
                        } else {
                            // 跨 block，判断是否是外层到内层
                            llvm::outs() << "#########\n";
                            bool dstIsInnerBlock = false;
                            mlir::Operation *dstParentOp = dstBlock->getParentOp();

                            // 向上查找，看 dstBlock 是否在 srcBlock 的区域内
                            while (dstParentOp) {
                                if (dstParentOp->getBlock() == srcBlock) {
                                    dstIsInnerBlock = true;
                                    break;
                                }
                                if (dstParentOp->getBlock()) {
                                    dstParentOp = dstParentOp->getBlock()->getParentOp();
                                } else {
                                    break;
                                }
                            }
                            if (dstIsInnerBlock) {

                                insertSyncAndMovementForCrossBlock(inputOp, op, inputType, currentType,
                                                              builder, syncFlag % 14, dstIsInnerBlock, valueTypes, main_graph);
                                syncFlag ++;
                            }
                        }
                    }
                }
            }
        });

        // llvm::outs() << "\n函数 " << funcOp.getName() << " 统计:\n";
        // llvm::outs() << "  - 插入的总同步操作数: " << syncFlag << "\n";
        funcOp.walk([&](hivm::CopyOp copyOp) {
            llvm::outs()<<copyOp<<"  sss\n\n\n\n";
            rewriteCopyChainForCbub(copyOp, dyn_cast<MemRefType>(copyOp.getOperands()[1].getType()).getShape(), builder);
        });
        GraphManager::getInstance().registerGraph(funcName, shared_graph);
    }

    // llvm::outs()<<module<<"  after dag sync\n\n\n";
    // llvm::outs()<<module<<"  after dag sync\n\n\n";
    // llvm::outs()<<module<<"  after dag sync\n\n\n";
    // llvm::outs()<<module<<"  after dag sync\n\n\n";

}


std::unique_ptr<OperationPass<ModuleOp>> mlir::triton::createDAGSyncPass()
{
    return std::make_unique<DAGSyncPass>();
}
