#include <memory>

#include "epu/memory.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotAnalysis.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/Passes.h"
#include "triton-shared/Conversion/StructuredToMemref/StructuredToMemref.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Casting.h"

#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton-shared/Transform/common_utils.h"
#define GEN_PASS_DEF_SPLITCOMPUTATIONALOP
#include "evas/Transform/Linalg/Passes.h.inc"

namespace mlir::triton::ev {
namespace {
using mlir::func::FuncOp;
using mlir::linalg::LinalgOp;

static constexpr llvm::StringRef kSchedulePrimitive = "schedule_primitive";

using Cluster = llvm::SmallVector<Operation *, 8>;
raw_ostream &operator<<(raw_ostream &os, const Cluster &cluster) {
  os << "[\n";
  for (size_t i = 0; i < cluster.size(); ++i) {
    if (i != 0) {
      os << ", "; // Separate elements with a comma
    }
    if (cluster[i] != nullptr) {
      // Assuming you want to output the address of the Operation object
      os << "Operation " << i << ": " << *(cluster[i])
         << "\n"; // Output the address (or use any relevant member function)
      // Alternatively, you can call a member function to display additional
      // information if desired cluster[i]->print();
    } else {
      os << "nullptr\n"; // Handle nullptr entries
    }
  }
  os << "]\n";
  return os;
}

void reorderCluster(Cluster &cluster) {
  if (cluster.empty())
    return;
  std::sort(cluster.begin(), cluster.end(),
            [](Operation *a, Operation *b) { return a->isBeforeInBlock(b); });
}

bool isComputationalOp(Operation *op) {
  StringRef opName = op->getName().getStringRef();
  return opName.starts_with("linalg") && opName != "linalg.yield" && opName != "linalg.generic";
}

bool isOnlyUser(Operation *A, Operation *B) {
  // Check if A has any results
  if (A->getNumResults() == 0)
    return false;
  // Check all results of A
  for (Value result : A->getResults()) {
    // If any result has no uses, return false
    if (result.use_empty())
      return false;
    // Check if all uses of this result are in operation B
    for (OpOperand &use : result.getUses()) {
      if (use.getOwner() != B)
        return false;
    }
  }
  // All results of A are only used by B
  return true;
}

bool usedOnlyInCluster(Operation *op, const Cluster &cluster) {
  for (Value result : op->getResults()) {
    for (OpOperand &use : result.getUses()) {
      if (!llvm::is_contained(cluster, use.getOwner()))
        return false;
    }
  }
  return true;
}

SmallVector<Value, 4> getInputsOfCluster(const Cluster &cluster) {
  llvm::SmallVector<Value, 4> inputs;
  llvm::SmallDenseSet<Value> inputSet;
  llvm::SmallDenseSet<Operation *> opSet;
  bool hasScatter = false;

  for (Operation *op : cluster) {
    if (isa<linalg::ScatterOp>(op)) {
      hasScatter = true;
    }
    bool inserted = opSet.insert(op).second;
    (void)inserted;
    assert(inserted && "cluster contains duplicate operations");
  }

  for (Operation *op : cluster) {
    for (Value operand : op->getOperands()) {
      Operation *defOp = operand.getDefiningOp();
      if (opSet.find(defOp) != opSet.end()) {
        // skip if defining op is in the cluster
        continue;
      }
      if (inputSet.insert(operand).second) {
        inputs.push_back(operand);
      }
    }
    if (hasScatter) {
      // Since the input of scatter is also the output, the  input will be added
      // again as the output parameter.
      inputs.push_back(op->getOperand(0));
    }
  }
  return inputs;
}

SmallVector<Value, 4> getOutputsOfCluster(const Cluster &cluster) {
  llvm::SmallVector<Value, 4> outputs;
  llvm::SmallDenseSet<Operation *> opSet;
  for (Operation *op : cluster) {
    // Should add all the operations recursively because a value might be used
    // by an operation of an inner region.
    op->walk([&](Operation *innerOp) {
      bool inserted = opSet.insert(innerOp).second;
      (void)inserted;
      assert(inserted && "cluster contains duplicate operations");
    });
  }

  for (Operation *op : cluster) {
    for (Value result : op->getResults()) {
      bool hasExternalUser =
          llvm::any_of(result.getUses(), [&](OpOperand &use) {
            return !opSet.count(use.getOwner());
          });
      if (hasExternalUser) {
        outputs.push_back(result);
      }
    }
  }
  return outputs;
}

Operation *getFirstOpInCluster(const Cluster &cluster) {
  Operation *firstOp = *std::min_element(
      cluster.begin(), cluster.end(),
      [](Operation *x, Operation *y) { return x->isBeforeInBlock(y); });
  return firstOp;
}

Operation *getLastOpInCluster(const Cluster &cluster) {
  Operation *lastOp = *std::max_element(
      cluster.begin(), cluster.end(),
      [](Operation *x, Operation *y) { return x->isBeforeInBlock(y); });
  return lastOp;
}

void moveConsumer(const Cluster &cluster) {
  Operation *firstOp = getFirstOpInCluster(cluster);
  Operation *lastOp = getLastOpInCluster(cluster);

  llvm::SmallDenseSet<Operation *> fusedSet(cluster.begin(), cluster.end());
  llvm::SmallDenseSet<Operation *> consumerSet;

  llvm::SmallVector<Operation *, 4> consumersVec;
  auto firstIter = firstOp->getIterator();
  auto lastIter = lastOp->getIterator();

  for (Operation &curOp : llvm::make_range(firstIter, lastIter)) {
    // isn't fused op && consumer's op
    // move this after fusion op
    if (!fusedSet.contains(&curOp)) {
      // fused op's consumer or consumer's consumer
      bool isConsumer =
          llvm::any_of(curOp.getOperands(), [&fusedSet, &consumerSet](Value v) {
            auto op = v.getDefiningOp();
            return fusedSet.contains(op) || consumerSet.contains(op);
          });
      if (isConsumer) {
        consumerSet.insert(&curOp);
        consumersVec.push_back(&curOp);
      }
    }
  }

  for (auto op : llvm::reverse(consumersVec)) {
    op->moveAfter(lastOp);
  }
}

bool isCallOpRet(Value v) {
  auto op = v.getDefiningOp();
  if (!op)
    return false;
  if (isa<func::CallOp>(op)) {
    return true;
  }
  for (auto operand : op->getOperands()) {
    if (isCallOpRet(operand))
      return true;
  }
  return false;
}

bool isScalarType(Type type) {
  return isa<IntegerType>(type) || isa<FloatType>(type) ||
         isa<ComplexType>(type) || isa<IndexType>(type);
}

Operation *findDstInput(Operation *op) {
  if (!op)
    return nullptr;
  if (isa_and_nonnull<bufferization::AllocTensorOp>(op)) {
    return op;
  }
  if (isa_and_nonnull<bufferization::ToTensorOp>(op)) {
    return op;
  }
  if (op->getDialect()->getNamespace() ==
      tensor::TensorDialect::getDialectNamespace()) {
    return findDstInput(op->getOperand(0).getDefiningOp());
  } else if (auto fillOp = llvm::dyn_cast_or_null<linalg::FillOp>(op)) {
    return findDstInput(fillOp.getOperand(1).getDefiningOp());
  }
  return nullptr;
}

Cluster findDestinationOps(Operation *op,
                           const llvm::SmallDenseSet<Operation *> &dstInputSet,
                           const Cluster &cls) {
  Cluster destinationOps;
  // Process each operand of the operation
  for (Value input : op->getOperands()) {
    Operation *inputOp = input.getDefiningOp();
    // Skip if this is an output dstInput
    if (!inputOp || dstInputSet.contains(inputOp))
      continue;
    // Skip if the input op is not used only in the cluster
    if (!usedOnlyInCluster(inputOp, cls))
      continue;
    // If this is an allocation operation, add it directly
    if (isa<bufferization::AllocTensorOp>(inputOp)) {
      destinationOps.push_back(inputOp);
      continue;
    }
    // Recursively process tensor dialect operations
    if (inputOp->getDialect()->getNamespace() ==
        tensor::TensorDialect::getDialectNamespace()) {
      // Get destination ops from the tensor op
      Cluster attachedOps = findDestinationOps(inputOp, dstInputSet, cls);
      // If we found destination ops, add them and the tensor op
      if (!attachedOps.empty()) {
        destinationOps.append(attachedOps);
        destinationOps.push_back(inputOp);
      }
    }
  }

  return destinationOps;
}

std::string getFuncName(int clusterIdx) {
  std::ostringstream nameStream;
  nameStream << "sub_kernel_" << clusterIdx;
  return nameStream.str();
}

class SplitComputationalOpPass
    : public ::impl::SplitComputationalOpBase<SplitComputationalOpPass> {
public:
  int getClusterIndex(Operation *op) {
    for (auto indexedCluster : llvm::enumerate(clusters)) {
      auto cluster = indexedCluster.value();
      for (auto clusterOp : cluster) {
        if (clusterOp == op)
          return indexedCluster.index();
      }
    }
    return -1;
  }

  int mergeClusters(int c1, int c2) {
    if (c1 == c2)
      return c1;
    if (c1 > c2) {
      return mergeClusters(c2, c1);
    }
    Cluster cluster2 = clusters[c2];
    clusters.erase(clusters.begin() + c2);
    clusters[c1].append(cluster2.begin(), cluster2.end());
    return c1;
  }

  void InitClusters(FuncOp funcOp) {
    funcOp.walk([this](Operation *op) {
      if (isComputationalOp(op)) {
        Cluster newCls;
        newCls.push_back(op);
        clusters.push_back(newCls);
      }
      return WalkResult::advance();
    });
  }

  bool ConnnectedTo(const Cluster &clsA, const Cluster &clsB) {
    auto valuesA = getOutputsOfCluster(clsA);
    auto valuesB = getInputsOfCluster(clsB);
    for (auto in : valuesA) {
      for (auto out : valuesB) {
        if (in == out)
          return true;
      }
    }
    return false;
  }

  bool HasSingleOutput(const Cluster &clsA) {
    auto outputs = getOutputsOfCluster(clsA);
    return outputs.size() <= 1;
  }

  Cluster getMergedCls(const Cluster &clsA, const Cluster &clsB) {
    Cluster ret = clsA;
    ret.append(clsB.begin(), clsB.end());
    return ret;
  }

  template <typename T> bool hasLinalgOp(const Cluster &cls) {
    for (auto op : cls) {
      if (isa<T>(op))
        return true;
    }
    return false;
  }

  template <typename T> bool IsolatedPattern(const Cluster &cls) {
    if (!hasLinalgOp<T>(cls))
      return true;
    for (auto op : cls) {
      if (isa<T>(op) && !(isa<bufferization::AllocTensorOp>(op)))
        return false;
    }
    return true;
  }

  template <typename... Ts> bool TryIsolatedPattern(const Cluster &cls) {
    return (... && IsolatedPattern<Ts>(cls));
  }

  bool TryRestrictedFusePattern(const Cluster &cls) {
    // Multi-output is not supported by evofc for now
    if (!HasSingleOutput(cls))
      return false;

    for (Operation *op : cls) {
      bool isReduceOp = true;
      SmallVector<int64_t> dims{}, shapes{};
      auto inputDimFunc =
          [&](auto op) {
            dims.push_back(op.getDimension());
            auto shape = op.getInput().getType().getShape();
            shapes.assign(shape.begin(), shape.end());
          };
      auto inputDimsFunc = [&](auto op) {
        dims.assign(op.getDimensions().begin(), op.getDimensions().end());
        auto shape = op.getInput().getType().getShape();
        shapes.assign(shape.begin(), shape.end());
      };
      auto inputsDimsFunc = [&](auto op) {
        dims.assign(op.getDimensions().begin(), op.getDimensions().end());
        auto type = op.getInputs()[0].getType();
        if (auto shapedType = dyn_cast<mlir::ShapedType>(type)) {
          auto shape = shapedType.getShape();
          shapes.assign(shape.begin(), shape.end());
        } else if (auto tensorType = dyn_cast<mlir::RankedTensorType>(type)) {
          auto shape = tensorType.getShape();
          shapes.assign(shape.begin(), shape.end());
        } else if (auto memrefType = dyn_cast<mlir::MemRefType>(type)) {
          auto shape = memrefType.getShape();
          shapes.assign(shape.begin(), shape.end());
        }
      };
      llvm::TypeSwitch<Operation *>(op)
          .Case<linalg::TopPOp>(inputDimFunc)
          .Case<linalg::NormOp>(inputDimFunc)
          .Case<linalg::ReduceMedianOp>(inputsDimsFunc)
          .Case<linalg::SoftmaxOp>(inputDimFunc)
          .Case<linalg::LogSoftmaxOp>(inputDimFunc)
          .Case<linalg::ReduceOp>(inputsDimsFunc)
          .Case<linalg::ReduceMeanOp>(inputsDimsFunc)
          .Case<linalg::TopkOp>(inputDimFunc)
          .Case<linalg::SortOp>(inputDimFunc)
          .Case<linalg::ArgmaxOp>(inputDimsFunc)
          .Case<linalg::CumsumOp>(inputDimFunc)
          .Case<linalg::InterpolateOp>(inputDimsFunc)
          .Case<linalg::FlipOp>(inputDimsFunc)
          .Default([&](auto op) { isReduceOp = false; });
      if (!isReduceOp)
        continue;

      bool parallelizable = false;
      for (int i = 0; i < shapes.size(); i++) {
        auto it = std::find(dims.begin(), dims.end(), i);
        if (it == dims.end() && shapes[i] != 1) {
          parallelizable = true;
          break;
        }
      }
      auto *ctx = op->getContext();
      op->setAttr("isParallelizable", mlir::BoolAttr::get(ctx, parallelizable));
      if (!parallelizable) {
        return false;
      }
    }

    if (!TryIsolatedPattern<linalg::SoftmaxOp, linalg::LayernormOp,
                            linalg::BroadcastOp, linalg::TransposeOp,
                            linalg::TakeOp, linalg::ScatterOp, linalg::EvPadOp,
                            linalg::SliceOp>(cls))
      return false;

    // transpose
    return true;
  }

  bool CanFuseTo(const Cluster &clsA, const Cluster &clsB) {
    if (!ConnnectedTo(clsA, clsB))
      return false;

    Operation *lastOpA = getLastOpInCluster(clsA);
    Operation *firstOpB = getFirstOpInCluster(clsB);
    Cluster clsMid;
    // Check if lastOpA and firstOpB are in the same block
    if (lastOpA->getBlock() != firstOpB->getBlock())
      return false;
    for (auto &op : llvm::make_range(std::next(lastOpA->getIterator()),
                                     firstOpB->getIterator())) {
      clsMid.push_back(&op);
    }
    // annotate op to cut the fusion    
    if (utils::getAnnotation(lastOpA)) return false;
    // check if the middle cluster is connected to both clsA and clsB
    if (!clsMid.empty() && ConnnectedTo(clsA, clsMid) &&
        ConnnectedTo(clsMid, clsB)) {
      return false;
    }
    auto tryMerged = getMergedCls(clsA, clsB);
    return TryRestrictedFusePattern(tryMerged);
  }

  void FuseClustersWithDefUse() {
    if (clusters.size() <= 1)
      return;
    for (size_t index = 0; index < clusters.size() - 1; ++index) {
      if (CanFuseTo(clusters[index], clusters[index + 1])) {
        (void)mergeClusters(index, index + 1);
        FuseClustersWithDefUse();
        return;
      }
    }
  }

  Cluster getAttachedCluster(const Cluster &cls) {
    Cluster ret = cls;
    auto outputs = getOutputsOfCluster(cls);
    auto outputsSet =
        llvm::SmallDenseSet<Value>(outputs.begin(), outputs.end());
    llvm::SmallDenseSet<Operation *> dstInputSet;
    // find all the dst input ops that correspond to the cluster outputs
    for (auto op : cls) {
      auto dstOp = cast<DestinationStyleOpInterface>(op);
      for (auto [idx, output] : llvm::enumerate(op->getResults())) {
        if (outputsSet.contains(output)) {
          auto init =
              findDstInput(dstOp.getDpsInitOperand(idx)->get().getDefiningOp());
            dstInputSet.insert(init);
            outputToDstInput[output] = init;
        }
      }
    }
    for (auto op : cls) {
      auto dst_ops = findDestinationOps(op, dstInputSet, cls);
      ret.append(dst_ops);
    }
    reorderCluster(ret);
    return ret;
  }


  void setMemscopeForAllocTensorOp(bufferization::AllocTensorOp allocOp, OpBuilder &b) {
    if (auto annotateOp = utils::getAnnotation(allocOp.getOperation())){
      auto meminfo = annotateOp.getMeminfo();
      allocOp.setMemorySpaceAttr(b.getI64IntegerAttr((int64_t)meminfo.getScope()));
    } else {
      allocOp.setMemorySpaceAttr(b.getI64IntegerAttr((int64_t)mlir::triton::MemScope::L2));
    }
    if (mlir::isa<mlir::RankedTensorType>(allocOp.getType())) {
      auto tensorType = mlir::cast<mlir::RankedTensorType>(allocOp.getType());
      if (tensorType.getRank() == 0 ||
          (tensorType.getRank() == 1 && tensorType.getShape()[0] == 1) ||
          (tensorType.getRank() == 2 && tensorType.getShape()[0] == 1 && tensorType.getShape()[1] == 1)) {
        allocOp.setMemorySpaceAttr(b.getI64IntegerAttr((int64_t)mlir::triton::MemScope::DDR));
      }
    }
  }

  bool isAnnotatedPrefetch(Operation *op) {
    if (auto annotateOp = utils::getAnnotation(op)){
      auto meminfo = annotateOp.getMeminfo();
      return meminfo.getPrefetch();
    }
    return false;
  }

  void annotatePrefetchToSubKernel(bufferization::AllocTensorOp allocOp, func::CallOp callOp, OpBuilder &b) {
    if (isAnnotatedPrefetch(allocOp)) {
      callOp->setAttr(mlir::ev::prefetchName, b.getBoolAttr(true));
    }
  }

  func::FuncOp createFuncOpWithCluster(OpBuilder &b, StringRef subFnName,
                                       ValueRange inputs, ValueRange outputs,
                                       const Cluster &cluster,
                                       Operation *insertionPoint) {
    Operation *lastOp = getLastOpInCluster(cluster);
    llvm::SmallVector<Location, 4> locations;
    locations.reserve(cluster.size());
    for (Operation *op : cluster) {
      locations.push_back(op->getLoc());
    }
    Location fusedLoc = FusedLoc::get(lastOp->getContext(), locations);

    llvm::SmallVector<Type, 4> outputTypes;
    outputTypes.reserve(outputs.size());
    for (Value v : outputs) {
      outputTypes.push_back(v.getType());
    }
    llvm::SmallVector<Type, 4> inputTypes;
    inputTypes.reserve(inputs.size());
    for (Value v : inputs) {
      inputTypes.push_back(v.getType());
    }

    moveConsumer(cluster);

    auto subFnType = b.getFunctionType(inputTypes, outputTypes);
    b.setInsertionPoint(insertionPoint);
    func::FuncOp subFnOp =
        b.create<func::FuncOp>(fusedLoc, subFnName, subFnType);
    subFnOp.setSymVisibility("private");
    b.setInsertionPoint(lastOp);
    auto callOp = b.create<func::CallOp>(fusedLoc, subFnOp, inputs);
    callOp->setAttr(kSchedulePrimitive, b.getBoolAttr(true));
    // callOp->setAttr(mlir::ev::addrName,
    //                 b.getArrayAttr(SmallVector<Attribute>(
    //                     callOp.getNumResults(), b.getI64IntegerAttr(-1))));
    Block *block = subFnOp.addEntryBlock();
    b.setInsertionPoint(block, block->end());
    IRMapping bvm;
    for (auto inputAndArg : llvm::zip(inputs, subFnOp.getArguments())) {
      bvm.map(std::get<0>(inputAndArg), std::get<1>(inputAndArg));
    }
    for (Operation *op : cluster) {
      b.clone(*op, bvm);
    }
    llvm::SmallVector<Value, 4> funcReturns;
    for (Value output : outputs) {
      funcReturns.push_back(bvm.lookupOrDefault(output));
    }
    b.create<func::ReturnOp>(fusedLoc, funcReturns);

    for (auto outputAndResult : llvm::zip(outputs, callOp.getResults())) {
      Value output = std::get<0>(outputAndResult);
      // replace the use of output with the destination alloc op
      Operation *dstInputOp = outputToDstInput[output];
      Value dstInputValue = dstInputOp->getResult(0);
      Value callResult = std::get<1>(outputAndResult);
      for (OpOperand &use : llvm::make_early_inc_range(output.getUses())) {
        use.set(dstInputValue);
      }
      // 只对 bufferization::AllocTensorOp 设置内存空间和预取属性
      if (auto tensorAllocOp = dyn_cast<bufferization::AllocTensorOp>(dstInputOp)) {
        setMemscopeForAllocTensorOp(tensorAllocOp, b);
        // todo：需要考虑totensor的情况
        annotatePrefetchToSubKernel(tensorAllocOp, callOp, b);
      }
    }

    // erase dead ops in the end
    for (Operation *op : llvm::reverse(cluster)) {
      if (op->use_empty()) {
        op->erase();
      }
    }

    return subFnOp;
  }
  FailureOr<func::FuncOp> createFuncOpWithCluster(OpBuilder &b,
                                                  StringRef subFnName,
                                                  const Cluster &cluster,
                                                  Operation *insertionPoint) {
    auto attachedCluster = getAttachedCluster(cluster);
    llvm::SmallVector<Value, 4> inputs = getInputsOfCluster(attachedCluster);
    llvm::SmallVector<Value, 4> outputs = getOutputsOfCluster(attachedCluster);
    return createFuncOpWithCluster(b, subFnName, inputs, outputs,
                                   attachedCluster, insertionPoint);
  }
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp m = getOperation();
    // set funcName fixed for finding the outer func in further process
    auto f = *(m.getOps<func::FuncOp>().begin());
    const std::string funcName = "kernel";
    f.setName(funcName);
    InitClusters(f);
    FuseClustersWithDefUse();
    OpBuilder b(f);

    SymbolTable symTable(m);
    for (auto c : llvm::enumerate(clusters)) {
      llvm::outs() << c.value() << "\n";
      FailureOr<func::FuncOp> subFnOp = createFuncOpWithCluster(
          b, getFuncName(c.index()), c.value(), f.getOperation());
      assert(mlir::succeeded(subFnOp) && "create FuncOp failed");
      symTable.insert(*subFnOp);
    }
  }

private:
  SmallVector<Cluster> clusters;
  llvm::SmallDenseMap<Value, Operation *> outputToDstInput;
};

} // namespace

std::unique_ptr<mlir::Pass> createSplitComputationalOpPass() {
  return std::make_unique<SplitComputationalOpPass>();
}

} // namespace mlir::triton::ev
