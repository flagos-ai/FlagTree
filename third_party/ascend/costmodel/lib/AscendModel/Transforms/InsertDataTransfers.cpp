//===- InsertDataTransfers.cpp - Insert Cube/Vector memory transfers ------===//
//
// This pass inserts explicit memory transfer operations when data flows
// between Vector and Cube compute units on Ascend 910B.
//
// Data flow model (910B architecture):
//
//   Vector path:  HBM ──MTE2──▶ UB ──▶ Vector ──▶ UB ──MTE3──▶ HBM
//   Cube path:    HBM ──MTE2──▶ L1 ──MTE1──▶ L0A/L0B ──▶ Cube ──▶ L0C
//   ──FixPipe──▶ HBM
//
// When Vector result is used by Cube (e.g., arith.mulf -> tt.dot):
//   Insert: vector_store (MTE3: UB → HBM) + cube_load (MTE2: HBM → L1)
//
// When Cube result is used by Vector (e.g., tt.dot -> arith.addf):
//   Insert: cube_store (FixPipe: L0C → HBM) + vector_load (MTE2: HBM → UB)
//
//===----------------------------------------------------------------------===//

#include "AscendModel/IR/AscendModelDialect.h"
#include "AscendModel/Transforms/Passes.h"
#include "AscendModel/Utils.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir {
namespace ascend {

#define GEN_PASS_DEF_INSERTDATATRANSFERSPASS
#include "AscendModel/Transforms/Passes.h.inc"

namespace {

// Use utility functions from Utils.h
using utils::getByteSize;

//===----------------------------------------------------------------------===//
// Helper: Classify operations by compute path
//===----------------------------------------------------------------------===//

enum class ComputePath {
  Cube,   // Matrix operations: matmul
  Vector, // SIMD operations: elementwise, reduce, etc.
  Memory, // Memory operations: load/store
  Unknown
};

/// Classify an operation by its compute path.
static ComputePath getComputePath(Operation *op) {
  // Cube operations
  if (isa<MatmulOp>(op))
    return ComputePath::Cube;

  // Vector operations
  if (isa<AddOp, SubOp, MulOp, DivOp, MaxOp, MinOp, ExpOp, LogOp, SqrtOp,
          RsqrtOp, TanhOp, SigmoidOp, NegOp, AbsOp, ReluOp, CastOp, ReduceSumOp,
          ReduceMaxOp, ReduceMinOp, ReduceProdOp, BroadcastOp, SelectOp>(op))
    return ComputePath::Vector;

  // Memory operations
  if (isa<VectorLoadOp, VectorStoreOp, CubeLoadOp, CubeStoreOp>(op))
    return ComputePath::Memory;

  // Arith/Math tensor ops are Vector (before conversion)
  StringRef dialect = op->getDialect()->getNamespace();
  if (dialect == "arith" || dialect == "math") {
    // Check if it operates on tensors
    for (Type resultType : op->getResultTypes()) {
      if (isa<TensorType>(resultType))
        return ComputePath::Vector;
    }
  }

  return ComputePath::Unknown;
}

//===----------------------------------------------------------------------===//
// Pass Implementation
//===----------------------------------------------------------------------===//

struct InsertDataTransfersPass
    : public impl::InsertDataTransfersPassBase<InsertDataTransfersPass> {
  using InsertDataTransfersPassBase::InsertDataTransfersPassBase;

  void runOnOperation() override {
    ModuleOp module = getOperation();

    // Track which values have been transferred to avoid duplicates
    llvm::DenseMap<Value, Value> vectorToHBM; // Value in UB -> Value in HBM
    llvm::DenseMap<Value, Value> hbmToL1;     // Value in HBM -> Value in L1

    // Collect all MatmulOps first (to avoid iterator invalidation)
    SmallVector<MatmulOp> matmulOps;
    module.walk([&](MatmulOp op) { matmulOps.push_back(op); });

    // Process each MatmulOp
    for (MatmulOp matmulOp : matmulOps) {
      OpBuilder builder(matmulOp);
      Location loc = matmulOp.getLoc();

      // Null attrs for optional parameters
      ::mlir::IntegerAttr nullAttr;
      // Memory-space attrs for transfer ops (tilesim migration, root cause 1).
      auto hbm = builder.getStringAttr("hbm");
      auto l1 = builder.getStringAttr("l1");
      auto l0c = builder.getStringAttr("l0c");
      auto ub = builder.getStringAttr("ub");

      // === Step 1: Handle inputs to Cube ===
      // MatmulOp has lhs and rhs operands
      SmallVector<Value, 2> operands = {matmulOp.getLhs(), matmulOp.getRhs()};

      for (unsigned i = 0; i < operands.size(); ++i) {
        Value operand = operands[i];
        Operation *defOp = operand.getDefiningOp();

        // Skip if no defining op (block argument)
        if (!defOp)
          continue;

        // Skip if already a memory load
        if (isa<CubeLoadOp, VectorLoadOp>(defOp))
          continue;

        ComputePath path = getComputePath(defOp);

        // If operand comes from Vector path, insert transfers
        if (path == ComputePath::Vector) {
          auto tensorType = dyn_cast<RankedTensorType>(operand.getType());
          if (!tensorType)
            continue;

          uint64_t bytes = static_cast<uint64_t>(getByteSize(tensorType));

          // Check if we already transferred this value
          Value hbmValue;
          auto it = vectorToHBM.find(operand);
          if (it != vectorToHBM.end()) {
            hbmValue = it->second;
          } else {
            // Insert vector_store: UB → HBM (MTE3)
            builder.create<VectorStoreOp>(loc, operand, bytes, ub, hbm,
                                          nullAttr, nullAttr);
            hbmValue = operand; // After store, conceptually in HBM
            vectorToHBM[operand] = hbmValue;
          }

          // Check if we already loaded to L1
          Value l1Value;
          auto it2 = hbmToL1.find(hbmValue);
          if (it2 != hbmToL1.end()) {
            l1Value = it2->second;
          } else {
            // Insert cube_load: HBM → L1 (MTE2)
            auto cubeLoad = builder.create<CubeLoadOp>(
                loc, tensorType, hbmValue, bytes, hbm, l1, nullAttr, nullAttr);
            l1Value = cubeLoad.getResult();
            hbmToL1[hbmValue] = l1Value;
          }

          // Replace operand in matmul
          matmulOp->setOperand(i, l1Value);
        }
      }

      // === Step 2: Handle output from Cube ===
      // Check if any user is a Vector operation

      Value matmulResult = matmulOp.getResult();
      auto resultType = dyn_cast<RankedTensorType>(matmulResult.getType());
      if (!resultType)
        continue;

      // Collect Vector users
      SmallVector<OpOperand *> vectorUsers;
      for (OpOperand &use : matmulResult.getUses()) {
        Operation *user = use.getOwner();
        ComputePath userPath = getComputePath(user);
        if (userPath == ComputePath::Vector) {
          vectorUsers.push_back(&use);
        }
      }

      if (vectorUsers.empty())
        continue;

      // Insert after matmul
      builder.setInsertionPointAfter(matmulOp);

      uint64_t bytes = static_cast<uint64_t>(getByteSize(resultType));

      // Insert cube_store: L0C → HBM (FixPipe)
      builder.create<CubeStoreOp>(loc, matmulResult, bytes, l0c, hbm, nullAttr,
                                  nullAttr);

      // Insert vector_load: HBM → UB (MTE2)
      auto vectorLoad = builder.create<VectorLoadOp>(
          loc, resultType, matmulResult, bytes, hbm, ub, nullAttr, nullAttr);

      // Replace uses in Vector operations
      for (OpOperand *use : vectorUsers) {
        use->set(vectorLoad.getResult());
      }
    }
  }
};

} // namespace
} // namespace ascend
} // namespace mlir
