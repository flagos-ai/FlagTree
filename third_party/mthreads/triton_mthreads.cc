#include "Dialect/MTGPU/IR/Dialect.h"
#include "Dialect/MUSA/IR/Dialect.h"
#include "MTGPUToLLVM/Passes.h"
#include "TritonMUSAGPUToLLVM/Passes.h"
#include "TritonMUSAGPUTransforms/Passes.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/MTVM/MTVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/LLVMTranslationInterface.h"
#include "passes.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/IR/CallingConv.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include <algorithm>
#include <cstdint>
#include <functional>
#include <numeric>
#include <optional>
#include <pybind11/pybind11.h>
#include <string>

namespace py = pybind11;

#ifdef __TLE__
void init_triton_musa_tle_ir(py::module m);
void init_triton_musa_tle_frontend_passes_ttgpuir(py::module m);
void init_triton_musa_tle_dialect_passes_ttgpuir(py::module m);
void register_triton_musa_tle_dialects(mlir::DialectRegistry &registry);
#endif

namespace {

unsigned canonicalizeLaneShuffles(llvm::Module &module) {
  unsigned changed = 0;
  for (auto &fn : module)
    for (auto &bb : fn)
      for (auto &inst : bb) {
        auto *call = llvm::dyn_cast<llvm::CallInst>(&inst);
        auto *callee = call ? call->getCalledFunction() : nullptr;
        if (!callee || callee->getName() != "llvm.musa.shfl.idx.sync.i32" ||
            call->arg_size() != 4)
          continue;
        auto *control =
            llvm::dyn_cast<llvm::ConstantInt>(call->getArgOperand(2));
        if (!control || control->getZExtValue() != 12319 ||
            !llvm::isa<llvm::ConstantPointerNull>(call->getArgOperand(3)))
          continue;
        auto *index =
            llvm::dyn_cast<llvm::BinaryOperator>(call->getArgOperand(1));
        if (!index || index->getOpcode() != llvm::Instruction::Xor)
          continue;
        auto *mask = llvm::dyn_cast<llvm::ConstantInt>(index->getOperand(1));
        llvm::Value *lane = index->getOperand(0);
        if (!mask) {
          mask = llvm::dyn_cast<llvm::ConstantInt>(index->getOperand(0));
          lane = index->getOperand(1);
        }
        if (!mask || mask->getBitWidth() != 32 || mask->getZExtValue() >= 32)
          continue;
        auto *lowBits = llvm::dyn_cast<llvm::BinaryOperator>(lane);
        if (!lowBits || lowBits->getOpcode() != llvm::Instruction::And)
          continue;
        auto *bits = llvm::dyn_cast<llvm::ConstantInt>(lowBits->getOperand(1));
        llvm::Value *tid = lowBits->getOperand(0);
        if (!bits) {
          bits = llvm::dyn_cast<llvm::ConstantInt>(lowBits->getOperand(0));
          tid = lowBits->getOperand(1);
        }
        auto *read = llvm::dyn_cast<llvm::CallInst>(tid);
        auto *reg = read ? read->getCalledFunction() : nullptr;
        if (!bits || bits->getZExtValue() != 31 || !reg ||
            reg->getName() != "llvm.musa.read.ptx.sreg.tid.x" ||
            read->arg_size() != 0)
          continue;
        auto *replacement = module.getFunction("llvm.musa.shfl.xor.sync.i32");
        if (!replacement) {
          replacement = llvm::Function::Create(
              callee->getFunctionType(), llvm::GlobalValue::ExternalLinkage,
              "llvm.musa.shfl.xor.sync.i32", module);
          replacement->setAttributes(callee->getAttributes());
        }
        if (replacement->getFunctionType() != callee->getFunctionType())
          continue;
        call->setCalledFunction(replacement);
        call->setArgOperand(1, mask);
        ++changed;
      }
  return changed;
}

llvm::CallInst *emitBurstAdd(llvm::IRBuilder<> &builder, llvm::Module &module,
                             llvm::Value *a, llvm::Value *b) {
  auto *ty = llvm::cast<llvm::FixedVectorType>(a->getType());
  auto *type = llvm::FunctionType::get(ty, {ty, ty}, false);
  auto name = ty->getNumElements() == 4 ? "llvm.musa.add.f.bst4.src01"
                                        : "llvm.musa.add.f.bst2.src01";
  return builder.CreateCall(module.getOrInsertFunction(name, type), {a, b});
}

llvm::CallInst *emitScalarAdd(llvm::IRBuilder<> &builder, llvm::Value *a,
                              llvm::Value *b) {
  // Multiplication by exactly one preserves the FP32 add's single rounding.
  return builder.CreateIntrinsic(
      llvm::Intrinsic::fma, {a->getType()},
      {a, llvm::ConstantFP::get(a->getType(), 1.), b});
}

// A packed FP8 source can also feed a pairwise reduction. Only one lane
// of a shuffle with three poison lanes has a defined sum. Keep its operand
// order and the position of the addition in the original basic block.
unsigned lowerPackPairSums(llvm::Module &module) {
  llvm::SmallVector<std::pair<llvm::BinaryOperator *, unsigned>> pairs;
  for (auto &fn : module) {
    if (fn.hasFnAttribute(llvm::Attribute::StrictFP))
      continue;
    for (auto &bb : fn) {
      llvm::SmallPtrSet<llvm::Value *, 16> packed;
      for (auto &inst : bb) {
        if (auto *call = llvm::dyn_cast<llvm::CallInst>(&inst)) {
          auto *callee = call->getCalledFunction();
          if (callee && callee->getName() == "llvm.musa.f2e4m3.rn.bst4")
            packed.insert(call->getArgOperand(0));
        }
        auto *add = llvm::dyn_cast<llvm::BinaryOperator>(&inst);
        if (!add || add->getOpcode() != llvm::Instruction::FAdd ||
            add->getFastMathFlags().any())
          continue;
        for (unsigned side : {0u, 1u}) {
          auto *v = add->getOperand(side);
          auto *shuffle = llvm::dyn_cast<llvm::ShuffleVectorInst>(
              add->getOperand(1 - side));
          if (!packed.contains(v) || !shuffle || shuffle->getOperand(0) != v ||
              shuffle->getShuffleMask().size() != 4)
            continue;
          auto mask = shuffle->getShuffleMask();
          if (mask == llvm::ArrayRef<int>({1, -1, -1, -1}))
            pairs.emplace_back(add, 0);
          else if (mask == llvm::ArrayRef<int>({-1, -1, 3, -1}))
            pairs.emplace_back(add, 2);
        }
      }
    }
  }
  for (auto [add, lane] : pairs) {
    llvm::IRBuilder<> b(add);
    auto *sum =
        emitScalarAdd(b, b.CreateExtractElement(add->getOperand(0), lane),
                      b.CreateExtractElement(add->getOperand(1), lane));
    add->replaceAllUsesWith(b.CreateInsertElement(
        llvm::PoisonValue::get(add->getType()), sum, lane));
    add->eraseFromParent();
  }
  return pairs.size();
}

unsigned demandedSqmmaAddLanes(llvm::Value *value,
                               llvm::DenseMap<llvm::Value *, unsigned> &memo,
                               llvm::SmallPtrSetImpl<llvm::Value *> &visiting) {
  auto *type = llvm::dyn_cast<llvm::FixedVectorType>(value->getType());
  if (!type || type->getNumElements() > 32)
    return ~0u;
  unsigned width = type->getNumElements();
  unsigned all = width == 32 ? ~0u : ((1u << width) - 1);
  auto found = memo.find(value);
  if (found != memo.end())
    return found->second;
  if (!visiting.insert(value).second)
    return all;
  unsigned demanded = 0;
  for (auto *user : value->users()) {
    if (auto *extract = llvm::dyn_cast<llvm::ExtractElementInst>(user)) {
      auto *index =
          llvm::dyn_cast<llvm::ConstantInt>(extract->getIndexOperand());
      if (!index || index->getZExtValue() >= width) {
        demanded = all;
        break;
      }
      demanded |= 1u << index->getZExtValue();
      continue;
    }
    if (auto *shuffle = llvm::dyn_cast<llvm::ShuffleVectorInst>(user)) {
      auto *outputType =
          llvm::dyn_cast<llvm::FixedVectorType>(shuffle->getType());
      if (!outputType || outputType->getNumElements() > 32) {
        demanded = all;
        break;
      }
      unsigned output = demandedSqmmaAddLanes(shuffle, memo, visiting);
      for (auto [i, lane] : llvm::enumerate(shuffle->getShuffleMask())) {
        if (i >= 32 || !(output & (1u << i)) || lane < 0)
          continue;
        if (lane < static_cast<int>(width) && shuffle->getOperand(0) == value)
          demanded |= 1u << lane;
        if (lane >= static_cast<int>(width) && shuffle->getOperand(1) == value)
          demanded |= 1u << (lane - width);
      }
      continue;
    }
    if (auto *insert = llvm::dyn_cast<llvm::InsertElementInst>(user)) {
      auto *index = llvm::dyn_cast<llvm::ConstantInt>(insert->getOperand(2));
      if (insert->getOperand(0) != value || !index ||
          index->getZExtValue() >= width) {
        demanded = all;
        break;
      }
      demanded |= demandedSqmmaAddLanes(insert, memo, visiting) &
                  ~(1u << index->getZExtValue());
      continue;
    }
    // These operations are pointwise and preserve the lane count. PHI
    // recursion is deliberately handled as all-live on cycles.
    auto *inst = llvm::dyn_cast<llvm::Instruction>(user);
    auto *outType =
        inst ? llvm::dyn_cast<llvm::FixedVectorType>(inst->getType()) : nullptr;
    bool pointwise =
        inst && (llvm::isa<llvm::BinaryOperator, llvm::UnaryOperator,
                           llvm::PHINode, llvm::SelectInst>(inst));
    if (auto *call = llvm::dyn_cast_or_null<llvm::CallBase>(inst))
      if (auto *callee = call->getCalledFunction())
        pointwise = callee->getIntrinsicID() == llvm::Intrinsic::fma;
    if (pointwise && outType && outType->getNumElements() == width) {
      demanded |= demandedSqmmaAddLanes(inst, memo, visiting);
      continue;
    }
    demanded = all;
    break;
  }
  visiting.erase(value);
  memo[value] = demanded;
  return demanded;
}

void packIndependentAdds(
    llvm::Module &module,
    const llvm::SmallPtrSetImpl<llvm::CallInst *> &scalarAdds) {

  for (auto &fn : module) {
    if (fn.isDeclaration() || fn.hasFnAttribute(llvm::Attribute::StrictFP))
      continue;
    for (auto &bb : fn) {
      llvm::SmallVector<llvm::CallInst *, 4> batch;
      llvm::SmallVector<llvm::Instruction *, 16> moves;
      llvm::SmallPtrSet<llvm::Instruction *, 16> planned;

      auto flush = [&]() {
        if (batch.size() >= 2) {
          unsigned width = batch.size() >= 4 ? 4 : 2;
          auto *anchor = batch.front();
          for (auto *inst : moves)
            inst->moveBefore(anchor);
          llvm::IRBuilder<> builder(anchor);
          auto *vec = llvm::FixedVectorType::get(builder.getFloatTy(), width);
          llvm::Value *a = llvm::PoisonValue::get(vec);
          llvm::Value *b = llvm::PoisonValue::get(vec);
          for (unsigned i = 0; i < width; ++i) {
            a = builder.CreateInsertElement(a, batch[i]->getArgOperand(0), i);
            b = builder.CreateInsertElement(b, batch[i]->getArgOperand(2), i);
          }
          auto *call = emitBurstAdd(builder, module, a, b);
          call->setDebugLoc(anchor->getDebugLoc());
          for (unsigned i = 0; i < width; ++i)
            batch[i]->replaceAllUsesWith(builder.CreateExtractElement(call, i));
          for (unsigned i = 0; i < width; ++i)
            batch[i]->eraseFromParent();
        }
        batch.clear();
        moves.clear();
        planned.clear();
      };
      // Snapshot the original order; flush removes earlier batch members only.
      llvm::SmallVector<llvm::Instruction *, 128> original;
      for (auto &inst : bb)
        original.push_back(&inst);
      for (auto *inst : original) {
        auto *call = llvm::dyn_cast<llvm::CallInst>(inst);
        if (call && scalarAdds.contains(call)) {
          auto *anchor = batch.empty() ? call : batch.front();
          unsigned previousMoves = moves.size();
          // A later member may depend on an earlier ADD through extract/insert
          // views. Reject that batch; hoisting a call or arithmetic is
          // forbidden.
          std::function<bool(llvm::Value *)> prepare = [&](llvm::Value *value) {
            auto *def = llvm::dyn_cast<llvm::Instruction>(value);
            if (!def || def->getParent() != &bb || def->comesBefore(anchor))
              return true;
            if (def == anchor || llvm::is_contained(batch, def))
              return false;
            if (planned.contains(def))
              return true;
            if (!llvm::isa<llvm::ExtractElementInst, llvm::InsertElementInst,
                           llvm::ShuffleVectorInst, llvm::BitCastInst>(def))
              return false;
            for (auto &operand : def->operands())
              if (!prepare(operand.get()))
                return false;
            planned.insert(def);
            moves.push_back(def);
            return true;
          };
          if (!prepare(call->getArgOperand(0)) ||
              !prepare(call->getArgOperand(2))) {
            // A failed candidate must not contribute partially collected
            // register views to the preceding batch's hoist plan.
            moves.resize(previousMoves);
            planned.clear();
            for (auto *move : moves)
              planned.insert(move);
            flush();
          }
          batch.push_back(call);
          if (batch.size() == 4)
            flush();
          continue;
        }
        if (call || inst->mayReadOrWriteMemory() ||
            inst->mayHaveSideEffects() || inst->isTerminator())
          flush();
      }
      flush();
    }
  }
}

unsigned lowerMp31Fp32Adds(llvm::Module &module) {
  unsigned changed = lowerPackPairSums(module);
  llvm::SmallVector<llvm::BinaryOperator *> adds;
  for (auto &fn : module) {
    if (fn.hasFnAttribute(llvm::Attribute::StrictFP))
      continue;
    for (auto &bb : fn) {
      bool pending = false;
      for (auto &inst : bb) {
        if (auto *call = llvm::dyn_cast<llvm::CallBase>(&inst)) {
          auto *callee = call->getCalledFunction();
          if (callee && callee->getName().starts_with("llvm.musa.sqmma.")) {
            auto name = callee->getName();
            if (name == "llvm.musa.sqmma.wait")
              pending = false;
            else if (name.ends_with(".mma"))
              pending = name.starts_with("llvm.musa.sqmma.e4m3.") ||
                        name.starts_with("llvm.musa.sqmma.e5m2.");
          }
        }
        auto *add = llvm::dyn_cast<llvm::BinaryOperator>(&inst);
        if (!pending || !add || add->getOpcode() != llvm::Instruction::FAdd ||
            add->getFastMathFlags().any())
          continue;
        auto *vec = llvm::dyn_cast<llvm::FixedVectorType>(add->getType());
        if (add->getType()->isFloatTy() ||
            (vec && vec->getElementType()->isFloatTy() &&
             (vec->getNumElements() == 2 || vec->getNumElements() == 4)))
          adds.push_back(add);
      }
    }
  }

  llvm::DenseMap<llvm::Value *, unsigned> masks;
  llvm::SmallPtrSet<llvm::Value *, 16> visiting;
  for (auto *add : adds)
    if (add->getType()->isVectorTy())
      demandedSqmmaAddLanes(add, masks, visiting);
  llvm::SmallPtrSet<llvm::CallInst *, 32> scalarAdds;
  for (auto *add : adds) {
    llvm::IRBuilder<> b(add);
    auto *lhs = add->getOperand(0);
    auto *rhs = add->getOperand(1);
    auto *vec = llvm::dyn_cast<llvm::FixedVectorType>(add->getType());
    llvm::Value *result;
    if (!vec) {
      auto *sum = emitScalarAdd(b, lhs, rhs);
      scalarAdds.insert(sum);
      result = sum;
    } else if (masks.lookup(add) == (1u << vec->getNumElements()) - 1) {
      result = emitBurstAdd(b, module, lhs, rhs);
    } else {
      result = llvm::PoisonValue::get(vec);
      for (unsigned i = 0; i < vec->getNumElements(); ++i) {
        if (!(masks.lookup(add) & (1u << i)))
          continue;
        auto *sum = emitScalarAdd(b, b.CreateExtractElement(lhs, i),
                                  b.CreateExtractElement(rhs, i));
        scalarAdds.insert(sum);
        result = b.CreateInsertElement(result, sum, i);
      }
    }
    add->replaceAllUsesWith(result);
    add->eraseFromParent();
    ++changed;
  }
  if (!scalarAdds.empty())
    packIndependentAdds(module, scalarAdds);
  return changed;
}

llvm::Function *findPrimaryKernel(llvm::Module &module,
                                  llvm::StringRef kernelNameHint) {
  if (!kernelNameHint.empty()) {
    if (llvm::Function *fn = module.getFunction(kernelNameHint)) {
      if (!fn->isDeclaration())
        return fn;
    }
  }
  for (llvm::Function &fn : module) {
    if (!fn.isDeclaration() &&
        fn.getLinkage() == llvm::GlobalValue::ExternalLinkage)
      return &fn;
  }
  for (llvm::Function &fn : module) {
    if (!fn.isDeclaration())
      return &fn;
  }
  return nullptr;
}

bool hasMusaAnnotation(llvm::NamedMDNode *annotations, const llvm::Function &fn,
                       llvm::StringRef key) {
  if (!annotations)
    return false;
  for (llvm::MDNode *node : annotations->operands()) {
    if (!node || node->getNumOperands() < 3)
      continue;
    auto *valueMD = llvm::dyn_cast<llvm::ValueAsMetadata>(node->getOperand(0));
    auto *keyMD = llvm::dyn_cast<llvm::MDString>(node->getOperand(1));
    if (!valueMD || !keyMD)
      continue;
    auto *annotatedFn = llvm::dyn_cast<llvm::Function>(valueMD->getValue());
    if (annotatedFn != &fn)
      continue;
    if (keyMD->getString() == key)
      return true;
  }
  return false;
}

void addMusaAnnotation(llvm::Module &module, llvm::Function &fn,
                       llvm::StringRef key, int32_t value) {
  llvm::NamedMDNode *annotations =
      module.getOrInsertNamedMetadata("musa.annotations");
  if (hasMusaAnnotation(annotations, fn, key))
    return;

  llvm::LLVMContext &ctx = module.getContext();
  llvm::MDNode *node = llvm::MDNode::get(
      ctx, {llvm::ValueAsMetadata::get(&fn), llvm::MDString::get(ctx, key),
            llvm::ConstantAsMetadata::get(
                llvm::ConstantInt::get(llvm::Type::getInt32Ty(ctx), value))});
  annotations->addOperand(node);
}

bool moduleUsesMulhiHelper(const llvm::Module &module) {
  for (const llvm::Function &fn : module) {
    if (fn.isDeclaration())
      continue;
    for (const llvm::BasicBlock &block : fn) {
      for (const llvm::Instruction &inst : block) {
        auto *call = llvm::dyn_cast<llvm::CallBase>(&inst);
        if (!call)
          continue;
        const llvm::Function *callee = call->getCalledFunction();
        if (!callee)
          continue;
        llvm::StringRef calleeName = callee->getName();
        if (calleeName == "__mt_umulhi" || calleeName == "__mt_umul64hi")
          return true;
      }
    }
  }
  return false;
}

constexpr llvm::StringLiteral kTMETailDivisibilityAttr =
    "musa.tme_tail_divisibility";

std::optional<int64_t> getElementSizeBytes(mlir::Type type) {
  int64_t bits = type.getIntOrFloatBitWidth();
  if (bits <= 0 || bits % 8 != 0)
    return std::nullopt;
  return bits / 8;
}

bool blockShapeSatisfiesTME(mlir::triton::TensorDescType descTy,
                            int64_t elemBytes) {
  auto shape = descTy.getBlockType().getShape();
  return !shape.empty() && shape.back() > 0 &&
         (shape.back() * elemBytes) % 16 == 0;
}

bool valueIsKnownMultipleOf(mlir::Value value, int64_t divisor,
                            unsigned callDepth = 0) {
  if (divisor <= 1)
    return true;
  if (callDepth > 16)
    return false;

  mlir::Attribute constantAttr;
  if (mlir::matchPattern(value, mlir::m_Constant(&constantAttr))) {
    if (auto intAttr = mlir::dyn_cast<mlir::IntegerAttr>(constantAttr))
      return intAttr.getInt() % divisor == 0;
  }

  if (auto arg = mlir::dyn_cast<mlir::BlockArgument>(value)) {
    auto func = mlir::dyn_cast_or_null<mlir::triton::FuncOp>(
        arg.getOwner()->getParentOp());
    if (!func)
      return false;
    auto attr = func.getArgAttrOfType<mlir::IntegerAttr>(arg.getArgNumber(),
                                                         "tt.divisibility");
    if (attr && attr.getInt() % divisor == 0)
      return true;

    auto module = func->getParentOfType<mlir::ModuleOp>();
    if (!module)
      return false;
    bool foundCall = false;
    bool allCallOperandsSatisfy = true;
    module.walk([&](mlir::triton::CallOp call) {
      if (call.resolveCallable() != func.getOperation())
        return;
      foundCall = true;
      if (arg.getArgNumber() >= call.getNumOperands() ||
          !valueIsKnownMultipleOf(call.getOperand(arg.getArgNumber()), divisor,
                                  callDepth + 1))
        allCallOperandsSatisfy = false;
    });
    return foundCall && allCallOperandsSatisfy;
  }

  if (mlir::Operation *def = value.getDefiningOp()) {
    auto attr = def->getAttrOfType<mlir::IntegerAttr>("tt.divisibility");
    return attr && attr.getInt() % divisor == 0;
  }
  return false;
}

bool requiresTMETailPointerFallback(mlir::ModuleOp module) {
  bool requiresFallback = false;

  module.walk([&](mlir::triton::MakeTensorDescOp op) {
    if (requiresFallback)
      return;
    auto descTy = op.getType();
    auto elemBytes =
        getElementSizeBytes(descTy.getBlockType().getElementType());
    if (!elemBytes || !blockShapeSatisfiesTME(descTy, *elemBytes)) {
      requiresFallback = true;
      return;
    }
    if (*elemBytes >= 4)
      return;
    int64_t elementDivisor = 4 / std::gcd<int64_t>(4, *elemBytes);
    if (op.getShape().empty() ||
        !valueIsKnownMultipleOf(op.getShape().back(), elementDivisor))
      requiresFallback = true;
  });

  if (requiresFallback)
    return true;

  module.walk([&](mlir::triton::FuncOp func) {
    if (requiresFallback)
      return;
    if (func.getVisibility() != mlir::SymbolTable::Visibility::Public)
      return;
    for (auto [idx, arg] : llvm::enumerate(func.getArguments())) {
      auto descTy = mlir::dyn_cast<mlir::triton::TensorDescType>(arg.getType());
      if (!descTy)
        continue;
      auto elemBytes =
          getElementSizeBytes(descTy.getBlockType().getElementType());
      if (!elemBytes || !blockShapeSatisfiesTME(descTy, *elemBytes)) {
        requiresFallback = true;
        return;
      }
      if (*elemBytes >= 4)
        continue;
      auto attr = func.getArgAttrOfType<mlir::IntegerAttr>(
          idx, kTMETailDivisibilityAttr);
      if (!attr || attr.getInt() % 4 != 0) {
        requiresFallback = true;
        return;
      }
    }
  });

  return requiresFallback;
}

} // namespace

void init_triton_musa_passes_ttgpuir(py::module m) {
  using namespace mlir::triton;
  m.def("add_mtgpu_to_llvm", [](mlir::PassManager &pm, int32_t capability) {
    pm.addPass(mlir::triton::createConvertMTGPUToLLVMPass(capability));
  });
  m.def(
      "add_to_llvmir",
      [](mlir::PassManager &pm, int32_t capability, bool enableFp8Burst2,
         bool enableVectorArithmetic) {
        pm.addPass(mlir::triton::createConvertTritonMUSAGPUToLLVMPass(
            capability, enableFp8Burst2, enableVectorArithmetic));
      },
      py::arg("pm"), py::arg("capability"),
      py::arg("enable_fp8_burst2") = false,
      py::arg("enable_vector_arithmetic") = false);
  m.def("add_allocate_shared_memory", [](mlir::PassManager &pm,
                                         int32_t capability) {
    pm.addPass(mlir::triton::createAllocateMUSASharedMemoryPass(capability));
  });
  ADD_PASS_OPTION_WRAPPER_2("add_pipeline", mlir::createTritonMUSAGPUPipeline,
                            int, bool);
  ADD_PASS_WRAPPER_0("add_accelerate_matmul",
                     mlir::createTritonMUSAGPUAccelerateMatmul);
  ADD_PASS_WRAPPER_0(
      "add_canonicalize_sqmma_result_conversions",
      mlir::createTritonMUSAGPUCanonicalizeSqmmaResultConversions);
  ADD_PASS_WRAPPER_0("add_convert_sqmma_to_mtgpu",
                     mlir::createTritonMUSAGPUConvertSqmmaToMTGPU);
  ADD_PASS_WRAPPER_0("add_finalize_barriers",
                     mlir::createTritonMUSAGPUFinalizeBarriers);
  ADD_PASS_WRAPPER_0("add_issue_barrier_insertion",
                     mlir::createTritonMUSAGPUIssueBarrierInsertion);
  ADD_PASS_OPTION_WRAPPER_1("add_mark_inplace_loads",
                            mlir::createTritonMUSAGPUMarkInplaceLoads,
                            std::string);
  ADD_PASS_WRAPPER_0("add_optimize_accumulator_init",
                     mlir::createTritonMUSAGPUOptimizeAccumulatorInit);
  ADD_PASS_WRAPPER_0("add_optimize_dot_operands",
                     mlir::createTritonMUSAGPUOptimizeDotOperands);
  ADD_PASS_WRAPPER_0("add_tme_lowering", mlir::createTritonMUSAGPUTMELowering);
  ADD_PASS_WRAPPER_0("add_optimize_descriptor_encoding",
                     mlir::createTritonMUSAGPUOptimizeDescriptorEncoding);
}

void init_triton_mthreads(py::module &&m) {
#ifdef __TLE__
  init_triton_musa_tle_ir(m.def_submodule("ir"));
#endif // __TLE__

  auto passes = m.def_submodule("passes");
  auto ttgpuir = passes.def_submodule("ttgpuir");
  init_triton_musa_passes_ttgpuir(ttgpuir);
#ifdef __TLE__
  init_triton_musa_tle_frontend_passes_ttgpuir(ttgpuir);
  init_triton_musa_tle_dialect_passes_ttgpuir(ttgpuir);
#endif // __TLE__

  // load dialects
  m.def("load_dialects", [](mlir::MLIRContext &context) {
    mlir::DialectRegistry registry;
    registry
        .insert<mlir::triton::mtgpu::MTGPUDialect,
                mlir::triton::musa::MUSADialect, mlir::vector::VectorDialect>();
#ifdef __TLE__
    register_triton_musa_tle_dialects(registry);
#endif
    mlir::registerLLVMDialectTranslation(registry);
    mlir::registerMTVMDialectTranslation(registry);
    context.appendDialectRegistry(registry);
    context.loadAllAvailableDialects();
  });

  m.def("requires_tme_tail_pointer_fallback", [](mlir::ModuleOp &module) {
    return requiresTMETailPointerFallback(module);
  });

  m.def("attach_datalayout", [](llvm::Module &module) {
    const std::string dataLayout = "e-p:64:64:64:64-"
                                   "p1:64:64:64:64-"
                                   "p2:64:64:64:64-"
                                   "p3:32:32-"
                                   "p4:32:32-"
                                   "p5:64:64-"
                                   "i64:64-"
                                   "v16:16-"
                                   "v24:32-"
                                   "v32:32-"
                                   "v48:64-"
                                   "v96:128";
    module.setDataLayout(dataLayout);
  });

  m.def("decorate_kernel_abi",
        [](llvm::Module &module, const std::string &kernelNameHint,
           int32_t maxntidx) -> std::string {
          llvm::Function *kernel = findPrimaryKernel(module, kernelNameHint);
          if (!kernel)
            return "";

          kernel->setCallingConv(llvm::CallingConv::MTGPU_KERNEL);
          addMusaAnnotation(module, *kernel, "kernel", 1);
          addMusaAnnotation(module, *kernel, "maxntidx",
                            std::max<int32_t>(1, maxntidx));
          return kernel->getName().str();
        });

  m.def("canonicalize_lane_shuffles", &canonicalizeLaneShuffles);
  m.def("lower_mp31_fp32_adds", &lowerMp31Fp32Adds);
  m.def("module_uses_mulhi_helper",
        [](llvm::Module &module) { return moduleUsesMulhiHelper(module); });
}
