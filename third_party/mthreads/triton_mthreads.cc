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
#include "llvm/Analysis/InstructionSimplify.h"
#include "llvm/IR/CallingConv.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InlineAsm.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Transforms/InstCombine/InstCombine.h"
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

unsigned tieSignedVectorConversions(llvm::Module &module) {
  unsigned changed = 0;
  for (auto &fn : module) {
    if (fn.hasFnAttribute(llvm::Attribute::StrictFP))
      continue;
    llvm::SmallVector<llvm::SIToFPInst *> conversions;
    for (auto &bb : fn)
      for (auto &inst : bb)
        if (auto *cast = llvm::dyn_cast<llvm::SIToFPInst>(&inst)) {
          auto *src = llvm::dyn_cast<llvm::FixedVectorType>(cast->getSrcTy());
          auto *dst = llvm::dyn_cast<llvm::FixedVectorType>(cast->getDestTy());
          if (src && dst && src->getNumElements() == 4 &&
              dst->getNumElements() == 4 &&
              src->getElementType()->isIntegerTy(32) &&
              dst->getElementType()->isFloatTy())
            conversions.push_back(cast);
        }
    for (auto *cast : conversions) {

      auto *type =
          llvm::FunctionType::get(cast->getSrcTy(), {cast->getSrcTy()}, false);
      auto *assembly = llvm::InlineAsm::get(type, "CVT.RTE.BST4 $0.f32, $1.s32",
                                            "=R,0", false);
      llvm::IRBuilder<> builder(cast);
      builder.SetCurrentDebugLocation(cast->getDebugLoc());
      auto *bits = builder.CreateCall(assembly, {cast->getOperand(0)},
                                      "musa.cvt.tied.bits");
      bits->setDoesNotAccessMemory();
      bits->setDoesNotThrow();
      auto *result =
          builder.CreateBitCast(bits, cast->getDestTy(), "musa.cvt.tied.fp32");
      cast->replaceAllUsesWith(result);
      cast->eraseFromParent();
      ++changed;
    }
  }
  return changed;
}

unsigned fuseFp8PackPairSums(llvm::Module &module) {
  unsigned changed = 0;
  llvm::SmallPtrSet<llvm::Function *, 8> modified;
  auto &context = module.getContext();
  auto *f32 = llvm::Type::getFloatTy(context);
  auto *i32 = llvm::Type::getInt32Ty(context);
  auto *v4f = llvm::FixedVectorType::get(f32, 4);
  auto *resultTy = llvm::StructType::get(context, {i32, f32, f32});
  auto *signature =
      llvm::FunctionType::get(resultTy, {v4f, f32, f32, f32, f32}, false);
  for (auto &fn : module) {
    if (fn.hasFnAttribute(llvm::Attribute::StrictFP))
      continue;
    for (auto &bb : fn) {
      llvm::SmallVector<llvm::CallInst *> casts;
      for (auto &inst : bb) {
        auto *call = llvm::dyn_cast<llvm::CallInst>(&inst);
        auto *assembly =
            call ? llvm::dyn_cast<llvm::InlineAsm>(call->getCalledOperand())
                 : nullptr;
        if (assembly && !assembly->hasSideEffects() &&
            assembly->getAsmString() == "CVT.RTE.DENORM.BST4 $0.e4m3, $1.f32" &&
            assembly->getConstraintString() == "=R,R" &&
            !call->isConvergent() && call->getType() == i32 &&
            call->arg_size() == 1 && call->getArgOperand(0)->getType() == v4f)
          casts.push_back(call);
      }
      for (auto *cast : casts) {
        auto *vector = cast->getArgOperand(0);
        llvm::BinaryOperator *pairs[2] = {nullptr, nullptr};
        bool reversed[2] = {false, false};
        for (auto *user : vector->users()) {
          auto *add = llvm::dyn_cast<llvm::BinaryOperator>(user);
          if (!add || add->getOpcode() != llvm::Instruction::FAdd ||
              add->getParent() != &bb || !cast->comesBefore(add))
            continue;
          auto *other = add->getOperand(add->getOperand(0) == vector ? 1 : 0);
          auto *shuffle = llvm::dyn_cast<llvm::ShuffleVectorInst>(other);
          if (!shuffle || shuffle->getOperand(0) != vector ||
              shuffle->getType() != v4f)
            continue;
          const auto mask = shuffle->getShuffleMask();
          for (unsigned pair = 0; pair < 2; ++pair) {
            const unsigned lane = pair * 2;
            bool exact = mask[lane] == int(lane + 1);
            for (unsigned j = 0; j < 4; ++j)
              if (j != lane && mask[j] >= 0)
                exact = false;
            if (exact && !pairs[pair]) {
              pairs[pair] = add;
              reversed[pair] = add->getOperand(0) != vector;
            }
          }
        }
        if (!pairs[0] || !pairs[1])
          continue;

        llvm::IRBuilder<> builder(cast);
        builder.SetCurrentDebugLocation(cast->getDebugLoc());
        llvm::SmallVector<llvm::Value *> inputs{vector};
        for (unsigned lane = 0; lane < 4; ++lane) {
          unsigned sourceLane = reversed[lane / 2] ? lane ^ 1 : lane;
          inputs.push_back(builder.CreateExtractElement(vector, sourceLane));
        }
        // Earlier outputs cannot overlap still-live inputs. All outputs are
        // defined by real instructions.
        std::string body = "CVT.RTE.DENORM.BST4 $0.e4m3, $3.f32\n\t"
                           "FOP.ADD $1, $4, $5\n\tFOP.ADD $2, $6, $7";
        auto *assembly = llvm::InlineAsm::get(signature, body,
                                              "=&R,=&R,=R,R,R,R,R,R", false);
        auto *fused = builder.CreateCall(assembly, inputs, "musa.pack.pairs");
        fused->setDoesNotThrow();
        fused->setMemoryEffects(llvm::MemoryEffects::none());
        cast->replaceAllUsesWith(builder.CreateExtractValue(fused, 0));

        for (unsigned pair = 0; pair < 2; ++pair) {
          auto *value = builder.CreateExtractValue(fused, pair + 1);
          auto *replacement = builder.CreateInsertElement(
              llvm::PoisonValue::get(v4f), value, pair * 2);
          pairs[pair]->replaceAllUsesWith(replacement);
          pairs[pair]->eraseFromParent();
        }

        cast->eraseFromParent();
        ++changed;
        modified.insert(&fn);
      }
    }
  }
  // Canonicalize the single-defined-lane vectors we just introduced before
  // passing the module to the SDK. In particular, turn the surviving local
  // reduction into scalar operations instead of feeding poison-heavy vector
  // expressions to the SDK's burst-combine pass. No FP fast-math is enabled.
  if (!modified.empty()) {
    llvm::LoopAnalysisManager loops;
    llvm::FunctionAnalysisManager functions;
    llvm::CGSCCAnalysisManager cgscc;
    llvm::ModuleAnalysisManager modules;
    llvm::PassBuilder builder;
    builder.registerModuleAnalyses(modules);
    builder.registerCGSCCAnalyses(cgscc);
    builder.registerFunctionAnalyses(functions);
    builder.registerLoopAnalyses(loops);
    builder.crossRegisterProxies(loops, functions, cgscc, modules);
    for (auto *fn : modified) {
      llvm::FunctionPassManager passes;
      passes.addPass(llvm::InstCombinePass());
      passes.run(*fn, functions);
    }
  }
  return changed;
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

unsigned orderSqmmaOverlapAdds(llvm::Module &module, unsigned maxAdds = 0,
                               bool tieInput = false) {
  llvm::SmallVector<std::pair<llvm::BinaryOperator *, bool>, 32> work;
  for (auto &fn : module) {
    if (fn.isDeclaration() || fn.hasFnAttribute(llvm::Attribute::StrictFP))
      continue;
    for (auto &bb : fn) {
      bool pendingFP8 = false;
      unsigned orderedInRegion = 0;
      for (auto &inst : bb) {
        if (auto *call = llvm::dyn_cast<llvm::CallBase>(&inst)) {
          if (auto *callee = call->getCalledFunction()) {
            auto name = callee->getName();
            if (name == "llvm.musa.sqmma.wait")
              pendingFP8 = false;
            else if (name.starts_with("llvm.musa.sqmma.") &&
                     name.ends_with(".mma")) {
              pendingFP8 = name.starts_with("llvm.musa.sqmma.e4m3.") ||
                           name.starts_with("llvm.musa.sqmma.e5m2.");
              orderedInRegion = 0;
            }
          }
        }
        auto *add = llvm::dyn_cast<llvm::BinaryOperator>(&inst);
        if (pendingFP8 && add && add->getOpcode() == llvm::Instruction::FAdd &&
            !add->getFastMathFlags().any()) {
          auto *vec = llvm::dyn_cast<llvm::FixedVectorType>(add->getType());
          if (add->getType()->isFloatTy() ||
              (vec && vec->getElementType()->isFloatTy() &&
               (vec->getNumElements() == 2 || vec->getNumElements() == 4))) {
            bool ordered = !maxAdds || orderedInRegion < maxAdds;
            work.push_back({add, ordered});
            if (ordered)
              ++orderedInRegion;
          }
        }
      }
    }
  }
  // Analyze every original use before changing any vector's users.
  llvm::DenseMap<llvm::Value *, unsigned> demanded;
  llvm::SmallPtrSet<llvm::Value *, 16> visiting;
  for (auto [add, mustOrder] : work)
    if (add->getType()->isVectorTy())
      demandedSqmmaAddLanes(add, demanded, visiting);
  unsigned changed = 0;
  for (auto [add, mustOrder] : work) {
    llvm::IRBuilder<> builder(add);
    auto *vec = llvm::dyn_cast<llvm::FixedVectorType>(add->getType());
    unsigned mask = vec ? demanded.lookup(add) : 1;
    if (!mask)
      continue;

    if (vec && mask != ((1u << vec->getNumElements()) - 1)) {
      auto *scalar = vec->getElementType();
      auto *type = llvm::FunctionType::get(scalar, {scalar, scalar}, false);
      // Tying a destructive ADD is a register-allocation constraint only.
      // LLVM must preserve any later use of the old input with a copy; this
      // neither changes rounding nor removes the original synchronization.
      std::string constraints = tieInput ? "=R,0,R" : "=R,R,R";
      if (mustOrder)
        constraints += ",~{memory}";
      auto *code = llvm::InlineAsm::get(type, "FOP.ADD $0, $1, $2", constraints,
                                        mustOrder);
      llvm::Value *result = llvm::PoisonValue::get(vec);
      for (unsigned i = 0; i < vec->getNumElements(); ++i) {
        if (!(mask & (1u << i)))
          continue;
        auto *ordered = builder.CreateCall(
            code, {builder.CreateExtractElement(add->getOperand(0), i),
                   builder.CreateExtractElement(add->getOperand(1), i)});
        ordered->setDebugLoc(add->getDebugLoc());
        ordered->setDoesNotThrow();
        result = builder.CreateInsertElement(result, ordered, i);
      }
      add->replaceAllUsesWith(result);
      add->eraseFromParent();
      changed += mustOrder;
      continue;
    }
    auto *type = llvm::FunctionType::get(
        add->getType(), {add->getType(), add->getType()}, false);
    std::string opcode = "FOP.ADD";
    if (auto *vec = llvm::dyn_cast<llvm::FixedVectorType>(add->getType()))
      opcode += ".BST" + std::to_string(vec->getNumElements());
    std::string constraints = tieInput ? "=R,0,R" : "=R,R,R";
    if (mustOrder)
      constraints += ",~{memory}";
    auto *code = llvm::InlineAsm::get(type, opcode + " $0, $1, $2", constraints,
                                      mustOrder);
    auto *ordered =
        builder.CreateCall(code, {add->getOperand(0), add->getOperand(1)});
    ordered->setDebugLoc(add->getDebugLoc());
    ordered->setDoesNotThrow();
    add->replaceAllUsesWith(ordered);
    add->eraseFromParent();
    changed += mustOrder;
  }

  return changed;
}

unsigned packSqmmaOverlapAdds(llvm::Module &module) {
  unsigned packed = 0;
  for (auto &fn : module) {
    if (fn.isDeclaration() || fn.hasFnAttribute(llvm::Attribute::StrictFP))
      continue;
    for (auto &bb : fn) {
      llvm::SmallVector<llvm::CallInst *, 4> batch;
      llvm::SmallVector<llvm::Instruction *, 16> moves;
      llvm::SmallPtrSet<llvm::Instruction *, 16> planned;
      llvm::InlineAsm *batchCode = nullptr;
      bool pendingFP8 = false;
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
            b = builder.CreateInsertElement(b, batch[i]->getArgOperand(1), i);
          }
          auto *type = llvm::FunctionType::get(vec, {vec, vec}, false);
          auto *code = llvm::InlineAsm::get(
              type, "FOP.ADD.BST" + std::to_string(width) + " $0, $1, $2",
              batchCode->getConstraintString(), batchCode->hasSideEffects());
          auto *call = builder.CreateCall(code, {a, b});
          call->setDoesNotThrow();
          call->setDebugLoc(anchor->getDebugLoc());
          for (unsigned i = 0; i < width; ++i)
            batch[i]->replaceAllUsesWith(builder.CreateExtractElement(call, i));
          for (unsigned i = 0; i < width; ++i)
            batch[i]->eraseFromParent();
          ++packed;
        }
        batch.clear();
        moves.clear();
        planned.clear();
        batchCode = nullptr;
      };
      // Snapshot the original order; flush removes earlier batch members only.
      llvm::SmallVector<llvm::Instruction *, 128> original;
      for (auto &inst : bb)
        original.push_back(&inst);
      for (auto *inst : original) {
        auto *call = llvm::dyn_cast<llvm::CallInst>(inst);
        auto *code =
            call && call->isInlineAsm()
                ? llvm::dyn_cast<llvm::InlineAsm>(call->getCalledOperand())
                : nullptr;
        bool candidate = pendingFP8 && code && call->getType()->isFloatTy() &&
                         call->arg_size() == 2 &&
                         code->getAsmString() == "FOP.ADD $0, $1, $2" &&
                         (code->getConstraintString() == "=R,R,R,~{memory}" ||
                          code->getConstraintString() == "=R,0,R,~{memory}" ||
                          code->getConstraintString() == "=R,R,R" ||
                          code->getConstraintString() == "=R,0,R");
        if (candidate) {
          if (batchCode && batchCode != code)
            flush();
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
              !prepare(call->getArgOperand(1))) {
            // A failed candidate must not contribute partially collected
            // register views to the preceding batch's hoist plan.
            moves.resize(previousMoves);
            planned.clear();
            for (auto *move : moves)
              planned.insert(move);
            flush();
          }
          if (batch.empty())
            batchCode = code;
          batch.push_back(call);
          if (batch.size() == 4)
            flush();
          continue;
        }
        if (call || inst->mayReadOrWriteMemory() ||
            inst->mayHaveSideEffects() || inst->isTerminator())
          flush();
        if (call)
          if (auto *callee = call->getCalledFunction()) {
            auto name = callee->getName();
            if (name == "llvm.musa.sqmma.wait")
              pendingFP8 = false;
            else if (name.starts_with("llvm.musa.sqmma.") &&
                     name.ends_with(".mma"))
              pendingFP8 = name.starts_with("llvm.musa.sqmma.e4m3.") ||
                           name.starts_with("llvm.musa.sqmma.e5m2.");
          }
      }
      flush();
    }
  }
  return packed;
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
      [](mlir::PassManager &pm, int32_t capability, bool enableFp8Burst2) {
        pm.addPass(mlir::triton::createConvertTritonMUSAGPUToLLVMPass(
            capability, enableFp8Burst2));
      },
      py::arg("pm"), py::arg("capability"),
      py::arg("enable_fp8_burst2") = false);
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
  m.def("tie_signed_vector_conversions", &tieSignedVectorConversions,
        py::arg("module"));
  m.def("fuse_fp8_pack_pair_sums", &fuseFp8PackPairSums, py::arg("module"));
  m.def("order_sqmma_overlap_adds", &orderSqmmaOverlapAdds, py::arg("module"),
        py::arg("max_adds") = 0, py::arg("tie_input") = false);
  m.def("pack_sqmma_overlap_adds", &packSqmmaOverlapAdds);
  m.def("module_uses_mulhi_helper",
        [](llvm::Module &module) { return moduleUsesMulhiHelper(module); });
}
