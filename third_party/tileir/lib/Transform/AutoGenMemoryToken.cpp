#include "Transform/Passes.h"
#include "TritonToTileIR/Utils.h"
#include "Utils/Utils.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/Support/Debug.h"

using namespace mlir;

// MLIR pass TableGen now uses per-pass macros (GEN_PASS_DEF_*).
namespace mlir {
namespace triton {
#define GEN_PASS_DEF_AUTOGENMEMORYTOKEN
#include "Transform/Passes.h.inc"
} // namespace triton
} // namespace mlir

#define DEBUG_TYPE "add-memory-token"

/*
 * This Pass file aims to add memory tokens automatically to ensure tileIR's
 * compatibility with Triton. We add memory tokens based on the following rules:
 *  - If a kernel contains memory ops with input token, which means user has
 * already added some tokens in the kernel, we will keep the original token flow
 * unchanged and do nothing.
 *  - If a kernel contains a triton debug_barrier op, we add memory tokens for
 * all memory ops in a sequential way.
 *  - If a kernel contains sets of memory ops which acesses the same data, we
 * will apply memory tokens to maintain their access order.
 *
 * Implementation:
 *  We organize memory ops into sequances, where each sequence access the same
 * memory data and their access order need to be maintained by memory token. To
 * distinguish different sequances, we assign SID for each sequence and add
 * function getMemOpSeqId to map op to its sequence SID. There are 2 types of
 * memory ops:
 *    - one is ptr memory ops, which uses tensor of pointers like LoadPtrTkoOp.
 *    - the other is view memory ops, which uses tensor of views like
 * LoadViewTkoOp. These two kind of memory ops use different ways to represent
 * their memory accessing pattern. So for ptr memory ops, we hash their ptr
 * value as SID; for view ops, we hash their view value and index values as SID.
 *  The main transformation is done in Pass AutoGenMemoryTokenPass, which
 * performs two walks for the entire input IR. One is to collect memory sequence
 * info, and after processing collected data (to make sure there are memory
 * tokens required to be added), another walk is performed to add memory tokens
 * based on the sequence info.
 *
 * In this version of implementation, there are some scenarios we cannot handle:
 *    1. if some ptr ops and view ops access the same data, we will not be able
 * to detect that and put them into the same sequence.
 *    2. if some memory ops' access memory overlap, we will not be able to find
 * out.
 *    3. if users pass 2 ptrs pointing to the same memory location, we will not
 * be able to find out.
 */

namespace {

using namespace mlir::triton;

using SeqId = std::size_t;
using SeqIdSet = llvm::SmallSet<SeqId, 16>;
using SeqIdVec = SmallVector<SeqId>;
static const SeqId BARRIER_SEQ_ID = 1;

using SeqTokensBase = DenseMap<SeqId, Value>;
class SeqTokens : public SeqTokensBase {
  SeqIdSet updatedSids;

public:
  SeqTokens() = default;

  void update(SeqId id, Value token) {
    (*this)[id] = token;
    updatedSids.insert(id);
  }

  void update(const SeqTokens &newTokens) {
    for (const auto &pair : newTokens)
      update(pair.first, pair.second);
  }

  SeqIdVec
  aggregate(const SmallVector<std::reference_wrapper<SeqTokens>> &tokenSets) {
    SeqIdSet allUpdatedSids;
    for (auto &tokenRef : tokenSets) {
      for (const auto &p : tokenRef.get())
        allUpdatedSids.insert(p.first);
    }

    // Here we use a vector to make sure all results have the same sid order
    SeqIdVec allUpdatedSidsVec(allUpdatedSids.begin(), allUpdatedSids.end());

    auto genResult = [&](SeqTokens &tokens) {
      for (const auto &sid : allUpdatedSidsVec) {
        if (tokens.find(sid) == tokens.end())
          tokens[sid] = (*this)[sid];
      }
    };

    for (auto &tokenRef : tokenSets)
      genResult(tokenRef.get());

    return allUpdatedSidsVec;
  }

  SeqTokens getUpdatedTokens() {
    SeqTokens outTokens;
    for (const auto &sid : updatedSids)
      outTokens[sid] = (*this)[sid];
    return outTokens;
  }

  void cleanUpdatedSids() { updatedSids.clear(); }
};
using OpToTokens = SmallVector<std::pair<Operation *, SeqTokens>, 4>;

struct MemSeqInfo {
  size_t writeMemOpCounter = 0;
  size_t memOpCounter = 0; // used for both preprocessing and transform
  bool ignored = false;

  MemSeqInfo() = default;
};

struct BlockMemSeqs {
  // collected data from preprocessing walk
  bool hasBarrierOp = false;
  bool hasMemToken = false;
  DenseMap<SeqId, MemSeqInfo> srcToMemSeqInfoMap;

  // runtime data for transform walk
  int loop_level = 0;
  int if_level = 0;

  BlockMemSeqs() = default;
  SeqTokens getBlockInitTokens(Block *block, IRRewriter &rewriter) {
    rewriter.setInsertionPointToStart(block);
    SeqTokens tokens;
    for (const auto &pair : srcToMemSeqInfoMap) {
      if (pair.second.ignored)
        continue; // only make new token for un-ignored sequences
      auto makeTkOp =
          cuda_tile::MakeTokenOp::create(rewriter, block->front().getLoc());
      tokens[pair.first] = makeTkOp.getResult();
    }
    return tokens;
  }

  void clear() {
    hasBarrierOp = false;
    hasMemToken = false;
    loop_level = 0;
    if_level = 0;
    srcToMemSeqInfoMap.clear();
  }
};

bool isMemOp(Operation *op) {
  return isa<cuda_tile::LoadPtrTkoOp, cuda_tile::StorePtrTkoOp,
             cuda_tile::LoadViewTkoOp, cuda_tile::StoreViewTkoOp,
             cuda_tile::AtomicRMWTkoOp, cuda_tile::AtomicCASTkoOp>(op);
}

bool isWriteMemOp(Operation *op) {
  return isa<cuda_tile::StorePtrTkoOp, cuda_tile::StoreViewTkoOp,
             cuda_tile::AtomicRMWTkoOp, cuda_tile::AtomicCASTkoOp>(op);
}

class AutoGenMemoryTokenPass
    : public ::mlir::triton::impl::AutoGenMemoryTokenBase<
          AutoGenMemoryTokenPass> {
  // Data members
  BlockMemSeqs currentBlockMemSeqs;

  /// Generate SeqId for a specific memory op
  SeqId getMemOpSeqId(Operation *op) {
    if (currentBlockMemSeqs.hasBarrierOp)
      return BARRIER_SEQ_ID;

    Value val = Value();
    SmallVector<Value> indexes;
    if (auto loadOp = dyn_cast<cuda_tile::LoadPtrTkoOp>(op)) {
      val = loadOp.getSource();
      if (loadOp.getToken())
        currentBlockMemSeqs.hasMemToken = true;
    } else if (auto storeOp = dyn_cast<cuda_tile::StorePtrTkoOp>(op)) {
      val = storeOp.getDestination();
      if (storeOp.getToken())
        currentBlockMemSeqs.hasMemToken = true;
    } else if (auto atmRMWOp = dyn_cast<cuda_tile::AtomicRMWTkoOp>(op)) {
      val = atmRMWOp.getPointers();
      if (atmRMWOp.getToken())
        currentBlockMemSeqs.hasMemToken = true;
    } else if (auto atmCASOp = dyn_cast<cuda_tile::AtomicCASTkoOp>(op)) {
      val = atmCASOp.getPointers();
      if (atmCASOp.getToken())
        currentBlockMemSeqs.hasMemToken = true;
    } else if (auto viewOp = dyn_cast<cuda_tile::LoadViewTkoOp>(op)) {
      val = viewOp.getView();
      indexes = viewOp.getIndex();
      if (viewOp.getToken())
        currentBlockMemSeqs.hasMemToken = true;
    } else if (auto viewOp = dyn_cast<cuda_tile::StoreViewTkoOp>(op)) {
      val = viewOp.getView();
      indexes = viewOp.getIndex();
      if (viewOp.getToken())
        currentBlockMemSeqs.hasMemToken = true;
    }
    // TODO: does different order of index generate the same hash value?
    if (!val)
      return 0;
    llvm::hash_code hash = mlir::hash_value(val);
    for (const Value &val : indexes)
      hash = llvm::hash_combine(hash, mlir::hash_value(val));
    return hash;
  }

  /// Get function/entry block and name from operation
  Block *getFuncBlock(Operation *op, std::string &funcName) {
    if (auto entryOp = dyn_cast<cuda_tile::EntryOp>(op)) {
      funcName = entryOp.getSymName().str();
      return &entryOp.getBody().front();
    }
    return nullptr;
  }

  /// Handle memory op
  /// 1. add input token to op's operands(if token is not null)
  /// 2. update operandSegmentSizes attribute(if exists)
  /// 3. return the updated token value from op's result values.
  template <typename OpTy>
  Value updateMemOpWithToken(OpTy *op, Value token, IRRewriter &rewriter) {
    SmallVector<Value> newOperands = llvm::to_vector(op->getOperands());

    if (token) {
      // append token operand
      newOperands.push_back(token);
      // update operand segment sizes attribute
      if (auto segmentSizesAttr = op->getAttr("operandSegmentSizes")) {
        auto arrayAttr = dyn_cast<DenseI32ArrayAttr>(segmentSizesAttr);
        SmallVector<int32_t> newSegmentSizes =
            llvm::to_vector(arrayAttr.asArrayRef());
        newSegmentSizes.back() =
            1; // the last segment indicates whether token operand exists
        op->setAttr("operandSegmentSizes",
                    rewriter.getDenseI32ArrayAttr(newSegmentSizes));
      }
    }
    op->setOperands(newOperands);

    return op->getResults().back();
  }

  /// Handle terminator ops by adding token to its operands.
  template <typename OpTy>
  void updateTermOpWithToken(OpTy *op, SeqTokens &tokens, SeqIdVec &sids) {
    SmallVector<Value> newOperands = llvm::to_vector(op->getOperands());
    // use sids to ensure the order of tokens
    for (auto s : sids)
      newOperands.push_back(tokens[s]);
    op->setOperands(newOperands);
  }

  SeqTokens handleIfOpTokens(cuda_tile::IfOp ifOp, SeqTokens tokens,
                             IRRewriter &rewriter, OpToTokens *termOps) {
    tokens.cleanUpdatedSids();
    currentBlockMemSeqs.if_level++;

    // handle token in then and else block
    SeqTokens thenTokens =
        addMemTokenForBlock(ifOp.getThenBlock(), tokens, rewriter, termOps);
    SeqTokens elseTokens =
        addMemTokenForBlock(ifOp.getElseBlock(), tokens, rewriter, termOps);
    SmallVector<std::reference_wrapper<SeqTokens>> tokenList{
        std::ref(thenTokens), std::ref(elseTokens)};
    SeqIdVec sids = tokens.aggregate(tokenList);

    // skip those sequences which will not be used in later memory ops
    if (!currentBlockMemSeqs.loop_level) {
      SeqIdVec newSids;
      for (const auto &sid : sids) {
        if (currentBlockMemSeqs.srcToMemSeqInfoMap[sid].memOpCounter == 0) {
          thenTokens.erase(thenTokens.find(sid));
          elseTokens.erase(elseTokens.find(sid));
        } else
          newSids.push_back(sid);
      }
      sids.swap(newSids);
    }

    auto insertTermOp = [&]() {
      rewriter.createBlock(&ifOp.getElseRegion());
      rewriter.setInsertionPointToEnd(ifOp.getElseBlock());
      cuda_tile::YieldOp::create(rewriter, ifOp.getLoc());
    };

    // if either branch has memory token update, we need to update terminate ops
    // of this ifOp
    if (!sids.empty() && isa<cuda_tile::YieldOp>(ifOp.getThenTerminator())) {
      updateTermOpWithToken(ifOp.getThenTerminator(), thenTokens, sids);
      if (!ifOp.getElseTerminator())
        insertTermOp();
      updateTermOpWithToken(ifOp.getElseTerminator(), elseTokens, sids);

      // append token type to ifOp's return type
      rewriter.setInsertionPointAfter(ifOp);
      SmallVector<Type> resultTypes = llvm::to_vector(ifOp.getResultTypes());
      resultTypes.append(sids.size(),
                         cuda_tile::TokenType::get(rewriter.getContext()));
      auto newIfOp =
          cuda_tile::IfOp::create(rewriter, ifOp.getLoc(), resultTypes,
                                  ifOp.getCondition(), ifOp->getAttrs());
      rewriter.inlineRegionBefore(ifOp.getThenRegion(), newIfOp.getThenRegion(),
                                  newIfOp.getThenRegion().begin());
      rewriter.inlineRegionBefore(ifOp.getElseRegion(), newIfOp.getElseRegion(),
                                  newIfOp.getElseRegion().begin());
      // update token
      for (size_t i = 0; i < sids.size(); i++)
        tokens.update(sids[i], newIfOp.getResult(ifOp.getNumResults() + i));

      // replace old result values with new ones, except new token
      SmallVector<Value> replacementResults;
      for (size_t i = 0; i < ifOp.getNumResults(); i++)
        replacementResults.push_back(newIfOp.getResult(i));
      rewriter.replaceOp(ifOp, replacementResults);
    } else if (isa<cuda_tile::BreakOp, cuda_tile::ContinueOp>(
                   ifOp.getThenTerminator())) {
      if (!ifOp.getElseTerminator())
        insertTermOp();
      if (termOps == nullptr) {
        ifOp.emitWarning("unexpected terminator op in if op");
      } else {
        termOps->push_back({ifOp.getThenTerminator(), thenTokens});
        termOps->push_back({ifOp.getElseTerminator(), elseTokens});
      }
    }

    currentBlockMemSeqs.if_level--;
    return tokens.getUpdatedTokens();
  }

  SeqTokens handleForOpTokens(cuda_tile::ForOp forOp, SeqTokens tokens,
                              IRRewriter &rewriter) {
    tokens.cleanUpdatedSids();
    currentBlockMemSeqs.loop_level++;

    // handle token in body block
    OpToTokens localTermOps;
    SeqTokens forTokens =
        addMemTokenForBlock(forOp.getBody(), tokens, rewriter, &localTermOps);
    SmallVector<std::reference_wrapper<SeqTokens>> tokenList{
        std::ref(forTokens)};
    for (auto &opTokenPair : localTermOps)
      tokenList.push_back(std::ref(opTokenPair.second));
    SeqIdVec sids = tokens.aggregate(tokenList);

    // add token to terminator
    if (!sids.empty()) {
      // add token to terminator recursively
      updateTermOpWithToken(forOp.getBody()->getTerminator(), forTokens, sids);
      for (auto &pair : localTermOps)
        updateTermOpWithToken(pair.first, pair.second, sids);

      // append token type to forOp's init values
      SmallVector<Value> newInitArgs = llvm::to_vector(forOp.getInitValues());
      for (const auto &sid : sids)
        newInitArgs.push_back(tokens[sid]);
      // create new loop op
      rewriter.setInsertionPointAfter(forOp);
      auto newForOp = cuda_tile::ForOp::create(
          rewriter, forOp.getLoc(), forOp.getLowerBound(),
          forOp.getUpperBound(), forOp.getStep(), newInitArgs);

      // copy block body
      Block *newBlock = &newForOp.getRegion().front();
      SmallVector<Value> blockArgs;
      size_t originNumArgs = forOp.getBody()->getNumArguments();
      for (auto arg : forOp.getBody()->getArguments())
        blockArgs.push_back(newBlock->getArgument(arg.getArgNumber()));
      rewriter.mergeBlocks(forOp.getBody(), newBlock, blockArgs);

      // update token usage in loop body
      for (size_t i = 0; i < sids.size(); i++) {
        auto arg = newBlock->getArgument(originNumArgs + i);
        tokens[sids[i]].replaceAllUsesExcept(arg, newForOp);
        tokens.update(sids[i], newForOp.getResult(forOp.getNumResults() + i));
      }

      // replace old result values with new ones, except new token
      SmallVector<Value> replacementResults;
      for (size_t i = 0; i < forOp.getNumResults(); i++)
        replacementResults.push_back(newForOp.getResult(i));
      rewriter.replaceOp(forOp, replacementResults);
    }

    currentBlockMemSeqs.loop_level--;
    return tokens.getUpdatedTokens();
  }

  SeqTokens handleLoopOpTokens(cuda_tile::LoopOp loopOp, SeqTokens tokens,
                               IRRewriter &rewriter) {
    tokens.cleanUpdatedSids();
    currentBlockMemSeqs.loop_level++;

    // handle token in body block
    OpToTokens localTermOps;
    SeqTokens loopTokens =
        addMemTokenForBlock(loopOp.getBody(), tokens, rewriter, &localTermOps);
    SmallVector<std::reference_wrapper<SeqTokens>> tokenList{
        std::ref(loopTokens)};
    for (auto &opTokenPair : localTermOps)
      tokenList.push_back(std::ref(opTokenPair.second));
    SeqIdVec sids = tokens.aggregate(tokenList);

    // add token to terminator
    if (!sids.empty()) {
      // add token to terminator recursively
      updateTermOpWithToken(loopOp.getBody()->getTerminator(), loopTokens,
                            sids);
      for (auto &pair : localTermOps)
        updateTermOpWithToken(pair.first, pair.second, sids);

      // append token type to loopOp's operand
      SmallVector<Value> newOperands = llvm::to_vector(loopOp->getOperands());
      for (const auto &sid : sids)
        newOperands.push_back(tokens[sid]);
      // append token type to loopOp's return type
      auto tokenType = cuda_tile::TokenType::get(rewriter.getContext());
      SmallVector<Type> resultTypes = llvm::to_vector(loopOp.getResultTypes());
      resultTypes.append(sids.size(), tokenType);
      // create new loop op
      rewriter.setInsertionPointAfter(loopOp);
      auto newLoopOp =
          cuda_tile::LoopOp::create(rewriter, loopOp.getLoc(), resultTypes,
                                    newOperands, loopOp->getAttrs());

      // append token to loop block's argument list
      Block *newBlock = rewriter.createBlock(&newLoopOp.getRegion());
      for (Type type : loopOp.getBody()->getArgumentTypes())
        newBlock->addArgument(type, loopOp.getLoc());
      for (const auto &sid : sids)
        newBlock->addArgument(tokenType, loopOp.getLoc());

      // copy block body
      size_t originNumArgs = loopOp.getBody()->getNumArguments();
      SmallVector<Value> blockArgs;
      for (auto arg : loopOp.getBody()->getArguments())
        blockArgs.push_back(newBlock->getArgument(arg.getArgNumber()));
      rewriter.mergeBlocks(loopOp.getBody(), newBlock, blockArgs);

      // update token usage in loop body
      for (size_t i = 0; i < sids.size(); i++) {
        auto arg = newBlock->getArgument(originNumArgs + i);
        tokens[sids[i]].replaceAllUsesExcept(arg, newLoopOp);
        tokens.update(sids[i], newLoopOp.getResult(loopOp.getNumResults() + i));
      }

      // replace old result values with new ones, except new token
      SmallVector<Value> replacementResults;
      for (size_t i = 0; i < loopOp.getNumResults(); i++)
        replacementResults.push_back(newLoopOp.getResult(i));
      rewriter.replaceOp(loopOp, replacementResults);
    }

    currentBlockMemSeqs.loop_level--;
    return tokens.getUpdatedTokens();
  }

  /// Propagates memory tokens through a block and its nested control flow.
  ///
  /// This function performs a pre-order walk of all operations in the block,
  /// adding memory tokens to memory operations in sequential order. For control
  /// flow operations (if/for/loop), it recursively processes nested blocks and
  /// updates tokens appropriately.
  ///
  /// @param block: The block to process.
  /// @param tokens: The initial tokens to propagate, the size of tokens should
  /// be
  ///                the number of sequences which requires adding memory
  ///                tokens.
  /// @param rewriter: IR rewriter for modifications.
  /// @param termOps: Optional collector for terminator operations with their
  ///                 tokens.
  ///         (e.g. loopOp -> ifOp -> breakOp)
  /// @return The final token value after processing all operations in the
  /// block.
  ///         The result will only contain the updated token value.
  SeqTokens addMemTokenForBlock(Block *block, SeqTokens tokens,
                                IRRewriter &rewriter,
                                OpToTokens *termOps = nullptr) {
    tokens.cleanUpdatedSids();
    if (!block || block->empty())
      return SeqTokens();

    block->walk<WalkOrder::PreOrder>([&](Operation *childOp) {
      if (auto ifOp = dyn_cast<cuda_tile::IfOp>(childOp)) {
        SeqTokens newTokens = handleIfOpTokens(ifOp, tokens, rewriter, termOps);
        tokens.update(newTokens);
        return WalkResult::skip();
      } else if (auto forOp = dyn_cast<cuda_tile::ForOp>(childOp)) {
        SeqTokens newTokens = handleForOpTokens(forOp, tokens, rewriter);
        tokens.update(newTokens);
        return WalkResult::skip();
      } else if (auto loopOp = dyn_cast<cuda_tile::LoopOp>(childOp)) {
        SeqTokens newTokens = handleLoopOpTokens(loopOp, tokens, rewriter);
        tokens.update(newTokens);
        return WalkResult::skip();
      } else if (isMemOp(childOp)) {
        SeqId sid = getMemOpSeqId(childOp);
        auto &seq = currentBlockMemSeqs.srcToMemSeqInfoMap[sid];
        // only add memory token for memory op sequences with more than 1 memory
        // op
        if (!seq.ignored) {
          tokens.update(sid,
                        updateMemOpWithToken(childOp, tokens[sid], rewriter));
          seq.memOpCounter--;
        }
      }
      return WalkResult::advance();
    });

    return tokens.getUpdatedTokens();
  };

public:
  AutoGenMemoryTokenPass() = default;
  AutoGenMemoryTokenPass(bool enable_autogen_alias_mem_token) {
    this->enable_autogen_alias_mem_token = enable_autogen_alias_mem_token;
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp mod = getOperation();
    IRRewriter rewriter(context);

    mod->walk<WalkOrder::PreOrder>([&](Operation *outOp) {
      std::string fname;
      Block *body = getFuncBlock(outOp, fname);
      if (!body)
        return WalkResult::advance();

      // 1. Preprocess walk: traverse the block to collect info
      currentBlockMemSeqs.clear();
      body->walk([&](Operation *op) {
        //    1.1 check if func/entry op contains debug_barrier op, if yes, all
        //    memory ops will map to the same SeqId
        if (isa<mlir::gpu::BarrierOp>(op)) {
          currentBlockMemSeqs.hasBarrierOp = true;
          rewriter.eraseOp(op);
          return WalkResult::advance();
        }
        //    1.2 record all memory ops (possibly needs to be ordered)
        if (!isMemOp(op))
          return WalkResult::advance();
        SeqId sid = getMemOpSeqId(op);
        //    1.3 check if any memory op has input token, if yes, no need to
        //    proceed
        if (currentBlockMemSeqs.hasMemToken)
          return WalkResult::interrupt();
        currentBlockMemSeqs.srcToMemSeqInfoMap[sid].memOpCounter++;
        if (isWriteMemOp(op))
          currentBlockMemSeqs.srcToMemSeqInfoMap[sid].writeMemOpCounter++;
        LLVM_DEBUG(llvm::errs()
                   << "Add Operation to SeqId " << sid << ": " << *op << "\n");
        return WalkResult::advance();
      });

      // 2. Check phase: walk through collected info to decide whether to run
      // transform walk
      //      2.1 if no barrier op and disable autogen alias mem token, skip
      if (this->enable_autogen_alias_mem_token == false &&
          !currentBlockMemSeqs.hasBarrierOp) {
        return WalkResult::skip();
      }
      //      2.2 if contains user-defined mem token, skip
      if (currentBlockMemSeqs.hasMemToken) {
        if (currentBlockMemSeqs.hasBarrierOp) {
          outOp->emitWarning(
              "debug_barrier should not be added when memory tokens are added"
              "manually, debug_barrier op will be ignored.");
        }
        return WalkResult::skip();
      }
      //      2.3 map all mem op to a single sequence if there is debug_barrier
      //      op
      if (currentBlockMemSeqs.hasBarrierOp) {
        size_t writeMemOpCounter = 0, memOpCounter = 0;
        for (auto &p : currentBlockMemSeqs.srcToMemSeqInfoMap) {
          writeMemOpCounter += p.second.writeMemOpCounter;
          memOpCounter += p.second.memOpCounter;
        }
        currentBlockMemSeqs.srcToMemSeqInfoMap.clear();
        currentBlockMemSeqs.srcToMemSeqInfoMap[BARRIER_SEQ_ID] = {
            writeMemOpCounter, memOpCounter, false};
      }
      //      2.4 ignore sequences with only 1 memory op,
      //          ignore sequences with no write ops
      bool willModify = false;
      for (auto &p : currentBlockMemSeqs.srcToMemSeqInfoMap) {
        if (p.second.memOpCounter <= 1 || !p.second.writeMemOpCounter)
          p.second.ignored = true;
        else
          willModify = true;
      }
      if (!willModify) {
        LLVM_DEBUG(llvm::errs()
                   << "[AutoGenMemoryTokenPass] will not modify IR: " << fname
                   << ".\n");
        return WalkResult::skip();
      } else {
        LLVM_DEBUG(llvm::errs() << "[AutoGenMemoryTokenPass] will modify IR: "
                                << fname << ".\n");
        LLVM_DEBUG(llvm::errs()
                   << "[AutoGenMemoryTokenPass] Memory Sequence Info Map:\n");
        for (const auto &pair : currentBlockMemSeqs.srcToMemSeqInfoMap) {
          LLVM_DEBUG(llvm::errs()
                     << "\tSeqId: " << pair.first
                     << ", memOpCounter: " << pair.second.memOpCounter
                     << ", ignored: " << pair.second.ignored << "\n");
        }
      }

      // 3. Transform walk: traverse all ops recursively in the mod again to add
      // memory tokens
      auto tokens = currentBlockMemSeqs.getBlockInitTokens(body, rewriter);
      addMemTokenForBlock(body, tokens, rewriter);
      return WalkResult::skip();
    });
  }
};

} // namespace

std::unique_ptr<Pass> mlir::triton::createAutoGenMemoryTokenPass() {
  return std::make_unique<AutoGenMemoryTokenPass>();
}

std::unique_ptr<Pass> mlir::triton::createAutoGenMemoryTokenPass(
    bool enable_autogen_alias_mem_token) {
  return std::make_unique<AutoGenMemoryTokenPass>(
      enable_autogen_alias_mem_token);
}
