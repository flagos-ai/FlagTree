// Interprocedural pipe control-state lowering, ported from feature/tle_mega
// bbae8cad6. Included inside WSLowerToken's implementation namespace.
#pragma once

enum TokenBarrierRequirement : unsigned {
  TokenNeedsNone = 0,
  TokenNeedsFull = 1,
  TokenNeedsEmpty = 2,
};

struct TokenBarrierValues {
  Value full;
  Value empty;
};

static unsigned getDirectTokenBarrierRequirement(Operation *user) {
  if (isa<ttnvws::ProducerCommitOp, ttnvws::ConsumerWaitOp>(user))
    return TokenNeedsFull;
  if (isa<ttnvws::ProducerAcquireOp, ttnvws::ConsumerReleaseOp>(user))
    return TokenNeedsEmpty;
  return TokenNeedsNone;
}

static FailureOr<tt::FuncOp> getTokenCallee(ModuleOp module,
                                             tt::CallOp call) {
  auto callee = module.lookupSymbol<tt::FuncOp>(call.getCallee());
  if (!callee)
    return call.emitOpError("cannot resolve token-bearing callee ")
           << call.getCallee();
  if (callee.isExternal() || !callee.getBody().hasOneBlock())
    return call.emitOpError("token-bearing callee must have one body block");
  return callee;
}

static FailureOr<unsigned> getTokenBarrierRequirement(
    Value token, ModuleOp module, DenseMap<Value, unsigned> &requirements,
    DenseSet<Value> &active) {
  if (auto it = requirements.find(token); it != requirements.end())
    return it->second;
  if (!active.insert(token).second)
    return emitError(token.getLoc())
           << "recursive token flow through device calls is unsupported";

  unsigned requirement = TokenNeedsNone;
  for (OpOperand &use : token.getUses()) {
    Operation *user = use.getOwner();
    unsigned direct = getDirectTokenBarrierRequirement(user);
    if (direct != TokenNeedsNone) {
      requirement |= direct;
      continue;
    }
    if (auto wsOp = dyn_cast<ttg::WarpSpecializeOp>(user)) {
      unsigned operandNumber = use.getOperandNumber();
      if (operandNumber >= wsOp.getExplicitCaptures().size())
        return wsOp.emitOpError("token must be an explicit capture");
      for (Region *region : wsOp.getPartitionRegions()) {
        if (operandNumber >= region->getNumArguments())
          return wsOp.emitOpError("token capture has no partition argument");
        auto nested = getTokenBarrierRequirement(
            region->getArgument(operandNumber), module, requirements, active);
        if (failed(nested))
          return failure();
        requirement |= *nested;
      }
      continue;
    }
    if (auto call = dyn_cast<tt::CallOp>(user)) {
      auto callee = getTokenCallee(module, call);
      if (failed(callee))
        return failure();
      unsigned operandNumber = use.getOperandNumber();
      if (operandNumber >= callee->getNumArguments())
        return call.emitOpError("token operand has no callee argument");
      auto nested = getTokenBarrierRequirement(
          callee->getArgument(operandNumber), module, requirements, active);
      if (failed(nested))
        return failure();
      requirement |= *nested;
      continue;
    }
    return user->emitOpError("unexpected use of an NVWS token");
  }

  active.erase(token);
  requirements[token] = requirement;
  return requirement;
}

static LogicalResult collectTokenLifecycleUsers(
    Value token, ModuleOp module, DenseSet<Value> &visited,
    DenseSet<Operation *> &lifecycleUsers) {
  if (!visited.insert(token).second)
    return success();
  for (OpOperand &use : token.getUses()) {
    Operation *user = use.getOwner();
    if (getDirectTokenBarrierRequirement(user) != TokenNeedsNone) {
      lifecycleUsers.insert(user);
      continue;
    }
    if (auto wsOp = dyn_cast<ttg::WarpSpecializeOp>(user)) {
      unsigned operandNumber = use.getOperandNumber();
      for (Region *region : wsOp.getPartitionRegions()) {
        if (failed(collectTokenLifecycleUsers(
                region->getArgument(operandNumber), module, visited,
                lifecycleUsers)))
          return failure();
      }
      continue;
    }
    if (auto call = dyn_cast<tt::CallOp>(user)) {
      auto callee = getTokenCallee(module, call);
      if (failed(callee))
        return failure();
      if (failed(collectTokenLifecycleUsers(
              callee->getArgument(use.getOperandNumber()), module, visited,
              lifecycleUsers)))
        return failure();
      continue;
    }
    return user->emitOpError("unexpected use of an NVWS token");
  }
  return success();
}

static bool haveCompatibleTokenContracts(ttnvws::CreateTokenOp lhs,
                                         ttnvws::CreateTokenOp rhs) {
  if (lhs.getResult().getType() != rhs.getResult().getType() ||
      lhs.getLoadType() != rhs.getLoadType())
    return false;
  for (StringRef attr :
       ArrayRef<StringRef>{"full_count", "empty_count",
                           kTleInferFullCountOffsetAttr}) {
    if (lhs->getAttr(attr) != rhs->getAttr(attr))
      return false;
  }
  return true;
}

static void recordOriginalOperand(
    std::map<Operation *, SmallVector<unsigned>> &operands, Operation *op,
    unsigned operandNumber) {
  SmallVector<unsigned> &indices = operands[op];
  if (!llvm::is_contained(indices, operandNumber))
    indices.push_back(operandNumber);
}

static LogicalResult lowerTokenOperationsThroughCalls(Operation *parentOp,
                                                       int numCTAs) {
  auto module = dyn_cast<ModuleOp>(parentOp);
  if (!module)
    return parentOp->emitOpError(
        "interprocedural NVWS token lowering requires a module");

  SmallVector<ttnvws::CreateTokenOp> createTokens;
  module.walk([&](ttnvws::CreateTokenOp op) { createTokens.push_back(op); });
  if (createTokens.empty())
    return success();

  DenseMap<Value, unsigned> requirements;
  DenseSet<Value> active;
  DenseMap<Operation *, DenseSet<Operation *>> lifecycleUsersByToken;
  DenseSet<Operation *> allLifecycleUsers;
  SmallVector<ttnvws::CreateTokenOp> liveCreateTokens;
  for (ttnvws::CreateTokenOp create : createTokens) {
    auto requirement = getTokenBarrierRequirement(
        create.getResult(), module, requirements, active);
    if (failed(requirement))
      return failure();
    if (*requirement == TokenNeedsNone && create.getResult().use_empty()) {
      create.erase();
      continue;
    }
    if (*requirement == TokenNeedsNone)
      return create.emitOpError("has no reachable lifecycle users");
    liveCreateTokens.push_back(create);
    DenseSet<Value> visited;
    DenseSet<Operation *> users;
    if (failed(collectTokenLifecycleUsers(create.getResult(), module, visited,
                                          users)))
      return failure();
    allLifecycleUsers.insert(users.begin(), users.end());
    lifecycleUsersByToken.try_emplace(create.getOperation(), std::move(users));
  }
  createTokens = std::move(liveCreateTokens);
  if (createTokens.empty())
    return success();

  DenseMap<Operation *, unsigned> inferredParticipantCounts;
  for (Operation *user : allLifecycleUsers) {
    auto commit = dyn_cast<ttnvws::ProducerCommitOp>(user);
    if (!commit || !commit->hasAttr(kTleInferArriveCountAttr))
      continue;
    std::optional<unsigned> count;
    if (failed(recordInferredParticipantCount(commit, count)))
      return failure();
    assert(count && "inferred participant commit must produce a count");
    inferredParticipantCounts[user] = *count;
  }

  MLIRContext *context = module.getContext();
  Attribute sharedMemorySpace = ttg::SharedMemorySpaceAttr::get(context);
  auto barrierCTALayout = ttg::CTAEncodingAttr::getDefault(context, 1);
  auto barrierEncoding = ttg::SwizzledSharedEncodingAttr::get(
      context, 1, 1, 1, {0}, barrierCTALayout);

  DenseMap<Value, TokenBarrierValues> barrierValues;
  DenseMap<Value, ttnvws::CreateTokenOp> tokenOrigins;
  DenseMap<Operation *, unsigned> tokenFullCounts;
  for (ttnvws::CreateTokenOp create : createTokens) {
    OpBuilder builder(create);
    Location loc = create.getLoc();
    unsigned requirement = requirements.lookup(create.getResult());
    Type arrayType = ttg::MemDescType::get(
        {create.getNumBuffers(), 1}, builder.getI64Type(), barrierEncoding,
        sharedMemorySpace, /*mutableMemory=*/true);
    Type slotType = ttg::MemDescType::get(
        {1}, builder.getI64Type(), barrierEncoding, sharedMemorySpace,
        /*mutableMemory=*/true);

    TokenBarrierValues values;
    if (requirement & TokenNeedsFull)
      values.full = ttg::LocalAllocOp::create(builder, loc, arrayType, Value());
    if (requirement & TokenNeedsEmpty)
      values.empty =
          ttg::LocalAllocOp::create(builder, loc, arrayType, Value());
    barrierValues[create.getResult()] = values;
    tokenOrigins[create.getResult()] = create;

    unsigned fullCount = create.getLoadType() ==
                                 ttnvws::TokenLoadType::TMALoadOp
                             ? 1
                             : THREADS_PER_TASK;
    if (create.getLoadType() != ttnvws::TokenLoadType::TMALoadOp) {
      if (auto count = getTokenCountOverride(create, "full_count"))
        fullCount = *count;
    }
    if (auto offset =
            getTokenCountOverride(create, kTleInferFullCountOffsetAttr)) {
      std::optional<unsigned> inferred;
      for (Operation *user : lifecycleUsersByToken.lookup(create)) {
        auto it = inferredParticipantCounts.find(user);
        if (it == inferredParticipantCounts.end())
          continue;
        if (inferred && *inferred != it->second)
          return user->emitOpError("inconsistent inferred participant counts ")
                 << *inferred << " and " << it->second;
        inferred = it->second;
      }
      if (!inferred)
        return create.emitOpError(
            "requires an inferred participant producer commit for token "
            "full_count inference");
      fullCount = *inferred + *offset;
      create->removeAttr(kTleInferFullCountOffsetAttr);
    }
    tokenFullCounts[create.getOperation()] = fullCount;

    unsigned emptyCount = THREADS_PER_TASK;
    if (auto count = getTokenCountOverride(create, "empty_count"))
      emptyCount = *count;
    for (unsigned i = 0; i < create.getNumBuffers(); ++i) {
      Value idx = arith::ConstantIntOp::create(builder, loc, i, 32);
      if (values.full) {
        Value slot = ttg::MemDescIndexOp::create(builder, loc, slotType,
                                                 values.full, idx);
        ttng::InitBarrierOp::create(builder, loc, slot, fullCount);
      }
      if (values.empty) {
        Value slot = ttg::MemDescIndexOp::create(builder, loc, slotType,
                                                 values.empty, idx);
        ttng::InitBarrierOp::create(builder, loc, slot, emptyCount);
      }
    }
    assert(numCTAs == 1 && "remote CTA is not supported yet");
    if (values.full || values.empty)
      mlir::gpu::BarrierOp::create(builder, loc);
  }

  std::map<Operation *, SmallVector<unsigned>> callTokenOperands;
  // Call operands must follow the callee's ABI, not create-token traversal
  // order. Different callers can bind its token parameters in different order.
  std::map<Operation *, std::map<unsigned, Value>> callBarrierOperands;
  std::map<Operation *, SmallVector<unsigned>> functionTokenArguments;
  std::map<Operation *, SmallVector<unsigned>> wsTokenOperands;
  DenseSet<Value> propagated;
  std::function<LogicalResult(Value)> propagate =
      [&](Value token) -> LogicalResult {
    if (!propagated.insert(token).second)
      return success();
    TokenBarrierValues values = barrierValues.lookup(token);
    ttnvws::CreateTokenOp origin = tokenOrigins.lookup(token);
    if (!origin)
      return emitError(token.getLoc()) << "token has no create-token origin";

    for (OpOperand &use : llvm::make_early_inc_range(token.getUses())) {
      Operation *user = use.getOwner();
      if (getDirectTokenBarrierRequirement(user) != TokenNeedsNone)
        continue;
      if (auto wsOp = dyn_cast<ttg::WarpSpecializeOp>(user)) {
        unsigned operandNumber = use.getOperandNumber();
        unsigned requirement = requirements.lookup(token);
        recordOriginalOperand(wsTokenOperands, wsOp, operandNumber);
        unsigned fullArgument = 0;
        unsigned emptyArgument = 0;
        if (requirement & TokenNeedsFull) {
          wsOp->insertOperands(wsOp.getNumOperands(), values.full);
          fullArgument = wsOp.getNumOperands() - 1;
          for (Region *region : wsOp.getPartitionRegions())
            region->addArgument(values.full.getType(), values.full.getLoc());
        }
        if (requirement & TokenNeedsEmpty) {
          wsOp->insertOperands(wsOp.getNumOperands(), values.empty);
          emptyArgument = wsOp.getNumOperands() - 1;
          for (Region *region : wsOp.getPartitionRegions())
            region->addArgument(values.empty.getType(), values.empty.getLoc());
        }
        for (Region *region : wsOp.getPartitionRegions()) {
          Value nested = region->getArgument(operandNumber);
          TokenBarrierValues nestedValues;
          if (requirement & TokenNeedsFull)
            nestedValues.full = region->getArgument(fullArgument);
          if (requirement & TokenNeedsEmpty)
            nestedValues.empty = region->getArgument(emptyArgument);
          barrierValues[nested] = nestedValues;
          tokenOrigins[nested] = origin;
          if (failed(propagate(nested)))
            return failure();
        }
        continue;
      }
      if (auto call = dyn_cast<tt::CallOp>(user)) {
        auto callee = getTokenCallee(module, call);
        if (failed(callee))
          return failure();
        unsigned operandNumber = use.getOperandNumber();
        Value calleeToken = callee->getArgument(operandNumber);
        unsigned requirement = requirements.lookup(calleeToken);
        TokenBarrierValues calleeValues = barrierValues.lookup(calleeToken);
        if (!calleeValues.full && !calleeValues.empty) {
          if (requirement & TokenNeedsFull) {
            unsigned index = callee->getNumArguments();
            if (failed(callee->insertArgument(index, values.full.getType(),
                                              DictionaryAttr{},
                                              calleeToken.getLoc())))
              return callee->emitOpError("cannot append full-barrier argument");
            calleeValues.full = callee->getArgument(index);
          }
          if (requirement & TokenNeedsEmpty) {
            unsigned index = callee->getNumArguments();
            if (failed(callee->insertArgument(index, values.empty.getType(),
                                              DictionaryAttr{},
                                              calleeToken.getLoc())))
              return callee->emitOpError(
                  "cannot append empty-barrier argument");
            calleeValues.empty = callee->getArgument(index);
          }
          barrierValues[calleeToken] = calleeValues;
          tokenOrigins[calleeToken] = origin;
          recordOriginalOperand(functionTokenArguments, callee->getOperation(),
                                operandNumber);
        } else {
          ttnvws::CreateTokenOp previous = tokenOrigins.lookup(calleeToken);
          if (!previous || !haveCompatibleTokenContracts(previous, origin))
            return call.emitOpError(
                "shared token-bearing callee has incompatible pipe contracts");
        }
        if (requirement & TokenNeedsFull)
          callBarrierOperands[call][cast<BlockArgument>(calleeValues.full)
                                        .getArgNumber()] = values.full;
        if (requirement & TokenNeedsEmpty)
          callBarrierOperands[call][cast<BlockArgument>(calleeValues.empty)
                                        .getArgNumber()] = values.empty;
        recordOriginalOperand(callTokenOperands, call, operandNumber);
        if (failed(propagate(calleeToken)))
          return failure();
        continue;
      }
      return user->emitOpError("unexpected use of an NVWS token");
    }
    return success();
  };

  for (ttnvws::CreateTokenOp create : createTokens)
    if (failed(propagate(create.getResult())))
      return failure();

  for (auto &[op, operands] : callBarrierOperands) {
    for (auto &[index, value] : operands) {
      if (index != op->getNumOperands())
        return op->emitOpError("incomplete lowered token call ABI");
      op->insertOperands(index, value);
    }
  }

  SmallVector<Operation *> deprecatedOps;
  for (Operation *user : allLifecycleUsers) {
    Value token;
    if (auto op = dyn_cast<ttnvws::ProducerAcquireOp>(user))
      token = op.getToken();
    else if (auto op = dyn_cast<ttnvws::ProducerCommitOp>(user))
      token = op.getToken();
    else if (auto op = dyn_cast<ttnvws::ConsumerWaitOp>(user))
      token = op.getToken();
    else
      token = cast<ttnvws::ConsumerReleaseOp>(user).getToken();
    TokenBarrierValues values = barrierValues.lookup(token);
    ttnvws::CreateTokenOp origin = tokenOrigins.lookup(token);
    if (!origin)
      return user->emitOpError("has no propagated create-token origin");

    OpBuilder builder(user);
    Type slotType = ttg::MemDescType::get(
        {1}, builder.getI64Type(), barrierEncoding, sharedMemorySpace,
        /*mutableMemory=*/true);
    auto fullSlot = [&](Value idx) -> Value {
      assert(values.full && "full-barrier user requires a full array");
      return ttg::MemDescIndexOp::create(builder, user->getLoc(), slotType,
                                         values.full, idx);
    };
    auto emptySlot = [&](Value idx) -> Value {
      assert(values.empty && "empty-barrier user requires an empty array");
      return ttg::MemDescIndexOp::create(builder, user->getLoc(), slotType,
                                         values.empty, idx);
    };

    if (auto op = dyn_cast<ttnvws::ProducerAcquireOp>(user)) {
      Value barrier = emptySlot(op.getIdx());
      setAsyncTaskIds(barrier.getDefiningOp(), getAsyncTaskIds(user));
      processProducerAcquireOp(builder, op, barrier);
    } else if (auto op = dyn_cast<ttnvws::ProducerCommitOp>(user)) {
      Value barrier = fullSlot(op.getIdx());
      setAsyncTaskIds(barrier.getDefiningOp(), getAsyncTaskIds(user));
      if (op.getCommitKind() ==
          ttnvws::ProducerCommitKind::TmaCopyBarrierArrive) {
        if (failed(processProducerCommitTmaCopyOp(builder, op, barrier)))
          return failure();
      } else {
        processProducerCommitOp(builder, op, barrier, origin.getLoadType(),
                                tokenFullCounts.lookup(origin.getOperation()));
      }
    } else if (auto op = dyn_cast<ttnvws::ConsumerWaitOp>(user)) {
      Value barrier = fullSlot(op.getIdx());
      setAsyncTaskIds(barrier.getDefiningOp(), getAsyncTaskIds(user));
      processConsumerWaitOp(builder, op, barrier);
    } else {
      auto releaseOp = cast<ttnvws::ConsumerReleaseOp>(user);
      Value barrier = emptySlot(releaseOp.getIdx());
      setAsyncTaskIds(barrier.getDefiningOp(), getAsyncTaskIds(user));
      unsigned releaseCount = THREADS_PER_TASK;
      if (auto attr =
              releaseOp->getAttrOfType<IntegerAttr>("release_count"))
        releaseCount = static_cast<unsigned>(attr.getInt());
      processConsumerReleaseOp(builder, releaseOp, barrier, numCTAs,
                               releaseCount);
    }
    deprecatedOps.push_back(user);
  }

  for (Operation *op : deprecatedOps)
    op->erase();

  auto eraseOriginalOperands = [](auto op, SmallVector<unsigned> indices) {
    llvm::sort(indices, std::greater<unsigned>());
    for (unsigned index : indices)
      op->eraseOperand(index);
  };
  for (auto &[op, indices] : callTokenOperands)
    eraseOriginalOperands(cast<tt::CallOp>(op), indices);
  for (auto &[op, indices] : functionTokenArguments) {
    auto function = cast<tt::FuncOp>(op);
    llvm::sort(indices, std::greater<unsigned>());
    for (unsigned index : indices) {
      if (!function.getArgument(index).use_empty())
        return function.emitOpError("lowered token argument still has uses");
      if (failed(function.eraseArgument(index)))
        return function.emitOpError("cannot erase lowered token argument");
    }
  }
  for (auto &[op, indices] : wsTokenOperands) {
    auto wsOp = cast<ttg::WarpSpecializeOp>(op);
    llvm::sort(indices, std::greater<unsigned>());
    for (unsigned index : indices) {
      for (Region *region : wsOp.getPartitionRegions()) {
        if (!region->getArgument(index).use_empty())
          return wsOp.emitOpError("lowered token capture still has uses");
        region->eraseArgument(index);
      }
      wsOp->eraseOperand(index);
    }
  }
  for (ttnvws::CreateTokenOp create : createTokens) {
    if (!create.getResult().use_empty())
      return create.emitOpError("lowered token still has uses");
    create.erase();
  }

  coalesceMBarrierInitBarriers(parentOp);
  return success();
}
