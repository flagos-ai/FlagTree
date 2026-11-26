#include "triton/Dialect/TritonXPU/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/SetVector.h"

namespace mlir::triton::xpu {

#define GEN_PASS_DEF_TRITONXPUINSERTDEVICEINIT
#include "triton/Dialect/TritonXPU/Transforms/Passes.h.inc"

namespace {

// LoopGrid may have placed the handle's constant or arithmetic inside a loop.
// Rebuild only expressions that depend solely on kernel arguments or constants.
Value materializeCommAtEntry(Value comm, Block &entry, OpBuilder &builder,
                             IRMapping &mapping) {
  if (auto arg = dyn_cast<BlockArgument>(comm))
    return arg.getOwner() == &entry ? comm : Value();
  if (Value mapped = mapping.lookupOrNull(comm))
    return mapped;
  Operation *def = comm.getDefiningOp();
  if (def->getName().getDialectNamespace() !=
          arith::ArithDialect::getDialectNamespace() ||
      !isPure(def))
    return {};
  for (Value operand : def->getOperands()) {
    Value mapped = materializeCommAtEntry(operand, entry, builder, mapping);
    if (!mapped)
      return {};
    mapping.map(operand, mapped);
  }
  builder.clone(*def, mapping);
  return mapping.lookup(comm);
}

LogicalResult insertDeviceInit(triton::FuncOp func, bool hasImplicitParams) {
  SmallVector<DeviceInitOp> initializers;
  llvm::SetVector<Value> explicitComms;
  llvm::SetVector<Value> inferredComms;
  bool usesImplicitComm = false;
  func.walk([&](Operation *op) {
    Value comm = llvm::TypeSwitch<Operation *, Value>(op)
                     .Case<DeviceInitOp>([&](DeviceInitOp init) {
                       initializers.push_back(init);
                       explicitComms.insert(init.getComm());
                       return Value();
                     })
                     .Case<GetDeviceIdOp, RemotePointerOp>([&](auto use) {
                       Value comm = use.getComm();
                       usesImplicitComm |= !comm;
                       return comm;
                     })
                     .Case<GetLocalRankOp, GetNumPesOp, DistributedBarrierOp,
                           SignalOp, SignalWaitOp>(
                         [](auto use) { return use.getComm(); })
                     .Default([](Operation *) { return Value(); });
    if (comm)
      inferredComms.insert(comm);
  });
  const auto &comms = explicitComms.empty() ? inferredComms : explicitComms;
  if (comms.size() > 1)
    return func.emitOpError("FlagCX device initialization requires a single "
                            "communicator; select it with tle.init");
  Value comm = comms.empty() ? Value() : comms.front();
  Block &entry = func.front();
  if (!comm && usesImplicitComm && hasImplicitParams)
    comm = entry.getArgument(0);
  // Low-level remote pointers with only an i64 memory handle supply no
  // communicator. An explicit init or the implicit distributed arguments can
  // provide it without changing the memory handle ABI.
  if (!comm)
    return success();

  OpBuilder builder(&entry, entry.begin());
  IRMapping mapping;
  Value entryComm = materializeCommAtEntry(comm, entry, builder, mapping);
  if (!entryComm)
    return func.emitOpError(
        "FlagCX communicator must be available at kernel entry");
  builder.create<DeviceInitOp>(func.getLoc(), entryComm);
  for (DeviceInitOp init : initializers)
    init.erase();
  return success();
}

struct TritonXPUInsertDeviceInit
    : impl::TritonXPUInsertDeviceInitBase<TritonXPUInsertDeviceInit> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    bool hasImplicitParams =
        module->hasAttr("xpu.implicit_distributed_params");
    for (auto func : module.getOps<triton::FuncOp>()) {
      if (func.isPublic() && !func.isExternal() &&
          failed(insertDeviceInit(func, hasImplicitParams))) {
        signalPassFailure();
        return;
      }
    }
  }
};

} // namespace
} // namespace mlir::triton::xpu
