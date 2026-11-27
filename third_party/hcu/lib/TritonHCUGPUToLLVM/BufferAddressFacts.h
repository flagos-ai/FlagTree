#ifndef TRITON_HCU_BUFFER_ADDRESS_FACTS_H
#define TRITON_HCU_BUFFER_ADDRESS_FACTS_H

// Facts about exact lowered SSA aggregates, shared by address planners. Range,
// affine cancellation, lane ownership and mask equivalence remain separate
// proofs. In particular, no predicate grants an offset range fact.
class BufferAddressFacts {
  DenseMap<Value, Value> normalized;
  unsigned budget = 65536;

public:
  Value scalar(Value value) {
    if (auto it = normalized.find(value); it != normalized.end())
      return it->second;
    Value original = value;
    while (budget) {
      --budget;
      Operation *op = value.getDefiningOp();
      if (op && isa<LLVM::BitcastOp, UnrealizedConversionCastOp>(op) &&
          op->getNumOperands() == 1 && op->getNumResults() == 1 &&
          op->getOperand(0).getType() == value.getType()) {
        value = op->getOperand(0);
        continue;
      }
      auto extract = value.getDefiningOp<LLVM::ExtractValueOp>();
      if (!extract)
        break;
      auto position = extract.getPosition();
      auto type =
          dyn_cast<LLVM::LLVMStructType>(extract.getContainer().getType());
      if (position.size() != 1 || !type || type.isOpaque() || position[0] < 0 ||
          uint64_t(position[0]) >= type.getBody().size() ||
          type.getBody()[position[0]] != value.getType())
        return {};
      Value container = extract.getContainer(), field;
      while (budget) {
        --budget;
        auto insert = container.getDefiningOp<LLVM::InsertValueOp>();
        if (!insert || insert.getPosition().size() != 1 ||
            insert.getContainer().getType() != type || insert.getType() != type)
          return {};
        int64_t index = insert.getPosition()[0];
        if (index < 0 || uint64_t(index) >= type.getBody().size() ||
            insert.getValue().getType() != type.getBody()[index])
          return {};
        if (index == position[0]) {
          field = insert.getValue();
          break;
        }
        container = insert.getContainer();
      }
      if (!field)
        return {};
      value = field;
    }
    if (!budget)
      return {};
    normalized[original] = value;
    return value;
  }

  std::optional<uint64_t> constant(Value value) {
    value = scalar(value);
    Attribute attr;
    if (value && matchPattern(value, m_Constant(&attr)))
      if (auto integer = dyn_cast<IntegerAttr>(attr))
        if (integer.getValue().getBitWidth() <= 64)
          return integer.getValue().getZExtValue();
    return std::nullopt;
  }
};
#endif
