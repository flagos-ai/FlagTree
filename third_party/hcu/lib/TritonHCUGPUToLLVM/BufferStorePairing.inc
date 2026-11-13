// Included by the buffer store lowering. The store operands already represent
// values in one lane. Only their order within that lane can change.
class BufferStorePairing {
  struct Affine {
    llvm::SmallDenseMap<Value, uint64_t, 4> terms;
    uint64_t constant = 0;

    void add(const Affine &other, uint64_t factor, uint64_t mask) {
      constant = (constant + factor * other.constant) & mask;
      for (auto [value, coefficient] : other.terms) {
        auto &term = terms[value];
        term = (term + factor * coefficient) & mask;
        if (!term)
          terms.erase(value);
      }
    }
  };

  struct Facts {
    uint64_t zero;
    uint64_t one;
  };

  DenseMap<Value, Value> scalars;
  DenseMap<Value, SmallVector<Value>> aggregates;
  DenseMap<Value, Facts> known;
  DenseMap<Value, Affine> forms;
  // Evaluate only expressions whose inputs are constants and the workgroup
  // thread id. Normalize their finite tables so distinct SSA expressions
  // with the same lane-dependent part cancel in an address difference.
  struct LaneValues {
    uint64_t first = 0;
    SmallVector<uint64_t, 0> varying;

    uint64_t at(unsigned lane) const {
      return varying.empty() ? first : varying[lane];
    }
  };
  DenseMap<Value, std::optional<LaneValues>> laneValues;
  DenseMap<size_t, SmallVector<Value, 1>> laneClasses;
  DenseMap<Value, LaneValues> laneTerms;
  unsigned threads;
  unsigned budget = 65536;

  static unsigned width(Value value) {
    return cast<IntegerType>(value.getType()).getWidth();
  }

  static uint64_t mask(unsigned bits) {
    return llvm::maskTrailingOnes<uint64_t>(bits);
  }

  // Unpack only exact insert/extract chains. An opaque aggregate never
  // contributes an address or a mask proof.
  Value scalar(Value value) {
    if (auto it = scalars.find(value); it != scalars.end())
      return it->second;
    Value original = value;
    while (budget) {
      --budget;
      if (auto cast = value.getDefiningOp<LLVM::BitcastOp>();
          cast && cast.getArg().getType() == value.getType()) {
        value = cast.getArg();
        continue;
      }
      if (auto cast = value.getDefiningOp<UnrealizedConversionCastOp>();
          cast && cast.getInputs().size() == 1 && cast.getOutputs().size() == 1 &&
          cast.getInputs().front().getType() == value.getType()) {
        value = cast.getInputs().front();
        continue;
      }
      auto extract = value.getDefiningOp<LLVM::ExtractValueOp>();
      if (!extract || extract.getPosition().size() != 1)
        break;
      int64_t index = extract.getPosition()[0];
      Value container = extract.getContainer();
      auto type = dyn_cast<LLVM::LLVMStructType>(container.getType());
      if (!type || type.isOpaque() || type.getBody().size() > 256 || index < 0 ||
          uint64_t(index) >= type.getBody().size())
        break;
      auto [it, inserted] = aggregates.try_emplace(container);
      if (inserted) {
        auto &fields = it->second;
        fields.resize(type.getBody().size());
        while (budget) {
          --budget;
          auto insert = container.getDefiningOp<LLVM::InsertValueOp>();
          if (!insert || insert.getPosition().size() != 1)
            break;
          auto field = insert.getPosition()[0];
          if (!fields[field])
            fields[field] = insert.getValue();
          container = insert.getContainer();
        }
      }
      Value field = it->second[index];
      if (!field || field.getType() != value.getType())
        break;
      value = field;
    }
    scalars.try_emplace(original, value);
    return value;
  }

  std::optional<uint64_t> constant(Value value) {
    Attribute attr;
    if (matchPattern(scalar(value), m_Constant(&attr)))
      if (auto integer = dyn_cast<IntegerAttr>(attr))
        if (integer.getValue().getBitWidth() <= 64)
          return integer.getValue().getZExtValue();
    return std::nullopt;
  }

  bool evaluateLanes(Value value, unsigned depth = 0) {
    value = scalar(value);
    if (auto it = laneValues.find(value); it != laneValues.end())
      return it->second.has_value();
    if (depth > 48 || !budget)
      return false;
    --budget;
    auto save = [&](std::optional<LaneValues> result) {
      bool valid = result.has_value();
      laneValues.try_emplace(value, std::move(result));
      return valid;
    };
    if (auto c = constant(value))
      return save(LaneValues{*c, {}});
    if (auto id = value.getDefiningOp<mlir::gpu::ThreadIdOp>();
        id && id.getDimension() == mlir::gpu::Dimension::x) {
      LaneValues result;
      for (unsigned i = 0; i < threads; ++i)
        result.varying.push_back(i);
      return save(std::move(result));
    }
    auto *op = value.getDefiningOp();
    auto type = dyn_cast<IntegerType>(value.getType());
    if (!op || !type || type.getWidth() > 64)
      return save(std::nullopt);
    unsigned size = type.getWidth();
    using Fold = std::function<std::optional<uint64_t>(uint64_t, uint64_t)>;
    Fold fold = llvm::TypeSwitch<Operation *, Fold>(op)
        .Case<LLVM::AddOp, arith::AddIOp>(
            [](auto) { return [](uint64_t a, uint64_t b) { return a + b; }; })
        .Case<LLVM::SubOp, arith::SubIOp>(
            [](auto) { return [](uint64_t a, uint64_t b) { return a - b; }; })
        .Case<LLVM::MulOp, arith::MulIOp>(
            [](auto) { return [](uint64_t a, uint64_t b) { return a * b; }; })
        .Case<LLVM::AndOp>(
            [](auto) { return [](uint64_t a, uint64_t b) { return a & b; }; })
        .Case<LLVM::OrOp>(
            [](auto) { return [](uint64_t a, uint64_t b) { return a | b; }; })
        .Case<LLVM::XOrOp>(
            [](auto) { return [](uint64_t a, uint64_t b) { return a ^ b; }; })
        .Case<LLVM::ShlOp, LLVM::LShrOp, LLVM::AShrOp>([&](auto shift) {
          return [size, left = isa<LLVM::ShlOp>(shift.getOperation()),
                  arithmetic = isa<LLVM::AShrOp>(shift.getOperation())](
                     uint64_t a, uint64_t b) -> std::optional<uint64_t> {
            if (b >= size)
              return std::nullopt;
            if (left)
              return a << b;
            if (arithmetic && (a & (uint64_t(1) << (size - 1))))
              return (a >> b) | (mask(size) & ~mask(size - b));
            return a >> b;
          };
        })
        .Case<LLVM::URemOp, LLVM::UDivOp>([](auto op) {
          return [rem = isa<LLVM::URemOp>(op.getOperation())](
                     uint64_t a, uint64_t b) -> std::optional<uint64_t> {
            if (!b)
              return std::nullopt;
            return rem ? a % b : a / b;
          };
        })
        .Case<LLVM::TruncOp, LLVM::ZExtOp, arith::IndexCastOp>(
            [](auto) { return [](uint64_t a, uint64_t) { return a; }; })
        .Case<LLVM::SExtOp>([&](auto ext) {
          unsigned inputWidth = width(ext.getArg());
          return [inputWidth](uint64_t a, uint64_t) {
            return a & (uint64_t(1) << (inputWidth - 1))
                       ? a | ~mask(inputWidth)
                       : a;
          };
        })
        .Default([](Operation *) { return Fold{}; });
    if (!fold || !llvm::all_of(op->getOperands(), [&](Value input) {
          return evaluateLanes(input, depth + 1);
        }))
      return save(std::nullopt);

    const auto &a = *laneValues.find(scalar(op->getOperand(0)))->second;
    LaneValues zero;
    const auto &b = op->getNumOperands() == 2
                        ? *laneValues.find(scalar(op->getOperand(1)))->second
                        : zero;
    LaneValues result;
    unsigned count = a.varying.empty() && b.varying.empty() ? 1 : threads;
    for (unsigned i = 0; i < count; ++i) {
      auto folded = fold(a.at(i), b.at(i));
      if (!folded)
        return save(std::nullopt);
      result.varying.push_back(*folded & mask(size));
    }
    result.first = result.varying.front();
    if (llvm::all_of(result.varying, [&](auto v) { return v == result.first; }))
      result.varying.clear();
    return save(std::move(result));
  }

  Affine laneForm(Value value) {
    const auto &table = *laneValues.find(value)->second;
    Affine result;
    result.constant = table.first;
    if (table.varying.empty())
      return result;
    LaneValues normalized;
    for (uint64_t v : table.varying)
      normalized.varying.push_back((v - table.first) & mask(width(value)));
    auto &bucket = laneClasses[llvm::hash_combine(
        width(value), llvm::hash_combine_range(normalized.varying.begin(),
                                               normalized.varying.end()))];
    auto same = llvm::find_if(bucket, [&](Value other) {
      return laneTerms.find(other)->second.varying == normalized.varying;
    });
    if (same != bucket.end()) {
      result.terms.try_emplace(*same, 1);
      return result;
    }
    bucket.push_back(value);
    laneTerms.try_emplace(value, std::move(normalized));
    result.terms.try_emplace(value, 1);
    return result;
  }

  bool termEven(Value value) {
    if (auto it = laneTerms.find(value); it != laneTerms.end())
      return llvm::all_of(it->second.varying, [](uint64_t n) { return !(n & 1); });
    return bits(value).zero & 1;
  }

  Facts bits(Value value, unsigned depth = 0) {
    value = scalar(value);
    if (depth > 48 || !budget)
      return {};
    --budget;
    if (auto it = known.find(value); it != known.end())
      return it->second;
    unsigned size = width(value);
    if (size > 64)
      return {};
    uint64_t limit = mask(size);
    auto save = [&](uint64_t zero, uint64_t one = 0) {
      Facts result{zero & limit, one & limit};
      known.try_emplace(value, result);
      return result;
    };
    if (auto c = constant(value))
      return save(~*c, *c);
    auto *op = value.getDefiningOp();
    if (!op)
      return save(0);
    if (isa<LLVM::ZExtOp, LLVM::SExtOp, LLVM::TruncOp>(op)) {
      Value input = scalar(op->getOperand(0));
      Facts source = bits(input, depth + 1);
      unsigned sourceWidth = width(input);
      if (sourceWidth > 64)
        return save(0);
      if (isa<LLVM::ZExtOp>(op) ||
          (isa<LLVM::SExtOp>(op) && (source.zero >> (sourceWidth - 1))))
        source.zero |= ~mask(sourceWidth);
      return save(source.zero, source.one);
    }
    if (!isa<LLVM::AddOp, LLVM::SubOp, LLVM::MulOp, LLVM::AndOp,
             LLVM::OrOp, LLVM::XOrOp, LLVM::ShlOp, LLVM::LShrOp,
             LLVM::AShrOp>(op))
      return save(0);
    Value lhs = op->getOperand(0), rhs = op->getOperand(1);
    Facts a = bits(lhs, depth + 1), b = bits(rhs, depth + 1);
    if (isa<LLVM::AndOp>(op))
      return save(a.zero | b.zero, a.one & b.one);
    if (isa<LLVM::OrOp>(op))
      return save(a.zero & b.zero, a.one | b.one);
    if (isa<LLVM::XOrOp>(op))
      return save((a.zero & b.zero) | (a.one & b.one),
                  (a.zero & b.one) | (a.one & b.zero));
    if (isa<LLVM::ShlOp, LLVM::LShrOp, LLVM::AShrOp>(op)) {
      auto shift = constant(rhs);
      if (!shift || *shift >= size)
        return save(0);
      if (isa<LLVM::ShlOp>(op))
        return save((a.zero << *shift) | mask(*shift), a.one << *shift);
      uint64_t zero = a.zero >> *shift;
      if (isa<LLVM::LShrOp>(op) || (a.zero >> (size - 1)))
        zero |= ~mask(size - *shift);
      return save(zero, a.one >> *shift);
    }
    unsigned aLow = llvm::countr_one(a.zero), bLow = llvm::countr_one(b.zero);
    if (isa<LLVM::MulOp>(op))
      return save(mask(std::min(size, aLow + bLow)));
    unsigned low = std::min(aLow, bLow);
    uint64_t aMax = limit & ~a.zero, bMax = limit & ~b.zero;
    if (isa<LLVM::AddOp>(op) && aMax <= limit - bMax)
      return save(mask(low) | (limit & ~mask(llvm::bit_width(aMax + bMax))));
    return save(mask(low));
  }

  Affine affine(Value value, unsigned depth = 0) {
    value = scalar(value);
    unsigned size = width(value);
    auto opaque = [&] {
      Affine result;
      result.terms.try_emplace(value, 1);
      return result;
    };
    if (auto it = forms.find(value); it != forms.end())
      return it->second;
    if (depth > 48 || size > 64 || !budget)
      return opaque();
    --budget;
    auto save = [&](Affine result) {
      forms.try_emplace(value, result);
      return result;
    };
    if (auto c = constant(value)) {
      Affine result;
      result.constant = *c;
      return save(result);
    }
    if (evaluateLanes(value))
      return save(laneForm(value));
    auto *op = value.getDefiningOp();
    if (!op)
      return save(opaque());
    if (isa<LLVM::TruncOp, arith::TruncIOp>(op)) {
      Affine result;
      result.add(affine(op->getOperand(0), depth + 1), 1, mask(size));
      return save(result);
    }
    if (op->getNumOperands() != 2)
      return save(opaque());
    Value lhs = op->getOperand(0), rhs = op->getOperand(1);
    bool add = isa<LLVM::AddOp, arith::AddIOp>(op);
    if (isa<LLVM::OrOp, LLVM::XOrOp>(op))
      add = !(mask(size) & ~bits(lhs).zero & ~bits(rhs).zero);
    if (add || isa<LLVM::SubOp, arith::SubIOp>(op)) {
      Affine result = affine(lhs, depth + 1);
      result.add(affine(rhs, depth + 1),
                 isa<LLVM::SubOp, arith::SubIOp>(op) ? uint64_t(-1) : 1, mask(size));
      return save(result);
    }
    if (isa<LLVM::MulOp, arith::MulIOp>(op)) {
      if (!constant(rhs))
        std::swap(lhs, rhs);
      if (auto c = constant(rhs)) {
        Affine result;
        result.add(affine(lhs, depth + 1), *c, mask(size));
        return save(result);
      }
    }
    if (isa<LLVM::ShlOp, arith::ShLIOp>(op)) {
      if (auto shift = constant(rhs); shift && *shift < size) {
        Affine result;
        result.add(affine(lhs, depth + 1), uint64_t(1) << *shift, mask(size));
        return save(result);
      }
    }
    return save(opaque());
  }

  std::optional<uint64_t> difference(Value lo, Value hi, unsigned depth = 0) {
    lo = scalar(lo);
    hi = scalar(hi);
    if (lo == hi)
      return 0;
    if (lo.getType() != hi.getType() || width(lo) > 64 || depth > 16 || !budget)
      return std::nullopt;
    --budget;
    if (auto *a = lo.getDefiningOp(), *b = hi.getDefiningOp();
        a && b && a->getName() == b->getName() &&
        isa<LLVM::ZExtOp, LLVM::SExtOp>(a)) {
      // Extending an even/odd pair preserves its difference. The addition
      // of one cannot carry out of the low bit, including across the sign bit.
      Value x = a->getOperand(0), y = b->getOperand(0);
      auto delta = difference(x, y, depth + 1);
      if (delta && (*delta == 0 || (*delta == 1 && even(x))))
        return delta;
    }
    auto a = affine(lo), b = affine(hi);
    if (a.terms == b.terms)
      return (b.constant - a.constant) & mask(width(lo));
    auto isAdd = [&](Operation *op) {
      if (isa_and_nonnull<LLVM::AddOp>(op))
        return true;
      return isa_and_nonnull<LLVM::OrOp, LLVM::XOrOp>(op) &&
             !(mask(width(lo)) & ~bits(op->getOperand(0)).zero &
               ~bits(op->getOperand(1)).zero);
    };
    auto *x = lo.getDefiningOp(), *y = hi.getDefiningOp();
    if (!isAdd(x) || !isAdd(y))
      return std::nullopt;
    for (unsigned swap = 0; swap < 2; ++swap) {
      auto first = difference(x->getOperand(0), y->getOperand(swap), depth + 1);
      auto second = difference(x->getOperand(1), y->getOperand(1 - swap), depth + 1);
      if (first && second)
        return (*first + *second) & mask(width(lo));
    }
    return std::nullopt;
  }

  bool even(Value value) {
    auto form = affine(value);
    return !(form.constant & 1) &&
           llvm::all_of(form.terms, [&](auto term) {
             return !(term.second & 1) || termEven(term.first);
           });
  }

  bool sameMask(Value a, Value b, unsigned depth = 0) {
    a = scalar(a);
    b = scalar(b);
    if (a == b)
      return true;
    if (depth > 8 || !budget)
      return false;
    --budget;
    if (auto x = constant(a), y = constant(b); x && y)
      return *x == *y;
    auto lhsAnd = a.getDefiningOp<LLVM::AndOp>();
    auto rhsAnd = b.getDefiningOp<LLVM::AndOp>();
    if (lhsAnd && rhsAnd)
      return (sameMask(lhsAnd.getLhs(), rhsAnd.getLhs(), depth + 1) &&
              sameMask(lhsAnd.getRhs(), rhsAnd.getRhs(), depth + 1)) ||
             (sameMask(lhsAnd.getLhs(), rhsAnd.getRhs(), depth + 1) &&
              sameMask(lhsAnd.getRhs(), rhsAnd.getLhs(), depth + 1));
    auto x = a.getDefiningOp<LLVM::ICmpOp>();
    auto y = b.getDefiningOp<LLVM::ICmpOp>();
    if (!x || !y || x.getPredicate() != y.getPredicate() ||
        difference(x.getRhs(), y.getRhs()) != 0)
      return false;
    auto delta = difference(x.getLhs(), y.getLhs());
    if (delta == 0)
      return true;
    using P = LLVM::ICmpPredicate;
    if ((x.getPredicate() != P::ult && x.getPredicate() != P::slt) ||
        !even(x.getRhs()))
      return false;
    // For an even bound, x and x+1 agree on x < bound whenever x is
    // even. Treat the two predicates symmetrically.
    return (delta == 1 && even(x.getLhs())) ||
           (difference(y.getLhs(), x.getLhs()) == 1 && even(y.getLhs()));
  }

  bool preservesRawBufferBounds(Value offset) {
    // Scaling the i32 element offset by two wraps modulo 2^32. For a
    // dword-aligned address, the only case where the two original halfword
    // stores have different descriptor validity is the pair that starts
    // two bytes before NUM_RECORDS.
    constexpr uint64_t split = (BufferStoreScalarOffset::Records - 2) / 2;
    const auto form = affine(offset);
    unsigned knownLow = 31;
    SmallVector<std::pair<const LaneValues *, uint64_t>, 4> laneContributions;
    for (auto [term, coefficient] : form.terms) {
      if (auto it = laneTerms.find(term); it != laneTerms.end()) {
        laneContributions.emplace_back(&it->second, coefficient);
        continue;
      }
      knownLow = std::min<unsigned>(
          knownLow, llvm::countr_zero(coefficient) + llvm::countr_one(bits(term).zero));
    }
    if (knownLow <= 1)
      return false;
    uint64_t residueMask = mask(knownLow);
    for (unsigned lane = 0; lane < threads; ++lane) {
      uint64_t local = form.constant;
      for (auto [table, coefficient] : laneContributions)
        local += coefficient * table->at(lane);
      if (((local ^ split) & residueMask) == 0)
        return false;
    }
    return true;
  }

public:
  explicit BufferStorePairing(unsigned threads) : threads(threads) {}

  using Pairs = SmallVector<std::pair<unsigned, unsigned>>;

  Pairs plan(ArrayRef<Value> offsets, ArrayRef<Value> masks, uint32_t redundant) {
    unsigned count = offsets.size();
    if (count < 2 || count > 256 || !llvm::isPowerOf2_32(count) ||
        (!masks.empty() && masks.size() != count))
      return {};
    Pairs best;
    unsigned canonicalCount = count >> llvm::popcount(redundant & (count - 1));
    for (unsigned bit = 1; bit < count && budget; bit *= 2) {
      if (redundant & bit)
        continue;
      Pairs pairs;
      for (unsigned i = 0; i < count && budget; ++i) {
        if (i & (bit | redundant))
          continue;
        unsigned lo = i, hi = i | bit;
        if (difference(offsets[lo], offsets[hi]) != 1)
          std::swap(lo, hi);
        if (even(offsets[lo]) && difference(offsets[lo], offsets[hi]) == 1 &&
            (masks.empty() || sameMask(masks[lo], masks[hi])) &&
            preservesRawBufferBounds(offsets[lo]))
          pairs.emplace_back(lo, hi);
      }
      if (pairs.size() * 2 == canonicalCount)
        return pairs;
      if (pairs.size() > best.size())
        best = std::move(pairs);
    }
    return best;
  }
};
