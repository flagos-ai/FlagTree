// Included in LoadStoreOpToLLVM.cpp's anonymous namespace. No shape selector.
// A plan changes neither descriptor bounds nor the predicate. Its facts prove
// the full original lowered scalar offset over all input bit patterns, before
// splitting an exact nonnegative core+constant. No mask facts authorize bounds.
class BufferStoreScalarOffset {
public:
  static constexpr uint64_t Records = uint64_t(INT32_MAX)-1;
  static constexpr uint64_t MaxElement = (Records-2)/2; // Full i16 must fit.
  struct Node {
    enum Kind {Keep,Zero,Add,Mul,SExt,ZExt,Trunc} kind=Keep;
    Value value;Type type;std::shared_ptr<Node> a,b;
    uint64_t constant=0,factor=0,coreMax=0;
  };
  using Plan=std::shared_ptr<Node>;
private:
  struct Facts {uint64_t maximum,zero;unsigned width;};
  DenseMap<Value,Value> normalized;
  DenseMap<Value,Facts> factsMemo;
  DenseMap<Value,Plan> planMemo;
  llvm::SmallDenseSet<Value,32> active;
  DenseMap<const Node*,Value> emitted;
  unsigned aggregateSteps=0,factSteps=0,planSteps=0;
  static uint64_t widthMask(unsigned width) {return width==64?~uint64_t(0):(uint64_t(1)<<width)-1;}
  static uint64_t upperZeros(uint64_t maximum,unsigned width) {
    uint64_t used=0;while(maximum){used=(used<<1)|1;maximum>>=1;}
    return widthMask(width)&~used;
  }
  static unsigned lowZeros(uint64_t zero,unsigned width) {
    unsigned n=0;while(n<width&&((zero>>n)&1))++n;return n;
  }
  std::optional<uint64_t> integerConstant(Value value) {
    value=scalar(value);if(!value)return {};
    Attribute attr;if(!matchPattern(value,m_Constant(&attr)))return {};
    auto c=dyn_cast<IntegerAttr>(attr);if(!c||c.getValue().getBitWidth()>64)return {};
    return c.getValue().getZExtValue(); // Bit pattern; negativity checked by bounds.
  }
  Value scalar(Value value) {
    if(auto it=normalized.find(value);it!=normalized.end())return it->second;
    Value original=value;
    while(true) {
      if(++aggregateSteps>32768)return {};
      if(auto cast=value.getDefiningOp<LLVM::BitcastOp>()) {
        auto type=dyn_cast<IntegerType>(value.getType());
        // Tensor constants can lower to exact i32->i32 / i64->i64 identity
        // bitcasts. Do not cross vector, pointer, float, or width-changing casts.
        if(!type||(type.getWidth()!=32&&type.getWidth()!=64)||
           cast->getOperand(0).getType()!=value.getType())return {};
        value=cast->getOperand(0);continue;
      }
      auto extract=value.getDefiningOp<LLVM::ExtractValueOp>();
      if(!extract)break;
      auto position=extract.getPosition();
      auto type=dyn_cast<LLVM::LLVMStructType>(extract.getContainer().getType());
      if(position.size()!=1||!type||type.isOpaque()||position[0]<0||
         uint64_t(position[0])>=type.getBody().size()||type.getBody()[position[0]]!=value.getType())return {};
      Value container=extract.getContainer();
      while(true) {
        if(++aggregateSteps>32768)return {};
        auto insert=container.getDefiningOp<LLVM::InsertValueOp>();
        if(!insert||insert.getPosition().size()!=1||insert.getContainer().getType()!=type||insert.getType()!=type)return {};
        int64_t field=insert.getPosition()[0];
        if(field<0||uint64_t(field)>=type.getBody().size()||insert.getValue().getType()!=type.getBody()[field])return {};
        if(field==position[0]) {value=insert.getValue();break;}
        container=insert.getContainer();
      }
    }
    normalized[original]=value;return value;
  }
  std::optional<Facts> facts(Value value,unsigned depth=0) {
    value=scalar(value);if(!value||depth>48||++factSteps>16384)return {};
    if(auto it=factsMemo.find(value);it!=factsMemo.end())return it->second;
    auto type=dyn_cast<IntegerType>(value.getType());
    if(!type||(type.getWidth()!=32&&type.getWidth()!=64))return {};
    unsigned width=type.getWidth();uint64_t mask=widthMask(width);
    auto save=[&](uint64_t maximum,uint64_t zero)->std::optional<Facts> {
      Facts result{maximum,zero&mask,width};factsMemo[value]=result;return result;
    };
    if(auto c=integerConstant(value))return save(*c,mask&~*c);
    // An opaque bit pattern can still be safely bounded by a subsequent AND.
    // It is never cloned or rematerialized, and alone cannot pass the range gate.
    auto unknown=[&](){return save(mask,0);};
    if(!active.insert(value).second)return {};
    auto erase=llvm::make_scope_exit([&]{active.erase(value);});
    Operation* op=value.getDefiningOp();if(!op)return unknown();
    if(isa<LLVM::SExtOp,LLVM::ZExtOp,LLVM::TruncOp>(op)) {
      auto a=facts(op->getOperand(0),depth+1);if(!a||a->maximum>MaxElement)return unknown();
      return save(a->maximum,upperZeros(a->maximum,width)|(a->zero&mask));
    }
    if(auto select=dyn_cast<LLVM::SelectOp>(op)) {
      // Join both complete value domains. The condition (including a store
      // mask) supplies no range facts and the original select stays intact.
      auto a=facts(select.getTrueValue(),depth+1);
      auto b=facts(select.getFalseValue(),depth+1);
      if(!a||!b||a->width!=width||b->width!=width)return unknown();
      uint64_t maximum=std::max(a->maximum,b->maximum);
      return save(maximum,(a->zero&b->zero)|upperZeros(maximum,width));
    }
    if(op->getNumOperands()!=2||op->getNumResults()!=1)return unknown();
    auto a=facts(op->getOperand(0),depth+1),b=facts(op->getOperand(1),depth+1);
    if(!a||!b||a->width!=width||b->width!=width)return unknown();
    if(isa<LLVM::AndOp>(op)) {
      uint64_t zero=a->zero|b->zero;
      return save(std::min({a->maximum,b->maximum,mask&~zero}),zero);
    }
    if(isa<LLVM::OrOp,LLVM::XOrOp>(op)) {
      // Either bitwise result is zero wherever both operands are known zero.
      // This bounds XOR without assuming it is addition or rewriting its core.
      uint64_t zero=a->zero&b->zero,maximum=mask&~zero;
      if(a->maximum<=MaxElement&&b->maximum<=MaxElement-a->maximum)
        maximum=std::min(maximum,a->maximum+b->maximum);
      return save(maximum,zero|upperZeros(maximum,width));
    }
    if(isa<LLVM::LShrOp,LLVM::AShrOp>(op)) {
      auto shift=integerConstant(op->getOperand(1));
      if(!shift||*shift>=width||(isa<LLVM::AShrOp>(op)&&a->maximum>MaxElement))return unknown();
      uint64_t maximum=a->maximum>>*shift;
      return save(maximum,(a->zero>>*shift)|upperZeros(maximum,width));
    }
    if(isa<LLVM::UDivOp,LLVM::URemOp>(op)) {
      // A positive constant makes these bounds valid for every unsigned
      // input bit pattern. Do not infer facts for a possibly-zero divisor.
      auto divisor=integerConstant(op->getOperand(1));
      if(!divisor||!*divisor)return unknown();
      uint64_t maximum=isa<LLVM::UDivOp>(op)
          ? a->maximum/ *divisor : std::min(a->maximum,*divisor-1);
      return save(maximum,upperZeros(maximum,width));
    }
    if(a->maximum>MaxElement||b->maximum>MaxElement)return unknown();
    if(isa<LLVM::AddOp>(op)&&a->maximum<=MaxElement-b->maximum) {
      uint64_t maximum=a->maximum+b->maximum;unsigned low=std::min(lowZeros(a->zero,width),lowZeros(b->zero,width));
      uint64_t lowMask=low==64?mask:(uint64_t(1)<<low)-1;
      return save(maximum,upperZeros(maximum,width)|lowMask);
    }
    if(isa<LLVM::MulOp>(op)&&(!b->maximum||a->maximum<=MaxElement/b->maximum)) {
      uint64_t maximum=a->maximum*b->maximum;unsigned low=std::min(width,lowZeros(a->zero,width)+lowZeros(b->zero,width));
      uint64_t lowMask=low==64?mask:(uint64_t(1)<<low)-1;
      return save(maximum,upperZeros(maximum,width)|lowMask);
    }
    if(isa<LLVM::ShlOp>(op)) {
      auto shift=integerConstant(op->getOperand(1));
      if(!shift||*shift>=width||*shift>=31||a->maximum>(MaxElement>>*shift))return unknown();
      uint64_t maximum=a->maximum<<*shift;
      return save(maximum,(a->zero<<*shift)|((uint64_t(1)<<*shift)-1)|upperZeros(maximum,width));
    }
    return unknown();
  }
  Plan split(Value value,unsigned depth=0) {
    value=scalar(value);if(!value||depth>48||++planSteps>16384)return {};
    if(auto it=planMemo.find(value);it!=planMemo.end())return it->second;
    auto full=facts(value);if(!full||full->maximum>MaxElement)return {};
    Plan result=std::make_shared<Node>();result->value=value;result->type=value.getType();result->coreMax=full->maximum;
    auto finish=[&]()->Plan {
      if(result->constant>full->maximum)return {};
      result->coreMax=full->maximum-result->constant;
      planMemo[value]=result;return result;
    };
    if(auto c=integerConstant(value)) {
      result->kind=Node::Zero;result->constant=*c;return finish();
    }
    Operation* op=value.getDefiningOp();if(!op)return finish();
    bool add=isa<LLVM::AddOp>(op);
    if(isa<LLVM::OrOp,LLVM::XOrOp>(op)) {
      auto a=facts(op->getOperand(0)),b=facts(op->getOperand(1));
      // OR/XOR=ADD only if all potentially-one bit sets are independently
      // disjoint. An overlapping XOR remains the original intact core.
      add=a&&b&&!(widthMask(full->width)&~a->zero&~b->zero);
    }
    if(add) {
      auto a=split(op->getOperand(0),depth+1),b=split(op->getOperand(1),depth+1);
      if(!a||!b||a->constant>MaxElement-b->constant)return finish();
      result->kind=Node::Add;result->a=a;result->b=b;result->constant=a->constant+b->constant;return finish();
    }
    if(isa<LLVM::MulOp>(op)) {
      for(unsigned side=0;side<2;++side)if(auto factor=integerConstant(op->getOperand(side))) {
        if(*factor>MaxElement)continue;
        auto a=split(op->getOperand(1-side),depth+1);if(!a||(*factor&&a->constant>MaxElement/ *factor))continue;
        result->kind=Node::Mul;result->a=a;result->factor=*factor;result->constant=a->constant* *factor;return finish();
      }
    }
    if(isa<LLVM::ShlOp>(op)) {
      auto shift=integerConstant(op->getOperand(1));
      if(shift&&*shift<31)if(auto a=split(op->getOperand(0),depth+1)) {
        uint64_t factor=uint64_t(1)<<*shift;
        if(a->constant<=MaxElement/factor) {
          result->kind=Node::Mul;result->a=a;result->factor=factor;result->constant=a->constant*factor;return finish();
        }
      }
    }
    if(isa<LLVM::SExtOp,LLVM::ZExtOp,LLVM::TruncOp>(op)) {
      auto a=split(op->getOperand(0),depth+1);
      if(a) {
        result->kind=isa<LLVM::SExtOp>(op)?Node::SExt:isa<LLVM::ZExtOp>(op)?Node::ZExt:Node::Trunc;
        result->a=a;result->constant=a->constant;return finish();
      }
    }
    return finish(); // Unrecognized core is reused intact; no constant removed.
  }
public:
  Plan plan(Value offset, unsigned vectorElements=1) {
    // The descriptor measures bytes and SOFFSET can bypass its bounds. Prove
    // the complete payload, not only its first i16. Keep the original scalar
    // domain/normalization and require every vector byte to fit independently
    // of the mask before exposing any constant through SOFFSET.
    if(!offset.getType().isInteger(32) || !vectorElements ||
       vectorElements>8 || (vectorElements&(vectorElements-1)))return {};
    auto result=split(offset);
    uint64_t maxVectorStart=(Records-2ull*vectorElements)/2;
    return result && result->constant && result->constant<=maxVectorStart &&
        result->coreMax<=maxVectorStart-result->constant ? result : Plan{};
  }
  Value emitCore(const Plan& plan,ConversionPatternRewriter& rewriter,Location loc) {
    if(auto it=emitted.find(plan.get());it!=emitted.end())return it->second;
    auto b=TritonLLVMOpBuilder(loc,rewriter);Value value;
    if(plan->kind==Node::Keep)value=plan->value;
    else if(plan->kind==Node::Zero)value=b.int_val(cast<IntegerType>(plan->type).getWidth(),0);
    else {
      Value a=emitCore(plan->a,rewriter,loc);
      if(plan->kind==Node::Add)value=b.add(a,emitCore(plan->b,rewriter,loc));
      else if(plan->kind==Node::Mul)value=b.mul(a,b.int_val(cast<IntegerType>(plan->type).getWidth(),plan->factor));
      else if(plan->kind==Node::SExt)value=b.sext(plan->type,a);
      else if(plan->kind==Node::ZExt)value=b.zext(plan->type,a);
      else value=b.trunc(plan->type,a);
    }
    emitted[plan.get()]=value;return value;
  }
};
