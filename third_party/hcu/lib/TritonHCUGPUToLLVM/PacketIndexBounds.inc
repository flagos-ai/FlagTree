// Exact signed intervals from ordinary scalar SSA and a canonical counted CF
// loop. No launch bounds, imported certificates, assumptions, shapes or tuning
// configurations are used. Arithmetic that may wrap loses its range proof.
struct PacketIndexBounds { int64_t lo, hi; };
static bool packetBoundsFit(__int128 lo,__int128 hi,Type type) {
  auto integer=dyn_cast<IntegerType>(type);
  if(!integer || integer.getWidth()<2 || integer.getWidth()>64)return false;
  __int128 limit=__int128(1)<<(integer.getWidth()-1);
  return lo>=-limit && hi<limit && lo<=hi;
}
static std::optional<int64_t> packetCounterStep(Value next,Value phi) {
  auto add=next.getDefiningOp<arith::AddIOp>();if(!add)return std::nullopt;
  if(add.getLhs()==phi)return legacyV0SafeInteger(add.getRhs());
  if(add.getRhs()==phi)return legacyV0SafeInteger(add.getLhs());
  return std::nullopt;
}
static std::optional<PacketIndexBounds> packetCountedPhiBounds(
    BlockArgument arg,Operation *anchor) {
  Block *header=arg.getOwner();
  auto test=dyn_cast<cf::CondBranchOp>(header->getTerminator());
  if(!test || test.getTrueDest()!=anchor->getBlock() ||
      !test.getTrueDestOperands().empty())return std::nullopt;
  Block *body=test.getTrueDest();
  if(body==header || body->getSinglePredecessor()!=header ||
      test.getFalseDest()==body || test.getFalseDest()==header)return std::nullopt;
  auto cmp=test.getCondition().getDefiningOp<arith::CmpIOp>();
  auto iv=cmp?dyn_cast<BlockArgument>(cmp.getLhs()):BlockArgument{};
  if(!iv || iv.getOwner()!=header || cmp->getBlock()!=header ||
      cmp.getPredicate()!=arith::CmpIPredicate::slt)return std::nullopt;
  auto bound=legacyV0SafeInteger(cmp.getRhs());if(!bound)return std::nullopt;
  auto back=dyn_cast<cf::BranchOp>(body->getTerminator());
  if(!back || back.getDest()!=header ||
      back.getDestOperands().size()!=header->getNumArguments())return std::nullopt;
  auto predecessors=llvm::to_vector(header->getPredecessors());
  if(predecessors.size()!=2)return std::nullopt;
  Block *entry=nullptr;
  for(Block *pred:predecessors)if(pred!=body) {
    if(entry)return std::nullopt;
    entry=pred;
  }
  if(!entry || entry==header)return std::nullopt;
  auto enter=dyn_cast<cf::BranchOp>(entry->getTerminator());
  if(!enter || enter.getDest()!=header ||
      enter.getDestOperands().size()!=header->getNumArguments())return std::nullopt;
  unsigned primary=iv.getArgNumber(),secondary=arg.getArgNumber();
  auto initial=legacyV0SafeInteger(enter.getDestOperands()[primary]);
  auto step=packetCounterStep(back.getDestOperands()[primary],iv);
  auto seed=legacyV0SafeInteger(enter.getDestOperands()[secondary]);
  auto advance=packetCounterStep(back.getDestOperands()[secondary],arg);
  if(!initial || !step || *step<=0 || !seed || !advance || *initial>=*bound)
    return std::nullopt;
  __int128 trips=(__int128(*bound)-*initial+*step-1)/ *step;
  __int128 finalPrimary=__int128(*initial)+trips* *step;
  __int128 finalSecondary=__int128(*seed)+trips* *advance;
  if(!packetBoundsFit(*initial,finalPrimary,iv.getType()) ||
      !packetBoundsFit(std::min<__int128>(*seed,finalSecondary),
                       std::max<__int128>(*seed,finalSecondary),arg.getType()))
    return std::nullopt;
  __int128 last=__int128(*seed)+(trips-1)* *advance;
  return PacketIndexBounds{int64_t(std::min<__int128>(*seed,last)),
                           int64_t(std::max<__int128>(*seed,last))};
}
static std::optional<PacketIndexBounds> packetIndexBounds(
    Value value,Operation *anchor,llvm::SmallPtrSetImpl<void *> &active,unsigned &fuel) {
  if(!fuel)return std::nullopt;--fuel;
  auto integer=dyn_cast<IntegerType>(value.getType());
  if(!integer || integer.getWidth()<2 || integer.getWidth()>64)return std::nullopt;
  if(auto c=legacyV0SafeInteger(value))return PacketIndexBounds{*c,*c};
  if(!active.insert(value.getAsOpaquePointer()).second)return std::nullopt;
  auto result=[&]()->std::optional<PacketIndexBounds> {
    if(auto arg=dyn_cast<BlockArgument>(value))return packetCountedPhiBounds(arg,anchor);
    Operation *op=value.getDefiningOp();if(!op)return std::nullopt;
    auto range=[&](Value v){return packetIndexBounds(v,anchor,active,fuel);};
    if(isa<arith::ExtSIOp,arith::ExtUIOp,arith::TruncIOp>(op)) {
      auto source=range(op->getOperand(0));
      if(!source || (isa<arith::ExtUIOp>(op) && source->lo<0) ||
          !packetBoundsFit(source->lo,source->hi,value.getType()))return std::nullopt;
      return source;
    }
    if(auto select=dyn_cast<arith::SelectOp>(op)) {
      auto a=range(select.getTrueValue()),b=range(select.getFalseValue());
      if(!a || !b)return std::nullopt;
      return PacketIndexBounds{std::min(a->lo,b->lo),std::max(a->hi,b->hi)};
    }
    if(op->getNumOperands()!=2)return std::nullopt;
    auto a=range(op->getOperand(0)),b=range(op->getOperand(1));
    if(isa<arith::MaxSIOp,arith::MinSIOp>(op)) {
      // A clamp bounds an otherwise unknown integer too. Use its full signed
      // source domain, never an assumed nonnegative program/loop index.
      auto full=[&](Value v) {
        unsigned w=cast<IntegerType>(v.getType()).getWidth();
        __int128 limit=__int128(1)<<(w-1);
        return PacketIndexBounds{int64_t(-limit),int64_t(limit-1)};
      };
      if(!a)a=full(op->getOperand(0));if(!b)b=full(op->getOperand(1));
      if(isa<arith::MaxSIOp>(op))return PacketIndexBounds{std::max(a->lo,b->lo),std::max(a->hi,b->hi)};
      return PacketIndexBounds{std::min(a->lo,b->lo),std::min(a->hi,b->hi)};
    }
    if(!a || !b)return std::nullopt;
    __int128 lo,hi;
    if(isa<arith::AddIOp>(op)){lo=__int128(a->lo)+b->lo;hi=__int128(a->hi)+b->hi;}
    else if(isa<arith::SubIOp>(op)){lo=__int128(a->lo)-b->hi;hi=__int128(a->hi)-b->lo;}
    else if(isa<arith::MulIOp>(op)) {
      __int128 v[]={__int128(a->lo)*b->lo,__int128(a->lo)*b->hi,
                    __int128(a->hi)*b->lo,__int128(a->hi)*b->hi};
      lo=*std::min_element(v,v+4);hi=*std::max_element(v,v+4);
    } else if(isa<arith::DivSIOp,arith::RemSIOp>(op) && a->lo>=0 &&
              b->lo==b->hi && b->lo>0) {
      if(isa<arith::DivSIOp>(op)){lo=a->lo/b->lo;hi=a->hi/b->lo;}
      else {lo=0;hi=std::min(a->hi,b->lo-1);}
    } else return std::nullopt;
    if(!packetBoundsFit(lo,hi,value.getType()))return std::nullopt;
    return PacketIndexBounds{int64_t(lo),int64_t(hi)};
  }();
  active.erase(value.getAsOpaquePointer());return result;
}
static std::optional<PacketIndexBounds> packetIndexBounds(Value v,Operation *anchor) {
  llvm::SmallPtrSet<void *,32> active;unsigned fuel=512;return packetIndexBounds(v,anchor,active,fuel);
}
