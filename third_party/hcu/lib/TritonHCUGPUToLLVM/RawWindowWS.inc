// Independent raw-window CTA protocol. No legacy MLS/shape gates are imported.
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Verifier.h"
#include "Dialect/TritonHCUGPU/IR/RawWindowLayout.h"
#include "triton/Dialect/TritonGPU/IR/LinearLayoutConversions.h"
#include "llvm/Support/MathExtras.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/APInt.h"
#include <optional>
#include <functional>
#include <string>
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/SmallPtrSet.h"
namespace {
namespace rawws {
constexpr llvm::StringLiteral Request="ttg.hcu.raw_window_ws";
constexpr llvm::StringLiteral Witness="ttg.hcu.raw_window_ws.internal_witness";
constexpr llvm::StringLiteral Staggered="ttg.hcu.raw_window_ws.staggered";
constexpr llvm::StringLiteral DisableWaveSpecialization="ttg.hcu.raw_window_ws.disable_wave_specialization";
constexpr llvm::StringLiteral DisableLgkmCleanup="ttg.hcu.raw_window_ws.disable_lgkm_cleanup";
struct Segment {bool read=false,copy=false,wait=false,valid=true;unsigned copies=0,waits=0;SmallVector<unsigned> readWidths;};
struct Proof {
  Attribute token;unsigned base=0,producerSites=0,consumerSites=0;
  bool staggered=false;
  // Per actor, one segment before each dynamic CTA and one terminal segment.
  SmallVector<SmallVector<Segment>> segments;
};
// Two invocations of this pass bracket type/CF lowering in one thread. Imported
// attrs alone cannot manufacture a proof: the freshly issued distinct token
// must be present in this same-process registry, bound to the same ModuleOp.
static thread_local llvm::DenseMap<Operation*,Proof> proofs;
static Value capture(Value v) {
  if(auto a=dyn_cast<BlockArgument>(v))
    if(auto p=dyn_cast<WarpSpecializePartitionsOp>(a.getOwner()->getParentOp())) {
      auto xs=p.getParentOp().getExplicitCaptures();
      if(a.getOwner()==&a.getOwner()->getParent()->front() && a.getArgNumber()<xs.size())return capture(xs[a.getArgNumber()]);
    }
  return v;
}
using Env=llvm::DenseMap<Value,int64_t>;
static std::optional<bool> compareBits(StringRef predicate,const llvm::APInt&a,const llvm::APInt&b) {
  if(predicate=="eq")return a==b;if(predicate=="ne")return a!=b;
  if(predicate=="slt")return a.slt(b);if(predicate=="sle")return a.sle(b);
  if(predicate=="sgt")return a.sgt(b);if(predicate=="sge")return a.sge(b);
  if(predicate=="ult")return a.ult(b);if(predicate=="ule")return a.ule(b);
  if(predicate=="ugt")return a.ugt(b);if(predicate=="uge")return a.uge(b);
  return {};
}
// Environments store mathematical/signed int64 values, including i1 true=1.
// Construct a uint64 bit pattern first, then truncate to the IR width. Never
// request APInt's signed-fit constructor for an arbitrary N-bit bit pattern.
static llvm::APInt integerBitPattern(unsigned width,int64_t value) {
  assert(width>=1&&width<=64);
  return llvm::APInt(64,static_cast<uint64_t>(value)).trunc(width);
}
static unsigned scalarWidth(Type t) {
  if(auto i=dyn_cast<IntegerType>(t))return i.getWidth();
  return t.isIndex()?32:0;
}
static std::optional<int64_t> integer(Value v,const Env& env,unsigned depth=0,bool comparisons=false) {
  if(depth>64||!v.getType().isIntOrIndex())return {};
  auto found=env.find(v);if(found!=env.end())return found->second;
  Value root=capture(v);if(root!=v)return integer(root,env,depth+1,comparisons);
  auto op=v.getDefiningOp();if(!op)return {};
  if(auto c=dyn_cast<arith::ConstantOp>(op))
    if(auto a=dyn_cast<IntegerAttr>(c.getValue()))return a.getInt();
  StringRef n=op->getName().getStringRef();
  if(op->getNumOperands()==1 && (n=="arith.index_cast"||n=="arith.extsi"||n=="arith.extui"||n=="arith.trunci")) {
    auto x=integer(op->getOperand(0),env,depth+1,comparisons);if(!x)return {};
    auto srcType=dyn_cast<IntegerType>(op->getOperand(0).getType());
    auto dstType=dyn_cast<IntegerType>(v.getType());
    unsigned sw=srcType?srcType.getWidth():32,dw=dstType?dstType.getWidth():32;
    if(sw>64||dw>64)return {};
    llvm::APInt bits=integerBitPattern(sw,*x);
    bits=n=="arith.extui"?bits.zextOrTrunc(dw):bits.sextOrTrunc(dw);
    return bits.getSExtValue();
  }
  if(auto cmp=dyn_cast<arith::CmpIOp>(op)) {
    if(!comparisons)return {};
    auto x=integer(cmp.getLhs(),env,depth+1,comparisons),y=integer(cmp.getRhs(),env,depth+1,comparisons);
    unsigned width=scalarWidth(cmp.getLhs().getType());
    if(!x||!y||!width||width>64)return {};
    auto result=compareBits(arith::stringifyCmpIPredicate(cmp.getPredicate()),integerBitPattern(width,*x),integerBitPattern(width,*y));
    if(result)return int64_t(*result);return {};
  }
  if(auto select=dyn_cast<arith::SelectOp>(op)) {
    if(!comparisons)return {};
    auto c=integer(select.getCondition(),env,depth+1,comparisons);if(!c)return {};
    return integer(*c?select.getTrueValue():select.getFalseValue(),env,depth+1,comparisons);
  }
  if(op->getNumOperands()!=2)return {};
  auto x=integer(op->getOperand(0),env,depth+1,comparisons),y=integer(op->getOperand(1),env,depth+1,comparisons);
  if(!x||!y)return {};
  auto it=dyn_cast<IntegerType>(v.getType());unsigned width=it?it.getWidth():32;
  if(width>64)return {};
  llvm::APInt a=integerBitPattern(width,*x),b=integerBitPattern(width,*y),z(width,0);
  if(n=="arith.addi")z=a+b;else if(n=="arith.subi")z=a-b;
  else if(n=="arith.muli")z=a*b;else if(n=="arith.andi")z=a&b;
  else if(n=="arith.ori")z=a|b;else if(n=="arith.xori")z=a^b;
  else if(n=="arith.divsi"||n=="arith.remsi") {
    if(b.isZero()||(a.isMinSignedValue()&&b.isAllOnes()))return {};
    z=n=="arith.divsi"?a.sdiv(b):a.srem(b);
  } else if(n=="arith.divui"||n=="arith.remui") {
    if(b.isZero())return {};z=n=="arith.divui"?a.udiv(b):a.urem(b);
  } else return {};
  return z.getSExtValue();
}
struct Window {Value owner;int64_t slot;bool operator==(const Window& x)const{return owner==x.owner&&slot==x.slot;}};
static std::optional<Window> window(Value v,const Env& env,bool comparisons=false) {
  v=capture(v);auto ix=v.getDefiningOp<MemDescIndexOp>();if(!ix)return {};
  Value root=capture(ix.getSrc());auto alloc=root.getDefiningOp<LocalAllocOp>();
  auto slot=integer(ix.getIndex(),env,0,comparisons);if(!alloc||!slot)return {};
  auto t=alloc.getType();
  if(alloc.getSrc()||t.getRank()!=2||t.getShape()[1]!=16384||t.getShape()[0]<1||t.getShape()[0]>2||*slot<0||*slot>=t.getShape()[0]||alloc.getAlignmentOrDefault()<32768)return {};
  return Window{root,*slot};
}
#include "RawWindowStoreSuffix.inc"
enum class Kind {Copy,Wait,Barrier,Read};
struct Event {Kind kind;Operation* op;std::optional<Window> view;};
struct Trace {
  bool producer,staggered=false;unsigned relativeWave=0,base=0;
  SmallVector<Event> events;std::string why;unsigned budget=0;
  explicit Trace(bool p,bool s=false,unsigned w=0,unsigned b=0):producer(p),staggered(s),relativeWave(w),base(b){}
  bool reject(StringRef s){why=s.str();return false;}
  bool append(Kind k,Operation* op,std::optional<Window> w={}) {
    if(events.size()>=65536)return reject("event budget exceeded");
    events.push_back({k,op,w});return true;
  }
  bool block(Block& b,Env& env) {
    for(Operation& op:b) {
      if(++budget>1048576)return reject("finite interpretation budget exceeded");
      if(auto loop=dyn_cast<scf::ForOp>(&op)) {
        auto lb=integer(loop.getLowerBound(),env,0,staggered),ub=integer(loop.getUpperBound(),env,0,staggered),step=integer(loop.getStep(),env,0,staggered);
        if(!lb||!ub||!step||*step<=0||!llvm::hasSingleElement(loop.getRegion()))return reject("only finite positive-step single-block SCF loops are supported");
        // Carry scalar values only when exact. Tensor accumulators are opaque;
        // any attempt to use them as a slot/control value will reject later.
        for(auto pair:llvm::zip(loop.getRegionIterArgs(),loop.getInitArgs())) {
          auto x=integer(std::get<1>(pair),env,0,staggered);
          if(x)env[std::get<0>(pair)]=*x;else env.erase(std::get<0>(pair));
        }
        auto ivType=dyn_cast<IntegerType>(loop.getInductionVar().getType());
        unsigned ivWidth=ivType?ivType.getWidth():32; // target Index conversion is32.
        if(ivWidth<2||ivWidth>64)return reject("unsupported SCF induction width");
        int64_t ivMax=llvm::APInt::getSignedMaxValue(ivWidth).getSExtValue();
        int64_t ivMin=llvm::APInt::getSignedMinValue(ivWidth).getSExtValue();
        if(*lb<ivMin||*lb>ivMax||*ub<ivMin||*ub>ivMax||*step>ivMax)return reject("SCF bounds outside actual induction width");
        int64_t iv=*lb;unsigned iterations=0;
        while(iv<*ub) {
          if(++iterations>65536)return reject("SCF trip budget exceeded");
          env[loop.getInductionVar()]=iv;
          if(!block(*loop.getBody(),env))return false;
          auto yield=cast<scf::YieldOp>(loop.getBody()->getTerminator());
          Env next=env;
          for(auto pair:llvm::zip(loop.getRegionIterArgs(),yield.getOperands())) {
            auto x=integer(std::get<1>(pair),env,0,staggered);
            if(x)next[std::get<0>(pair)]=*x;else next.erase(std::get<0>(pair));
          }
          env=std::move(next);
          if(iv>ivMax-*step)return reject("final SCF induction update overflows");
          iv+=*step;
        }
        for(auto pair:llvm::zip(loop.getResults(),loop.getRegionIterArgs())) {
          auto x=integer(std::get<1>(pair),env,0,staggered);
          if(x)env[std::get<0>(pair)]=*x;else env.erase(std::get<0>(pair));
        }
        continue;
      }
      if(auto branch=dyn_cast<scf::IfOp>(&op)) {
        if(!staggered)return reject("SCF conditional must be folded before raw WS proof");
        auto condition=integer(branch.getCondition(),env,0,staggered);
        if(!condition) {
          // Unknown output conditions may not bypass any protocol event. Both
          // terminal branches are checked, never assumed or interpreted true.
          if(!producer&&earlyStoreOnlySuffix(branch))continue;
          return reject("unknown/lane-dependent SCF condition outside a proved store-only suffix");
        }
        Region& chosen=*condition?branch.getThenRegion():branch.getElseRegion();
        if(chosen.empty()) {if(branch.getNumResults())return reject("missing SCF result branch");continue;}
        if(!llvm::hasSingleElement(chosen))return reject("multi-block SCF branch");
        if(!block(chosen.front(),env))return false;
        auto yield=dyn_cast<scf::YieldOp>(chosen.front().getTerminator());
        if(!yield||yield.getNumOperands()!=branch.getNumResults())return reject("invalid SCF branch yield");
        Env next=env;
        for(auto pair:llvm::zip(branch.getResults(),yield.getOperands())) {
          auto value=integer(std::get<1>(pair),env,0,staggered);
          if(value)next[std::get<0>(pair)]=*value;else next.erase(std::get<0>(pair));
        }
        env=std::move(next);continue;
      }
      if(op.getNumRegions())return reject("unknown nested region/control");
      StringRef name=op.getName().getStringRef();
      if(name=="hcug.warp_group_id") {
        if(!staggered||producer||op.getNumOperands()||op.getNumResults()!=1||!op.getResult(0).getType().isInteger(32))return reject("invalid wave-group query context/type");
        auto size=op.getAttrOfType<IntegerAttr>("groupSize");
        if(!size||size.getInt()<=0||size.getInt()>base||!llvm::isPowerOf2_64(size.getInt()))return reject("invalid wave-group size");
        env[op.getResult(0)]=relativeWave/size.getInt();continue;
      }
      if(name=="hcug.raw_window_copy") {
        if(!producer)return reject("copy in compute region");
        auto w=window(op.getOperand(1),env,staggered);if(!w)return reject("copy requires exact whole-window slot");
        if(!append(Kind::Copy,&op,w))return false;continue;
      }
      if(name=="hcug.raw_window_wait") {
        if(!producer)return reject("raw wait in compute region");
        if(!append(Kind::Wait,&op))return false;continue;
      }
      if(isa<LocalBarrierOp>(&op)) {if(!append(Kind::Barrier,&op))return false;continue;}
      if(name=="hcug.raw_window_read") {
        if(producer)return reject("read in producer region");
        if(staggered&&!isa_and_nonnull<UnitAttr>(op.getAttr("hcu.full_read_completion")))return reject("staggered read requires explicit full LDS completion");
        auto w=window(op.getOperand(0),env,staggered);if(!w)return reject("read requires exact whole-window slot");
        if(!append(Kind::Read,&op,w))return false;continue;
      }
      if(isa<scf::YieldOp,WarpYieldOp,WarpReturnOp>(&op))continue;
      if(isa<MemDescIndexOp,LocalAllocOp>(&op))continue;
      if(name.contains("barrier")||name.contains("async")||name.contains("wait"))return reject("unknown synchronization event");
      if(!producer && isa<triton::MatrixLoadOp>(&op) && isa_and_nonnull<UnitAttr>(op.getAttr("hcu.structured_register")))continue;
      if(!producer && (isa<triton::StoreOp>(&op)||name=="hcug.buffer_store"))continue;
      if(!producer && name=="tt.elementwise_inline_asm") {
        auto text=op.getAttrOfType<StringAttr>("asm_string");
        if(text&&(text.getValue().empty()||text.getValue()=="s_setprio 0"||text.getValue()=="s_setprio 1"))continue;
        return reject("unknown compute inline asm");
      }
      if(op.hasTrait<OpTrait::IsTerminator>()||name.contains("call")||name.contains("asm"))return reject("unknown control/call/asm operation");
      if(isPure(&op))continue;
      return reject("unknown memory/effect operation");
    }
    return true;
  }
};
static bool coverRead(Operation* op,llvm::BitVector& covered,Attribute storage,Attribute basis) {
  if(op->getAttr("storageShape")!=storage||op->getAttr("basis")!=basis)return false;
  auto shape=dyn_cast<DenseI32ArrayAttr>(storage),origin=op->getAttrOfType<DenseI32ArrayAttr>("origin"),order=op->getAttrOfType<DenseI32ArrayAttr>("order");
  auto t=dyn_cast<RankedTensorType>(op->getResult(0).getType());
  if(!shape||shape.size()!=2||!origin||origin.size()!=2||!order||order.size()!=2||!t||t.getRank()!=2||order[0]<0||order[0]>1||order[1]!=1-order[0])return false;
  if(int64_t(shape[0])*shape[1]!=16384||origin[0]<0||origin[1]<0||int64_t(origin[0])+t.getShape()[order[0]]>shape[0]||int64_t(origin[1])+t.getShape()[order[1]]>shape[1])return false;
  for(int64_t r=0;r<t.getShape()[order[0]];++r)for(int64_t c=0;c<t.getShape()[order[1]];++c)
    covered.set((r+origin[0])*shape[1]+c+origin[1]);
  return true;
}
// Compare actors at dynamic CTA cuts. Within a cut actors are concurrent:
// full completion is local until the cut, never permission to overwrite early.
using Cuts=SmallVector<SmallVector<Event>>;
static Cuts cuts(const Trace& t) {
  Cuts result(1);for(const Event& e:t.events) {
    if(e.kind==Kind::Barrier)result.emplace_back();else result.back().push_back(e);
  }return result;
}
// Same finite vector legality as RawWindowReadConversion; no guessing that an
// IR tensor read is a single hardware request. This describes actual pre-native
// LLVM loads. Native event/readiness remains a separate parent-owned guard.
static std::optional<SmallVector<unsigned>> readLoadWidths(Operation* op,unsigned base) {
  namespace rw=mlir::triton::hcugpu::rawwindow;
  auto ty=dyn_cast<RankedTensorType>(op->getResult(0).getType());
  auto storage=op->getAttrOfType<DenseI32ArrayAttr>("storageShape"),basis=op->getAttrOfType<DenseI32ArrayAttr>("basis");
  auto origin=op->getAttrOfType<DenseI32ArrayAttr>("origin"),order=op->getAttrOfType<DenseI32ArrayAttr>("order");
  if(!ty||ty.getRank()!=2||!isa<DistributedEncodingTrait>(ty.getEncoding())||!storage||storage.size()!=2||!basis||basis.size()!=14||!origin||origin.size()!=2||!order||order.size()!=2)return {};
  std::array<unsigned,14> bits{};for(unsigned i=0;i<14;++i)bits[i]=basis[i];
  auto chart=rw::SourceLayout::get(storage[0],storage[1],bits);if(!chart)return {};
  auto ll=triton::gpu::toLinearLayout(ty);auto* ctx=op->getContext();
  auto reg=StringAttr::get(ctx,"register"),lane=StringAttr::get(ctx,"lane"),warp=StringAttr::get(ctx,"warp"),block=StringAttr::get(ctx,"block");
  if(!ll.hasInDim(reg)||!ll.hasInDim(lane)||!ll.hasInDim(warp)||!ll.hasInDim(block)||ll.getInDimSize(lane)!=64||ll.getInDimSize(warp)!=base||ll.getInDimSize(block)!=1)return {};
  unsigned registers=ll.getInDimSize(reg);if(!registers||uint64_t(registers)*64*base>1048576)return {};
  auto address=[&](unsigned r,unsigned l,unsigned w) {
    auto xy=ll.apply({{reg,r},{lane,l},{warp,w},{block,0}});
    unsigned row=xy[order[0]].second+origin[0],col=xy[order[1]].second+origin[1];
    return chart->address(row*chart->columns+col);
  };
  for(unsigned width:{8u,4u,2u,1u}) {
    if(registers%width)continue;bool valid=true;
    for(unsigned w=0;w<base&&valid;++w)for(unsigned l=0;l<64&&valid;++l)for(unsigned r=0;r<registers&&valid;r+=width) {
      unsigned first=address(r,l,w);if(first%width){valid=false;break;}
      for(unsigned j=1;j<width;++j)if(address(r+j,l,w)!=first+j){valid=false;break;}
    }
    if(valid)return SmallVector<unsigned>(registers/width,width);
  }return {};
}
static SmallVector<Segment> projection(const Cuts& cs,unsigned base,llvm::DenseMap<Operation*,SmallVector<unsigned>>& cache) {
  SmallVector<Segment> result;
  for(const auto& es:cs) {Segment s;for(const Event& e:es) {
    s.read|=e.kind==Kind::Read;s.copy|=e.kind==Kind::Copy;s.wait|=e.kind==Kind::Wait;
    if(e.kind==Kind::Copy)s.copies+=16384/(4*64*8); // RawWindowCopy ABI: 16B/lane, four waves.
    if(e.kind==Kind::Wait)++s.waits;
    if(e.kind==Kind::Read) {
      auto found=cache.find(e.op);
      if(found==cache.end()) {
        auto widths=readLoadWidths(e.op,base);
        if(!widths){s.valid=false;continue;}
        found=cache.try_emplace(e.op,std::move(*widths)).first;
      }
      s.readWidths.append(found->second.begin(),found->second.end());
    }
  }result.push_back(std::move(s));}return result;
}
static LogicalResult staggeredEarly(ModuleOp mod,WarpSpecializeOp ws,unsigned base) {
  llvm::errs()<<"[rawws-v15] staggeredEarly-entry\n";
  auto reject=[&](StringRef s)->LogicalResult{return mod.emitError("raw-window-ws staggered: ")<<s;};
  Trace producer(true,true,0,base);Env env;
  if(!producer.block(ws.getPartitionRegions()[0]->front(),env))return reject(producer.why);
  for(unsigned w=1;w<4;++w) {
    Trace other(true,true,w,base);Env e;
    if(!other.block(ws.getPartitionRegions()[0]->front(),e))return reject(other.why);
    if(other.events.size()!=producer.events.size())return reject("producer waves have different event counts");
    for(unsigned i=0;i<other.events.size();++i) {
      const auto& a=other.events[i];const auto& b=producer.events[i];
      if(a.kind!=b.kind||a.op!=b.op||bool(a.view)!=bool(b.view)||(a.view&&!(*a.view==*b.view)))return reject("producer waves have different event/slot paths");
    }
  }
  // Strict producer skeleton: T copy,A,matching-wait,B phases, then two
  // no-copy terminal CTAs. No ended-wave barrier participation is assumed.
  auto& p=producer.events;size_t position=0;unsigned phases=0;
  Value owner;std::optional<Window> previous;
  while(position<p.size()&&p[position].kind==Kind::Copy) {
    if(p.size()-position<4)return reject("incomplete producer phase");
    const auto& cp=p[position];
    if(p[position+1].kind!=Kind::Barrier||p[position+2].kind!=Kind::Wait||p[position+3].kind!=Kind::Barrier||p[position+2].op->getOperand(0).getDefiningOp()!=cp.op)return reject("producer must copy,A,matching VM0,B");
    if(!owner)owner=cp.view->owner;
    if(owner!=cp.view->owner||(previous&&*previous==*cp.view))return reject("producer owner/alternating-window mismatch");
    previous=cp.view;position+=4;++phases;
  }
  if(!phases||p.size()-position!=2||p[position].kind!=Kind::Barrier||p[position+1].kind!=Kind::Barrier)return reject("requires exactly two no-copy terminal CTAs");
  Cuts pc=cuts(producer);SmallVector<Trace> consumers;
  for(unsigned w=0;w<base;++w) {
    Trace c(false,true,w,base);Env e;
    if(!c.block(ws.getDefaultRegion().front(),e))return reject(c.why);
    if(cuts(c).size()!=pc.size())return reject("unequal dynamic CTA ordinal count");
    consumers.push_back(std::move(c));
  }
  // Producer has no wave-group query. All source control must have exact
  // scalar values; otherwise Trace rejected rather than assuming uniformity.
  SmallVector<Cuts> cc;for(const auto& c:consumers)cc.push_back(cuts(c));
  struct Frame {
    Operation* copy=nullptr;bool published=false,pending=false,complete=false;
    SmallVector<llvm::BitVector> covered;SmallVector<SmallVector<Operation*>> reads;
  } frames[2];
  for(auto& f:frames)for(unsigned w=0;w<base;++w) {f.covered.emplace_back(16384,false);f.reads.emplace_back();}
  auto retiredWhole=[&](const Frame& f) {
    if(!f.published||!llvm::all_of(f.covered,[](const llvm::BitVector& b){return b.all();}))return false;
    // Each wave executes the same distributed tensor views (possibly distinct
    // SSA ops in different branches). Their union is therefore the complete
    // full-parent tensor, not a falsely complete per-wave logical rectangle.
    for(unsigned w=1;w<base;++w) {
      if(f.reads[w].size()!=f.reads[0].size())return false;
      for(unsigned r=0;r<f.reads[0].size();++r) {
        auto a=f.reads[0][r],b=f.reads[w][r];
        if(a->getAttrDictionary()!=b->getAttrDictionary()||a->getResult(0).getType()!=b->getResult(0).getType())return false;
      }
    }return true;
  };
  for(unsigned cut=0;cut<pc.size();++cut) {
    // Derive the write set before processing either actor. Completing a read
    // inside this same interval cannot order a producer on another wave.
    std::optional<Window> written;
    for(const auto& e:pc[cut])if(e.kind==Kind::Copy) {
      if(written)return reject("multiple copies in one CTA interval");written=e.view;
    }
    for(unsigned w=0;w<base;++w)for(const auto& e:cc[w][cut]) {
      if(e.kind!=Kind::Read)return reject("unexpected compute queue event");
      if(e.view->owner!=owner)return reject("compute read has another physical owner");
      if(written&&*written==*e.view)return reject("cross-wave read/copy overlap before the same CTA");
      Frame& f=frames[e.view->slot];
      if(!f.copy||!f.published||f.pending)return reject("compute read before completed DMA publication");
      if(!coverRead(e.op,f.covered[w],f.copy->getAttr("storageShape"),f.copy->getAttr("basis")))return reject("compute chart does not match published frame");
      f.reads[w].push_back(e.op);
      // Trace already required hcu.full_read_completion on each read. The
      // lowerer must materialize LGKM0; late proof verifies actual completion.
    }
    for(const auto& e:pc[cut]) {
      if(e.kind==Kind::Copy) {
        Frame& f=frames[e.view->slot];
        if(f.pending)return reject("overwrite while DMA still pending");
        if(f.copy&&!retiredWhole(f))return reject("overwrite before whole old-frame reads retired at a prior CTA");
        f.copy=e.op;f.pending=true;f.complete=false;f.published=false;
        for(auto& bits:f.covered)bits.reset();for(auto& reads:f.reads)reads.clear();
      } else if(e.kind==Kind::Wait) {
        Operation* cp=e.op->getOperand(0).getDefiningOp();Frame* found=nullptr;
        for(auto& f:frames)if(f.copy==cp&&f.pending)found=&f;
        if(!found)return reject("wait without the exact pending copy");
        found->complete=true;
      } else return reject("unexpected producer interval event");
    }
    // A cut exists after every segment except the final one. Publication and
    // cross-wave retirement are acquired only there, not at a local wait.
    if(cut+1<pc.size())for(auto& f:frames)if(f.pending&&f.complete) {
      f.pending=false;f.published=true;
    }
  }
  for(const auto& f:frames)if(f.copy&&(f.pending||!retiredWhole(f)))return reject("terminal frame not completely consumed/retired");
  // No queue event may occur after the last terminal barrier. This also makes
  // the common join independent of terminal reads or outstanding DMA.
  if(!pc.back().empty())return reject("producer terminal queue event");
  for(const auto& c:cc)if(!c.back().empty())return reject("consumer terminal queue event");
  Proof proof;proof.base=base;proof.staggered=true;
  llvm::DenseMap<Operation*,SmallVector<unsigned>> cache;
  for(const auto& c:cc)proof.segments.push_back(projection(c,base,cache));
  for(unsigned w=0;w<4;++w)proof.segments.push_back(projection(pc,base,cache));
  for(const auto& actor:proof.segments)for(const auto& s:actor)if(!s.valid)return reject("finite distributed read lowering plan unavailable");
  llvm::SmallPtrSet<Operation*,16> ps,cs;
  for(const auto& e:p)if(e.kind==Kind::Barrier)ps.insert(e.op);
  for(const auto& c:consumers)for(const auto& e:c.events)if(e.kind==Kind::Barrier)cs.insert(e.op);
  proof.producerSites=ps.size();proof.consumerSites=cs.size();
  proof.token=DistinctAttr::create(UnitAttr::get(mod.getContext()));
  mod->setAttr(Witness,proof.token);proofs[mod.getOperation()]=std::move(proof);
  Builder b(mod.getContext());ws.setWarpGroupStartIds(ArrayRef<int32_t>{int32_t(base)});
  mod->setAttr("ttg.total-num-warps",b.getI32IntegerAttr(base+4));
  mod->setAttr("ttg.hcu.raw_window_ws.phases",b.getI32IntegerAttr(phases));
  return success();
}
static LogicalResult early(ModuleOp mod,bool wdra) {
  llvm::errs()<<"[rawws-v15] early-entry before-registry\n";
  proofs.erase(mod.getOperation());mod->removeAttr(Witness);
  SmallVector<WarpSpecializeOp> xs;mod.walk([&](WarpSpecializeOp op){xs.push_back(op);});
  auto reject=[&](StringRef s)->LogicalResult{return mod.emitError("raw-window-ws: ")<<s;};
  const StringRef optOutKeys[]={DisableWaveSpecialization,DisableLgkmCleanup};
  for(StringRef key:optOutKeys)
    if(auto attr=mod->getAttr(key);attr&&!isa<UnitAttr>(attr))return reject("optimization opt-out request must be UnitAttr");
  llvm::errs()<<"[rawws-v15] early-optouts-checked\n";
  auto base=mod->getAttrOfType<IntegerAttr>("ttg.num-warps"),ctas=mod->getAttrOfType<IntegerAttr>("ttg.num-ctas"),wave=mod->getAttrOfType<IntegerAttr>("ttg.threads-per-warp");
  if(wdra||!base||(base.getInt()!=4&&base.getInt()!=8)||!ctas||ctas.getInt()!=1||!wave||wave.getInt()!=64||xs.size()!=1)return reject("requires one wave64 CTA, one WS, base4/8, and no WDRA");
  auto ws=xs.front();
  unsigned entries=0;triton::FuncOp entry;
  for(auto f:mod.getOps<triton::FuncOp>())if(f.isPublic()&&!f.getBody().empty()){++entries;entry=f;}
  if(entries!=1||ws->getParentOfType<triton::FuncOp>()!=entry)return reject("requires one public entry containing the WS");
  if(ws.getNumResults()||ws.getPartitionRegions().size()!=1||ws.getPartitionNumWarps().size()!=1||ws.getPartitionNumWarps()[0]!=4||!llvm::hasSingleElement(ws.getDefaultRegion())||!llvm::hasSingleElement(*ws.getPartitionRegions()[0]))return reject("requires void compute region and one four-wave producer");
  if(mod->hasAttr(Staggered)) {
    if(!isa<UnitAttr>(mod->getAttr(Staggered)))return reject("staggered request must be UnitAttr");
    llvm::errs()<<"[rawws-v15] early-staggered-call begin base="<<base.getInt()<<"\n";
    LogicalResult staggeredResult=staggeredEarly(mod,ws,unsigned(base.getInt()));
    llvm::errs()<<"[rawws-v15] early-staggered-call end failed="<<failed(staggeredResult)<<"\n";
    return staggeredResult;
  }
  Trace p{true},c{false};Env env;
  if(!p.block(ws.getPartitionRegions()[0]->front(),env))return reject(p.why);
  if(!c.block(ws.getDefaultRegion().front(),env))return reject(c.why);
  size_t pi=0,ci=0;unsigned phases=0;std::optional<Window> previous;
  while(pi<p.events.size()) {
    if(p.events.size()-pi<4)return reject("incomplete producer phase");
    auto copy=p.events[pi],a=p.events[pi+1],wait=p.events[pi+2],b=p.events[pi+3];
    if(copy.kind!=Kind::Copy||a.kind!=Kind::Barrier||wait.kind!=Kind::Wait||b.kind!=Kind::Barrier||wait.op->getOperand(0).getDefiningOp()!=copy.op)return reject("producer must copy,A,matching VM0,B");
    if(previous&&previous->owner!=copy.view->owner)return reject("multiple physical owners are unsupported");
    if(previous&&*previous==*copy.view)return reject("producer may not overwrite immediately preceding compute frame before its A barrier");
    if(ci+2>c.events.size()||c.events[ci].kind!=Kind::Barrier||c.events[ci+1].kind!=Kind::Barrier)return reject("compute must arrive at A and B before reading");
    ci+=2;llvm::BitVector covered(16384,false);unsigned reads=0;
    while(ci<c.events.size()&&c.events[ci].kind==Kind::Read) {
      auto read=c.events[ci++];
      if(!(*read.view==*copy.view)||!coverRead(read.op,covered,copy.op->getAttr("storageShape"),copy.op->getAttr("basis")))return reject("compute source chart/window does not match published producer phase");
      ++reads;
    }
    if(!reads||!covered.all())return reject("compute must cover the complete published logical window before next phase");
    previous=copy.view;pi+=4;++phases;
  }
  if(!phases||ci!=c.events.size())return reject("unequal or empty producer/consumer phase traces");
  Proof proof;proof.base=base.getInt();
  llvm::SmallPtrSet<Operation*,16> producerSites,consumerSites;
  for(const Event& e:p.events)if(e.kind==Kind::Barrier)producerSites.insert(e.op);
  for(const Event& e:c.events)if(e.kind==Kind::Barrier)consumerSites.insert(e.op);
  proof.producerSites=producerSites.size();proof.consumerSites=consumerSites.size();
  if(!proof.producerSites||proof.producerSites!=proof.consumerSites)return reject("unequal static CTA sites");
  proof.token=DistinctAttr::create(UnitAttr::get(mod.getContext()));
  mod->setAttr(Witness,proof.token);proofs[mod.getOperation()]=proof;
  Builder builder(mod.getContext());ws.setWarpGroupStartIds(ArrayRef<int32_t>{int32_t(proof.base)});
  mod->setAttr("ttg.total-num-warps",builder.getI32IntegerAttr(proof.base+4));
  mod->setAttr("ttg.hcu.raw_window_ws.phases",builder.getI32IntegerAttr(phases));
  return success();
}
// Late verification uses the real post-CF successor graph, not lexical CTA
// counts. All 64 lane conditions must agree at each branch; readfirstlane is
// evaluated in lane zero's actual scalar environment, never this lane's phi.
using LaneEnvs=SmallVector<Env,64>;
static StringRef callee(Operation* op) {
  if(isa<LLVM::CallOp>(op))
    if(auto s=op->getAttrOfType<FlatSymbolRefAttr>("callee"))return s.getValue();
  if(isa<LLVM::CallIntrinsicOp>(op))
    if(auto s=op->getAttrOfType<StringAttr>("intrin"))return s.getValue();
  return {};
}
static std::optional<int64_t> lateInteger(Value v,const LaneEnvs& envs,unsigned wave,unsigned lane,unsigned depth=0) {
  if(depth>96||!v.getType().isIntOrIndex())return {};
  auto found=envs[lane].find(v);if(found!=envs[lane].end())return found->second;
  Value root=capture(v);if(root!=v)return lateInteger(root,envs,wave,lane,depth+1);
  Operation* op=v.getDefiningOp();if(!op)return {};
  StringRef n=op->getName().getStringRef(),fn=callee(op);
  if(n=="rocdl.workitem.id.x"||fn=="llvm.amdgcn.workitem.id.x")return wave*64+lane;
  if(n=="gpu.thread_id") {
    auto dim=op->getAttrOfType<mlir::gpu::DimensionAttr>("dimension");
    if(dim&&dim.getValue()==mlir::gpu::Dimension::x)return wave*64+lane;
    return {};
  }
  if(n=="rocdl.readfirstlane"||fn=="llvm.amdgcn.readfirstlane") {
    if(op->getNumOperands()!=1)return {};
    return lateInteger(op->getOperand(0),envs,wave,0,depth+1);
  }
  if(n=="arith.constant"||n=="llvm.mlir.constant") {
    if(auto c=op->getAttrOfType<IntegerAttr>("value"))return c.getInt();return {};
  }
  if(auto cmp=dyn_cast<arith::CmpIOp>(op)) {
    auto x=lateInteger(cmp.getLhs(),envs,wave,lane,depth+1),y=lateInteger(cmp.getRhs(),envs,wave,lane,depth+1);
    unsigned width=scalarWidth(cmp.getLhs().getType());if(!x||!y||!width||width>64)return {};
    auto result=compareBits(arith::stringifyCmpIPredicate(cmp.getPredicate()),integerBitPattern(width,*x),integerBitPattern(width,*y));
    if(result)return int64_t(*result);return {};
  }
  if(auto cmp=dyn_cast<LLVM::ICmpOp>(op)) {
    auto x=lateInteger(cmp.getLhs(),envs,wave,lane,depth+1),y=lateInteger(cmp.getRhs(),envs,wave,lane,depth+1);
    unsigned width=scalarWidth(cmp.getLhs().getType());if(!x||!y||!width||width>64)return {};
    const llvm::APInt a=integerBitPattern(width,*x),b=integerBitPattern(width,*y);
    bool result=false;
    switch(cmp.getPredicate()) {
    case LLVM::ICmpPredicate::eq:result=a==b;break;case LLVM::ICmpPredicate::ne:result=a!=b;break;
    case LLVM::ICmpPredicate::slt:result=a.slt(b);break;case LLVM::ICmpPredicate::sle:result=a.sle(b);break;
    case LLVM::ICmpPredicate::sgt:result=a.sgt(b);break;case LLVM::ICmpPredicate::sge:result=a.sge(b);break;
    case LLVM::ICmpPredicate::ult:result=a.ult(b);break;case LLVM::ICmpPredicate::ule:result=a.ule(b);break;
    case LLVM::ICmpPredicate::ugt:result=a.ugt(b);break;case LLVM::ICmpPredicate::uge:result=a.uge(b);break;
    default:return {};
    }return int64_t(result);
  }
  if(auto sel=dyn_cast<LLVM::SelectOp>(op)) {
    auto condition=lateInteger(sel.getCondition(),envs,wave,lane,depth+1);if(!condition)return {};
    return lateInteger(*condition?sel.getTrueValue():sel.getFalseValue(),envs,wave,lane,depth+1);
  }
  if(op->getNumOperands()==1) {
    auto x=lateInteger(op->getOperand(0),envs,wave,lane,depth+1);
    unsigned sw=scalarWidth(op->getOperand(0).getType()),dw=scalarWidth(v.getType());
    if(!x||!sw||!dw||sw>64||dw>64)return {};
    llvm::APInt a=integerBitPattern(sw,*x);
    if(n=="llvm.zext"||n=="arith.extui")return a.zextOrTrunc(dw).getSExtValue();
    if(n=="llvm.sext"||n=="llvm.trunc"||n=="arith.extsi"||n=="arith.trunci"||n=="arith.index_cast"||n=="builtin.unrealized_conversion_cast")return a.sextOrTrunc(dw).getSExtValue();
    if(n=="llvm.bitcast"&&sw==dw)return *x;
    return {};
  }
  if(op->getNumOperands()!=2)return {};
  auto x=lateInteger(op->getOperand(0),envs,wave,lane,depth+1),y=lateInteger(op->getOperand(1),envs,wave,lane,depth+1);
  unsigned width=scalarWidth(v.getType());if(!x||!y||!width||width>64)return {};
  llvm::APInt a=integerBitPattern(width,*x),b=integerBitPattern(width,*y),z(width,0);
  if(n=="llvm.add"||n=="arith.addi")z=a+b;else if(n=="llvm.sub"||n=="arith.subi")z=a-b;else if(n=="llvm.mul"||n=="arith.muli")z=a*b;
  else if(n=="llvm.and")z=a&b;else if(n=="llvm.or")z=a|b;else if(n=="llvm.xor")z=a^b;
  else if(n=="llvm.lshr"||n=="llvm.ashr"||n=="llvm.shl") {
    if(b.uge(width))return {};unsigned shift=b.getZExtValue();
    z=n=="llvm.lshr"?a.lshr(shift):n=="llvm.ashr"?a.ashr(shift):a.shl(shift);
  } else if(n=="llvm.udiv"||n=="llvm.urem") {
    if(b.isZero())return {};z=n=="llvm.udiv"?a.udiv(b):a.urem(b);
  } else if(n=="llvm.sdiv"||n=="llvm.srem") {
    if(b.isZero()||(a.isMinSignedValue()&&b.isAllOnes()))return {};
    z=n=="llvm.sdiv"?a.sdiv(b):a.srem(b);
  } else return {};
  return z.getSExtValue();
}
// Exact compiler-generated scalar clamp. No general inline-asm purity claim.
static bool falseOrAbsentFlag(Operation* op,StringRef name) {
  Attribute a=op->getAttr(name);if(!a)return true;
  auto b=dyn_cast<BoolAttr>(a);return b&&!b.getValue();
}
static bool knownScalarClamp(Operation* op) {
  auto text=op->getAttrOfType<StringAttr>("asm_string"),constraints=op->getAttrOfType<StringAttr>("constraints");
  if(!text||!constraints||text.getValue()!=
      "s_sub_u32 $0, $1, $2\n\ts_cmp_ge_u32 $1, $2\n\ts_cselect_b32 $0, $0, 0"||
      constraints.getValue()!="=&s,s,s,~{scc}"||op->getNumOperands()!=2||op->getNumResults()!=1||
      !op->getResult(0).getType().isInteger(32)||!op->getOperand(0).getType().isInteger(32)||!op->getOperand(1).getType().isInteger(32)||
      !falseOrAbsentFlag(op,"has_side_effects")||!falseOrAbsentFlag(op,"is_align_stack"))return false;
  auto attrs=op->getAttrOfType<ArrayAttr>("operand_attrs");
  return !op->hasAttr("operand_attrs")||(attrs&&attrs.empty());
}
static bool knownResourceConstructor(Operation* op) {
  if((!isa<ROCDL::MakeBufferRsrcOp>(op)&&callee(op)!="llvm.amdgcn.make.buffer.rsrc.p8.p1")||op->getNumOperands()!=4||op->getNumResults()!=1)return false;
  auto input=dyn_cast<LLVM::LLVMPointerType>(op->getOperand(0).getType());
  auto output=dyn_cast<LLVM::LLVMPointerType>(op->getResult(0).getType());
  return input&&input.getAddressSpace()==1&&output&&output.getAddressSpace()==8&&
      op->getOperand(1).getType().isInteger(16)&&op->getOperand(2).getType().isInteger(32)&&op->getOperand(3).getType().isInteger(32);
}
// Only the actual BF16 epilogue classification intrinsic, not arbitrary
// CallIntrinsicOp. It is register-only and cannot complete any queue request.
static bool knownFPClass(Operation* op) {
  if(!isa<LLVM::CallIntrinsicOp>(op)||callee(op)!="llvm.is.fpclass"||op->getNumOperands()!=2||op->getNumResults()!=1||
      !op->getOperand(0).getType().isF32()||!op->getOperand(1).getType().isInteger(32)||!op->getResult(0).getType().isInteger(1))return false;
  auto constant=op->getOperand(1).getDefiningOp<LLVM::ConstantOp>();
  auto mask=constant?dyn_cast<IntegerAttr>(constant.getValue()):IntegerAttr{};
  return mask&&mask.getType().isInteger(32)&&mask.getInt()>=0&&mask.getInt()<=0x3ff;
}
static bool knownGlobalI16Store(Operation* op) {
  if(!isa<ROCDL::RawPtrBufferStoreOp>(op)||op->getNumOperands()!=5||op->getNumResults())return false;
  auto resource=dyn_cast<LLVM::LLVMPointerType>(op->getOperand(1).getType());
  return op->getOperand(0).getType().isInteger(16)&&resource&&resource.getAddressSpace()==8&&
      op->getOperand(2).getType().isInteger(32)&&op->getOperand(3).getType().isInteger(32)&&op->getOperand(4).getType().isInteger(32);
}
static bool knownWorkgroupQuery(Operation* op) {
  StringRef n=op->getName().getStringRef();
  return (n=="rocdl.workgroup.id.x"||n=="rocdl.workgroup.id.y")&&op->getNumOperands()==0&&op->getNumResults()==1&&op->getResult(0).getType().isInteger(32);
}
// Exact provenance call emitted by frozen LocalBarrierOpConversion. It is
// not a wait or an additional CTA. A real adjacent barrier was already modeled.
static bool knownLocalBarrierMarker(Operation* op) {
  if(callee(op)!="__triton_hcu_local_barrier_marker"||op->getNumOperands()||op->getNumResults())return false;
  Operation* before=op->getPrevNode();
  if(!before||!isa<ROCDL::SBarrierOp,ROCDL::BarrierOp>(before))return false;
  auto mod=op->getParentOfType<ModuleOp>();
  auto declaration=mod.lookupSymbol<LLVM::LLVMFuncOp>("__triton_hcu_local_barrier_marker");
  if(!declaration||!declaration.getBody().empty())return false;
  auto type=declaration.getFunctionType();
  return type.getNumParams()==0&&!type.isVarArg()&&isa<LLVM::LLVMVoidType>(type.getReturnType());
}
// Exact global register load family. This never grants shared DMA/wait
// credit. Descriptor/coordinate legality belongs to the unchanged MLS proof.
static bool knownRawRegisterGPRLoad(Operation* op) {
  if(!isa<LLVM::CallOp>(op)||op->getNumOperands()!=4||op->getNumResults()!=1)
    return false;
  StringRef fn=callee(op);Type result=op->getResult(0).getType();
  bool payload=(fn=="llvm.amdgcn.raw.ptr.buffer.load.i16"&&result.isInteger(16))||
    (fn=="llvm.amdgcn.raw.ptr.buffer.load.i32"&&result.isInteger(32));
  if(auto v=dyn_cast<VectorType>(result))
    payload=v.getRank()==1&&v.getElementType().isInteger(32)&&
      ((fn=="llvm.amdgcn.raw.ptr.buffer.load.v2i32"&&v.getNumElements()==2)||
       (fn=="llvm.amdgcn.raw.ptr.buffer.load.v4i32"&&v.getNumElements()==4));
  auto ptr=dyn_cast<LLVM::LLVMPointerType>(op->getOperand(0).getType());
  if(!payload||!ptr||ptr.getAddressSpace()!=8)return false;
  for(unsigned i=1;i<4;++i)if(!op->getOperand(i).getType().isInteger(32))return false;
  Operation* aux=op->getOperand(3).getDefiningOp();if(!aux)return false;
  if(aux->getName().getStringRef()!="llvm.mlir.constant"&&
     aux->getName().getStringRef()!="arith.constant")return false;
  auto attr=aux->getAttrOfType<IntegerAttr>("value");
  return attr&&attr.getType().isInteger(32)&&attr.getInt()==0;
}
struct LateTrace {
  unsigned wave;const SmallVector<Segment>& expected;std::string why;
  LaneEnvs envs=LaneEnvs(64);SmallVector<Segment> segments{Segment{}};
  SmallVector<Operation*> branches;bool pendingRead=false,pendingCopy=false;
  unsigned budget=0;bool consumer=false;
  LateTrace(unsigned w,const SmallVector<Segment>& e,bool c=false):wave(w),expected(e),consumer(c){}
  bool reject(StringRef reason){why=reason.str();return false;}
  bool finishStoreOnlySuffix(Operation* branch,Region& region) {
    // This discharges no pending request and gives no DMA publication credit.
    return consumer&&!pendingRead&&!pendingCopy&&segments.size()==expected.size()&&
        lateStoreOnlySuffix(branch,region);
  }
  // gfx936 uses the major9 encoding shared by MemoryCounterWait lowering:
  // VM=[15:14,3:0], LGKM=[11:8]. Both representations use this one transition.
  bool applyWaitMask(int64_t mask) {
    if(mask<0||mask>65535)return reject("wait mask outside the gfx936 16-bit encoding");
    if((mask&0x0f00)==0)pendingRead=false;
    if((mask&0xc00f)==0&&pendingCopy) {pendingCopy=false;segments.back().wait=true;++segments.back().waits;}
    return true;
  }
  bool transfer(Block* dest,ValueRange operands,Block*& current) {
    if(!dest||operands.size()!=dest->getNumArguments())return reject("invalid direct branch argument list");
    LaneEnvs next=envs;
    for(unsigned lane=0;lane<64;++lane)for(auto pair:llvm::zip(dest->getArguments(),operands)) {
      auto x=lateInteger(std::get<1>(pair),envs,wave,lane);
      if(x)next[lane][std::get<0>(pair)]=*x;else next[lane].erase(std::get<0>(pair));
    }
    envs=std::move(next);current=dest;return true;
  }
  std::optional<bool> condition(Value v) {
    std::optional<bool> result;
    for(unsigned lane=0;lane<64;++lane) {
      auto x=lateInteger(v,envs,wave,lane);if(!x)return {};
      bool b=*x!=0;if(result&&*result!=b)return {};result=b;
    }return result;
  }
  bool run(Region& region) {
    if(region.empty())return reject("empty late actor");Block* current=&region.front();
    while(current) {
      if(current->getParent()!=&region)return reject("side entry/exit in late actor CFG");
      bool terminated=false;
      for(Operation& op:*current) {
        if(++budget>4194304)return reject("late finite CFG budget exceeded");
        StringRef n=op.getName().getStringRef(),fn=callee(&op);
        if(isa<WarpYieldOp,WarpReturnOp>(&op)) {
          if(pendingRead||pendingCopy)return reject("pending shared read/DMA at actor exit");
          current=nullptr;terminated=true;break;
        }
        if(auto br=dyn_cast<LLVM::BrOp>(&op)) {
          if(!transfer(br.getDest(),br.getDestOperands(),current))return false;
          terminated=true;break;
        }
        if(auto br=dyn_cast<LLVM::CondBrOp>(&op)) {
          auto choice=condition(br.getCondition());
          if(!choice) {
            if(!finishStoreOnlySuffix(&op,region))return reject("unknown/lane-dependent late branch outside a proved empty-queue store-only suffix");
            branches.push_back(&op);current=nullptr;terminated=true;break;
          }
          branches.push_back(&op);
          if(!transfer(*choice?br.getTrueDest():br.getFalseDest(),*choice?br.getTrueDestOperands():br.getFalseDestOperands(),current))return false;
          terminated=true;break;
        }
        if(auto br=dyn_cast<cf::BranchOp>(&op)) {
          if(!transfer(br.getDest(),br.getDestOperands(),current))return false;
          terminated=true;break;
        }
        if(auto br=dyn_cast<cf::CondBranchOp>(&op)) {
          auto choice=condition(br.getCondition());
          if(!choice) {
            if(!finishStoreOnlySuffix(&op,region))return reject("unknown/lane-dependent late CF branch outside a proved empty-queue store-only suffix");
            branches.push_back(&op);current=nullptr;terminated=true;break;
          }
          branches.push_back(&op);
          if(!transfer(*choice?br.getTrueDest():br.getFalseDest(),*choice?br.getTrueDestOperands():br.getFalseDestOperands(),current))return false;
          terminated=true;break;
        }
        if(op.getNumRegions()||op.getNumSuccessors()||op.hasTrait<OpTrait::IsTerminator>())return reject("unknown late control/region");
        if(isa<LLVM::CallIntrinsicOp>(&op)) {
          if(!knownFPClass(&op)) {why="unknown/malformed late CallIntrinsicOp: "+fn.str();return false;}
          continue;
        }
        if(isa<ROCDL::MakeBufferRsrcOp>(&op)) {
          if(!knownResourceConstructor(&op))return reject("malformed late ROCDL make.buffer.rsrc signature");
          continue;
        }
        if(isa<ROCDL::RawPtrBufferStoreOp>(&op)) {
          if(!knownGlobalI16Store(&op))return reject("malformed late ROCDL raw.ptr.buffer.store signature");
          continue; // Global VM store, no LDS event or DMA publication credit.
        }
        if(n=="rocdl.workgroup.id.x"||n=="rocdl.workgroup.id.y") {
          if(!knownWorkgroupQuery(&op))return reject("malformed late ROCDL workgroup-id signature");
          continue; // CTA-uniform but not a known constant in branch evaluation.
        }
        if(isa<ROCDL::SBarrierOp,ROCDL::BarrierOp>(&op)) {
          // Every shared read must have a real full wait before its publication
          // cut. DMA may remain pending across A but must match the early trace.
          if(pendingRead)return reject("late CTA reached with unretired LDS reads");
          segments.emplace_back();if(segments.size()>expected.size())return reject("extra late dynamic CTA");continue;
        }
        if(fn=="llvm.hcu.raw.buffer.load.lds") {
          if(pendingRead)return reject("late copy before pending local reads retired");
          segments.back().copy=true;++segments.back().copies;pendingCopy=true;continue;
        }
        if(fn=="llvm.amdgcn.s.waitcnt") {
          if(op.getNumOperands()!=1)return reject("unknown wait signature");
          auto mask=lateInteger(op.getOperand(0),envs,wave,0);if(!mask)return reject("unknown wait count");
          if(!applyWaitMask(*mask))return false;
          continue;
        }
        if(auto wait=dyn_cast<ROCDL::SWaitcntOp>(&op)) {
          auto mask=wait.getBitfieldAttr();
          if(op.getNumOperands()||op.getNumResults()||!mask||!mask.getType().isInteger(32))return reject("malformed rocdl.s.waitcnt bitfield signature");
          if(!applyWaitMask(mask.getInt()))return false;
          continue;
        }
        if(auto load=dyn_cast<LLVM::LoadOp>(&op)) {
          auto ptr=dyn_cast<LLVM::LLVMPointerType>(load.getAddr().getType());
          if(!ptr||ptr.getAddressSpace()!=3)return reject("unmodelled late ordinary load");
          auto type=dyn_cast<VectorType>(load.getType());
          if(!type||type.getRank()!=1||!(type.getElementType().isInteger(16)||type.getElementType().isF16()||type.getElementType().isBF16()))return reject("unexpected late LDS vector type");
          segments.back().readWidths.push_back(type.getNumElements());
          segments.back().read=true;pendingRead=true;continue;
        }
        if(auto store=dyn_cast<LLVM::StoreOp>(&op)) {
          auto ptr=dyn_cast<LLVM::LLVMPointerType>(store.getAddr().getType());
          if(!ptr||ptr.getAddressSpace()!=1)return reject("unmodelled late store/AS3 write");continue;
        }
        if(n=="llvm.inline_asm") {
          auto text=op.getAttrOfType<StringAttr>("asm_string");
          if(knownScalarClamp(&op))continue;
          if(!text||(text.getValue()!=""&&text.getValue()!="s_setprio 0"&&text.getValue()!="s_setprio 1")) {
            why="unknown late inline asm: text="+(text?text.getValue().str():"<missing>");
            if(auto c=op.getAttrOfType<StringAttr>("constraints"))why+=" constraints="+c.getValue().str();
            return false;
          }
          continue;
        }
        // Known register-only target operations do not create LDS queue events.
        if(n=="rocdl.readfirstlane"||n=="rocdl.workitem.id.x"||n=="gpu.thread_id"||
           (n.starts_with("rocdl.")&&(n.contains("mmac")||n.contains("mfma"))))continue;
        if(!fn.empty()) {
          if(fn=="__triton_hcu_local_barrier_marker") {
            if(!knownLocalBarrierMarker(&op))return reject("malformed/misplaced late local-barrier provenance marker");
            continue; // No queue transition, no projection event, no CTA count.
          }
          if(fn=="llvm.amdgcn.make.buffer.rsrc.p8.p1") {
            if(!knownResourceConstructor(&op))return reject("malformed late make.buffer.rsrc.p8.p1 signature");
            continue;
          }
          if(fn.starts_with("llvm.amdgcn.raw.ptr.buffer.load.")) {
            if(!knownRawRegisterGPRLoad(&op))return reject("malformed raw GPR load signature/aux");
            continue; // VM register read only; no shared publication credit.
          }
          if(fn.starts_with("llvm.amdgcn.struct.ptr.buffer.load.")&&!fn.contains("lds"))continue;
          if(fn.starts_with("llvm.amdgcn.raw.ptr.buffer.store.")||fn.starts_with("llvm.amdgcn.raw.buffer.store."))continue;
          if(fn=="llvm.amdgcn.readfirstlane"||fn=="llvm.amdgcn.workitem.id.x"||fn.starts_with("llvm.hcu.mmac")||fn.starts_with("llvm.amdgcn.mfma"))continue;
          why="unknown late call/intrinsic: "+fn.str();return false;
        }
        if(n.contains("barrier")||n.contains("wait")||n.contains("async")||n.contains("call")) {why="unknown late memory/sync event: "+n.str();return false;}
        if(!isPure(&op)) {why="unknown late effect operation: "+n.str();return false;}
      }
      if(!terminated)return reject("late block without supported terminator");
    }
    if(segments.size()!=expected.size())return reject("missing late dynamic CTA");
    for(unsigned i=0;i<segments.size();++i) {
      if(segments[i].read!=expected[i].read||segments[i].copy!=expected[i].copy||segments[i].wait!=expected[i].wait||segments[i].copies!=expected[i].copies||segments[i].waits!=expected[i].waits||segments[i].readWidths!=expected[i].readWidths)return reject("late ordinal memory/wait projection mismatch");
    }
    return true;
  }
};
#include "RawWindowWaveSpecialize.inc"
#include "RawWindowLgkmEmpty.inc"

static bool verifyStaggeredLate(WarpSpecializeOp ws,const Proof& proof) {
  if(proof.segments.size()!=proof.base+4)return false;
  for(unsigned wave=0;wave<proof.base+4;++wave) {
    LateTrace trace{wave,proof.segments[wave],wave<proof.base};
    Region& region=wave<proof.base?ws.getDefaultRegion():*ws.getPartitionRegions()[0];
    if(!trace.run(region)) {ws.emitError("raw-window-ws late actor ")<<wave<<": "<<trace.why;return false;}
  }
  return true;
}
static bool uniformCapture(LLVM::LLVMFuncOp f,Value v,llvm::SmallPtrSetImpl<void*>&seen) {
  if(!seen.insert(v.getAsOpaquePointer()).second)return true;
  if(auto a=dyn_cast<BlockArgument>(v))return a.getOwner()==&f.getBody().front();
  auto op=v.getDefiningOp();if(!op||!isPure(op))return false;
  StringRef n=op->getName().getStringRef();
  if(!(n.starts_with("llvm.")||n=="builtin.unrealized_conversion_cast")||n.contains("call")||n.contains("load")||n.contains("intrinsic"))return false;
  return llvm::all_of(op->getOperands(),[&](Value x){return uniformCapture(f,x,seen);});
}
static LogicalResult late(LLVM::LLVMFuncOp func) {
  SmallVector<WarpSpecializeOp> ops;func.walk([&](WarpSpecializeOp ws){ops.push_back(ws);});
  if(ops.empty())return success(); // public intrinsic declarations/helpers.
  auto mod=func->getParentOfType<ModuleOp>();auto found=proofs.find(mod.getOperation());
  const StringRef optOutKeys[]={DisableWaveSpecialization,DisableLgkmCleanup};
  for(StringRef key:optOutKeys)
    if(auto attr=mod->getAttr(key);attr&&!isa<UnitAttr>(attr))return func.emitError("raw-window-ws: optimization opt-out request must be UnitAttr");
  if(found==proofs.end()||mod->getAttr(Witness)!=found->second.token)return func.emitError("raw-window-ws: missing same-process fresh pre-CF proof");
  Proof proof=found->second;proofs.erase(found);mod->removeAttr(Witness);
  if(ops.size()!=1)return func.emitError("raw-window-ws: one post-CF WS required");
  auto ws=ops.front();
  if(ws.getNumResults()||ws.getPartitionRegions().size()!=1)return ws.emitError("raw-window-ws: invalid post-CF partition");
  unsigned counts[2]={0,0};unsigned i=0;
  for(Region* region:{&ws.getDefaultRegion(),ws.getPartitionRegions()[0]}) {
    bool unknown=false;
    region->walk([&](Operation* op){
      counts[i]+=isa<ROCDL::SBarrierOp,ROCDL::BarrierOp>(op);
      StringRef n=op->getName().getStringRef();unknown|=n.contains("abarrier")||n.contains("ebarrier");
    });
    if(unknown)return ws.emitError("raw-window-ws: unsupported late synchronization");++i;
  }
  if(counts[0]!=proof.consumerSites||counts[1]!=proof.producerSites)return ws.emitError("raw-window-ws: late CTA sites differ from fresh early trace");
  if(proof.staggered) {
    Builder diagnostic(ws.getContext());
    if(!mod->hasAttr(DisableWaveSpecialization))specializeWaveControl(ws,proof);
    else {
      mod->setAttr("ttg.hcu.raw_window_ws.wave_classes",diagnostic.getI32IntegerAttr(0));
      mod->setAttr("ttg.hcu.raw_window_ws.wave_source_sites",diagnostic.getI32IntegerAttr(0));
      mod->setAttr("ttg.hcu.raw_window_ws.wave_folded_sites",diagnostic.getI32IntegerAttr(0));
      mod->setAttr("ttg.hcu.raw_window_ws.wave_specialization",diagnostic.getStringAttr("explicitly-disabled"));
    }
    if(!mod->hasAttr(DisableLgkmCleanup))cleanupMustEmptyLgkm(ws);
    else {
      mod->setAttr("ttg.hcu.raw_window_ws.lgkm_empty_deleted",diagnostic.getI32IntegerAttr(0));
      mod->setAttr("ttg.hcu.raw_window_ws.lgkm_empty_cleanup",diagnostic.getStringAttr("explicitly-disabled"));
    }
  }
  if(proof.staggered&&!verifyStaggeredLate(ws,proof))return ws.emitError("raw-window-ws: actual per-wave CF/barrier/read-completion projection differs from fresh proof");
  for(Value capture:ws.getExplicitCaptures()) {
    llvm::SmallPtrSet<void*,32> seen;if(!uniformCapture(func,capture,seen))return ws.emitError("raw-window-ws: nonuniform or non-rematerializable LLVM capture");
  }
  elideTrivialCaptures(func,ops);
  if(ws.getNumOperands()||ws.getPartitionRegions()[0]->getNumArguments())return ws.emitError("raw-window-ws: capture elision incomplete");
  auto& blocks=func.getBody().getBlocks();Block* before=ws->getBlock();
  TritonLLVMIRRewriter b(ws.getLoc(),OpBuilder::atBlockEnd(before));
  Block* after=b.splitBlock(before,ws->getIterator());
  Block* compute=&ws.getDefaultRegion().front();Block* producer=&ws.getPartitionRegions()[0]->front();
  for(Region* region:{&ws.getDefaultRegion(),ws.getPartitionRegions()[0]})region->walk([&](Operation* op){
    if(isa<WarpYieldOp,WarpReturnOp>(op)){TritonLLVMIRRewriter term(op->getLoc(),op);term.replaceOpWithNewOp<LLVM::BrOp>(op,ValueRange{},after);}
  });
  b.setInsertionPointToEnd(before);Value tid=b.create<mlir::gpu::ThreadIdOp>(mlir::gpu::Dimension::x);
  tid=b.create<arith::IndexCastOp>(b.getI32Type(),tid);
  b.create<LLVM::CondBrOp>(b.icmp_ult(tid,b.i32_val(proof.base*64)),compute,producer);
  blocks.splice(after->getIterator(),ws.getDefaultRegion().getBlocks());
  blocks.splice(after->getIterator(),ws.getPartitionRegions()[0]->getBlocks());ws.erase();
  b.setInsertionPointToStart(after);b.create<ROCDL::SBarrierOp>();return success();
}
} // namespace rawws
} // namespace
