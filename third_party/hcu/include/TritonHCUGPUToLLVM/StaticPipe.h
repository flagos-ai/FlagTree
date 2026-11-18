#ifndef HCU_TLE_STATIC_PIPE_H
#define HCU_TLE_STATIC_PIPE_H

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/IR/Matchers.h"
#include "tle/dialect/include/IR/Dialect.h"
#include "tle/dialect/include/IR/ExactSMEM.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "Dialect/TritonHCUGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/LinearLayoutConversions.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "llvm/ADT/SmallPtrSet.h"

namespace mlir::triton::HCU::static_pipe {
namespace ttg = mlir::triton::gpu;
namespace te = mlir::triton::tle;
inline constexpr const char *iterationAttr = "tle.hcu.static_pipe_iteration";
inline constexpr const char *preparedAttr = "tle.hcu.static_pipe_prepared";
inline constexpr const char *payloadAttr = "tle.hcu.static_pipe_payload";
inline constexpr const char *stateAttr = "tle.hcu.static_pipe_state";

inline std::optional<int64_t> constant(Value v) {
  APInt n;
  if (matchPattern(v, m_ConstantInt(&n))) return n.getBitWidth()==1 ? n.getZExtValue() : n.getSExtValue();
  if (auto cast = v.getDefiningOp<arith::ExtSIOp>()) return constant(cast.getIn());
  if (auto cast = v.getDefiningOp<arith::ExtUIOp>()) return constant(cast.getIn());
  if (auto add = v.getDefiningOp<arith::AddIOp>()) {
    auto a=constant(add.getLhs()), b=constant(add.getRhs());
    if (a && b) return *a+*b;
  }
  if (auto mul = v.getDefiningOp<arith::MulIOp>()) {
    auto a=constant(mul.getLhs()), b=constant(mul.getRhs());
    if (a && b) return *a * *b;
  }
  return std::nullopt;
}
// Evaluate only the integer expressions admitted by the public stage/phase API.
// The bounded trip count lets us check every iteration without trusting hints.
inline std::optional<int64_t> evaluate(Value v,Value iv,int64_t n) {
  if (v==iv) return n;
  if (auto c=constant(v)) return c;
  auto op=v.getDefiningOp();if (!op || op->getNumResults()!=1) return std::nullopt;
  if (isa<arith::ExtSIOp,arith::ExtUIOp,arith::TruncIOp,arith::IndexCastOp>(op))
    return evaluate(op->getOperand(0),iv,n);
  if (op->getNumOperands()!=2) return std::nullopt;
  auto a=evaluate(op->getOperand(0),iv,n),b=evaluate(op->getOperand(1),iv,n);
  if (!a || !b) return std::nullopt;
  if (isa<arith::AddIOp>(op)) return *a+*b;
  if (isa<arith::MulIOp>(op)) return *a * *b;
  if (isa<arith::SubIOp>(op)) return *a-*b;
  if (isa<arith::DivSIOp,arith::DivUIOp>(op) && *b>0 && *a>=0) return *a / *b;
  if (isa<arith::RemSIOp,arith::RemUIOp>(op) && *b>0 && *a>=0) return *a % *b;
  if (isa<arith::AndIOp>(op)) return *a & *b;
  if (isa<arith::ShLIOp>(op) && *b>=0 && *b<31) return *a << *b;
  if (isa<arith::ShRUIOp,arith::ShRSIOp>(op) && *a>=0 && *b>=0 && *b<31) return *a >> *b;
  if (auto c=dyn_cast<arith::CmpIOp>(op)) {
    if (c.getPredicate()==arith::CmpIPredicate::ne) return int64_t(*a!=*b);
    if (c.getPredicate()==arith::CmpIPredicate::eq) return int64_t(*a==*b);
  }
  return std::nullopt;
}
inline constexpr const char *countedAttr="tle.hcu.counted_pipe_verified";
inline constexpr const char *pairAttr="tle.hcu.counted_pipe_pair";
inline Value root(Value v) { return te::resolveExactSMEMRootSource(v); }
inline bool isPipe(Operation *op) { return op->getName().getStringRef().starts_with("tle.pipe."); }
inline OperandRange fields(Operation *op) {
  if (auto p=dyn_cast<te::PipeCreateOp>(op)) return p.getFields();
  if (auto p=dyn_cast<te::PipeWriterAcquireOp>(op)) return p.getFields();
  if (auto p=dyn_cast<te::PipeWriterCommitOp>(op)) return p.getFields();
  if (auto p=dyn_cast<te::PipeReaderWaitOp>(op)) return p.getFields();
  return cast<te::PipeReaderReleaseOp>(op).getFields();
}
inline Value stage(Operation *op) {
  if (auto p=dyn_cast<te::PipeWriterAcquireOp>(op)) return p.getStage();
  if (auto p=dyn_cast<te::PipeWriterCommitOp>(op)) return p.getStage();
  if (auto p=dyn_cast<te::PipeReaderWaitOp>(op)) return p.getStage();
  return cast<te::PipeReaderReleaseOp>(op).getStage();
}
inline bool multiple16(Value v) {
  if (auto n=constant(v)) return *n%16==0;
  if (auto m=v.getDefiningOp<arith::MulIOp>())
    return multiple16(m.getLhs()) || multiple16(m.getRhs());
  if (auto a=v.getDefiningOp<arith::AddIOp>())
    return multiple16(a.getLhs()) && multiple16(a.getRhs());
  return false;
}
inline bool aligned16(Value v) {
  if (auto a=dyn_cast<BlockArgument>(v)) {
    auto f=dyn_cast<triton::FuncOp>(a.getOwner()->getParentOp());
    if (!f) return false;
    auto div=f.getArgAttrOfType<IntegerAttr>(a.getArgNumber(),"tt.divisibility");
    return div && div.getInt()%16==0;
  }
  if (auto add=v.getDefiningOp<triton::AddPtrOp>())
    return aligned16(add.getPtr()) && multiple16(add.getOffset());
  return false;
}
struct Plan {
  te::PipeCreateOp create;
  ttg::WarpSpecializeOp ws;
  triton::FuncOp func;
  ttg::LocalAllocOp payload, state;
  SmallVector<ttg::LocalAllocOp> payloads;
  int payloadIndex(Value value) {
    value=root(value);
    for(auto [i,p]:llvm::enumerate(payloads))if(p.getResult()==value)return i;
    return -1;
  }
  SmallVector<Operation *> writer, reader;
  SmallVector<ttg::TMACopyOp> copies;
  SmallVector<Operation *> reads;
  unsigned rounds=0;
  bool counted=false;
  Value writerIV,readerIV;
};

// Exact two-wave coordinate proof, using the same target linear layout as LLVM.
// A row may be strided between waves; no row may be shared by both waves.
inline bool mmaParent(Attribute attr) {
  auto enc=dyn_cast<ttg::AMDMfmaEncodingAttr>(attr);
  return enc && enc.getVersion()==3 && enc.getWarpsPerCTA()==ArrayRef<unsigned>{2,1} &&
      enc.getInstrShape()==ArrayRef<unsigned>{16,16,32} && !enc.getIsTransposed() &&
      static_cast<unsigned>(enc.getMmacLayout())==2 && enc.getElementBitWidth()==32 &&
      enc.getTilesPerWarp()==ArrayRef<unsigned>{1,1};
}
inline bool coordinateCoverage(Attribute enc,bool rowOwned,SmallVectorImpl<unsigned> *ownerResult=nullptr) {
  auto ll=ttg::toLinearLayout(ArrayRef<int64_t>{64,64},enc);
  auto ctx=enc.getContext(); auto name=[&](StringRef n){return StringAttr::get(ctx,n);};
  auto size=[&](StringRef n) {auto it=ll.getBases().find(name(n));return it==ll.getBases().end()?1:int(1u<<it->second.size());};
  if (size("lane")!=64 || size("warp")!=2 || size("block")!=1 || size("register")>128) return false;
  SmallVector<unsigned> seen(4096,0),owners(64,0);
  for (unsigned w=0;w<2;++w) {
    SmallVector<unsigned> perWave(4096,0);
    for (int lane=0;lane<64;++lane) for (int reg=0;reg<size("register");++reg) {
      SmallVector<std::pair<StringAttr,int32_t>> in;
      for (auto dim:ll.getInDimNames()) {
        int v=dim==name("lane")?lane:dim==name("warp")?int(w):dim==name("register")?reg:0;
        in.push_back({dim,v});
      }
      auto out=ll.apply(in);int row=-1,col=-1;
      for (auto [dim,v]:out) {if(dim==name("dim0"))row=v;if(dim==name("dim1"))col=v;}
      if(row<0 || row>=64 || col<0 || col>=64)return false;
      ++seen[row*64+col];++perWave[row*64+col];owners[row]|=1u<<w;
    }
    if (!rowOwned && llvm::any_of(perWave,[](unsigned n){return n==0;}))return false;
  }
  if (!rowOwned) return true; // Dot B may replicate, but each wave sees every element.
  if (ownerResult) ownerResult->assign(owners.begin(),owners.end());
  return llvm::all_of(seen,[](unsigned n){return n==1;}) &&
      llvm::all_of(owners,[](unsigned n){return n==1 || n==2;}) &&
      llvm::count(owners,1)==32 && llvm::count(owners,2)==32;
}
inline bool rowEncoding(Attribute attr) {
  if (mmaParent(attr)) return coordinateCoverage(attr,true);
  auto enc=dyn_cast<ttg::BlockedEncodingAttr>(attr);
  bool known=enc && (((enc.getSizePerThread()==ArrayRef<unsigned>{32,1} &&
      enc.getThreadsPerWarp()==ArrayRef<unsigned>{1,64}) ||
      ((enc.getSizePerThread()==ArrayRef<unsigned>{8,4} || enc.getSizePerThread()==ArrayRef<unsigned>{4,4}) &&
      enc.getThreadsPerWarp()==ArrayRef<unsigned>{4,16}))) &&
      enc.getWarpsPerCTA()==ArrayRef<unsigned>{2,1} && enc.getOrder()==ArrayRef<unsigned>{1,0};
  return known && coordinateCoverage(attr,true);
}
inline bool bytePayloadType(Type type) {return type.isInteger(8) || isa<Float8E4M3FNType>(type);}
inline bool dotAEncoding(Attribute attr) {
  auto enc=dyn_cast<ttg::DotOperandEncodingAttr>(attr);
  return enc && enc.getOpIdx()==0 && enc.getKWidth()==8 && mmaParent(enc.getParent()) && coordinateCoverage(attr,true);
}
inline bool dotBEncoding(Attribute attr) {
  auto enc=dyn_cast<ttg::DotOperandEncodingAttr>(attr);
  return enc && enc.getOpIdx()==1 && enc.getKWidth()==8 && mmaParent(enc.getParent()) && coordinateCoverage(attr,false);
}
// Replace an exact load -> transpose -> staging allocation -> dot load with a
// transposed view of the original held pipe slot. The full lifecycle is proved
// again below, after rewriting. No new load or alias escapes the held interval.
inline LogicalResult foldPipeDotStaging(ModuleOp mod) {
  if (!mod->hasAttr("tle.hcu.software_ws_2x2")) return success();
  SmallVector<ttg::LocalAllocOp> allocations;
  mod.walk([&](ttg::LocalAllocOp a){allocations.push_back(a);});
  for (auto alloc:allocations) {
    if (alloc->getNumOperands()!=1 || !alloc.getResult().hasOneUse())continue;
    auto dotLoad=dyn_cast<ttg::LocalLoadOp>(*alloc.getResult().getUsers().begin());
    auto sourceType=dyn_cast<RankedTensorType>(alloc->getOperand(0).getType());
    if (dotLoad && sourceType && sourceType.getShape()==ArrayRef<int64_t>{64,64} &&
        bytePayloadType(sourceType.getElementType()) && rowEncoding(sourceType.getEncoding()) &&
        dotAEncoding(dotLoad.getType().getEncoding()) && !dotLoad.getToken()) {
      OpBuilder b(dotLoad);auto converted=ttg::ConvertLayoutOp::create(b,dotLoad.getLoc(),dotLoad.getType(),alloc->getOperand(0));
      dotLoad.replaceAllUsesWith(converted.getResult());dotLoad.erase();alloc.erase();continue;
    }
    auto trans=alloc->getOperand(0).getDefiningOp<triton::TransOp>();
    auto source=trans?trans.getSrc().getDefiningOp<ttg::LocalLoadOp>():alloc->getOperand(0).getDefiningOp<ttg::LocalLoadOp>();
    auto stage=source?source.getSrc().getDefiningOp<ttg::MemDescIndexOp>():ttg::MemDescIndexOp();
    auto load=dyn_cast<ttg::LocalLoadOp>(*alloc.getResult().getUsers().begin());
    auto rootAlloc=stage?root(stage.getSrc()).getDefiningOp<ttg::LocalAllocOp>():ttg::LocalAllocOp();
    if (!source || !stage || !load || !rootAlloc || !rootAlloc->hasAttr(payloadAttr) ||
        (trans && !trans.getResult().hasOneUse()) || !source.getResult().hasOneUse() || source.getToken() || load.getToken() ||
        (trans && trans.getOrder()!=ArrayRef<int32_t>{1,0}) || source.getType().getShape()!=ArrayRef<int64_t>{64,64} ||
        !bytePayloadType(source.getType().getElementType()) || !rowEncoding(source.getType().getEncoding()) ||
        !dotBEncoding(load.getType().getEncoding()))continue;
    OpBuilder b(load);Value memory=source.getSrc();
    if(trans)memory=ttg::MemDescTransOp::create(b,load.getLoc(),memory,trans.getOrder()).getResult();
    auto replacement=ttg::LocalLoadOp::create(b,load.getLoc(),load.getType(),memory,Value());
    load.replaceAllUsesWith(replacement.getResult());load.erase();alloc.erase();if(trans)trans.erase();source.erase();
  }
  return success();
}
inline bool registerStateType(Type type,bool gpuPhase) {
  auto ty=dyn_cast<RankedTensorType>(type);
  if (!ty || !ty.getElementType().isF32()) return false;
  if (ty.getShape()==ArrayRef<int64_t>{64,64}) return !gpuPhase || rowEncoding(ty.getEncoding());
  if (ty.getShape()!=ArrayRef<int64_t>{64}) return false;
  if (!gpuPhase) return true;
  auto enc=dyn_cast<ttg::SliceEncodingAttr>(ty.getEncoding());
  return enc && enc.getDim()==1 && rowEncoding(enc.getParent());
}
inline bool sameWaveStateConvert(ttg::ConvertLayoutOp op) {
  auto src=op.getSrc().getType(),dst=op.getType();
  if (src.getShape()!=dst.getShape() || src.getElementType()!=dst.getElementType())return false;
  Attribute a=src.getEncoding(),b=dst.getEncoding();
  if (src.getRank()==1) {
    if (!registerStateType(src,true) || !registerStateType(dst,true))return false;
    a=cast<ttg::SliceEncodingAttr>(a).getParent();b=cast<ttg::SliceEncodingAttr>(b).getParent();
  } else {
    if (src.getShape()!=ArrayRef<int64_t>{64,64} || (!src.getElementType().isF32() && !bytePayloadType(src.getElementType())))return false;
    if (!(rowEncoding(a) || dotAEncoding(a)) || !(rowEncoding(b) || dotAEncoding(b)))return false;
  }
  SmallVector<unsigned> oa,ob;
  return coordinateCoverage(a,true,&oa) && coordinateCoverage(b,true,&ob) && oa==ob;
}
// The fourth entry argument is an optional read-only Q tensor, captured by the
// reader. Its 64x64 footprint must be valid and disjoint from output/status,
// as with ordinary Triton global inputs; it never aliases the pipe's LDS.
inline Value globalInputRoot(Value value,unsigned depth=0) {
  if (depth>32)return Value();
  if (auto arg=dyn_cast<BlockArgument>(value)) {
    if (isa<triton::FuncOp>(arg.getOwner()->getParentOp()))return arg;
    if (auto partitions=dyn_cast<ttg::WarpSpecializePartitionsOp>(arg.getOwner()->getParentOp())) {
      auto ws=partitions->getParentOfType<ttg::WarpSpecializeOp>();
      if (ws && arg.getArgNumber()<ws->getNumOperands())return globalInputRoot(ws->getOperand(arg.getArgNumber()),depth+1);
    }
    return Value();
  }
  if (auto op=value.getDefiningOp<triton::AddPtrOp>())return globalInputRoot(op.getPtr(),depth+1);
  if (auto op=value.getDefiningOp<triton::SplatOp>())return globalInputRoot(op.getSrc(),depth+1);
  if (auto op=value.getDefiningOp<ttg::ConvertLayoutOp>())return globalInputRoot(op.getSrc(),depth+1);
  return Value();
}
inline bool qPrefixLoad(Operation *op,triton::FuncOp func,Type elementType) {
  if (func.getNumArguments()<4 || op->getNumResults()!=1)return false;
  Value ptr;
  if (auto load=dyn_cast<triton::LoadOp>(op))ptr=load.getPtr();
  else if (auto load=dyn_cast<hcugpu::BufferLoadOp>(op))ptr=load.getPtr();
  if (!ptr || globalInputRoot(ptr)!=func.getArgument(3))return false;
  auto ty=dyn_cast<RankedTensorType>(op->getResult(0).getType());
  Type pt=ptr.getType();if(auto t=dyn_cast<RankedTensorType>(pt))pt=t.getElementType();
  auto pointer=dyn_cast<triton::PointerType>(pt);
  return ty && ty.getShape()==ArrayRef<int64_t>{64,64} && ty.getElementType()==elementType &&
      pointer && pointer.getAddressSpace()==1 && pointer.getPointeeType()==elementType;
}
inline bool ownWaveReduction(Operation *op,bool gpuPhase) {
  auto reduce=dyn_cast<triton::ReduceOp>(op);
  if (!reduce || reduce.getAxis()!=1 || reduce->getNumOperands()!=1 || reduce->getNumResults()!=1 ||
      !registerStateType(reduce->getOperand(0).getType(),gpuPhase) ||
      cast<RankedTensorType>(reduce->getOperand(0).getType()).getRank()!=2 ||
      !registerStateType(reduce->getResult(0).getType(),gpuPhase) ||
      !reduce.getCombineOp().hasOneBlock()) return false;
  auto &block=reduce.getCombineOp().front();
  if (block.getNumArguments()!=2 || !block.getArgument(0).getType().isF32() ||
      !block.getArgument(1).getType().isF32() || std::distance(block.begin(),block.end())!=2) return false;
  auto &combine=block.front(); auto ret=dyn_cast<triton::ReduceReturnOp>(block.getTerminator());
  return isa<arith::AddFOp,arith::MaxNumFOp,arith::MaximumFOp>(combine) &&
      combine.getOperand(0)==block.getArgument(0) && combine.getOperand(1)==block.getArgument(1) &&
      ret && ret->getNumOperands()==1 && ret->getOperand(0)==combine.getResult(0);
}
// An epilogue may compute from carried registers, never from a post-release load.
inline bool registerEpilogue(Value value,scf::ForOp loop,bool &usesState,unsigned depth=0) {
  if (depth>128) return false;
  for (Value result:loop.getResults()) if (value==result) {usesState=true;return true;}
  Type type=value.getType();if (auto t=dyn_cast<RankedTensorType>(type)) type=t.getElementType();
  if (!isa<FloatType,IntegerType,IndexType>(type)) return false;
  auto op=value.getDefiningOp();
  if (!op || op==loop.getOperation() || loop->isProperAncestor(op) || op->getNumRegions() ||
      !isMemoryEffectFree(op) || isa<triton::CallOp>(op)) return false;
  for (Value operand:op->getOperands()) if (!registerEpilogue(operand,loop,usesState,depth+1)) return false;
  return true;
}

// Full structural proof is rerun before LLVM conversion. It does not rely on
// caller-supplied proof attributes. The first mode is deliberately straight-line.
inline FailureOr<Plan> inspect(ModuleOp mod, bool gpuPhase) {
  SmallVector<te::PipeCreateOp> creates;
  SmallVector<ttg::WarpSpecializeOp> groups;
  bool unsupported=false;
  mod.walk([&](Operation *op) {
    if (auto c=dyn_cast<te::PipeCreateOp>(op)) creates.push_back(c);
    if (auto w=dyn_cast<ttg::WarpSpecializeOp>(op)) groups.push_back(w);
    if (isPipe(op) && !isa<te::PipeCreateOp,te::PipeWriterAcquireOp,
        te::PipeWriterCommitOp,te::PipeReaderWaitOp,te::PipeReaderReleaseOp>(op)) {
      op->emitError("static HCU pipe does not support close/drop/one-shot extensions"); unsupported=true;
    }
  });
  if (unsupported) return failure();
  if (creates.size()!=1 || groups.size()!=1) {
    mod.emitError("static HCU pipe requires exactly one pipe and one 2+2 partition"); return failure();
  }
  if (gpuPhase) {
    auto target=mod->getAttrOfType<StringAttr>("ttg.target");
    auto warps=mod->getAttrOfType<IntegerAttr>("ttg.num-warps");
    if (!target || target.getValue()!="hip:gfx938" ||
        ttg::TritonGPUDialect::getNumCTAs(mod)!=1 ||
        ttg::TritonGPUDialect::getThreadsPerWarp(mod)!=64 ||
        (!warps || warps.getInt()!=2) || !mod->hasAttr("tle.hcu.software_ws_2x2")) {
      mod.emitError("static pipe requires gfx938, wave64 and fixed 2+2 CTA dispatch");return failure();
    }
  }
  Plan p; p.create=creates.front(); p.ws=groups.front();
  p.func=p.create->getParentOfType<triton::FuncOp>();
  if (!p.func || p.ws->getParentOfType<triton::FuncOp>()!=p.func ||
      !p.func.getBody().hasOneBlock() || p.create->getBlock()!=p.ws->getBlock() ||
      !p.create->isBeforeInBlock(p.ws) || !p.ws.getDefaultRegion().hasOneBlock() ||
      p.ws.getPartitionRegions().size()!=1 ||
      !p.ws.getPartitionRegions()[0]->hasOneBlock() ||
      p.ws.getPartitionNumWarps()!=ArrayRef<int32_t>{2} || p.ws.getNumResults()) {
    p.create.emitError("static HCU pipe requires a single straight-line default writer and reader worker"); return failure();
  }
  auto status=mod->getAttrOfType<IntegerAttr>("tle.hcu.static_pipe_status_arg");
  auto enabled=mod->getAttrOfType<IntegerAttr>("tle.hcu.static_pipe_enabled");
  if (!enabled || enabled.getInt()!=1 || !status || status.getInt()<0 ||
      status.getInt()>=p.func.getNumArguments()) {
    p.create.emitError("static HCU pipe requires an explicit per-CTA i32 error-status argument index"); return failure();
  }
  auto ptr=dyn_cast<triton::PointerType>(p.func.getArgument(status.getInt()).getType());
  if (!ptr || !ptr.getPointeeType().isInteger(32) || ptr.getAddressSpace()!=1 || !p.func.getArgument(status.getInt()).use_empty()) {
    p.create.emitError("static HCU pipe status argument must be an otherwise-unused i32 global pointer"); return failure();
  }
  auto readers=p.create->getAttrOfType<ArrayAttr>("readers");
  auto one=p.create->getAttrOfType<BoolAttr>("one_shot");
  unsigned fieldCount=p.create.getFields().size();
  unsigned payloadCount=fieldCount-(gpuPhase?1:0);
  if (p.create.getCapacity()!=2 || p.create.getScope()!="cta" || (payloadCount!=1 && payloadCount!=2 && payloadCount!=4) ||
      (readers && !readers.empty()) || (one && one.getValue())) {
    p.create.emitError("HCU pipe requires capacity2, one byte field or ordered INT8-K/FP8-V pairs (1 or 2 D64 fragments), default reader and cyclic CTA scope");return failure();
  }
  for(unsigned i=0;i<payloadCount;++i) {
    auto alloc=root(p.create.getFields()[i]).getDefiningOp<ttg::LocalAllocOp>();
    if(!alloc || alloc->getNumOperands() || alloc->getBlock()!=p.create->getBlock() ||
        alloc.getAlignmentOrDefault()<256 || llvm::is_contained(p.payloads,alloc)) {
      p.create.emitError("pipe payloads must be distinct fresh aligned allocations");return failure();
    }
    auto ty=alloc.getType();auto layout=dyn_cast<ttg::SwizzledSharedEncodingAttr>(ty.getEncoding());
    if(ty.getShape()!=ArrayRef<int64_t>{2,64,64} || ty.getAllocShape()!=ty.getShape() ||
        !bytePayloadType(ty.getElementType()) || !layout || layout.getVec()!=32 || layout.getPerPhase()!=2 ||
        layout.getMaxPhase()!=2 || layout.getOrder()!=ArrayRef<unsigned>{2,1,0} ||
        (payloadCount>1 && ((i<payloadCount/2 && !ty.getElementType().isInteger(8)) ||
                            (i>=payloadCount/2 && !isa<Float8E4M3FNType>(ty.getElementType()))))) {
      p.create.emitError("unsupported pipe payload type, shape or layout");return failure();
    }
    p.payloads.push_back(alloc);
  }
  p.payload=p.payloads.front();
  if (gpuPhase) {
    p.state=root(p.create.getFields().back()).getDefiningOp<ttg::LocalAllocOp>();
    if (!p.state || p.state->getNumOperands() || p.state.getAlignmentOrDefault()<256 ||
        p.state.getType().getShape()!=ArrayRef<int64_t>{2,8} ||
        !p.state.getType().getElementType().isInteger(32) ||
        p.state.getType().getAllocShape()!=ArrayRef<int64_t>{2,8} ||
        p.state->getBlock()!=p.create->getBlock() || llvm::is_contained(p.payloads,p.state)) {
      p.create.emitError("invalid private pipe epoch allocation"); return failure();
    }
  }
  if (gpuPhase) {
    auto layout=dyn_cast<ttg::SwizzledSharedEncodingAttr>(p.state.getType().getEncoding());
    if (!layout || layout.getVec()!=1 || layout.getPerPhase()!=1 || layout.getMaxPhase()!=1 ||
        layout.getOrder()!=ArrayRef<unsigned>{1,0}) {p.create.emitError("invalid epoch state layout");return failure();}
  }
  SmallVector<Operation*> writerOps,readerOps;
  unsigned writerTrip=0,readerTrip=0;
  auto collect=[&](Region &region,SmallVectorImpl<Operation*> &ops,Value &iv,unsigned &trip,bool writer) -> LogicalResult {
    scf::ForOp loop;
    for (Operation &op:region.front()) {
      if (auto f=dyn_cast<scf::ForOp>(op)) {
        if (loop || !f.getRegion().hasOneBlock() || (writer && f.getNumResults()) || f.getNumResults()>8 ||
            constant(f.getLowerBound())!=0 || constant(f.getStep())!=1)
          return op.emitError("pipe loop requires zero-based unit-step and at most eight reader FP32 register states");
        auto ub=constant(f.getUpperBound());
        if (!ub || *ub<2 || *ub>4096) return op.emitError("pipe loop trip count must be a compile-time integer in 2..4096");
        for (auto [result,initial]:llvm::zip(f.getResults(),f.getInitArgs())) {
          if (!registerStateType(result.getType(),gpuPhase))
            return f.emitError("reader state requires own-row 64-vector or 64x64 FP32 register encoding");
          auto init=initial;
          while (true) {
            if (auto layout=init.getDefiningOp<te::SetLayoutOp>()) {init=layout.getSrc();continue;}
            if (auto layout=init.getDefiningOp<ttg::ConvertLayoutOp>()) {init=layout.getSrc();continue;}
            break;
          }
          auto c=init.getDefiningOp<arith::ConstantOp>();
          auto elements=c?dyn_cast<DenseElementsAttr>(c.getValue()):DenseElementsAttr();
          if (!elements || !elements.isSplat()) return f.emitError("reader state must start at constant zero or negative infinity");
          auto v=elements.getSplatValue<APFloat>();
          if (!v.isZero() && !(v.isInfinity() && v.isNegative()))
            return f.emitError("reader state must start at constant zero or negative infinity");
        }
        loop=f;iv=f.getInductionVar();trip=*ub;continue;
      }
      if (op.getName().getDialectNamespace()=="scf") return op.emitError("unsupported nested pipe control flow");
    }
    if (loop) {
      auto enabled=mod->getAttrOfType<IntegerAttr>("tle.hcu.allow_counted_pipe");
      if (!enabled || enabled.getInt()!=1) return loop.emitError("counted pipe requires explicit compiler opt-in");
      for (Operation &op:region.front()) if (&op!=loop.getOperation() &&
          !isMemoryEffectFree(&op) && !isa<ttg::WarpYieldOp,ttg::WarpReturnOp>(op)) {
        bool allowed=!writer && op.isBeforeInBlock(loop.getOperation()) &&
            qPrefixLoad(&op,p.func,p.payload.getType().getElementType());
        if (!writer && loop.getNumResults() && loop->isBeforeInBlock(&op)) {
          if (auto store=dyn_cast<triton::StoreOp>(op)) {
            bool usesState=false;
            bool pure=registerEpilogue(store.getValue(),loop,usesState);
            auto type=store.getPtr().getType();
            if (auto t=dyn_cast<RankedTensorType>(type)) type=t.getElementType();
            auto ptr=dyn_cast<triton::PointerType>(type);
            allowed=pure && usesState && ptr && ptr.getAddressSpace()==1 && ptr.getPointeeType().isF32() &&
                (p.func.getNumArguments()<4 || globalInputRoot(store.getPtr())==p.func.getArgument(1));
          }
        }
        if (!allowed) return op.emitError("unproved effect outside counted pipe body");
      }
      for (Operation &op:loop.getBody()->without_terminator()) {
        if (op.getNumRegions() && (writer || !ownWaveReduction(&op,gpuPhase)))
          return op.emitError("only single-input own-wave add/max reductions are supported inside counted pipe");
        ops.push_back(&op);
      }
    } else for (Operation &op:region.front()) ops.push_back(&op);
    return success();
  };
  if (failed(collect(p.ws.getDefaultRegion(),writerOps,p.writerIV,writerTrip,true)) ||
      failed(collect(*p.ws.getPartitionRegions()[0],readerOps,p.readerIV,readerTrip,false))) return failure();
  if (writerTrip!=readerTrip) {p.create.emitError("producer and consumer counted trip counts differ");return failure();}
  p.counted=writerTrip!=0;
  auto matches=[&](Value v,bool writer,unsigned k,int mode) {
    auto expected=[&](unsigned i)->int64_t {return mode==0 ? i%2 : mode==1 ? (i/2)%2 : i*64;};
    if (!p.counted) return constant(v)==expected(k);
    for (unsigned i=0;i<writerTrip;++i)
      if (evaluate(v,writer?p.writerIV:p.readerIV,i)!=expected(i)) return false;
    return true;
  };
  auto checkLifecycle=[&](Operation *op,unsigned k,bool phase) -> LogicalResult {
    auto list=fields(op);
    if (list.size()!=fieldCount)return failure();
    for(unsigned i=0;i<payloadCount;++i)if(root(list[i])!=p.payloads[i].getResult())return failure();
    if ((gpuPhase && root(list.back())!=p.state.getResult()) || !matches(stage(op),isa<te::PipeWriterAcquireOp,te::PipeWriterCommitOp>(op),k,0) ||
        op->getAttr("field_names")!=p.create->getAttr("field_names") ||
        op->getAttr("scope")!=p.create->getAttr("scope") ||
        op->getAttrOfType<IntegerAttr>("capacity").getInt()!=2) return failure();
    if (auto name=op->getAttrOfType<StringAttr>("reader_name");name && !name.getValue().empty()) return failure();
    if (phase) {
      Value v=isa<te::PipeWriterAcquireOp>(op) ? cast<te::PipeWriterAcquireOp>(op).getPhase() : cast<te::PipeReaderWaitOp>(op).getPhase();
      if (!matches(v,isa<te::PipeWriterAcquireOp>(op),k,1)) return failure();
    }
    return success();
  };
  bool open=false; unsigned k=0,copies=0,waits=0;llvm::SmallPtrSet<Operation*,4> written,readFields;
  hcugpu::MatrixLoadToLocalOp pendingIssue; ttg::AsyncCommitGroupOp pendingCommit; bool waited=false;
  for (Operation *item:writerOps) {
    Operation &op=*item;
    if (isa<te::PipeWriterAcquireOp>(op)) {
      if (open || k>=16 || failed(checkLifecycle(&op,k,true))) {
        op.emitError("unbalanced/static stage-phase writer acquire"); return failure();
      }
      open=true;copies=0;waits=0;written.clear();pendingIssue=nullptr;pendingCommit=nullptr;waited=false;p.writer.push_back(&op);continue;
    }
    if (auto copy=dyn_cast<ttg::TMACopyOp>(op)) {
      auto view=copy.getDst().getDefiningOp<ttg::MemDescIndexOp>();
      auto desc=copy.getSrc().getDefiningOp<triton::MakeTensorDescOp>();
      if (gpuPhase || !open || ++copies>payloadCount || !view || p.payloadIndex(view.getSrc())<0 ||
          !written.insert(root(view.getSrc()).getDefiningOp()).second ||
          !matches(view.getIndex(),true,k,0) || !desc ||
          desc.getShape().size()!=2 || desc.getStrides().size()!=2 || copy.getIndices().size()!=2 ||
          constant(desc.getShape()[1])!=64 || constant(desc.getStrides()[0])!=64 ||
          constant(desc.getStrides()[1])!=1 || !matches(copy.getIndices()[0],true,k,2) ||
          constant(copy.getIndices()[1])!=0 || !aligned16(desc.getBase())) {
        op.emitError("pipe writer requires one full in-bounds aligned descriptor copy per stage"); return failure();
      }
      p.copies.push_back(copy);continue;
    }
    if (auto issue=dyn_cast<hcugpu::MatrixLoadToLocalOp>(op)) {
      auto view=issue.getDest().getDefiningOp<ttg::MemDescIndexOp>();
      if (!gpuPhase || !open || ++copies>payloadCount || !view || p.payloadIndex(view.getSrc())<0 ||
          !written.insert(root(view.getSrc()).getDefiningOp()).second ||
          !matches(view.getIndex(),true,k,0) || !issue.getBoundaryCheck().empty() ||
          issue.getTensorShape()!=ArrayRef<int32_t>{64,64} ||
          issue.getShape().size()!=2 || constant(issue.getShape()[1])!=64 ||
          issue.getStrides().size()!=2 || constant(issue.getStrides()[0])!=64 || constant(issue.getStrides()[1])!=1 ||
          issue.getIndices().size()!=2 || !matches(issue.getIndices()[0],true,k,2) || constant(issue.getIndices()[1])!=0 ||
          !aligned16(issue.getBase())) {op.emitError("unproved lowered pipe MLS transport");return failure();}
      pendingIssue=issue;pendingCommit=nullptr;waited=false;continue;
    }
    if (auto commit=dyn_cast<ttg::AsyncCommitGroupOp>(op)) {
      if (!gpuPhase || !pendingIssue || pendingCommit || commit->getNumOperands()!=1 || commit->getOperand(0)!=pendingIssue.getResult()) {
        op.emitError("unproved pipe async commit token");return failure();
      }
      pendingCommit=commit;continue;
    }
    if (auto wait=dyn_cast<hcugpu::AsyncWaitOp>(op)) {
      if (!gpuPhase || !pendingCommit || waited || wait->getNumOperands()!=1 ||
          wait->getOperand(0)!=pendingCommit.getResult() || wait.getNumInst()!=0) {
        op.emitError("pipe transport requires issuer wait0");return failure();
      }
      waited=true;++waits;continue;
    }
    if (isa<te::PipeWriterCommitOp>(op)) {
      if (!open || (copies!=payloadCount || written.size()!=payloadCount || (gpuPhase && waits!=payloadCount)) || failed(checkLifecycle(&op,k,false))) {
        op.emitError("unbalanced static pipe writer commit"); return failure();
      }
      open=false;++k;p.writer.push_back(&op);continue;
    }
    if (isPipe(&op) || (!isMemoryEffectFree(&op) && !isa<ttg::WarpYieldOp>(op))) {
      op.emitError("unproved operation in static pipe writer");return failure();
    }
  }
  if (open || k==0 || k>16) {p.create.emitError("empty/unbalanced static writer sequence");return failure();}
  if (p.counted && k!=1) {p.create.emitError("counted writer must contain exactly one lifecycle per iteration");return failure();}
  p.rounds=p.counted?writerTrip:k;
  if (!gpuPhase) for (auto c:p.copies) {
    auto d=c.getSrc().getDefiningOp<triton::MakeTensorDescOp>();
    if (constant(d.getShape()[0])!=int64_t(p.rounds*64)) {
      c.emitError("pipe descriptor rows must exactly cover the static sequence");return failure();
    }
  }
  open=false;k=0;unsigned loads=0;bool released=false;
  for (Operation *item:readerOps) {
    Operation &op=*item;
    if (isa<te::PipeReaderWaitOp>(op)) {
      if (open || k>=(p.counted?1:p.rounds) || failed(checkLifecycle(&op,k,true))) {op.emitError("unbalanced static reader wait");return failure();}
      open=true;loads=0;readFields.clear();released=false;p.reader.push_back(&op);continue;
    }
    if (auto load=dyn_cast<ttg::LocalLoadOp>(op)) {
      Value src=load.getSrc();auto trans=src.getDefiningOp<ttg::MemDescTransOp>();
      bool transposed=bool(trans);
      if (trans) {if(trans.getOrder()!=ArrayRef<int32_t>{1,0})return load.emitError("unsupported pipe shared transpose");src=trans.getSrc();}
      auto view=src.getDefiningOp<ttg::MemDescIndexOp>();
      auto ty=cast<RankedTensorType>(load.getType());
      bool layoutOK=transposed ? dotBEncoding(ty.getEncoding()) : (rowEncoding(ty.getEncoding()) || dotBEncoding(ty.getEncoding()));
      if (!gpuPhase || !open || ++loads>payloadCount || !view || p.payloadIndex(view.getSrc())<0 ||
          !readFields.insert(root(view.getSrc()).getDefiningOp()).second ||
          !matches(view.getIndex(),false,k,0) || ty.getShape()!=ArrayRef<int64_t>{64,64} || !layoutOK) {
        op.emitError("reader requires full-tile own-row local load");return failure();
      }
      p.reads.push_back(&op);continue;
    }
    if (auto load=dyn_cast<triton::LoadOp>(op)) {
      auto pointers=load.getPtr().getDefiningOp<te::LocalPointersOp>();
      auto view=pointers ? pointers->getOperand(0).getDefiningOp<ttg::MemDescIndexOp>() : ttg::MemDescIndexOp();
      if (gpuPhase || !open || ++loads>payloadCount || !pointers || !view || p.payloadIndex(view.getSrc())<0 ||
          !readFields.insert(root(view.getSrc()).getDefiningOp()).second ||
          !matches(view.getIndex(),false,k,0)) {op.emitError("reader must load only its acquired pipe slot");return failure();}
      p.reads.push_back(&op);continue;
    }
    if (auto pointers=dyn_cast<te::LocalPointersOp>(op)) {
      for (Operation *u:pointers->getResult(0).getUsers())
        if (!isa<triton::LoadOp>(u)) {op.emitError("pipe shared pointers may only be read");return failure();}
      continue;
    }
    if (auto store=dyn_cast<hcugpu::BufferStoreOp>(op)) {
      auto pointer=dyn_cast<triton::PointerType>(store.getPtr().getType());
      if (!open || !pointer || pointer.getAddressSpace()!=1 ||
          (p.func.getNumArguments()>=4 && globalInputRoot(store.getPtr())!=p.func.getArgument(1))) {op.emitError("pipe buffered store must target global output");return failure();}
      continue;
    }
    if (auto store=dyn_cast<triton::StoreOp>(op)) {
      auto type=store.getPtr().getType();
      if (auto t=dyn_cast<RankedTensorType>(type)) type=t.getElementType();
      auto pointer=dyn_cast<triton::PointerType>(type);
      bool registerOnlyAfterRelease=!open && released && !p.reads.empty() && store.getValue()==p.reads.back()->getResult(0);
      if ((!open && !registerOnlyAfterRelease) || !pointer || pointer.getAddressSpace()!=1 ||
          (p.func.getNumArguments()>=4 && globalInputRoot(store.getPtr())!=p.func.getArgument(1))) {op.emitError("reader store requires acquired interval or the released tile's register value");return failure();}
      continue;
    }
    if (isa<te::PipeReaderReleaseOp>(op)) {
      if (!open || loads!=payloadCount || readFields.size()!=payloadCount || failed(checkLifecycle(&op,k,false))) {op.emitError("unbalanced reader release");return failure();}
      open=false;released=true;++k;p.reader.push_back(&op);continue;
    }
    if (isPipe(&op) || (!isMemoryEffectFree(&op) && !ownWaveReduction(&op,gpuPhase) && !isa<ttg::WarpReturnOp>(op))) {
      op.emitError("unproved operation in static pipe reader");return failure();
    }
  }
  if (open || k!=(p.counted?1:p.rounds)) {p.create.emitError("writer/reader static trip counts differ");return failure();}
  // All aliases of the payload must stay within this pipe or its stage views.
  auto validUser=[&](Operation *u) {
    return isPipe(u) || u==p.ws.getOperation() || isa<ttg::MemDescIndexOp>(u);
  };
  for(auto payload:p.payloads)for (Operation *u:payload.getResult().getUsers()) if (!validUser(u)) {
    u->emitError("pipe payload escapes its validated lifecycle");return failure();
  }
  for (BlockArgument arg:p.ws.getPartitionRegions()[0]->front().getArguments()) {
    if (p.payloadIndex(arg)>=0) for (Operation *u:arg.getUsers()) if (!validUser(u)) {
      u->emitError("pipe reader payload escapes its lifecycle");return failure();
    }
  }
  if (gpuPhase) {
    auto validStateUser=[&](Operation *u) {return isPipe(u) || u==p.ws.getOperation();};
    for (Operation *u:p.state.getResult().getUsers()) if (!validStateUser(u)) {
      u->emitError("private epoch storage escapes its protocol");return failure();
    }
    for (BlockArgument arg:p.ws.getPartitionRegions()[0]->front().getArguments())
      if (root(arg)==p.state.getResult()) for (Operation *u:arg.getUsers()) if (!validStateUser(u)) {
        u->emitError("private reader epoch storage escapes its protocol");return failure();
      }
  }
  return p;
}

inline LogicalResult prepare(ModuleOp mod) {
  bool found=false;mod.walk([&](te::PipeCreateOp){found=true;});
  if (!found) return success();
  auto result=inspect(mod,false);if (failed(result)) return failure();
  auto &p=*result;OpBuilder b(p.create);auto loc=p.create.getLoc();
  auto cta=ttg::CTAEncodingAttr::getDefault(mod.getContext(),2);
  auto shared=ttg::SwizzledSharedEncodingAttr::get(mod.getContext(),1,1,1,ArrayRef<unsigned>{1,0},cta);
  auto type=ttg::MemDescType::get({2,8},b.getI32Type(),shared,ttg::SharedMemorySpaceAttr::get(mod.getContext()),true);
  auto state=ttg::LocalAllocOp::create(b,loc,type,Value());state->setAttr("alignment",b.getI32IntegerAttr(256));
  state->setAttr(stateAttr,b.getUnitAttr());for(auto payload:p.payloads)payload->setAttr(payloadAttr,b.getUnitAttr());
  p.ws->insertOperands(p.ws->getNumOperands(),state.getResult());
  auto readerState=p.ws.getPartitionRegions()[0]->front().addArgument(type,loc);
  auto names=llvm::to_vector(p.create->getAttrOfType<ArrayAttr>("field_names"));names.push_back(b.getStringAttr("epoch_state"));
  auto append=[&](Operation *op,Value v) {
    op->insertOperands(fields(op).size(),v);op->setAttr("field_names",b.getArrayAttr(names));
    op->setAttr(preparedAttr,b.getUnitAttr());
  };
  append(p.create,state.getResult());
  for (auto [i,op]:llvm::enumerate(p.writer)) {append(op,state.getResult());op->setAttr(iterationAttr,b.getI32IntegerAttr(i/2));}
  for (auto [i,op]:llvm::enumerate(p.reader)) {append(op,readerState);op->setAttr(iterationAttr,b.getI32IntegerAttr(i/2));}
  for (auto [i,copy]:llvm::enumerate(p.copies)) copy->setAttr(iterationAttr,b.getI32IntegerAttr(i/p.payloads.size()));
  p.create->setAttr("tle.hcu.static_pipe_rounds",b.getI32IntegerAttr(p.rounds));
  return success();
}
inline constexpr const char *verifiedAttr="tle.hcu.static_pipe_verified";
inline constexpr const char *transportAttr="tle.hcu.static_pipe_transport_verified";
inline constexpr const char *accessAttr="tle.hcu.static_pipe_access_verified";
inline LogicalResult verifyGPU(ModuleOp mod) {
  mod->removeAttr(verifiedAttr);mod->removeAttr(countedAttr);
  mod->removeAttr("tle.hcu.wave_local_scratch_proven");mod->removeAttr("tle.hcu.mma_layout_proof");
  mod.walk([](Operation *op){op->removeAttr("tle.hcu.wave_local_scratch");});
  mod.walk([](Operation *op) {op->removeAttr(transportAttr);op->removeAttr(accessAttr);op->removeAttr(pairAttr);op->removeAttr(countedAttr);});
  bool found=false;mod.walk([&](te::PipeCreateOp){found=true;});
  if (!found) return success();
  auto result=inspect(mod,true);if (failed(result)) return failure();
  auto &p=*result;OpBuilder b(mod.getContext());
  SmallVector<Attribute> proofs;
  p.ws.walk([&](ttg::ConvertLayoutOp convert) {
    if (!sameWaveStateConvert(convert))return;
    convert->setAttr("tle.hcu.wave_local_scratch",b.getUnitAttr());
    proofs.push_back(b.getDictionaryAttr({b.getNamedAttr("source",TypeAttr::get(convert.getSrc().getType())),
        b.getNamedAttr("target",TypeAttr::get(convert.getType())),
        b.getNamedAttr("source_linear",b.getStringAttr(ttg::toLinearLayout(convert.getSrc().getType()).toString())),
        b.getNamedAttr("target_linear",b.getStringAttr(ttg::toLinearLayout(convert.getType()).toString()))}));
  });
  if (!proofs.empty()) {
    mod->setAttr("tle.hcu.wave_local_scratch_proven",b.getUnitAttr());
    mod->setAttr("tle.hcu.mma_layout_proof",b.getArrayAttr(proofs));
  }
  unsigned n=0;
  p.ws.getDefaultRegion().walk([&](hcugpu::MatrixLoadToLocalOp issue) {
    if (constant(issue.getShape()[0])!=int64_t(p.rounds*64)) return;
    issue->setAttr(transportAttr,b.getUnitAttr());issue->setAttr(accessAttr,b.getUnitAttr());
    issue->setAttr(iterationAttr,b.getI32IntegerAttr(n++/p.payloads.size()));
  });
  if (n!=(p.counted?1:p.rounds)*p.payloads.size()) {p.create.emitError("static pipe transport count changed");return failure();}
  if (p.counted) {
    mod->setAttr(countedAttr,b.getUnitAttr());p.create->setAttr(countedAttr,b.getUnitAttr());
    for (Operation *op:p.writer) op->setAttr(pairAttr,b.getI32IntegerAttr(0));
    for (Operation *op:p.reader) op->setAttr(pairAttr,b.getI32IntegerAttr(1));
  }
  for (auto [i,op]:llvm::enumerate(p.writer)) {op->setAttr(verifiedAttr,b.getUnitAttr());op->setAttr(iterationAttr,b.getI32IntegerAttr(i/2));}
  for (auto [i,op]:llvm::enumerate(p.reader)) {op->setAttr(verifiedAttr,b.getUnitAttr());op->setAttr(iterationAttr,b.getI32IntegerAttr(i/2));}
  for (Operation *op:p.reads) {
    op->setAttr(accessAttr,b.getUnitAttr());
    if (isa<triton::LoadOp>(op)) op->getOperand(0).getDefiningOp()->setAttr(accessAttr,b.getUnitAttr());
  }
  p.create->setAttr(verifiedAttr,b.getUnitAttr());mod->setAttr(verifiedAttr,b.getI32IntegerAttr(1));
  return success();
}
inline bool isTransportWait(hcugpu::AsyncWaitOp wait,hcugpu::MatrixLoadToLocalOp &issue) {
  if (wait.getNumInst()!=0 || wait->getNumOperands()!=1) return false;
  auto commit=wait->getOperand(0).getDefiningOp<ttg::AsyncCommitGroupOp>();
  if (!commit || commit->getNumOperands()!=1) return false;
  issue=commit->getOperand(0).getDefiningOp<hcugpu::MatrixLoadToLocalOp>();
  return issue && issue->hasAttr(transportAttr) && issue->getParentOfType<ModuleOp>()->hasAttr(verifiedAttr);
}
inline bool synchronizedPair(Operation *a,Operation *b) {
  if (!a || !b) return false;
  auto mod=a->getParentOfType<ModuleOp>();
  if (!mod || !mod->hasAttr(verifiedAttr)) return false;
  return (isPipe(a) && a->hasAttr(verifiedAttr)) || (isPipe(b) && b->hasAttr(verifiedAttr)) ||
         (a->hasAttr(accessAttr) && b->hasAttr(accessAttr));
}
} // namespace mlir::triton::HCU::static_pipe
#endif
