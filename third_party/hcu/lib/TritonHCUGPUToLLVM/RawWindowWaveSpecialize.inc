// Included inside namespace rawws, after LateTrace. No shape/layout selector.
// This only specializes control: it grants no descriptor-uniformity authority.
// Complete old source/static proofs and post-clone per-wave memory projection
// are required. Original live IR is untouched until the trial passes.
static bool waveInvariantLeaf(Value v, llvm::SmallPtrSetImpl<void*>& active,
                              unsigned depth=0) {
  if(depth>96||!scalarWidth(v.getType())||scalarWidth(v.getType())>64||isa<BlockArgument>(v))return false;
  Operation* op=v.getDefiningOp();if(!op||!active.insert(v.getAsOpaquePointer()).second)return false;
  StringRef n=op->getName().getStringRef(),fn=callee(op);
  bool leaf=false,recursive=false;
  if(n=="llvm.mlir.constant"||n=="arith.constant") {
    auto a=op->getAttrOfType<IntegerAttr>("value");
    leaf=bool(a)&&op->getNumOperands()==0&&op->getNumResults()==1;
  } else if(n=="rocdl.workitem.id.x"||fn=="llvm.amdgcn.workitem.id.x") {
    leaf=op->getNumOperands()==0&&op->getNumResults()==1&&v.getType().isInteger(32);
  } else if(n=="gpu.thread_id") {
    auto dim=op->getAttrOfType<mlir::gpu::DimensionAttr>("dimension");
    leaf=dim&&dim.getValue()==mlir::gpu::Dimension::x&&op->getNumOperands()==0&&op->getNumResults()==1;
  } else if(n=="rocdl.readfirstlane"||fn=="llvm.amdgcn.readfirstlane") {
    recursive=op->getNumOperands()==1&&op->getNumResults()==1&&v.getType().isInteger(32)&&op->getOperand(0).getType().isInteger(32);
  } else {
    // All operands, including both select arms, must have invariant leaves.
    // A known empty-env selected arm alone must not hide a PHI/kernel argument.
    recursive=llvm::is_contained(ArrayRef<StringRef>{
      "llvm.add","llvm.sub","llvm.mul","llvm.and","llvm.or","llvm.xor",
      "llvm.lshr","llvm.ashr","llvm.shl","llvm.udiv","llvm.urem","llvm.sdiv","llvm.srem",
      "llvm.icmp","llvm.select","llvm.zext","llvm.sext","llvm.trunc","llvm.bitcast",
      "arith.addi","arith.subi","arith.muli","arith.cmpi","arith.select",
      "arith.extui","arith.extsi","arith.trunci","arith.index_cast",
      "builtin.unrealized_conversion_cast"},n)&&op->getNumResults()==1&&isPure(op);
  }
  bool ok=leaf||(recursive&&llvm::all_of(op->getOperands(),[&](Value x){return waveInvariantLeaf(x,active,depth+1);}));
  active.erase(v.getAsOpaquePointer());return ok;
}
struct WaveBranchSite {Operation* op;SmallVector<uint8_t> choices;};
static void specializeWaveControl(WarpSpecializeOp ws,const Proof& proof) {
  auto mod=ws->getParentOfType<ModuleOp>();Builder attrs(ws.getContext());
  auto report=[&](unsigned classes,unsigned sourceSites,unsigned folded,StringRef reason) {
    mod->setAttr("ttg.hcu.raw_window_ws.wave_classes",attrs.getI32IntegerAttr(classes));
    mod->setAttr("ttg.hcu.raw_window_ws.wave_source_sites",attrs.getI32IntegerAttr(sourceSites));
    mod->setAttr("ttg.hcu.raw_window_ws.wave_folded_sites",attrs.getI32IntegerAttr(folded));
    mod->setAttr("ttg.hcu.raw_window_ws.wave_specialization",attrs.getStringAttr(reason));
  };
  report(0,0,0,"not-applied");
  Region& source=ws.getDefaultRegion();
  if(!proof.staggered||source.empty()||source.front().getNumArguments()||!proof.base) {report(0,0,0,"unsupported-entry");return;}
  SmallVector<WaveBranchSite> sites;
  for(Block& block:source)for(Operation& op:block) {
    if(op.getNumRegions()) {report(0,0,0,"nested-region");return;}
    Value condition;
    if(auto b=dyn_cast<LLVM::CondBrOp>(&op))condition=b.getCondition();
    else if(auto b=dyn_cast<cf::CondBranchOp>(&op))condition=b.getCondition();
    if(!condition)continue;
    llvm::SmallPtrSet<void*,32> active;
    if(!waveInvariantLeaf(condition,active))continue;
    WaveBranchSite site{&op,{}};bool valid=true;LaneEnvs empty(64);
    for(unsigned wave=0;wave<proof.base&&valid;++wave) {
      std::optional<bool> choice;
      for(unsigned lane=0;lane<64;++lane) {
        auto x=lateInteger(condition,empty,wave,lane);
        if(!x||(choice&&*choice!=bool(*x))) {valid=false;break;}
        choice=bool(*x);
      }
      if(valid)site.choices.push_back(uint8_t(*choice));
    }
    if(valid)sites.push_back(std::move(site));
  }
  if(sites.empty()) {report(0,0,0,"no-proved-invariant-sites");return;}
  // Equality of the entire proven branch signature determines a class. No
  // contiguous-wave, number-of-classes, tile or parent-layout assumption.
  SmallVector<SmallVector<unsigned>> classes;
  for(unsigned wave=0;wave<proof.base;++wave) {
    bool found=false;
    for(auto& group:classes) {
      bool same=llvm::all_of(sites,[&](const WaveBranchSite& s){return s.choices[wave]==s.choices[group.front()];});
      if(same){group.push_back(wave);found=true;break;}
    }
    if(!found)classes.push_back(SmallVector<unsigned>{wave});
  }
  if(classes.size()<2) {report(0,sites.size(),0,"single-signature-no-clone");return;}
  // A generic finite code-growth budget, not a hardware geometry selector.
  unsigned ops=0;source.walk([&](Operation*){++ops;});
  if(uint64_t(ops)*classes.size()>262144) {report(0,sites.size(),0,"clone-operation-budget");return;}
  // The replacement must already have a container while cloning/building:
  // OpBuilder and Region::cloneInto may query Region::getContext().
  Operation* trialOp=ws->clone();OpBuilder anchored(ws.getOperation());anchored.insert(trialOp);
  auto eraseTrial=llvm::make_scope_exit([&]{trialOp->erase();});
  auto trial=cast<WarpSpecializeOp>(trialOp);
  Region& replacement=trial.getDefaultRegion();
  replacement.dropAllReferences();replacement.getBlocks().clear();
  auto* dispatch=new Block;replacement.push_back(dispatch);
  SmallVector<Block*> entries;unsigned folded=0;
  for(const auto& group:classes) {
    IRMapping map;source.cloneInto(&replacement,map);entries.push_back(map.lookup(&source.front()));
    for(const auto& site:sites) {
      Operation* cloned=map.lookupOrNull(site.op);
      if(!cloned) {report(0,sites.size(),0,"missing-clone-map");return;}
      bool take=site.choices[group.front()];OpBuilder b(cloned);
      if(auto branch=dyn_cast<LLVM::CondBrOp>(cloned)) {
        Block* dest=take?branch.getTrueDest():branch.getFalseDest();
        ValueRange args=take?branch.getTrueDestOperands():branch.getFalseDestOperands();
        b.create<LLVM::BrOp>(branch.getLoc(),args,dest);
      } else if(auto branch=dyn_cast<cf::CondBranchOp>(cloned)) {
        Block* dest=take?branch.getTrueDest():branch.getFalseDest();
        ValueRange args=take?branch.getTrueDestOperands():branch.getFalseDestOperands();
        b.create<cf::BranchOp>(branch.getLoc(),dest,args);
      } else {report(0,sites.size(),0,"unexpected-cloned-branch");return;}
      cloned->erase();++folded;
    }
  }
  TritonLLVMIRRewriter b(ws.getLoc(),OpBuilder::atBlockEnd(dispatch));
  Value tid=b.create<mlir::gpu::ThreadIdOp>(mlir::gpu::Dimension::x);
  tid=b.create<arith::IndexCastOp>(b.getI32Type(),tid);
  Value wave=b.create<ROCDL::ReadfirstlaneOp>(b.getI32Type(),b.udiv(tid,b.i32_val(64)));
  for(unsigned c=0;c+1<classes.size();++c) {
    Value match;
    for(unsigned w:classes[c]) {
      Value eq=b.icmp_eq(wave,b.i32_val(w));match=match?b.or_(match,eq):eq;
    }
    Block* next=new Block;replacement.push_back(next);
    b.create<LLVM::CondBrOp>(match,entries[c],next);b.setInsertionPointToEnd(next);
  }
  b.create<LLVM::BrOp>(ValueRange{},entries.back());
  // Attached trial preserves symbol/provenance/capture context. Scope exit
  // erases it on every return; original source remains unchanged until commit.
  if(failed(mlir::verify(trialOp))) {report(0,sites.size(),0,"trial-SSA-verifier-rejected");return;}
  for(unsigned w=0;w<proof.base;++w) {
    LateTrace trace{w,proof.segments[w],true};
    if(!trace.run(trial.getDefaultRegion())) {
      std::string reason="trial-per-wave-rejected: "+trace.why;
      report(0,sites.size(),0,reason);return;
    }
  }
  ws.getDefaultRegion().takeBody(trial.getDefaultRegion());
  report(classes.size(),sites.size(),folded,"applied-fresh-per-wave-verified");
}
