#ifndef LLVMXPUIR_OPS
#define LLVMXPUIR_OPS

include "mlir/IR/OpBase.td" // Trait
include "mlir/IR/DialectBase.td"
include "mlir/Dialect/LLVMIR/LLVMOpBase.td" // LLVM_OpBase
include "mlir/Interfaces/InferTypeOpInterface.td" // SameOperandsAndResultType


//===----------------------------------------------------------------------===//
// LLVMXPU dialect definitions
//===----------------------------------------------------------------------===//

def LLVMXPU_Dialect : Dialect {
  let name = "llvm_xpu";

  let cppNamespace = "::mlir::LLVM::XPU";

  let hasOperationAttrVerify = 1;

  let dependentDialects = ["LLVM::LLVMDialect"];

  let extraClassDeclaration = [{
    /// Get the name of the attribute used to annotate external kernel
    /// functions.
    static StringRef getKernelFuncAttrName() { return "xpu.kernel"; }
  }];

  // let useDefaultAttributePrinterParser = 1;
}



//===----------------------------------------------------------------------===//
// LLVMXPU op definitions
//===----------------------------------------------------------------------===//

class XPU_Op<string mnemonic, list<Trait> traits = []> :
  LLVM_OpBase<LLVMXPU_Dialect, mnemonic, traits> {
}


//===----------------------------------------------------------------------===//
// LLVMXPU intrinsic operations
//===----------------------------------------------------------------------===//

class XPU_IntrOp<string mnem, list<Trait> traits = [],
                  int numResults>
  : LLVM_IntrOpBase<LLVMXPU_Dialect, mnem, "xpu_" # !subst(".", "_", mnem),
                    /*list<int> overloadedResults=*/[],
                    /*list<int> overloadedOperands=*/[],
                    traits, numResults>;


def XPU_GM2LMOp : XPU_IntrOp<"gm2lm", [], 0> {
    let arguments = (ins LLVM_AnyPointer:$src, // TODO[dyq]: Check i8?
                         LLVM_AnyPointer:$dst,
                         I32:$offset,
                         I32:$size);
    string llvmBuilder = [{
        // XPU2, offset only has 7 bits, so we let LLVM fold the value
        auto addr  = builder.CreatePtrToInt($src, $offset->getType());
        auto faddr = builder.CreateAdd(addr, $offset);
        auto srcptr  = builder.CreateIntToPtr(faddr, $src->getType());
        auto zero = builder.getInt32(0);
        auto *mod = builder.GetInsertBlock()->getModule();
        auto *vTy = builder.getVoidTy();
        auto *fTy = llvm::FunctionType::get(vTy, {$dst->getType(), srcptr->getType(), zero->getType(), $size->getType()}, false);
        auto fn = mod->getOrInsertFunction("llvm.xpu.gm2lm", fTy);
        builder.CreateCall(fTy, fn.getCallee(), {$dst, srcptr, zero, $size});
    }];
}

def XPU_LM2GMOp : XPU_IntrOp<"lm2gm", [], 0> {
    let arguments = (ins LLVM_AnyPointer:$src,
                         LLVM_AnyPointer:$dst,
                         I32:$offset,
                         I32:$size);
    string llvmBuilder = [{
        // XPU2, offset only has 7 bits, so we let LLVM fold the value
        auto addr  = builder.CreatePtrToInt($dst, $offset->getType());
        auto faddr = builder.CreateAdd(addr, $offset);
        auto dstptr  = builder.CreateIntToPtr(faddr, $dst->getType());
        auto zero = builder.getInt32(0);
        auto *mod = builder.GetInsertBlock()->getModule();
        auto *vTy = builder.getVoidTy();
        auto *fTy = llvm::FunctionType::get(vTy, {dstptr->getType(), zero->getType(), $src->getType(), $size->getType()}, false);
        auto fn = mod->getOrInsertFunction("llvm.xpu.lm2gm", fTy);
        builder.CreateCall(fTy, fn.getCallee(), {dstptr, zero, $src, $size});
    }];
}

def XPU_SM2GMOp : XPU_IntrOp<"sm2gm", [], 0> {
    let arguments = (ins LLVM_AnyPointer:$src,
                         LLVM_AnyPointer:$dst,
                         I32:$offset,
                         I32:$size);
    string llvmBuilder = [{
        // XPU2, offset only has 7 bits, so we let LLVM fold the value
        auto addr  = builder.CreatePtrToInt($dst, $offset->getType());
        auto faddr = builder.CreateAdd(addr, $offset);
        auto dstptr  = builder.CreateIntToPtr(faddr, $dst->getType());
        auto zero = builder.getInt32(0);
        auto *mod = builder.GetInsertBlock()->getModule();
        auto *vTy = builder.getVoidTy();
        auto *fTy = llvm::FunctionType::get(vTy, {dstptr->getType(), zero->getType(), $src->getType(), $size->getType()}, false);
        auto fn = mod->getOrInsertFunction("llvm.xpu.sm2gm", fTy);
        builder.CreateCall(fTy, fn.getCallee(), {dstptr, zero, $src, $size});
    }];
}


def XPU_GM2LMOp_v3 : XPU_IntrOp<"gm2lm_v3", [], 0> {
  let arguments = (ins LLVM_AnyPointer:$src,
                       LLVM_AnyPointer:$dst,
                       I32:$offset,
                       I32:$size);
  string llvmBuilder = [{
    auto isrc = builder.CreatePtrToInt($src, builder.getInt64Ty());
    auto ioft = builder.CreateZExtOrTrunc($offset, builder.getInt64Ty());
    auto asrc = builder.CreateAdd(isrc, ioft);
    auto fsrc = builder.CreateIntToPtr(asrc, $src->getType());
    auto zero = builder.getInt32(0);
    auto *module = builder.GetInsertBlock()->getModule();
    auto *voidTy = builder.getVoidTy();
    auto *ptrTy = $dst->getType();
    auto *i32Ty = builder.getInt32Ty();
    auto *funcTy = llvm::FunctionType::get(voidTy, {ptrTy, fsrc->getType(), i32Ty, i32Ty}, false);
    auto callee = module->getOrInsertFunction("llvm.xpu.gm2lm_v3", funcTy);
    builder.CreateCall(funcTy, callee.getCallee(), {$dst, fsrc, zero, $size});
  }];
}

def XPU_LM2GMOp_v3 : XPU_IntrOp<"lm2gm_v3", [], 0> {
  let arguments = (ins LLVM_AnyPointer:$src,
                       LLVM_AnyPointer:$dst,
                       I32:$offset,
                       I32:$size);
  string llvmBuilder = [{
    auto idst = builder.CreatePtrToInt($dst, builder.getInt64Ty());
    auto ioft = builder.CreateZExtOrTrunc($offset, builder.getInt64Ty());
    auto adst = builder.CreateAdd(idst, ioft);
    auto fdst = builder.CreateIntToPtr(adst, $dst->getType());
    auto zero = builder.getInt32(0);
    auto *module_lm = builder.GetInsertBlock()->getModule();
    auto *voidTy_lm = builder.getVoidTy();
    auto *i32Ty_lm = builder.getInt32Ty();
    auto *funcTy_lm = llvm::FunctionType::get(voidTy_lm, {fdst->getType(), $src->getType(), i32Ty_lm, i32Ty_lm}, false);
    auto callee_lm = module_lm->getOrInsertFunction("llvm.xpu.lm2gm_v3", funcTy_lm);
    builder.CreateCall(funcTy_lm, callee_lm.getCallee(), {fdst, $src, zero, $size});
  }];
}

def XPU_SM2GMOp_v3 : XPU_IntrOp<"sm2gm_v3", [], 0> {
  let arguments = (ins LLVM_AnyPointer:$src,
                       LLVM_AnyPointer:$dst,
                       I32:$offset,
                       I32:$size);
  string llvmBuilder = [{
    auto idst = builder.CreatePtrToInt($dst, builder.getInt64Ty());
    auto ioft = builder.CreateZExtOrTrunc($offset, builder.getInt64Ty());
    auto adst = builder.CreateAdd(idst, ioft);
    auto fdst = builder.CreateIntToPtr(adst, $dst->getType());
    auto zero = builder.getInt32(0);
    auto *mod = builder.GetInsertBlock()->getModule();
    auto *vTy = builder.getVoidTy();
    auto *i32Ty = builder.getInt32Ty();
    auto *fTy = llvm::FunctionType::get(vTy, {fdst->getType(), $src->getType(), i32Ty, i32Ty}, false);
    auto fn = mod->getOrInsertFunction("llvm.xpu.sm2gm_v3", fTy);
    builder.CreateCall(fTy, fn.getCallee(), {fdst, $src, zero, $size});
  }];
}

//===----------------------------------------------------------------------===//
// XPU special register op definitions
//===----------------------------------------------------------------------===//

class XPU_SpecialRegisterOp<string mnemonic, list<Trait> traits = []> :
  XPU_IntrOp<mnemonic, !listconcat(traits, [Pure]), 1> {
  let arguments = (ins);
  let assemblyFormat = "attr-dict `:` type($res)";
}

//===----------------------------------------------------------------------===//
// Physiscal cluster index and range (0-7)
//===----------------------------------------------------------------------===//

def XPU_ClusterIdOp : XPU_SpecialRegisterOp<"cluster_id"> {
  string llvmBuilder = [{
    auto *module = builder.GetInsertBlock()->getModule();
    auto *i32Ty = builder.getInt32Ty();
    auto *funcTy = llvm::FunctionType::get(i32Ty, {}, false);
    auto callee = module->getOrInsertFunction("llvm.xpu.cluster_id", funcTy);
    $res = builder.CreateCall(funcTy, callee.getCallee(), {});
  }];
}

//===----------------------------------------------------------------------===//
// Core index and range
//===----------------------------------------------------------------------===//

def XPU_CoreIdOp : XPU_SpecialRegisterOp<"core_id"> {
  string llvmBuilder = [{
    auto *module = builder.GetInsertBlock()->getModule();
    auto *i32Ty = builder.getInt32Ty();
    auto *funcTy = llvm::FunctionType::get(i32Ty, {}, false);
    auto callee = module->getOrInsertFunction("llvm.xpu.core_id", funcTy);
    $res = builder.CreateCall(funcTy, callee.getCallee(), {});
  }];
}

//===----------------------------------------------------------------------===//
// XPU load parameters op definitions
//===----------------------------------------------------------------------===//

def XPU_LoadParamOp : XPU_Op<"load_param"> {
  let arguments = (ins I32:$num);
  let results = (outs I32:$res);
  string llvmBuilder = [{
    auto *module = builder.GetInsertBlock()->getModule();
    auto *i32Ty = builder.getInt32Ty();
    auto *funcTy = llvm::FunctionType::get(i32Ty, {i32Ty}, false);
    auto callee = module->getOrInsertFunction("llvm.xpu.load_param", funcTy);
    auto *num = $num;
    if (num->getType() != i32Ty && num->getType()->isIntegerTy())
      num = builder.CreateIntCast(num, i32Ty, /*isSigned=*/false);
    $res = builder.CreateCall(funcTy, callee.getCallee(), {num});
  }];
  let assemblyFormat = "$num attr-dict";
}

//===----------------------------------------------------------------------===//
// XPU mfence v2 op definitions
//===----------------------------------------------------------------------===//

def XPU_MfenceOp : XPU_Op<"mfence"> {
  let arguments = (ins I32:$num);
  string llvmBuilder = [{
    auto *module_mf = builder.GetInsertBlock()->getModule();
    auto *voidTy_mf = builder.getVoidTy();
    auto *i32Ty_mf = builder.getInt32Ty();
    auto *funcTy_mf = llvm::FunctionType::get(voidTy_mf, {i32Ty_mf}, false);
    auto callee_mf = module_mf->getOrInsertFunction("llvm.xpu.mfence.v2", funcTy_mf);
    builder.CreateCall(funcTy_mf, callee_mf.getCallee(), {$num});
  }];
  let assemblyFormat = "$num attr-dict";
}

//===----------------------------------------------------------------------===//
// XPU barrier for inter-cluster
// mfence + sync_cluster
//===----------------------------------------------------------------------===//
def XPU_BarrierOp : XPU_Op<"barrier"> {
  let arguments = (ins);
  string llvmBuilder = [{
    auto *module = builder.GetInsertBlock()->getModule();
    auto *voidTy = builder.getVoidTy();
    auto *i32Ty = builder.getInt32Ty();
    auto five = builder.getInt32(7);
    auto *mfTy = llvm::FunctionType::get(voidTy, {i32Ty}, false);
    auto mfCallee = module->getOrInsertFunction("llvm.xpu.mfence.v2", mfTy);
    builder.CreateCall(mfTy, mfCallee.getCallee(), {five});
    auto mask = builder.getInt32(65535);
    auto *syncTy = llvm::FunctionType::get(voidTy, {i32Ty}, false);
    auto syncCallee = module->getOrInsertFunction("llvm.xpu.csr_set_sync_group", syncTy);
    builder.CreateCall(syncTy, syncCallee.getCallee(), {mask});
  }];
  let assemblyFormat = "attr-dict";
}

//===----------------------------------------------------------------------===//
// XPU set haddr op definitions
//===----------------------------------------------------------------------===//

def XPU_SetHaddrOp : XPU_Op<"set_haddr"> {
  let arguments = (ins I32:$num);
  string llvmBuilder = [{
    auto *mod = builder.GetInsertBlock()->getModule();
    auto *vTy = builder.getVoidTy();
    auto *i32Ty = builder.getInt32Ty();
    auto *fTy = llvm::FunctionType::get(vTy, {i32Ty}, false);
    auto fn = mod->getOrInsertFunction("llvm.xpu.set_haddr", fTy);
    builder.CreateCall(fTy, fn.getCallee(), {$num});
  }];
  let assemblyFormat = "$num attr-dict";
}

//===----------------------------------------------------------------------===//
// XPU log definitions
//===----------------------------------------------------------------------===//

def XPU_LogOp : XPU_Op<"log", [Elementwise, SameOperandsAndResultType, SameOperandsAndResultShape]> {
  let arguments = (ins F32:$operand);
  let results = (outs F32:$result);
  string llvmBuilder = [{
    auto *mod = builder.GetInsertBlock()->getModule();
    auto *f32Ty = builder.getFloatTy();
    auto *fTy = llvm::FunctionType::get(f32Ty, {f32Ty}, false);
    auto fn = mod->getOrInsertFunction("llvm.xpu.log2f", fTy);
    $result = builder.CreateCall(fTy, fn.getCallee(), {$operand});
  }];
  let assemblyFormat = "$operand attr-dict `:` type($result)";
}

//===----------------------------------------------------------------------===//
// XPU min definitions
//===----------------------------------------------------------------------===//
def XPU_MinOp : XPU_IntrOp<"min", [], 0> {
    let arguments = (ins I32:$lhs,
                         I32:$rhs);
    let results = (outs I32:$res);
    string llvmBuilder = [{
        // XTDK only supports llvm.xpu.min
        auto *mod = builder.GetInsertBlock()->getModule();
        auto *i32Ty = builder.getInt32Ty();
        auto *fTy = llvm::FunctionType::get(i32Ty, {i32Ty, i32Ty}, false);
        auto fn = mod->getOrInsertFunction("llvm.smin.i32", fTy);
        $res = builder.CreateCall(fTy, fn.getCallee(), {$lhs, $rhs});
    }];
}

//===----------------------------------------------------------------------===//
// XPU VGatherF definitions
//===----------------------------------------------------------------------===//
def XPU_VGatherFOp : XPU_IntrOp<"vgather_mask16_mr1", [], 0> {
    let arguments = (ins LLVM_AnyPointer:$src,
                         LLVM_AnyVector:$mask);
    let results = (outs LLVM_AnyVector:$res);
    string llvmBuilder = [{
        auto *mod = builder.GetInsertBlock()->getModule();
        auto *fTy = llvm::FunctionType::get($mask->getType(), {$src->getType(), $mask->getType()}, false);
        auto fn = mod->getOrInsertFunction("llvm.xpu.vgather_mask16_mr1", fTy);
        $res = builder.CreateCall(fTy, fn.getCallee(), {$src, $mask});
    }];
}

def XPU_VGatherHFOp : XPU_IntrOp<"vgather_mr1", [], 0> {
    let arguments = (ins LLVM_AnyPointer:$src,
                         LLVM_AnyVector:$mask);
    let results = (outs LLVM_AnyVector:$res);
    string llvmBuilder = [{
        auto *mod = builder.GetInsertBlock()->getModule();
        auto *i32Ty = builder.getInt32Ty();
        auto *fTy = llvm::FunctionType::get(i32Ty, {i32Ty, i32Ty}, false);
        auto fn = mod->getOrInsertFunction("llvm.xpu.vgather_mr1", fTy);
        $res = builder.CreateCall(fTy, fn.getCallee(), {$src, $mask});
    }];
}

//===----------------------------------------------------------------------===//
// XPU vvor_f_mh_rn definitions
//===----------------------------------------------------------------------===//
def XPU_VVOR_F_MHOp : XPU_IntrOp<"vvor_f_mh_rn", [], 0> {
    let arguments = (ins LLVM_AnyVector:$mask,
                         LLVM_AnyVector:$a,
                         LLVM_AnyVector:$b,
                         LLVM_AnyVector:$c);
    let results = (outs LLVM_AnyVector:$res);
    string llvmBuilder = [{
        auto *mod = builder.GetInsertBlock()->getModule();
        auto *i32Ty = builder.getInt32Ty();
        auto *fTy = llvm::FunctionType::get(i32Ty, {i32Ty, i32Ty, i32Ty, i32Ty}, false);
        auto fn = mod->getOrInsertFunction("llvm.xpu.vvor_f_mh_rn", fTy);
        $res = builder.CreateCall(fTy, fn.getCallee(), {$mask, $a, $b, $c});
    }];
}

def XPU_VVOR_HF_MHOp : XPU_IntrOp<"vvor_hf_mh_rn", [], 0> {
    let arguments = (ins LLVM_AnyVector:$mask,
                         LLVM_AnyVector:$a,
                         LLVM_AnyVector:$b,
                         LLVM_AnyVector:$c);
    let results = (outs LLVM_AnyVector:$res);
    string llvmBuilder = [{
        auto *mod = builder.GetInsertBlock()->getModule();
        auto *i32Ty = builder.getInt32Ty();
        auto *fTy = llvm::FunctionType::get(i32Ty, {i32Ty, i32Ty, i32Ty, i32Ty}, false);
        auto fn = mod->getOrInsertFunction("llvm.xpu.vvor_hf_mh_rn", fTy);
        $res = builder.CreateCall(fTy, fn.getCallee(), {$mask, $a, $b, $c});
    }];
}

def XPU_VVOR_S_MHOp : XPU_IntrOp<"vvor_s_mh", [], 0> {
    let arguments = (ins LLVM_AnyVector:$mask,
                         LLVM_AnyVector:$a,
                         LLVM_AnyVector:$b,
                         LLVM_AnyVector:$c);
    let results = (outs LLVM_AnyVector:$res);
    string llvmBuilder = [{
        auto *mod = builder.GetInsertBlock()->getModule();
        auto *i32Ty = builder.getInt32Ty();
        auto *fTy = llvm::FunctionType::get(i32Ty, {i32Ty, i32Ty, i32Ty, i32Ty}, false);
        auto fn = mod->getOrInsertFunction("llvm.xpu.vvor_s_mh", fTy);
        $res = builder.CreateCall(fTy, fn.getCallee(), {$mask, $a, $b, $c});
    }];
}

//===----------------------------------------------------------------------===//
// XPU vload/vstore definitions
//===----------------------------------------------------------------------===//
def XPU_VLOAD_MZOp : XPU_IntrOp<"vload_mz", [], 0> {
    let arguments = (ins LLVM_AnyPointer:$src_ptr,
                         LLVM_AnyVector:$mask);
    let results = (outs LLVM_AnyVector:$res);
    string llvmBuilder = [{
        auto zero = builder.getInt32(0);
        auto *mod = builder.GetInsertBlock()->getModule();
        auto *i32Ty = builder.getInt32Ty();
        auto *fTy = llvm::FunctionType::get(i32Ty, {i32Ty, $src_ptr->getType(), i32Ty}, false);
        auto fn = mod->getOrInsertFunction("llvm.xpu.vload_mz", fTy);
        $res = builder.CreateCall(fTy, fn.getCallee(), {$mask, $src_ptr, zero});
    }];
}

def XPU_VLOAD_MHOp : XPU_IntrOp<"vload_mh", [], 0> {
    let arguments = (ins LLVM_AnyPointer:$src_ptr,
                         LLVM_AnyVector:$dst_data,
                         LLVM_AnyVector:$mask);
    let results = (outs LLVM_AnyVector:$res);
    string llvmBuilder = [{
        auto zero = builder.getInt32(0);
        auto *mod = builder.GetInsertBlock()->getModule();
        auto *i32Ty = builder.getInt32Ty();
        auto *fTy = llvm::FunctionType::get(i32Ty, {i32Ty, $src_ptr->getType(), i32Ty, i32Ty}, false);
        auto fn = mod->getOrInsertFunction("llvm.xpu.vload_mh", fTy);
        $res = builder.CreateCall(fTy, fn.getCallee(), {$mask, $src_ptr, $dst_data, zero});
    }];
}

def XPU_VSTORE_MHOp : XPU_IntrOp<"vstore_mh", [], 0> {
    let arguments = (ins LLVM_AnyVector:$src,
                         LLVM_AnyPointer:$dst_Ptr,
                         LLVM_AnyVector:$mask);
    string llvmBuilder = [{
        auto zero = builder.getInt32(0);
        auto *mod = builder.GetInsertBlock()->getModule();
        auto *vTy = builder.getVoidTy();
        auto *i32Ty = builder.getInt32Ty();
        auto *fTy = llvm::FunctionType::get(vTy, {i32Ty, i32Ty, $dst_Ptr->getType(), i32Ty}, false);
        auto fn = mod->getOrInsertFunction("llvm.xpu.vstore_mh", fTy);
        builder.CreateCall(fTy, fn.getCallee(), {$src, $mask, $dst_Ptr, zero});
    }];
}

//===----------------------------------------------------------------------===//
// XPU svsllp/svsrlp definitions
//===----------------------------------------------------------------------===//
def XPU_SVSLLPOp : XPU_IntrOp<"svsllp", [], 0> {
    let arguments = (ins I32:$offset,
                         LLVM_AnyVector:$src);
    let results = (outs LLVM_AnyVector:$res);
    string llvmBuilder = [{
        auto *mod = builder.GetInsertBlock()->getModule();
        auto *i32Ty = builder.getInt32Ty();
        auto *fTy = llvm::FunctionType::get(i32Ty, {i32Ty, i32Ty}, false);
        auto fn = mod->getOrInsertFunction("llvm.xpu.svsllp_s", fTy);
        $res = builder.CreateCall(fTy, fn.getCallee(), {$offset, $src});
    }];
}

def XPU_SVSRLPOp : XPU_IntrOp<"svsrlp", [], 0> {
    let arguments = (ins I32:$offset,
                         LLVM_AnyVector:$src);
    let results = (outs LLVM_AnyVector:$res);
    string llvmBuilder = [{
        auto *mod = builder.GetInsertBlock()->getModule();
        auto *i32Ty = builder.getInt32Ty();
        auto *fTy = llvm::FunctionType::get(i32Ty, {i32Ty, i32Ty}, false);
        auto fn = mod->getOrInsertFunction("llvm.xpu.svsrlp_s", fTy);
        $res = builder.CreateCall(fTy, fn.getCallee(), {$offset, $src});
    }];
}

//===----------------------------------------------------------------------===//
// XPU vmerge definitions
//===----------------------------------------------------------------------===//
def XPU_VMERGE_L_HFOp : XPU_IntrOp<"vmerge_l_hf", [], 0> {
    let arguments = (ins LLVM_AnyVector:$src1,
                         LLVM_AnyVector:$src2);
    let results = (outs LLVM_AnyVector:$res);
    string llvmBuilder = [{
        auto *mod = builder.GetInsertBlock()->getModule();
        auto *i32Ty = builder.getInt32Ty();
        auto *fTy = llvm::FunctionType::get(i32Ty, {i32Ty, i32Ty}, false);
        auto fn = mod->getOrInsertFunction("llvm.xpu.vmerge_l_hf", fTy);
        $res = builder.CreateCall(fTy, fn.getCallee(), {$src1, $src2});
    }];
}

def XPU_VMERGE_H_HFOp : XPU_IntrOp<"vmerge_h_hf", [], 0> {
    let arguments = (ins LLVM_AnyVector:$src1,
                         LLVM_AnyVector:$src2);
    let results = (outs LLVM_AnyVector:$res);
    string llvmBuilder = [{
        auto *mod = builder.GetInsertBlock()->getModule();
        auto *i32Ty = builder.getInt32Ty();
        auto *fTy = llvm::FunctionType::get(i32Ty, {i32Ty, i32Ty}, false);
        auto fn = mod->getOrInsertFunction("llvm.xpu.vmerge_h_hf", fTy);
        $res = builder.CreateCall(fTy, fn.getCallee(), {$src1, $src2});
    }];
}

//===----------------------------------------------------------------------===//
// XPU vscatter definitions
//===----------------------------------------------------------------------===//
def XPU_SCATTER_MHOp : XPU_IntrOp<"vscatter_mh", [], 0> {
    let arguments = (ins LLVM_AnyVector:$value,
                         LLVM_AnyVector:$mask,
                         LLVM_AnyPointer:$dst_ptr,
                         LLVM_AnyVector:$offset);
    string llvmBuilder = [{
        auto *mod = builder.GetInsertBlock()->getModule();
        auto *vTy = builder.getVoidTy();
        auto *i32Ty = builder.getInt32Ty();
        auto *fTy = llvm::FunctionType::get(vTy, {i32Ty, i32Ty, $dst_ptr->getType(), i32Ty}, false);
        auto fn = mod->getOrInsertFunction("llvm.xpu.vscatter_mh", fTy);
        builder.CreateCall(fTy, fn.getCallee(), {$value, $mask, $dst_ptr, $offset});
    }];
}

//===----------------------------------------------------------------------===//
// XPU vshuffle definitions
//===----------------------------------------------------------------------===//
def XPU_VSHUFFLE2Op : XPU_IntrOp<"vshuffle2", [], 0> {
    let arguments = (ins LLVM_AnyVector:$src);
    let results = (outs LLVM_AnyVector:$res);
    string llvmBuilder = [{
        auto *mod = builder.GetInsertBlock()->getModule();
        auto *i32Ty = builder.getInt32Ty();
        auto *fTy = llvm::FunctionType::get(i32Ty, {i32Ty}, false);
        auto fn = mod->getOrInsertFunction("llvm.xpu.vshuffle2_hf", fTy);
        $res = builder.CreateCall(fTy, fn.getCallee(), {$src});
    }];
}

#endif // LLVMXPUIR_OPS
