#ifndef TT2TTX_CONVERSION_PASSES
#define TT2TTX_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def ConvertTritonToTritonXPU: Pass<"convert-triton-to-triton-xpu", "mlir::ModuleOp"> {
    let summary = "Convert Triton to TritonXPU";
    let description = [{

    }];
    let constructor = "mlir::triton::createConvertTritonToTritonXPUPass()";

    let dependentDialects = ["mlir::arith::ArithDialect",
                             "mlir::math::MathDialect",
                             // TODO: Does this pass depend on SCF?
                             "mlir::scf::SCFDialect",
                             "mlir::triton::TritonDialect",
                             // Yes: this pass stamps the TritonGPU module
                             // attributes and its encodings reference
                             // CTAEncodingAttr, so the dialect has to be
                             // loaded. Without it the pass only works behind
                             // another pass that loaded TritonGPU already --
                             // running it alone in triton-opt died with
                             // "can't create Attribute
                             // 'mlir::triton::gpu::CTAEncodingAttr' because
                             // storage uniquer isn't initialized".
                             "mlir::triton::gpu::TritonGPUDialect",
                             "mlir::triton::xpu::TritonXPUDialect"];

   let options = [
       Option<"xpu_arch", "xpu_arch",
               "uint32_t", /*default*/"3",
               "XPU Architecture">,
       Option<"buffer_size", "buffer_size",
              "uint32_t", /*default*/"512",
              "bytes for local memory buffer">,
       Option<"core_num", "core_num",
              "uint32_t", /*default*/"64",
              "xpu spec core_num">,
       Option<"isTLE", "is-tle",
              "bool", /*default*/"false",
              "enable TLE-only conversion behavior (keeps LM ptr tensors "
              "unencoded); must stay false for the normal pipeline">
   ];
}


#endif // TT2TTX_CONVERSION_PASSES
