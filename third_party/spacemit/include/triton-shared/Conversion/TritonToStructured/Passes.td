#ifndef TRITON_TO_STRUCTURED_CONVERSION_PASSES
#define TRITON_TO_STRUCTURED_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def TritonToStructured : Pass<"triton-to-structured", "mlir::ModuleOp"> {
  let summary = "Convert Triton non-block pointer to TritonStructured dialect";
  let constructor = "triton::createTritonToStructuredPass()";
  let options = [
      Option<"enableMakeGatherScatterTensorPtr", "enable-make-gather-scatter", "bool", /*default*/"true",
             "Enable make_gather_scatter_tptr support">,
      Option<"runPrepassOnly", "run-prepass-only", "bool", /*default*/"false",
             "Only run the pre-processing pass which inserts tts.get_structured_state ops used in scf.for">,
      Option<"skipPrepass", "skip-prepass", "bool", /*default*/"false",
             "Skip the prepass">,
      Option<"useUnsafeMask", "use-unsafe-mask", "bool", /*default*/"false",
             "Assume that the mask bounds are never less than starting offsets. May produce incorrect results.">
  ];
}

#endif
