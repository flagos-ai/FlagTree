#ifndef ILUVATAR_TRITON_CONVERSION_TRITONGPU_TO_DECOMPOSEUNSUPPORTEDCONVERSIONS_H
#define ILUVATAR_TRITON_CONVERSION_TRITONGPU_TO_DECOMPOSEUNSUPPORTEDCONVERSIONS_H

#define FLAGTREE_SPEC_Conversion_TritonGPUToLLVM_DecomposeUnsupportedConversions \
  template void decomposeTensorCoreToDotLayoutConversion<                        \
      triton::gpu::IluvatarMmaEncodingAttr>(ModuleOp, ShortcutFn)

#endif // ILUVATAR_TRITON_CONVERSION_TRITONGPU_TO_DECOMPOSEUNSUPPORTEDCONVERSIONS_H
