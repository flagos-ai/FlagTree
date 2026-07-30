#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

namespace opts {

// command line option for the type of sanitizer
static cl::opt<std::string>
    SanitizerType("sanitizer-type",
                  cl::desc("Type of sanitizer being used: AddressSanitizer = "
                           "asan, ThreadSanitizer = tsan"),
                  cl::value_desc("string"));

} // namespace opts

namespace {

struct SanitizerAttributes : PassInfoMixin<SanitizerAttributes> {
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &) {
    // add an attribute to the function depending on the type of sanitizer
    if (opts::SanitizerType == "asan") {
      F.addFnAttr(Attribute::SanitizeAddress);
    } else if (opts::SanitizerType == "tsan") {
      F.addFnAttr(Attribute::SanitizeThread);
    }

    // this pass modifies all function attributes
    return PreservedAnalyses::none();
  }

  // Without isRequired returning true, this pass will be skipped for functions
  // decorated with the optnone LLVM attribute. Note that clang -O0 decorates
  // all functions with optnone.
  static bool isRequired() { return true; }
};
} // namespace

llvm::PassPluginLibraryInfo getSanitizerAttributesPluginInfo() {
  return {LLVM_PLUGIN_API_VERSION, "SanitizerAttributes", LLVM_VERSION_STRING,
          [](PassBuilder &PB) {
            PB.registerPipelineParsingCallback(
                [](StringRef Name, FunctionPassManager &FPM,
                   ArrayRef<PassBuilder::PipelineElement>) {
                  if (Name == "sanitizer-attributes") {
                    FPM.addPass(SanitizerAttributes());
                    return true;
                  }
                  return false;
                });
          }};
}

extern "C" LLVM_ATTRIBUTE_WEAK ::llvm::PassPluginLibraryInfo
llvmGetPassPluginInfo() {
  return getSanitizerAttributesPluginInfo();
}
