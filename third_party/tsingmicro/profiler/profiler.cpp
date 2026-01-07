#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Support/Alignment.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/SourceMgr.h"
#include <cassert>
#include <fstream>
#include <map>
#include <string>
using namespace llvm;

#define DEBUG_TYPE "profiler"

const static std::string kernel_start = "kernel_s";
const static std::string kernel_end = "kernel_e";

static std::map<uint32_t, std::string> orderDesc;

cl::OptionCategory ProfileCommon("Common profile options");
static cl::OptionCategory *ProfileCategories[] = {&ProfileCommon};

cl::list<std::string>
    TracePoints("trace-points", cl::CommaSeparated,
                cl::desc("Function name which need do profiling tracing"),
                cl::value_desc("func1,func2,func3,..."), cl::Prefix,
                cl::cat(ProfileCommon));

cl::opt<std::string> InputIR(cl::Positional, cl::desc("The input IR file"),
                             cl::cat(ProfileCommon));

cl::opt<std::string> OutFile("o", cl::desc("Set the output IR file"),
                             cl::value_desc("filename"),
                             cl::init("/tmp/temp.ll"), cl::cat(ProfileCommon));

cl::opt<int> Index("index", cl::desc("The ir index to be processed"),
                   cl::value_desc("int"), cl::init(0), cl::cat(ProfileCommon));

/// Handling Config Generator command options with LLVM CommandLine facilities
void initCommandLine(int argc, char **argv) {
  // Hide LLVM command options, display Config Generator only command options.
  cl::HideUnrelatedOptions(ArrayRef(ProfileCategories));

  // Parse command line options.
  cl::ParseCommandLineOptions(argc, argv, "Profile command line usage\n\n ");
}

void parse(const char *path, std::unique_ptr<Module> &program,
           LLVMContext &ctx) {
  SMDiagnostic error;

  program = parseIRFile(path, error, ctx);
  if (!program) {
    printf("Failed to parse IR file\n");
    error.print(path, errs());

    exit(-1);
  }
}

void writeOrderDesc(std::map<uint32_t, std::string> desc) {
  std::ofstream outFile("profile_desc.ini");
  if (!outFile.is_open()) {
    printf("Failed to open file: profile_desc.ini\n");
    return;
  }

  outFile << "[" << "profile_order_desc" << "]" << std::endl;
  for (const auto &pair : desc) {
    outFile << "o" << pair.first << "=" << pair.second << std::endl;
  }
  outFile.close();
}

void dump(const char *path, std::unique_ptr<Module> &program) {
  std::string ir;
  raw_string_ostream stream(ir);
  program->print(stream, nullptr);

  std::ofstream output(path);
  output << ir;
  output.close();
  // llvm::errs() << "Dumping IR: " << ir << "\n";
}

void process(const std::unique_ptr<Module> &program, IRBuilder<> &builder) {
  SmallVector<StringRef> tracePoints(TracePoints.begin(), TracePoints.end());
  if (tracePoints.empty()) {
    printf("No trace points specified, only statistic total kernel execution "
           "time\n");
  }

  SmallVector<Function *> functions;
  for (auto &func : program->getFunctionList()) {
    if (func.isDeclaration()) {
      LLVM_DEBUG(llvm::dbgs() << "Function " << func.getName()
                              << " is a declaration, skipping.\n");
      continue;
    }
    functions.push_back(&func);
  }

  assert(functions.size() == 1 && "Support only one triton kernel\n");
  auto triton_kernel = functions[0];

  // Insert profiling trace function in the entry block
  auto EntryBlock = &triton_kernel->getEntryBlock();
  builder.SetInsertPoint(EntryBlock, EntryBlock->begin());
  auto group_id = builder.getInt32(Index);              // Group ID
  auto event_init_value = builder.getInt32(0xFFFFFFFF); // Event ID
  auto event_id_ptr =
      builder.CreateAlloca(builder.getInt32Ty(), nullptr, "event_id");
  // Store event init value
  builder.CreateStore(event_init_value, event_id_ptr);
  const auto add_profile_trace_point = program->getFunction("addOrderProfile");
  const auto tsm_wait_finish_point = program->getFunction("TsmWaitfinish");
  assert(add_profile_trace_point &&
         "Function 'addOrderProfile' not found in the module");
  int order_id = 0;

  orderDesc.insert({order_id, kernel_start});
  builder.CreateCall(add_profile_trace_point,
                     {group_id, builder.getInt8(order_id++), event_id_ptr});

  // Insert profiling trace function for each trace point
  bool isAddWait = false;
  for (auto func : tracePoints) {
    const auto function = program->getFunction(func);

    if (!function) {
      printf("Function '%s' not found.\n", func.data());
      continue;
    }
    for (const auto &user : function->users()) {
      // 确保该引用实际上是一条调用指令
      if (!isa<CallInst>(user))
        continue;
      const auto call_instruction = cast<CallInst>(user);
      builder.SetInsertPoint(call_instruction);

      orderDesc.insert({order_id, std::string(func.data()) + "_s"});
      builder.CreateCall(add_profile_trace_point,
                         {group_id, builder.getInt8(order_id++), event_id_ptr});
      builder.SetInsertPoint(call_instruction->getNextNode());

      builder.CreateCall(tsm_wait_finish_point);
      orderDesc.insert({order_id, std::string(func.data()) + "_e"});
      builder.CreateCall(add_profile_trace_point,
                         {group_id, builder.getInt8(order_id++), event_id_ptr});
      isAddWait = true;
    }
  }

  // Insert print function in the exit block
  const auto print_order_by_event = program->getFunction("printOrderByEvent");
  assert(print_order_by_event &&
         "Function 'printOrderByEvent' not found in the module");
  auto EndBlock = &triton_kernel->back();
  builder.SetInsertPoint(EndBlock->getTerminator());

  if (!isAddWait) {
    builder.CreateCall(tsm_wait_finish_point);
  }
  orderDesc.insert({order_id, kernel_end});
  builder.CreateCall(add_profile_trace_point,
                     {group_id, builder.getInt8(order_id++), event_id_ptr});

  builder.CreateCall(print_order_by_event, {group_id, event_id_ptr});
}

void create_add_order_profile(const std::unique_ptr<Module> &program,
                              IRBuilder<> &builder) {
  std::vector<Type *> args = {
      builder.getInt32Ty() /* groupId */, builder.getInt8Ty() /* orderId */,
      PointerType::get(builder.getInt32Ty(), 0) /* eventId */};
  auto function_type = FunctionType::get(builder.getInt64Ty(), args, false);

  program->getOrInsertFunction("addOrderProfile", function_type);
}

void create_print_profile(const std::unique_ptr<Module> &program,
                          IRBuilder<> &builder) {
  // void printOrderByEvent(uint32_t groupId, uint32_t *eventId)
  std::vector<Type *> args = {
      builder.getInt32Ty() /* groupId */,
      PointerType::get(builder.getInt32Ty(), 0) /* eventId */
  };
  auto function_type = FunctionType::get(builder.getVoidTy(), args, false);

  program->getOrInsertFunction("printOrderByEvent", function_type);
}

void create_tsm_wait_finish(const std::unique_ptr<Module> &program,
                            IRBuilder<> &builder) {
  // uint8_t TsmWaitfinish();
  std::vector<Type *> args = {};
  auto function_type = FunctionType::get(builder.getInt8Ty(), args, false);

  program->getOrInsertFunction("TsmWaitfinish", function_type);
}

int main(int argc, char *argv[]) {
  initCommandLine(argc, argv);
  orderDesc.clear();

  LLVMContext context;
  std::unique_ptr<Module> program = nullptr;
  parse(InputIR.c_str(), program, context);

  LLVM_DEBUG(llvm::dbgs() << "Loaded IR: "
                          << program->getModuleIdentifier().data() << "\n");
  // dump(OutFile.c_str(), program);
  IRBuilder builder(context);

  create_add_order_profile(program, builder);
  create_print_profile(program, builder);
  create_tsm_wait_finish(program, builder);
  process(program, builder);

  LLVM_DEBUG(llvm::dbgs() << "Verification: " << verifyModule(*program, &dbgs())
                          << "\n");
  dump(OutFile.c_str(), program);

  return 0;
}
