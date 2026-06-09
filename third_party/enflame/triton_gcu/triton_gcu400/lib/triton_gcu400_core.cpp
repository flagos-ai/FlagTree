/**
 * Copyright 2024-2026 Enflame. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *  http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// triton_gcu400_core.cpp -- In-process replacement for triton-gcu400-opt.
//
// Same architecture as gcu300, with extra GCUWS dialect registration.

#include "triton_gcu400_core.h"

#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "Dialect/GCUWS/IR/Dialect.h"
#include "RegisterGCUDialects.h"

#include "triton/Conversion/TritonToTritonGPU/Passes.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"

#ifdef ENABLE_TRITON_DISTRIBUTED
#include "TritonDistributed/Dialect/Distributed/IR/Dialect.h"
#include "TritonDistributed/Dialect/SIMT/IR/Dialect.h"
#endif

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"

#include "mlir/IR/Diagnostics.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllExtensions.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

#include "llvm/Support/Debug.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"

static std::once_flag g_passRegFlag;
static void ensurePassesRegistered() {
  std::call_once(g_passRegFlag, []() {
    mlir::triton::gpu::registerTritonGPUPasses();
    mlir::triton::registerConvertTritonToTritonGPUPass();
  });
}

static Gcu400String makeString(const std::string &s) {
  char *buf = static_cast<char *>(std::malloc(s.size() + 1));
  std::memcpy(buf, s.data(), s.size());
  buf[s.size()] = '\0';
  return {buf, s.size()};
}

static Gcu400String nullString() { return {nullptr, 0}; }

// ===== Pipeline Implementation =====

struct Gcu400Pipeline_ {
  std::vector<std::pair<std::string, std::string>> passes;
  std::string lastError;
  bool printGeneric = false;
  bool printAfterAll = false;
  bool disableThreading = false;
  bool printModuleScope = false;
  bool enableTiming = false;

  Gcu400Pipeline_() { ensurePassesRegistered(); }
};

extern "C" {

Gcu400Pipeline gcu400_pipeline_create(void) {
  try {
    return new Gcu400Pipeline_();
  } catch (const std::exception &e) {
    llvm::errs() << "[gcu400_pipeline_create] exception: " << e.what() << "\n";
    return nullptr;
  }
}

void gcu400_pipeline_destroy(Gcu400Pipeline p) { delete p; }

void gcu400_pipeline_add_pass(Gcu400Pipeline p, const char *pass_name,
                              const char *options_str) {
  if (!p || !pass_name)
    return;

  llvm::StringRef name(pass_name);

  if (name == "mlir-print-op-generic") {
    p->printGeneric = true;
    return;
  }
  if (name == "mlir-print-ir-after-all") {
    p->printAfterAll = true;
    return;
  }
  if (name == "mlir-disable-threading") {
    p->disableThreading = true;
    return;
  }
  if (name == "mlir-print-ir-module-scope") {
    p->printModuleScope = true;
    return;
  }
  if (name == "mlir-timing") {
    p->enableTiming = true;
    return;
  }
  if (name == "mlir-timing-display") {
    return;
  }

  p->passes.emplace_back(name.str(), options_str ? options_str : std::string());
}

Gcu400String gcu400_pipeline_run(Gcu400Pipeline p, const char *input,
                                 size_t input_len) {
  if (!p)
    return nullString();
  p->lastError.clear();

  mlir::DialectRegistry registry;
  mlir::gcu::registerGCUDialects(registry);
  registry
      .insert<mlir::triton::TritonDialect, mlir::triton::gpu::TritonGPUDialect,
              mlir::triton::gcuws::GCUWSDialect>();
#ifdef ENABLE_TRITON_DISTRIBUTED
  registry.insert<mlir::triton::distributed::DistributedDialect,
                  mlir::triton::simt::SIMTDialect>();
#endif

  mlir::MLIRContext ctx(registry);
  ctx.loadAllAvailableDialects();
  if (p->disableThreading)
    ctx.disableMultithreading();

  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(
      llvm::MemoryBuffer::getMemBuffer(llvm::StringRef(input, input_len),
                                       "<input>", false),
      llvm::SMLoc());

  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceFile<mlir::ModuleOp>(sourceMgr, &ctx);
  if (!module) {
    p->lastError = "Failed to parse input MLIR";
    return nullString();
  }

  mlir::PassManager pm(&ctx, "builtin.module");
  if (p->printAfterAll)
    pm.enableIRPrinting(
        /*shouldPrintBeforePass=*/{},
        /*shouldPrintAfterPass=*/
        [](mlir::Pass *, mlir::Operation *) { return true; },
        p->printModuleScope);
  if (p->enableTiming)
    pm.enableTiming();

  for (const auto &[name, opts] : p->passes) {
    std::string spec(name);
    if (!opts.empty())
      spec += "{" + opts + "}";
    if (spec.find('(') != std::string::npos) {
      if (mlir::failed(mlir::parsePassPipeline(spec, pm))) {
        p->lastError = "Failed to add pass: " + spec;
        llvm::errs() << "[gcu400_pipeline_run] " << p->lastError << "\n";
        return nullString();
      }
      continue;
    }
    if (mlir::succeeded(mlir::parsePassPipeline(spec, pm, llvm::nulls())))
      continue;
    std::string wrapped = "any(" + spec + ")";
    if (mlir::failed(mlir::parsePassPipeline(wrapped, pm))) {
      p->lastError = "Failed to add pass: " + spec;
      llvm::errs() << "[gcu400_pipeline_run] " << p->lastError << "\n";
      return nullString();
    }
  }

  if (mlir::failed(pm.run(module.get()))) {
    p->lastError = "PassManager execution failed";
    return nullString();
  }

  std::string output;
  llvm::raw_string_ostream os(output);
  mlir::OpPrintingFlags flags;
  if (p->printGeneric)
    flags.printGenericOpForm();
  module->print(os, flags);
  return makeString(output);
}

const char *gcu400_pipeline_last_error(Gcu400Pipeline p) {
  return p ? p->lastError.c_str() : "null pipeline";
}

static Gcu400String runOptOnChunk(llvm::StringRef chunk,
                                  const std::vector<std::string> &passArgs,
                                  bool printGeneric, bool printAfterAll,
                                  bool disableThreading, bool printModuleScope,
                                  bool enableTiming,
                                  bool allowUnregisteredDialects = false,
                                  bool verifyDiagnostics = false) {
  mlir::DialectRegistry registry;
  mlir::gcu::registerGCUDialects(registry);
  registry
      .insert<mlir::triton::TritonDialect, mlir::triton::gpu::TritonGPUDialect,
              mlir::triton::gcuws::GCUWSDialect>();
#ifdef ENABLE_TRITON_DISTRIBUTED
  registry.insert<mlir::triton::distributed::DistributedDialect,
                  mlir::triton::simt::SIMTDialect>();
#endif

  mlir::MLIRContext ctx(registry);
  ctx.loadAllAvailableDialects();
  if (disableThreading)
    ctx.disableMultithreading();
  if (allowUnregisteredDialects)
    ctx.allowUnregisteredDialects(true);
  if (verifyDiagnostics)
    ctx.printOpOnDiagnostic(false);

  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(
      llvm::MemoryBuffer::getMemBuffer(chunk, "<input>", false), llvm::SMLoc());

  std::unique_ptr<mlir::SourceMgrDiagnosticVerifierHandler> verifyHandler;
  if (verifyDiagnostics)
    verifyHandler = std::make_unique<mlir::SourceMgrDiagnosticVerifierHandler>(
        sourceMgr, &ctx);

  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceFile<mlir::ModuleOp>(sourceMgr, &ctx);
  if (!module) {
    if (verifyHandler) {
      if (mlir::succeeded(verifyHandler->verify()))
        return makeString("");
    }
    return nullString();
  }

  mlir::PassManager pm(&ctx, "builtin.module");
  if (printAfterAll)
    pm.enableIRPrinting(
        /*shouldPrintBeforePass=*/{},
        /*shouldPrintAfterPass=*/
        [](mlir::Pass *, mlir::Operation *) { return true; }, printModuleScope);
  if (enableTiming)
    pm.enableTiming();

  for (const auto &arg : passArgs) {
    std::string normalized = arg;
    if (normalized.find('{') == std::string::npos) {
      auto eq = normalized.find('=');
      if (eq != std::string::npos)
        normalized =
            normalized.substr(0, eq) + "{" + normalized.substr(eq + 1) + "}";
    }
    if (normalized.find('(') != std::string::npos) {
      if (mlir::failed(mlir::parsePassPipeline(normalized, pm))) {
        llvm::errs() << "[gcu400_run_opt] failed to parse pass pipeline: "
                     << normalized << "\n";
        return nullString();
      }
      continue;
    }
    if (mlir::succeeded(mlir::parsePassPipeline(normalized, pm, llvm::nulls())))
      continue;
    std::string wrapped = "any(" + normalized + ")";
    if (mlir::failed(mlir::parsePassPipeline(wrapped, pm))) {
      llvm::errs() << "[gcu400_run_opt] failed to parse pass pipeline: "
                   << normalized << "\n";
      return nullString();
    }
  }

  bool passesOk = mlir::succeeded(pm.run(module.get()));

  if (verifyHandler) {
    bool verifyOk = mlir::succeeded(verifyHandler->verify());
    if (!verifyOk)
      return nullString();
  } else if (!passesOk) {
    return nullString();
  }

  std::string output;
  llvm::raw_string_ostream os(output);
  mlir::OpPrintingFlags flags;
  if (printGeneric)
    flags.printGenericOpForm();
  module->print(os, flags);
  return makeString(output);
}

Gcu400String gcu400_run_opt(const char *input, size_t input_len,
                            const char *const *args, int num_args) {
  ensurePassesRegistered();

  bool printGeneric = false;
  bool printAfterAll = false;
  bool disableThreading = false;
  bool printModuleScope = false;
  bool enableTiming = false;
  bool splitInputFile = false;
  bool allowUnregisteredDialects = false;
  bool verifyDiagnostics = false;
  std::vector<std::string> passArgs;

  for (int i = 0; i < num_args; ++i) {
    llvm::StringRef a(args[i]);
#ifndef NDEBUG
    if (a == "-debug" || a == "--debug") {
      llvm::DebugFlag = true;
      continue;
    }
    if (a.starts_with("-debug-only=") || a.starts_with("--debug-only=")) {
      llvm::DebugFlag = true;
      llvm::StringRef val = a.substr(a.find('=') + 1);
      llvm::setCurrentDebugType(val.data());
      continue;
    }
#endif
    if (a == "-mlir-print-op-generic") {
      printGeneric = true;
      continue;
    }
    if (a == "-mlir-print-ir-after-all") {
      printAfterAll = true;
      continue;
    }
    if (a == "-mlir-disable-threading") {
      disableThreading = true;
      continue;
    }
    if (a == "-mlir-print-ir-module-scope") {
      printModuleScope = true;
      continue;
    }
    if (a.starts_with("-mlir-timing") || a.starts_with("--mlir-timing")) {
      enableTiming = true;
      continue;
    }
    if (a == "-split-input-file" || a == "--split-input-file") {
      splitInputFile = true;
      continue;
    }
    if (a == "-allow-unregistered-dialect" ||
        a == "--allow-unregistered-dialect") {
      allowUnregisteredDialects = true;
      continue;
    }
    if (a == "-verify-diagnostics" || a == "--verify-diagnostics") {
      verifyDiagnostics = true;
      continue;
    }
    if (a.starts_with("-mlir-print-ir-before"))
      continue;
    llvm::StringRef stripped = a;
    stripped.consume_front("--");
    stripped.consume_front("-");
    passArgs.emplace_back(stripped.str());
  }

  llvm::StringRef fullInput(input, input_len);

  if (!splitInputFile) {
    return runOptOnChunk(fullInput, passArgs, printGeneric, printAfterAll,
                         disableThreading, printModuleScope, enableTiming,
                         allowUnregisteredDialects, verifyDiagnostics);
  }

  constexpr llvm::StringLiteral kSplitMarker("// -----");
  std::string combined;
  bool first = true;
  llvm::StringRef remaining = fullInput;

  while (!remaining.empty()) {
    auto [chunk, rest] = remaining.split(kSplitMarker);
    remaining = rest;

    llvm::StringRef trimmed = chunk.trim();
    if (trimmed.empty())
      continue;

    Gcu400String result =
        runOptOnChunk(chunk, passArgs, printGeneric, printAfterAll,
                      disableThreading, printModuleScope, enableTiming,
                      allowUnregisteredDialects, verifyDiagnostics);
    if (!result.data)
      return nullString();

    if (!first)
      combined += "\n// -----\n";
    combined.append(result.data, result.len);
    gcu400_string_free(result);
    first = false;
  }

  return makeString(combined);
}

void gcu400_string_free(Gcu400String s) {
  std::free(const_cast<char *>(s.data));
}

// ---------------------------------------------------------------------------
// gcu400_opt_main -- full MlirOptMain entry point
//
// MLIR standard passes are registered through MLIRRegisterAllPasses linked
// into this .so (static initializers).  Only GCU / Triton-specific passes
// need explicit registration here.
// ---------------------------------------------------------------------------

int gcu400_opt_main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  mlir::gcu::registerGCUDialects(registry);
  registry
      .insert<mlir::triton::TritonDialect, mlir::triton::gpu::TritonGPUDialect,
              mlir::triton::gcuws::GCUWSDialect>();
#ifdef ENABLE_TRITON_DISTRIBUTED
  registry.insert<mlir::triton::distributed::DistributedDialect,
                  mlir::triton::simt::SIMTDialect>();
#endif
  mlir::registerAllExtensions(registry);

  static std::once_flag optRegFlag;
  std::call_once(optRegFlag, []() {
    mlir::triton::gpu::registerTritonGPUPasses();
    mlir::triton::registerConvertTritonToTritonGPUPass();
  });

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "GCU400 optimizer driver\n", registry));
}

} // extern "C"
