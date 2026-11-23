//===- HIVMAnalysis.cpp - HIVM performance analysis ----------------------===//
//
// This implementation consumes HIVM through MLIR-native ingestion and
// schedules the resulting execution graph with either a static scheduler or
// a discrete-event simulator.
//
//===----------------------------------------------------------------------===//

#include "AscendModel/Analysis/HIVMAnalysis.h"

#ifdef TRITONSIM_HAS_BISHENGIR_HIVM
#include "bishengir/Dialect/Annotation/IR/Annotation.h"
#include "bishengir/Dialect/HACC/IR/HACC.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#endif

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/Parser/Parser.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstring>
#include <limits>
#include <optional>
#include <queue>
#include <sstream>
#include <tuple>

using namespace mlir::ascend;

namespace {

struct ParsedOp {
  HIVMOp op;
  std::string definedValue;
  std::vector<std::string> operandValues;
  std::vector<mlir::Value> mlirResults;
  std::vector<HIVMPipe> barrierPipes;
  std::string syncCoreType;
  std::string senderEvent;
  std::string receiverEvent;
  std::string eventId;
  mlir::Value syncIdValue;
  HIVMPipe senderPipe = HIVMPipe::Unknown;
  HIVMPipe receiverPipe = HIVMPipe::Unknown;
};

static int64_t estimateMmadL1MTE1Cycles(const ParsedOp &parsed,
                                        const HardwareConfig &config);

static int64_t estimateND2NZCycles(const ParsedOp &parsed,
                                   const HardwareConfig &config);

static std::vector<HIVMOp> expandMacroOp(const ParsedOp &parsed,
                                         const HardwareConfig &config) {
  std::vector<HIVMOp> ops;
  llvm::StringRef name = parsed.op.opName;
  if (name == "mmadL1") {
    // MTE1 (L1 -> L0A/L0B) and Cube (compute on L0) are on different pipes
    // and overlap in the hardware pipeline.  Each gets its own duration.
    HIVMOp mte = parsed.op;
    mte.opName = "mmadL1.mte1";
    mte.pipe = HIVMPipe::MTE1;
    mte.duration = estimateMmadL1MTE1Cycles(parsed, config);
    mte.isSyncOp = false;
    mte.isBarrier = false;

    HIVMOp cube = parsed.op;
    cube.opName = "mmadL1.cube";
    cube.pipe = HIVMPipe::Cube;
    // Cube gets the full compute duration (startup + tile cycles), NOT the
    // leftover after subtracting MTE1.  The two pipes overlap.
    cube.duration = std::max<int64_t>(1, parsed.op.duration);
    cube.isSyncOp = false;
    cube.isBarrier = false;
    ops.push_back(std::move(mte));
    ops.push_back(std::move(cube));
    return ops;
  }

  if (name == "matmul" || name == "mix_matmul" || name == "mix_group_matmul") {
    int64_t preload = std::max<int64_t>(1, config.getMTE2StartupLatency());
    int64_t drain = std::max<int64_t>(1, config.getMTE3StartupLatency());
    int64_t compute =
        std::max<int64_t>(1, parsed.op.duration - preload - drain);

    HIVMOp mte2 = parsed.op;
    mte2.opName = name.str() + ".mte2";
    mte2.pipe = HIVMPipe::CubeMTE2;
    mte2.duration = preload;
    mte2.isSyncOp = false;
    mte2.isBarrier = false;

    HIVMOp cube = parsed.op;
    cube.opName = name.str() + ".cube";
    cube.pipe = HIVMPipe::Cube;
    cube.duration = compute;
    cube.isSyncOp = false;
    cube.isBarrier = false;

    HIVMOp mte3 = parsed.op;
    mte3.opName = name.str() + ".mte3";
    mte3.pipe = HIVMPipe::MTE3;
    mte3.duration = drain;
    mte3.isSyncOp = false;
    mte3.isBarrier = false;

    ops.push_back(std::move(mte2));
    ops.push_back(std::move(cube));
    ops.push_back(std::move(mte3));
    return ops;
  }

  ops.push_back(parsed.op);
  return ops;
}

static void attachSyncMetadata(ParsedOp &parsed) {
  parsed.op.senderPipe = parsed.senderPipe;
  parsed.op.receiverPipe = parsed.receiverPipe;
  parsed.op.eventId = parsed.eventId;
}

struct EventKey {
  HIVMPipe sender = HIVMPipe::Unknown;
  HIVMPipe receiver = HIVMPipe::Unknown;
  std::string eventId;

  bool operator<(const EventKey &other) const {
    return std::tie(sender, receiver, eventId) <
           std::tie(other.sender, other.receiver, other.eventId);
  }
};

struct EventInstanceKey {
  EventKey key;
  int64_t generation = 0;

  bool operator<(const EventInstanceKey &other) const {
    return std::tie(key.sender, key.receiver, key.eventId, generation) <
           std::tie(other.key.sender, other.key.receiver, other.key.eventId,
                    other.generation);
  }
};

struct LoopFrame {
  int braceDepth = 0;
  int64_t tripCount = 1;
};

struct AnalysisState {
  llvm::DenseMap<mlir::Value, int64_t> constants;
  llvm::DenseMap<mlir::Value, int64_t> boundValues;
  llvm::DenseMap<mlir::Value, size_t> valueProducers;
  llvm::DenseMap<mlir::Value, std::string> bufferRoots;
  std::map<std::string, int64_t> bufferSlots;
  std::map<std::string, int64_t> bufferVersions;
  std::map<EventKey, size_t> eventProducers;
  std::map<EventKey, int64_t> eventGenerations;
  std::map<HIVMPipe, size_t> latestPipeProducer;
  std::map<std::string, int64_t> argBindings;
};

static llvm::StringRef trim(llvm::StringRef s) { return s.trim(); }

static int64_t ceilDiv(int64_t num, int64_t den) {
  if (den <= 0)
    return 0;
  return (num + den - 1) / den;
}

/// Resolve an ambiguous core type (empty or "CUBE_OR_VECTOR") by inspecting the
/// enclosing func.func's name for AIC/AIV markers.
static llvm::StringRef resolveCoreTypeFromFunc(mlir::Operation *op,
                                               llvm::StringRef current) {
  if (current == "CUBE" || current == "AIC" || current == "VECTOR" ||
      current == "AIV")
    return current;
  if (auto parentFunc = op->getParentOfType<mlir::func::FuncOp>()) {
    llvm::StringRef funcName = parentFunc.getName();
    if (funcName.contains("aic") || funcName.contains("AIC") ||
        funcName.contains("cube"))
      return "CUBE";
    if (funcName.contains("aiv") || funcName.contains("AIV") ||
        funcName.contains("vector") || funcName.contains("mix"))
      return "VECTOR";
  }
  return current;
}

static bool isVectorDomainPipe(HIVMPipe pipe) {
  return pipe == HIVMPipe::Vector || pipe == HIVMPipe::VectorMTE2 ||
         pipe == HIVMPipe::MTE3;
}

static bool isCubeDomainPipe(HIVMPipe pipe) {
  return pipe == HIVMPipe::Cube || pipe == HIVMPipe::CubeMTE2 ||
         pipe == HIVMPipe::MTE1 || pipe == HIVMPipe::FixPipe;
}

static HIVMPipe disambiguateMTE2Pipe(HIVMPipe pipe, HIVMPipe peerPipe,
                                     llvm::StringRef coreType) {
  if (pipe != HIVMPipe::VectorMTE2)
    return pipe;
  if (isVectorDomainPipe(peerPipe))
    return HIVMPipe::VectorMTE2;
  if (isCubeDomainPipe(peerPipe))
    return HIVMPipe::CubeMTE2;
  if (coreType == "VECTOR" || coreType == "AIV")
    return HIVMPipe::VectorMTE2;
  if (coreType == "CUBE" || coreType == "AIC")
    return HIVMPipe::CubeMTE2;
  return HIVMPipe::VectorMTE2;
}

static HIVMPipe selectMTE2PipeForSpaces(llvm::StringRef srcSpace,
                                        llvm::StringRef dstSpace,
                                        llvm::StringRef coreType) {
  if (dstSpace == "ub" || srcSpace == "ub")
    return HIVMPipe::VectorMTE2;
  if (dstSpace == "l1" || dstSpace == "l0a" || dstSpace == "l0b" ||
      dstSpace == "l0c" || srcSpace == "l1" || srcSpace == "l0a" ||
      srcSpace == "l0b" || srcSpace == "l0c")
    return HIVMPipe::CubeMTE2;
  if (coreType == "CUBE" || coreType == "AIC")
    return HIVMPipe::CubeMTE2;
  return HIVMPipe::VectorMTE2;
}

static int64_t getElementByteWidth(llvm::StringRef typeToken) {
  llvm::StringRef t = trim(typeToken);
  if (t == "f16" || t == "bf16" || t == "i16")
    return 2;
  if (t == "f32" || t == "i32")
    return 4;
  if (t == "f64" || t == "i64")
    return 8;
  if (t == "i8" || t == "ui8" || t == "si8")
    return 1;
  return 0;
}

static int64_t parseMemRefElementCount(llvm::StringRef line) {
  size_t memrefPos = line.find("memref<");
  if (memrefPos == llvm::StringRef::npos)
    return 0;
  size_t addrPos = line.find(", #hivm.address_space<", memrefPos);
  if (addrPos == llvm::StringRef::npos)
    return 0;
  llvm::StringRef shapeAndType = line.slice(memrefPos + 7, addrPos);
  llvm::SmallVector<llvm::StringRef, 8> parts;
  shapeAndType.split(parts, 'x', -1, false);
  if (parts.empty())
    return 0;

  int64_t count = 1;
  for (size_t i = 0; i + 1 < parts.size(); ++i) {
    llvm::StringRef dim = trim(parts[i]);
    if (dim == "?" || dim.empty())
      return 0;
    int64_t value = 0;
    if (dim.getAsInteger(10, value))
      return 0;
    count *= value;
  }
  return count;
}

static int64_t parseMemRefBytes(llvm::StringRef line) {
  size_t memrefPos = line.find("memref<");
  if (memrefPos == llvm::StringRef::npos)
    return 0;
  size_t addrPos = line.find(", #hivm.address_space<", memrefPos);
  if (addrPos == llvm::StringRef::npos)
    return 0;
  llvm::StringRef shapeAndType = line.slice(memrefPos + 7, addrPos);
  llvm::SmallVector<llvm::StringRef, 8> parts;
  shapeAndType.split(parts, 'x', -1, false);
  if (parts.empty())
    return 0;

  llvm::StringRef elemType = trim(parts.back());
  int64_t elemBytes = getElementByteWidth(elemType);
  if (elemBytes <= 0)
    return 0;

  int64_t count = 1;
  for (size_t i = 0; i + 1 < parts.size(); ++i) {
    llvm::StringRef dim = trim(parts[i]);
    if (dim == "?" || dim.empty())
      return 0;
    int64_t value = 0;
    if (dim.getAsInteger(10, value))
      return 0;
    count *= value;
  }

  return count * elemBytes;
}

static int64_t estimateMmadL1MTE1Cycles(const ParsedOp &parsed,
                                        const HardwareConfig &config) {
  int64_t inputBytes = 0;
  for (const std::string &buffer : parsed.op.readBuffers)
    inputBytes += parseMemRefBytes(buffer);
  if (inputBytes <= 0)
    inputBytes = std::max<int64_t>(parsed.op.bytes, 1);

  double bandwidth = config.getMemoryBandwidthBytesPerCycle("mte1");
  if (bandwidth <= 0.0)
    bandwidth = std::max(1.0, config.getMemoryBandwidthBytesPerCycle("l1"));
  int64_t transferCycles = std::max<int64_t>(
      1, static_cast<int64_t>(std::ceil(inputBytes / bandwidth)));
  int64_t startupCycles =
      std::max<int64_t>(4, config.getMTE2StartupLatency() / 5);
  return startupCycles + transferCycles;
}

static int64_t estimateND2NZCycles(const ParsedOp &parsed,
                                   const HardwareConfig &config) {
  int64_t bytes = std::max<int64_t>(parsed.op.bytes, 1);
  // nd2nz is not a plain HBM->L1 DMA. The layout conversion runs on the
  // cube-side transfer path but sustains lower throughput than a normal
  // cube_mte2 transport, so keep it on a dedicated calibration path.
  double baseBandwidth = config.getMemoryBandwidthBytesPerCycle("cube_mte2");
  if (baseBandwidth <= 0.0)
    baseBandwidth =
        std::max(1.0, config.getMemoryBandwidthBytesPerCycle("hbm"));
  double effectiveBandwidth = std::max(1.0, baseBandwidth * 0.5);
  int64_t transferCycles = std::max<int64_t>(
      1, static_cast<int64_t>(std::ceil(bytes / effectiveBandwidth)));
  int64_t startupCycles =
      std::max<int64_t>(16, config.getMTE2StartupLatency() / 3);
  return startupCycles + transferCycles;
}

static std::string canonicalizeAddressSpace(llvm::StringRef space) {
  llvm::StringRef s = trim(space);
  if (s == "gm")
    return "gm";
  if (s == "ub")
    return "ub";
  if (s == "l1" || s == "cbuf")
    return "l1";
  if (s == "ca" || s == "l0a")
    return "l0a";
  if (s == "cb" || s == "l0b")
    return "l0b";
  if (s == "cc" || s == "l0c")
    return "l0c";
  return s.str();
}

#ifdef TRITONSIM_HAS_BISHENGIR_HIVM
static std::string getCanonicalTypeAddressSpace(mlir::Type type) {
  auto memref = llvm::dyn_cast<mlir::MemRefType>(type);
  if (!memref)
    return "";
  mlir::Attribute memSpace = memref.getMemorySpace();
  auto addrAttr =
      llvm::dyn_cast_or_null<mlir::hivm::AddressSpaceAttr>(memSpace);
  if (!addrAttr)
    return "";
  switch (addrAttr.getAddressSpace()) {
  case mlir::hivm::AddressSpace::GM:
    return "gm";
  case mlir::hivm::AddressSpace::L1:
    return "l1";
  case mlir::hivm::AddressSpace::L0A:
    return "l0a";
  case mlir::hivm::AddressSpace::L0B:
    return "l0b";
  case mlir::hivm::AddressSpace::L0C:
    return "l0c";
  case mlir::hivm::AddressSpace::UB:
    return "ub";
  default:
    return "";
  }
}
#endif

static int64_t getTypeByteWidth(mlir::Type type) {
  if (!type)
    return 0;
  if (auto intType = llvm::dyn_cast<mlir::IntegerType>(type))
    return std::max<int64_t>(1, intType.getWidth() / 8);
  if (auto floatType = llvm::dyn_cast<mlir::FloatType>(type))
    return std::max<int64_t>(1, floatType.getWidth() / 8);
  return 0;
}

static int64_t getShapedTypeElementCount(mlir::Type type) {
  auto shaped = llvm::dyn_cast<mlir::ShapedType>(type);
  if (!shaped || !shaped.hasStaticShape())
    return 0;
  int64_t count = 1;
  for (int64_t dim : shaped.getShape())
    count *= dim;
  return count;
}

static int64_t getShapedTypeBytes(mlir::Type type) {
  auto shaped = llvm::dyn_cast<mlir::ShapedType>(type);
  if (!shaped)
    return 0;
  int64_t count = getShapedTypeElementCount(type);
  if (count <= 0)
    return 0;
  return count * getTypeByteWidth(shaped.getElementType());
}

static int64_t inferValueBytes(mlir::Value value) {
  if (!value)
    return 0;
  return getShapedTypeBytes(value.getType());
}

static int64_t inferValueElements(mlir::Value value) {
  if (!value)
    return 0;
  return getShapedTypeElementCount(value.getType());
}

static int getLineNumberFromLocation(mlir::Location loc) {
  return llvm::TypeSwitch<mlir::Location, int>(loc)
      .Case<mlir::FileLineColLoc>(
          [](auto fileLoc) { return static_cast<int>(fileLoc.getLine()); })
      .Case<mlir::NameLoc>([](auto nameLoc) {
        return getLineNumberFromLocation(nameLoc.getChildLoc());
      })
      .Case<mlir::FusedLoc>([](auto fusedLoc) {
        for (mlir::Location child : fusedLoc.getLocations()) {
          int line = getLineNumberFromLocation(child);
          if (line > 0)
            return line;
        }
        return 0;
      })
      .Default([](mlir::Location) { return 0; });
}

static std::string renderOperation(mlir::Operation *op) {
  std::string storage;
  llvm::raw_string_ostream os(storage);
  mlir::OpPrintingFlags flags;
  flags.elideLargeElementsAttrs();
  op->print(os, flags);
  os.flush();
  return storage;
}

static void eraseAttributeAssignment(std::string &line, llvm::StringRef name) {
  std::string needle = name.str();
  for (;;) {
    size_t pos = line.find(needle);
    if (pos == std::string::npos)
      return;

    size_t valueStart = line.find('#', pos + needle.size());
    if (valueStart == std::string::npos)
      return;
    size_t valueEnd = line.find('>', valueStart);
    if (valueEnd == std::string::npos)
      return;
    ++valueEnd;

    size_t eraseStart = pos;
    while (eraseStart > 0 && line[eraseStart - 1] == ' ')
      --eraseStart;
    if (eraseStart > 0 && line[eraseStart - 1] == ',') {
      --eraseStart;
      while (eraseStart > 0 && line[eraseStart - 1] == ' ')
        --eraseStart;
    } else {
      while (valueEnd < line.size() && line[valueEnd] == ' ')
        ++valueEnd;
      if (valueEnd < line.size() && line[valueEnd] == ',')
        ++valueEnd;
      while (valueEnd < line.size() && line[valueEnd] == ' ')
        ++valueEnd;
    }

    line.erase(eraseStart, valueEnd - eraseStart);
  }
}

static std::string sanitizeMlirBuffer(llvm::StringRef buffer) {
  // Pre-process to remove custom dialect attributes/types that require
  // registered dialects.  When built without BiShengIR, the parser cannot
  // handle #hivm.address_space<...>, #hacc.arg_type<...>, etc.
  std::string preprocessed;
  {
    llvm::SmallVector<llvm::StringRef, 0> lines;
    buffer.split(lines, '\n');
    llvm::raw_string_ostream os(preprocessed);
    for (llvm::StringRef line : lines) {
      llvm::StringRef trimmed = line.trim();
      if (trimmed.starts_with("warning: ") ||
          trimmed.ends_with("warning generated."))
        break;
      if (trimmed.starts_with("ld.lld:") || trimmed.starts_with("[ERROR]") ||
          trimmed.starts_with("[WARNING]") || trimmed.starts_with("[INFO]"))
        continue;
      std::string l = line.str();
      // Replace #hivm.address_space<xxx> with integer memory space
      while (auto pos = l.find("#hivm.address_space<")) {
        auto end = l.find('>', pos);
        if (end == std::string::npos)
          break;
        auto space = llvm::StringRef(l).slice(pos + 20, end);
        int num = 0;
        if (space == "gm")
          num = 0;
        else if (space == "ub")
          num = 1;
        else if (space == "l1")
          num = 2;
        else if (space == "l0a")
          num = 3;
        else if (space == "l0b")
          num = 4;
        else if (space == "l0c")
          num = 5;
        else if (space == "cbuf")
          num = 6;
        l.replace(pos, end - pos + 1, std::to_string(num));
      }
      // Strip whole custom attribute assignments whose values require
      // external dialect parsers.
      eraseAttributeAssignment(l, "hacc.arg_type");
      eraseAttributeAssignment(l, "hivm.func_core_type");
      eraseAttributeAssignment(l, "hacc.function_kind");
      os << l << "\n";
    }
    os.flush();
  }

  return preprocessed;
}

static bool startsWithHivmOp(mlir::Operation *op) {
  return op->getName().getStringRef().starts_with("hivm.hir.");
}

static llvm::StringRef getLeafOpName(mlir::Operation *op) {
  llvm::StringRef fullName = op->getName().getStringRef();
  size_t dot = fullName.rfind('.');
  if (dot == llvm::StringRef::npos)
    return fullName;
  return fullName.drop_front(dot + 1);
}

#ifdef TRITONSIM_HAS_BISHENGIR_HIVM
static HIVMPipe convertTypedPipe(mlir::hivm::PIPE pipe) {
  switch (pipe) {
  case mlir::hivm::PIPE::PIPE_V:
    return HIVMPipe::Vector;
  case mlir::hivm::PIPE::PIPE_MTE2:
  case mlir::hivm::PIPE::VIRTUAL_PIPE_MTE2_L1A:
  case mlir::hivm::PIPE::VIRTUAL_PIPE_MTE2_L1B:
    return HIVMPipe::VectorMTE2;
  case mlir::hivm::PIPE::PIPE_MTE3:
    return HIVMPipe::MTE3;
  case mlir::hivm::PIPE::PIPE_S:
    return HIVMPipe::Scalar;
  case mlir::hivm::PIPE::PIPE_FIX:
    return HIVMPipe::FixPipe;
  case mlir::hivm::PIPE::PIPE_M:
    return HIVMPipe::Cube;
  case mlir::hivm::PIPE::PIPE_MTE1:
    return HIVMPipe::MTE1;
  case mlir::hivm::PIPE::PIPE_ALL:
    return HIVMPipe::All;
  default:
    return HIVMPipe::Unknown;
  }
}

static std::string stringifyTypedPipe(mlir::hivm::PIPE pipe) {
  return mlir::hivm::stringifyPIPE(pipe).str();
}

static std::string
stringifyTypedCore(std::optional<mlir::hivm::TCoreType> core) {
  if (!core)
    return "";
  return mlir::hivm::stringifyTCoreType(*core).str();
}

static std::string canonicalizeStaticEventToken(llvm::StringRef token) {
  if (token.consume_front("EVENT_ID"))
    return token.str();
  return token.str();
}

static bool populateTypedHivmOp(mlir::Operation *op, ParsedOp &parsed) {
  auto isTypedCubeOpName = [&](llvm::StringRef opName) {
    return opName == "matmul" || opName == "mix_matmul" ||
           opName == "mix_group_matmul" || opName == "mmadL1";
  };
  parsed.op.opName = getLeafOpName(op).str();

  if (auto pipeIface = llvm::dyn_cast<mlir::hivm::OpPipeInterface>(op)) {
    if (pipeIface.isSinglePipeOp()) {
      parsed.op.pipe = convertTypedPipe(pipeIface.getPipe());
    } else if (!isTypedCubeOpName(parsed.op.opName)) {
      parsed.op.pipe = convertTypedPipe(pipeIface.getOutPipe());
    }
  }
  if (auto coreIface = llvm::dyn_cast<mlir::hivm::CoreTypeInterface>(op))
    parsed.op.coreType = stringifyTypedCore(coreIface.getCoreType());
  if (parsed.op.coreType.empty()) {
    if (auto inferIface =
            llvm::dyn_cast<mlir::hivm::InferCoreTypeInterface>(op))
      parsed.op.coreType = stringifyTypedCore(inferIface.inferCoreType());
  }
  // Resolve "CUBE_OR_VECTOR" to a concrete core type using the enclosing
  // function name.  This is critical for disambiguating PIPE_MTE2 into
  // PIPE_MTE2_C vs PIPE_MTE2_V for pipe_barrier ops.
  parsed.op.coreType = resolveCoreTypeFromFunc(op, parsed.op.coreType).str();

  if (llvm::isa<mlir::hivm::LoadOp>(op))
    parsed.op.pipe = selectMTE2PipeForSpaces(
        getCanonicalTypeAddressSpace(op->getOperand(0).getType()),
        op->getNumOperands() > 1
            ? getCanonicalTypeAddressSpace(op->getOperand(1).getType())
            : llvm::StringRef(),
        parsed.op.coreType);
  else if (llvm::isa<mlir::hivm::StoreOp>(op))
    parsed.op.pipe = HIVMPipe::MTE3;
  else if (llvm::isa<mlir::hivm::FixpipeOp>(op))
    parsed.op.pipe = HIVMPipe::FixPipe;
  else if (parsed.op.opName == "nd2nz")
    parsed.op.pipe = HIVMPipe::CubeMTE2;
  else if (parsed.op.opName == "nz2nd")
    parsed.op.pipe = HIVMPipe::MTE3;
  else if (auto copyOp = llvm::dyn_cast<mlir::hivm::CopyOp>(op)) {
    std::string srcSpace =
        getCanonicalTypeAddressSpace(copyOp.getSrc().getType());
    std::string dstSpace =
        getCanonicalTypeAddressSpace(copyOp.getDst().getType());
    if (srcSpace == "ub" && dstSpace == "l1")
      parsed.op.pipe = HIVMPipe::MTE3;
    else if (srcSpace == "gm" && dstSpace == "l1")
      parsed.op.pipe = HIVMPipe::CubeMTE2;
    else if (srcSpace == "l0c" && dstSpace == "gm")
      parsed.op.pipe = HIVMPipe::FixPipe;
    else if (srcSpace == "ub" && dstSpace == "ub")
      parsed.op.pipe = HIVMPipe::Vector;
  } else if (parsed.op.opName == "convert_layout" ||
             parsed.op.opName == "pointer_cast")
    parsed.op.pipe = HIVMPipe::Unknown;
  else if (llvm::isa<mlir::hivm::MmadL1Op, mlir::hivm::MatmulOp,
                     mlir::hivm::MixMatmulOp, mlir::hivm::MixGroupMatmulOp>(op))
    parsed.op.pipe = HIVMPipe::Cube;

  if (auto barrier = llvm::dyn_cast<mlir::hivm::PipeBarrierOp>(op)) {
    parsed.op.opName = "pipe_barrier";
    parsed.op.isSyncOp = true;
    parsed.op.isBarrier = true;
    HIVMPipe rawPipe = convertTypedPipe(barrier.getPipe().getPipe());
    parsed.op.pipe =
        disambiguateMTE2Pipe(rawPipe, HIVMPipe::Unknown, parsed.op.coreType);
    parsed.barrierPipes.push_back(parsed.op.pipe);
    return true;
  }
  if (auto setFlag = llvm::dyn_cast<mlir::hivm::SetFlagOp>(op)) {
    parsed.op.opName = "set_flag";
    parsed.op.isSyncOp = true;
    parsed.senderEvent = stringifyTypedPipe(setFlag.getSetPipe().getPipe());
    parsed.receiverEvent = stringifyTypedPipe(setFlag.getWaitPipe().getPipe());
    parsed.eventId = stringifyTypedEvent(setFlag.getStaticEventId(),
                                         setFlag.getDynamicEventId());
    parsed.syncIdValue = setFlag.getDynamicEventId();
    parsed.senderPipe = disambiguateMTE2Pipe(
        convertTypedPipe(setFlag.getSetPipe().getPipe()),
        convertTypedPipe(setFlag.getWaitPipe().getPipe()), parsed.op.coreType);
    parsed.receiverPipe =
        disambiguateMTE2Pipe(convertTypedPipe(setFlag.getWaitPipe().getPipe()),
                             parsed.senderPipe, parsed.op.coreType);
    parsed.op.pipe = parsed.senderPipe;
    return true;
  }
  if (auto waitFlag = llvm::dyn_cast<mlir::hivm::WaitFlagOp>(op)) {
    parsed.op.opName = "wait_flag";
    parsed.op.isSyncOp = true;
    parsed.senderEvent = stringifyTypedPipe(waitFlag.getSetPipe().getPipe());
    parsed.receiverEvent = stringifyTypedPipe(waitFlag.getWaitPipe().getPipe());
    parsed.eventId = stringifyTypedEvent(waitFlag.getStaticEventId(),
                                         waitFlag.getDynamicEventId());
    parsed.syncIdValue = waitFlag.getDynamicEventId();
    parsed.senderPipe = disambiguateMTE2Pipe(
        convertTypedPipe(waitFlag.getSetPipe().getPipe()),
        convertTypedPipe(waitFlag.getWaitPipe().getPipe()), parsed.op.coreType);
    parsed.receiverPipe =
        disambiguateMTE2Pipe(convertTypedPipe(waitFlag.getWaitPipe().getPipe()),
                             parsed.senderPipe, parsed.op.coreType);
    parsed.op.pipe = parsed.receiverPipe;
    return true;
  }
  if (auto syncSet = llvm::dyn_cast<mlir::hivm::SyncBlockSetOp>(op)) {
    parsed.op.opName = "sync_block_set";
    parsed.op.isSyncOp = true;
    parsed.syncCoreType =
        mlir::hivm::stringifyTCoreType(syncSet.getTcoreType().getTcoretype())
            .str();
    parsed.op.coreType = parsed.syncCoreType;
    parsed.senderEvent = stringifyTypedPipe(syncSet.getTpipe().getPipe());
    parsed.receiverEvent = stringifyTypedPipe(syncSet.getPipe().getPipe());
    parsed.senderPipe = disambiguateMTE2Pipe(
        convertTypedPipe(syncSet.getTpipe().getPipe()),
        convertTypedPipe(syncSet.getPipe().getPipe()), parsed.op.coreType);
    parsed.receiverPipe =
        disambiguateMTE2Pipe(convertTypedPipe(syncSet.getPipe().getPipe()),
                             parsed.senderPipe, parsed.op.coreType);
    parsed.eventId = stringifyTypedFlag(syncSet.getStaticFlagId(),
                                        syncSet.getDynamicFlagId());
    parsed.syncIdValue = syncSet.getDynamicFlagId();
    parsed.op.pipe = parsed.senderPipe;
    return true;
  }
  if (auto syncWait = llvm::dyn_cast<mlir::hivm::SyncBlockWaitOp>(op)) {
    parsed.op.opName = "sync_block_wait";
    parsed.op.isSyncOp = true;
    parsed.op.isBarrier = true;
    parsed.syncCoreType =
        mlir::hivm::stringifyTCoreType(syncWait.getTcoreType().getTcoretype())
            .str();
    parsed.op.coreType = parsed.syncCoreType;
    parsed.senderEvent = stringifyTypedPipe(syncWait.getTpipe().getPipe());
    parsed.receiverEvent = stringifyTypedPipe(syncWait.getPipe().getPipe());
    // The sender pipe lives on the *opposite* core in cross-core sync.
    llvm::StringRef senderCoreType =
        (parsed.op.coreType == "CUBE" || parsed.op.coreType == "AIC") ? "VECTOR"
                                                                      : "CUBE";
    parsed.senderPipe = disambiguateMTE2Pipe(
        convertTypedPipe(syncWait.getTpipe().getPipe()),
        convertTypedPipe(syncWait.getPipe().getPipe()), senderCoreType);
    parsed.receiverPipe =
        disambiguateMTE2Pipe(convertTypedPipe(syncWait.getPipe().getPipe()),
                             parsed.senderPipe, parsed.op.coreType);
    parsed.eventId = stringifyTypedFlag(syncWait.getStaticFlagId(),
                                        syncWait.getDynamicFlagId());
    parsed.syncIdValue = syncWait.getDynamicFlagId();
    parsed.op.pipe = HIVMPipe::All;
    parsed.barrierPipes.push_back(HIVMPipe::All);
    return true;
  }
  if (auto syncBlock = llvm::dyn_cast<mlir::hivm::SyncBlockOp>(op)) {
    parsed.op.opName = "sync_block";
    parsed.op.isSyncOp = true;
    parsed.op.isBarrier = true;
    parsed.op.pipe = HIVMPipe::All;
    if (auto inferredCore = syncBlock.inferCoreType())
      parsed.op.coreType = mlir::hivm::stringifyTCoreType(*inferredCore).str();
    return true;
  }

  return startsWithHivmOp(op);
}
#endif

static std::string stringifyAttribute(mlir::Attribute attr) {
  if (!attr)
    return "";
  std::string storage;
  llvm::raw_string_ostream os(storage);
  attr.print(os);
  os.flush();
  return storage;
}

static std::pair<std::string, std::string>
parseLoadStoreSpaces(llvm::StringRef line);

static HIVMPipe parsePipeToken(llvm::StringRef text) {
  if (text.contains("PIPE_ALL"))
    return HIVMPipe::All;
  if (text.contains("PIPE_MTE3"))
    return HIVMPipe::MTE3;
  if (text.contains("PIPE_MTE2"))
    return HIVMPipe::VectorMTE2;
  if (text.contains("PIPE_MTE1"))
    return HIVMPipe::MTE1;
  if (text.contains("PIPE_FIX"))
    return HIVMPipe::FixPipe;
  if (text.contains("PIPE_M"))
    return HIVMPipe::Cube;
  if (text.contains("PIPE_V"))
    return HIVMPipe::Vector;
  if (text.contains("PIPE_S"))
    return HIVMPipe::Scalar;
  return HIVMPipe::Unknown;
}

static std::string parseEventToken(llvm::StringRef text) {
  size_t pos = text.find("EVENT_ID");
  if (pos == llvm::StringRef::npos)
    return "";
  size_t end = pos;
  while (
      end < text.size() &&
      (std::isalnum(static_cast<unsigned char>(text[end])) || text[end] == '_'))
    ++end;
  return text.slice(pos, end).str();
}

#ifndef TRITONSIM_HAS_BISHENGIR_HIVM
static std::string inferGenericCoreType(mlir::Operation *op) {
  if (auto parentFunc = op->getParentOfType<mlir::func::FuncOp>()) {
    std::string attr =
        stringifyAttribute(parentFunc->getAttr("hivm.func_core_type"));
    if (llvm::StringRef(attr).contains("AIC") ||
        llvm::StringRef(attr).contains("CUBE"))
      return "CUBE";
    if (llvm::StringRef(attr).contains("AIV") ||
        llvm::StringRef(attr).contains("VECTOR"))
      return "VECTOR";
  }
  return resolveCoreTypeFromFunc(op, "").str();
}

static bool populateGenericHivmOp(mlir::Operation *op, ParsedOp &parsed) {
  parsed.op.opName = getLeafOpName(op).str();
  parsed.op.coreType = inferGenericCoreType(op);
  std::string opText = renderOperation(op);
  auto spaces = parseLoadStoreSpaces(opText);

  if (parsed.op.opName == "load") {
    parsed.op.pipe = selectMTE2PipeForSpaces(spaces.first, spaces.second,
                                             parsed.op.coreType);
  } else if (parsed.op.opName == "store") {
    parsed.op.pipe = HIVMPipe::MTE3;
  } else if (parsed.op.opName == "fixpipe") {
    parsed.op.pipe = HIVMPipe::FixPipe;
  } else if (parsed.op.opName == "nd2nz") {
    parsed.op.pipe = HIVMPipe::CubeMTE2;
  } else if (parsed.op.opName == "nz2nd") {
    parsed.op.pipe = HIVMPipe::MTE3;
  } else if (parsed.op.opName == "copy") {
    if (spaces.second == "gm")
      parsed.op.pipe = HIVMPipe::MTE3;
    else if (spaces.second == "l1")
      parsed.op.pipe = HIVMPipe::CubeMTE2;
    else
      parsed.op.pipe = HIVMPipe::Vector;
  } else if (parsed.op.opName == "pointer_cast" ||
             parsed.op.opName == "convert_layout") {
    parsed.op.pipe = HIVMPipe::Unknown;
  } else if (parsed.op.opName == "matmul" || parsed.op.opName == "mix_matmul" ||
             parsed.op.opName == "mix_group_matmul" ||
             parsed.op.opName == "mmadL1") {
    parsed.op.pipe = HIVMPipe::Cube;
  } else {
    parsed.op.pipe = HIVMPipe::Vector;
  }

  if (parsed.op.opName == "pipe_barrier") {
    parsed.op.isSyncOp = true;
    parsed.op.isBarrier = true;
    parsed.op.pipe = disambiguateMTE2Pipe(
        parsePipeToken(stringifyAttribute(op->getAttr("pipe"))),
        HIVMPipe::Unknown, parsed.op.coreType);
    parsed.barrierPipes.push_back(parsed.op.pipe);
    return true;
  }
  if (parsed.op.opName == "set_flag" || parsed.op.opName == "wait_flag") {
    parsed.op.isSyncOp = true;
    HIVMPipe setPipe =
        parsePipeToken(stringifyAttribute(op->getAttr("set_pipe")));
    HIVMPipe waitPipe =
        parsePipeToken(stringifyAttribute(op->getAttr("wait_pipe")));
    parsed.senderPipe =
        disambiguateMTE2Pipe(setPipe, waitPipe, parsed.op.coreType);
    parsed.receiverPipe =
        disambiguateMTE2Pipe(waitPipe, parsed.senderPipe, parsed.op.coreType);
    parsed.eventId =
        parseEventToken(stringifyAttribute(op->getAttr("static_event_id")));
    parsed.op.pipe = parsed.op.opName == "set_flag" ? parsed.senderPipe
                                                    : parsed.receiverPipe;
    attachSyncMetadata(parsed);
    return true;
  }

  return true;
}
#endif

static std::string renderValueToken(mlir::Value value) {
  if (!value)
    return "";
  std::string storage;
  llvm::raw_string_ostream os(storage);
  os << value;
  os.flush();
  return storage;
}

static std::string renderOpaqueValueToken(mlir::Value value) {
  if (!value)
    return "";
  std::string storage;
  llvm::raw_string_ostream os(storage);
  os << "ssa@" << value.getAsOpaquePointer();
  os.flush();
  return storage;
}

static std::optional<int64_t>
evaluateAffineExpr(mlir::AffineExpr expr, mlir::AffineMap map,
                   llvm::ArrayRef<int64_t> inputs);

static bool resolveMLIRValueImpl(mlir::Value value, const AnalysisState &state,
                                 int64_t &resolved,
                                 llvm::SmallDenseSet<mlir::Value, 8> &visited);

static bool resolveAffineApply(mlir::affine::AffineApplyOp affineApply,
                               const AnalysisState &state, int64_t &resolved,
                               llvm::SmallDenseSet<mlir::Value, 8> &visited) {
  llvm::SmallVector<int64_t, 8> inputs;
  inputs.reserve(affineApply.getOperands().size());
  for (mlir::Value operand : affineApply.getOperands()) {
    int64_t operandValue = 0;
    if (!resolveMLIRValueImpl(operand, state, operandValue, visited))
      return false;
    inputs.push_back(operandValue);
  }
  auto result = evaluateAffineExpr(affineApply.getAffineMap().getResult(0),
                                   affineApply.getAffineMap(), inputs);
  if (!result)
    return false;
  resolved = *result;
  return true;
}

static bool resolveMLIRValueImpl(mlir::Value value, const AnalysisState &state,
                                 int64_t &resolved,
                                 llvm::SmallDenseSet<mlir::Value, 8> &visited) {
  if (!visited.insert(value).second)
    return false;
  auto finish = [&](bool ok) {
    visited.erase(value);
    return ok;
  };

  auto cstIt = state.constants.find(value);
  if (cstIt != state.constants.end()) {
    resolved = cstIt->second;
    return finish(true);
  }
  auto boundIt = state.boundValues.find(value);
  if (boundIt != state.boundValues.end()) {
    resolved = boundIt->second;
    return finish(true);
  }
  mlir::Operation *defOp = value.getDefiningOp();
  if (!defOp || defOp->getNumResults() != 1)
    return finish(false);

  if (auto constantOp = llvm::dyn_cast<mlir::arith::ConstantOp>(defOp)) {
    if (auto intAttr =
            llvm::dyn_cast<mlir::IntegerAttr>(constantOp.getValue())) {
      resolved = intAttr.getInt();
      return finish(true);
    }
  }
  if (auto castOp = llvm::dyn_cast<mlir::arith::IndexCastOp>(defOp))
    return finish(
        resolveMLIRValueImpl(castOp.getIn(), state, resolved, visited));
  if (auto castOp = llvm::dyn_cast<mlir::arith::IndexCastUIOp>(defOp))
    return finish(
        resolveMLIRValueImpl(castOp.getIn(), state, resolved, visited));
  if (auto truncOp = llvm::dyn_cast<mlir::arith::TruncIOp>(defOp))
    return finish(
        resolveMLIRValueImpl(truncOp.getIn(), state, resolved, visited));
  if (auto extOp = llvm::dyn_cast<mlir::arith::ExtSIOp>(defOp))
    return finish(
        resolveMLIRValueImpl(extOp.getIn(), state, resolved, visited));
  if (auto addOp = llvm::dyn_cast<mlir::arith::AddIOp>(defOp)) {
    int64_t lhs = 0, rhs = 0;
    if (!resolveMLIRValueImpl(addOp.getLhs(), state, lhs, visited) ||
        !resolveMLIRValueImpl(addOp.getRhs(), state, rhs, visited))
      return finish(false);
    resolved = lhs + rhs;
    return finish(true);
  }
  if (auto subOp = llvm::dyn_cast<mlir::arith::SubIOp>(defOp)) {
    int64_t lhs = 0, rhs = 0;
    if (!resolveMLIRValueImpl(subOp.getLhs(), state, lhs, visited) ||
        !resolveMLIRValueImpl(subOp.getRhs(), state, rhs, visited))
      return finish(false);
    resolved = lhs - rhs;
    return finish(true);
  }
  if (auto mulOp = llvm::dyn_cast<mlir::arith::MulIOp>(defOp)) {
    int64_t lhs = 0, rhs = 0;
    bool lhsResolved =
        resolveMLIRValueImpl(mulOp.getLhs(), state, lhs, visited);
    if (lhsResolved && lhs == 0) {
      resolved = 0;
      return finish(true);
    }
    bool rhsResolved =
        resolveMLIRValueImpl(mulOp.getRhs(), state, rhs, visited);
    if (rhsResolved && rhs == 0) {
      resolved = 0;
      return finish(true);
    }
    if (!lhsResolved || !rhsResolved)
      return finish(false);
    resolved = lhs * rhs;
    return finish(true);
  }
  if (auto divOp = llvm::dyn_cast<mlir::arith::DivSIOp>(defOp)) {
    int64_t lhs = 0;
    if (!resolveMLIRValueImpl(divOp.getLhs(), state, lhs, visited))
      return finish(false);
    if (lhs == 0) {
      resolved = 0;
      return finish(true);
    }
    int64_t rhs = 0;
    if (!resolveMLIRValueImpl(divOp.getRhs(), state, rhs, visited) || rhs == 0)
      return finish(false);
    resolved = lhs / rhs;
    return finish(true);
  }
  if (auto remOp = llvm::dyn_cast<mlir::arith::RemSIOp>(defOp)) {
    int64_t lhs = 0;
    if (!resolveMLIRValueImpl(remOp.getLhs(), state, lhs, visited))
      return finish(false);
    if (lhs == 0) {
      resolved = 0;
      return finish(true);
    }
    int64_t rhs = 0;
    if (!resolveMLIRValueImpl(remOp.getRhs(), state, rhs, visited) || rhs == 0)
      return finish(false);
    resolved = lhs % rhs;
    return finish(true);
  }
  if (auto minOp = llvm::dyn_cast<mlir::arith::MinSIOp>(defOp)) {
    int64_t lhs = 0, rhs = 0;
    if (!resolveMLIRValueImpl(minOp.getLhs(), state, lhs, visited) ||
        !resolveMLIRValueImpl(minOp.getRhs(), state, rhs, visited))
      return finish(false);
    resolved = std::min(lhs, rhs);
    return finish(true);
  }
  if (auto cmpOp = llvm::dyn_cast<mlir::arith::CmpIOp>(defOp)) {
    int64_t lhs = 0, rhs = 0;
    if (!resolveMLIRValueImpl(cmpOp.getLhs(), state, lhs, visited) ||
        !resolveMLIRValueImpl(cmpOp.getRhs(), state, rhs, visited))
      return finish(false);
    switch (cmpOp.getPredicate()) {
    case mlir::arith::CmpIPredicate::eq:
      resolved = lhs == rhs;
      return finish(true);
    case mlir::arith::CmpIPredicate::ne:
      resolved = lhs != rhs;
      return finish(true);
    case mlir::arith::CmpIPredicate::slt:
      resolved = lhs < rhs;
      return finish(true);
    case mlir::arith::CmpIPredicate::sle:
      resolved = lhs <= rhs;
      return finish(true);
    case mlir::arith::CmpIPredicate::sgt:
      resolved = lhs > rhs;
      return finish(true);
    case mlir::arith::CmpIPredicate::sge:
      resolved = lhs >= rhs;
      return finish(true);
    case mlir::arith::CmpIPredicate::ult:
      resolved = static_cast<uint64_t>(lhs) < static_cast<uint64_t>(rhs);
      return finish(true);
    case mlir::arith::CmpIPredicate::ule:
      resolved = static_cast<uint64_t>(lhs) <= static_cast<uint64_t>(rhs);
      return finish(true);
    case mlir::arith::CmpIPredicate::ugt:
      resolved = static_cast<uint64_t>(lhs) > static_cast<uint64_t>(rhs);
      return finish(true);
    case mlir::arith::CmpIPredicate::uge:
      resolved = static_cast<uint64_t>(lhs) >= static_cast<uint64_t>(rhs);
      return finish(true);
    }
  }
  if (auto selectOp = llvm::dyn_cast<mlir::arith::SelectOp>(defOp)) {
    int64_t cond = 0, trueValue = 0, falseValue = 0;
    if (!resolveMLIRValueImpl(selectOp.getCondition(), state, cond, visited) ||
        !resolveMLIRValueImpl(selectOp.getTrueValue(), state, trueValue,
                              visited) ||
        !resolveMLIRValueImpl(selectOp.getFalseValue(), state, falseValue,
                              visited))
      return finish(false);
    resolved = cond != 0 ? trueValue : falseValue;
    return finish(true);
  }
  if (auto affineApply = llvm::dyn_cast<mlir::affine::AffineApplyOp>(defOp))
    return finish(resolveAffineApply(affineApply, state, resolved, visited));

  return finish(false);
}

static bool resolveMLIRValue(mlir::Value value, const AnalysisState &state,
                             int64_t &resolved) {
  llvm::SmallDenseSet<mlir::Value, 8> visited;
  return resolveMLIRValueImpl(value, state, resolved, visited);
}

static std::string canonicalizeSyncId(mlir::Value value,
                                      const AnalysisState &state) {
  if (!value)
    return "";

  int64_t resolved = 0;
  if (resolveMLIRValue(value, state, resolved))
    return std::to_string(resolved);

  auto producerIt = state.valueProducers.find(value);
  if (producerIt != state.valueProducers.end())
    return ("ssa_producer_" + std::to_string(producerIt->second));

  return renderOpaqueValueToken(value);
}

static bool parseForTripCount(mlir::scf::ForOp forOp,
                              const AnalysisState &state, int64_t &tripCount) {
  int64_t lb = 0;
  int64_t ub = 0;
  int64_t step = 0;
  if (!resolveMLIRValue(forOp.getLowerBound(), state, lb) ||
      !resolveMLIRValue(forOp.getUpperBound(), state, ub) ||
      !resolveMLIRValue(forOp.getStep(), state, step) || step <= 0 || ub < lb) {
    tripCount = 1;
    return false;
  }
  tripCount = std::max<int64_t>(1, ceilDiv(ub - lb, step));
  return true;
}

static bool captureConstant(mlir::Operation *op, AnalysisState &state) {
  auto constantOp = llvm::dyn_cast<mlir::arith::ConstantOp>(op);
  if (!constantOp || op->getNumResults() != 1)
    return false;

  mlir::Attribute valueAttr = constantOp.getValue();
  if (auto intAttr = llvm::dyn_cast<mlir::IntegerAttr>(valueAttr)) {
    state.constants[op->getResult(0)] = intAttr.getInt();
    return true;
  }
  return false;
}

static std::optional<int64_t>
evaluateAffineExpr(mlir::AffineExpr expr, mlir::AffineMap map,
                   llvm::ArrayRef<int64_t> inputs);

static bool resolveMLIRValue(mlir::Value value, const AnalysisState &state,
                             int64_t &resolved);

static std::optional<int64_t>
evaluateAffineExpr(mlir::AffineExpr expr, mlir::AffineMap map,
                   llvm::ArrayRef<int64_t> inputs) {
  if (auto constant = llvm::dyn_cast<mlir::AffineConstantExpr>(expr))
    return constant.getValue();
  if (auto dim = llvm::dyn_cast<mlir::AffineDimExpr>(expr)) {
    unsigned pos = dim.getPosition();
    if (pos < inputs.size())
      return inputs[pos];
    return std::nullopt;
  }
  if (auto symbol = llvm::dyn_cast<mlir::AffineSymbolExpr>(expr)) {
    unsigned pos = map.getNumDims() + symbol.getPosition();
    if (pos < inputs.size())
      return inputs[pos];
    return std::nullopt;
  }
  if (auto binary = llvm::dyn_cast<mlir::AffineBinaryOpExpr>(expr)) {
    auto lhs = evaluateAffineExpr(binary.getLHS(), map, inputs);
    auto rhs = evaluateAffineExpr(binary.getRHS(), map, inputs);
    if (!lhs || !rhs)
      return std::nullopt;
    switch (binary.getKind()) {
    case mlir::AffineExprKind::Add:
      return *lhs + *rhs;
    case mlir::AffineExprKind::Mul:
      return *lhs * *rhs;
    case mlir::AffineExprKind::Mod:
      if (*rhs == 0)
        return std::nullopt;
      return *lhs % *rhs;
    case mlir::AffineExprKind::FloorDiv:
      if (*rhs == 0)
        return std::nullopt;
      return llvm::divideFloorSigned(*lhs, *rhs);
    case mlir::AffineExprKind::CeilDiv:
      if (*rhs == 0)
        return std::nullopt;
      return llvm::divideCeilSigned(*lhs, *rhs);
    default:
      return std::nullopt;
    }
  }
  return std::nullopt;
}

static bool captureDerivedScalarValue(mlir::Operation *op,
                                      AnalysisState &state) {
  if (op->getNumResults() != 1)
    return false;

  auto recordValue = [&](int64_t value) {
    state.boundValues[op->getResult(0)] = value;
    return true;
  };

  if (op->getName().getStringRef() == "hivm.hir.get_block_idx" ||
      op->getName().getStringRef() == "get_block_idx") {
    auto it = state.argBindings.find("pid_x");
    if (it == state.argBindings.end())
      it = state.argBindings.find("program_id_x");
    if (it != state.argBindings.end())
      return recordValue(it->second);
    return false;
  }

  if (auto castOp = llvm::dyn_cast<mlir::arith::IndexCastOp>(op)) {
    int64_t resolved = 0;
    if (resolveMLIRValue(castOp.getIn(), state, resolved))
      return recordValue(resolved);
    return false;
  }

  if (auto castOp = llvm::dyn_cast<mlir::arith::IndexCastUIOp>(op)) {
    int64_t resolved = 0;
    if (resolveMLIRValue(castOp.getIn(), state, resolved))
      return recordValue(resolved);
    return false;
  }

  if (auto truncOp = llvm::dyn_cast<mlir::arith::TruncIOp>(op)) {
    int64_t resolved = 0;
    if (resolveMLIRValue(truncOp.getIn(), state, resolved))
      return recordValue(resolved);
    return false;
  }

  if (auto extOp = llvm::dyn_cast<mlir::arith::ExtSIOp>(op)) {
    int64_t resolved = 0;
    if (resolveMLIRValue(extOp.getIn(), state, resolved))
      return recordValue(resolved);
    return false;
  }

  if (auto addOp = llvm::dyn_cast<mlir::arith::AddIOp>(op)) {
    int64_t lhs = 0, rhs = 0;
    if (resolveMLIRValue(addOp.getLhs(), state, lhs) &&
        resolveMLIRValue(addOp.getRhs(), state, rhs))
      return recordValue(lhs + rhs);
    return false;
  }

  if (auto subOp = llvm::dyn_cast<mlir::arith::SubIOp>(op)) {
    int64_t lhs = 0, rhs = 0;
    if (resolveMLIRValue(subOp.getLhs(), state, lhs) &&
        resolveMLIRValue(subOp.getRhs(), state, rhs))
      return recordValue(lhs - rhs);
    return false;
  }

  if (auto mulOp = llvm::dyn_cast<mlir::arith::MulIOp>(op)) {
    int64_t lhs = 0, rhs = 0;
    if (resolveMLIRValue(mulOp.getLhs(), state, lhs) &&
        resolveMLIRValue(mulOp.getRhs(), state, rhs))
      return recordValue(lhs * rhs);
    return false;
  }

  if (auto divOp = llvm::dyn_cast<mlir::arith::DivSIOp>(op)) {
    int64_t lhs = 0, rhs = 0;
    if (resolveMLIRValue(divOp.getLhs(), state, lhs) &&
        resolveMLIRValue(divOp.getRhs(), state, rhs) && rhs != 0)
      return recordValue(lhs / rhs);
    return false;
  }

  if (auto remOp = llvm::dyn_cast<mlir::arith::RemSIOp>(op)) {
    int64_t lhs = 0, rhs = 0;
    if (resolveMLIRValue(remOp.getLhs(), state, lhs) &&
        resolveMLIRValue(remOp.getRhs(), state, rhs) && rhs != 0)
      return recordValue(lhs % rhs);
    return false;
  }

  if (auto minOp = llvm::dyn_cast<mlir::arith::MinSIOp>(op)) {
    int64_t lhs = 0, rhs = 0;
    if (resolveMLIRValue(minOp.getLhs(), state, lhs) &&
        resolveMLIRValue(minOp.getRhs(), state, rhs))
      return recordValue(std::min(lhs, rhs));
    return false;
  }

  if (auto cmpOp = llvm::dyn_cast<mlir::arith::CmpIOp>(op)) {
    int64_t lhs = 0, rhs = 0;
    if (!resolveMLIRValue(cmpOp.getLhs(), state, lhs) ||
        !resolveMLIRValue(cmpOp.getRhs(), state, rhs))
      return false;
    bool result = false;
    switch (cmpOp.getPredicate()) {
    case mlir::arith::CmpIPredicate::eq:
      result = lhs == rhs;
      break;
    case mlir::arith::CmpIPredicate::ne:
      result = lhs != rhs;
      break;
    case mlir::arith::CmpIPredicate::slt:
      result = lhs < rhs;
      break;
    case mlir::arith::CmpIPredicate::sle:
      result = lhs <= rhs;
      break;
    case mlir::arith::CmpIPredicate::sgt:
      result = lhs > rhs;
      break;
    case mlir::arith::CmpIPredicate::sge:
      result = lhs >= rhs;
      break;
    case mlir::arith::CmpIPredicate::ult:
      result = static_cast<uint64_t>(lhs) < static_cast<uint64_t>(rhs);
      break;
    case mlir::arith::CmpIPredicate::ule:
      result = static_cast<uint64_t>(lhs) <= static_cast<uint64_t>(rhs);
      break;
    case mlir::arith::CmpIPredicate::ugt:
      result = static_cast<uint64_t>(lhs) > static_cast<uint64_t>(rhs);
      break;
    case mlir::arith::CmpIPredicate::uge:
      result = static_cast<uint64_t>(lhs) >= static_cast<uint64_t>(rhs);
      break;
    }
    return recordValue(result ? 1 : 0);
  }

  if (auto selectOp = llvm::dyn_cast<mlir::arith::SelectOp>(op)) {
    int64_t cond = 0, trueValue = 0, falseValue = 0;
    if (resolveMLIRValue(selectOp.getCondition(), state, cond) &&
        resolveMLIRValue(selectOp.getTrueValue(), state, trueValue) &&
        resolveMLIRValue(selectOp.getFalseValue(), state, falseValue))
      return recordValue(cond != 0 ? trueValue : falseValue);
    return false;
  }

  if (auto affineApply = llvm::dyn_cast<mlir::affine::AffineApplyOp>(op)) {
    llvm::SmallVector<int64_t, 8> inputs;
    inputs.reserve(affineApply.getOperands().size());
    for (mlir::Value operand : affineApply.getOperands()) {
      int64_t resolved = 0;
      if (!resolveMLIRValue(operand, state, resolved))
        return false;
      inputs.push_back(resolved);
    }
    auto result = evaluateAffineExpr(affineApply.getAffineMap().getResult(0),
                                     affineApply.getAffineMap(), inputs);
    if (result)
      return recordValue(*result);
    return false;
  }

  return false;
}

static std::string getOrCreateBufferRoot(mlir::Value value,
                                         AnalysisState &state) {
  auto it = state.bufferRoots.find(value);
  if (it != state.bufferRoots.end())
    return it->second;
  std::string root = renderValueToken(value);
  state.bufferRoots[value] = root;
  return root;
}

static void captureBufferMetadata(mlir::Operation *op, AnalysisState &state) {
#ifdef TRITONSIM_HAS_BISHENGIR_HIVM
  if (auto markOp = llvm::dyn_cast<mlir::annotation::MarkOp>(op)) {
    mlir::Value src = markOp.getSrc();
    if (markOp.isAnnotatedByStaticAttr("hivm.multi_buffer")) {
      if (auto intAttr = llvm::dyn_cast_or_null<mlir::IntegerAttr>(
              markOp.getStaticAttrValue("hivm.multi_buffer"))) {
        std::string root = getOrCreateBufferRoot(src, state);
        state.bufferSlots[root] = std::max<int64_t>(1, intAttr.getInt());
      }
    }
    return;
  }
#endif

  if (auto subviewOp = llvm::dyn_cast<mlir::memref::SubViewOp>(op)) {
    auto it = state.bufferRoots.find(subviewOp.getSource());
    if (it != state.bufferRoots.end())
      state.bufferRoots[subviewOp.getResult()] = it->second;
    return;
  }
  if (auto castOp = llvm::dyn_cast<mlir::memref::ReinterpretCastOp>(op)) {
    auto it = state.bufferRoots.find(castOp.getSource());
    if (it != state.bufferRoots.end())
      state.bufferRoots[castOp.getResult()] = it->second;
    return;
  }
  if (auto castOp = llvm::dyn_cast<mlir::memref::CastOp>(op)) {
    auto it = state.bufferRoots.find(castOp.getSource());
    if (it != state.bufferRoots.end())
      state.bufferRoots[castOp.getResult()] = it->second;
    return;
  }
}

static void dedupeBufferList(std::vector<std::string> &buffers) {
  std::sort(buffers.begin(), buffers.end());
  buffers.erase(std::unique(buffers.begin(), buffers.end()), buffers.end());
}

static bool isLikelyWritingOp(llvm::StringRef opName) {
  return opName == "load" || opName == "copy" || opName == "vadd" ||
         opName == "vmul" || opName == "vcast" || opName == "vbrc" ||
         opName == "vreduce" || opName == "fixpipe" || opName == "nd2nz" ||
         opName == "nz2nd" || opName == "mmadL1" || opName == "matmul" ||
         opName == "mix_matmul" || opName == "mix_group_matmul";
}

static void attachBufferAccessMetadata(mlir::Operation *op, ParsedOp &parsed,
                                       AnalysisState &state) {
  if (parsed.op.opName == "pointer_cast" && op->getNumResults() == 1 &&
      mlir::isa<mlir::MemRefType>(op->getResult(0).getType())) {
    getOrCreateBufferRoot(op->getResult(0), state);
  }

  llvm::SmallVector<std::string, 4> rootedOperands;
  for (mlir::Value operand : op->getOperands()) {
    auto it = state.bufferRoots.find(operand);
    if (it != state.bufferRoots.end())
      rootedOperands.push_back(it->second);
  }

  if (rootedOperands.empty())
    return;

  if (isLikelyWritingOp(parsed.op.opName)) {
    parsed.op.writeBuffers.push_back(rootedOperands.back());
    parsed.op.multiBufferSlots = std::max<int64_t>(
        parsed.op.multiBufferSlots, state.bufferSlots[rootedOperands.back()]);
    rootedOperands.pop_back();
  }

  for (const std::string &root : rootedOperands)
    parsed.op.readBuffers.push_back(root);
  dedupeBufferList(parsed.op.readBuffers);
  dedupeBufferList(parsed.op.writeBuffers);
}

static std::pair<std::string, std::string>
parseLoadStoreSpaces(llvm::StringRef line) {
  llvm::SmallVector<std::string, 2> spaces;
  size_t pos = 0;
  while (spaces.size() < 2) {
    pos = line.find("#hivm.address_space<", pos);
    if (pos == llvm::StringRef::npos)
      break;
    pos += strlen("#hivm.address_space<");
    size_t end = line.find('>', pos);
    if (end == llvm::StringRef::npos)
      break;
    spaces.push_back(canonicalizeAddressSpace(line.slice(pos, end)));
    pos = end + 1;
  }
  if (spaces.size() < 2)
    return {"", ""};
  return {spaces[0], spaces[1]};
}

static std::map<std::string, int64_t>
parseArgBindings(llvm::StringRef bindings) {
  std::map<std::string, int64_t> result;
  llvm::SmallVector<llvm::StringRef, 8> entries;
  bindings.split(entries, ',', -1, false);
  for (llvm::StringRef entry : entries) {
    auto kv = trim(entry).split('=');
    if (kv.first.empty() || kv.second.empty())
      continue;
    int64_t value = 0;
    if (trim(kv.second).getAsInteger(10, value))
      continue;
    result[trim(kv.first).str()] = value;
  }
  return result;
}

static bool isZeroCostOp(llvm::StringRef opName) {
  return opName == "pointer_cast" || opName == "convert_layout";
}

static bool isCubeOpName(llvm::StringRef opName) {
  return opName == "matmul" || opName == "mix_matmul" ||
         opName == "mix_group_matmul" || opName == "mmadL1";
}

static int64_t estimateDuration(const ParsedOp &parsed,
                                const HardwareConfig &config) {
  llvm::StringRef opName = parsed.op.opName;
  llvm::StringRef line = parsed.op.text;

  auto startupForPipe = [&](HIVMPipe pipe) -> int64_t {
    switch (pipe) {
    case HIVMPipe::Vector:
      return config.getVectorStartupLatency();
    case HIVMPipe::VectorMTE2:
    case HIVMPipe::CubeMTE2:
      return config.getMTE2StartupLatency();
    case HIVMPipe::MTE3:
      return config.getMTE3StartupLatency();
    case HIVMPipe::FixPipe:
      return config.getFixPipeStartupLatency();
    case HIVMPipe::Cube:
      return config.getCubeStartupLatency();
    case HIVMPipe::MTE1:
      return std::max<int64_t>(1, config.getMTE2StartupLatency() / 2);
    case HIVMPipe::Scalar:
      return 1;
    case HIVMPipe::All:
      return config.getPipeBarrierCyclesPerIter();
    case HIVMPipe::Unknown:
      return 1;
    }
    return 1;
  };

  auto estimateSpaceTransferCycles =
      [&](llvm::StringRef srcSpace, llvm::StringRef dstSpace, int64_t bytes,
          int64_t startup) -> int64_t {
    int64_t clampedBytes = std::max<int64_t>(bytes, 1);
    auto chooseBandwidth = [&](llvm::StringRef name) {
      double bw = config.getMemoryBandwidthBytesPerCycle(name);
      return bw > 0.0 ? bw : 0.0;
    };
    double srcBw = chooseBandwidth(srcSpace);
    double dstBw = chooseBandwidth(dstSpace);
    double bw = 0.0;
    if (srcBw > 0.0 && dstBw > 0.0)
      bw = std::min(srcBw, dstBw);
    else
      bw = std::max(srcBw, dstBw);
    if (bw <= 0.0)
      bw = static_cast<double>(config.getVectorWidthBytes());
    int latency = std::max(config.getMemoryLatencyCycles(srcSpace),
                           config.getMemoryLatencyCycles(dstSpace));
    int64_t transfer = std::max<int64_t>(
        1, static_cast<int64_t>(std::ceil(clampedBytes / bw)));
    return startup + latency + transfer;
  };

  if (isZeroCostOp(opName))
    return 0;

  if (opName == "set_mask_norm" || opName == "get_block_idx")
    return 1;

  if (opName == "set_flag")
    return 1;
  if (opName == "wait_flag")
    return 2;
  if (opName == "sync_block_set") {
    HIVMPipe sender = parsed.senderPipe;
    HIVMPipe receiver = parsed.receiverPipe;
    int64_t crossPipe = sender != receiver ? 1 : 0;
    int64_t crossCore = parsed.op.coreType == "CUBE" ? 2 : 1;
    return 1 + crossPipe + crossCore;
  }
  if (opName == "sync_block_wait") {
    HIVMPipe receiver = parsed.receiverPipe;
    int64_t base = std::max<int64_t>(2, startupForPipe(receiver) / 8);
    if (parsed.op.coreType == "CUBE")
      base += 2;
    return base;
  }
  if (opName == "sync_block")
    return std::max<int64_t>(config.getPipeBarrierCyclesPerIter(),
                             config.getVectorStartupLatency());

  // pipe_barrier drains the in-flight instructions on the target pipe.
  // Cost is the pipeline depth, NOT the full startup latency.
  if (opName == "pipe_barrier") {
    if (parsed.barrierPipes.empty())
      return 8;
    if (llvm::is_contained(parsed.barrierPipes, HIVMPipe::All))
      return 64;
    HIVMPipe pipe = parsed.barrierPipes.front();
    switch (pipe) {
    case HIVMPipe::Vector:
      return 4;
    case HIVMPipe::VectorMTE2:
    case HIVMPipe::CubeMTE2:
      return 16;
    case HIVMPipe::MTE3:
      return 16;
    case HIVMPipe::FixPipe:
      return 8;
    case HIVMPipe::Cube:
      return 8;
    default:
      return 8;
    }
  }

  if (opName == "load") {
    int64_t bytes = parsed.op.bytes;
    auto spaces = parseLoadStoreSpaces(line);
    if (spaces.first == "gm" && spaces.second == "ub")
      return config.getMTE2StartupLatency() +
             config.estimateMemoryCycles("vector_mte2",
                                         std::max<int64_t>(bytes, 1));
    if (spaces.first == "gm" && spaces.second == "l1")
      return config.getMTE2StartupLatency() +
             config.estimateMemoryCycles("cube_mte2",
                                         std::max<int64_t>(bytes, 1));
    return config.getMTE2StartupLatency() +
           config.estimateMemoryCyclesWithLatency("hbm",
                                                  std::max<int64_t>(bytes, 1));
  }

  if (opName == "store") {
    int64_t bytes = parsed.op.bytes;
    auto spaces = parseLoadStoreSpaces(line);
    if (spaces.first == "ub" && spaces.second == "gm")
      return config.getMTE3StartupLatency() +
             config.estimateMemoryCycles("mte3", std::max<int64_t>(bytes, 1));
    if (spaces.first == "l0c" && spaces.second == "gm")
      return config.getFixPipeStartupLatency() +
             config.estimateMemoryCycles("fixpipe",
                                         std::max<int64_t>(bytes, 1));
    return config.getMTE3StartupLatency() +
           config.estimateMemoryCyclesWithLatency("hbm",
                                                  std::max<int64_t>(bytes, 1));
  }

  if (opName == "copy") {
    int64_t bytes =
        std::max<int64_t>(parsed.op.bytes, config.getVectorWidthBytes());
    auto spaces = parseLoadStoreSpaces(line);
    if (spaces.first == "ub" && spaces.second == "ub")
      return estimateSpaceTransferCycles("ub", "ub", bytes,
                                         config.getVectorStartupLatency());
    if (spaces.first == "gm" && spaces.second == "l1")
      return config.getMTE2StartupLatency() +
             config.estimateMemoryCycles("cube_mte2", bytes);
    if (spaces.first == "ub" && spaces.second == "l1")
      return estimateSpaceTransferCycles("ub", "l1", bytes,
                                         config.getMTE3StartupLatency());
    if (spaces.first == "l0c" && spaces.second == "gm")
      return config.getFixPipeStartupLatency() +
             config.estimateMemoryCycles("fixpipe", bytes);
    return estimateSpaceTransferCycles("ub", "ub", bytes,
                                       startupForPipe(parsed.op.pipe));
  }

  auto vectorCycles = [&](int opCost) -> int64_t {
    int64_t elems =
        std::max<int64_t>(parsed.op.elements, config.getVectorWidthElements());
    return config.getVectorStartupLatency() +
           ceilDiv(elems, config.getVectorWidthElements()) * opCost;
  };

  auto isVectorALUOp = [&](llvm::StringRef name) {
    return name == "vadd" || name == "vmul" || name == "vsub" ||
           name == "vcast" || name == "vexp" || name == "vdiv" ||
           name == "vlog" || name == "vsqrt" || name == "vrsqrt" ||
           name == "vtanh" || name == "vsigmoid" || name == "vreduce";
  };
  if (isVectorALUOp(opName))
    return vectorCycles(config.getVectorOpCyclesPerInstruction(opName));
  if (opName == "vbrc") {
    if (parsed.op.pipe == HIVMPipe::VectorMTE2 ||
        parsed.op.pipe == HIVMPipe::CubeMTE2) {
      int64_t bytes =
          std::max<int64_t>(parsed.op.bytes, config.getVectorWidthBytes());
      return estimateSpaceTransferCycles(
          "ub", parsed.op.pipe == HIVMPipe::CubeMTE2 ? "l1" : "ub", bytes,
          config.getMTE2StartupLatency());
    }
    return vectorCycles(1);
  }
  if (opName == "fixpipe") {
    int64_t bytes = std::max<int64_t>(parsed.op.bytes, 1);
    auto spaces = parseLoadStoreSpaces(line);
    if (spaces.second == "ub")
      return estimateSpaceTransferCycles("l0c", "ub", bytes,
                                         config.getFixPipeStartupLatency());
    if (spaces.second == "gm")
      return config.getFixPipeStartupLatency() +
             config.estimateMemoryCycles("fixpipe", bytes);
    return config.getFixPipeStartupLatency() +
           config.estimateMemoryCycles("fixpipe", bytes);
  }

  if (opName == "nd2nz") {
    return estimateND2NZCycles(parsed, config);
  }

  if (opName == "nz2nd") {
    int64_t bytes = std::max<int64_t>(parsed.op.bytes, 1);
    return config.getMTE3StartupLatency() +
           config.estimateMemoryCycles("mte3", bytes);
  }

  if (isCubeOpName(opName)) {
    int64_t totalElements = 0;
    size_t memrefCount = 0;
    size_t searchPos = 0;
    while ((searchPos = line.find("memref<", searchPos)) !=
               llvm::StringRef::npos &&
           memrefCount < 3) {
      totalElements += parseMemRefElementCount(line.drop_front(searchPos));
      ++memrefCount;
      searchPos += 7;
    }
    if (memrefCount >= 3) {
      int64_t lhs = std::max<int64_t>(1, totalElements / 3);
      int64_t rhs = std::max<int64_t>(1, totalElements / 3);
      int64_t out = std::max<int64_t>(1, totalElements - lhs - rhs);
      int64_t M = std::max<int64_t>(1, static_cast<int64_t>(std::sqrt(out)));
      int64_t N = std::max<int64_t>(1, out / M);
      int64_t K = std::max<int64_t>(1, lhs / M);
      return config.getCubeStartupLatency() +
             config.estimateCubeCycles(M, N, K);
    }
    return config.getCubeStartupLatency() + 16;
  }

  return 1;
}

static void
addLatestPipeDependency(HIVMPipe pipe,
                        const std::map<HIVMPipe, size_t> &latestPipeProducer,
                        ParsedOp &parsed) {
  auto it = latestPipeProducer.find(pipe);
  if (it != latestPipeProducer.end())
    parsed.op.dependsOn.push_back(it->second);
}

static bool pipeBelongsToCore(HIVMPipe pipe, llvm::StringRef coreType) {
  bool isCubeCore = coreType == "CUBE" || coreType == "AIC";
  bool isVectorCore = coreType == "VECTOR" || coreType == "AIV";
  if (isCubeCore) {
    return pipe == HIVMPipe::Cube || pipe == HIVMPipe::MTE1 ||
           pipe == HIVMPipe::CubeMTE2 || pipe == HIVMPipe::FixPipe ||
           pipe == HIVMPipe::Scalar;
  }
  if (isVectorCore) {
    return pipe == HIVMPipe::Vector || pipe == HIVMPipe::VectorMTE2 ||
           pipe == HIVMPipe::MTE3 || pipe == HIVMPipe::Scalar;
  }
  return false;
}

static llvm::SmallVector<HIVMPipe, 5>
getCoreBarrierPipes(llvm::StringRef coreType) {
  if (coreType == "CUBE" || coreType == "AIC")
    return {HIVMPipe::Cube, HIVMPipe::MTE1, HIVMPipe::CubeMTE2,
            HIVMPipe::FixPipe, HIVMPipe::Scalar};
  if (coreType == "VECTOR" || coreType == "AIV")
    return {HIVMPipe::Vector, HIVMPipe::VectorMTE2, HIVMPipe::MTE3,
            HIVMPipe::Scalar};
  return {};
}

static void ingestParsedOp(const ParsedOp &parsed, AnalysisState &state,
                           HIVMAnalysisReport &report,
                           const HardwareConfig &config) {
  ParsedOp mutableParsed = parsed;
  if (mutableParsed.syncIdValue) {
    std::string canonicalSyncId =
        canonicalizeSyncId(mutableParsed.syncIdValue, state);
    if (!canonicalSyncId.empty())
      mutableParsed.op.eventId = canonicalSyncId;
  }
  EventKey opEventKey{mutableParsed.senderPipe, mutableParsed.receiverPipe,
                      mutableParsed.op.eventId};

  // Enforce program order within each pipe: every op depends on the previous
  // op on the same pipe, matching hardware sequential execution semantics.
  if (mutableParsed.op.pipe != HIVMPipe::Unknown &&
      mutableParsed.op.pipe != HIVMPipe::All)
    addLatestPipeDependency(mutableParsed.op.pipe, state.latestPipeProducer,
                            mutableParsed);

  mutableParsed.op.readBufferVersions.reserve(
      mutableParsed.op.readBuffers.size());
  for (const std::string &root : mutableParsed.op.readBuffers) {
    auto it = state.bufferVersions.find(root);
    mutableParsed.op.readBufferVersions.push_back(
        it != state.bufferVersions.end() ? it->second : 0);
  }
  mutableParsed.op.writeBufferVersions.reserve(
      mutableParsed.op.writeBuffers.size());
  for (const std::string &root : mutableParsed.op.writeBuffers) {
    int64_t nextVersion = state.bufferVersions[root] + 1;
    state.bufferVersions[root] = nextVersion;
    mutableParsed.op.writeBufferVersions.push_back(nextVersion);
  }

  if (mutableParsed.op.isBarrier) {
    if (mutableParsed.barrierPipes.empty() ||
        llvm::is_contained(mutableParsed.barrierPipes, HIVMPipe::All)) {
      for (const auto &entry : state.latestPipeProducer) {
        if (mutableParsed.op.pipe == HIVMPipe::All &&
            !mutableParsed.op.coreType.empty() &&
            !pipeBelongsToCore(entry.first, mutableParsed.op.coreType))
          continue;
        mutableParsed.op.dependsOn.push_back(entry.second);
      }
    } else {
      for (HIVMPipe pipe : mutableParsed.barrierPipes)
        addLatestPipeDependency(pipe, state.latestPipeProducer, mutableParsed);
    }
  }

  if (mutableParsed.op.opName == "wait_flag" ||
      mutableParsed.op.opName == "sync_block_wait") {
    auto genIt = state.eventGenerations.find(opEventKey);
    if (genIt != state.eventGenerations.end())
      mutableParsed.op.eventGeneration = genIt->second;
    auto eventIt = state.eventProducers.find(opEventKey);
    if (eventIt != state.eventProducers.end())
      mutableParsed.op.dependsOn.push_back(eventIt->second);
    addLatestPipeDependency(mutableParsed.op.pipe, state.latestPipeProducer,
                            mutableParsed);
  }

  std::vector<HIVMOp> expandedOps = expandMacroOp(mutableParsed, config);
  size_t firstExpandedId = report.operations.size();
  size_t previousExpandedId = std::numeric_limits<size_t>::max();
  for (HIVMOp &expanded : expandedOps) {
    expanded.id = report.operations.size();
    if (previousExpandedId != std::numeric_limits<size_t>::max())
      expanded.dependsOn.push_back(previousExpandedId);
    // Expanded sub-ops may land on a different pipe than the parsed op;
    // enforce program order on each sub-op's actual pipe.
    if (expanded.pipe != mutableParsed.op.pipe &&
        expanded.pipe != HIVMPipe::Unknown && expanded.pipe != HIVMPipe::All) {
      auto it = state.latestPipeProducer.find(expanded.pipe);
      if (it != state.latestPipeProducer.end())
        expanded.dependsOn.push_back(it->second);
    }
    previousExpandedId = expanded.id;
    report.operations.push_back(std::move(expanded));
  }

  if (mutableParsed.op.opName == "set_flag" ||
      mutableParsed.op.opName == "sync_block_set") {
    int64_t nextGeneration = state.eventGenerations[opEventKey] + 1;
    state.eventGenerations[opEventKey] = nextGeneration;
    mutableParsed.op.eventGeneration = nextGeneration;
    for (size_t idx = firstExpandedId;
         idx <= previousExpandedId && idx < report.operations.size(); ++idx)
      report.operations[idx].eventGeneration = nextGeneration;
    state.eventProducers[opEventKey] = firstExpandedId;
  }

  for (mlir::Value result : mutableParsed.mlirResults)
    state.valueProducers[result] = previousExpandedId;

  for (size_t idx = firstExpandedId;
       idx <= previousExpandedId && idx < report.operations.size(); ++idx) {
    HIVMOp &expanded = report.operations[idx];
    if (expanded.pipe == HIVMPipe::All && expanded.isBarrier) {
      // A PIPE_ALL barrier blocks all pipes on its core.  Register it as the
      // latest producer for every pipe in that core so subsequent ops on any
      // of those pipes depend on the barrier completing.
      for (HIVMPipe p : getCoreBarrierPipes(expanded.coreType))
        state.latestPipeProducer[p] = expanded.id;
    } else if (expanded.pipe != HIVMPipe::All &&
               expanded.pipe != HIVMPipe::Unknown) {
      state.latestPipeProducer[expanded.pipe] = expanded.id;
    }
  }
}

static void analyzeParsedOperation(mlir::Operation *op, int64_t loopMultiplier,
                                   AnalysisState &state,
                                   HIVMAnalysisReport &report,
                                   const HardwareConfig &config,
                                   bool replayIterations);

static void analyzeParsedRegion(mlir::Region &region, int64_t loopMultiplier,
                                AnalysisState &state,
                                HIVMAnalysisReport &report,
                                const HardwareConfig &config,
                                bool replayIterations) {
  for (mlir::Block &block : region) {
    for (mlir::Operation &op : block)
      analyzeParsedOperation(&op, loopMultiplier, state, report, config,
                             replayIterations);
  }
}

static void seedLoopCarriedState(mlir::scf::ForOp forOp,
                                 const AnalysisState &parentState,
                                 AnalysisState &loopState) {
  mlir::Block &body = forOp.getRegion().front();
  mlir::Block::BlockArgListType bodyArgs = body.getArguments();
  unsigned iterArgOffset = 1;
  for (auto [idx, initArg] : llvm::enumerate(forOp.getInitArgs())) {
    if (iterArgOffset + idx >= bodyArgs.size())
      break;
    mlir::BlockArgument regionArg = bodyArgs[iterArgOffset + idx];
    if (auto rootIt = parentState.bufferRoots.find(initArg);
        rootIt != parentState.bufferRoots.end()) {
      loopState.bufferRoots[regionArg] = rootIt->second;
    }
    if (auto producerIt = parentState.valueProducers.find(initArg);
        producerIt != parentState.valueProducers.end()) {
      loopState.valueProducers[regionArg] = producerIt->second;
    }
    if (auto constantIt = parentState.constants.find(initArg);
        constantIt != parentState.constants.end()) {
      loopState.constants[regionArg] = constantIt->second;
    }
    if (auto boundIt = parentState.boundValues.find(initArg);
        boundIt != parentState.boundValues.end()) {
      loopState.boundValues[regionArg] = boundIt->second;
    }
  }
}

static void propagateLoopResults(mlir::scf::ForOp forOp,
                                 const AnalysisState &loopState,
                                 AnalysisState &parentState) {
  auto yieldOp = llvm::dyn_cast<mlir::scf::YieldOp>(
      forOp.getRegion().front().getTerminator());
  if (!yieldOp)
    return;

  for (auto [idx, yielded] : llvm::enumerate(yieldOp.getOperands())) {
    if (idx >= forOp.getNumResults())
      break;
    mlir::Value result = forOp.getResult(idx);
    if (auto producerIt = loopState.valueProducers.find(yielded);
        producerIt != loopState.valueProducers.end()) {
      parentState.valueProducers[result] = producerIt->second;
    }
    if (auto rootIt = loopState.bufferRoots.find(yielded);
        rootIt != loopState.bufferRoots.end()) {
      parentState.bufferRoots[result] = rootIt->second;
    }
    if (auto constantIt = loopState.constants.find(yielded);
        constantIt != loopState.constants.end()) {
      parentState.constants[result] = constantIt->second;
    }
    if (auto boundIt = loopState.boundValues.find(yielded);
        boundIt != loopState.boundValues.end()) {
      parentState.boundValues[result] = boundIt->second;
    }
  }

  parentState.bufferSlots = loopState.bufferSlots;
  parentState.bufferVersions = loopState.bufferVersions;
  parentState.eventProducers = loopState.eventProducers;
  parentState.eventGenerations = loopState.eventGenerations;
  parentState.latestPipeProducer = loopState.latestPipeProducer;
}

static void advanceLoopCarriedState(mlir::scf::ForOp forOp,
                                    AnalysisState &loopState) {
  auto yieldOp = llvm::dyn_cast<mlir::scf::YieldOp>(
      forOp.getRegion().front().getTerminator());
  if (!yieldOp)
    return;

  mlir::Block &body = forOp.getRegion().front();
  mlir::Block::BlockArgListType bodyArgs = body.getArguments();
  unsigned iterArgOffset = 1;
  for (auto [idx, yielded] : llvm::enumerate(yieldOp.getOperands())) {
    if (iterArgOffset + idx >= bodyArgs.size())
      break;
    mlir::BlockArgument regionArg = bodyArgs[iterArgOffset + idx];

    if (auto producerIt = loopState.valueProducers.find(yielded);
        producerIt != loopState.valueProducers.end()) {
      loopState.valueProducers[regionArg] = producerIt->second;
    } else {
      loopState.valueProducers.erase(regionArg);
    }

    if (auto rootIt = loopState.bufferRoots.find(yielded);
        rootIt != loopState.bufferRoots.end()) {
      loopState.bufferRoots[regionArg] = rootIt->second;
    } else {
      loopState.bufferRoots.erase(regionArg);
    }

    if (auto constantIt = loopState.constants.find(yielded);
        constantIt != loopState.constants.end()) {
      loopState.constants[regionArg] = constantIt->second;
    } else {
      loopState.constants.erase(regionArg);
    }

    if (auto boundIt = loopState.boundValues.find(yielded);
        boundIt != loopState.boundValues.end()) {
      loopState.boundValues[regionArg] = boundIt->second;
    } else {
      loopState.boundValues.erase(regionArg);
    }
  }
}

static void analyzeParsedOperation(mlir::Operation *op, int64_t loopMultiplier,
                                   AnalysisState &state,
                                   HIVMAnalysisReport &report,
                                   const HardwareConfig &config,
                                   bool replayIterations) {
  if (captureConstant(op, state))
    return;
  captureDerivedScalarValue(op, state);

  captureBufferMetadata(op, state);

  if (auto funcOp = llvm::dyn_cast<mlir::func::FuncOp>(op)) {
    // Each function (AIC / AIV) runs on its own core in parallel.  Use a
    // fresh per-function state so that pipe-ordering and event tracking do
    // not bleed across functions.  Only preserve arg-bindings.
    AnalysisState funcState;
    funcState.argBindings = state.argBindings;
    unsigned userArgIndex = 0;
    for (auto [idx, arg] : llvm::enumerate(funcOp.getArguments())) {
      if (funcOp.getArgAttr(idx, "hacc.arg_type"))
        continue;
      auto bindIt =
          funcState.argBindings.find("arg" + std::to_string(userArgIndex++));
      if (bindIt != funcState.argBindings.end())
        funcState.boundValues[arg] = bindIt->second;
    }
    analyzeParsedRegion(funcOp.getBody(), loopMultiplier, funcState, report,
                        config, replayIterations);
    return;
  }

  if (auto forOp = llvm::dyn_cast<mlir::scf::ForOp>(op)) {
    int64_t tripCount = 1;
    bool hasConcreteTripCount = parseForTripCount(forOp, state, tripCount);
    int64_t lowerBound = 0;
    int64_t step = 1;
    bool hasConcreteInductionValue =
        resolveMLIRValue(forOp.getLowerBound(), state, lowerBound) &&
        resolveMLIRValue(forOp.getStep(), state, step);
    int64_t nestedMultiplier = loopMultiplier * std::max<int64_t>(tripCount, 1);
    report.maxLoopMultiplier =
        std::max<int64_t>(report.maxLoopMultiplier, nestedMultiplier);
    AnalysisState loopState = state;
    seedLoopCarriedState(forOp, state, loopState);
    if (replayIterations && hasConcreteTripCount && tripCount > 1) {
      for (int64_t iter = 0; iter < tripCount; ++iter) {
        if (hasConcreteInductionValue)
          loopState.boundValues[forOp.getInductionVar()] =
              lowerBound + iter * step;
        analyzeParsedRegion(op->getRegion(0), loopMultiplier, loopState, report,
                            config, replayIterations);
        if (iter + 1 < tripCount)
          advanceLoopCarriedState(forOp, loopState);
      }
    } else {
      if (hasConcreteInductionValue)
        loopState.boundValues[forOp.getInductionVar()] = lowerBound;
      int64_t bodyMultiplier = (replayIterations && hasConcreteTripCount)
                                   ? loopMultiplier
                                   : nestedMultiplier;
      analyzeParsedRegion(op->getRegion(0), bodyMultiplier, loopState, report,
                          config, replayIterations);
    }
    propagateLoopResults(forOp, loopState, state);
    return;
  }

  if (startsWithHivmOp(op)) {
    std::string opText = renderOperation(op);
    ParsedOp parsed;
#ifdef TRITONSIM_HAS_BISHENGIR_HIVM
    if (!populateTypedHivmOp(op, parsed))
      return;
    parsed.op.lineNumber = getLineNumberFromLocation(op->getLoc());
#else
    if (!populateGenericHivmOp(op, parsed))
      return;
    parsed.op.lineNumber = getLineNumberFromLocation(op->getLoc());
#endif
    parsed.op.loopMultiplier = loopMultiplier;
    parsed.op.text = opText;
    if (parsed.op.opName.empty())
      parsed.op.opName = getLeafOpName(op).str();
    parsed.mlirResults.assign(op->result_begin(), op->result_end());

    for (mlir::Value operand : op->getOperands()) {
      auto it = state.valueProducers.find(operand);
      if (it != state.valueProducers.end())
        parsed.op.dependsOn.push_back(it->second);
    }

    if (op->getNumResults() > 0) {
      parsed.op.bytes = inferValueBytes(op->getResult(0));
      parsed.op.elements = inferValueElements(op->getResult(0));
    }
    if (parsed.op.bytes == 0 || parsed.op.elements == 0) {
      for (mlir::Value operand : op->getOperands()) {
        parsed.op.bytes = std::max(parsed.op.bytes, inferValueBytes(operand));
        parsed.op.elements =
            std::max(parsed.op.elements, inferValueElements(operand));
      }
    }
    if (parsed.op.bytes == 0)
      parsed.op.bytes = parseMemRefBytes(parsed.op.text);
    if (parsed.op.elements == 0)
      parsed.op.elements = parseMemRefElementCount(parsed.op.text);
    attachSyncMetadata(parsed);
    attachBufferAccessMetadata(op, parsed, state);
    parsed.op.duration = estimateDuration(parsed, config);
    ingestParsedOp(parsed, state, report, config);
  } else if (!llvm::isa<mlir::scf::YieldOp>(op) &&
             !llvm::isa<mlir::func::ReturnOp>(op) && op->getNumResults() > 0) {
    // Non-hivm ops with results are scalar operations (arith, memref, etc.)
    // that execute on PIPE_S.  They produce SSA values consumed by hivm.hir
    // ops, creating cross-pipe dependencies via def-use chains.
    ParsedOp parsed;
    parsed.op.opName = getLeafOpName(op).str();
    parsed.op.pipe = HIVMPipe::Scalar;
    parsed.op.loopMultiplier = loopMultiplier;
    parsed.op.lineNumber = getLineNumberFromLocation(op->getLoc());
    parsed.op.text = renderOperation(op);
    parsed.op.duration = 1; // scalar ops take 1 cycle
    parsed.mlirResults.assign(op->result_begin(), op->result_end());

    // Determine core type from function context
    if (auto parentFunc = op->getParentOfType<mlir::func::FuncOp>()) {
      llvm::StringRef funcName = parentFunc.getName();
      if (funcName.contains("aic") || funcName.contains("AIC") ||
          funcName.contains("cube"))
        parsed.op.coreType = "CUBE";
      else if (funcName.contains("aiv") || funcName.contains("AIV") ||
               funcName.contains("vector") || funcName.contains("mix"))
        parsed.op.coreType = "VECTOR";
    }

    // SSA dependencies: if this op uses a value produced by a previous op
    for (mlir::Value operand : op->getOperands()) {
      auto it = state.valueProducers.find(operand);
      if (it != state.valueProducers.end())
        parsed.op.dependsOn.push_back(it->second);
    }

    ingestParsedOp(parsed, state, report, config);
  }

  for (mlir::Region &region : op->getRegions())
    analyzeParsedRegion(region, loopMultiplier, state, report, config,
                        replayIterations);
}

static void finalizeScheduledReport(HIVMAnalysisReport &report,
                                    const HardwareConfig &config) {
  std::map<HIVMPipe, int64_t> pipeAvailableAt;
  for (HIVMOp &op : report.operations) {
    int64_t earliest = 0;
    for (size_t depId : op.dependsOn) {
      if (depId < report.operations.size())
        earliest = std::max(earliest, report.operations[depId].endCycle);
    }

    if (op.isBarrier) {
      int64_t start = earliest;
      if (op.pipe == HIVMPipe::All) {
        for (const auto &entry : pipeAvailableAt)
          start = std::max(start, entry.second);
        op.startCycle = start;
        op.endCycle = start + op.duration;
        for (auto &entry : pipeAvailableAt)
          entry.second = op.endCycle;
      } else {
        start = std::max(start, pipeAvailableAt[op.pipe]);
        op.startCycle = start;
        op.endCycle = start + op.duration;
        pipeAvailableAt[op.pipe] = op.endCycle;
      }
    } else if (op.pipe == HIVMPipe::Unknown) {
      op.startCycle = earliest;
      op.endCycle = earliest + op.duration;
    } else {
      int64_t start = std::max(earliest, pipeAvailableAt[op.pipe]);
      op.startCycle = start;
      op.endCycle = start + op.duration;
      pipeAvailableAt[op.pipe] = op.endCycle;
    }

    report.oneIterationCycles =
        std::max(report.oneIterationCycles, op.endCycle);
    report.totalBusyCycles += op.duration;
    report.opCount++;
    if (op.pipe == HIVMPipe::Scalar && op.opName != "set_flag" &&
        op.opName != "wait_flag" && op.opName != "sync_block_set" &&
        op.opName != "sync_block_wait" && op.opName != "sync_block" &&
        op.opName != "pipe_barrier" && op.opName != "get_block_idx" &&
        op.opName != "get_block_num" && op.opName != "get_sub_block_idx" &&
        op.opName != "get_sub_block_num" && op.opName != "set_mask_norm" &&
        op.opName != "pointer_cast" && op.opName != "convert_layout")
      report.unknownOpCount++;
    if (op.isSyncOp) {
      report.syncCycles += op.duration;
      report.syncOpCount++;
    }
    if (op.isBarrier) {
      report.barrierCycles += op.duration;
      report.barrierCount++;
    }
    if (op.pipe != HIVMPipe::All && op.pipe != HIVMPipe::Unknown) {
      report.pipeBusyCycles[op.pipe] += op.duration;
      report.weightedPipeCycles[op.pipe] += op.duration * op.loopMultiplier;
    }
  }

  int64_t globalBarrierWeightedCycles = 0;
  for (const HIVMOp &op : report.operations) {
    if (op.isBarrier && op.pipe == HIVMPipe::All)
      globalBarrierWeightedCycles += op.duration * op.loopMultiplier;
  }
  for (const auto &entry : report.weightedPipeCycles)
    report.weightedCycles = std::max(report.weightedCycles, entry.second);
  report.weightedCycles += globalBarrierWeightedCycles;
  if (report.weightedCycles == 0)
    report.weightedCycles = report.oneIterationCycles;
}

struct CompletionEvent {
  int64_t time = 0;
  size_t opId = 0;

  bool operator>(const CompletionEvent &other) const {
    return std::tie(time, opId) > std::tie(other.time, other.opId);
  }
};

struct BufferSlotState {
  int64_t writableAt = 0;
  int64_t readableAt = 0;
  int64_t version = 0;
};

struct BufferRootState {
  std::vector<BufferSlotState> slots;
  int64_t latestReadableAt = 0;
  int64_t latestVersion = 0;
  std::map<int64_t, int64_t> versionReadableAt;
  std::map<int64_t, size_t> versionToSlot;
};

static llvm::StringRef getSyncBlockSourceCore(const HIVMOp &op) {
  bool isCubeCore = op.coreType == "CUBE" || op.coreType == "AIC";
  if (op.opName == "sync_block_set")
    return isCubeCore ? "AIC" : "AIV";
  if (op.opName == "sync_block_wait")
    return isCubeCore ? "AIV" : "AIC";
  return "";
}

static void normalizeSyncBlockGenerations(HIVMAnalysisReport &report) {
  std::map<std::pair<std::string, std::string>, int64_t> setGeneration;
  std::map<std::pair<std::string, std::string>, int64_t> waitGeneration;
  for (HIVMOp &op : report.operations) {
    if ((op.opName != "sync_block_set" && op.opName != "sync_block_wait") ||
        op.eventId.empty())
      continue;
    llvm::StringRef sourceCore = getSyncBlockSourceCore(op);
    if (sourceCore.empty())
      continue;
    auto key = std::make_pair(op.eventId, sourceCore.str());
    if (op.opName == "sync_block_set")
      op.eventGeneration = ++setGeneration[key];
    else
      op.eventGeneration = ++waitGeneration[key];
  }
}

/// After generation normalization, wire explicit dependency edges from each
/// sync_block_set to its matching sync_block_wait so the DES respects
/// cross-core ordering.  Without this, the wait may be scheduled before the
/// set completes (they live in different func::FuncOps with independent state).
static void wireCrossCoreSyncDependencies(HIVMAnalysisReport &report) {
  // Key: (eventId, sourceCore, generation) → set-op id
  using SyncKey = std::tuple<std::string, std::string, int64_t>;
  std::map<SyncKey, size_t> setOpById;
  for (HIVMOp &op : report.operations) {
    if (op.opName != "sync_block_set" || op.eventId.empty())
      continue;
    llvm::StringRef sourceCore = getSyncBlockSourceCore(op);
    if (sourceCore.empty())
      continue;
    SyncKey key{op.eventId, sourceCore.str(), op.eventGeneration};
    setOpById[key] = op.id;
  }
  for (HIVMOp &op : report.operations) {
    if (op.opName != "sync_block_wait" || op.eventId.empty())
      continue;
    llvm::StringRef sourceCore = getSyncBlockSourceCore(op);
    if (sourceCore.empty())
      continue;
    // sync_block_wait's sourceCore returns the core that *set* the flag
    // (the opposite core), which matches the set-op's sourceCore.
    SyncKey key{op.eventId, sourceCore.str(), op.eventGeneration};
    auto it = setOpById.find(key);
    if (it != setOpById.end())
      op.dependsOn.push_back(it->second);
  }
}

static void finalizeDiscreteEventReport(HIVMAnalysisReport &report,
                                        const HardwareConfig &config) {
  normalizeSyncBlockGenerations(report);
  wireCrossCoreSyncDependencies(report);

  const size_t numOps = report.operations.size();
  if (numOps == 0) {
    report.weightedCycles = 0;
    return;
  }

  std::vector<size_t> remainingDeps(numOps, 0);
  std::vector<int64_t> readyAt(numOps, 0);
  std::vector<llvm::SmallVector<size_t, 4>> successors(numOps);
  std::vector<bool> queued(numOps, false);
  std::vector<bool> started(numOps, false);
  std::vector<bool> completed(numOps, false);
  std::deque<size_t> readyOps;
  std::priority_queue<CompletionEvent, std::vector<CompletionEvent>,
                      std::greater<CompletionEvent>>
      completions;
  std::map<HIVMPipe, int64_t> pipeAvailableAt;
  std::map<EventInstanceKey, int64_t> flagEventVisibleAt;
  std::map<EventInstanceKey, int64_t> blockSyncVisibleAt;
  std::map<std::string, BufferRootState> bufferStates;
  std::map<size_t, std::vector<std::pair<std::string, size_t>>>
      writeSlotAssignments;
  size_t completedCount = 0;

  for (const HIVMOp &op : report.operations) {
    for (const std::string &root : op.writeBuffers) {
      auto &state = bufferStates[root];
      if (state.slots.empty()) {
        int64_t count = std::max<int64_t>(1, op.multiBufferSlots);
        for (int64_t i = 0; i < count; ++i)
          state.slots.push_back(BufferSlotState{});
      }
      state.versionReadableAt.emplace(0, 0);
    }
    for (const std::string &root : op.readBuffers) {
      auto [it, inserted] = bufferStates.try_emplace(root, BufferRootState{});
      it->second.versionReadableAt.emplace(0, 0);
    }
  }

  for (size_t opId = 0; opId < numOps; ++opId) {
    HIVMOp &op = report.operations[opId];
    remainingDeps[opId] = op.dependsOn.size();
    for (size_t depId : op.dependsOn) {
      if (depId < numOps)
        successors[depId].push_back(opId);
    }
    if (remainingDeps[opId] == 0) {
      readyOps.push_back(opId);
      queued[opId] = true;
    }
  }

  auto completeOp = [&](size_t opId, int64_t time) {
    if (completed[opId])
      return;
    HIVMOp &op = report.operations[opId];
    op.endCycle = time;
    completed[opId] = true;
    ++completedCount;
    report.oneIterationCycles =
        std::max(report.oneIterationCycles, op.endCycle);
    report.totalBusyCycles += op.duration;
    report.opCount++;
    if (op.pipe == HIVMPipe::Scalar && op.opName != "set_flag" &&
        op.opName != "wait_flag" && op.opName != "sync_block_set" &&
        op.opName != "sync_block_wait" && op.opName != "sync_block" &&
        op.opName != "pipe_barrier" && op.opName != "get_block_idx" &&
        op.opName != "get_block_num" && op.opName != "get_sub_block_idx" &&
        op.opName != "get_sub_block_num" && op.opName != "set_mask_norm" &&
        op.opName != "pointer_cast" && op.opName != "convert_layout")
      report.unknownOpCount++;
    if (op.isSyncOp) {
      report.syncCycles += op.duration;
      report.syncOpCount++;
      if ((op.opName == "set_flag" || op.opName == "sync_block_set") &&
          !op.eventId.empty()) {
        EventInstanceKey key{{op.senderPipe, op.receiverPipe, op.eventId},
                             op.eventGeneration};
        if (op.opName == "sync_block_set")
          blockSyncVisibleAt[key] = time;
        else
          flagEventVisibleAt[key] = time;
      }
    }
    auto slotIt = writeSlotAssignments.find(opId);
    if (slotIt != writeSlotAssignments.end()) {
      for (const auto &[root, slotIndex] : slotIt->second) {
        auto rootIt = bufferStates.find(root);
        if (rootIt == bufferStates.end() ||
            slotIndex >= rootIt->second.slots.size())
          continue;
        BufferRootState &state = rootIt->second;
        BufferSlotState &slot = state.slots[slotIndex];
        auto rootVersionIt = llvm::find(op.writeBuffers, root);
        if (rootVersionIt == op.writeBuffers.end())
          continue;
        size_t bufferIdx =
            std::distance(op.writeBuffers.begin(), rootVersionIt);
        if (bufferIdx >= op.writeBufferVersions.size())
          continue;
        int64_t version = op.writeBufferVersions[bufferIdx];
        slot.readableAt = time;
        slot.version = version;
        state.latestVersion = std::max(state.latestVersion, version);
        state.latestReadableAt = std::max(state.latestReadableAt, time);
        state.versionReadableAt[version] = time;
        state.versionToSlot[version] = slotIndex;
      }
    }
    if (op.isBarrier) {
      report.barrierCycles += op.duration;
      report.barrierCount++;
    }
    if (op.pipe != HIVMPipe::All && op.pipe != HIVMPipe::Unknown) {
      report.pipeBusyCycles[op.pipe] += op.duration;
      report.weightedPipeCycles[op.pipe] += op.duration * op.loopMultiplier;
    }
    for (size_t succId : successors[opId]) {
      readyAt[succId] = std::max(readyAt[succId], time);
      if (remainingDeps[succId] > 0)
        --remainingDeps[succId];
      if (remainingDeps[succId] == 0 && !queued[succId]) {
        readyOps.push_back(succId);
        queued[succId] = true;
      }
    }
  };

  auto computeStartTime = [&](const HIVMOp &op) -> int64_t {
    int64_t start = readyAt[op.id];
    for (size_t idx = 0; idx < op.readBuffers.size(); ++idx) {
      const std::string &root = op.readBuffers[idx];
      auto it = bufferStates.find(root);
      if (it == bufferStates.end())
        continue;
      int64_t version =
          idx < op.readBufferVersions.size() ? op.readBufferVersions[idx] : 0;
      auto readableIt = it->second.versionReadableAt.find(version);
      if (readableIt != it->second.versionReadableAt.end())
        start = std::max(start, readableIt->second);
      else if (it->second.latestVersion >= version)
        start = std::max(start, it->second.latestReadableAt);
    }
    for (const std::string &root : op.writeBuffers) {
      auto it = bufferStates.find(root);
      if (it != bufferStates.end() && !it->second.slots.empty()) {
        int64_t slotReady = std::numeric_limits<int64_t>::max();
        for (const BufferSlotState &slot : it->second.slots)
          slotReady = std::min(slotReady, slot.writableAt);
        start = std::max(start, slotReady);
      }
    }
    if ((op.opName == "wait_flag" || op.opName == "sync_block_wait") &&
        !op.eventId.empty()) {
      EventInstanceKey key{{op.senderPipe, op.receiverPipe, op.eventId},
                           op.eventGeneration};
      auto &visibleAt = op.opName == "sync_block_wait" ? blockSyncVisibleAt
                                                       : flagEventVisibleAt;
      auto it = visibleAt.find(key);
      if (it != visibleAt.end())
        start = std::max(start, it->second);
    }
    if (op.pipe == HIVMPipe::Unknown)
      return start;
    if (op.isBarrier && op.pipe == HIVMPipe::All) {
      if (op.coreType.empty()) {
        for (const auto &entry : pipeAvailableAt)
          start = std::max(start, entry.second);
      } else {
        for (const auto &entry : pipeAvailableAt) {
          if (pipeBelongsToCore(entry.first, op.coreType))
            start = std::max(start, entry.second);
        }
      }
      return start;
    }
    return std::max(start, pipeAvailableAt[op.pipe]);
  };

  auto startOp = [&](size_t opId, int64_t startTime) {
    HIVMOp &op = report.operations[opId];
    started[opId] = true;
    op.startCycle = startTime;
    const int64_t endTime = startTime + op.duration;
    for (size_t idx = 0; idx < op.readBuffers.size(); ++idx) {
      const std::string &root = op.readBuffers[idx];
      auto it = bufferStates.find(root);
      if (it == bufferStates.end())
        continue;
      int64_t version =
          idx < op.readBufferVersions.size() ? op.readBufferVersions[idx] : 0;
      if (version <= 0)
        continue;
      auto slotIt = it->second.versionToSlot.find(version);
      if (slotIt == it->second.versionToSlot.end())
        continue;
      size_t slotIndex = slotIt->second;
      if (slotIndex >= it->second.slots.size())
        continue;
      it->second.slots[slotIndex].writableAt =
          std::max(it->second.slots[slotIndex].writableAt, endTime);
    }
    for (const std::string &root : op.writeBuffers) {
      auto it = bufferStates.find(root);
      if (it == bufferStates.end() || it->second.slots.empty())
        continue;
      BufferRootState &state = it->second;
      size_t bestSlot = 0;
      int64_t bestTime = state.slots.front().writableAt;
      for (size_t i = 1; i < state.slots.size(); ++i) {
        if (state.slots[i].writableAt < bestTime) {
          bestTime = state.slots[i].writableAt;
          bestSlot = i;
        }
      }
      state.slots[bestSlot].writableAt = endTime;
      writeSlotAssignments[opId].push_back({root, bestSlot});
    }
    if (op.pipe != HIVMPipe::Unknown) {
      if (op.isBarrier && op.pipe == HIVMPipe::All) {
        auto barrierPipes = getCoreBarrierPipes(op.coreType);
        if (barrierPipes.empty()) {
          for (auto &entry : pipeAvailableAt)
            entry.second = endTime;
        } else {
          for (HIVMPipe barrierPipe : barrierPipes)
            pipeAvailableAt[barrierPipe] = endTime;
        }
      } else {
        pipeAvailableAt[op.pipe] = endTime;
      }
    }
    if (op.duration == 0)
      completeOp(opId, endTime);
    else
      completions.push({endTime, opId});
  };

  int64_t currentTime = 0;
  while (completedCount < numOps) {
    bool startedAny = false;
    size_t readyCount = readyOps.size();
    for (size_t i = 0; i < readyCount; ++i) {
      size_t opId = readyOps.front();
      readyOps.pop_front();
      HIVMOp &op = report.operations[opId];
      if (started[opId] || completed[opId])
        continue;
      int64_t startTime = computeStartTime(op);
      if (startTime <= currentTime) {
        startOp(opId, currentTime);
        startedAny = true;
      } else {
        readyOps.push_back(opId);
      }
    }

    while (!completions.empty() && completions.top().time <= currentTime) {
      size_t opId = completions.top().opId;
      completions.pop();
      if (!completed[opId]) {
        completeOp(opId, currentTime);
      }
    }

    if (startedAny)
      continue;

    int64_t nextTime = std::numeric_limits<int64_t>::max();
    if (!completions.empty())
      nextTime = std::min(nextTime, completions.top().time);
    for (size_t opId : readyOps)
      nextTime = std::min(nextTime, computeStartTime(report.operations[opId]));

    if (nextTime == std::numeric_limits<int64_t>::max())
      break;
    currentTime = std::max(currentTime, nextTime);

    while (!completions.empty() && completions.top().time <= currentTime) {
      size_t opId = completions.top().opId;
      completions.pop();
      if (!completed[opId]) {
        completeOp(opId, currentTime);
      }
    }
  }

  int64_t globalBarrierWeightedCycles = 0;
  for (const HIVMOp &op : report.operations) {
    if (op.isBarrier && op.pipe == HIVMPipe::All)
      globalBarrierWeightedCycles += op.duration * op.loopMultiplier;
  }
  for (const auto &entry : report.weightedPipeCycles)
    report.weightedCycles = std::max(report.weightedCycles, entry.second);
  report.weightedCycles += globalBarrierWeightedCycles;
  if (report.weightedCycles == 0)
    report.weightedCycles = report.oneIterationCycles;
}

} // namespace

llvm::StringRef HIVMAnalyzer::stringifySchedulerMode(HIVMSchedulerMode mode) {
  switch (mode) {
  case HIVMSchedulerMode::Static:
    return "static";
  case HIVMSchedulerMode::DES:
    return "des";
  }
  return "static";
}

llvm::StringRef HIVMAnalyzer::stringifyPipe(HIVMPipe pipe) {
  switch (pipe) {
  case HIVMPipe::Vector:
    return "PIPE_V";
  case HIVMPipe::VectorMTE2:
    return "PIPE_MTE2_V";
  case HIVMPipe::CubeMTE2:
    return "PIPE_MTE2_C";
  case HIVMPipe::MTE3:
    return "PIPE_MTE3";
  case HIVMPipe::Scalar:
    return "PIPE_S";
  case HIVMPipe::FixPipe:
    return "PIPE_FIX";
  case HIVMPipe::Cube:
    return "PIPE_M";
  case HIVMPipe::MTE1:
    return "PIPE_MTE1";
  case HIVMPipe::All:
    return "PIPE_ALL";
  case HIVMPipe::Unknown:
    return "PIPE_UNKNOWN";
  }
  return "PIPE_UNKNOWN";
}

HIVMAnalyzer::HIVMAnalyzer(const HardwareConfig &config,
                           llvm::StringRef argBindings,
                           HIVMSchedulerMode schedulerMode)
    : config(config), argBindingsStr(argBindings.str()),
      schedulerMode(schedulerMode) {}

bool HIVMAnalyzer::analyzeFile(llvm::StringRef path, HIVMAnalysisReport &report,
                               std::string &error) const {
  auto fileOrErr = llvm::MemoryBuffer::getFile(path);
  if (!fileOrErr) {
    error = "failed to read HIVM file: " + path.str();
    return false;
  }

  llvm::StringRef rawBuffer = fileOrErr.get()->getBuffer();
  std::string sanitized = sanitizeMlirBuffer(rawBuffer);
  report = HIVMAnalysisReport();
  report.sourcePath = path.str();
  report.sourceMode = "direct-hivm";
  report.schedulerMode = schedulerMode;

  {
    mlir::DialectRegistry registry;
    registry.insert<mlir::BuiltinDialect, mlir::affine::AffineDialect,
                    mlir::func::FuncDialect, mlir::arith::ArithDialect,
                    mlir::memref::MemRefDialect, mlir::scf::SCFDialect>();
#ifdef TRITONSIM_HAS_BISHENGIR_HIVM
    registry.insert<mlir::annotation::AnnotationDialect,
                    mlir::hacc::HACCDialect, mlir::hivm::HIVMDialect>();
#endif
    mlir::MLIRContext context(registry);
    context.allowUnregisteredDialects();

    std::string parseDiagnostics;
    mlir::ScopedDiagnosticHandler diagHandler(
        &context, [&](mlir::Diagnostic &diag) {
          llvm::raw_string_ostream os(parseDiagnostics);
          diag.print(os);
          os << "\n";
          return mlir::success();
        });

    llvm::SmallVector<llvm::StringRef, 2> parseCandidates;
    parseCandidates.push_back(rawBuffer);
    if (sanitized != rawBuffer)
      parseCandidates.push_back(sanitized);
    for (llvm::StringRef buffer : parseCandidates) {
      if (auto module =
              mlir::parseSourceString<mlir::ModuleOp>(buffer, &context)) {
        if (!analyzeModule(*module, report, error))
          return false;
        report.sourcePath = path.str();
        report.sourceMode = "direct-hivm";
        return true;
      }
    }
    error = "failed to parse HIVM MLIR module";
    if (!parseDiagnostics.empty())
      error += ":\n" + parseDiagnostics;
    return false;
  }
}

bool HIVMAnalyzer::analyzeModule(mlir::ModuleOp module,
                                 HIVMAnalysisReport &report,
                                 std::string &error) const {
  if (!module) {
    error = "null module passed to HIVM analysis";
    return false;
  }

  report = HIVMAnalysisReport();
  report.sourcePath = "<module>";
  report.sourceMode = "mlir-pass";
  report.schedulerMode = schedulerMode;

  AnalysisState state;
  state.argBindings = parseArgBindings(argBindingsStr);
  analyzeParsedRegion(module.getBodyRegion(), 1, state, report, config,
                      schedulerMode == HIVMSchedulerMode::DES);
  if (schedulerMode == HIVMSchedulerMode::DES)
    finalizeDiscreteEventReport(report, config);
  else
    finalizeScheduledReport(report, config);
  return true;
}

void HIVMAnalysisReport::print(llvm::raw_ostream &os,
                               const HardwareConfig &config) const {
  os << "=== HIVM Analysis ===\n";
  os << "Source mode: " << sourceMode << "\n";
  os << "Source: " << sourcePath << "\n";
  os << "Scheduler: " << HIVMAnalyzer::stringifySchedulerMode(schedulerMode)
     << "\n";
  os << "Hardware: " << config.getName() << " @ "
     << llvm::format("%.2f", config.getClockFrequencyGHz()) << " GHz\n\n";

  os << "Summary:\n";
  os << "  Operations: " << opCount << "\n";
  os << "  Sync ops: " << syncOpCount << "\n";
  os << "  Barriers: " << barrierCount << "\n";
  os << "  Scalar/unknown fallback ops: " << unknownOpCount << "\n";
  os << "  One-iteration critical path: " << oneIterationCycles << " cycles ("
     << llvm::format("%.3f", config.cyclesToMicroseconds(oneIterationCycles))
     << " us)\n";
  os << "  Weighted pipe max: " << weightedCycles << " cycles ("
     << llvm::format("%.3f", config.cyclesToMicroseconds(weightedCycles))
     << " us)\n";
  os << "  Sync cycles: " << syncCycles << "\n";
  os << "  Barrier cycles: " << barrierCycles << "\n";
  os << "  Max loop multiplier: " << maxLoopMultiplier << "\n\n";

  os << "Per-pipe utilization (one iteration):\n";
  for (const auto &entry : pipeBusyCycles) {
    double util = oneIterationCycles > 0 ? static_cast<double>(entry.second) /
                                               oneIterationCycles * 100.0
                                         : 0.0;
    os << "  " << HIVMAnalyzer::stringifyPipe(entry.first) << ": "
       << entry.second << " cycles, " << llvm::format("%.1f", util) << "%\n";
  }

  os << "\nPer-pipe weighted cycles:\n";
  for (const auto &entry : weightedPipeCycles) {
    os << "  " << HIVMAnalyzer::stringifyPipe(entry.first) << ": "
       << entry.second << "\n";
  }

  os << "\nTop operations by weighted cost:\n";
  std::vector<const HIVMOp *> sorted;
  sorted.reserve(operations.size());
  for (const HIVMOp &op : operations)
    sorted.push_back(&op);
  std::sort(sorted.begin(), sorted.end(),
            [](const HIVMOp *lhs, const HIVMOp *rhs) {
              return lhs->duration * lhs->loopMultiplier >
                     rhs->duration * rhs->loopMultiplier;
            });

  size_t limit = std::min<size_t>(10, sorted.size());
  for (size_t i = 0; i < limit; ++i) {
    const HIVMOp *op = sorted[i];
    os << "  line " << op->lineNumber << " " << op->opName << " ["
       << HIVMAnalyzer::stringifyPipe(op->pipe)
       << "]: " << (op->duration * op->loopMultiplier) << " weighted cycles";
    if (!op->coreType.empty())
      os << ", core=" << op->coreType;
    if (op->bytes > 0)
      os << ", " << op->bytes << " bytes";
    if (op->elements > 0)
      os << ", " << op->elements << " elems";
    os << "\n";
  }
}

void HIVMAnalysisReport::emitPerfettoTrace(llvm::raw_ostream &os,
                                           const HardwareConfig &config) const {
  // Assign each pipe a unique tid.  Pipes are grouped into AIC (Cube core)
  // and AIV (Vector core) processes so that Perfetto renders them separately.
  //   AIC pid=1 : Cube, MTE1, CubeMTE2, FixPipe, Scalar(AIC)
  //   AIV pid=2 : Vector, VectorMTE2, MTE3, Scalar(AIV)
  //   Shared pid=3 : All, Unknown  (cross-core barriers / unclassified)
  constexpr int kPidAIC = 1;
  constexpr int kPidAIV = 2;
  constexpr int kPidShared = 3;

  auto pipeTid = [](HIVMPipe pipe) -> int {
    switch (pipe) {
    case HIVMPipe::Cube:
      return 1;
    case HIVMPipe::MTE1:
      return 2;
    case HIVMPipe::CubeMTE2:
      return 3;
    case HIVMPipe::FixPipe:
      return 4;
    case HIVMPipe::Scalar:
      return 5;
    case HIVMPipe::Vector:
      return 1;
    case HIVMPipe::VectorMTE2:
      return 2;
    case HIVMPipe::MTE3:
      return 3;
    case HIVMPipe::All:
      return 1;
    case HIVMPipe::Unknown:
      return 5; // Coalesce with Scalar — metadata/address ops
    }
    return 5;
  };

  auto pipePid = [&](HIVMPipe pipe, llvm::StringRef coreType) -> int {
    switch (pipe) {
    case HIVMPipe::Cube:
    case HIVMPipe::MTE1:
    case HIVMPipe::CubeMTE2:
    case HIVMPipe::FixPipe:
      return kPidAIC;
    case HIVMPipe::Vector:
    case HIVMPipe::VectorMTE2:
    case HIVMPipe::MTE3:
      return kPidAIV;
    case HIVMPipe::Scalar:
      // Scalar exists on both cores; assign by op's core_type.
      if (coreType == "CUBE" || coreType == "AIC")
        return kPidAIC;
      return kPidAIV;
    case HIVMPipe::All:
    case HIVMPipe::Unknown:
      if (coreType == "CUBE" || coreType == "AIC")
        return kPidAIC;
      if (coreType == "VECTOR" || coreType == "AIV")
        return kPidAIV;
      return kPidShared;
    }
    return kPidShared;
  };

  auto cyclesToTraceUs = [&](int64_t cycles) -> double {
    return config.cyclesToMicroseconds(cycles);
  };
  auto joinStrings = [](const std::vector<std::string> &values) {
    std::string joined;
    llvm::raw_string_ostream ss(joined);
    for (size_t i = 0; i < values.size(); ++i) {
      if (i)
        ss << ";";
      ss << values[i];
    }
    ss.flush();
    return joined;
  };
  auto joinInts = [](const std::vector<int64_t> &values) {
    std::string joined;
    llvm::raw_string_ostream ss(joined);
    for (size_t i = 0; i < values.size(); ++i) {
      if (i)
        ss << ";";
      ss << values[i];
    }
    ss.flush();
    return joined;
  };

  os << "{\n  \"traceEvents\": [\n";
  bool first = true;
  auto emitComma = [&]() {
    if (!first)
      os << ",\n";
    first = false;
  };

  // Process names for AIC, AIV, and Shared groups.
  emitComma();
  os << "    {\"ph\":\"M\",\"pid\":" << kPidAIC
     << ",\"tid\":0,\"name\":\"process_name\",\"args\":{\"name\":\"AIC (Cube "
        "Core)\"}}";
  emitComma();
  os << "    {\"ph\":\"M\",\"pid\":" << kPidAIV
     << ",\"tid\":0,\"name\":\"process_name\",\"args\":{\"name\":\"AIV (Vector "
        "Core)\"}}";
  emitComma();
  os << "    {\"ph\":\"M\",\"pid\":" << kPidShared
     << ",\"tid\":0,\"name\":\"process_name\",\"args\":{\"name\":\"Shared\"}}";

  // AIC pipes: Cube, MTE1, CubeMTE2, FixPipe, Scalar(AIC)
  for (HIVMPipe pipe : {HIVMPipe::Cube, HIVMPipe::MTE1, HIVMPipe::CubeMTE2,
                        HIVMPipe::FixPipe}) {
    emitComma();
    os << "    {\"ph\":\"M\",\"pid\":" << kPidAIC
       << ",\"tid\":" << pipeTid(pipe)
       << ",\"name\":\"thread_name\",\"args\":{\"name\":\""
       << HIVMAnalyzer::stringifyPipe(pipe) << "\"}}";
  }
  // Scalar thread under AIC
  emitComma();
  os << "    {\"ph\":\"M\",\"pid\":" << kPidAIC
     << ",\"tid\":" << pipeTid(HIVMPipe::Scalar)
     << ",\"name\":\"thread_name\",\"args\":{\"name\":\"Scalar\"}}";

  // AIV pipes: Vector, VectorMTE2, MTE3, Scalar(AIV)
  for (HIVMPipe pipe :
       {HIVMPipe::Vector, HIVMPipe::VectorMTE2, HIVMPipe::MTE3}) {
    emitComma();
    os << "    {\"ph\":\"M\",\"pid\":" << kPidAIV
       << ",\"tid\":" << pipeTid(pipe)
       << ",\"name\":\"thread_name\",\"args\":{\"name\":\""
       << HIVMAnalyzer::stringifyPipe(pipe) << "\"}}";
  }
  // Scalar thread under AIV
  emitComma();
  os << "    {\"ph\":\"M\",\"pid\":" << kPidAIV
     << ",\"tid\":" << pipeTid(HIVMPipe::Scalar)
     << ",\"name\":\"thread_name\",\"args\":{\"name\":\"Scalar\"}}";

  // Shared process: cross-core barrier track
  emitComma();
  os << "    {\"ph\":\"M\",\"pid\":" << kPidShared
     << ",\"tid\":" << pipeTid(HIVMPipe::All)
     << ",\"name\":\"thread_name\",\"args\":{\"name\":\""
     << HIVMAnalyzer::stringifyPipe(HIVMPipe::All) << "\"}}";

  for (const HIVMOp &op : operations) {
    // Skip zero-cycle metadata ops that are not real scheduled work.
    if (op.opName == "pointer_cast" || op.opName == "convert_layout")
      continue;
    emitComma();
    os << "    {\"ph\":\"X\",\"pid\":" << pipePid(op.pipe, op.coreType)
       << ",\"tid\":" << pipeTid(op.pipe)
       << ",\"ts\":" << llvm::format("%.3f", cyclesToTraceUs(op.startCycle))
       << ",\"dur\":" << llvm::format("%.3f", cyclesToTraceUs(op.duration))
       << ",\"name\":\"" << op.opName << "\",\"args\":{"
       << "\"line\":" << op.lineNumber << ",\"cycles\":" << op.duration
       << ",\"loop_multiplier\":" << op.loopMultiplier
       << ",\"bytes\":" << op.bytes << ",\"elements\":" << op.elements
       << ",\"event_id\":\"" << op.eventId << "\""
       << ",\"event_generation\":" << op.eventGeneration
       << ",\"sender_pipe\":\"" << HIVMAnalyzer::stringifyPipe(op.senderPipe)
       << "\""
       << ",\"receiver_pipe\":\""
       << HIVMAnalyzer::stringifyPipe(op.receiverPipe) << "\""
       << ",\"read_buffers\":\"" << joinStrings(op.readBuffers) << "\""
       << ",\"write_buffers\":\"" << joinStrings(op.writeBuffers) << "\""
       << ",\"read_versions\":\"" << joinInts(op.readBufferVersions) << "\""
       << ",\"write_versions\":\"" << joinInts(op.writeBufferVersions) << "\""
       << ",\"core_type\":\"" << op.coreType << "\""
       << ",\"sync\":" << (op.isSyncOp ? "true" : "false")
       << ",\"barrier\":" << (op.isBarrier ? "true" : "false") << "}}";
  }

  // Emit flow events linking sync_block_set → sync_block_wait across cores.
  // Build a map from (eventId, generation) → set-op index, then match waits.
  {
    // Key: (eventId, generation, sourceCore)
    using SyncKey = std::tuple<std::string, int64_t, std::string>;
    std::map<SyncKey, std::vector<size_t>> setOps;
    std::map<SyncKey, std::vector<size_t>> waitOps;
    for (size_t i = 0; i < operations.size(); ++i) {
      const HIVMOp &op = operations[i];
      if (op.opName == "sync_block_set" && !op.eventId.empty()) {
        bool isCube = op.coreType == "CUBE" || op.coreType == "AIC";
        SyncKey key{op.eventId, op.eventGeneration, isCube ? "AIC" : "AIV"};
        setOps[key].push_back(i);
      } else if (op.opName == "sync_block_wait" && !op.eventId.empty()) {
        // Wait on CUBE core means waiting for AIV→AIC signal.
        bool isCube = op.coreType == "CUBE" || op.coreType == "AIC";
        SyncKey key{op.eventId, op.eventGeneration, isCube ? "AIV" : "AIC"};
        waitOps[key].push_back(i);
      }
    }
    int64_t flowId = 0;
    for (auto &[key, setIndices] : setOps) {
      auto it = waitOps.find(key);
      if (it == waitOps.end())
        continue;
      auto &waits = it->second;
      size_t pairs = std::min(setIndices.size(), waits.size());
      for (size_t p = 0; p < pairs; ++p) {
        const HIVMOp &setOp = operations[setIndices[p]];
        const HIVMOp &waitOp = operations[waits[p]];
        // Flow start at set-op end time
        emitComma();
        os << "    {\"ph\":\"s\",\"id\":" << flowId
           << ",\"pid\":" << pipePid(setOp.pipe, setOp.coreType)
           << ",\"tid\":" << pipeTid(setOp.pipe) << ",\"ts\":"
           << llvm::format("%.3f",
                           cyclesToTraceUs(setOp.startCycle + setOp.duration))
           << ",\"name\":\"sync\",\"cat\":\"sync\"}";
        // Flow finish at wait-op start time
        emitComma();
        os << "    {\"ph\":\"f\",\"id\":" << flowId
           << ",\"pid\":" << pipePid(waitOp.pipe, waitOp.coreType)
           << ",\"tid\":" << pipeTid(waitOp.pipe) << ",\"ts\":"
           << llvm::format("%.3f", cyclesToTraceUs(waitOp.startCycle))
           << ",\"name\":\"sync\",\"cat\":\"sync\",\"bp\":\"e\"}";
        ++flowId;
      }
    }
  }

  os << "\n  ],\n  \"displayTimeUnit\": \"us\"\n}\n";
}

void HIVMAnalysisReport::emitDESGraph(llvm::raw_ostream &os,
                                      const HardwareConfig &config) const {
  auto joinStrVec = [](const std::vector<std::string> &v) {
    std::string s;
    llvm::raw_string_ostream ss(s);
    ss << "[";
    for (size_t i = 0; i < v.size(); ++i) {
      if (i)
        ss << ",";
      ss << "\"" << v[i] << "\"";
    }
    ss << "]";
    ss.flush();
    return s;
  };
  auto joinIntVec = [](const std::vector<int64_t> &v) {
    std::string s;
    llvm::raw_string_ostream ss(s);
    ss << "[";
    for (size_t i = 0; i < v.size(); ++i) {
      if (i)
        ss << ",";
      ss << v[i];
    }
    ss << "]";
    ss.flush();
    return s;
  };
  auto joinSizeVec = [](const std::vector<size_t> &v) {
    std::string s;
    llvm::raw_string_ostream ss(s);
    ss << "[";
    for (size_t i = 0; i < v.size(); ++i) {
      if (i)
        ss << ",";
      ss << v[i];
    }
    ss << "]";
    ss.flush();
    return s;
  };

  os << "{\n";
  os << "  \"clock_ghz\": "
     << llvm::format("%.3f", config.getClockFrequencyGHz()) << ",\n";
  os << "  \"operations\": [\n";
  for (size_t i = 0; i < operations.size(); ++i) {
    const HIVMOp &op = operations[i];
    if (i)
      os << ",\n";
    os << "    {"
       << "\"id\":" << op.id << ",\"name\":\"" << op.opName << "\""
       << ",\"pipe\":\"" << HIVMAnalyzer::stringifyPipe(op.pipe) << "\""
       << ",\"duration\":" << op.duration << ",\"line\":" << op.lineNumber
       << ",\"depends_on\":" << joinSizeVec(op.dependsOn)
       << ",\"is_sync\":" << (op.isSyncOp ? "true" : "false")
       << ",\"is_barrier\":" << (op.isBarrier ? "true" : "false")
       << ",\"event_id\":\"" << op.eventId << "\""
       << ",\"event_generation\":" << op.eventGeneration
       << ",\"sender_pipe\":\"" << HIVMAnalyzer::stringifyPipe(op.senderPipe)
       << "\""
       << ",\"receiver_pipe\":\""
       << HIVMAnalyzer::stringifyPipe(op.receiverPipe) << "\""
       << ",\"core_type\":\"" << op.coreType << "\""
       << ",\"bytes\":" << op.bytes << ",\"elements\":" << op.elements
       << ",\"loop_multiplier\":" << op.loopMultiplier
       << ",\"multi_buffer_slots\":" << op.multiBufferSlots
       << ",\"read_buffers\":" << joinStrVec(op.readBuffers)
       << ",\"write_buffers\":" << joinStrVec(op.writeBuffers)
       << ",\"read_versions\":" << joinIntVec(op.readBufferVersions)
       << ",\"write_versions\":" << joinIntVec(op.writeBufferVersions) << "}";
  }
  os << "\n  ]\n}\n";
}
