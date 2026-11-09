//===- HardwareConfig.cpp - Hardware Configuration Implementation ---------===//
//
// This file implements the HardwareConfig class for loading and querying
// hardware parameters from JSON configuration files.
//
//===----------------------------------------------------------------------===//

#include "AscendModel/HardwareConfig.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"
#include <cmath>
#include <cstdlib>
#include <mutex>

using namespace mlir::ascend;

//===----------------------------------------------------------------------===//
// Global Config Instance
//===----------------------------------------------------------------------===//

static std::unique_ptr<HardwareConfig> globalConfig;

HardwareConfig &mlir::ascend::getHardwareConfig() {
  if (!globalConfig) {
    globalConfig = HardwareConfig::getDefault910B();
  }
  return *globalConfig;
}

void mlir::ascend::setHardwareConfig(std::unique_ptr<HardwareConfig> config) {
  globalConfig = std::move(config);
}

bool mlir::ascend::loadHardwareConfigFromFile(llvm::StringRef path,
                                              std::string &error) {
  auto config = HardwareConfig::loadFromFile(path);
  if (!config) {
    error = "Failed to load hardware config from: " + path.str();
    return false;
  }
  if (!config->validate(error)) {
    return false;
  }
  setHardwareConfig(std::move(config));
  return true;
}

std::shared_ptr<const HardwareConfig>
mlir::ascend::loadHardwareConfigForAnalysis(llvm::StringRef path,
                                            std::string &error) {
  if (path.empty()) {
    static std::once_flag defaultConfigOnce;
    static std::shared_ptr<const HardwareConfig> defaultConfig;
    static std::string defaultConfigError;
    std::call_once(defaultConfigOnce, []() {
      auto config = HardwareConfig::getDefault910B();
      if (!config) {
        defaultConfigError = "Failed to load default hardware config";
        return;
      }
      if (!config->validate(defaultConfigError)) {
        return;
      }
      defaultConfig = std::move(config);
    });
    error = defaultConfigError;
    return defaultConfig;
  }

  auto config = HardwareConfig::loadFromFile(path);
  if (!config) {
    error = "Failed to load hardware config from: " + path.str();
    return nullptr;
  }
  if (!config->validate(error)) {
    return nullptr;
  }
  return std::shared_ptr<const HardwareConfig>(std::move(config));
}

//===----------------------------------------------------------------------===//
// HardwareConfig Implementation
//===----------------------------------------------------------------------===//

HardwareConfig::HardwareConfig() : clockFreqGHz(1.0) {}

HardwareConfig::~HardwareConfig() = default;

std::unique_ptr<HardwareConfig>
HardwareConfig::loadFromFile(llvm::StringRef path) {
  auto bufferOrErr = llvm::MemoryBuffer::getFile(path);
  if (!bufferOrErr) {
    llvm::errs() << "Error reading file: " << path << "\n";
    return nullptr;
  }

  auto json = llvm::json::parse(bufferOrErr.get()->getBuffer());
  if (!json) {
    llvm::errs() << "Error parsing JSON: " << llvm::toString(json.takeError())
                 << "\n";
    return nullptr;
  }

  return loadFromJSON(*json);
}

std::unique_ptr<HardwareConfig>
HardwareConfig::loadFromJSON(const llvm::json::Value &json) {
  auto config = std::make_unique<HardwareConfig>();
  std::string error;
  if (!config->parseJSON(json, error)) {
    llvm::errs() << "Error parsing hardware config: " << error << "\n";
    return nullptr;
  }
  return config;
}

//===----------------------------------------------------------------------===//
// Default 910B Configuration
//===----------------------------------------------------------------------===//

std::unique_ptr<HardwareConfig> HardwareConfig::getDefault910B() {
  // Try to load from standard config locations
  std::vector<std::string> searchPaths = {
      "configs/ascend_910b.json",       // Current directory
      "../configs/ascend_910b.json",    // Parent directory
      "../../configs/ascend_910b.json", // Two levels up
      "../config/ascend_910b.json",     // Alternative naming
      "config/ascend_910b.json",
  };

  // Also check environment variable for config path
  if (const char *envPath = std::getenv("ASCEND_CONFIG_PATH")) {
    searchPaths.insert(searchPaths.begin(),
                       std::string(envPath) + "/ascend_910b.json");
  }

  for (const auto &path : searchPaths) {
    if (llvm::sys::fs::exists(path)) {
      auto config = loadFromFile(path);
      if (config) {
        return config;
      }
    }
  }

  // Fallback to defaults if no config file found
  llvm::errs() << "Warning: Costmodel config file not found, using default "
                  "ascend_910b.json\n";
  return createHardcodedDefault910B();
}

/// Populate the tilesim-migrated micro-architecture tables with the 910B1
/// measured values (bandwidth_910B1.csv / vec_cycle_910B1.csv / 910B1.yaml).
/// Used by the hardcoded fallback so the migrated model is available even
/// when the JSON config file cannot be located. The JSON config is the
/// authoritative source in production; this mirrors the same numbers.
void HardwareConfig::populateTilesimDefaults910B() {
  // ---- Bandwidth tables (GB/s) ----
  // Core-independent movers.
  auto addScalar = [&](llvm::StringRef key, double gbps) {
    BandwidthTable t;
    t.singleGbps = gbps;
    bandwidthTables[key.str()] = std::move(t);
  };
  addScalar("hbm:l2", 33.68666667); // GM->SHM(L2)
  addScalar("l2:hbm", 34.34291667); // SHM->GM
  addScalar("l2:ub", 137.0);        // L2->UB
  addScalar("ub:l2", 93.0);         // UB->L2
  addScalar("hbm:l1", 135.0);       // GM->L1 (cube MTE2)
  addScalar("l2:l1", 221.0);        // L2->L1
  addScalar("l1:l2", 221.0);        // L1->L2
  addScalar("l1:l0a", 441.0);       // MTE1 A
  addScalar("l1:l0b", 220.5);       // MTE1 B
  addScalar("hbm:l0b", 104.5);
  addScalar("l0c:hbm", 70.0); // FixPipe
  addScalar("l0c:l2", 70.0);
  addScalar("l0c:l1", 70.0);
  addScalar("hbm:l2_agg", 1638.4); // aggregate HBM bandwidth

  // GM->UB per-core (vector MTE2): bandwidth degrades as cores contend.
  {
    static const double gmUb[] = {
        100.9,       64.5,        88.29666667, 77.945,      84.852,
        80.82666667, 73.31571429, 68.93375,    73.22888889, 67.586,
        71.13727273, 68.08833333, 69.14615385, 72.60785714, 74.99866667,
        73.26375,    72.05058824, 70.66833333, 73.27,       68.7705,
        68.72190476, 66.07909091, 66.33826087, 65.08041667, 65.3448,
        62.61884615, 62.15148148, 58.95964286, 58.23,       56.146,
        56.48,       53.3059375,  51.55151515, 49.97264706, 48.51342857,
        46.525,      46.06621622, 43.74315789, 42.97435897, 41.5595,
        40.44609756, 39.31904762, 38.47302326, 37.03886364, 36.33777778,
        35.40717391, 34.83446809, 33.68666667};
    BandwidthTable t;
    t.hasPerCore = true;
    for (int c = 1; c <= 48; ++c)
      t.perCoreGbps[c] = gmUb[c - 1];
    t.singleGbps = gmUb[47];
    bandwidthTables["hbm:ub"] = std::move(t);
  }
  // UB->GM per-core (MTE3).
  {
    static const double ubGm[] = {
        188.46,        210.53,        117.2733333,   130.295,       108.964,
        119.54,        117.280446425, 115.02089285,  112.761339275, 110.5017857,
        108.242232125, 105.98267855,  103.723124975, 101.4635714,   100.0593333,
        100.373125,    96.81470588,   98.78555556,   90.44578947,   88.896,
        83.38571429,   83.77727273,   77.92608696,   76.65166667,   71.0792,
        70.58615385,   66.58407407,   66.42357143,   61.56655172,   59.854,
        56.1083871,    55.9315625,    53.0,          51.53,         49.45828571,
        48.11111111,   46.14540541,   45.26947368,   43.51512821,   42.56075,
        40.9797561,    40.19857143,   38.46488372,   38.07454545,   36.684,
        36.17608696,   34.70893617,   34.34291667};
    BandwidthTable t;
    t.hasPerCore = true;
    for (int c = 1; c <= 48; ++c)
      t.perCoreGbps[c] = ubGm[c - 1];
    t.singleGbps = ubGm[47];
    bandwidthTables["ub:hbm"] = std::move(t);
  }

  // ---- Vector instruction cycle tables (intrinsic -> dtype -> triplet) ----
  auto addVec = [&](llvm::StringRef inst, int bits, int compute, int head,
                    int interval) {
    llvm::StringRef dt = (bits == 32) ? "fp32" : "fp16";
    vecCycleTables[inst.str()][dt.str()] =
        VecCycleEntry{compute, head, interval};
  };
  // (intrinsic, dtype, computing, head, interval) from vec_cycle_910B1.csv.
  addVec("VADD", 16, 2, 13, 18);
  addVec("VADD", 32, 2, 13, 18);
  addVec("VSUB", 16, 2, 13, 18);
  addVec("VSUB", 32, 2, 13, 18);
  addVec("VMUL", 16, 2, 13, 19);
  addVec("VMUL", 32, 2, 13, 19);
  addVec("VDIV", 16, 8, 13, 25);
  addVec("VDIV", 32, 4, 13, 25);
  addVec("VMAX", 16, 2, 14, 16);
  addVec("VMAX", 32, 2, 14, 16);
  addVec("VMIN", 16, 2, 14, 16);
  addVec("VMIN", 32, 2, 14, 16);
  addVec("VEXP", 16, 4, 13, 24);
  addVec("VEXP", 32, 2, 13, 24);
  addVec("VSQRT", 16, 2, 14, 25);
  addVec("VSQRT", 32, 2, 14, 25);
  addVec("VABS", 16, 1, 13, 16);
  addVec("VABS", 32, 1, 14, 16);
  addVec("RELU", 16, 1, 14, 16);
  addVec("RELU", 32, 1, 14, 16);
  addVec("LOG", 16, 2, 26, 14);
  addVec("LOG", 32, 2, 26, 14);
  addVec("VSEL", 16, 2, 13, 13);
  addVec("VSEL", 32, 2, 13, 14);
  addVec("VBRCB", 16, 0, 0, 18);
  addVec("VBRCB", 32, 0, 0, 18);
  addVec("VCMPV_NE", 16, 2, 14, 22);
  addVec("VCMPV_NE", 32, 2, 14, 22);
  addVec("VCMPV_GE", 16, 2, 14, 22);
  addVec("VCMPV_GE", 32, 2, 14, 22);
  addVec("CMPV_EQ", 16, 2, 13, 22);
  addVec("CMPV_EQ", 32, 2, 13, 22);
  addVec("VCGADD", 32, 1, 14, 24);
  addVec("VCGMAX", 32, 1, 14, 17);
  addVec("VCGMIN", 32, 1, 14, 17);
  addVec("VREDUCEV2", 32, 14, 14, 20);
  addVec("VCOPY", 16, 1, 11, 13);
  addVec("VCOPY", 32, 1, 11, 13);
  addVec("CONV_F322F16", 32, 1, 16, 14);
  addVec("CONV_F162F32", 16, 2, 13, 15);
  addVec("CONV_F322BF16", 32, 1, 14, 16);
  addVec("CONV_BF162F32", 16, 2, 14, 18);

  // ---- Cube (GEMM) model ----
  cubeModel.basicM = 16;
  cubeModel.basicKNumerator = 32;
  cubeModel.basicN = 16;
  cubeModel.l0TileLimitKb = 32;
  cubeModel.repeatCycles["fp32"] = 2;
  cubeModel.repeatCycles["fp16"] = 1;
  cubeModel.repeatCycles["bf16"] = 1;
  cubeModel.repeatCycles["int8"] = 1;

  activeBandwidthCores = 48;
  // tilesim's engineering by-rule-pipeline path (eval_by_rule_pipe.py, the
  // "main simulation loop" this costmodel is aligned to) sets
  // enable_small_pkt_bw=True; the base config default (False) is not the
  // accuracy target. Global pkt_param mirrors tilesim 910B1.yaml.
  enableSmallPacketBw = true;
  globalSmallPacket.enabled = true;
  globalSmallPacket.thresholdBytes = 256;
  globalSmallPacket.a64 = 0.295;
  globalSmallPacket.b64 = 12.061;
  globalSmallPacket.a128 = 0.272;
  globalSmallPacket.b128 = 16.820;
  globalSmallPacket.a256 = 0.322;
  globalSmallPacket.b256 = 24.004;
  hasGlobalSmallPacket = true;
  // Mutex unit cliques (tilesim MutexComponents / pipe_exclusive_config).
  // 910B: AIV MTE2 (vec load) and MTE3 (vec store) share one pipeline.
  mutexGroups.clear();
  mutexGroups.push_back({"vec_mte2", "mte3"});
}

/// Hardcoded fallback configuration (used when JSON file not found)
std::unique_ptr<HardwareConfig> HardwareConfig::createHardcodedDefault910B() {
  auto config = std::make_unique<HardwareConfig>();

  config->name = "Ascend 910B";
  config->vendor = "Huawei";
  config->version = "1.0";
  config->clockFreqGHz = 1.85;

  // Memory spaces
  // HBM: 32GB, 1.6TB/s
  {
    MemorySpace hbm;
    hbm.name = "hbm";
    hbm.type = MemoryType::OffChip;
    hbm.sizeBytes = 32ULL * 1024 * 1024 * 1024;
    hbm.bandwidthBytesPerCycle = (1.6e12) / (config->clockFreqGHz * 1e9);
    hbm.latencyCycles = 200;
    config->memorySpaces["hbm"] = std::move(hbm);
  }

  // L2: 192MB shared
  {
    MemorySpace l2;
    l2.name = "l2";
    l2.type = MemoryType::OnChipShared;
    l2.sizeBytes = 192 * 1024 * 1024;
    l2.bandwidthBytesPerCycle = (3200e9) / (config->clockFreqGHz * 1e9);
    l2.latencyCycles = 50;
    config->memorySpaces["l2"] = std::move(l2);
  }

  // L1: 1MB for Cube staging
  {
    MemorySpace l1;
    l1.name = "l1";
    l1.type = MemoryType::OnChipLocal;
    l1.sizeBytes = 1024 * 1024;
    l1.bandwidthBytesPerCycle = (6400e9) / (config->clockFreqGHz * 1e9);
    l1.latencyCycles = 10;
    l1.description = "Cube data staging buffer";
    config->memorySpaces["l1"] = std::move(l1);
  }

  // L0A: 64KB left matrix input
  {
    MemorySpace l0a;
    l0a.name = "l0a";
    l0a.type = MemoryType::RegisterFile;
    l0a.sizeBytes = 64 * 1024;
    l0a.bandwidthBytesPerCycle = 0; // Internal
    l0a.latencyCycles = 0;
    l0a.description = "Left matrix input for Cube";
    config->memorySpaces["l0a"] = std::move(l0a);
  }

  // L0B: 64KB right matrix input
  {
    MemorySpace l0b;
    l0b.name = "l0b";
    l0b.type = MemoryType::RegisterFile;
    l0b.sizeBytes = 64 * 1024;
    l0b.bandwidthBytesPerCycle = 0;
    l0b.latencyCycles = 0;
    l0b.description = "Right matrix input for Cube";
    config->memorySpaces["l0b"] = std::move(l0b);
  }

  // L0C: 256KB Cube output
  {
    MemorySpace l0c;
    l0c.name = "l0c";
    l0c.type = MemoryType::RegisterFile;
    l0c.sizeBytes = 256 * 1024;
    l0c.bandwidthBytesPerCycle = 0;
    l0c.latencyCycles = 0;
    l0c.description = "Cube output accumulator";
    config->memorySpaces["l0c"] = std::move(l0c);
  }

  // UB: 256KB Unified Buffer for Vector
  {
    MemorySpace ub;
    ub.name = "ub";
    ub.type = MemoryType::OnChipLocal;
    ub.sizeBytes = 256 * 1024;
    ub.bandwidthBytesPerCycle = (3200e9) / (config->clockFreqGHz * 1e9);
    ub.latencyCycles = 5;
    ub.description = "Unified Buffer for Vector compute";
    config->memorySpaces["ub"] = std::move(ub);
  }

  // Compute units
  // Cube: 320 TFLOPS FP16, 16x16x16 tiles
  {
    ComputeUnit cube;
    cube.name = "cube";
    cube.type = ComputeUnitType::MatrixEngine;
    cube.tflopsFP16 = 320;
    cube.tflopsFP32 = 160;
    cube.tflopsINT8 = 640;
    cube.tileM = 16;
    cube.tileN = 16;
    cube.tileK = 16;
    // Fractal sizes per data type
    // Fractal sizes per data type (m x k x n)
    cube.fractalSizes["fp16"] = FractalSize{16, 16, 16};
    cube.fractalSizes["bf16"] = FractalSize{16, 16, 16};
    cube.fractalSizes["fp32"] = FractalSize{16, 8, 16};
    cube.fractalSizes["int8"] = FractalSize{16, 32, 16};
    cube.inputSpaces = {"l0a", "l0b"};
    cube.outputSpace = "l0c";
    cube.supportedDtypes = {"fp16", "bf16", "fp32", "int8"};
    config->computeUnits["cube"] = std::move(cube);
  }

  // Vector: 128 FP16 elements per cycle
  {
    ComputeUnit vector;
    vector.name = "vector";
    vector.type = ComputeUnitType::SIMDEngine;
    vector.tflopsFP16 = 20;
    vector.tflopsFP32 = 10;
    vector.tflopsINT8 = 0;
    vector.widthElements = 128;
    vector.widthBytes = 256;
    vector.computeSpace = "ub";
    vector.supportedOps = {
        "add",  "sub",        "mul",        "div",   "max",      "min",
        "exp",  "log",        "sqrt",       "rsqrt", "tanh",     "sigmoid",
        "relu", "reduce_sum", "reduce_max", "cast",  "broadcast"};
    vector.supportedDtypes = {"fp32", "fp16", "bf16", "int32", "int8"};
    config->computeUnits["vector"] = std::move(vector);
  }

  config->vectorOpCyclesPerInstruction["vadd"] = 1;
  config->vectorOpCyclesPerInstruction["vsub"] = 1;
  config->vectorOpCyclesPerInstruction["vmul"] = 1;
  config->vectorOpCyclesPerInstruction["vcast"] = 1;
  config->vectorOpCyclesPerInstruction["vreduce"] = 2;
  config->vectorOpCyclesPerInstruction["vexp"] = 4;
  config->vectorOpCyclesPerInstruction["vdiv"] = 3;
  config->vectorOpCyclesPerInstruction["vlog"] = 3;
  config->vectorOpCyclesPerInstruction["vtanh"] = 3;
  config->vectorOpCyclesPerInstruction["vsigmoid"] = 3;
  config->vectorOpCyclesPerInstruction["vsqrt"] = 2;
  config->vectorOpCyclesPerInstruction["vrsqrt"] = 2;

  // Data movers
  // MTE2 (Cube): HBM -> L1
  {
    DataMover dm;
    dm.name = "cube_mte2";
    dm.description = "Cube input: HBM to L1";
    dm.srcSpace = "hbm";
    dm.dstSpaces = {"l1"};
    dm.bandwidthBytesPerCycle = (200e9) / (config->clockFreqGHz * 1e9);
    dm.maxBurstBytes = 65536;
    dm.alignmentBytes = 32;
    config->dataMovers["cube_mte2"] = std::move(dm);
  }

  // MTE1: L1 -> L0A/L0B
  {
    DataMover dm;
    dm.name = "mte1";
    dm.description = "Cube input: L1 to L0A/L0B";
    dm.srcSpace = "l1";
    dm.dstSpaces = {"l0a", "l0b"};
    dm.bandwidthBytesPerCycle = (400e9) / (config->clockFreqGHz * 1e9);
    dm.maxBurstBytes = 32768;
    dm.alignmentBytes = 32;
    config->dataMovers["mte1"] = std::move(dm);
  }

  // FixPipe: L0C -> HBM
  {
    DataMover dm;
    dm.name = "fixpipe";
    dm.description = "Cube output: L0C to HBM";
    dm.srcSpace = "l0c";
    dm.dstSpaces = {"hbm"};
    dm.bandwidthBytesPerCycle = (200e9) / (config->clockFreqGHz * 1e9);
    dm.maxBurstBytes = 65536;
    dm.alignmentBytes = 32;
    dm.supportsAccumulate = true;
    dm.supportsCast = true;
    config->dataMovers["fixpipe"] = std::move(dm);
  }

  // MTE2 (Vector): HBM -> UB
  {
    DataMover dm;
    dm.name = "vector_mte2";
    dm.description = "Vector input: HBM to UB";
    dm.srcSpace = "hbm";
    dm.dstSpaces = {"ub"};
    dm.bandwidthBytesPerCycle = (200e9) / (config->clockFreqGHz * 1e9);
    dm.maxBurstBytes = 65536;
    dm.alignmentBytes = 32;
    config->dataMovers["vector_mte2"] = std::move(dm);
  }

  // MTE3: UB -> HBM
  {
    DataMover dm;
    dm.name = "mte3";
    dm.description = "Vector output: UB to HBM";
    dm.srcSpace = "ub";
    dm.dstSpaces = {"hbm"};
    dm.bandwidthBytesPerCycle = (200e9) / (config->clockFreqGHz * 1e9);
    dm.maxBurstBytes = 65536;
    dm.alignmentBytes = 32;
    config->dataMovers["mte3"] = std::move(dm);
  }

  // Pipeline paths
  {
    PipelinePath cubePath;
    cubePath.name = "cube_path";
    cubePath.stages = {"cube_mte2", "mte1", "cube", "fixpipe"};
    cubePath.description = "Matrix multiplication data flow";
    config->pipelinePaths["cube_path"] = std::move(cubePath);
  }
  {
    PipelinePath vectorPath;
    vectorPath.name = "vector_path";
    vectorPath.stages = {"vector_mte2", "vector", "mte3"};
    vectorPath.description = "Vector computation data flow";
    config->pipelinePaths["vector_path"] = std::move(vectorPath);
  }

  // Parallelism
  config->parallelismFlags["cube_and_vector"] = true;
  config->parallelismFlags["cube_mte2_and_fixpipe"] = true;
  config->parallelismFlags["vector_mte2_and_mte3"] = true;

  // tilesim-migrated micro-architecture tables (bandwidth/vec/cube).
  config->populateTilesimDefaults910B();

  return config;
}

//===----------------------------------------------------------------------===//
// JSON Parsing
//===----------------------------------------------------------------------===//

static MemoryType parseMemoryType(llvm::StringRef typeStr) {
  if (typeStr == "off_chip")
    return MemoryType::OffChip;
  if (typeStr == "on_chip_shared")
    return MemoryType::OnChipShared;
  if (typeStr == "on_chip_local")
    return MemoryType::OnChipLocal;
  if (typeStr == "register_file")
    return MemoryType::RegisterFile;
  return MemoryType::OnChipLocal;
}

static ComputeUnitType parseComputeUnitType(llvm::StringRef typeStr) {
  if (typeStr == "matrix_engine")
    return ComputeUnitType::MatrixEngine;
  if (typeStr == "simd_engine")
    return ComputeUnitType::SIMDEngine;
  if (typeStr == "scalar_engine")
    return ComputeUnitType::ScalarEngine;
  return ComputeUnitType::SIMDEngine;
}

bool HardwareConfig::parseJSON(const llvm::json::Value &json,
                               std::string &error) {
  const auto *root = json.getAsObject();
  if (!root) {
    error = "Root must be a JSON object";
    return false;
  }

  // Basic info
  if (auto n = root->getString("name"))
    name = n->str();
  if (auto v = root->getString("vendor"))
    vendor = v->str();
  if (auto ver = root->getString("version"))
    version = ver->str();

  // Clock
  if (const auto *clock = root->getObject("clock")) {
    if (auto freq = clock->getNumber("frequency_ghz"))
      clockFreqGHz = *freq;
  }

  // Memory spaces
  if (const auto *memSpaces = root->getObject("memory_spaces")) {
    for (const auto &kv : *memSpaces) {
      const auto *obj = kv.second.getAsObject();
      if (!obj)
        continue;

      MemorySpace space;
      space.name = kv.first.str();

      if (auto t = obj->getString("type"))
        space.type = parseMemoryType(*t);

      // Size (support multiple units)
      if (auto gb = obj->getNumber("size_gb"))
        space.sizeBytes = static_cast<size_t>(*gb * 1024 * 1024 * 1024);
      else if (auto mb = obj->getNumber("size_mb"))
        space.sizeBytes = static_cast<size_t>(*mb * 1024 * 1024);
      else if (auto kb = obj->getNumber("size_kb"))
        space.sizeBytes = static_cast<size_t>(*kb * 1024);

      // Bandwidth
      if (auto tbps = obj->getNumber("bandwidth_tbps"))
        space.bandwidthBytesPerCycle = (*tbps * 1e12) / (clockFreqGHz * 1e9);
      else if (auto gbps = obj->getNumber("bandwidth_gbps"))
        space.bandwidthBytesPerCycle = (*gbps * 1e9) / (clockFreqGHz * 1e9);

      if (auto lat = obj->getInteger("latency_cycles"))
        space.latencyCycles = *lat;

      if (auto desc = obj->getString("description"))
        space.description = desc->str();

      memorySpaces[space.name] = std::move(space);
    }
  }

  // Compute units
  if (const auto *units = root->getObject("compute_units")) {
    for (const auto &kv : *units) {
      const auto *obj = kv.second.getAsObject();
      if (!obj)
        continue;

      ComputeUnit unit;
      unit.name = kv.first.str();

      if (auto t = obj->getString("type"))
        unit.type = parseComputeUnitType(*t);

      if (auto v = obj->getNumber("tflops_fp16"))
        unit.tflopsFP16 = *v;
      if (auto v = obj->getNumber("tflops_fp32"))
        unit.tflopsFP32 = *v;
      if (auto v = obj->getNumber("tflops_int8"))
        unit.tflopsINT8 = *v;

      if (auto v = obj->getInteger("tile_m"))
        unit.tileM = *v;
      if (auto v = obj->getInteger("tile_n"))
        unit.tileN = *v;
      if (auto v = obj->getInteger("tile_k"))
        unit.tileK = *v;

      // Parse fractal sizes per dtype
      if (const auto *fractalObj = obj->getObject("fractal_sizes")) {
        for (const auto &fkv : *fractalObj) {
          const auto *sizeObj = fkv.second.getAsObject();
          if (!sizeObj)
            continue;

          FractalSize fs;
          if (auto m = sizeObj->getInteger("m"))
            fs.m = *m;
          if (auto n = sizeObj->getInteger("n"))
            fs.n = *n;
          if (auto k = sizeObj->getInteger("k"))
            fs.k = *k;
          unit.fractalSizes[fkv.first.str()] = fs;
        }
      }

      if (auto v = obj->getInteger("width_elements"))
        unit.widthElements = *v;
      if (auto v = obj->getInteger("width_bytes"))
        unit.widthBytes = *v;

      if (auto s = obj->getString("output_space"))
        unit.outputSpace = s->str();
      if (auto s = obj->getString("compute_space"))
        unit.computeSpace = s->str();

      if (const auto *arr = obj->getArray("input_spaces")) {
        for (const auto &v : *arr) {
          if (auto s = v.getAsString())
            unit.inputSpaces.push_back(s->str());
        }
      }

      if (const auto *arr = obj->getArray("supported_ops")) {
        for (const auto &v : *arr) {
          if (auto s = v.getAsString())
            unit.supportedOps.push_back(s->str());
        }
      }

      if (const auto *arr = obj->getArray("supported_dtypes")) {
        for (const auto &v : *arr) {
          if (auto s = v.getAsString())
            unit.supportedDtypes.push_back(s->str());
        }
      }

      computeUnits[unit.name] = std::move(unit);
    }
  }

  // Data movers
  if (const auto *movers = root->getObject("data_movers")) {
    for (const auto &kv : *movers) {
      const auto *obj = kv.second.getAsObject();
      if (!obj)
        continue;

      DataMover dm;
      dm.name = kv.first.str();

      if (auto s = obj->getString("description"))
        dm.description = s->str();
      if (auto s = obj->getString("src_space"))
        dm.srcSpace = s->str();
      if (auto s = obj->getString("dst_space"))
        dm.dstSpaces.push_back(s->str());

      if (const auto *arr = obj->getArray("dst_spaces")) {
        dm.dstSpaces.clear();
        for (const auto &v : *arr) {
          if (auto s = v.getAsString())
            dm.dstSpaces.push_back(s->str());
        }
      }

      if (auto gbps = obj->getNumber("bandwidth_gbps"))
        dm.bandwidthBytesPerCycle = (*gbps * 1e9) / (clockFreqGHz * 1e9);

      if (auto v = obj->getInteger("max_burst_bytes"))
        dm.maxBurstBytes = *v;
      if (auto v = obj->getInteger("alignment_bytes"))
        dm.alignmentBytes = *v;
      if (auto v = obj->getBoolean("supports_accumulate"))
        dm.supportsAccumulate = *v;
      if (auto v = obj->getBoolean("supports_cast"))
        dm.supportsCast = *v;

      dataMovers[dm.name] = std::move(dm);
    }
  }

  // Pipeline
  if (const auto *pipeline = root->getObject("pipeline")) {
    for (const auto &kv : *pipeline) {
      if (kv.first == "parallelism") {
        if (const auto *par = kv.second.getAsObject()) {
          for (const auto &pkv : *par) {
            if (auto v = pkv.second.getAsBoolean())
              parallelismFlags[pkv.first.str()] = *v;
          }
        }
        continue;
      }

      const auto *obj = kv.second.getAsObject();
      if (!obj)
        continue;

      PipelinePath path;
      path.name = kv.first.str();

      if (const auto *arr = obj->getArray("stages")) {
        for (const auto &v : *arr) {
          if (auto s = v.getAsString())
            path.stages.push_back(s->str());
        }
      }

      if (auto s = obj->getString("description"))
        path.description = s->str();

      pipelinePaths[path.name] = std::move(path);
    }
  }

  if (const auto *calibration = root->getObject("calibration")) {
    if (const auto *vecOps =
            calibration->getObject("vector_op_cycles_per_vec_instruction")) {
      auto readInt = [&](llvm::StringRef key, llvm::StringRef opName) {
        if (auto v = vecOps->getInteger(key))
          vectorOpCyclesPerInstruction[opName] = static_cast<int>(*v);
      };
      readInt("simple_ops_add_sub_mul_etc", "vadd");
      readInt("simple_ops_add_sub_mul_etc", "vsub");
      readInt("simple_ops_add_sub_mul_etc", "vmul");
      readInt("simple_ops_add_sub_mul_etc", "vcast");
      readInt("exp", "vexp");
      readInt("log", "vlog");
      readInt("tanh", "vtanh");
      readInt("sigmoid", "vsigmoid");
      readInt("sqrt", "vsqrt");
      readInt("rsqrt", "vrsqrt");
      readInt("div", "vdiv");
    }
  }

  //===------------------------------------------------------------------===//
  // tilesim-migrated micro-architecture tables (root cause ①/②)
  // All fields are optional; absence leaves the (less accurate) legacy
  // estimation path intact, preserving backward compatibility.
  //===------------------------------------------------------------------===//

  // Bandwidth tables keyed by "src:dst". Supports core-independent
  // ("bandwidth_gbps") and per-core ("per_core_gbps": {coreNum: gbps})
  // forms, plus optional small-packet fitting coefficients.
  if (const auto *bwTables = root->getObject("bandwidth_tables")) {
    for (const auto &kv : *bwTables) {
      const auto *obj = kv.second.getAsObject();
      if (!obj)
        continue;
      BandwidthTable tbl;
      if (auto gbps = obj->getNumber("bandwidth_gbps")) {
        tbl.singleGbps = *gbps;
      } else if (const auto *perCore = obj->getObject("per_core_gbps")) {
        tbl.hasPerCore = true;
        for (const auto &pkv : *perCore) {
          if (auto v = pkv.second.getAsNumber()) {
            int cores = std::stoi(pkv.first.str());
            tbl.perCoreGbps[cores] = *v;
          }
        }
        // singleGbps fallback = the largest (most-saturated) core entry.
        if (!tbl.perCoreGbps.empty())
          tbl.singleGbps = tbl.perCoreGbps.rbegin()->second;
      }
      if (const auto *sp = obj->getObject("small_packet")) {
        tbl.smallPacket.enabled = true;
        if (auto t = sp->getInteger("threshold_bytes"))
          tbl.smallPacket.thresholdBytes = *t;
        auto readPair = [&](llvm::StringRef key, double &a, double &b) {
          if (const auto *arr = sp->getArray(key); arr && arr->size() >= 2) {
            if (auto va = (*arr)[0].getAsNumber())
              a = *va;
            if (auto vb = (*arr)[1].getAsNumber())
              b = *vb;
          }
        };
        readPair("64B", tbl.smallPacket.a64, tbl.smallPacket.b64);
        readPair("128B", tbl.smallPacket.a128, tbl.smallPacket.b128);
        readPair("256B", tbl.smallPacket.a256, tbl.smallPacket.b256);
      }
      bandwidthTables[kv.first.str()] = std::move(tbl);
    }
  }

  // Vector instruction cycle tables: intrinsic -> dtype ->
  // {compute,head,interval}.
  if (const auto *vecTables = root->getObject("vec_cycle_tables")) {
    for (const auto &kv : *vecTables) {
      const auto *byDtype = kv.second.getAsObject();
      if (!byDtype)
        continue;
      llvm::StringMap<VecCycleEntry> inner;
      for (const auto &dkv : *byDtype) {
        const auto *entry = dkv.second.getAsObject();
        if (!entry)
          continue;
        VecCycleEntry e;
        if (auto v = entry->getInteger("compute"))
          e.compute = *v;
        if (auto v = entry->getInteger("head"))
          e.head = *v;
        if (auto v = entry->getInteger("interval"))
          e.interval = *v;
        inner[dkv.first.str()] = e;
      }
      vecCycleTables[kv.first.str()] = std::move(inner);
    }
  }

  // Cube (GEMM) micro-architecture: throughput [m,k,n], repeat cycles, L0
  // limit.
  if (const auto *cube = root->getObject("cube_model")) {
    if (const auto *arr = cube->getArray("throughput");
        arr && arr->size() >= 3) {
      if (auto v = (*arr)[0].getAsInteger())
        cubeModel.basicM = *v;
      if (auto v = (*arr)[1].getAsInteger())
        cubeModel.basicKNumerator = *v;
      if (auto v = (*arr)[2].getAsInteger())
        cubeModel.basicN = *v;
    }
    if (const auto *rc = cube->getObject("repeat_cycles")) {
      for (const auto &rkv : *rc) {
        if (auto v = rkv.second.getAsInteger())
          cubeModel.repeatCycles[rkv.first.str()] = static_cast<int>(*v);
      }
    }
    if (auto v = cube->getInteger("l0_tile_limit_kb"))
      cubeModel.l0TileLimitKb = *v;
  }

  // tilesim tuning knobs.
  if (const auto *ts = root->getObject("tilesim")) {
    if (auto v = ts->getInteger("active_bandwidth_cores"))
      activeBandwidthCores = *v;
    if (auto v = ts->getBoolean("enable_small_packet_bw"))
      enableSmallPacketBw = *v;
    // Global pkt_param (tilesim 910B1.yaml pkt_param): one (a,b) set per size
    // bucket applied to every mover. Sits behind any per-table small_packet
    // entry parsed above from bandwidth_tables.
    if (const auto *sp = ts->getObject("small_packet")) {
      SmallPacketCoeffs c;
      c.enabled = true;
      if (auto t = sp->getInteger("threshold_bytes"))
        c.thresholdBytes = *t;
      auto readPair = [&](llvm::StringRef key, double &a, double &b) {
        if (const auto *arr = sp->getArray(key); arr && arr->size() >= 2) {
          if (auto va = (*arr)[0].getAsNumber())
            a = *va;
          if (auto vb = (*arr)[1].getAsNumber())
            b = *vb;
        }
      };
      readPair("64B", c.a64, c.b64);
      readPair("128B", c.a128, c.b128);
      readPair("256B", c.a256, c.b256);
      globalSmallPacket = c;
      hasGlobalSmallPacket = true;
    }
    // Mutex unit cliques (tilesim MutexComponents): object with a "cliques"
    // array of unit-name arrays, e.g. {"cliques":[["vec_mte2","mte3"]]}.
    // Units in a clique share a pipeline and cannot run in parallel.
    if (const auto *mg = ts->getObject("mutex_groups")) {
      mutexGroups.clear();
      if (const auto *cliques = mg->getArray("cliques")) {
        for (const auto &clique : *cliques) {
          if (const auto *units = clique.getAsArray()) {
            std::vector<std::string> group;
            for (const auto &u : *units)
              if (auto s = u.getAsString())
                group.push_back(s->str());
            if (group.size() >= 2)
              mutexGroups.push_back(std::move(group));
          }
        }
      }
    }
  }

  return true;
}

//===----------------------------------------------------------------------===//
// Query Methods
//===----------------------------------------------------------------------===//

const MemorySpace *HardwareConfig::getMemorySpace(llvm::StringRef name) const {
  auto it = memorySpaces.find(name);
  return it != memorySpaces.end() ? &it->second : nullptr;
}

double HardwareConfig::getMemoryBandwidthTBps(llvm::StringRef name) const {
  if (const auto *space = getMemorySpace(name)) {
    return space->bandwidthBytesPerCycle * clockFreqGHz * 1e9 / 1e12;
  }
  return 0;
}

double
HardwareConfig::getMemoryBandwidthBytesPerCycle(llvm::StringRef name) const {
  if (const auto *space = getMemorySpace(name)) {
    return space->bandwidthBytesPerCycle;
  }
  return 0;
}

int HardwareConfig::getMemoryLatencyCycles(llvm::StringRef name) const {
  if (const auto *space = getMemorySpace(name)) {
    return space->latencyCycles;
  }
  return 0;
}

size_t HardwareConfig::getMemorySizeBytes(llvm::StringRef name) const {
  if (const auto *space = getMemorySpace(name)) {
    return space->sizeBytes;
  }
  return 0;
}

std::vector<std::string> HardwareConfig::getMemorySpaceNames() const {
  std::vector<std::string> names;
  for (const auto &kv : memorySpaces) {
    names.push_back(kv.first().str());
  }
  return names;
}

const ComputeUnit *HardwareConfig::getComputeUnit(llvm::StringRef name) const {
  auto it = computeUnits.find(name);
  return it != computeUnits.end() ? &it->second : nullptr;
}

double HardwareConfig::getCubeTFlopsFP16() const {
  if (const auto *cube = getComputeUnit("cube")) {
    return cube->tflopsFP16;
  }
  return 320; // Default
}

void HardwareConfig::getCubeTileSize(int &m, int &n, int &k) const {
  if (const auto *cube = getComputeUnit("cube")) {
    m = cube->tileM;
    n = cube->tileN;
    k = cube->tileK;
    return;
  }
  m = n = k = 16; // Default
}

void HardwareConfig::getCubeFractalSize(int elementBits, int &m, int &n,
                                        int &k) const {
  if (const auto *cube = getComputeUnit("cube")) {
    // Determine dtype key from element bits
    llvm::StringRef dtypeKey;
    if (elementBits == 8) {
      dtypeKey = "int8";
    } else if (elementBits == 32) {
      dtypeKey = "fp32";
    } else {
      // Default to fp16 for 16-bit types (fp16, bf16)
      dtypeKey = "fp16";
    }

    // Look up fractal size for this dtype
    auto it = cube->fractalSizes.find(dtypeKey);
    if (it != cube->fractalSizes.end()) {
      m = it->second.m;
      n = it->second.n;
      k = it->second.k;
      return;
    }

    // Fall back to default tile size
    m = cube->tileM;
    n = cube->tileN;
    k = cube->tileK;
    return;
  }

  // Ultimate fallback: FP16 defaults
  m = 16;
  n = 16;
  k = 16;
}

llvm::StringRef HardwareConfig::getCubeOutputSpace() const {
  if (const auto *cube = getComputeUnit("cube")) {
    return cube->outputSpace;
  }
  return "l0c";
}

double HardwareConfig::getVectorTFlopsFP32() const {
  if (const auto *vec = getComputeUnit("vector")) {
    return vec->tflopsFP32;
  }
  return 10; // Default
}

int HardwareConfig::getVectorWidthElements() const {
  if (const auto *vec = getComputeUnit("vector")) {
    return vec->widthElements;
  }
  return 128; // Default
}

int HardwareConfig::getVectorWidthBytes() const {
  if (const auto *vec = getComputeUnit("vector")) {
    return vec->widthBytes;
  }
  return 256; // Default
}

llvm::StringRef HardwareConfig::getVectorComputeSpace() const {
  if (const auto *vec = getComputeUnit("vector")) {
    return vec->computeSpace;
  }
  return "ub";
}

int HardwareConfig::getVectorOpCyclesPerInstruction(
    llvm::StringRef opName) const {
  auto it = vectorOpCyclesPerInstruction.find(opName);
  if (it != vectorOpCyclesPerInstruction.end())
    return it->second;
  if (opName == "vreduce")
    return 2;
  return 1;
}

double HardwareConfig::getHBMBandwidthGBs() const {
  if (const auto *hbm = getMemorySpace("hbm")) {
    // Convert bytes/cycle to GB/s
    return hbm->bandwidthBytesPerCycle * clockFreqGHz * 1e9 / 1e9;
  }
  return 1600.0; // Default: 1.6 TB/s = 1600 GB/s
}

double HardwareConfig::getHBMBandwidthTBs() const {
  return getHBMBandwidthGBs() / 1000.0;
}

double HardwareConfig::getCubeTFLOPS() const { return getCubeTFlopsFP16(); }

double HardwareConfig::getVectorTFLOPS() const { return getVectorTFlopsFP32(); }

int HardwareConfig::getMTE2StartupLatency() const {
  // MTE2 DMA pipeline startup: address translation + HBM access setup.
  // Previously inflated to 320 to absorb pipe_barrier fill latency, but
  // pipe_barrier now has its own cost model.  Restored to reflect actual
  // MTE2 transfer startup overhead.
  return 50;
}

int HardwareConfig::getMTE3StartupLatency() const { return 40; }

int HardwareConfig::getFixPipeStartupLatency() const { return 30; }

int HardwareConfig::getCubeStartupLatency() const { return 20; }

int HardwareConfig::getVectorStartupLatency() const {
  // Calibrated: 35 cycles (was 10).
  // Reflects UB read-after-write penalty between dependent vector instructions
  // in a serial dependency chain (e.g., sub → exp → reduce in softmax).
  return 35;
}

double HardwareConfig::getAIVScalarOverheadFactor() const {
  // Calibrated from _attn_fwd profiling (BM=48/64 steady-state):
  //   aiv_vec_ratio = 0.211  →  scalar+overhead fraction = 1 - 0.211 = 0.789
  //   effective factor = (1 - vec_ratio) / vec_ratio = 0.789 / 0.211 ≈ 3.74
  // Applying this to vec_cycles gives total_aiv_cycles that matches the
  // observed aiv_time, capturing scalar (loop/barrier) + idle fractions.
  return 3.74;
}

int HardwareConfig::getNumAICCores() const {
  // Block Dim = 20 in profiling runs.
  return 20;
}

int HardwareConfig::getNumAIVCores() const {
  // Mix Block Dim = 40 → 40 AIV cores (2 per AIC block × 20 blocks).
  return 40;
}

int HardwareConfig::getPipeBarrierCyclesPerIter() const {
  // Calibrated from BM=64, 1-wave execution:
  //   AIV wall time = 31.993 μs = 59 187 cycles
  //   Active fraction (vec+scalar+mte) = 61.1% → idle = 38.9% = 23 044 cycles
  //   With N_iter = 3 inner iterations → 23 044 / 3 ≈ 7 500 cycles per barrier.
  return 7500;
}

const DataMover *HardwareConfig::getDataMover(llvm::StringRef name) const {
  auto it = dataMovers.find(name);
  return it != dataMovers.end() ? &it->second : nullptr;
}

std::vector<std::string> HardwareConfig::getDataMoverNames() const {
  std::vector<std::string> names;
  for (const auto &kv : dataMovers) {
    names.push_back(kv.first().str());
  }
  return names;
}

//===----------------------------------------------------------------------===//
// Performance Estimation
//===----------------------------------------------------------------------===//

int64_t HardwareConfig::estimateCubeCycles(int64_t M, int64_t N,
                                           int64_t K) const {
  int tileM = 16;
  int tileN = 16;
  int tileK = 16;
  getCubeTileSize(tileM, tileN, tileK);
  auto ceilDiv = [](int64_t a, int64_t b) -> int64_t {
    return b > 0 ? (a + b - 1) / b : 0;
  };
  return std::max<int64_t>(1, ceilDiv(M, tileM) * ceilDiv(N, tileN) *
                                  ceilDiv(K, tileK));
}

int64_t HardwareConfig::estimateVectorCycles(int64_t numElements) const {
  int width = getVectorWidthElements();
  if (width <= 0)
    return std::max<int64_t>(1, numElements);
  return std::max<int64_t>(1, (numElements + width - 1) / width);
}

int64_t HardwareConfig::estimateMemoryCycles(llvm::StringRef moverName,
                                             int64_t bytes) const {
  const DataMover *mover = getDataMover(moverName);
  if (!mover || mover->bandwidthBytesPerCycle <= 0.0)
    return std::max<int64_t>(1, bytes);
  return std::max<int64_t>(1, static_cast<int64_t>(std::ceil(
                                  bytes / mover->bandwidthBytesPerCycle)));
}

int64_t HardwareConfig::estimateMemoryCyclesWithLatency(llvm::StringRef space,
                                                        int64_t bytes) const {
  const MemorySpace *mem = getMemorySpace(space);
  if (!mem || mem->bandwidthBytesPerCycle <= 0.0)
    return std::max<int64_t>(1, bytes);
  return std::max<int64_t>(1, mem->latencyCycles +
                                  static_cast<int64_t>(std::ceil(
                                      bytes / mem->bandwidthBytesPerCycle)));
}

//===----------------------------------------------------------------------===//
// tilesim-migrated micro-architecture queries (root cause ①/②)
//===----------------------------------------------------------------------===//

namespace {
/// Map an element bit width to the dtype key used in the migrated tables.
/// 16-bit types (fp16/bf16) share the FP16 bucket, matching tilesim's CSV.
llvm::StringRef elementBitsToDtypeKey(int elementBits) {
  if (elementBits == 8)
    return "int8";
  if (elementBits == 32)
    return "fp32";
  if (elementBits == 64)
    return "fp64";
  return "fp16"; // fp16 / bf16
}

/// tilesim small-packet bandwidth convention: 1 GB/s == 1073.741824 B/us
/// (it treats "GB" as GiB when converting the measured GB/s figures).
constexpr double kBytesPerUsPerGbs =
    (1024.0 * 1024.0 * 1024.0) / 1e6; // 1073.74...
} // namespace

BandwidthEntry HardwareConfig::lookupBandwidth(llvm::StringRef src,
                                               llvm::StringRef dst, int coreNum,
                                               int64_t pktBytes) const {
  BandwidthEntry result;
  std::string key = (src + ":" + dst).str();
  auto it = bandwidthTables.find(key);
  if (it != bandwidthTables.end()) {
    const BandwidthTable &tbl = it->second;
    double gbps = tbl.singleGbps;
    if (tbl.hasPerCore && !tbl.perCoreGbps.empty()) {
      // tilesim selects the exact core-count row; clamp out-of-range counts
      // to the nearest tabulated value (no interpolation, matching tilesim).
      int cores = coreNum > 0 ? coreNum : activeBandwidthCores;
      int minCores = tbl.perCoreGbps.begin()->first;
      int maxCores = tbl.perCoreGbps.rbegin()->first;
      if (cores <= minCores) {
        gbps = tbl.perCoreGbps.begin()->second;
      } else if (cores >= maxCores) {
        gbps = tbl.perCoreGbps.rbegin()->second;
      } else {
        auto eit = tbl.perCoreGbps.find(cores);
        gbps = (eit != tbl.perCoreGbps.end())
                   ? eit->second
                   : tbl.perCoreGbps.rbegin()->second;
      }
    }
    result.bwGBs = gbps;

    // Small-packet fitting (tilesim pkt_param). tilesim keeps a single global
    // pkt_param applied to every mover; a per-table small_packet entry
    // overrides it. Opt-in via enableSmallPacketBw.
    if (enableSmallPacketBw && pktBytes > 0) {
      const SmallPacketCoeffs *sp = nullptr;
      if (tbl.smallPacket.enabled)
        sp = &tbl.smallPacket;
      else if (hasGlobalSmallPacket && globalSmallPacket.enabled)
        sp = &globalSmallPacket;
      if (sp && pktBytes < sp->thresholdBytes) {
        double a = 0, b = 0;
        if (pktBytes < 64) {
          a = sp->a64;
          b = sp->b64;
        } else if (pktBytes < 128) {
          a = sp->a128;
          b = sp->b128;
        } else {
          a = sp->a256;
          b = sp->b256;
        }
        if (b > 0) {
          // tilesim: bw[B/us] = b / (a*b + pkt_GB) * 1024^3/1e6
          double pktGb =
              static_cast<double>(pktBytes) / (1024.0 * 1024.0 * 1024.0);
          double bwBus = b / (a * b + pktGb) * kBytesPerUsPerGbs;
          result.bwGBs = bwBus / kBytesPerUsPerGbs; // back to GB/s
          result.isSmallPacket = true;
        }
      }
    }
  } else {
    // Unknown (src,dst): fall back to aggregate HBM bandwidth so estimates
    // degrade gracefully instead of returning zero.
    result.bwGBs = getHBMBandwidthGBs();
  }
  return result;
}

VecCycleEntry HardwareConfig::lookupVecCycle(llvm::StringRef intrinsic,
                                             int elementBits) const {
  auto it = vecCycleTables.find(intrinsic);
  if (it != vecCycleTables.end()) {
    const auto &byDtype = it->second;
    llvm::StringRef dt = elementBitsToDtypeKey(elementBits);
    auto dit = byDtype.find(dt);
    if (dit != byDtype.end())
      return dit->second;
    // tilesim falls back to FP32 then warns; we silently fall back.
    auto fp32 = byDtype.find("fp32");
    if (fp32 != byDtype.end())
      return fp32->second;
  }
  return VecCycleEntry{1, 0, 0};
}

int64_t HardwareConfig::estimateTransferCycles(llvm::StringRef src,
                                               llvm::StringRef dst,
                                               int64_t bytes,
                                               int coreNum) const {
  BandwidthEntry bw = lookupBandwidth(src, dst, coreNum, bytes);
  if (bw.bwGBs <= 0)
    bw.bwGBs = getHBMBandwidthGBs();
  if (bw.bwGBs <= 0)
    bw.bwGBs = 1600.0;
  // Replicate tilesim: latency[us] = bytes / (bw[B/us]); cycles = latency *
  // clock[MHz].
  double bwBus = bw.bwGBs * kBytesPerUsPerGbs;
  double latencyUs = static_cast<double>(bytes) / bwBus;
  double cycles = latencyUs * (clockFreqGHz * 1000.0);
  return static_cast<int64_t>(std::ceil(cycles));
}

void HardwareConfig::getCubeModelThroughput(int elementBits, int &basicM,
                                            int &basicK, int &basicN) const {
  basicM = cubeModel.basicM;
  int elemBytes = (elementBits + 7) / 8;
  basicK = elemBytes > 0 ? cubeModel.basicKNumerator / elemBytes
                         : cubeModel.basicKNumerator;
  basicN = cubeModel.basicN;
}

int HardwareConfig::getCubeModelRepeatCycles(int elementBits) const {
  llvm::StringRef dt = elementBitsToDtypeKey(elementBits);
  auto it = cubeModel.repeatCycles.find(dt);
  if (it != cubeModel.repeatCycles.end())
    return it->second;
  // tilesim: fp32 -> 2, fp16/bf16/int8 -> 1.
  return (elementBits == 32) ? 2 : 1;
}

int HardwareConfig::getCubeModelL0TileLimitBytes() const {
  return cubeModel.l0TileLimitKb * 1024;
}

bool HardwareConfig::areMutexUnits(llvm::StringRef a, llvm::StringRef b) const {
  if (a.empty() || b.empty() || a == b)
    return false;
  for (const auto &group : mutexGroups) {
    bool hasA = false, hasB = false;
    for (const auto &u : group) {
      if (u == a)
        hasA = true;
      if (u == b)
        hasB = true;
    }
    if (hasA && hasB)
      return true;
  }
  return false;
}

//===----------------------------------------------------------------------===//
// Pipeline
//===----------------------------------------------------------------------===//

const PipelinePath *
HardwareConfig::getPipelinePath(llvm::StringRef name) const {
  auto it = pipelinePaths.find(name);
  return it != pipelinePaths.end() ? &it->second : nullptr;
}

bool HardwareConfig::canRunInParallel(llvm::StringRef path1,
                                      llvm::StringRef path2) const {
  // Check explicit parallelism flags
  std::string key = path1.str() + "_and_" + path2.str();
  auto it = parallelismFlags.find(key);
  if (it != parallelismFlags.end())
    return it->second;

  // Try reverse order
  key = path2.str() + "_and_" + path1.str();
  it = parallelismFlags.find(key);
  if (it != parallelismFlags.end())
    return it->second;

  return false;
}

//===----------------------------------------------------------------------===//
// Validation
//===----------------------------------------------------------------------===//

bool HardwareConfig::validate(std::string &error) const {
  if (clockFreqGHz <= 0) {
    error = "Clock frequency must be positive";
    return false;
  }

  if (memorySpaces.empty()) {
    error = "At least one memory space must be defined";
    return false;
  }

  if (computeUnits.empty()) {
    error = "At least one compute unit must be defined";
    return false;
  }

  // Validate data mover references
  for (const auto &kv : dataMovers) {
    const auto &dm = kv.second;
    if (!dm.srcSpace.empty() && !getMemorySpace(dm.srcSpace)) {
      error = "Data mover '" + dm.name + "' references unknown source space '" +
              dm.srcSpace + "'";
      return false;
    }
    for (const auto &dst : dm.dstSpaces) {
      if (!getMemorySpace(dst)) {
        error = "Data mover '" + dm.name +
                "' references unknown destination space '" + dst + "'";
        return false;
      }
    }
  }

  return true;
}
