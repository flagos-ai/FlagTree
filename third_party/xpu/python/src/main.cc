#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Signals.h"
#include <optional>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>

namespace py = pybind11;

namespace {

enum class FlagCXTeamKind { INTRA, INTER, WORLD };
enum class FlagCXCoopKind { THREAD, WARP, BLOCK };
enum class SyncScope { SYSTEM, DEVICE, BLOCK, THREAD };
enum class MemoryOrder { RELAXED, ACQUIRE, RELEASE, ACQ_REL };

void init_triton_xpu_tle_compat(py::module &&m) {
  auto attr = m.def_submodule("attr");

  py::enum_<FlagCXTeamKind>(attr, "FlagCXTeamKind")
      .value("Intra", FlagCXTeamKind::INTRA)
      .value("Inter", FlagCXTeamKind::INTER)
      .value("World", FlagCXTeamKind::WORLD)
      .def_static("from_str",
                  [](const std::string &name) -> std::optional<FlagCXTeamKind> {
                    if (name == "intra" || name == "intra_node")
                      return FlagCXTeamKind::INTRA;
                    if (name == "inter" || name == "inter_node")
                      return FlagCXTeamKind::INTER;
                    if (name == "world")
                      return FlagCXTeamKind::WORLD;
                    return std::nullopt;
                  })
      .def_static("from_int", [](int value) -> std::optional<FlagCXTeamKind> {
        if (value < 0 || value > 2)
          return std::nullopt;
        return static_cast<FlagCXTeamKind>(value);
      });
  py::enum_<FlagCXCoopKind>(attr, "FlagCXCoopKind")
      .value("Thread", FlagCXCoopKind::THREAD)
      .value("Warp", FlagCXCoopKind::WARP)
      .value("Block", FlagCXCoopKind::BLOCK)
      .def_static("from_str",
                  [](const std::string &name) -> std::optional<FlagCXCoopKind> {
                    if (name == "thread")
                      return FlagCXCoopKind::THREAD;
                    if (name == "warp")
                      return FlagCXCoopKind::WARP;
                    if (name == "block")
                      return FlagCXCoopKind::BLOCK;
                    return std::nullopt;
                  });
  py::enum_<SyncScope>(attr, "SyncScope")
      .value("System", SyncScope::SYSTEM)
      .value("Device", SyncScope::DEVICE)
      .value("Block", SyncScope::BLOCK)
      .value("Thread", SyncScope::THREAD)
      .def_static("from_str",
                  [](const std::string &name) -> std::optional<SyncScope> {
                    if (name == "system")
                      return SyncScope::SYSTEM;
                    if (name == "device")
                      return SyncScope::DEVICE;
                    if (name == "block")
                      return SyncScope::BLOCK;
                    if (name == "thread")
                      return SyncScope::THREAD;
                    return std::nullopt;
                  });
  py::enum_<MemoryOrder>(attr, "MemoryOrder")
      .value("Relaxed", MemoryOrder::RELAXED)
      .value("Acquire", MemoryOrder::ACQUIRE)
      .value("Release", MemoryOrder::RELEASE)
      .value("AcqRel", MemoryOrder::ACQ_REL)
      .def_static("from_str",
                  [](const std::string &name) -> std::optional<MemoryOrder> {
                    if (name == "relaxed")
                      return MemoryOrder::RELAXED;
                    if (name == "acquire")
                      return MemoryOrder::ACQUIRE;
                    if (name == "release")
                      return MemoryOrder::RELEASE;
                    if (name == "acqrel" || name == "acq_rel")
                      return MemoryOrder::ACQ_REL;
                    return std::nullopt;
                  });

  m.def_submodule("utils");
}

} // namespace

#define FOR_EACH_1(MACRO, X) MACRO(X)
#define FOR_EACH_2(MACRO, X, ...) MACRO(X) FOR_EACH_1(MACRO, __VA_ARGS__)
#define FOR_EACH_3(MACRO, X, ...) MACRO(X) FOR_EACH_2(MACRO, __VA_ARGS__)
#define FOR_EACH_4(MACRO, X, ...) MACRO(X) FOR_EACH_3(MACRO, __VA_ARGS__)
#define FOR_EACH_5(MACRO, X, ...) MACRO(X) FOR_EACH_4(MACRO, __VA_ARGS__)
#define FOR_EACH_6(MACRO, X, ...) MACRO(X) FOR_EACH_5(MACRO, __VA_ARGS__)
#define FOR_EACH_7(MACRO, X, ...) MACRO(X) FOR_EACH_6(MACRO, __VA_ARGS__)
#define FOR_EACH_8(MACRO, X, ...) MACRO(X) FOR_EACH_7(MACRO, __VA_ARGS__)

#define FOR_EACH_NARG(...) FOR_EACH_NARG_(__VA_ARGS__, FOR_EACH_RSEQ_N())
#define FOR_EACH_NARG_(...) FOR_EACH_ARG_N(__VA_ARGS__)
#define FOR_EACH_ARG_N(_1, _2, _3, _4, _5, _6, _7, _8, N, ...) N
#define FOR_EACH_RSEQ_N() 8, 7, 6, 5, 4, 3, 2, 1, 0

#define CONCATENATE(x, y) CONCATENATE1(x, y)
#define CONCATENATE1(x, y) x##y

#define FOR_EACH(MACRO, ...)                                                   \
  CONCATENATE(FOR_EACH_, FOR_EACH_NARG_HELPER(__VA_ARGS__))(MACRO, __VA_ARGS__)
#define FOR_EACH_NARG_HELPER(...) FOR_EACH_NARG(__VA_ARGS__)

// New macro to remove parentheses
#define REMOVE_PARENS(...) __VA_ARGS__

// Intermediate macro to ensure correct expansion
#define FOR_EACH_P_INTERMEDIATE(MACRO, ...) FOR_EACH(MACRO, __VA_ARGS__)

// Modified FOR_EACH to handle parentheses
#define FOR_EACH_P(MACRO, ARGS_WITH_PARENS)                                    \
  FOR_EACH_P_INTERMEDIATE(MACRO, REMOVE_PARENS ARGS_WITH_PARENS)

#define DECLARE_BACKEND(name) void init_triton_##name(pybind11::module &&m);

#define INIT_BACKEND(name) init_triton_##name(m.def_submodule(#name));

void init_triton_env_vars(pybind11::module &m);
void init_triton_ir(pybind11::module &&m);
void init_triton_llvm(pybind11::module &&m);
void init_triton_interpreter(pybind11::module &&m);
void init_triton_passes(pybind11::module &&m);
void init_triton_stacktrace_hook(pybind11::module &m);
void init_gluon_ir(pybind11::module &&m);
void init_linear_layout(pybind11::module &&m);
void init_native_specialize(pybind11::module &m);
FOR_EACH_P(DECLARE_BACKEND, TRITON_BACKENDS_TUPLE)

PYBIND11_MODULE(libtriton, m) {
  m.doc() = "Python bindings to the C++ Triton API";
  init_triton_stacktrace_hook(m);
  init_triton_env_vars(m);
  init_native_specialize(m);
  init_triton_ir(m.def_submodule("ir"));
  init_triton_passes(m.def_submodule("passes"));
  init_triton_interpreter(m.def_submodule("interpreter"));
  init_triton_llvm(m.def_submodule("llvm"));
  init_linear_layout(m.def_submodule("linear_layout"));
  init_gluon_ir(m.def_submodule("gluon_ir"));
  init_triton_xpu_tle_compat(m.def_submodule("tle"));
  FOR_EACH_P(INIT_BACKEND, TRITON_BACKENDS_TUPLE)
}
