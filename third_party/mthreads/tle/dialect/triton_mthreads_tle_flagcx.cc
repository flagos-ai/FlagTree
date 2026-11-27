#ifdef __TLE__
#ifdef FLAGCX_ENABLED

#include "ir.h"
#include "tle/dialect/include/IR/Dialect.h"
#include "tle/dialect/include/IR/VerifyUtils.h"
#include <optional>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

namespace py = pybind11;
namespace tle = mlir::triton::tle;

using mlir::DenseI32ArrayAttr;
using mlir::IntegerAttr;
using mlir::OpState;
using mlir::StringAttr;
using mlir::Type;
using mlir::Value;
using ::TritonOpBuilder;

void init_triton_mthreads_tle_flagcx_attr(py::module m) {
  py::enum_<tle::SignalOpKind>(m, "SignalOpKind")
      .value("Inc", tle::SignalOpKind::INC)
      .value("Add", tle::SignalOpKind::ADD)
      .def_static(
          "from_str",
          [](std::string name) { return tle::symbolizeSignalOpKind(name); },
          py::arg("name"));
  py::enum_<tle::FlagCXTeamKind>(m, "FlagCXTeamKind")
      .value("Intra", tle::FlagCXTeamKind::INTRA)
      .value("Inter", tle::FlagCXTeamKind::INTER)
      .value("World", tle::FlagCXTeamKind::WORLD)
      .def_static(
          "from_str",
          [](std::string name) { return tle::symbolizeFlagCXTeamKind(name); },
          py::arg("name"))
      .def_static(
          "from_int",
          [](int value) -> std::optional<tle::FlagCXTeamKind> {
            if (value < 0 || value > tle::getMaxEnumValForFlagCXTeamKind())
              return std::nullopt;
            return static_cast<tle::FlagCXTeamKind>(value);
          },
          py::arg("value"));
  py::enum_<tle::FlagCXCoopKind>(m, "FlagCXCoopKind")
      .value("Thread", tle::FlagCXCoopKind::THREAD)
      .value("Warp", tle::FlagCXCoopKind::WARP)
      .value("Block", tle::FlagCXCoopKind::BLOCK)
      .def_static(
          "from_str",
          [](std::string name) { return tle::symbolizeFlagCXCoopKind(name); },
          py::arg("name"));
  py::enum_<tle::SignalWaitKind>(m, "SignalWaitKind")
      .value("Signal", tle::SignalWaitKind::SIGNAL)
      .value("Shadow", tle::SignalWaitKind::SHADOW)
      .value("Counter", tle::SignalWaitKind::COUNTER)
      .def_static(
          "from_str",
          [](std::string name) { return tle::symbolizeSignalWaitKind(name); },
          py::arg("name"));
  py::enum_<tle::SyncScope>(m, "SyncScope")
      .value("System", tle::SyncScope::SYSTEM)
      .value("Device", tle::SyncScope::DEVICE)
      .value("Block", tle::SyncScope::BLOCK)
      .value("Thread", tle::SyncScope::THREAD)
      .def_static(
          "from_str",
          [](std::string name) { return tle::symbolizeSyncScope(name); },
          py::arg("name"));
  py::enum_<tle::MemoryOrder>(m, "MemoryOrder")
      .value("Relaxed", tle::MemoryOrder::RELAXED)
      .value("Acquire", tle::MemoryOrder::ACQUIRE)
      .value("Release", tle::MemoryOrder::RELEASE)
      .value("AcqRel", tle::MemoryOrder::ACQ_REL)
      .def_static(
          "from_str",
          [](std::string name) { return tle::parseMemoryOrder(name); },
          py::arg("name"));
}

void init_triton_mthreads_tle_flagcx_utils(py::module m) {
  m.def(
      "verify_signal",
      [](tle::SignalOpKind kind, std::optional<Value> value,
         tle::SyncScope scope) {
        if (auto err = tle::Signal::verifySignalOp(
                kind, value.value_or(Value()), scope))
          throw py::value_error(*err);
      },
      py::arg("kind"), py::arg("value") = py::none(),
      py::arg("scope") = tle::SyncScope::SYSTEM);
  m.def(
      "verify_signal_wait",
      [](tle::SignalWaitKind kind, std::optional<Value> target,
         tle::MemoryOrder order) {
        if (auto err = tle::Signal::verifySignalWaitOp(
                kind, target.value_or(Value()), order))
          throw py::value_error(*err);
      },
      py::arg("kind"), py::arg("target") = py::none(),
      py::arg("order") = tle::MemoryOrder::ACQUIRE);
}

void init_triton_mthreads_tle_flagcx_ir(py::module /*m*/) {
  auto *builderClsPtr = ir::getBuilderClass();
  if (!builderClsPtr)
    throw std::runtime_error("triton IR builder class is not initialized");
  auto &builderCls = *builderClsPtr;

  builderCls
      .def("create_distributed_barrier",
           [](TritonOpBuilder &self) -> void {
             self.create<tle::DistributedBarrierOp>(
                 Value(), StringAttr(), StringAttr(), tle::MemoryOrderAttr(),
                 StringAttr(), IntegerAttr(), IntegerAttr(),
                 tle::SyncScopeAttr(), IntegerAttr(), DenseI32ArrayAttr(),
                 DenseI32ArrayAttr(), DenseI32ArrayAttr(), DenseI32ArrayAttr());
           })
      .def(
          "create_distributed_barrier",
          [](TritonOpBuilder &self, std::optional<Value> src,
             size_t barrier_index = 0, const std::string &space = "device",
             const std::string &group_kind = "block",
             tle::MemoryOrder order = tle::MemoryOrder::ACQ_REL,
             const std::string &barrier_kind = "sync", size_t context_id = 0,
             tle::SyncScope memory_scope = tle::SyncScope::SYSTEM) -> void {
            auto &builder = self.getBuilder();
            auto getOptStrAttr = [&](const std::string &s) -> StringAttr {
              return s.empty() ? StringAttr() : builder.getStringAttr(s);
            };
            self.create<tle::DistributedBarrierOp>(
                src.value_or(Value()), getOptStrAttr(space),
                getOptStrAttr(barrier_kind),
                builder.getAttr<tle::MemoryOrderAttr>(order),
                getOptStrAttr(group_kind),
                builder.getI32IntegerAttr(static_cast<int32_t>(barrier_index)),
                builder.getI32IntegerAttr(static_cast<int32_t>(context_id)),
                builder.getAttr<tle::SyncScopeAttr>(memory_scope), IntegerAttr(),
                DenseI32ArrayAttr(), DenseI32ArrayAttr(), DenseI32ArrayAttr(),
                DenseI32ArrayAttr());
          },
          py::arg("src") = py::none(), py::arg("barrier_index"),
          py::arg("space"), py::arg("group_kind"), py::arg("order"),
          py::arg("barrier_kind"), py::arg("context_id") = 0,
          py::arg("memory_scope") = "system")
      .def(
          "create_distributed_barrier",
          [](TritonOpBuilder &self, const std::string &groupKind,
             const std::vector<int32_t> &groupShape,
             const std::vector<int32_t> &groupAxes,
             const std::vector<int32_t> &groupMask,
             const std::vector<int32_t> &groupDomainShape) -> void {
            auto &builder = self.getBuilder();
            auto *ctx = builder.getContext();
            StringAttr kindAttr;
            IntegerAttr rankAttr;
            DenseI32ArrayAttr shapeAttr, axesAttr, maskAttr, domainShapeAttr;
            if (!groupKind.empty())
              kindAttr = builder.getStringAttr(groupKind);
            if (!groupShape.empty() || !groupAxes.empty() ||
                !groupMask.empty() || !groupDomainShape.empty()) {
              rankAttr = builder.getI32IntegerAttr(
                  static_cast<int32_t>(groupShape.size()));
              if (!groupShape.empty())
                shapeAttr = DenseI32ArrayAttr::get(ctx, groupShape);
              if (!groupAxes.empty())
                axesAttr = DenseI32ArrayAttr::get(ctx, groupAxes);
              if (!groupMask.empty())
                maskAttr = DenseI32ArrayAttr::get(ctx, groupMask);
              if (!groupDomainShape.empty())
                domainShapeAttr = DenseI32ArrayAttr::get(ctx, groupDomainShape);
            }
            self.create<tle::DistributedBarrierOp>(
                Value(), StringAttr(), StringAttr(), tle::MemoryOrderAttr(),
                kindAttr, IntegerAttr(), IntegerAttr(), tle::SyncScopeAttr(),
                rankAttr, shapeAttr, axesAttr, maskAttr, domainShapeAttr);
          },
          py::arg("group_kind"), py::arg("group_shape"), py::arg("group_axes"),
          py::arg("group_mask"),
          py::arg("group_domain_shape") = std::vector<int32_t>{})
      .def(
          "create_signal",
          [](TritonOpBuilder &self, Value comm, Value peer, Value slotId,
             std::optional<Value> value, tle::SignalOpKind signalOp,
             tle::FlagCXTeamKind teamKind, tle::FlagCXCoopKind coopKind,
             int32_t contextId, tle::SyncScope scope) -> void {
            auto &builder = self.getBuilder();
            if (auto err = tle::Signal::verifySignalOp(
                    signalOp, value.value_or(Value()), scope))
              throw py::value_error(*err);
            self.create<tle::SignalOp>(
                comm, peer, slotId, value.value_or(Value()),
                builder.getAttr<tle::SignalOpKindAttr>(signalOp),
                builder.getAttr<tle::FlagCXTeamKindAttr>(teamKind),
                builder.getAttr<tle::FlagCXCoopKindAttr>(coopKind),
                builder.getI32IntegerAttr(contextId),
                builder.getAttr<tle::SyncScopeAttr>(scope));
          },
          py::arg("comm"), py::arg("peer"), py::arg("slot_id"),
          py::arg("value"), py::arg("signal_op"), py::arg("team_kind"),
          py::arg("coop_kind"), py::arg("context_id"), py::arg("scope"))
      .def(
          "create_signal_wait",
          [](TritonOpBuilder &self, Value comm_dev_ptr, Value slot_id,
             tle::SignalWaitKind wait_kind, std::optional<Value> target,
             tle::FlagCXCoopKind coop_kind, int32_t contextId,
             tle::MemoryOrder order) -> void {
            auto &builder = self.getBuilder();
            if (auto err = tle::Signal::verifySignalWaitOp(
                    wait_kind, target.value_or(Value()), order))
              throw py::value_error(*err);
            self.create<tle::SignalWaitOp>(
                comm_dev_ptr, slot_id,
                builder.getAttr<tle::SignalWaitKindAttr>(wait_kind),
                target.value_or(Value()),
                builder.getAttr<tle::FlagCXCoopKindAttr>(coop_kind),
                builder.getI32IntegerAttr(contextId),
                builder.getAttr<tle::MemoryOrderAttr>(order));
          },
          py::arg("comm"), py::arg("slot_id"), py::arg("wait_kind"),
          py::arg("target"), py::arg("coop_kind"), py::arg("context_id"),
          py::arg("order"))
      .def(
          "create_remote_pointers",
          [](TritonOpBuilder &self, Type resultTy, std::optional<Value> src,
             Value shardId, const std::string &space,
             std::optional<Value> offset, std::optional<Value> comm,
             std::optional<int32_t> contextId,
             std::optional<tle::FlagCXCoopKind> coopKind) -> OpState {
            auto &builder = self.getBuilder();
            static const std::unordered_set<std::string> valid = {
                "cluster", "chiplet", "device", "node"};
            if (valid.find(space) == valid.end())
              throw std::invalid_argument(
                  "Invalid space: " + space +
                  ". Expected one of: cluster, chiplet, device, node.");
            IntegerAttr contextIdAttr =
                contextId ? builder.getI32IntegerAttr(*contextId)
                          : IntegerAttr();
            tle::FlagCXCoopKindAttr coopKindAttr =
                coopKind ? builder.getAttr<tle::FlagCXCoopKindAttr>(*coopKind)
                         : tle::FlagCXCoopKindAttr();
            return self.create<tle::RemotePointersOp>(
                resultTy, src.value_or(Value()), comm.value_or(Value()),
                shardId, builder.getStringAttr(space), offset.value_or(Value()),
                contextIdAttr, coopKindAttr);
          },
          py::arg("resultTy"), py::arg("src") = py::none(), py::arg("shardId"),
          py::arg("space"), py::arg("offset") = py::none(),
          py::arg("comm") = py::none(), py::arg("context_id") = py::none(),
          py::arg("coop_kind") = py::none())
      .def("get_device_id",
           [](TritonOpBuilder &self, Type resultTy,
              std::optional<Value> src) -> Value {
             return self.create<tle::GetDeviceIdOp>(resultTy,
                                                    src.value_or(Value()));
           })
      .def("get_world_rank",
           [](TritonOpBuilder &self, Type resultTy, Value src) -> Value {
             return self.create<tle::GetWorldRankOp>(resultTy, src);
           })
      .def("get_n_pes",
           [](TritonOpBuilder &self, Type resultTy,
              std::optional<Value> src) -> Value {
             return self.create<tle::GetNumPesOp>(resultTy,
                                                  src.value_or(Value()));
           });
}

void init_triton_mthreads_tle_flagcx(py::module m) {
  init_triton_mthreads_tle_flagcx_attr(m.def_submodule("attr"));
  init_triton_mthreads_tle_flagcx_utils(m.def_submodule("utils"));
  init_triton_mthreads_tle_flagcx_ir(m);
}

#endif // FLAGCX_ENABLED
#endif // __TLE__
