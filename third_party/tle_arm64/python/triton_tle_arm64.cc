//===- triton_tle_arm64.cc - ARM64 TLE builder injection ----------*- C++
//-*-===//
//
// Pybind plugin that injects ARM64 TLE CPU ops into TritonOpBuilder.
//
// Mirrors the calling philosophy of third_party/tle/python/triton_tle_dsa.cc:
// the builder methods are registered with pybind11 .def() on the core
// TritonOpBuilder class (obtained from triton._C.libtriton.ir), and each one
// loads the TritonCPU dialect defensively before building its op.
//
// The MLIR op classes (cpu::SdotGemvOp, cpu::FusedMlpOp, etc.) live in
// flagtree-cpu's TritonCPU dialect (TritonCPUOps.td) and compile into
// libtriton.so. This plugin only adds the Python builder methods so that
// tle_ops.py can call _builder.create_cpu_*().
//
//===----------------------------------------------------------------------===//

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Value.h"

#include "ir.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonCPU/IR/Dialect.h"

namespace py = pybind11;
using namespace mlir;
using namespace triton;

static void init_triton_tle_arm64_ir(py::module m) {
  (void)m;
  auto core_ir = py::module::import("triton._C.libtriton.ir");
  auto builder_cls =
      core_ir.attr("builder").cast<py::class_<TritonOpBuilder>>();

  builder_cls
      .def("create_cpu_fused_decode_step",
           [](TritonOpBuilder &self, Value &tok_id, Value &pos, Value &embed,
              Value &layer_ptrs, Value &kc, Value &vc, Value &rcos, Value &rsin,
              Value &fnorm, Value &lm_packed, Value &lm_scale, Value &hidden,
              Value &hd, Value &nh, Value &nkv, Value &inter, Value &vocab,
              Value &nlayers, Value &maxseq, float rms_eps) -> Value {
             self.getContext()->getOrLoadDialect<cpu::TritonCPUDialect>();
             auto op = self.create<cpu::FusedDecodeStepOp>(
                 tok_id, pos, embed, layer_ptrs, kc, vc, rcos, rsin, fnorm,
                 lm_packed, lm_scale, hidden, hd, nh, nkv, inter, vocab,
                 nlayers, maxseq, self.getBuilder().getF32FloatAttr(rms_eps));
             return op.getNextToken();
           })
      .def("create_cpu_fused_transformer_layer",
           [](TritonOpBuilder &self, Value &hidden, Value &wq, Value &wk,
              Value &wv, Value &wo, Value &wq_s, Value &wk_s, Value &wv_s,
              Value &wo_s, Value &q_norm, Value &k_norm, Value &cos_emb,
              Value &sin_emb, Value &k_cache, Value &v_cache, Value &cache_pos,
              Value &max_seq, Value &gate, Value &up, Value &down,
              Value &gate_s, Value &up_s, Value &down_s, Value &in_norm,
              Value &post_norm, Value &hidden_dim, Value &head_dim,
              Value &n_heads, Value &n_kv_heads, Value &intermediate,
              float rms_eps) {
             self.getContext()->getOrLoadDialect<cpu::TritonCPUDialect>();
             self.create<cpu::FusedTransformerLayerOp>(
                 hidden, wq, wk, wv, wo, wq_s, wk_s, wv_s, wo_s, q_norm, k_norm,
                 cos_emb, sin_emb, k_cache, v_cache, cache_pos, max_seq, gate,
                 up, down, gate_s, up_s, down_s, in_norm, post_norm, hidden_dim,
                 head_dim, n_heads, n_kv_heads, intermediate,
                 self.getBuilder().getF32FloatAttr(rms_eps));
           })
      .def("create_cpu_fused_mlp",
           [](TritonOpBuilder &self, Value &x, Value &gp, Value &up, Value &gs,
              Value &us, Value &out, Value &K, Value &N) {
             self.getContext()->getOrLoadDialect<cpu::TritonCPUDialect>();
             self.create<cpu::FusedMlpOp>(x, gp, up, gs, us, out, K, N);
           })
      .def("create_cpu_flash_attn_decode",
           [](TritonOpBuilder &self, Value &q, Value &k, Value &v, Value &out,
              Value &seq_len, Value &head_dim, float sm_scale, Value &num_heads,
              Value &num_kv_heads, Value &stride_kn, Value &stride_vn) {
             self.getContext()->getOrLoadDialect<cpu::TritonCPUDialect>();
             self.create<cpu::FlashAttnDecodeOp>(
                 q, k, v, out, seq_len, head_dim,
                 self.getBuilder().getF32FloatAttr(sm_scale), num_heads,
                 num_kv_heads, stride_kn, stride_vn);
           })
      .def("create_cpu_rms_norm",
           [](TritonOpBuilder &self, Value &x, Value &weight, Value &out,
              Value &D, float eps) {
             self.getContext()->getOrLoadDialect<cpu::TritonCPUDialect>();
             self.create<cpu::RmsNormOp>(
                 x, weight, out, D, self.getBuilder().getF32FloatAttr(eps));
           })
      .def("create_cpu_swiglu",
           [](TritonOpBuilder &self, Value &gate, Value &up, Value &out,
              Value &N) {
             self.getContext()->getOrLoadDialect<cpu::TritonCPUDialect>();
             self.create<cpu::SwigluOp>(gate, up, out, N);
           })
      .def("create_cpu_sdot_gemv",
           [](TritonOpBuilder &self, Value &a_ptr, Value &b_ptr, Value &c_ptr,
              Value &K, Value &N) {
             self.getContext()->getOrLoadDialect<cpu::TritonCPUDialect>();
             self.create<cpu::SdotGemvOp>(a_ptr, b_ptr, c_ptr, K, N);
           })
      .def("create_cpu_sdot_gemv_fused_bf16",
           [](TritonOpBuilder &self, Value &x_ptr, Value &b_ptr, Value &ws_ptr,
              Value &out_ptr, Value &K, Value &N) {
             self.getContext()->getOrLoadDialect<cpu::TritonCPUDialect>();
             self.create<cpu::SdotGemvFusedBf16Op>(x_ptr, b_ptr, ws_ptr,
                                                   out_ptr, K, N);
           })
      .def("create_cpu_sdot_pack_weights",
           [](TritonOpBuilder &self, Value &b_ptr, Value &bp_ptr, Value &K,
              Value &N) {
             self.getContext()->getOrLoadDialect<cpu::TritonCPUDialect>();
             self.create<cpu::SdotPackWeightsOp>(b_ptr, bp_ptr, K, N);
           })
      .def("create_cpu_neon_sdot",
           [](TritonOpBuilder &self, Value &acc, Value &a, Value &b) -> Value {
             self.getContext()->getOrLoadDialect<cpu::TritonCPUDialect>();
             return self.create<cpu::NeonSdotTensorOp>(acc.getType(), acc, a,
                                                       b);
           });
}

void init_triton_tle_arm64(py::module &&m) {
  init_triton_tle_arm64_ir(std::move(m));
}
