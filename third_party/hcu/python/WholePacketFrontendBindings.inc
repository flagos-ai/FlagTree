// Python construction of the existing bounded whole-packet IR. All physical
// ownership, uniformity, extent and chart checks still run in the lowering.
#include "Dialect/TritonHCUGPU/IR/Dialect.h"
#include "Dialect/TritonHCUGPU/IR/WholePacketLayout.h"
#include "mlir/IR/Builders.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Tools/LayoutUtils.h"
#include "triton/Tools/LinearLayout.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <stdexcept>

namespace {
namespace packet_frontend {
namespace tt = mlir::triton;
namespace tg = mlir::triton::gpu;
namespace hg = mlir::triton::hcugpu;
namespace wp = mlir::triton::hcugpu::wholepacket;
using namespace mlir;

static tg::CTAEncodingAttr oneCTA(MLIRContext *ctx, unsigned rank) {
  tt::LinearLayout::BasesT bases;
  bases[StringAttr::get(ctx, "block")] = {};
  return tg::CTAEncodingAttr::get(ctx,
      tt::LinearLayout(std::move(bases), tt::standardOutDimNames(ctx,rank)));
}
static ModuleOp enablePacketIR(OpBuilder &b) {
  Block *block=b.getInsertionBlock();
  auto module=block?block->getParentOp()->getParentOfType<ModuleOp>():ModuleOp{};
  if (!module) throw std::invalid_argument("packet construction needs a kernel module");
  module->setAttr("ttg.hcu.whole_packet_v1",b.getI32IntegerAttr(1));
  module->setAttr("ttg.hcu.whole_packet_global_bounds",b.getI32IntegerAttr(1));
  return module;
}
static tg::HCUMlsSharedEncodingAttr encoding(Value value) {
  auto type=dyn_cast<tg::MemDescType>(value.getType());
  auto enc=type?dyn_cast<tg::HCUMlsSharedEncodingAttr>(type.getEncoding())
               :tg::HCUMlsSharedEncodingAttr{};
  if (!enc || !wp::PacketSemantics::get(enc.getVersion(),enc.getOpIdx(),enc.getAlt2Kind()) || enc.getOpIdx()>1 ||
      type.getRank()!=2 || type.getElementType().getIntOrFloatBitWidth()!=16)
    throw std::invalid_argument("expected a rank-two 16-bit whole-packet view");
  return enc;
}
static DenseI32ArrayAttr chart(OpBuilder &b, Value view,
                              const std::vector<int32_t> &basis) {
  auto enc=encoding(view);
  auto type=cast<tg::MemDescType>(view.getType());
  if (basis.size()!=12) throw std::invalid_argument("packet source chart needs twelve bases");
  std::array<unsigned,12> values{};
  for (unsigned i=0;i<12;++i) {
    if (basis[i]<0 || basis[i]>=4096)
      throw std::invalid_argument("packet source basis is outside the complete packet");
    values[i]=unsigned(basis[i]);
  }
  // Frontend construction checks the finite chart with one wave; the lowering
  // and wait-count pass revalidate with the actual module warp count.
  auto semantics=*wp::PacketSemantics::get(enc.getVersion(),enc.getOpIdx(),enc.getAlt2Kind());
  bool valid=semantics.prefix
      ? bool(wp::SourceLayout::getActivePrefix(semantics.storageRows(type.getShape()[0],type.getShape()[1]),semantics.storageColumns(type.getShape()[0],type.getShape()[1]),1,values))
      : bool(wp::SourceLayout::get(type.getShape()[0],type.getShape()[1],8,values));
  if (!valid)
    throw std::invalid_argument("packet chart must be bijective and 16-byte contiguous");
  return b.getDenseI32ArrayAttr(basis);
}
static void scalarTypes(ValueRange values,unsigned width,unsigned count) {
  if (values.size()!=count) throw std::invalid_argument("wrong number of packet operands");
  for (Value v:values)
    if (!v || !v.getType().isInteger(width))
      throw std::invalid_argument("packet coordinate, extent or stride has wrong scalar type");
}
static void tokenTypes(ValueRange tokens) {
  for (Value v:tokens)
    if (!v || !isa<tg::AsyncTokenType>(v.getType()))
      throw std::invalid_argument("expected explicit async tokens");
}

static void bind(pybind11::module_ &m) {
  namespace py=pybind11;
  m.def("create_structured_register_load",[](OpBuilder &b,Location loc,Value base,
      Type resultType,const std::vector<Value> &shape,
      const std::vector<Value> &strides,const std::vector<Value> &indices)->Value {
    auto result=dyn_cast<RankedTensorType>(resultType);
    auto ptr=dyn_cast<tt::PointerType>(base.getType());
    if(!result || result.getRank()!=2 ||
       !(result.getElementType().isF16() || result.getElementType().isBF16()) ||
       !isa<tg::DistributedEncodingTrait>(result.getEncoding()) || !ptr ||
       ptr.getAddressSpace()!=1 || ptr.getPointeeType()!=result.getElementType())
      throw std::invalid_argument("structured register load needs matching AS1 f16/bf16 pointer and rank-two distributed result");
    scalarTypes(shape,32,2);scalarTypes(strides,64,2);scalarTypes(indices,32,3);
    SmallVector<int32_t> tile;
    for(int64_t d:result.getShape()) {
      if(d<1 || d>INT32_MAX) throw std::invalid_argument("invalid structured register tile");
      tile.push_back(int32_t(d));
    }
    auto op=b.create<tt::MatrixLoadOp>(loc,resultType,base,shape,strides,tile,
        indices,ArrayRef<int32_t>{},tt::CacheModifier::NONE,
        tt::EvictionPolicy::NORMAL,false);
    op->setAttr("hcu.structured_register",b.getUnitAttr());
    return op.getResult();
  });
  m.def("get_whole_packet_layout",[](OpBuilder &b,unsigned role,
                                      const std::vector<unsigned> &shape)->Attribute {
    if (role>1 || shape.size()!=2 || !shape[0] || !shape[1] ||
        (shape[0]&(shape[0]-1)) || (shape[1]&(shape[1]-1)) ||
        uint64_t(shape[0])*shape[1]!=wp::Elements)
      throw std::invalid_argument("whole-packet layout needs one complete 4096-element role");
    return tg::HCUMlsSharedEncodingAttr::get(b.getContext(),role,shape,
        16,0,256,ArrayRef<unsigned>{1,0},oneCTA(b.getContext(),2));
  });
  m.def("get_packet_prefix_layout",[](OpBuilder &b,unsigned role,
                                      const std::vector<unsigned> &shape)->Attribute {
    if (role>1 || shape.size()!=2 || !shape[0] || !shape[1] ||
        (shape[0]&(shape[0]-1)) || (shape[1]&(shape[1]-1)) ||
        shape[0]>wp::Elements || shape[1]>wp::Elements ||
        uint64_t(shape[0])*shape[1]>wp::Elements ||
        uint64_t(shape[0])*shape[1]<wp::Lanes*wp::VectorElements)
      throw std::invalid_argument("packet-prefix layout needs a whole-wave power-of-two active prefix");
    return tg::HCUMlsSharedEncodingAttr::get(b.getContext(),role,shape,
        16,0,257,ArrayRef<unsigned>{1,0},oneCTA(b.getContext(),2));
  });
  m.def("get_packet_storage_layout",[](OpBuilder &b,unsigned role,
        const std::vector<unsigned> &shape,unsigned transfer,
        const std::vector<unsigned> &order)->Attribute {
    if (role>1 || shape.size()!=2 || !shape[0] || !shape[1] ||
        (shape[0]&(shape[0]-1)) || (shape[1]&(shape[1]-1)) ||
        shape[0]>wp::Elements || shape[1]>wp::Elements ||
        uint64_t(shape[0])*shape[1]>wp::Elements ||
        uint64_t(shape[0])*shape[1]<wp::Lanes*wp::VectorElements ||
        transfer>1 || order.size()!=2 || order[0]>1 || order[1]>1 || order[0]==order[1])
      throw std::invalid_argument("invalid packet storage layout or rank-two permutation");
    unsigned payload=transfer | (order[0]<<1);
    return tg::HCUMlsSharedEncodingAttr::get(b.getContext(),role,shape,
        16,payload,258,ArrayRef<unsigned>{1,0},oneCTA(b.getContext(),2));
  });
  m.def("get_legacy_mmac_layout",[](OpBuilder &b,
        const std::vector<unsigned> &warps,const std::vector<unsigned> &tiles,
        bool transposed)->Attribute {
    auto power=[](unsigned x){return x && !(x&(x-1));};
    if (warps.size()!=2 || tiles.size()!=2 || !power(warps[0]) ||
        !power(warps[1]) || uint64_t(warps[0])*warps[1]>8 ||
        !power(tiles[0]) || !power(tiles[1]) || tiles[0]>16 || tiles[1]>16)
      throw std::invalid_argument("invalid legacy MMAC warp/tile layout");
    return tg::AMDMfmaEncodingAttr::get(b.getContext(),3,warps,
        ArrayRef<unsigned>{16,16,16},transposed,oneCTA(b.getContext(),2),
        tiles,32,tg::MmacLayout::LEGACY);
  });
  m.def("create_whole_packet_owner",[](OpBuilder &b,Location loc,
                                        Type element,unsigned slots)->Value {
    if (!(element.isF16() || element.isBF16()) || !wp::WholeFrame::get(slots,65536))
      throw std::invalid_argument("whole-packet owner needs 16-bit elements and 1..4 slots");
    enablePacketIR(b);
    auto layout=tg::SwizzledSharedEncodingAttr::get(b.getContext(),1,1,1,
        ArrayRef<unsigned>{0},oneCTA(b.getContext(),1));
    auto type=tg::MemDescType::get(ArrayRef<int64_t>{int64_t(slots),8192},
        element,layout,tg::SharedMemorySpaceAttr::get(b.getContext()),true);
    auto alloc=b.create<tg::LocalAllocOp>(loc,type);
    alloc->setAttr("alignment",b.getI32IntegerAttr(16384));
    return alloc.getResult();
  });
  m.def("create_whole_packet_copy",[](OpBuilder &b,Location loc,Value base,
        Value dest,const std::vector<Value> &shape,
        const std::vector<Value> &strides,const std::vector<Value> &indices,
        const std::vector<int32_t> &basis)->Value {
    auto enc=encoding(dest);
    auto ty=cast<tg::MemDescType>(dest.getType());
    auto ptr=dyn_cast<tt::PointerType>(base.getType());
    if (!ptr || ptr.getAddressSpace()!=1 || ptr.getPointeeType()!=ty.getElementType())
      throw std::invalid_argument("packet source needs a matching AS1 element pointer");
    scalarTypes(shape,32,2);scalarTypes(strides,64,2);scalarTypes(indices,32,3);
    auto sourceChart=chart(b,dest,basis);enablePacketIR(b);
    SmallVector<int32_t> tile;
    for (int64_t dim:ty.getShape()) tile.push_back(int32_t(dim));
    auto copy=b.create<hg::MatrixLoadToLocalOp>(loc,base,shape,strides,tile,
        indices,ArrayRef<int32_t>{},tt::CacheModifier::NONE,
        tt::EvictionPolicy::NORMAL,false,dest);
    copy->setAttr("hcu.packet.source_basis",sourceChart);
    return copy.getToken();
  });
  m.def("create_whole_packet_read",[](OpBuilder &b,Location loc,Value source,
        Type resultType,const std::vector<int32_t> &basis,py::object token)->Value {
    auto enc=encoding(source);auto src=cast<tg::MemDescType>(source.getType());
    auto result=dyn_cast<RankedTensorType>(resultType);
    auto dot=result?dyn_cast<tg::DotOperandEncodingAttr>(result.getEncoding())
                   :tg::DotOperandEncodingAttr{};
    // A distributed Linear chart has no dot role: the source encoding owns
    // that role. The lowering checks its complete physical read ownership,
    // source chart, bounds, aliases and tokenless WAR dependencies as usual.
    bool linear=result && isa<tg::LinearEncodingAttr>(result.getEncoding());
    if (!result || (!linear && (!dot || dot.getOpIdx()!=enc.getOpIdx())) ||
        result.getShape()!=src.getShape() || result.getElementType()!=src.getElementType())
      throw std::invalid_argument("packet read needs a matching dot or distributed Linear chart");
    // The existing HCU membar filter is RAW-only for token-attached local
    // loads. Keep the tested tokenless consumer contract and retain WAR sync.
    if (!token.is_none())
      throw std::invalid_argument("packet local reads retain tokenless WAR synchronization");
    auto sourceChart=chart(b,source,basis);
    auto read=b.create<tg::LocalLoadOp>(loc,resultType,source);
    read->setAttr("hcu.packet.source_basis",sourceChart);
    return read.getResult();
  },py::arg("builder"),py::arg("loc"),py::arg("source"),
     py::arg("result_type"),py::arg("basis"),py::arg("token")=py::none());
  m.def("get_whole_packet_token_type",[](OpBuilder &b)->Type {
    return tg::AsyncTokenType::get(b.getContext());
  });
  m.def("create_whole_packet_commit",[](OpBuilder &b,Location loc,
                                          const std::vector<Value> &tokens)->Value {
    tokenTypes(tokens);return b.create<tg::AsyncCommitGroupOp>(loc,tokens).getResult();
  });
  m.def("create_whole_packet_wait",[](OpBuilder &b,Location loc,
        const std::vector<Value> &tokens,int outstanding)->Value {
    tokenTypes(tokens);
    if (outstanding<0 || outstanding>63)
      throw std::invalid_argument("async outstanding group count is outside 0..63");
    return b.create<tg::AsyncWaitOp>(loc,tokens,outstanding).getResult();
  });
  m.def("create_packet_sched_barrier",[](OpBuilder &b,Location loc,int32_t mask) {
    if (mask<0) throw std::invalid_argument("scheduler barrier mask must be nonnegative");
    ROCDL::SchedBarrier::create(b,loc,mask);
  },py::arg("builder"),py::arg("loc"),py::arg("mask")=0);
  m.def("create_buffer_store",[](OpBuilder &b,Location loc,Value value,
        Value base,Value offsets,py::object mask) {
    Value pred=mask.is_none()?Value{}:mask.cast<Value>();
    b.create<hg::BufferStoreOp>(loc,value,base,offsets,Value{},tt::CacheModifier::NONE,pred);
  },py::arg("builder"),py::arg("loc"),py::arg("value"),py::arg("base"),
     py::arg("offsets"),py::arg("mask")=py::none());
}
} // namespace packet_frontend
static void initWholePacketFrontendBindings(pybind11::module_ &m) {
  packet_frontend::bind(m);
}
} // namespace
