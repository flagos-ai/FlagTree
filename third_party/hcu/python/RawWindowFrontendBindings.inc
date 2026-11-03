// Included after WholePacketFrontendBindings.inc, in the same frontend TU.
#include "Dialect/TritonHCUGPU/IR/RawWindowLayout.h"
namespace {
static void initRawWindowFrontendBindings(pybind11::module_ &m) {
  using namespace mlir; namespace tg=triton::gpu; namespace hg=triton::hcugpu;
  namespace rw=triton::hcugpu::rawwindow; namespace py=pybind11;
  auto ints=[](const std::vector<int32_t>& v){return ArrayRef<int32_t>(v);};
  m.def("create_raw_window_owner",[](OpBuilder& b,Location loc,Type element,unsigned slots)->Value {
    if(!(element.isF16()||element.isBF16())||slots<1||slots>2)
      throw std::invalid_argument("raw window owner needs half elements and one or two slots");
    auto layout=tg::SwizzledSharedEncodingAttr::get(b.getContext(),1,1,1,
      ArrayRef<unsigned>{0},packet_frontend::oneCTA(b.getContext(),1));
    auto type=tg::MemDescType::get(ArrayRef<int64_t>{int64_t(slots),rw::Elements},element,
      layout,tg::SharedMemorySpaceAttr::get(b.getContext()),true);
    auto owner=b.create<tg::LocalAllocOp>(loc,type);
    owner->setAttr("alignment",b.getI32IntegerAttr(rw::Bytes));return owner.getResult();
  });
  m.def("create_raw_window_copy",[](OpBuilder& b,Location loc,Value base,Value dest,
      Value rows,Value row,Value column,Value enabled,int pitch,
      const std::vector<int32_t>& shape,const std::vector<int32_t>& basis)->Value {
    auto op=b.create<hg::RawWindowCopyOp>(loc,tg::AsyncTokenType::get(b.getContext()),
      base,dest,rows,row,column,enabled,b.getI32IntegerAttr(pitch),
      b.getDenseI32ArrayAttr(shape),b.getDenseI32ArrayAttr(basis));return op.getToken();
  });
  m.def("create_raw_window_read",[](OpBuilder& b,Location loc,Value source,Type result,
      const std::vector<int32_t>& shape,const std::vector<int32_t>& basis,
      const std::vector<int32_t>& origin,const std::vector<int32_t>& order)->Value {
    return b.create<hg::RawWindowReadOp>(loc,result,source,b.getDenseI32ArrayAttr(shape),
      b.getDenseI32ArrayAttr(basis),b.getDenseI32ArrayAttr(origin),b.getDenseI32ArrayAttr(order)).getResult();
  });
  m.def("create_raw_window_read_full",[](OpBuilder& b,Location loc,Value source,Type result,
      const std::vector<int32_t>& shape,const std::vector<int32_t>& basis,
      const std::vector<int32_t>& origin,const std::vector<int32_t>& order)->Value {
    auto op=b.create<hg::RawWindowReadOp>(loc,result,source,b.getDenseI32ArrayAttr(shape),
      b.getDenseI32ArrayAttr(basis),b.getDenseI32ArrayAttr(origin),b.getDenseI32ArrayAttr(order));
    op->setAttr("hcu.full_read_completion",b.getUnitAttr());return op.getResult();
  });
  m.def("create_raw_window_read_completion",[](OpBuilder& b,Location loc,Value source,Type result,
      const std::vector<int32_t>& shape,const std::vector<int32_t>& basis,
      const std::vector<int32_t>& origin,const std::vector<int32_t>& order,
      bool localCompletion,bool globalCompletion)->Value {
    auto op=b.create<hg::RawWindowReadOp>(loc,result,source,b.getDenseI32ArrayAttr(shape),
      b.getDenseI32ArrayAttr(basis),b.getDenseI32ArrayAttr(origin),b.getDenseI32ArrayAttr(order));
    if(localCompletion)op->setAttr("hcu.full_read_completion",b.getUnitAttr());
    if(globalCompletion)op->setAttr("hcu.global_read_completion",b.getUnitAttr());
    return op.getResult();
  });
  m.def("create_raw_warp_group_id",[](OpBuilder& b,Location loc,int size)->Value {
    if(size<=0||!llvm::isPowerOf2_32(size))throw std::invalid_argument("invalid wave group size");
    return b.create<hg::WarpGroupIdOp>(loc,b.getI32Type(),b.getI32IntegerAttr(size)).getResult();
  });
  m.def("create_raw_window_wait",[](OpBuilder& b,Location loc,Value token) {
    b.create<hg::RawWindowWaitOp>(loc,token);
  });
}
}
