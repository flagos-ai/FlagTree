// Descriptor address decomposition is independent of the LDS chart. Keep
// the physical base and row bounds; only factor each valid local-row address.
// This gfx936 cache policy targets 256B-aligned source row pitches. Other
// alignments retain their measured physical-pitch distribution.
namespace {
static int64_t cacheDistributedRecordPitch(Operation *op, int64_t pitch,
                                           int64_t tileRows,
                                           int64_t tileColumns) {
  // Profitability varies with cache distribution, independently of validity.
  // The physical representation retains all row bounds and address proofs.
  auto policy = op->getParentOfType<ModuleOp>()->getAttrOfType<BoolAttr>(
      "ttg.hcu.normalize_buffer_record_pitch");
  if (policy && !policy.getValue())
    return pitch;
  if (pitch <= 0 || pitch >= 0x2000 || pitch % 128 || tileRows <= 0 ||
      tileColumns <= 0 || tileColumns > pitch)
    return pitch;
  uint64_t record = llvm::bit_floor(uint64_t(pitch));
  uint64_t residual = uint64_t(pitch) - record;
  if (!residual)
    return pitch;
  // The index and NR are unchanged, so rejected rows stay rejected. For valid
  // local rows, offset + row*2*residual must not wrap the hardware u32 offset.
  uint64_t columnBytes = 2 * uint64_t(tileColumns) + 16;
  if (columnBytes > UINT32_MAX ||
      uint64_t(tileRows - 1) > (UINT32_MAX - columnBytes) / (2 * residual))
    return pitch;
  return int64_t(record);
}
} // namespace

// Hardware mode and scalar resources come from this invocation's actual
// matrix operands and complete SharedLinear chart, not a source-layout ABI.
static LogicalResult lowerMatrixSharedCopy(
    triton::hcugpu::MatrixLoadToLocalOp op,
    triton::hcugpu::MatrixLoadToLocalOpAdaptor a,
    const mlir::triton::HCU::MatrixSharedCopyPlan &p,
    const LLVMTypeConverter &converter, ConversionPatternRewriter &rw) {
  using namespace mlir::triton::HCU;
  auto &rewriter = rw;
  auto loc = op.getLoc(); auto b = TritonLLVMOpBuilder(loc,rw);
  auto *ctx = rw.getContext();
  auto elem = converter.convertType(op.getDest().getType().getElementType());
  auto smem = getSharedMemoryObjectFromStruct(loc,a.getDest(),elem,rw);
  Value affine = smem.getShmemOffset(loc,rw,op.getDest().getType());
  Value lane,wave;
  {
    OpBuilder::InsertionGuard guard(rw);
    if (auto *region=warpSpecializeScope(op); region && !region->empty())
      rw.setInsertionPointToStart(&region->front());
    auto ids=getLaneAndWarpId(rw,loc);lane=ids.first;
    wave=rw.create<ROCDL::ReadfirstlaneOp>(loc,i32_ty,ids.second);
  }
  Value rowOrigin=a.getIndices()[p.rowAxis], colOrigin=a.getIndices()[p.columnAxis];
  bool completeColumns=p.wholeColumns || p.columnInBounds;
  bool uniformRectangle=(p.fullRows || p.wholeRows) && completeColumns;
  bool rowBoundsHandled=p.fullRows || uniformRectangle;
  Value columnValid=p.columnInBounds?b.i1_val(true):Value(b.and_(b.icmp_sge(colOrigin,b.i32_val(0)),b.icmp_ult(colOrigin,b.i32_val(p.columns))));
  Value enabled=b.icmp_ne(a.getIndices()[2],b.i32_val(0));
  Value base,records,rawUniform,rebase,rowBase, columnDelta;
  if (p.affinePartialColumns) {
    auto shape = op.getDest().getType().getShape();
    Value row = b.sext(i64_ty, rowOrigin);
    Value col = b.sext(i64_ty, colOrigin);
    Value rowValid = p.fullRows
        ? b.i1_val(true)
        : Value(b.and_(b.icmp_sge(row, b.i64_val(0)),
                       b.icmp_sle(b.add(row, b.i64_val(shape[p.rowAxis])),
                                  b.i64_val(p.rows))));
    Value columnsOverlap =
        b.and_(b.icmp_slt(col, b.i64_val(p.columns)),
               b.icmp_sgt(b.add(col, b.i64_val(shape[p.columnAxis])),
                          b.i64_val(0)));
    Value safeRow = b.select(rowValid, rowOrigin, b.i32_val(0));
    Value safeColumn = b.select(columnValid, colOrigin, b.i32_val(0));
    // Rebase a negative column origin at zero so a valid suffix can still
    // be copied. Valid vectors stay within the tile-local byte descriptor.
    columnDelta = b.sub(colOrigin, safeColumn);
    Value offset =
        b.shl(b.add(b.mul(b.sext(i64_ty, safeRow), b.i64_val(p.pitch)),
                    b.sext(i64_ty, safeColumn)),
              b.i64_val(1));
    base = b.gep(ptr_ty(ctx, 1), i8_ty, a.getBase(), offset);
    records = b.select(b.and_(enabled, b.and_(rowValid, columnsOverlap)),
                       b.i32_val(p.tileSpanBytes), b.i32_val(0));
  } else if (p.structured) {
    rowBase=b.select(b.icmp_sge(rowOrigin,b.i32_val(0)),rowOrigin,b.i32_val(0));
    Value safeCol=b.select(columnValid,colOrigin,b.i32_val(0));
    Value start=b.add(b.mul(b.zext(i64_ty,rowBase),b.i64_val(p.pitch)),b.sext(i64_ty,safeCol));
    Value byteOrigin=b.sub(b.shl(start,b.i64_val(1)),b.i64_val(16));
    base=b.gep(ptr_ty(ctx,1),i8_ty,a.getBase(),byteOrigin);
    Value remaining=memory_analysis::uniformSaturatingSub(loc,rw,b.i32_val(p.rows),rowBase);
    records=b.select(b.and_(enabled,columnValid),remaining,b.i32_val(0));
    if (!completeColumns) {
      // An invalid tile origin can still have a valid suffix. Retain the full
      // source and validate each vector; never clear that suffix wholesale.
      base=b.gep(ptr_ty(ctx,1),i8_ty,a.getBase(),b.i64_val(uint64_t(-16)));
      records=b.select(enabled,b.i32_val(p.rows),b.i32_val(0));
    }
  } else {
    Value start=b.add(b.mul(b.sext(i64_ty,rowOrigin),b.i64_val(p.pitch)),b.sext(i64_ty,colOrigin));
    rawUniform=b.shl(start,b.i64_val(1));
    rebase=b.select(b.icmp_sge(rawUniform,b.i64_val(0)),rawUniform,b.i64_val(0));
    base=b.gep(ptr_ty(ctx,1),i8_ty,a.getBase(),rebase);
    if (uniformRectangle) {
      // Scalar bounds or whole-tile alignment prove an all-valid/all-zero
      // rectangle. Keep invalid row/column origins on the safe source base and
      // issue NR0 so every destination byte is still overwritten with zero.
      auto shape=op.getDest().getType().getShape();
      Value ro=b.sext(i64_ty,rowOrigin);
      Value rowValid=p.fullRows?b.i1_val(true):Value(b.and_(
          b.icmp_sge(ro,b.i64_val(0)),
          b.icmp_sle(b.add(ro,b.i64_val(shape[p.rowAxis])),b.i64_val(p.rows))));
      Value safeRow=p.fullRows?rowOrigin:Value(b.select(rowValid,rowOrigin,b.i32_val(0)));
      Value safeCol=b.select(columnValid,colOrigin,b.i32_val(0));
      Value tileStart=b.add(b.mul(b.sext(i64_ty,safeRow),b.i64_val(p.pitch)),b.sext(i64_ty,safeCol));
      Value tileBytes=b.shl(tileStart,b.i64_val(1));
      base=b.gep(ptr_ty(ctx,1),i8_ty,a.getBase(),tileBytes);
      records=b.select(b.and_(enabled,b.and_(columnValid,rowValid)),b.i32_val(p.tileSpanBytes),b.i32_val(0));
    } else if (p.sourceBytes >= (uint64_t(1) << 31)) {
      // Large global tensors are legal: use a tile-local descriptor and prove
      // each requested row/column instead of truncating the global byte count.
      // Far-away invalid origins use the safe source base with NR0.
      auto shape=op.getDest().getType().getShape();
      Value ro=b.sext(i64_ty,rowOrigin), co=b.sext(i64_ty,colOrigin);
      Value rowsOverlap=b.and_(b.icmp_slt(ro,b.i64_val(p.rows)),
          b.icmp_sgt(b.add(ro,b.i64_val(shape[p.rowAxis])),b.i64_val(0)));
      Value colsOverlap=b.and_(b.icmp_slt(co,b.i64_val(p.columns)),
          b.icmp_sgt(b.add(co,b.i64_val(shape[p.columnAxis])),b.i64_val(0)));
      Value intersects=b.and_(rowsOverlap,colsOverlap);
      Value safeOrigin=b.select(intersects,rawUniform,b.i64_val(0));
      base=b.gep(ptr_ty(ctx,1),i8_ty,a.getBase(),safeOrigin);
      Value valid=b.and_(enabled,intersects);
      if (completeColumns) valid=b.and_(valid,columnValid);
      records=b.select(valid,b.i32_val(p.tileSpanBytes),b.i32_val(0));
    } else {
      Value fits=b.icmp_ule(rebase,b.i64_val(p.sourceBytes));
      // A very negative origin must not truncate back into a valid i32 byte
      // offset. Within this reach bound the relative offset is in
      // [-tileSpan,tileSpan], so all negative values are OOB after truncation.
      Value reaches=b.icmp_sge(rawUniform,b.i64_val(-int64_t(p.tileSpanBytes)));
      Value valid=b.and_(enabled,b.and_(fits,reaches));
      if (completeColumns) valid=b.and_(valid,columnValid);
      // On the valid path rebase fits before narrowing. Use a bounded
      // subtraction so LLVM can still prove the descriptor width after
      // simplifying the guard and promoting the arithmetic.
      Value rebase32 =b.trunc(i32_ty,rebase);
      Value remaining= LLVM::USubSat::create(
          rw, loc, i32_ty, b.i32_val(p.sourceBytes), rebase32);
      records=b.select(valid,remaining,b.i32_val(0));
    }
  }
  int64_t recordPitch = p.pitch;
  if (p.cacheSwizzle && (!p.structured || completeColumns)) {
    auto shape = op.getDest().getType().getShape();
    recordPitch = cacheDistributedRecordPitch(
        op, p.pitch, shape[p.rowAxis], shape[p.columnAxis]);
  }
  Value resource=rw.createOrFold<ROCDL::MakeBufferRsrcOp>(loc,ptr_ty(ctx,8),base,
      // Cache-bank swizzling is separate from descriptor address swizzling.
      // RAW keeps IDXEN disabled and byte bounds; the nonzero pitch supplies
      // cache distribution only. Never set the address-swizzle bit (bit15).
      b.int_val(16,p.cacheSwizzle?(uint64_t(recordPitch*2)|0x4000):0),
      b.zext(i64_ty, records), b.i32_val(0x20000));
  auto fn=getOrInsertHCUIntrinsic(op->getParentOfType<ModuleOp>(),loc,rw,
      p.structured?"llvm.amdgcn.struct.ptr.buffer.load.lds":"llvm.amdgcn.raw.ptr.buffer.load.lds",
      LLVM::LLVMVoidType::get(ctx),p.structured?
      ArrayRef<Type>{ptr_ty(ctx,8),ptr_ty(ctx,3),i32_ty,i32_ty,i32_ty,i32_ty,i32_ty,i32_ty}:
      ArrayRef<Type>{ptr_ty(ctx,8),ptr_ty(ctx,3),i32_ty,i32_ty,i32_ty,i32_ty,i32_ty});
  // The destination proof covers a dense, allocation-backed tile, including
  // zero-filled or disabled source vectors. Preserve that storage extent as a
  // standard LLVM fact; it says nothing about initialization or non-aliasing.
  // p.tileSpanBytes describes the strided global source, not this LDS extent.
  // Consume these extents before O3 under the existing wide-copy policy.
  // Other paths retain their original scopes: after O3 a canonicalized base
  // dereferenceability bound can join adjacent, independently scored slots.
  auto policy=op->getParentOfType<ModuleOp>()->getAttrOfType<IntegerAttr>(
      "ttg.hcu.prefer_wide_async_copy");
  if (policy && policy.getInt()==1) {
    Value tileBase=b.gep(ptr_ty(ctx,3),elem,smem.getBase(),affine);
    LLVM::AssumeOp::create(rw,loc,b.i1_val(true),"dereferenceable",
        ValueRange{tileBase,b.i64_val(uint64_t(2)*p.copies*p.waves*512)});
  }
  for (unsigned copy=0;copy<p.copies;++copy) {
    Value pw=b.add(wave,b.i32_val(copy*p.waves));
    Value ordinal=b.add(b.shl(pw,b.i32_val(9)),b.shl(lane,b.i32_val(3)));
    Value coords[2]={b.i32_val(0),b.i32_val(0)};
    for (unsigned bit=3;bit<p.bases.size();++bit) {
      Value digit=b.and_(b.lshr(ordinal,b.i32_val(bit)),b.i32_val(1));
      for (unsigned axis=0;axis<2;++axis)
        if (p.bases[bit][axis]) coords[axis]=b.xor_(coords[axis],b.mul(digit,b.i32_val(p.bases[bit][axis])));
    }
    Value physicalOffset=b.add(affine,b.shl(pw,b.i32_val(9)));
    Value dst=b.gep(ptr_ty(ctx,3),elem,smem.getBase(),physicalOffset);
    LLVM::CallOp call;
    if (p.structured) {
      Value row=b.add(coords[p.rowAxis],b.sub(rowOrigin,rowBase));
      Value col=coords[p.columnAxis];
      if (!completeColumns) {
        Value rowOffset = coords[p.rowAxis];
        Value globalRow =b.add(rowOrigin, rowOffset);
        Value globalCol = b.add(colOrigin, col);
        // MatrixCopyGeometry bounds the local coordinates below 2^20 and
        // the aligned column extent to [8, INT32_MAX]. The mathematical sum
        // of an i32 origin and a local coordinate is in [-2^31, 2^32).
        // Negative sums map above INT32_MAX in the unsigned comparison,
        // and positive sums are preserved even across the signed boundary.
        Value validCol = b.icmp_ule(globalCol,b.i32_val(p.columns - 8));
        Value outside = b.i32_val(0x80000000u);
        if (p.rows <= std::numeric_limits<int32_t>::max()) {
          Value validRow = b.icmp_ult(globalRow, b.i32_val(p.rows));
          // Any row index with bit 31 set is outside this descriptor.
          // Separate the column rejection bit from the changing row index
          // so an invariant column origin can share it across copies.
          Value columnReject = b.select(validCol, b.i32_val(0), outside);
          row = b.or_(b.select(validRow, globalRow, outside), columnReject);
        } else {
          Value wideRow =b.add(b.sext(i64_ty,rowOrigin),b.zext(i64_ty, rowOffset));
        Value validRow=b.and_(b.icmp_sge(wideRow,b.i64_val(0)),b.icmp_ult(wideRow,b.i64_val(p.rows)));
        row=b.select(b.and_(validCol,validRow),globalRow, outside);
        }
        col = globalCol;
      }
      Value offset = b.add(b.shl(col,b.i32_val(1)),b.i32_val(16));
      if (recordPitch != p.pitch)
        offset = b.add(offset, b.mul(row,b.i32_val(2*(p.pitch-recordPitch))));
      call=LLVM::createLLVMCallOp(rw,loc,fn,{resource,dst,b.i32_val(16),row,
          offset,b.i32_val(0),b.i32_val(0),b.i32_val(0)});
    } else if (p.affinePartialColumns) {
      // The aligned vector starts are below 2^20. Adding them to a signed
      // i32 origin neither wraps a nonnegative sum in u32 nor maps a negative
      // sum into the legal column range. Test all eight elements at once.
      Value column = b.add(colOrigin, coords[p.columnAxis]);
      Value valid = b.icmp_ule(column, b.i32_val(p.columns - 8));
      Value local = b.add(b.mul(coords[p.rowAxis], b.i32_val(p.pitch)),
                          b.add(coords[p.columnAxis], columnDelta));
      Value offset = b.select(valid, b.shl(local, b.i32_val(1)),
                              b.i32_val(0x80000000u));
      call = LLVM::createLLVMCallOp(rw, loc, fn,
                                   {resource, dst, b.i32_val(16), offset,b.i32_val(0),b.i32_val(0),b.i32_val(0)});
    } else {
      Value local=b.shl(b.add(b.mul(b.zext(i64_ty,coords[p.rowAxis]),b.i64_val(p.pitch)),
                             b.zext(i64_ty,coords[p.columnAxis])),b.i64_val(1));
      bool boxed=p.sourceBytes >= (uint64_t(1) << 31);
      Value relative=(boxed || uniformRectangle)?local:Value(b.add(local,b.sub(rawUniform,rebase)));
      Value offset=b.trunc(i32_ty,relative);
      if (!completeColumns || (boxed && !rowBoundsHandled)) {
        Value valid=b.i1_val(true);
        if (!completeColumns) {
          Value gc=b.add(b.sext(i64_ty,colOrigin),b.zext(i64_ty,coords[p.columnAxis]));
          valid=b.and_(b.icmp_sge(gc,b.i64_val(0)),b.icmp_ule(b.add(gc,b.i64_val(8)),b.i64_val(p.columns)));
        }
        if (boxed && !rowBoundsHandled) {
          Value gr=b.add(b.sext(i64_ty,rowOrigin),b.zext(i64_ty,coords[p.rowAxis]));
          valid=b.and_(valid,b.and_(b.icmp_sge(gr,b.i64_val(0)),b.icmp_slt(gr,b.i64_val(p.rows))));
        }
        offset=b.select(valid,offset,b.i32_val(0x80000000u));
      }
      call=LLVM::createLLVMCallOp(rw,loc,fn,{resource,dst,b.i32_val(16),offset,b.i32_val(0),b.i32_val(0),b.i32_val(0)});
    }
    if (auto alias=dyn_cast<LLVM::AliasAnalysisOpInterface>(call.getOperation())) addAsyncCopyAliasScope(alias);
  }
  rw.replaceOp(op,b.i32_val(0));return success();
}
