def ext_str_to_load_cache_modifier(cache_modifier):
    from triton._C.libtriton import ir
    if cache_modifier == ".cv":
        return ir.CACHE_MODIFIER.CV
    return None


def is_atomic_support_bf16():
    return True


def atomin_add_int64(sca_ty, builder, val, ptr, mask, sem, scope):
    from triton.language import core as tl
    from triton._C.libtriton import ir
    from triton.language.semantic import full, and_, cast, lshr, bitcast, add, _bool_like, where, shl, or_
    if sca_ty.is_int64():
        # Split it into low and high 32 bits and cast them to int32
        low_mask = full([], 0xFFFFFFFF, tl.int32, builder)
        val_low = and_(val, low_mask, builder)
        val_low_int32 = cast(val_low, tl.int32, builder)

        _32 = full([], 32, sca_ty, builder)
        val_shr = lshr(val, _32, builder)
        val_high = and_(val_shr, low_mask, builder)
        val_high_int32 = cast(val_high, tl.int32, builder)

        # Split the pointer into two addresses for low and high parts
        addr_low = bitcast(ptr, tl.pointer_type(tl.int32, 1), builder)
        one_int32 = full(addr_low.shape, 1, tl.int32, builder)
        addr_high = builder.create_addptr(addr_low.handle, one_int32.handle)

        # Perform atomic addition for the low 32 bits
        if ptr.type.is_block():
            sum_ty = tl.block_type(tl.int32, ptr.type.get_block_shapes())
        else:
            sum_ty = tl.int32
        old_value_low = tl.tensor(
            builder.create_atomic_rmw(ir.ATOMIC_OP.ADD, addr_low.handle, val_low_int32.handle, mask.handle, sem, scope),
            sum_ty)

        # Check for unsigned overflow in the low part and perform atomic addition for the high 32 bits
        sum_low = add(old_value_low, val_low_int32, builder)
        overflow = tl.tensor(builder.create_icmpULT(sum_low.handle, val_low_int32.handle),
                             _bool_like(sum_low))  # treat as unsigned
        _1 = full([], 1, tl.int32, builder)
        _0 = full([], 0, tl.int32, builder)
        val_high_adjusted = add(val_high_int32, where(overflow, _1, _0, builder), builder)
        old_value_high = tl.tensor(
            builder.create_atomic_rmw(ir.ATOMIC_OP.ADD, addr_high, val_high_adjusted.handle, mask.handle, sem, scope),
            sum_ty)

        # Combine the high and low results back into a 64-bit integer, treat low value as unisigned.
        old_value_low_int64 = tl.tensor(builder.create_int_cast(old_value_low.handle, tl.int64.to_ir(builder), False),
                                        tl.int64)
        old_value_high_int64 = cast(old_value_high, tl.int64, builder)
        old_value_high_shifted = shl(old_value_high_int64, _32, builder)
        old_value = or_(old_value_high_shifted, old_value_low_int64, builder)
        return old_value
    else:
        return None
