from triton.language import core as tl
from triton.language.semantic import TritonSemantic
from triton.backends.tileir.extend_core import tileir_tensor_descriptor


def _str_to_tileir_semantic(semantic_option):
    semantic_option = tl._unwrap_if_constexpr(semantic_option)
    semantic_list = ["weak", "relaxed", "acquire", "release", "acq_rel"]
    if semantic_option not in semantic_list:
        raise ValueError(f"semantic must be in {semantic_list}, but got {semantic_option}")
    return semantic_list.index(semantic_option)


def _str_to_tileir_scope(scope_option):
    scope_option = tl._unwrap_if_constexpr(scope_option)
    scope_list = ["tl_blk", "device", "sys"]
    if scope_option in scope_list:
        return scope_list.index(scope_option)
    return -1


def _check_memory_ordering(semantics, scope):
    if semantics > 0 and scope < 0:
        raise ValueError("scope must be tl_blk, device, or sys when semantics is not weak")
    if semantics == 0 and scope > 0:
        raise ValueError("scope should not be set when semantics is weak")


def _check_load_view_ordering(semantics):
    if semantics not in (0, 1, 2):
        raise ValueError("tl.ext.load_view_tko semantics must be one of ['weak', 'relaxed', 'acquire']")


def _check_store_view_ordering(semantics):
    if semantics not in (0, 1, 3):
        raise ValueError("tl.ext.store_view_tko semantics must be one of ['weak', 'relaxed', 'release']")


def _str_to_ele_ty(dtype):
    return {
        "f16": tl.float16,
        "f32": tl.float32,
        "f64": tl.float64,
        "i1": tl.int1,
        "i8": tl.int8,
        "i16": tl.int16,
        "i32": tl.int32,
        "i64": tl.int64,
        "ui1": tl.int1,
        "ui8": tl.uint8,
        "ui16": tl.uint16,
        "ui32": tl.uint32,
        "ui64": tl.uint64,
        "bf16": tl.bfloat16,
        "f8E5M2": tl.float8e5,
        "f8E4M3FN": tl.float8e4nv,
    }[str(dtype)]


def _convert_view_coords(coords, semantic):

    def convert_elem(elem):
        elem = tl._unwrap_if_constexpr(elem)
        if isinstance(elem, int):
            assert -(2**31) <= elem < 2**31, f"view ops support 32 bit indices, got {elem}"
            return semantic._convert_to_ir_values([elem], require_i64=False)[0]
        if isinstance(elem, tl.tensor):
            assert elem.dtype.is_int(), "expected an integer tile type for view indices"
            return elem.handle
        raise TypeError(f"unsupported element type for view indices: {type(elem)}")

    if hasattr(coords, "__iter__"):
        return [convert_elem(elem) for elem in coords]
    return [convert_elem(coords)]


def _tileir_capability(semantic):
    arch = getattr(getattr(semantic.builder, "options", None), "arch", None)
    if isinstance(arch, str):
        arch = arch.removeprefix("sm").removeprefix("_")
    return int(arch or 0)


class CudaTileSemantic(TritonSemantic):

    def make_tensor_descriptor(self, base, shape, strides, block_shape, padding_option="zero"):
        desc = super().make_tensor_descriptor(base, shape, strides, block_shape, padding_option)
        return tileir_tensor_descriptor(desc.handle, desc.shape, desc.strides, desc.block_type, base)

    def make_tensor_view(self, res_type, base, shapes, strides):
        return tl.tensor(
            self.builder.create_make_tensor_view(
                res_type.to_ir(self.builder),
                base.handle,
                self._convert_to_ir_values(shapes, require_i64=False),
                self._convert_to_ir_values(strides, require_i64=False),
            ),
            res_type,
        )

    def make_partition_view(self, src_tensor_view, tile_shape, tile_dim_map, res_type, pad_val):
        ret_op, ir_type = self.builder.create_make_partition_view(
            src_tensor_view.handle,
            tile_shape,
            tile_dim_map,
            pad_val,
        )
        res_type.ir_type = ir_type
        return tl.tensor(ret_op, res_type)

    def dim(self, src, index):
        return tl.tensor(self.builder.create_dim(src.handle, index), tl.int32)

    def load_view_tko(self, src, coords, mem_token, semantics, scope, token_ty, has_result_token, cost):
        coords = _convert_view_coords(coords, self)
        semantics = _str_to_tileir_semantic(semantics)
        scope = _str_to_tileir_scope(scope)
        _check_load_view_ordering(semantics)
        _check_memory_ordering(semantics, scope)
        data, res_token, shape, dtype = self.builder.create_load_view_tko(
            src.handle,
            coords,
            mem_token.handle if mem_token else None,
            semantics,
            scope,
            has_result_token,
            cost,
            _tileir_capability(self),
        )
        result = tl.tensor(data, tl.block_type(_str_to_ele_ty(dtype), shape))
        if has_result_token:
            return result, tl.tensor(res_token, token_ty)
        return result

    def store_view_tko(self, dst, value, coords, mem_token, semantics, scope, token_ty, has_result_token, cost):
        coords = _convert_view_coords(coords, self)
        semantics = _str_to_tileir_semantic(semantics)
        scope = _str_to_tileir_scope(scope)
        _check_store_view_ordering(semantics)
        _check_memory_ordering(semantics, scope)
        res_token = self.builder.create_store_view_tko(
            dst.handle,
            value.handle,
            coords,
            mem_token.handle if mem_token else None,
            semantics,
            scope,
            has_result_token,
            cost,
            _tileir_capability(self),
        )
        if has_result_token:
            return tl.tensor(res_token, token_ty)
        return tl.tensor(res_token, tl.void)

    def load_store_typechecking_impl(self, ptr, value, mask, *, is_store):
        """Check and convert pointer inputs; value is the store value or load's other."""
        if is_store and (ptr.type.is_const() or ptr.type.scalar.is_const()):
            raise ValueError("Cannot store to a constant pointer")
        if not ptr.type.scalar.is_ptr() or ptr.type.scalar.element_ty.is_block():
            op = "store_tko" if is_store else "load_tko"
            raise ValueError(f"{op} requires a scalar pointer or tensor of pointers, not a block pointer")
        if not is_store and value is not None and mask is None:
            raise ValueError("`other` cannot be provided without `mask`")
        if not ptr.type.is_block():
            if mask is not None and mask.type.is_block():
                raise ValueError("Mask argument cannot be block type if pointer argument is not a block")
            if value is not None and value.type.is_block():
                operand = "Value" if is_store else "Other"
                raise ValueError(f"{operand} argument cannot be block type if pointer argument is not a block")
        else:
            if mask is not None:
                mask = self.broadcast_impl_shape(mask, ptr.type.get_block_shapes())
            if value is not None:
                value = self.broadcast_impl_shape(value, ptr.type.get_block_shapes())

        ptr_ty = ptr.type.scalar
        elt_ty = ptr_ty.element_ty
        # Boolean memory uses bytes, as in Triton's ordinary pointer loads/stores.
        if elt_ty == tl.int1:
            elt_ty = tl.int8
            ptr = self.cast(ptr, tl.pointer_type(elt_ty, ptr_ty.address_space))
        if value is not None:
            value = self.cast(value, elt_ty)
        if mask is not None and not mask.type.scalar.is_bool():
            raise ValueError("Mask must have boolean scalar type")
        return ptr, value, mask

    def load_tko(self, ptr, mask, other, mem_token, semantics, scope, token_ty, has_result_token, cost):
        semantics = _str_to_tileir_semantic(semantics)
        scope = tl._unwrap_if_constexpr(scope)
        if scope not in ("", "tl_blk", "device", "sys") or (semantics == 0 and scope != ""):
            raise ValueError("scope must be empty for weak ordering, or tl_blk/device/sys otherwise")
        scope = _str_to_tileir_scope(scope)
        _check_memory_ordering(semantics, scope)
        if semantics not in (0, 1, 2):
            raise ValueError("load_tko requires weak/relaxed/acquire ordering")

        original_ptr_ty = ptr.type.scalar
        ptr, other, mask = self.load_store_typechecking_impl(ptr, other, mask, is_store=False)
        elt_ty = ptr.type.scalar.element_ty
        result_ty = tl.block_type(elt_ty, ptr.type.get_block_shapes()) if ptr.type.is_block() else elt_ty

        data, token = self.builder.create_load_tko(ptr.handle, mask.handle if mask is not None else None,
                                                   other.handle if other is not None else None,
                                                   mem_token.handle if mem_token is not None else None, semantics,
                                                   scope, cost, _tileir_capability(self))
        result = tl.tensor(data, result_ty)
        if original_ptr_ty.element_ty == tl.int1:
            result = self.cast(result, tl.int1)
        return (result, tl.tensor(token, token_ty)) if has_result_token else result

    def store_tko(self, ptr, value, mask, mem_token, semantics, scope, token_ty, has_result_token, cost):
        semantics = _str_to_tileir_semantic(semantics)
        scope = tl._unwrap_if_constexpr(scope)
        if scope not in ("", "tl_blk", "device", "sys") or (semantics == 0 and scope != ""):
            raise ValueError("scope must be empty for weak ordering, or tl_blk/device/sys otherwise")
        scope = _str_to_tileir_scope(scope)
        _check_memory_ordering(semantics, scope)
        if semantics not in (0, 1, 3):
            raise ValueError("store_tko requires weak/relaxed/release ordering")

        ptr, value, mask = self.load_store_typechecking_impl(ptr, value, mask, is_store=True)
        token = self.builder.create_store_tko(ptr.handle, value.handle, mask.handle if mask is not None else None,
                                              mem_token.handle if mem_token is not None else None, semantics, scope,
                                              cost, _tileir_capability(self))
        return tl.tensor(token, token_ty if has_result_token else tl.void)

    def atomic_typechecking_impl(self, ptr, value):
        if not ptr.type.scalar.is_ptr() or ptr.type.scalar.element_ty.is_block():
            raise ValueError("TKO atomics require a scalar pointer or tensor of pointers, not a block pointer")
        if ptr.type.is_const() or ptr.type.scalar.is_const():
            raise ValueError("Cannot store to a constant pointer")
        value = self.to_tensor(value)
        if ptr.type.is_block():
            value = self.broadcast_impl_shape(value, ptr.type.get_block_shapes())
        elif value.type.is_block():
            raise ValueError("Value argument cannot be block type if pointer argument is not a block")
        return self.cast(value, ptr.type.scalar.element_ty)

    def atomic_cas_tko(self, ptr, cmp, val, mem_token, semantics, scope, token_ty, has_result_token):
        cmp = self.atomic_typechecking_impl(ptr, cmp)
        val = self.atomic_typechecking_impl(ptr, val)
        semantics = _str_to_tileir_semantic(semantics)
        scope = _str_to_tileir_scope(scope)
        _check_memory_ordering(semantics, scope)
        result, res_token = self.builder.create_atomic_cas_tko(
            ptr.handle,
            cmp.handle,
            val.handle,
            mem_token.handle if mem_token else None,
            semantics,
            scope,
        )
        result = tl.tensor(result, val.type)
        if has_result_token:
            return result, tl.tensor(res_token, token_ty)
        return result

    def atomic_xchg_tko(self, ptr, val, mem_token, semantics, scope, token_ty, has_result_token):
        val = self.atomic_typechecking_impl(ptr, val)
        semantics = _str_to_tileir_semantic(semantics)
        scope = _str_to_tileir_scope(scope)
        _check_memory_ordering(semantics, scope)
        result, res_token = self.builder.create_atomic_xchg_tko(
            ptr.handle,
            val.handle,
            mem_token.handle if mem_token else None,
            semantics,
            scope,
        )
        result = tl.tensor(result, val.type)
        if has_result_token:
            return result, tl.tensor(res_token, token_ty)
        return result

    def create_mem_token(self):
        return self.builder.create_mem_token()

    def join_mem_tokens(self, token_a, token_b):
        return self.builder.create_join_mem_tokens(token_a.handle, token_b.handle)
