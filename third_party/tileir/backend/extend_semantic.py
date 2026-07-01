from triton.language import core as tl
from triton.language.semantic import TritonSemantic


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

    def create_mem_token(self):
        return self.builder.create_mem_token()

    def join_mem_tokens(self, token_a, token_b):
        return self.builder.create_join_mem_tokens(token_a.handle, token_b.handle)
