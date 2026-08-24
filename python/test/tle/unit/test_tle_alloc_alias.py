# flagtree tle
"""TLE alloc alias unit tests — mock-based, no GPU required."""

import pytest
import triton.language as tl
import triton.experimental.tle.language as tle


class TestAllocAlias:

    class _FakeTensor:

        def __init__(self, handle, ty):
            self.handle = handle
            self.type = ty

    class _FakeBuilder:

        def __init__(self):
            self.memdesc_type_args = None
            self.memdesc_alias_args = None
            self.swizzled_encoding_args = None
            self.auto_shared_layout_handles = []

        def get_half_ty(self):
            return "fp16"

        def make_swizzled_shared_encoding_attr(self, vector_size, per_phase, max_phase, order, ctas_per_cga,
                                               cta_split_num, cta_order):
            self.swizzled_encoding_args = (
                vector_size,
                per_phase,
                max_phase,
                list(order),
                list(ctas_per_cga),
                list(cta_split_num),
                list(cta_order),
            )
            return "fake_layout"

        def get_memdesc_type(self, shape, element_ty, layout, space, alloc_shape=None):
            self.memdesc_type_args = (list(shape), element_ty, layout, space, alloc_shape)
            return ("memdesc", tuple(shape), element_ty, layout, space,
                    None if alloc_shape is None else tuple(alloc_shape))

        def create_local_alloc(self, *args):
            return "alloc_handle"

        def create_memdesc_alias(self, result_ty, src, offset_bytes):
            self.memdesc_alias_args = (result_ty, src, offset_bytes)
            return "alias_handle"

        def mark_musa_tle_auto_shared_layout(self, handle):
            if handle != "alloc_handle":
                raise ValueError("auto shared layout marker requires a local allocation")
            self.auto_shared_layout_handles.append(handle)

    class _FakeSemantic:

        def __init__(self):
            self.builder = TestAllocAlias._FakeBuilder()

        def to_tensor(self, value):
            if isinstance(value, TestAllocAlias._FakeTensor):
                return value
            if isinstance(value, bool):
                return TestAllocAlias._FakeTensor(f"pred_{value}", tl.int1)
            if isinstance(value, int):
                return TestAllocAlias._FakeTensor(f"stage_{value}", tl.int32)
            raise TypeError(f"unsupported fake tensor input: {value!r}")

    def _make_buffer(self, shape):
        semantic = self._FakeSemantic()
        layout = tle.gpu.swizzled_shared_layout.make_default(len(shape))
        return (
            tle.gpu.buffered_tensor("base", tl.float16, shape, tle.gpu.smem, layout, semantic),
            semantic,
        )

    def test_alloc_alias_creates_typed_memdesc_alias_view(self):
        """alloc(alias=...) returns a typed view without creating a new allocation."""
        buffer, semantic = self._make_buffer([4, 16, 32])
        alias = tle.gpu.alloc(
            (2, 16, 16),
            tl.float16,
            layout=buffer.type.layout,
            alias=buffer,
            alias_offset_bytes=64,
            _semantic=semantic,
        )

        assert isinstance(alias, tle.gpu.buffered_tensor)
        assert alias.handle == "alias_handle"
        assert alias.shape == [2, 16, 16]
        assert alias.dtype == tl.float16
        assert alias.type.storage is tle.gpu.smem
        assert semantic.builder.memdesc_alias_args == (
            ("memdesc", (2, 16, 16), "fp16", "fake_layout", "smem", None),
            "base",
            64,
        )

    def test_alloc_alias_skips_mthreads_auto_layout_marker(self, monkeypatch):
        """The mthreads auto-layout marker only accepts local allocations."""
        from triton.experimental.tle.language.gpu import core as gpu_core

        monkeypatch.setattr(gpu_core.mthreads_common, "enabled", lambda: True)
        buffer, semantic = self._make_buffer([4, 16, 32])

        alias = tle.gpu.alloc(
            (2, 16, 16),
            tl.float16,
            alias=buffer,
            _semantic=semantic,
        )

        assert alias.handle == "alias_handle"
        assert semantic.builder.auto_shared_layout_handles == []

    def test_alloc_alias_rejects_init_value(self):
        buffer, semantic = self._make_buffer([4, 16, 32])
        init = self._FakeTensor("init", tl.float16)

        with pytest.raises(ValueError, match="alias mode cannot be combined"):
            tle.gpu.alloc(
                (2, 16, 16),
                tl.float16,
                layout=buffer.type.layout,
                init_value=init,
                alias=buffer,
                _semantic=semantic,
            )

    def test_alloc_alias_rejects_non_smem_buffer(self):
        """alias source must be a shared-memory buffered_tensor."""
        semantic = self._FakeSemantic()
        fake_buffer = self._FakeTensor("tmem_buf", tl.float16)

        with pytest.raises(ValueError, match="tle.buffered_tensor"):
            tle.gpu.alloc(
                (2, 16, 16),
                tl.float16,
                alias=fake_buffer,
                _semantic=semantic,
            )

    def test_alloc_alias_invalid_offset_type(self):
        """Bytes are a compile-time argument."""
        buffer, semantic = self._make_buffer([4, 16, 32])

        with pytest.raises(ValueError, match="compile-time integer"):
            tle.gpu.alloc(
                (2, 16, 16),
                tl.float16,
                layout=buffer.type.layout,
                alias=buffer,
                alias_offset_bytes=b"not_an_int",
                _semantic=semantic,
            )
