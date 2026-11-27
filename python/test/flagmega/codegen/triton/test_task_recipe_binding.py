"""An ordinary row entry must carry its task ownership into the callee ABI."""

import pytest

from triton.flagmega.codegen.triton.kernel_call_renderers import _bind_explicit_task_recipe
from triton.flagmega.codegen.triton.kernel_programs import _task_owner_partition
from triton.flagmega.errors import CodegenError
from triton.flagmega.ir.launch_recipe import TaskLaunchRecipe


def row_call(rows):
    recipe = TaskLaunchRecipe(rows, 4, "ordinary").to_data()
    call = {side: [{"buffers": [{"abi": {"pool_scope_stride_bytes": 0}}]}]
            for side in ("inputs", "outputs")}
    _bind_explicit_task_recipe(call, {"parameters": {"task_mapping": recipe}})
    return call, {"grid": [(rows + 3) // 4, 1, 1], "launch_recipe": recipe}


@pytest.mark.parametrize("rows", (1024, 12288, 17408, 248320))
def test_row_entry_uses_callee_recipe_instead_of_cooperative_mesh(rows):
    # This is the ABI-to-entry path which previously rejected the full model
    # despite selection having already declared its ordinary row recipe.
    call, request = row_call(rows)
    assert _task_owner_partition(request, {"grid": [32, 1, 1]}, [call]) is None


@pytest.mark.parametrize("rows", (1024, 12288, 17408, 248320))
@pytest.mark.parametrize("side", ("inputs", "outputs"))
def test_independent_row_entry_rejects_private_producer_or_consumer_storage(rows, side):
    call, request = row_call(rows)
    call[side][0]["buffers"][0]["abi"]["pool_scope_stride_bytes"] = 256
    with pytest.raises(CodegenError):
        _task_owner_partition(request, {"grid": [32, 1, 1]}, [call])


def test_conflicting_encoder_and_selected_task_ownership_is_rejected():
    call, _ = row_call(12288)
    different = TaskLaunchRecipe(12288, 8, "ordinary").to_data()
    with pytest.raises(CodegenError):
        _bind_explicit_task_recipe(call, {"parameters": {"task_mapping": different}})


def test_legacy_derived_mapping_does_not_become_an_explicit_recipe():
    call = {}
    _bind_explicit_task_recipe(call, {"parameters": {"task_mapping": {"tile_size": 16}}})
    assert "task_launch_recipe" not in call
