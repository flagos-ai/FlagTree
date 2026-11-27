# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
import pytest

from triton.flagmega.ir.execution_mesh import KernelProgramKind
from triton.flagmega.errors import IRSchemaError
from triton.flagmega.ir.launch_recipe import TaskLaunchRecipe


def test_ordinary_grid_can_exceed_resident_capacity():
    recipe = TaskLaunchRecipe(131073, 128, KernelProgramKind.ORDINARY)
    assert recipe.resolve(sm_count=132) == (1025, 1, 1)


def test_persistent_grid_is_resolved_from_the_runtime_device():
    recipe = TaskLaunchRecipe(131073, 128, KernelProgramKind.PERSISTENT, ctas_per_sm=2)
    assert recipe.resolve(sm_count=84) == (168, 1, 1)
    assert recipe.resolve(sm_count=132) == (264, 1, 1)
    assert recipe.resolve(sm_count=132, max_ctas=7) == (7, 1, 1)


def test_same_buffer_abi_for_different_runtime_grids():
    recipe = TaskLaunchRecipe(4103, 128, 'persistent', max_ctas=7)
    assert recipe.task_count == 33
    for count in (1, 2, 7, 33):
        tasks = [task for cta in range(count) for task in range(cta, recipe.task_count, count)]
        assert sorted(tasks) == list(range(recipe.task_count))
        elements = [index for task in tasks for index in range(task * 128, min((task + 1) * 128, 4103))]
        assert sorted(elements) == list(range(4103))
    assert TaskLaunchRecipe.from_data(recipe.to_data()) == recipe


@pytest.mark.parametrize('args', [(0, 128), (10, 0), (True, 128), (10, 1.5)])
def test_recipe_rejects_invalid_work_or_tiles(args):
    with pytest.raises(IRSchemaError):
        TaskLaunchRecipe(*args, 'ordinary')
