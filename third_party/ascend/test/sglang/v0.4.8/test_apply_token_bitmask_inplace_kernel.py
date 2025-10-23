import sys
import pytest
import triton
import torch
import triton.language as tl

sys.path.append("..")
import test_common


# source: python\sglang\srt\constrained\triton_ops\bitmask_ops.py
@triton.jit
def apply_token_bitmask_inplace_kernel(
    logits_ptr,
    bitmask_ptr,
    indices_ptr,
    num_rows,
    vocab_size,
    logits_strides,
    bitmask_strides,
    NUM_SMS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Apply a bitmask to logits in-place using Triton. The bitmask is a 01 bitwise compressed tensor,
    where 0 means the token is masked and 1 means the token is not masked. After applying the bitmask,
    the masked logits will be set to -inf.

    Parameters
    ----------
    logits_ptr : tl.tensor
        Pointer to the logits tensor to apply the bitmask to.

    bitmask_ptr : tl.tensor
        Pointer to the bitmask tensor to apply.

    indices_ptr : Optional[tl.tensor]
        Optional pointer to indices tensor specifying which rows to apply the mask to.

    num_rows : int
        Number of rows to process. If indices_ptr is provided, this is the number of unique indices.

    vocab_size : int
        Size of the vocabulary dimension. If the logits does not have a vocab padding, this is the
        same as the logits's second dimension. Otherwise, this is the actual size of the vocabulary.

    logits_strides : int
        Stride between rows in the logits tensor.

    bitmask_strides : int
        Stride between rows in the bitmask tensor.

    NUM_SMS : int
        Number of streaming multiprocessors to use.

    BLOCK_SIZE : int
        Size of processing blocks.
    """

    pid = tl.program_id(0)
    num_blocks = tl.cdiv(vocab_size, BLOCK_SIZE)
    for work_id in tl.range(pid, num_rows * num_blocks, NUM_SMS):
        row_id = work_id // num_blocks
        block_offset = (work_id % num_blocks) * BLOCK_SIZE
        batch_id = row_id if indices_ptr is None else tl.load(indices_ptr + row_id)
        offsets = block_offset + tl.arange(0, BLOCK_SIZE)
        bitmask_offsets = block_offset // 32 + tl.arange(0, BLOCK_SIZE // 32)
        vocab_mask = offsets < vocab_size
        packed_bitmask_mask = bitmask_offsets < bitmask_strides
        packed_bitmask = tl.load(
            bitmask_ptr + batch_id * bitmask_strides + bitmask_offsets,
            packed_bitmask_mask,
        )
        bitmask = ((packed_bitmask[:, None] >> (tl.arange(0, 32)[None, :])) & 1) == 0
        bitmask = bitmask.reshape(BLOCK_SIZE)

        tl.store(
            logits_ptr + batch_id * logits_strides + offsets,
            -float("inf"),
            vocab_mask & bitmask,
        )


def test_apply_token_bitmask_inplace_kernel(ptfile_path):
    try:
        data = torch.load(ptfile_path, map_location=torch.device('cpu'), weights_only=False)
    except Exception as e:
        pytest.fail(f"load file {ptfile_path} failed: {str(e)}")

    # ptfile format:
    # [input_data] (dict):
    #     key : value
    # [gpu_output] (dict):
    #     key : value
    # [grid] :
    #     (1,)
    input_data = test_common.convert_tensor_with_device_type(data["input_data"], device_type='npu')

    apply_token_bitmask_inplace_kernel[data["grid"]](**input_data)

    # compare the results of GPU and NPU.
    try:
        test_common.compare_data_precision(data["gpu_output"], input_data, device_type='cpu')
    except ValueError as e:
        pytest.fail(f"The testcase failed")