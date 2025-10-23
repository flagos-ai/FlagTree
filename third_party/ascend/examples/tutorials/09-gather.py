"""
Gather
===============
This is an example only for npu.
"""

import torch
import torch_npu
import triton
import triton.runtime.driver as driver
import triton.language as tl


# get device properties of npu
def get_npu_properties():
    device = torch.npu.current_device()
    return driver.active.utils.get_device_properties(device)


# a torch-version gather benchmark
def torch_gather(embeddings, idxes, default_value=0.0):
    # make the result tensor
    res = torch.empty((idxes.shape[0], embeddings.shape[-1]), dtype=embeddings.dtype, device=embeddings.device)

    # scatter embeddings
    res[idxes >= 0] = embeddings[idxes[idxes >= 0]]
    # set default values
    res[idxes < 0] = default_value

    return res


# triton-version gather's kernel
@triton.jit
def gather_kernel(embeddings_ptr, idxes_ptr, res_ptr, rows, cols, DEFAULT_VALUE: tl.constexpr, BIG_CORE_NUM: tl.constexpr, BIG_ROW_BLOCK_SIZE: tl.constexpr, COL_BLOCK_SIZE: tl.constexpr, COL_BLOCK_SIZE_SUB: tl.constexpr):
    SMALL_ROW_BLOCK_SIZE = BIG_ROW_BLOCK_SIZE - 1

    embedding_dtype = embeddings_ptr.type.element_ty
    default_value = tl.cast(DEFAULT_VALUE, dtype=embedding_dtype)
    default_embedding = tl.full((COL_BLOCK_SIZE_SUB, ), default_value, dtype=embedding_dtype)

    core_idx = tl.program_id(0)
    # compute the the size and start index of block
    row_block_size = BIG_ROW_BLOCK_SIZE if (core_idx < BIG_CORE_NUM) else SMALL_ROW_BLOCK_SIZE
    row_start_idx = (core_idx * BIG_ROW_BLOCK_SIZE) if (core_idx < BIG_CORE_NUM) else (BIG_CORE_NUM * BIG_ROW_BLOCK_SIZE + (core_idx - BIG_CORE_NUM) * SMALL_ROW_BLOCK_SIZE)

    # process blocks witn shape (row_block_size, COL_BLOCK_SIZE_SUB) one by one
    for col_idx in tl.range(0, COL_BLOCK_SIZE, COL_BLOCK_SIZE_SUB):
        emb_col_offsets = col_idx + tl.arange(0, COL_BLOCK_SIZE_SUB)
        emb_col_mask = emb_col_offsets < cols

        for row_idx in tl.range(row_start_idx, min(row_start_idx + row_block_size, rows)):
            idx_val = tl.load(idxes_ptr + row_idx)

            write_row_offset = row_idx * cols
            write_emb_mask = emb_col_mask

            if idx_val >= 0:
                read_row_offset = idx_val * cols
                read_emb_mask = emb_col_mask
                # read embedding
                embedding = tl.load(embeddings_ptr + read_row_offset + emb_col_offsets, mask=read_emb_mask)
                tl.store(res_ptr + write_row_offset + emb_col_offsets, embedding, write_emb_mask)
            else:
                # set default values
                tl.store(res_ptr + write_row_offset + emb_col_offsets, default_embedding, write_emb_mask)


# triton-version gather's host
def triton_gather(embeddings: torch.Tensor, indices: torch.Tensor, default_value=0.0):
    # constant settings for npu
    USE_SIZE = 96 * 1024
    CORE_NUM = get_npu_properties()["num_vectorcore"]

    n_rows = indices.shape[0]
    n_cols = embeddings.shape[1]
    # make the result tensor
    output = torch.empty(n_rows, n_cols, dtype=embeddings.dtype, device=embeddings.device)

    # when writing an npu kernel using triton,
    # you should note that the difference between BLOCK_SIZE and BLOCK_SIZE_SUB
    # BLOCK_SIZE specifies the size of data that are processed in one program
    col_size_aligned = triton.cdiv(embeddings.shape[-1] * embeddings.element_size(), 32) * 32 // embeddings.element_size()
    # the data are scattered to multiple programs, which can not be even
    # some process more data, some process less
    big_row_block_size = triton.cdiv(n_rows, CORE_NUM)
    big_core_num = CORE_NUM - ((big_row_block_size * CORE_NUM) - n_rows)
    col_block_size = col_size_aligned

    # BLOCK_SIZE_SUB specifies the size of data that are processed in one loop of a program
    max_col_block_size_sub = USE_SIZE // embeddings.element_size() // 2
    col_block_size_sub = min(col_size_aligned, max_col_block_size_sub)

    grid = (min(n_rows, CORE_NUM), triton.cdiv(n_cols, col_block_size))
    # launch the kernel
    gather_kernel[grid](embeddings, indices, output, n_rows, n_cols, default_value, BIG_CORE_NUM=big_core_num, BIG_ROW_BLOCK_SIZE=big_row_block_size, COL_BLOCK_SIZE=col_block_size, COL_BLOCK_SIZE_SUB=col_block_size_sub)

    return output


if __name__ == "__main__":
    for n_rows in (500, 1000):
        for n_cols in (16, 17, 31, 32, 63, 64, 128, 256, 819, 512, 1024, 8192, 1001, 2003, 17000):
            for index_num in (19, 123, 4321, 54321, 100, 200, 819, 500, 700, 1000):
                print(n_rows, n_cols, index_num, flush=True)

                indices = torch.randint(0, n_rows, (index_num, ), dtype=torch.int32).npu()
                embeddings = torch.randn(n_rows, n_cols, dtype=torch.float).npu()

                expect = torch_gather(embeddings, indices).cpu()
                actual = triton_gather(embeddings, indices).cpu()
                torch.npu.synchronize()
                mask = ~(expect == actual)

                error_count = mask.sum().item()
                total_count = mask.numel()
                print("error rate:", error_count / total_count, flush=True)

                print("error detail:")
                print("===========", flush=True)
                print(expect[mask], flush=True)
                print("===========", flush=True)
                print(actual[mask], flush=True)
                print("===========", flush=True)
                print(flush=True)
