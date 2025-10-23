"""
Gather sorted
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


# a torch-version gather_sorted benchmark
def torch_gather_sorted(embeddings, sorted_idxes, aux_idxes):
    # make the result tensor
    res = torch.empty((aux_idxes.shape[0], embeddings.shape[-1]), dtype=embeddings.dtype, device=embeddings.device)

    # scatter embeddings
    res[aux_idxes] = embeddings[sorted_idxes]

    return res


# triton-version gather_sorted's kernel
@triton.jit
def gather_sorted_kernel(embeddings_ptr, sorted_indices_ptr, aux_indices_ptr, res_ptr, rows, cols, DEFAULT_VALUE: tl.constexpr, BIG_CORE_NUM: tl.constexpr, BIG_ROW_BLOCK_SIZE: tl.constexpr, COL_BLOCK_SIZE: tl.constexpr, COL_BLOCK_SIZE_SUB: tl.constexpr):
    SMALL_ROW_BLOCK_SIZE = BIG_ROW_BLOCK_SIZE - 1

    emb_dtype = embeddings_ptr.type.element_ty
    default_value = tl.cast(DEFAULT_VALUE, dtype=emb_dtype)

    core_idx = tl.program_id(0)
    # compute the the size and start index of block
    row_block_size = BIG_ROW_BLOCK_SIZE if (core_idx < BIG_CORE_NUM) else SMALL_ROW_BLOCK_SIZE
    row_start_idx = (core_idx * BIG_ROW_BLOCK_SIZE) if (core_idx < BIG_CORE_NUM) else (BIG_CORE_NUM * BIG_ROW_BLOCK_SIZE + (core_idx - BIG_CORE_NUM) * SMALL_ROW_BLOCK_SIZE)

    # this version has 3-buffers, initilize for buffers
    row_block_size_0 = tl.cdiv(row_block_size, 3)
    remain_row_block_size = row_block_size - row_block_size_0
    row_block_size_1 = tl.cdiv(remain_row_block_size, 2)
    row_block_size_2 = remain_row_block_size - row_block_size_1
    
    row_start_idx_0 = row_start_idx
    row_start_idx_1 = row_start_idx + row_block_size_0
    row_start_idx_2 = row_start_idx + row_block_size_0 + row_block_size_1

    # process blocks witn shape (row_block_size, COL_BLOCK_SIZE_SUB) one by one
    for col_idx in tl.range(0, COL_BLOCK_SIZE, COL_BLOCK_SIZE_SUB):

        embedding_0 = tl.full((COL_BLOCK_SIZE_SUB, ), default_value, dtype=emb_dtype)
        embedding_1 = embedding_0 + 0
        embedding_2 = embedding_0 + 0

        emb_offsets = col_idx + tl.arange(0, COL_BLOCK_SIZE_SUB)
        emb_mask = emb_offsets < cols

        prev_embedding_idx_0 = tl.cast(-1, dtype=tl.int32)
        prev_embedding_idx_1 = tl.cast(-1, dtype=tl.int32)
        prev_embedding_idx_2 = tl.cast(-1, dtype=tl.int32)
        for row_idx in tl.range(row_start_idx_0, row_start_idx_1):
            # process the first buffer
            embedding_idx_0 = tl.load(sorted_indices_ptr + row_idx)
            res_idx_0 = tl.load(aux_indices_ptr + row_idx)

            if (embedding_idx_0 != 0) and (embedding_idx_0 != prev_embedding_idx_0):
                embedding_0 = tl.load(embeddings_ptr + embedding_idx_0 * cols + emb_offsets, emb_mask)
                tl.store(res_ptr + res_idx_0 * cols + emb_offsets, embedding_0, emb_mask)
            else:
                tl.store(res_ptr + res_idx_0 * cols + emb_offsets, embedding_0, emb_mask)

            prev_embedding_idx_0 = embedding_idx_0

            # process the second buffer
            if (row_idx + row_block_size_0) < (row_start_idx_1 + row_block_size_1):
                embedding_idx_1 = tl.load(sorted_indices_ptr + row_idx + row_block_size_0)
                res_idx_1 = tl.load(aux_indices_ptr + row_idx + row_block_size_0)

                if (embedding_idx_1 != 0) and (embedding_idx_1 != prev_embedding_idx_1):
                    embedding_1 = tl.load(embeddings_ptr + embedding_idx_1 * cols + emb_offsets, emb_mask)
                    tl.store(res_ptr + res_idx_1 * cols + emb_offsets, embedding_1, emb_mask)
                else:
                    tl.store(res_ptr + res_idx_1 * cols + emb_offsets, embedding_1, emb_mask)

                prev_embedding_idx_1 = embedding_idx_1

            # process the third buffer
            if (row_idx + row_block_size_0 + row_block_size_1) < (row_start_idx_2 + row_block_size_2):
                embedding_idx_2 = tl.load(sorted_indices_ptr + row_idx + row_block_size_0 + row_block_size_1)
                res_idx_2 = tl.load(aux_indices_ptr + row_idx + row_block_size_0 + row_block_size_1)

                if (embedding_idx_2 != 0) and (embedding_idx_2 != prev_embedding_idx_2):
                    embedding_2 = tl.load(embeddings_ptr + embedding_idx_2 * cols + emb_offsets, emb_mask)
                    tl.store(res_ptr + res_idx_2 * cols + emb_offsets, embedding_2, emb_mask)
                else:
                    tl.store(res_ptr + res_idx_2 * cols + emb_offsets, embedding_2, emb_mask)

                prev_embedding_idx_2 = embedding_idx_2


# triton-version gather_sorted's host
def triton_gather_sorted(embeddings: torch.Tensor, sorted_indices: torch.Tensor, aux_indices: torch.Tensor, default_value=1.0):
    # constant settings for npu
    ALIGNED = 32
    USE_SIZE = 96 * 1024
    CORE_NUM = get_npu_properties()["num_vectorcore"]

    n_rows = sorted_indices.shape[0]
    n_cols = embeddings.shape[1]
    # make the result tensor
    output = torch.empty(n_rows, n_cols, dtype=embeddings.dtype, device=embeddings.device)

    # when writing an npu kernel using triton,
    # you should note that the difference between BLOCK_SIZE and BLOCK_SIZE_SUB
    # BLOCK_SIZE specifies the size of data that are processed in one program
    col_size_aligned = triton.cdiv(embeddings.shape[-1] * embeddings.element_size(), ALIGNED) * ALIGNED // embeddings.element_size()
    # the data are scattered to multiple programs, which can not be even
    # some process more data, some process less
    big_row_block_size = triton.cdiv(n_rows, CORE_NUM)
    big_core_num = CORE_NUM - ((big_row_block_size * CORE_NUM) - n_rows)
    col_block_size = col_size_aligned
    # BLOCK_SIZE_SUB specifies the size of data that are processed in one loop of a program
    col_block_size_sub = min(1024, col_size_aligned)

    grid = (min(n_rows, CORE_NUM), triton.cdiv(n_cols, col_block_size))
    # launch the kernel
    gather_sorted_kernel[grid](embeddings, sorted_indices, aux_indices, output, n_rows, n_cols, default_value, BIG_CORE_NUM=big_core_num, BIG_ROW_BLOCK_SIZE=big_row_block_size, COL_BLOCK_SIZE=col_block_size, COL_BLOCK_SIZE_SUB=col_block_size_sub)

    return output


# genreate the desired inputs
def generate_inputs(index_shape, table_shape, dtype):
    sorted_indices = torch.randint(1, table_shape[0], index_shape, dtype=torch.int32).npu()
    mask = torch.rand_like(sorted_indices, dtype=torch.float).npu() < 0.2

    # make sorted_indices
    sorted_indices[mask] = 0
    sorted_indices, _ = torch.sort(sorted_indices)
    counts = torch.bincount(sorted_indices)
    _, _indices = torch.sort(counts[sorted_indices], descending=True, stable=True)
    sorted_indices = sorted_indices[_indices]

    # make aux_indicess
    aux_indices = torch.arange(0, index_shape[0], dtype=torch.int32).npu()
    _indices = torch.randperm(aux_indices.size(0))
    aux_indices = aux_indices[_indices]

    # make table, the first contains only 1.0
    table = torch.randn(table_shape, dtype=dtype).npu()
    table[0] = 1.0

    return table, sorted_indices, aux_indices


if __name__ == "__main__":
    for table_rows in (500, 1000):
        for table_cols in (16, 17, 31, 32, 63, 64, 128, 256, 819, 512, 1024, 8192, 1001, 2003, 17000):
            for index_num in (19, 123, 4321, 54321, 100, 200, 819, 500, 700, 1000):
                print(table_rows, table_cols, index_num, flush=True)

                table, sorted_indices, aux_indices = generate_inputs((index_num,), (table_rows, table_cols), torch.float)

                expect = torch_gather_sorted(table, sorted_indices, aux_indices).cpu()
                torch.npu.synchronize()
                actual = triton_gather_sorted(table, sorted_indices, aux_indices).cpu()
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
