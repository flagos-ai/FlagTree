import torch


def gems_assert_cosine_similarity(a, b, dtype, eps=1e-8):
    a_cpu = a.to("cpu")
    b_cpu = b.to(dtype)
    a_cpu = a_cpu.to(dtype=torch.float64)
    b_cpu = b_cpu.to(dtype=torch.float64)

    a_cpu = a_cpu.flatten()
    b_cpu = b_cpu.flatten()
    dim = 0

    is_nan_a = torch.isnan(a_cpu)
    is_nan_b = torch.isnan(b_cpu)
    both_nan = is_nan_a & is_nan_b

    is_inf_a = torch.isinf(a_cpu)
    is_inf_b = torch.isinf(b_cpu)
    both_inf = is_inf_a & is_inf_b & (torch.sign(a_cpu) == torch.sign(b_cpu))

    invalid_mask = both_nan | both_inf
    valid_mask = ~invalid_mask
    a_filtered = a_cpu[valid_mask]
    b_filtered = b_cpu[valid_mask]

    if len(a_filtered) == 0 and len(b_filtered) == 0:
        assert True
    elif len(a_filtered) == 0 and len(b_filtered) != 0:
        assert False, "The output of inf and nan results is misaligned"
    elif len(a_filtered) != 0 and len(b_filtered) == 0:
        assert False, "The output of inf and nan results is misaligned"
    else:
        dot_product = (a_filtered * b_filtered).sum(dim=dim)
        a_norm = a_filtered.norm(p=2, dim=dim)
        b_norm = b_filtered.norm(p=2, dim=dim)

        cosine_sim = dot_product / (a_norm * b_norm + eps)

        print(f"cosine_sim is: {cosine_sim*100:.4f}%")
        print(f"dot_product is: {dot_product}")
        print(f"X_norm*Y_norm is: {a_norm * b_norm + eps}")

        if torch.isnan(cosine_sim):
            print(f"cosine_sim is: {cosine_sim}")
            assert torch.isnan(cosine_sim)
        elif torch.isinf(cosine_sim):
            print(f"cosine_sim is: {cosine_sim}")
            assert torch.isinf(cosine_sim)
        elif cosine_sim < 0.9:
            print(f"cosine_sim is: {cosine_sim*100:.4f}%")
            assert False, f"cosine_sim < 90% ({cosine_sim*100:.4f}%)"
