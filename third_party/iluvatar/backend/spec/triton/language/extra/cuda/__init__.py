def language_extra_cuda_modify_all(all_array):
    return [a for a in all_array if a not in {"globaltimer", "smid"}]
