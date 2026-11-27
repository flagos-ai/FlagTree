#include "llama.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <vector>

int main(int argc, char **argv) {
    if (argc != 7 && argc != 8 && argc != 9) return 2;
    const int steps = std::stoi(argv[3]), repeats = std::stoi(argv[4]);
    const int initial_token = std::stoi(argv[5]);
    const bool cpu = std::stoi(argv[6]) != 0;
    if (steps < 1 || repeats < 1) return 2;
    std::vector<llama_token> forced;
    if (argc == 8) {
        std::ifstream file(argv[7]);
        int token;
        while (file >> token) forced.push_back(token);
        if (int(forced.size()) < steps) throw std::runtime_error("short input token file");
    }
    llama_backend_init();
    auto mp = llama_model_default_params();
    mp.n_gpu_layers = cpu ? 0 : -1;
    mp.use_extra_bufts = false;
    auto *model = llama_model_load_from_file(argv[1], mp);
    if (!model) return 3;
    const int vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    int prompt_token_count = 0;
    if (argc == 9) {
        const std::string prompt(argv[8]);
        const auto *vocabulary = llama_model_get_vocab(model);
        const int count = -llama_tokenize(vocabulary, prompt.data(), prompt.size(), nullptr, 0, true, true);
        prompt_token_count = count;
        forced.resize(count);
        if (llama_tokenize(vocabulary, prompt.data(), prompt.size(), forced.data(), count, true, true) != count)
            throw std::runtime_error("tokenization failed");
        if (count > steps) throw std::runtime_error("prompt exceeds decode steps");
    }
    auto cp = llama_context_default_params();
    cp.n_ctx = std::max(256, steps);
    cp.n_batch = cp.n_ubatch = 1;
    cp.n_threads = cp.n_threads_batch = 8;
    cp.type_k = cp.type_v = GGML_TYPE_BF16;
    cp.flash_attn_type = cpu ? LLAMA_FLASH_ATTN_TYPE_DISABLED : LLAMA_FLASH_ATTN_TYPE_ENABLED;
    auto *ctx = llama_init_from_model(model, cp);
    if (!ctx) return 4;
    auto batch = llama_batch_init(1, 0, 1);
    batch.n_tokens = 1;
    batch.n_seq_id[0] = 1;
    batch.seq_id[0][0] = 0;
    batch.logits[0] = 1;
    auto decode = [&](int token, int position) {
        batch.token[0] = token;
        batch.pos[0] = position;
        if (llama_decode(ctx, batch) != 0) throw std::runtime_error("llama_decode failed");
        llama_synchronize(ctx);
        float *logits = llama_get_logits_ith(ctx, -1);
        if (!logits) throw std::runtime_error("missing logits");
        return logits;
    };
    auto reset = [&]() { llama_memory_clear(llama_get_memory(ctx), true); };
    decode(initial_token, 0);
    reset();
    std::ofstream logits_file(argv[2], std::ios::binary);
    if (!logits_file) return 5;
    std::vector<int> inputs, outputs;
    int token = initial_token;
    for (int step = 0; step < steps; ++step) {
        if (step < int(forced.size())) token = forced[step];
        inputs.push_back(token);
        auto *logits = decode(token, step);
        for (int i = 0; i < vocab; ++i) {
            if (!std::isfinite(logits[i])) throw std::runtime_error("non-finite logits");
        }
        logits_file.write(reinterpret_cast<const char *>(logits), vocab * sizeof(float));
        token = std::max_element(logits, logits + vocab) - logits;
        outputs.push_back(token);
    }
    logits_file.close();
    std::vector<double> latencies;
    for (int repeat = 0; repeat <= repeats; ++repeat) {
        reset();
        for (int step = 0; step < steps; ++step) {
            const auto start = std::chrono::steady_clock::now();
            auto *logits = decode(inputs[step], step);
            const auto stop = std::chrono::steady_clock::now();
            if (std::max_element(logits, logits + vocab) - logits != outputs[step])
                throw std::runtime_error("reset changed output tokens");
            if (repeat) latencies.push_back(std::chrono::duration<double, std::milli>(stop - start).count());
        }
    }
    printf("{\"vocab\":%d,\"steps\":%d,\"repeats\":%d,\"layers\":%d,\"prompt_token_count\":%d,\"input_tokens\":[", vocab, steps, repeats, llama_model_n_layer(model), prompt_token_count);
    for (int i = 0; i < steps; ++i) printf("%s%d", i ? "," : "", inputs[i]);
    printf("],\"tokens\":[");
    for (int i = 0; i < steps; ++i) printf("%s%d", i ? "," : "", outputs[i]);
    printf("],\"latency_ms\":[");
    for (size_t i = 0; i < latencies.size(); ++i) printf("%s%.6f", i ? "," : "", latencies[i]);
    printf("]}\n");
    llama_batch_free(batch);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
}
