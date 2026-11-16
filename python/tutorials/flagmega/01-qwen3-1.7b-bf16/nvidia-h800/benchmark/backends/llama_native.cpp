// Native llama.cpp requests with backend greedy sampling and independent KV state.
// Reports host intervals only; no device timing is inferred from these intervals.
#include "llama.h"

#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include <time.h>
#include <unistd.h>

static uint64_t monotonic_ns() {
    timespec value{};
    if (clock_gettime(CLOCK_MONOTONIC, &value) != 0)
        throw std::runtime_error("CLOCK_MONOTONIC unavailable");
    return uint64_t(value.tv_sec) * 1000000000ULL + uint64_t(value.tv_nsec);
}

template <class T> static void array(std::ostream & out, const std::vector<T> & values) {
    out << '[';
    for (size_t i = 0; i < values.size(); ++i) out << (i ? "," : "") << values[i];
    out << ']';
}

struct Run {
    int repeat;
    uint64_t start_ns, end_ns, ttft_ns;
    llama_pos final_kv_position;
    std::vector<llama_token> tokens, consumed;
    std::vector<uint64_t> starts, ends;
};

int main(int argc, char ** argv) {
    try {
        if (argc != 7) throw std::runtime_error("Expected GGUF prompts.txt output.json outputs repeats warmups");
        const int outputs = std::stoi(argv[4]), repeats = std::stoi(argv[5]), warmups = std::stoi(argv[6]);
        if (outputs < 2 || repeats < 1 || warmups < 0) throw std::runtime_error("Invalid request counts");
        std::ifstream input(argv[2]);
        int scenarios = 0;
        if (!(input >> scenarios) || scenarios < 1) throw std::runtime_error("Invalid prompt file");
        std::vector<std::vector<llama_token>> prompts;
        for (int scenario = 0; scenario < scenarios; ++scenario) {
            int length = 0;
            if (!(input >> length) || length < 1 || length + outputs > 3072)
                throw std::runtime_error("Invalid prompt length");
            std::vector<llama_token> prompt(length);
            for (auto & token : prompt) if (!(input >> token) || token < 0 || token >= 151936)
                throw std::runtime_error("Invalid prompt token");
            prompts.push_back(prompt);
        }
        std::string extra;
        if (input >> extra) throw std::runtime_error("Trailing prompt input");

        llama_backend_init();
        ggml_backend_dev_t cuda_device = nullptr;
        for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
            auto device = ggml_backend_dev_get(i);
            if (std::string(ggml_backend_reg_name(ggml_backend_dev_backend_reg(device))) != "CUDA") continue;
            if (cuda_device) throw std::runtime_error("Require exactly one visible CUDA device");
            cuda_device = device;
        }
        if (!cuda_device) throw std::runtime_error("The pinned reference has no visible CUDA backend device");
        auto * probe = llama_sampler_init_greedy();
        const bool greedy_supported = probe->iface->backend_init &&
            probe->iface->backend_init(probe, ggml_backend_dev_buffer_type(cuda_device), 1);
        llama_sampler_free(probe);
        if (!greedy_supported) throw std::runtime_error("Native CUDA backend cannot offload greedy argmax");
        ggml_backend_dev_t devices[] = {cuda_device, nullptr};
        auto mp = llama_model_default_params();
        mp.devices = devices;
        mp.n_gpu_layers = -1;
        mp.split_mode = LLAMA_SPLIT_MODE_NONE;
        mp.main_gpu = 0;
        auto * model = llama_model_load_from_file(argv[1], mp);
        if (!model) throw std::runtime_error("Native model load failed");
        if (llama_model_n_layer(model) != 28 || llama_model_n_embd(model) != 2048 ||
            llama_vocab_n_tokens(llama_model_get_vocab(model)) != 151936)
            throw std::runtime_error("Model does not match the full Qwen3-1.7B contract");

        auto * sampler = llama_sampler_chain_init(llama_sampler_chain_default_params());
        llama_sampler_chain_add(sampler, llama_sampler_init_greedy());
        llama_sampler_seq_config sampler_config{0, sampler};
        auto cp = llama_context_default_params();
        cp.n_ctx = cp.n_batch = cp.n_ubatch = 3072;
        cp.n_seq_max = cp.n_outputs_max = cp.n_outputs_max_per_seq = 1;
        cp.n_threads = cp.n_threads_batch = 1;
        cp.type_k = cp.type_v = GGML_TYPE_BF16;
        cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
        cp.offload_kqv = cp.op_offload = true;
        cp.samplers = &sampler_config;
        cp.n_samplers = 1;
        auto * ctx = llama_init_from_model(model, cp);
        if (!ctx) throw std::runtime_error("Native context creation failed");
        auto batch = llama_batch_init(3072, 0, 1);

        auto sample = [&]() {
            const auto token = llama_get_sampled_token_ith(ctx, -1);
            if (token == LLAMA_TOKEN_NULL || token < 0 || token >= 151936)
                throw std::runtime_error("Backend greedy sampling returned no valid full-vocabulary token");
            llama_sampler_accept(sampler, token);
            llama_synchronize(ctx);
            return token;
        };
        auto decode = [&]() {
            const int status = llama_decode(ctx, batch);
            if (status != 0) throw std::runtime_error("Native llama_decode failed: " + std::to_string(status));
            return sample();
        };
        auto request = [&](const std::vector<llama_token> & prompt, int repeat) {
            llama_synchronize(ctx);
            llama_memory_clear(llama_get_memory(ctx), true);
            llama_sampler_reset(sampler);
            llama_synchronize(ctx);
            Run run{};
            run.repeat = repeat;
            run.consumed = prompt;
            run.start_ns = monotonic_ns();
            batch.n_tokens = int(prompt.size());
            for (size_t i = 0; i < prompt.size(); ++i) {
                batch.token[i] = prompt[i];
                batch.pos[i] = int(i);
                batch.n_seq_id[i] = 1;
                batch.seq_id[i][0] = 0;
                batch.logits[i] = i + 1 == prompt.size();
            }
            auto token = decode();
            run.ttft_ns = monotonic_ns() - run.start_ns;
            run.tokens.push_back(token);
            for (int output = 1; output < outputs; ++output) {
                const auto previous_token = token;
                const uint64_t start = monotonic_ns();
                batch.n_tokens = 1;
                batch.token[0] = previous_token;
                batch.pos[0] = int(prompt.size()) + output - 1;
                batch.n_seq_id[0] = 1;
                batch.seq_id[0][0] = 0;
                batch.logits[0] = true;
                token = decode();
                const uint64_t stop = monotonic_ns();
                run.starts.push_back(start);
                run.ends.push_back(stop);
                run.consumed.push_back(previous_token);
                run.tokens.push_back(token);
            }
            run.end_ns = monotonic_ns();
            run.final_kv_position = llama_memory_seq_pos_max(llama_get_memory(ctx), 0);
            if (run.final_kv_position != int(prompt.size()) + outputs - 2)
                throw std::runtime_error("Actual native KV position does not match the generated history");
            return run;
        };

        std::vector<std::vector<Run>> measured;
        for (const auto & prompt : prompts) {
            for (int i = 0; i < warmups; ++i) request(prompt, -1 - i);
            std::vector<Run> runs;
            for (int i = 0; i < repeats; ++i) runs.push_back(request(prompt, i));
            measured.push_back(runs);
        }
        std::ofstream out(argv[3]);
        if (!out) throw std::runtime_error("Unable to open raw output");
        out << std::setprecision(17);
        out << "{\"schema\":\"flagmega.llama-native-raw/v1\",\"pid\":" << getpid()
            << ",\"clock\":\"CLOCK_MONOTONIC\",\"unit\":\"ns\",\"layers\":28,\"vocab\":151936,"
            << "\"capabilities\":{\"cuda_greedy_backend_probe\":true,\"kv_type_requested\":\"BF16\","
            << "\"flash_attention_requested\":true},\"scenarios\":[";
        for (size_t i = 0; i < prompts.size(); ++i) {
            if (i) out << ',';
            out << "{\"prompt_length\":" << prompts[i].size() << ",\"runs\":[";
            for (size_t j = 0; j < measured[i].size(); ++j) {
                if (j) out << ',';
                const auto & r = measured[i][j];
                out << "{\"repeat_index\":" << r.repeat << ",\"request_start_ns\":" << r.start_ns
                    << ",\"request_end_ns\":" << r.end_ns << ",\"ttft_ms\":" << r.ttft_ns / 1e6
                    << ",\"final_kv_position\":" << r.final_kv_position << ",\"prompt_tokens\":";
                array(out, prompts[i]);
                out << ",\"token_ids\":"; array(out, r.tokens);
                out << ",\"consumed_token_ids\":"; array(out, r.consumed);
                out << ",\"step_start_ns\":"; array(out, r.starts);
                out << ",\"step_end_ns\":"; array(out, r.ends);
                out << ",\"decode_step_ms\":[";
                for (size_t k = 0; k < r.starts.size(); ++k)
                    out << (k ? "," : "") << (r.ends[k] - r.starts[k]) / 1e6;
                out << "]}";
            }
            out << "]}";
        }
        out << "]}\n";
        out.close();
        if (!out) throw std::runtime_error("Unable to write raw output");
        llama_batch_free(batch);
        llama_free(ctx);
        llama_sampler_free(sampler);
        llama_model_free(model);
        llama_backend_free();
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "llama native reference: " << error.what() << '\n';
        return 1;
    }
}
