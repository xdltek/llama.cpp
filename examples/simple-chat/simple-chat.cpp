#include "llama.h"

#include <chrono>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>
#include <fstream>

static void print_usage(int, char ** argv) {
    printf("\nexample usage:\n");
    printf("\n    %s -m model.gguf [-c context_size] [-ngl n_gpu_layers] [-if input.txt] [-of output.txt] [-ub n_ubatch]\n", argv[0]);
    printf("\n");
}

int main(int argc, char ** argv) {
    std::string model_path;
    int         ngl   = 99;
    int         n_ctx = 2048;
    int         n_ubatch = 128;
    std::string input_file;
    std::string output_file;

    // parse command line arguments
    for (int i = 1; i < argc; i++) {
        try {
            if (strcmp(argv[i], "-m") == 0) {
                if (i + 1 < argc) {
                    model_path = argv[++i];
                } else {
                    print_usage(argc, argv);
                    return 1;
                }
            } else if (strcmp(argv[i], "-c") == 0) {
                if (i + 1 < argc) {
                    n_ctx = std::stoi(argv[++i]);
                } else {
                    print_usage(argc, argv);
                    return 1;
                }
            } else if (strcmp(argv[i], "-ngl") == 0) {
                if (i + 1 < argc) {
                    ngl = std::stoi(argv[++i]);
                } else {
                    print_usage(argc, argv);
                    return 1;
                }
            } else if (strcmp(argv[i], "-if") == 0 || strcmp(argv[i], "--input-file") == 0) {
                if (i + 1 < argc) {
                    input_file = argv[++i];
                } else {
                    print_usage(argc, argv);
                    return 1;
                }
            } else if (strcmp(argv[i], "-of") == 0 || strcmp(argv[i], "--output-file") == 0) {
                if (i + 1 < argc) {
                    output_file = argv[++i];
                } else {
                    print_usage(argc, argv);
                    return 1;
                }
            }
            else if (strcmp(argv[i], "-ub") == 0) {
                if (i + 1 < argc) {
                    n_ubatch = std::stoi(argv[++i]);
                } else {
                    print_usage(argc, argv);
                    return 1;
                }
            } else {
                print_usage(argc, argv);
                return 1;
            }
        } catch (std::exception & e) {
            fprintf(stderr, "error: %s\n", e.what());
            print_usage(argc, argv);
            return 1;
        }
    }
    if (model_path.empty()) {
        print_usage(argc, argv);
        return 1;
    }

    // only print errors
    llama_log_set(
        [](enum ggml_log_level level, const char * text, void * /* user_data */) {
            if (level >= GGML_LOG_LEVEL_DEBUG) {
                fprintf(stderr, "%s", text);
            }
        },
        nullptr);

    // load dynamic backends
    ggml_backend_load_all();
    // initialize the model
    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers       = ngl;

    // **********add for test framework**********
    auto test_fw_load_start = std::chrono::high_resolution_clock::now();
    // **********add for test framework**********
    llama_model * model = llama_model_load_from_file(model_path.c_str(), model_params);
    if (!model) {
        fprintf(stderr, "%s: error: unable to load model\n", __func__);
        return 1;
    }
    const llama_vocab * vocab = llama_model_get_vocab(model);

    // initialize the context
    llama_context_params ctx_params = llama_context_default_params();

    // **********add for gaok**********
    ctx_params.n_ctx           = n_ctx;
    ctx_params.n_batch         = n_ctx;
    ctx_params.n_threads       = 1;
    ctx_params.n_threads_batch = 1;
    ctx_params.n_ubatch        = n_ubatch; // now only support 128
    ctx_params.offload_kqv     = true;

    // ctx_params.type_k = GGML_TYPE_F32;
    // ctx_params.type_v = GGML_TYPE_F32;

    ctx_params.type_k = GGML_TYPE_BF16; // now only support bf16
    ctx_params.type_v = GGML_TYPE_BF16;
    // **********add for gaok**********

    llama_context * ctx = llama_init_from_model(model, ctx_params);
    if (!ctx) {
        fprintf(stderr, "%s: error: failed to create the llama_context\n", __func__);
        return 1;
    }

    // initialize the sampler
    llama_sampler * smpl = llama_sampler_chain_init(llama_sampler_chain_default_params());
    // llama_sampler_chain_add(smpl, llama_sampler_init_min_p(0.05f, 1));
    // llama_sampler_chain_add(smpl, llama_sampler_init_temp(0.8f));
    // llama_sampler_chain_add(smpl, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));
    llama_sampler_chain_add(smpl, llama_sampler_init_greedy());
    // helper function to evaluate a prompt and generate a response
    bool context_exceeded = false;

    // **********add for test framework**********
    auto test_fw_load_end = std::chrono::high_resolution_clock::now();
    auto test_fw_load_duration = std::chrono::duration_cast<std::chrono::milliseconds>(test_fw_load_end - test_fw_load_start).count();
    printf("\n*** Model Load Time: %ld ms ***\n", test_fw_load_duration);
    fflush(stdout);
    // **********add for test framework**********
    

    auto generate = [&](const std::string & prompt, std::string & metrics) {
        std::string response;
        const bool  is_first = llama_memory_seq_pos_max(llama_get_memory(ctx), 0) == -1;

        // tokenize the prompt
        const int n_prompt_tokens = -llama_tokenize(vocab, prompt.c_str(), prompt.size(), NULL, 0, is_first, true);
        // compute aligned prompt size based on n_ubatch
        int aligned_prompt_size = n_prompt_tokens;
        {
            int ub = (int)ctx_params.n_ubatch;
            if (ub > 0 && n_prompt_tokens % ub != 0) {
                aligned_prompt_size = ((n_prompt_tokens / ub) + 1) * ub;
            }
        }

        char buf_metrics[512];
        snprintf(buf_metrics, sizeof(buf_metrics),
                 "prompt tokens number: %d\naligned prompt size: %d\n",
                 n_prompt_tokens, aligned_prompt_size);
        metrics += buf_metrics;
        printf("prompt tokens number: %d, aligned prompt size: %d\n", n_prompt_tokens, aligned_prompt_size);
        std::vector<llama_token> prompt_tokens(n_prompt_tokens);
        if (llama_tokenize(vocab, prompt.c_str(), prompt.size(), prompt_tokens.data(), prompt_tokens.size(), is_first,
                           true) < 0) {
            GGML_ABORT("failed to tokenize the prompt\n");
        }

        // prepare a batch for the prompt
        llama_batch batch = llama_batch_get_one(prompt_tokens.data(), prompt_tokens.size());
        llama_token new_token_id;
        // **********add for gaok**********
        int64_t     t_interval       = 0;
        int64_t     first_t_interval = 0;
        int64_t     ttft_ms          = -1;
        uint64_t    num              = 0;
        auto        prompt_start     = std::chrono::high_resolution_clock::now();
        // **********add for gaok**********
        while (true) {
            // check if we have enough space in the context to evaluate this batch
            int n_ctx      = llama_n_ctx(ctx);
            int n_ctx_used = llama_memory_seq_pos_max(llama_get_memory(ctx), 0) + 1;
            if (n_ctx_used + batch.n_tokens > n_ctx) {
                printf("\033[0m\n");
                fprintf(stderr, "context size exceeded\n");
                context_exceeded = true;
                break;
            }
            // **********add for gaok**********
            auto now    = std::chrono::system_clock::now();
            auto now_ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch());
            // **********add for gaok**********
            int  ret    = llama_decode(ctx, batch);
            if (ret != 0) {
                GGML_ABORT("failed to decode, ret = %d\n", ret);
            }
            // **********add for gaok**********
            num++;
            auto end    = std::chrono::system_clock::now();
            auto end_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end.time_since_epoch());
            if (num == 1) {
                first_t_interval = (end_ms.count() - now_ms.count());
            }
            if (num > 2) {
                t_interval += (end_ms.count() - now_ms.count());
            }

            // **********add for gaok**********

            // sample the next token
            new_token_id = llama_sampler_sample(smpl, ctx, -1);
            if (num == 1 && ttft_ms < 0) {
                auto first_token_end = std::chrono::high_resolution_clock::now();
                ttft_ms = std::chrono::duration_cast<std::chrono::milliseconds>(first_token_end - prompt_start).count();
            }

            // is it an end of generation?
            if (llama_vocab_is_eog(vocab, new_token_id)) {
                // **********add for gaok**********
                if (num > 1) {
                    double t_one_token = (double) t_interval / (num - 2);
                    int    n_batch     = n_prompt_tokens / ctx_params.n_ubatch;
                    int    r_batch     = n_prompt_tokens % ctx_params.n_ubatch;
                    double t_first_token = (double) first_t_interval / (r_batch == 0 ? n_prompt_tokens : (n_batch + 1) * ctx_params.n_ubatch);
                    snprintf(buf_metrics, sizeof(buf_metrics),
                             "\nttft ms: %ld\ninfer tokens number: %ld, prefill speed: %.3f tokens/s, decode speed: %.3f tokens/s\n",
                             ttft_ms, num, 1000 / t_first_token, 1000 / t_one_token);
                    metrics += buf_metrics;
                    printf("%s", buf_metrics);
                    num              = 0;
                    t_interval       = 0;
                    first_t_interval = 0;
                    ttft_ms          = -1;
                }
                // **********add for gaok**********
                break;
            }

            // convert the token to a string, print it and add it to the response
            char buf[256];
            int  n = llama_token_to_piece(vocab, new_token_id, buf, sizeof(buf), 0, true);
            if (n < 0) {
                GGML_ABORT("failed to convert token to piece\n");
            }
            std::string piece(buf, n);
            printf("%s", piece.c_str());
            fflush(stdout);
            response += piece;

            // prepare the next batch with the sampled token
            batch = llama_batch_get_one(&new_token_id, 1);
        }

        return response;
    };

    std::vector<llama_chat_message> messages;
    std::vector<char>               formatted(llama_n_ctx(ctx));
    int                             prev_len = 0;

    if (!input_file.empty()) {
        std::ifstream infile(input_file);
        if (!infile.is_open()) {
            fprintf(stderr, "error: failed to open input file %s\n", input_file.c_str());
            return 1;
        }
        std::ofstream outfile;
        if (!output_file.empty()) {
            outfile.open(output_file, std::ios::app);
            if (!outfile.is_open()) {
                fprintf(stderr, "error: failed to open output file %s\n", output_file.c_str());
                return 1;
            }
        }

        std::string line;
        while (std::getline(infile, line)) {
            if (line.empty()) continue;

            printf("\n\033[32m[Prompt]\033[0m %s\n", line.c_str());

            messages.clear();
            prev_len = 0;

            const char * tmpl = llama_model_chat_template(model, /* name */ nullptr);

            messages.push_back({ "user", strdup(line.c_str()) });
            int new_len =
                llama_chat_apply_template(tmpl, messages.data(), messages.size(), true, formatted.data(), formatted.size());
            if (new_len > (int) formatted.size()) {
                formatted.resize(new_len);
                new_len = llama_chat_apply_template(tmpl, messages.data(), messages.size(), true, formatted.data(),
                                                    formatted.size());
            }
            if (new_len < 0) {
                fprintf(stderr, "failed to apply the chat template\n");
                return 1;
            }

            std::string prompt(formatted.begin() + prev_len, formatted.begin() + new_len);

            printf("\033[33m");
            std::string metrics;
            std::string response = generate(prompt, metrics);
            printf("\n\033[0m");

            if (outfile.is_open()) {
                outfile << response << "\n\n";
                outfile << metrics << "\n";
                outfile.flush();
            }

            for (auto & msg : messages) {
                free(const_cast<char *>(msg.content));
            }
            messages.clear();

            if (context_exceeded) {
                break;
            }

            llama_memory_clear(llama_get_memory(ctx), true);
        }

        if (outfile.is_open()) {
            outfile.close();
        }
    } else {
        while (true) {
            // get user input
            printf("\033[32m> \033[0m");
            std::string user;
            std::getline(std::cin, user);

            if (user.empty()) {
                break;
            }

            const char * tmpl = llama_model_chat_template(model, /* name */ nullptr);

            // add the user input to the message list and format it
            messages.push_back({ "user", strdup(user.c_str()) });
            int new_len =
                llama_chat_apply_template(tmpl, messages.data(), messages.size(), true, formatted.data(), formatted.size());
            if (new_len > (int) formatted.size()) {
                formatted.resize(new_len);
                new_len = llama_chat_apply_template(tmpl, messages.data(), messages.size(), true, formatted.data(),
                                                    formatted.size());
            }
            if (new_len < 0) {
                fprintf(stderr, "failed to apply the chat template\n");
                return 1;
            }

            // remove previous messages to obtain the prompt to generate the response
            std::string prompt(formatted.begin() + prev_len, formatted.begin() + new_len);

            // generate a response
            printf("\033[33m");
            std::string metrics;
            std::string response = generate(prompt, metrics);
            printf("\n\033[0m");

            if (context_exceeded) {
                break;
            }

            // add the response to the messages
            messages.push_back({ "assistant", strdup(response.c_str()) });
            prev_len = llama_chat_apply_template(tmpl, messages.data(), messages.size(), false, nullptr, 0);
            if (prev_len < 0) {
                fprintf(stderr, "failed to apply the chat template\n");
                return 1;
            }
        }
    }

    // free resources
    for (auto & msg : messages) {
        free(const_cast<char *>(msg.content));
    }
    llama_sampler_free(smpl);
    llama_free(ctx);
    llama_model_free(model);

    return 0;
}
