/*
 * inference.c — CLI inference executable for SUB-AI C engine.
 *
 * Usage:
 *   ./inference --model model.bin --tokenizer data/tokenizer.json \
 *               --prompt "hello" --max_tokens 200 --temperature 0.8 --top_k 40
 *
 * Implements end-to-end autoregressive generation with streaming output.
 */

#include "loader.h"
#include "kvcache.h"
#include "model.h"
#include "sampler.h"
#include "tokenizer.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static void print_usage(const char *prog) {
    fprintf(stderr, "Usage: %s [options]\n", prog);
    fprintf(stderr, "Options:\n");
    fprintf(stderr, "  --model <path>        Path to binary model file (default: model.bin)\n");
    fprintf(stderr, "  --tokenizer <path>    Path to tokenizer.json (default: data/tokenizer.json)\n");
    fprintf(stderr, "  --prompt <str>        Input prompt string (default: \"\")\n");
    fprintf(stderr, "  --max_tokens <n>      Maximum number of tokens to generate (default: 200)\n");
    fprintf(stderr, "  --temperature <f>     Sampling temperature (default: 0.8)\n");
    fprintf(stderr, "  --top_k <n>           Top-K filtering limit (default: 40)\n");
    fprintf(stderr, "  --seed <n>            Random seed (default: current time)\n");
    fprintf(stderr, "  --help                Show this help message\n");
}

int main(int argc, char *argv[]) {
    const char *model_path     = "model.bin";
    const char *tokenizer_path = "data/tokenizer.json";
    const char *prompt         = "";
    int         max_tokens     = 200;
    float       temperature    = 0.8f;
    int         top_k          = 40;
    unsigned int seed          = (unsigned int)time(NULL);

    /* Parse CLI arguments */
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--model") == 0 && i + 1 < argc) {
            model_path = argv[++i];
        } else if (strcmp(argv[i], "--tokenizer") == 0 && i + 1 < argc) {
            tokenizer_path = argv[++i];
        } else if (strcmp(argv[i], "--prompt") == 0 && i + 1 < argc) {
            prompt = argv[++i];
        } else if (strcmp(argv[i], "--max_tokens") == 0 && i + 1 < argc) {
            max_tokens = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--temperature") == 0 && i + 1 < argc) {
            temperature = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--top_k") == 0 && i + 1 < argc) {
            top_k = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc) {
            seed = (unsigned int)strtoul(argv[++i], NULL, 10);
        } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            print_usage(argv[0]);
            return 0;
        } else {
            fprintf(stderr, "Unknown or incomplete argument: %s\n", argv[i]);
            print_usage(argv[0]);
            return 1;
        }
    }

    srand(seed);

    /* 1. Load model */
    ModelHeader hdr;
    ModelWeights weights;
    if (load_model(model_path, &hdr, &weights) != 0) {
        fprintf(stderr, "Failed to load model from %s\n", model_path);
        return 1;
    }

    /* 2. Load tokenizer */
    Tokenizer tok;
    if (tokenizer_init(&tok, tokenizer_path) != 0) {
        fprintf(stderr, "Failed to initialize tokenizer from %s\n", tokenizer_path);
        free_model(&weights);
        return 1;
    }

    /* 3. Initialize KV Cache */
    KVCache kv;
    int head_dim = (int)hdr.n_embd / (int)hdr.n_heads;
    if (kvcache_init(&kv, (int)hdr.n_layers, (int)hdr.context_len, (int)hdr.n_heads, head_dim) != 0) {
        fprintf(stderr, "Failed to initialize KV cache\n");
        tokenizer_free(&tok);
        free_model(&weights);
        return 1;
    }

    /* 4. Allocate logits buffer */
    float *logits = (float *)malloc(hdr.vocab_size * sizeof(float));
    if (!logits) {
        fprintf(stderr, "Failed to allocate logits buffer\n");
        kvcache_free(&kv);
        tokenizer_free(&tok);
        free_model(&weights);
        return 1;
    }

    /* 5. Encode prompt */
    int prompt_capacity = (int)hdr.context_len;
    int *prompt_ids = (int *)malloc(prompt_capacity * sizeof(int));
    int prompt_len = 0;

    if (strlen(prompt) > 0) {
        tokenizer_encode(&tok, prompt, prompt_ids, &prompt_len);
    }

    if (prompt_len == 0) {
        /* If prompt is empty, seed with token 0 */
        prompt_ids[0] = 0;
        prompt_len = 1;
    }

    if (prompt_len > (int)hdr.context_len - 1) {
        prompt_len = (int)hdr.context_len - 1;
    }

    /* 6. Prefill prompt */
    for (int pos = 0; pos < prompt_len; pos++) {
        int token = prompt_ids[pos];
        if (transformer_forward(logits, token, pos, &hdr, &weights, &kv) != 0) {
            fprintf(stderr, "Forward pass failed at position %d\n", pos);
            break;
        }
        printf("%s", tokenizer_decode_id(&tok, token));
        fflush(stdout);
    }

    /* 7. Autoregressive generation loop */
    int current_pos = prompt_len;
    int remaining = max_tokens;
    if (current_pos + remaining > (int)hdr.context_len) {
        remaining = (int)hdr.context_len - current_pos;
    }

    for (int step = 0; step < remaining; step++) {
        int next_token = sample_topk(logits, (int)hdr.vocab_size, temperature, top_k);
        printf("%s", tokenizer_decode_id(&tok, next_token));
        fflush(stdout);

        if (transformer_forward(logits, next_token, current_pos, &hdr, &weights, &kv) != 0) {
            fprintf(stderr, "Forward pass failed during generation at position %d\n", current_pos);
            break;
        }
        current_pos++;
    }
    printf("\n");

    /* 8. Clean up resources */
    free(prompt_ids);
    free(logits);
    kvcache_free(&kv);
    tokenizer_free(&tok);
    free_model(&weights);

    return 0;
}
