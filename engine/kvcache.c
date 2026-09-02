/*
 * kvcache.c — Key-Value cache implementation.
 *
 * Implements allocation, resetting, and freeing of the K and V cache buffers
 * used for accelerating autoregressive transformer generation.
 */

#include "kvcache.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int kvcache_init(KVCache *kv, int n_layers, int context_len, int n_heads, int head_dim) {
    if (!kv) return -1;
    memset(kv, 0, sizeof(*kv));

    kv->n_layers    = n_layers;
    kv->context_len = context_len;
    kv->n_heads     = n_heads;
    kv->head_dim    = head_dim;

    size_t total_floats = (size_t)n_layers * context_len * n_heads * head_dim;
    kv->k = (float *)calloc(total_floats, sizeof(float));
    kv->v = (float *)calloc(total_floats, sizeof(float));

    if (!kv->k || !kv->v) {
        fprintf(stderr, "Error: failed to allocate memory for KV cache (%zu floats)\n", total_floats);
        kvcache_free(kv);
        return -1;
    }

    return 0;
}

void kvcache_reset(KVCache *kv) {
    if (!kv || !kv->k || !kv->v) return;
    size_t total_floats = (size_t)kv->n_layers * kv->context_len * kv->n_heads * kv->head_dim;
    memset(kv->k, 0, total_floats * sizeof(float));
    memset(kv->v, 0, total_floats * sizeof(float));
}

void kvcache_free(KVCache *kv) {
    if (!kv) return;
    if (kv->k) {
        free(kv->k);
        kv->k = NULL;
    }
    if (kv->v) {
        free(kv->v);
        kv->v = NULL;
    }
    kv->n_layers = 0;
    kv->context_len = 0;
    kv->n_heads = 0;
    kv->head_dim = 0;
}
