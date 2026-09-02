/*
 * kvcache.h — Key-Value cache for autoregressive inference.
 *
 * Allocates and manages the K and V tensors across all layers and positions.
 * Shape for each tensor: [n_layers, context_len, n_heads, head_dim]
 * Total floats per cache (K or V) = n_layers * context_len * n_embd.
 */

#ifndef KVCACHE_H
#define KVCACHE_H

#include <stddef.h>

typedef struct {
    float *k;           /* [n_layers, context_len, n_heads, head_dim] */
    float *v;           /* [n_layers, context_len, n_heads, head_dim] */
    int    n_layers;
    int    context_len;
    int    n_heads;
    int    head_dim;
} KVCache;

int  kvcache_init(KVCache *kv, int n_layers, int context_len, int n_heads, int head_dim);
void kvcache_reset(KVCache *kv);
void kvcache_free(KVCache *kv);

#endif /* KVCACHE_H */
