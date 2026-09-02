/*
 * model.h — Transformer forward pass for SUB-AI C inference engine.
 *
 * Implements single-token autoregressive forward pass using KV-cache.
 */

#ifndef MODEL_H
#define MODEL_H

#include "loader.h"
#include "kvcache.h"

int transformer_forward(float *logits,
                        int token,
                        int pos,
                        const ModelHeader *hdr,
                        const ModelWeights *w,
                        KVCache *kv);

#endif /* MODEL_H */
