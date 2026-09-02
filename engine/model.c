/*
 * model.c — Transformer forward pass implementation for SUB-AI C inference engine.
 *
 * Implements token/position embedding, pre-LN multi-head causal self-attention
 * with KV-cache integration, pre-LN GELU MLP, final LayerNorm, and lm_head projection.
 */

#include "model.h"
#include "matmul.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

int transformer_forward(float *logits,
                        int token,
                        int pos,
                        const ModelHeader *hdr,
                        const ModelWeights *w,
                        KVCache *kv) {
    if (!logits || !hdr || !w || !kv) return -1;
    if (token < 0 || (uint32_t)token >= hdr->vocab_size) {
        fprintf(stderr, "Error: token %d out of vocab range [0, %u)\n", token, hdr->vocab_size);
        return -2;
    }
    if (pos < 0 || (uint32_t)pos >= hdr->context_len) {
        fprintf(stderr, "Error: position %d out of context range [0, %u)\n", pos, hdr->context_len);
        return -3;
    }

    int n_embd      = (int)hdr->n_embd;
    int n_heads     = (int)hdr->n_heads;
    int n_layers    = (int)hdr->n_layers;
    int vocab_size  = (int)hdr->vocab_size;
    int context_len = (int)hdr->context_len;
    int head_dim    = n_embd / n_heads;

    /* Allocate scratch buffers */
    float *x         = (float *)malloc(n_embd * sizeof(float));
    float *x_norm    = (float *)malloc(n_embd * sizeof(float));
    float *qkv       = (float *)malloc(3 * n_embd * sizeof(float));
    float *attn_out  = (float *)malloc(n_embd * sizeof(float));
    float *proj_out  = (float *)malloc(n_embd * sizeof(float));
    float *scores    = (float *)malloc((pos + 1) * sizeof(float));
    float *mlp_act   = (float *)malloc(4 * n_embd * sizeof(float));
    float *mlp_out   = (float *)malloc(n_embd * sizeof(float));

    if (!x || !x_norm || !qkv || !attn_out || !proj_out || !scores || !mlp_act || !mlp_out) {
        fprintf(stderr, "Error: failed to allocate scratch buffers for forward pass\n");
        free(x); free(x_norm); free(qkv); free(attn_out);
        free(proj_out); free(scores); free(mlp_act); free(mlp_out);
        return -4;
    }

    /* 1. Embeddings: x = token_emb[token] + pos_emb[pos] */
    const float *t_emb = w->token_emb + (size_t)token * n_embd;
    const float *p_emb = w->pos_emb + (size_t)pos * n_embd;
    for (int i = 0; i < n_embd; i++) {
        x[i] = t_emb[i] + p_emb[i];
    }

    float scale = 1.0f / sqrtf((float)head_dim);

    /* 2. Loop over transformer blocks */
    for (int l = 0; l < n_layers; l++) {
        /* 2a. Pre-LN for attention */
        layernorm(x_norm, x, w->ln1_gamma[l], w->ln1_beta[l], n_embd);

        /* 2b. QKV projection: x_norm @ qkv_kernel */
        matmul(qkv, x_norm, w->qkv_kernel[l], n_embd, 3 * n_embd);

        const float *q = qkv;
        const float *k = qkv + n_embd;
        const float *v = qkv + 2 * n_embd;

        /* 2c. Store K and V in KV-cache */
        size_t kv_offset_base = ((size_t)l * context_len + pos) * n_embd;
        memcpy(&kv->k[kv_offset_base], k, n_embd * sizeof(float));
        memcpy(&kv->v[kv_offset_base], v, n_embd * sizeof(float));

        /* 2d. Multi-head causal self-attention */
        for (int h = 0; h < n_heads; h++) {
            const float *q_h = q + h * head_dim;

            /* Compute attention scores for past positions t in 0..pos */
            for (int t = 0; t <= pos; t++) {
                size_t t_offset = ((size_t)l * context_len + t) * n_embd + (size_t)h * head_dim;
                const float *k_t = &kv->k[t_offset];
                float dot = 0.0f;
                for (int d = 0; d < head_dim; d++) {
                    dot += q_h[d] * k_t[d];
                }
                scores[t] = dot * scale;
            }

            softmax(scores, pos + 1);

            /* Weighted sum over V */
            for (int d = 0; d < head_dim; d++) {
                float sum = 0.0f;
                for (int t = 0; t <= pos; t++) {
                    size_t t_offset = ((size_t)l * context_len + t) * n_embd + (size_t)h * head_dim;
                    sum += scores[t] * kv->v[t_offset + d];
                }
                attn_out[h * head_dim + d] = sum;
            }
        }

        /* 2e. Attention projection & residual */
        matmul(proj_out, attn_out, w->proj_kernel[l], n_embd, n_embd);
        for (int i = 0; i < n_embd; i++) {
            x[i] += proj_out[i];
        }

        /* 2f. Pre-LN for MLP */
        layernorm(x_norm, x, w->ln2_gamma[l], w->ln2_beta[l], n_embd);

        /* 2g. MLP: FC1 -> GELU -> FC2 */
        matmul(mlp_act, x_norm, w->fc1_kernel[l], n_embd, 4 * n_embd);
        for (int i = 0; i < 4 * n_embd; i++) {
            mlp_act[i] = gelu(mlp_act[i]);
        }
        matmul(mlp_out, mlp_act, w->fc2_kernel[l], 4 * n_embd, n_embd);

        /* 2h. MLP residual */
        for (int i = 0; i < n_embd; i++) {
            x[i] += mlp_out[i];
        }
    }

    /* 3. Final LayerNorm */
    layernorm(x_norm, x, w->final_ln_gamma, w->final_ln_beta, n_embd);

    /* 4. LM Head projection (x_norm @ lm_head_kernel -> logits) */
    matmul(logits, x_norm, w->lm_head_kernel, n_embd, vocab_size);

    /* Clean up scratch memory */
    free(x);
    free(x_norm);
    free(qkv);
    free(attn_out);
    free(proj_out);
    free(scores);
    free(mlp_act);
    free(mlp_out);

    return 0;
}
