/*
 * loader.h — SUBA binary weight format loader declarations.
 *
 * Defines ModelHeader (256-byte little-endian header) and ModelWeights
 * (pointer struct into the flat weight buffer). Call load_model() to
 * read the .bin file and populate both structs; call free_model()
 * when done to release memory.
 *
 * Binary format:
 *   Header  : 256 bytes, 64 x uint32_t, little-endian
 *   Weights : packed float32, row-major, no padding
 */

#ifndef LOADER_H
#define LOADER_H

#include <stdint.h>

#define SUBA_MAGIC   0x53554241u   /* ASCII 'SUBA' */
#define SUBA_VERSION 1u
#define HEADER_INTS  64            /* 64 x uint32_t = 256 bytes */

typedef struct {
    uint32_t magic;
    uint32_t version;
    uint32_t vocab_size;
    uint32_t context_len;
    uint32_t n_embd;
    uint32_t n_heads;
    uint32_t n_layers;
    uint32_t _pad[57];
} ModelHeader;

typedef struct {
    float *data;            /* raw allocation — free this */
    float *token_emb;       /* [vocab_size, n_embd]       */
    float *pos_emb;         /* [context_len, n_embd]      */
    float **ln1_gamma;      /* [n_layers][n_embd]         */
    float **ln1_beta;       /* [n_layers][n_embd]         */
    float **qkv_kernel;     /* [n_layers][n_embd, 3*n_embd] */
    float **proj_kernel;    /* [n_layers][n_embd, n_embd]   */
    float **ln2_gamma;      /* [n_layers][n_embd]         */
    float **ln2_beta;       /* [n_layers][n_embd]         */
    float **fc1_kernel;     /* [n_layers][n_embd, 4*n_embd] */
    float **fc2_kernel;     /* [n_layers][4*n_embd, n_embd] */
    float *final_ln_gamma;  /* [n_embd]                   */
    float *final_ln_beta;   /* [n_embd]                   */
    float *lm_head_kernel;  /* [n_embd, vocab_size] weight-tied */
} ModelWeights;

int  load_model(const char *path, ModelHeader *hdr, ModelWeights *w);
void free_model(ModelWeights *w);

#endif /* LOADER_H */
