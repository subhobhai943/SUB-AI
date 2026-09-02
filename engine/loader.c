/*
 * loader.c — SUBA binary weight format loader implementation.
 *
 * Implements load_model() to read the 256-byte header and all packed float32
 * weights into contiguous memory, wiring up pointers for all layers.
 * Implements free_model() to safely release allocated memory.
 */

#include "loader.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int load_model(const char *path, ModelHeader *hdr, ModelWeights *w) {
    if (!path || !hdr || !w) return -1;
    memset(hdr, 0, sizeof(*hdr));
    memset(w, 0, sizeof(*w));

    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "Error: unable to open model file '%s'\n", path);
        return -1;
    }

    if (fread(hdr, sizeof(uint32_t), HEADER_INTS, f) != HEADER_INTS) {
        fprintf(stderr, "Error: failed to read 256-byte header from '%s'\n", path);
        fclose(f);
        return -2;
    }

    if (hdr->magic != SUBA_MAGIC) {
        fprintf(stderr, "Error: invalid magic 0x%08x (expected 0x%08x)\n", hdr->magic, SUBA_MAGIC);
        fclose(f);
        return -3;
    }

    if (hdr->version != SUBA_VERSION) {
        fprintf(stderr, "Error: unsupported format version %u (expected %u)\n", hdr->version, SUBA_VERSION);
        fclose(f);
        return -4;
    }

    uint32_t vocab_size  = hdr->vocab_size;
    uint32_t context_len = hdr->context_len;
    uint32_t n_embd      = hdr->n_embd;
    uint32_t n_layers    = hdr->n_layers;

    size_t total_floats = 0;
    total_floats += (size_t)vocab_size * n_embd;                      /* token_emb */
    total_floats += (size_t)context_len * n_embd;                     /* pos_emb */

    /* Per-layer weights */
    size_t layer_floats = (size_t)n_embd                              /* ln1_gamma */
                        + (size_t)n_embd                              /* ln1_beta */
                        + (size_t)n_embd * (3 * n_embd)               /* qkv_kernel */
                        + (size_t)n_embd * n_embd                     /* proj_kernel */
                        + (size_t)n_embd                              /* ln2_gamma */
                        + (size_t)n_embd                              /* ln2_beta */
                        + (size_t)n_embd * (4 * n_embd)               /* fc1_kernel */
                        + (size_t)(4 * n_embd) * n_embd;              /* fc2_kernel */
    total_floats += (size_t)n_layers * layer_floats;

    total_floats += (size_t)n_embd;                                   /* final_ln_gamma */
    total_floats += (size_t)n_embd;                                   /* final_ln_beta */
    total_floats += (size_t)n_embd * vocab_size;                     /* lm_head_kernel */

    w->data = (float *)malloc(total_floats * sizeof(float));
    if (!w->data) {
        fprintf(stderr, "Error: failed to allocate %zu bytes for model weights\n", total_floats * sizeof(float));
        fclose(f);
        return -5;
    }

    size_t read_floats = fread(w->data, sizeof(float), total_floats, f);
    fclose(f);

    if (read_floats != total_floats) {
        fprintf(stderr, "Error: read %zu floats, expected %zu\n", read_floats, total_floats);
        free(w->data);
        w->data = NULL;
        return -6;
    }

    /* Allocate layer pointer tables */
    w->ln1_gamma   = (float **)malloc(n_layers * sizeof(float *));
    w->ln1_beta    = (float **)malloc(n_layers * sizeof(float *));
    w->qkv_kernel  = (float **)malloc(n_layers * sizeof(float *));
    w->proj_kernel = (float **)malloc(n_layers * sizeof(float *));
    w->ln2_gamma   = (float **)malloc(n_layers * sizeof(float *));
    w->ln2_beta    = (float **)malloc(n_layers * sizeof(float *));
    w->fc1_kernel  = (float **)malloc(n_layers * sizeof(float *));
    w->fc2_kernel  = (float **)malloc(n_layers * sizeof(float *));

    if (!w->ln1_gamma || !w->ln1_beta || !w->qkv_kernel || !w->proj_kernel ||
        !w->ln2_gamma || !w->ln2_beta || !w->fc1_kernel || !w->fc2_kernel) {
        fprintf(stderr, "Error: failed to allocate layer pointer arrays\n");
        free_model(w);
        return -7;
    }

    float *ptr = w->data;
    w->token_emb = ptr;
    ptr += (size_t)vocab_size * n_embd;

    w->pos_emb = ptr;
    ptr += (size_t)context_len * n_embd;

    for (uint32_t i = 0; i < n_layers; i++) {
        w->ln1_gamma[i] = ptr;
        ptr += n_embd;
        w->ln1_beta[i] = ptr;
        ptr += n_embd;
        w->qkv_kernel[i] = ptr;
        ptr += (size_t)n_embd * (3 * n_embd);
        w->proj_kernel[i] = ptr;
        ptr += (size_t)n_embd * n_embd;
        w->ln2_gamma[i] = ptr;
        ptr += n_embd;
        w->ln2_beta[i] = ptr;
        ptr += n_embd;
        w->fc1_kernel[i] = ptr;
        ptr += (size_t)n_embd * (4 * n_embd);
        w->fc2_kernel[i] = ptr;
        ptr += (size_t)(4 * n_embd) * n_embd;
    }

    w->final_ln_gamma = ptr;
    ptr += n_embd;
    w->final_ln_beta = ptr;
    ptr += n_embd;

    w->lm_head_kernel = ptr;
    ptr += (size_t)n_embd * vocab_size;

    return 0;
}

void free_model(ModelWeights *w) {
    if (!w) return;
    if (w->data) {
        free(w->data);
        w->data = NULL;
    }
    if (w->ln1_gamma) { free(w->ln1_gamma); w->ln1_gamma = NULL; }
    if (w->ln1_beta) { free(w->ln1_beta); w->ln1_beta = NULL; }
    if (w->qkv_kernel) { free(w->qkv_kernel); w->qkv_kernel = NULL; }
    if (w->proj_kernel) { free(w->proj_kernel); w->proj_kernel = NULL; }
    if (w->ln2_gamma) { free(w->ln2_gamma); w->ln2_gamma = NULL; }
    if (w->ln2_beta) { free(w->ln2_beta); w->ln2_beta = NULL; }
    if (w->fc1_kernel) { free(w->fc1_kernel); w->fc1_kernel = NULL; }
    if (w->fc2_kernel) { free(w->fc2_kernel); w->fc2_kernel = NULL; }
}
