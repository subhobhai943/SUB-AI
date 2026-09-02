/*
 * tokenizer.h — Byte-level BPE tokenizer declarations for SUB-AI C engine.
 *
 * Reads tokenizer.json produced by the Python tokenizer training pipeline,
 * encodes input text to token ID sequences using learned BPE merges, and
 * decodes token IDs back into string pieces.
 */

#ifndef TOKENIZER_H
#define TOKENIZER_H

#include <stddef.h>

typedef struct {
    int p0;
    int p1;
    int new_id;
} BPEMerge;

typedef struct {
    char   *bytes;
    size_t  len;
} TokenPiece;

typedef struct {
    int         vocab_size;
    int         num_merges;
    BPEMerge   *merges;
    TokenPiece *tokens;
} Tokenizer;

int         tokenizer_init(Tokenizer *t, const char *json_path);
void        tokenizer_free(Tokenizer *t);
int         tokenizer_encode(Tokenizer *t, const char *text, int *out_ids, int *out_len);
const char *tokenizer_decode_id(Tokenizer *t, int id);

#endif /* TOKENIZER_H */
