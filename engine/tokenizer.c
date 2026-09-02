/*
 * tokenizer.c — Byte-level BPE tokenizer implementation for SUB-AI C engine.
 *
 * Implements tokenizer_init() (JSON parsing for merges and vocab size),
 * tokenizer_encode() (greedy lowest-rank pair merging), tokenizer_decode_id()
 * (string piece lookup), and tokenizer_free().
 */

#include "tokenizer.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

/* Helper to skip whitespace in a string pointer */
static const char *skip_ws(const char *p) {
    while (*p && isspace((unsigned char)*p)) p++;
    return p;
}

int tokenizer_init(Tokenizer *t, const char *json_path) {
    if (!t || !json_path) return -1;
    memset(t, 0, sizeof(*t));

    FILE *f = fopen(json_path, "rb");
    if (!f) {
        fprintf(stderr, "Error: unable to open tokenizer file '%s'\n", json_path);
        return -1;
    }

    fseek(f, 0, SEEK_END);
    long file_size = ftell(f);
    fseek(f, 0, SEEK_SET);

    if (file_size <= 0) {
        fclose(f);
        return -2;
    }

    char *buf = (char *)malloc(file_size + 1);
    if (!buf) {
        fclose(f);
        return -3;
    }

    size_t read_bytes = fread(buf, 1, file_size, f);
    fclose(f);
    buf[read_bytes] = '\0';

    /* Parse vocab_size */
    t->vocab_size = 8000; /* default fallback */
    const char *vpos = strstr(buf, "\"vocab_size\"");
    if (vpos) {
        vpos += strlen("\"vocab_size\"");
        vpos = skip_ws(vpos);
        if (*vpos == ':') vpos++;
        vpos = skip_ws(vpos);
        t->vocab_size = (int)strtol(vpos, NULL, 10);
    }

    /* First pass: count merges in "merges": [ ... ] */
    int merge_capacity = t->vocab_size;
    t->merges = (BPEMerge *)malloc(merge_capacity * sizeof(BPEMerge));
    if (!t->merges) {
        free(buf);
        return -4;
    }
    t->num_merges = 0;

    const char *mpos = strstr(buf, "\"merges\"");
    if (mpos) {
        mpos += strlen("\"merges\"");
        mpos = skip_ws(mpos);
        if (*mpos == ':') mpos++;
        mpos = skip_ws(mpos);
        if (*mpos == '[') {
            mpos++;
            while (*mpos && *mpos != ']') {
                mpos = skip_ws(mpos);
                if (*mpos == '[') {
                    mpos++;
                    char *endptr = NULL;
                    int p0 = (int)strtol(mpos, &endptr, 10);
                    if (endptr && endptr != mpos) {
                        mpos = skip_ws(endptr);
                        if (*mpos == ',') mpos++;
                        mpos = skip_ws(mpos);
                        int p1 = (int)strtol(mpos, &endptr, 10);
                        if (endptr && endptr != mpos) {
                            mpos = endptr;
                            while (*mpos && *mpos != ']') mpos++;
                            if (*mpos == ']') mpos++;

                            if (t->num_merges < merge_capacity) {
                                t->merges[t->num_merges].p0 = p0;
                                t->merges[t->num_merges].p1 = p1;
                                t->merges[t->num_merges].new_id = 256 + t->num_merges;
                                t->num_merges++;
                            }
                        }
                    }
                } else {
                    mpos++;
                }
            }
        }
    }

    free(buf);

    /* Allocate and initialize token pieces */
    t->tokens = (TokenPiece *)calloc(t->vocab_size, sizeof(TokenPiece));
    if (!t->tokens) {
        free(t->merges);
        t->merges = NULL;
        return -5;
    }

    /* Seed single bytes 0..255 */
    for (int i = 0; i < 256 && i < t->vocab_size; i++) {
        t->tokens[i].len = 1;
        t->tokens[i].bytes = (char *)malloc(2);
        t->tokens[i].bytes[0] = (char)(unsigned char)i;
        t->tokens[i].bytes[1] = '\0';
    }

    /* Reconstruct merged token strings */
    for (int m = 0; m < t->num_merges; m++) {
        int id = t->merges[m].new_id;
        if (id < t->vocab_size) {
            int p0 = t->merges[m].p0;
            int p1 = t->merges[m].p1;
            size_t l0 = (p0 >= 0 && p0 < t->vocab_size && t->tokens[p0].bytes) ? t->tokens[p0].len : 0;
            size_t l1 = (p1 >= 0 && p1 < t->vocab_size && t->tokens[p1].bytes) ? t->tokens[p1].len : 0;
            size_t total_l = l0 + l1;

            t->tokens[id].len = total_l;
            t->tokens[id].bytes = (char *)malloc(total_l + 1);
            if (l0 > 0) memcpy(t->tokens[id].bytes, t->tokens[p0].bytes, l0);
            if (l1 > 0) memcpy(t->tokens[id].bytes + l0, t->tokens[p1].bytes, l1);
            t->tokens[id].bytes[total_l] = '\0';
        }
    }

    /* Fill any remaining empty slots */
    for (int i = 0; i < t->vocab_size; i++) {
        if (!t->tokens[i].bytes) {
            t->tokens[i].len = 0;
            t->tokens[i].bytes = (char *)calloc(1, 1);
        }
    }

    return 0;
}

void tokenizer_free(Tokenizer *t) {
    if (!t) return;
    if (t->tokens) {
        for (int i = 0; i < t->vocab_size; i++) {
            if (t->tokens[i].bytes) {
                free(t->tokens[i].bytes);
                t->tokens[i].bytes = NULL;
            }
        }
        free(t->tokens);
        t->tokens = NULL;
    }
    if (t->merges) {
        free(t->merges);
        t->merges = NULL;
    }
    t->vocab_size = 0;
    t->num_merges = 0;
}

int tokenizer_encode(Tokenizer *t, const char *text, int *out_ids, int *out_len) {
    if (!t || !text || !out_ids || !out_len) return -1;
    size_t text_len = strlen(text);
    if (text_len == 0) {
        *out_len = 0;
        return 0;
    }

    int *ids = (int *)malloc(text_len * sizeof(int));
    if (!ids) return -2;

    for (size_t i = 0; i < text_len; i++) {
        ids[i] = (unsigned char)text[i];
    }
    int n = (int)text_len;

    /* Iterative BPE merging */
    while (n >= 2) {
        int best_rank = -1;
        int best_merge_idx = -1;

        for (int i = 0; i < n - 1; i++) {
            int p0 = ids[i];
            int p1 = ids[i + 1];

            for (int m = 0; m < t->num_merges; m++) {
                if (t->merges[m].p0 == p0 && t->merges[m].p1 == p1) {
                    if (best_rank == -1 || m < best_rank) {
                        best_rank = m;
                        best_merge_idx = m;
                    }
                    break;
                }
            }
        }

        if (best_rank == -1) {
            break;
        }

        int target_p0 = t->merges[best_merge_idx].p0;
        int target_p1 = t->merges[best_merge_idx].p1;
        int new_id    = t->merges[best_merge_idx].new_id;

        int write_pos = 0;
        for (int i = 0; i < n; ) {
            if (i < n - 1 && ids[i] == target_p0 && ids[i + 1] == target_p1) {
                ids[write_pos++] = new_id;
                i += 2;
            } else {
                ids[write_pos++] = ids[i++];
            }
        }
        n = write_pos;
    }

    for (int i = 0; i < n; i++) {
        out_ids[i] = ids[i];
    }
    *out_len = n;

    free(ids);
    return 0;
}

const char *tokenizer_decode_id(Tokenizer *t, int id) {
    if (!t || !t->tokens || id < 0 || id >= t->vocab_size) {
        return "";
    }
    return t->tokens[id].bytes ? t->tokens[id].bytes : "";
}
