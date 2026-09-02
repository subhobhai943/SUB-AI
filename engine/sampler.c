/*
 * sampler.c — Logit sampling implementations for SUB-AI inference engine.
 *
 * Implements greedy argmax selection and top-K temperature-scaled sampling
 * with categorical random selection.
 */

#include "sampler.h"
#include <math.h>
#include <stdlib.h>
#include <float.h>

typedef struct {
    float val;
    int   idx;
} ProbIndex;

int sample_argmax(const float *logits, int vocab_size) {
    if (!logits || vocab_size <= 0) return 0;
    int best_idx = 0;
    float best_val = logits[0];
    for (int i = 1; i < vocab_size; i++) {
        if (logits[i] > best_val) {
            best_val = logits[i];
            best_idx = i;
        }
    }
    return best_idx;
}

int sample_topk(float *logits, int vocab_size, float temperature, int top_k) {
    if (!logits || vocab_size <= 0) return 0;
    if (temperature <= 0.0f || top_k == 1) {
        return sample_argmax(logits, vocab_size);
    }

    if (top_k <= 0 || top_k > vocab_size) {
        top_k = vocab_size;
    }

    /* Scale logits by temperature */
    float inv_temp = 1.0f / temperature;

    ProbIndex *top = (ProbIndex *)malloc(top_k * sizeof(ProbIndex));
    if (!top) {
        return sample_argmax(logits, vocab_size);
    }

    /* Initialize top with first top_k elements */
    for (int i = 0; i < top_k; i++) {
        top[i].val = logits[i] * inv_temp;
        top[i].idx = i;
    }

    /* Find minimum in initial top_k */
    int min_pos = 0;
    for (int i = 1; i < top_k; i++) {
        if (top[i].val < top[min_pos].val) {
            min_pos = i;
        }
    }

    /* Scan rest of vocab */
    for (int i = top_k; i < vocab_size; i++) {
        float val = logits[i] * inv_temp;
        if (val > top[min_pos].val) {
            top[min_pos].val = val;
            top[min_pos].idx = i;
            /* Update min_pos */
            min_pos = 0;
            for (int j = 1; j < top_k; j++) {
                if (top[j].val < top[min_pos].val) {
                    min_pos = j;
                }
            }
        }
    }

    /* Compute softmax over the top_k elements */
    float max_val = top[0].val;
    for (int i = 1; i < top_k; i++) {
        if (top[i].val > max_val) {
            max_val = top[i].val;
        }
    }

    float sum = 0.0f;
    for (int i = 0; i < top_k; i++) {
        top[i].val = expf(top[i].val - max_val);
        sum += top[i].val;
    }

    /* Sample from distribution */
    float r = ((float)rand() / (float)RAND_MAX) * sum;
    float cum = 0.0f;
    int result = top[top_k - 1].idx;

    for (int i = 0; i < top_k; i++) {
        cum += top[i].val;
        if (r <= cum) {
            result = top[i].idx;
            break;
        }
    }

    free(top);
    return result;
}
