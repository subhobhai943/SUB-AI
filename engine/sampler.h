/*
 * sampler.h — Logit sampling methods for SUB-AI inference engine.
 *
 * Provides:
 * - apply_repetition_penalty: Penalizes recently generated tokens to eliminate looping
 * - sample_argmax: Greedy deterministic sampling (temperature = 0)
 * - sample_topk: Top-K temperature-scaled multinomial sampling
 */

#ifndef SAMPLER_H
#define SAMPLER_H

void apply_repetition_penalty(float *logits, int vocab_size, const int *context_tokens, int context_len, float penalty);
int  sample_argmax(const float *logits, int vocab_size);
int  sample_topk(float *logits, int vocab_size, float temperature, int top_k);

#endif /* SAMPLER_H */
