/*
 * sampler.h — Logit sampling methods for SUB-AI inference engine.
 *
 * Provides:
 * - sample_argmax: Greedy deterministic sampling (temperature = 0)
 * - sample_topk: Top-K temperature-scaled multinomial sampling
 */

#ifndef SAMPLER_H
#define SAMPLER_H

int sample_argmax(const float *logits, int vocab_size);
int sample_topk(float *logits, int vocab_size, float temperature, int top_k);

#endif /* SAMPLER_H */
