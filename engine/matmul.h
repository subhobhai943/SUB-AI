/*
 * matmul.h — Core math kernels for SUB-AI C inference engine.
 *
 * Provides CPU implementations of:
 * - matmul: Vector-matrix multiplication x @ W (TensorFlow layout: [in] @ [in, out] -> [out])
 * - softmax: Numerically stable in-place softmax
 * - gelu: Gaussian Error Linear Unit (tanh approximation)
 * - layernorm: Standard layer normalization with gamma and beta
 */

#ifndef MATMUL_H
#define MATMUL_H

void  matmul(float *out, const float *x, const float *W, int in_dim, int out_dim);
void  softmax(float *x, int n);
float gelu(float x);
void  layernorm(float *out, const float *x, const float *gamma, const float *beta, int n);

#endif /* MATMUL_H */
