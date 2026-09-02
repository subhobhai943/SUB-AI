/*
 * matmul.c — Core math kernels implementation for SUB-AI C inference engine.
 *
 * Implements vector-matrix multiplication, numerically stable softmax,
 * GELU activation function, and Layer Normalization.
 */

#include "matmul.h"
#include <math.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/*
 * matmul: computes out = x @ W
 *   x:      [in_dim]
 *   W:      [in_dim, out_dim] stored row-major (TF kernel layout)
 *   out:    [out_dim]
 */
void matmul(float *out, const float *x, const float *W, int in_dim, int out_dim) {
    for (int j = 0; j < out_dim; j++) {
        out[j] = 0.0f;
    }
    for (int i = 0; i < in_dim; i++) {
        float xi = x[i];
        const float *w_row = &W[(size_t)i * out_dim];
        for (int j = 0; j < out_dim; j++) {
            out[j] += xi * w_row[j];
        }
    }
}

/*
 * softmax: computes in-place numerically stable softmax over x[0..n-1]
 */
void softmax(float *x, int n) {
    if (n <= 0) return;
    float max_val = x[0];
    for (int i = 1; i < n; i++) {
        if (x[i] > max_val) {
            max_val = x[i];
        }
    }

    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }

    float inv_sum = (sum > 0.0f) ? (1.0f / sum) : 0.0f;
    for (int i = 0; i < n; i++) {
        x[i] *= inv_sum;
    }
}

/*
 * gelu: Gaussian Error Linear Unit (approximate tanh formulation)
 * Formula: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
 */
float gelu(float x) {
    const float k_sqrt_2_over_pi = 0.7978845608028654f; /* sqrt(2 / pi) */
    const float k_coeff = 0.044715f;
    float x_cubed = x * x * x;
    float inner = k_sqrt_2_over_pi * (x + k_coeff * x_cubed);
    return 0.5f * x * (1.0f + tanhf(inner));
}

/*
 * layernorm: standard layer normalization with learned scale (gamma) and shift (beta)
 * Formula: out = ((x - mean) / sqrt(var + eps)) * gamma + beta
 */
void layernorm(float *out, const float *x, const float *gamma, const float *beta, int n) {
    if (n <= 0) return;
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        sum += x[i];
    }
    float mean = sum / (float)n;

    float var_sum = 0.0f;
    for (int i = 0; i < n; i++) {
        float diff = x[i] - mean;
        var_sum += diff * diff;
    }
    float var = var_sum / (float)n;
    const float eps = 1e-5f;
    float inv_std = 1.0f / sqrtf(var + eps);

    for (int i = 0; i < n; i++) {
        float norm = (x[i] - mean) * inv_std;
        out[i] = norm * (gamma ? gamma[i] : 1.0f) + (beta ? beta[i] : 0.0f);
    }
}
