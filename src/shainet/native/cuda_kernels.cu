#include <cstdio>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <curand_kernel.h>
#include <math.h>

template <typename T> struct Convert;

template <> struct Convert<__half> {
  __device__ static float to_float(__half v) { return __half2float(v); }
  __device__ static __half from_float(float v) { return __float2half(v); }
};

template <> struct Convert<__nv_bfloat16> {
  __device__ static float to_float(__nv_bfloat16 v) {
    return __bfloat162float(v);
  }
  __device__ static __nv_bfloat16 from_float(float v) {
    return __float2bfloat16(v);
  }
};

template <> struct Convert<float> {
  __device__ static float to_float(float v) { return v; }
  __device__ static float from_float(float v) { return v; }
};


template <typename T>
__global__ void scale_kernel_t(T *data, float alpha, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= size)
    return;
  float v = Convert<T>::to_float(data[idx]);
  data[idx] = Convert<T>::from_float(v * alpha);
}

template <typename T>
__global__ void cross_entropy_loss_gradient_kernel_t(const T *pred, const T *target, T *grad, float *loss, int total) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total)
    return;

  T p = pred[idx];
  T t = target[idx];
  grad[idx] = p - t;

  float tp = Convert<T>::to_float(t);
  float pp = Convert<T>::to_float(p);
  float contrib = -tp * logf(fmaxf(pp, 1e-15f));
  atomicAdd(loss, contrib);
}

template <typename T>
void cross_entropy_loss_gradient_t(const T *pred, const T *target, T *grad,
                                   float *loss, int rows, int cols) {
  int total = rows * cols;
  cudaMemset(loss, 0, sizeof(float));
  int threads = 256;
  int blocks = (total + threads - 1) / threads;

  cross_entropy_loss_gradient_kernel_t<<<blocks, threads>>>(pred, target, grad,
                                                            loss, total);
  cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    printf("CUDA Error in cross_entropy_loss_gradient: %s\n",
           cudaGetErrorString(err));
  }
}

template <typename T>
__global__ void add_bias_kernel_t(T *mat, const T *bias, int rows, int cols) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= rows * cols)
    return;
  int col = idx % cols;
  mat[idx] = Convert<T>::from_float(Convert<T>::to_float(mat[idx]) +
                                    Convert<T>::to_float(bias[col]));
}

template <typename T>
__global__ void mse_loss_grad_kernel(const T *pred, const T *target, T *grad,
                                     float *loss, int total) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total)
    return;

  T p = pred[idx];
  T t = target[idx];
  T diff = p - t;
  grad[idx] = diff;
  float contrib = 0.5f * Convert<T>::to_float(diff) *
                  Convert<T>::to_float(diff);
  atomicAdd(loss, contrib);
}

template <typename T>
void mse_loss_gradient_t(const T *pred, const T *target, T *grad, float *loss,
                         int rows, int cols) {
  int total = rows * cols;
  cudaMemset(loss, 0, sizeof(float));
  int threads = 256;
  int blocks = (total + threads - 1) / threads;
  mse_loss_grad_kernel<<<blocks, threads>>>(pred, target, grad, loss, total);
  cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    printf("CUDA Error in mse_loss_gradient: %s\n", cudaGetErrorString(err));
  }
}

template <typename T>
__global__ void softmax_rows_kernel_t(T *out, const T *in, int rows, int cols) {
  int row = blockIdx.x;
  if (row >= rows)
    return;

  const T *row_in = in + row * cols;
  T *row_out = out + row * cols;

  float max_val = Convert<T>::to_float(row_in[0]);
  for (int j = 1; j < cols; ++j) {
    float v = Convert<T>::to_float(row_in[j]);
    if (v > max_val)
      max_val = v;
  }

  float sum = 0.0f;
  for (int j = 0; j < cols; ++j) {
    float e = expf(Convert<T>::to_float(row_in[j]) - max_val);
    row_out[j] = Convert<T>::from_float(e);
    sum += e;
  }

  for (int j = 0; j < cols; ++j) {
    float val = Convert<T>::to_float(row_out[j]) / sum;
    row_out[j] = Convert<T>::from_float(val);
  }
}

template <typename T>
__global__ void relu_backward_kernel_t(T *output, const T *input, const T *grad,
                                       int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= size)
    return;

  float in_val = Convert<T>::to_float(input[idx]);
  output[idx] = in_val > 0.0f ? grad[idx] : Convert<T>::from_float(0.0f);
}

template <typename T>
__global__ void dropout_kernel_t(T *out, const T *in, int rows, int cols,
                                 float drop_p, unsigned long long seed) {
  int row = blockIdx.x;
  if (row >= rows)
    return;
  curandState state;
  curand_init(seed + row, 0, 0, &state);
  const T *row_in = in + row * cols;
  T *row_out = out + row * cols;
  for (int j = 0; j < cols; ++j) {
    float r = curand_uniform(&state);
    row_out[j] = r < drop_p ? Convert<T>::from_float(0.0f) : row_in[j];
  }
}

template <typename T>
__global__ void gather_rows_kernel_t(T *out, const T *in, const int *ids,
                                     int rows, int cols) {
  int row = blockIdx.x;
  if (row >= rows)
    return;
  int id = ids[row];
  const T *row_in = in + id * cols;
  T *row_out = out + row * cols;
  for (int j = 0; j < cols; ++j) {
    row_out[j] = row_in[j];
  }
}

template <typename T>
__global__ void row_mean_var_kernel_t(const T *in, float *mean, float *var,
                                      int rows, int cols) {
  int row = blockIdx.x;
  if (row >= rows)
    return;
  const T *row_in = in + row * cols;
  float sum = 0.0f;
  float sq_sum = 0.0f;
  for (int j = 0; j < cols; ++j) {
    float v = Convert<T>::to_float(row_in[j]);
    sum += v;
    sq_sum += v * v;
  }
  float m = sum / cols;
  mean[row] = m;
  var[row] = sq_sum / cols - m * m;
}

template <typename T>
__global__ void transpose_kernel_t(T *out, const T *in, int rows, int cols) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= rows * cols)
    return;

  int row = idx / cols;
  int col = idx % cols;

  out[col * rows + row] = in[row * cols + col];
}

template <typename T>
__global__ void apply_layer_norm_kernel_t(T *out, const T *in,
                                          const float *mean, const float *var,
                                          int rows, int cols, float epsilon) {
  int row = blockIdx.x;
  if (row >= rows)
    return;
  const T *row_in = in + row * cols;
  T *row_out = out + row * cols;
  float m = mean[row];
  float denom = sqrtf(var[row] + epsilon);
  for (int j = 0; j < cols; ++j) {
    float v = Convert<T>::to_float(row_in[j]);
    row_out[j] = Convert<T>::from_float((v - m) / denom);
  }
}

// Generic kernel for other precisions
template <typename T>
__global__ void zero_matrix_kernel_t(T *matrix, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= size)
    return;

  matrix[idx] = Convert<T>::from_float(0.0f);
}

template <typename T>
__global__ void element_div_kernel_t(T *out, const T *a, const T *b, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= size)
    return;

  float denom = Convert<T>::to_float(b[idx]);
  float numer = Convert<T>::to_float(a[idx]);
  float res = denom == 0.0f ? 0.0f : numer / denom;
  out[idx] = Convert<T>::from_float(res);
}

template <typename T>
__global__ void element_mul_kernel_t(T *out, const T *a, const T *b,
                                     float alpha, float beta, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= size)
    return;

  float av = Convert<T>::to_float(a[idx]);
  float bv = Convert<T>::to_float(b[idx]);
  float ov = Convert<T>::to_float(out[idx]);
  out[idx] = Convert<T>::from_float(alpha * av * bv + beta * ov);
}

template <typename T>
__global__ void ger_kernel_t(const T *x, const T *y, T *a,
                             int m, int n, int lda, float alpha) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  if (row >= m || col >= n)
    return;

  float xv = Convert<T>::to_float(x[row]);
  float yv = Convert<T>::to_float(y[col]);
  int idx = row * lda + col;
  float av = Convert<T>::to_float(a[idx]);
  a[idx] = Convert<T>::from_float(av + alpha * xv * yv);
}

template <typename T>
void ger_t(const T *x, const T *y, T *a, int m, int n, int lda, float alpha) {
  dim3 block(16, 16);
  dim3 grid((n + block.x - 1) / block.x, (m + block.y - 1) / block.y);
  ger_kernel_t<<<grid, block>>>(x, y, a, m, n, lda, alpha);
  cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    printf("CUDA Error in ger kernel: %s\n", cudaGetErrorString(err));
  }
}

template <typename T>
__global__ void sum_cols_kernel_t(T *out, const T *in, int rows, int cols) {
  int col = blockIdx.x;
  if (col >= cols)
    return;
  float sum = 0.0f;
  for (int i = 0; i < rows; ++i) {
    sum += Convert<T>::to_float(in[i * cols + col]);
  }
  out[col] = Convert<T>::from_float(sum);
}

template <typename T>
__global__ void
softmax_cross_entropy_label_kernel_t(const T *pred, const int *labels, T *grad,
                                     float *loss, int rows, int cols) {
  int row = blockIdx.x;
  if (row >= rows)
    return;

  const T *row_pred = pred + row * cols;
  T *row_grad = grad + row * cols;

  // Find maximum value in the row for numerical stability
  float max_val = Convert<T>::to_float(row_pred[0]);
  for (int j = 1; j < cols; ++j) {
    float v = Convert<T>::to_float(row_pred[j]);
    if (v > max_val)
      max_val = v;
  }

  // Compute exponentials and their sum
  float sum = 0.0f;
  for (int j = 0; j < cols; ++j) {
    float e = expf(Convert<T>::to_float(row_pred[j]) - max_val);
    row_grad[j] = Convert<T>::from_float(e);
    sum += e;
  }

  // Normalize to obtain probabilities
  for (int j = 0; j < cols; ++j) {
    row_grad[j] = Convert<T>::from_float(
        Convert<T>::to_float(row_grad[j]) / sum);
  }

  int label = labels[row];
  if (label >= 0 && label < cols) {
    float p = Convert<T>::to_float(row_grad[label]);
    row_grad[label] = Convert<T>::from_float(p - 1.0f);
    float contrib = -logf(fmaxf(p, 1e-15f));
    atomicAdd(loss, contrib);
  }
}

template <typename T>
void softmax_cross_entropy_label_t(const T *pred, const int *labels, T *grad,
                                   float *loss, int rows, int cols) {
  cudaMemset(loss, 0, sizeof(float));
  softmax_cross_entropy_label_kernel_t<<<rows, 1>>>(pred, labels, grad, loss,
                                                    rows, cols);
  cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    printf("CUDA Error in softmax_cross_entropy_label: %s\n",
           cudaGetErrorString(err));
  }
}

template <typename T>
__global__ void softmax_cross_entropy_label_matrix_kernel_t(
    const T *pred, const T *labels, T *grad, float *loss, int rows, int cols) {
  int row = blockIdx.x;
  if (row >= rows)
    return;

  const T *row_pred = pred + row * cols;
  T *row_grad = grad + row * cols;

  float max_val = Convert<T>::to_float(row_pred[0]);
  for (int j = 1; j < cols; ++j) {
    float v = Convert<T>::to_float(row_pred[j]);
    if (v > max_val)
      max_val = v;
  }

  float sum = 0.0f;
  for (int j = 0; j < cols; ++j) {
    float e = expf(Convert<T>::to_float(row_pred[j]) - max_val);
    row_grad[j] = Convert<T>::from_float(e);
    sum += e;
  }

  for (int j = 0; j < cols; ++j) {
    row_grad[j] = Convert<T>::from_float(
        Convert<T>::to_float(row_grad[j]) / sum);
  }

  int label = (int)labels[row];
  if (label >= 0 && label < cols) {
    float p = Convert<T>::to_float(row_grad[label]);
    row_grad[label] = Convert<T>::from_float(p - 1.0f);
    float contrib = -logf(fmaxf(p, 1e-15f));
    atomicAdd(loss, contrib);
  }
}

template <typename T>
void softmax_cross_entropy_label_matrix_t(const T *pred, const T *labels,
                                          T *grad, float *loss, int rows,
                                          int cols) {
  cudaMemset(loss, 0, sizeof(float));
  softmax_cross_entropy_label_matrix_kernel_t<<<rows, 1>>>(pred, labels, grad,
                                                           loss, rows, cols);
  cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    printf("CUDA Error in softmax_cross_entropy_label_matrix: %s\n",
           cudaGetErrorString(err));
  }
}

template <typename T>
__global__ void row_sum_kernel_t(T *dst, const T *src, int rows, int cols) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (col >= cols)
    return;

  float sum = 0.0f;
  for (int row = 0; row < rows; ++row) {
    sum += Convert<T>::to_float(src[row * cols + col]);
  }
  float prev = Convert<T>::to_float(dst[col]);
  dst[col] = Convert<T>::from_float(prev + sum);
}

template <typename T>
__global__ void slice_cols_kernel_t(T *out, const T *in, int rows,
                                    int src_cols, int start, int len) {
  int row = blockIdx.x;
  int col = threadIdx.x;
  if (row >= rows || col >= len)
    return;
  out[row * len + col] = in[row * src_cols + start + col];
}

template <typename T>
__global__ void swiglu_backward_kernel_t(T *dest, const T *pre, const T *grad,
                                         int rows, int cols_half) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = rows * cols_half;
  if (idx >= total)
    return;

  int row = idx / cols_half;
  int col = idx % cols_half;
  int cols = cols_half * 2;

  const T *pre_row = pre + row * cols;
  T *dest_row = dest + row * cols;

  float a = Convert<T>::to_float(pre_row[col]);
  float b = Convert<T>::to_float(pre_row[col + cols_half]);
  float g = Convert<T>::to_float(grad[row * cols_half + col]);
  float sig = 1.0f / (1.0f + expf(-b));
  float sig_p = sig * (1.0f - sig);

  dest_row[col] = Convert<T>::from_float(g * sig);
  dest_row[col + cols_half] = Convert<T>::from_float(g * a * sig_p);
}

template <typename T>
__global__ void set_cols_kernel_t(T *out, const T *in, int rows,
                                  int dst_cols, int start, int len) {
  int row = blockIdx.x;
  int col = threadIdx.x;
  if (row >= rows || col >= len)
    return;
  out[row * dst_cols + start + col] = in[row * len + col];
}

// Host wrapper functions
extern "C" {
void softmax_rows(float *out, const float *in, int rows, int cols) {
  softmax_rows_kernel_t<float><<<rows, 1>>>(out, in, rows, cols);
  cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    printf("CUDA Error in softmax_rows: %s\n", cudaGetErrorString(err));
  }
}

void softmax_rows_fp16(__half *out, const __half *in, int rows, int cols) {
  softmax_rows_kernel_t<<<rows, 1>>>(out, in, rows, cols);
  cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    printf("CUDA Error in softmax_rows_fp16: %s\n", cudaGetErrorString(err));
  }
}

void softmax_rows_bf16(__nv_bfloat16 *out, const __nv_bfloat16 *in, int rows,
                       int cols) {
  softmax_rows_kernel_t<<<rows, 1>>>(out, in, rows, cols);
  cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    printf("CUDA Error in softmax_rows_bf16: %s\n", cudaGetErrorString(err));
  }
}

void softmax_rows_f32(float *out, const float *in, int rows, int cols) {
  softmax_rows_kernel_t<<<rows, 1>>>(out, in, rows, cols);
  cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    printf("CUDA Error in softmax_rows_f32: %s\n", cudaGetErrorString(err));
  }
}

void relu_backward(float *output, const float *input, const float *grad,
                   int size) {
  int threads_per_block = 256;
  int blocks = (size + threads_per_block - 1) / threads_per_block;

  relu_backward_kernel_t<float>
      <<<blocks, threads_per_block>>>(output, input, grad, size);
  cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    printf("CUDA Error in relu_backward: %s\n", cudaGetErrorString(err));
  }
}

void swiglu_backward(float *dest, const float *pre, const float *grad, int rows,
                     int cols_half) {
  int threads_per_block = 256;
  int total = rows * cols_half;
  int blocks = (total + threads_per_block - 1) / threads_per_block;

  swiglu_backward_kernel_t<float>
      <<<blocks, threads_per_block>>>(dest, pre, grad, rows, cols_half);
  cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    printf("CUDA Error in swiglu_backward: %s\n", cudaGetErrorString(err));
  }
}

void dropout(float *out, const float *in, int rows, int cols, float drop_p,
             unsigned long long seed) {
  dropout_kernel_t<float><<<rows, 1>>>(out, in, rows, cols, drop_p, seed);
  cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    printf("CUDA Error in dropout: %s\n", cudaGetErrorString(err));
  }
}

void dropout_fp16(__half *out, const __half *in, int rows, int cols,
                  float drop_p, unsigned long long seed) {
  dropout_kernel_t<<<rows, 1>>>(out, in, rows, cols, drop_p, seed);
  cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    printf("CUDA Error in dropout_fp16: %s\n", cudaGetErrorString(err));
  }
}

void dropout_bf16(__nv_bfloat16 *out, const __nv_bfloat16 *in, int rows,
                  int cols, float drop_p, unsigned long long seed) {
  dropout_kernel_t<<<rows, 1>>>(out, in, rows, cols, drop_p, seed);
  cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    printf("CUDA Error in dropout_bf16: %s\n", cudaGetErrorString(err));
  }
}

void dropout_f32(float *out, const float *in, int rows, int cols, float drop_p,
                 unsigned long long seed) {
  dropout_kernel_t<<<rows, 1>>>(out, in, rows, cols, drop_p, seed);
  cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    printf("CUDA Error in dropout_f32: %s\n", cudaGetErrorString(err));
  }
}

void gather_rows(float *out, const float *in, const int *ids, int rows,
                 int cols) {
  gather_rows_kernel_t<float><<<rows, 1>>>(out, in, ids, rows, cols);
  cudaDeviceSynchronize();
}

void gather_rows_fp16(__half *out, const __half *in, const int *ids, int rows,
                      int cols) {
  gather_rows_kernel_t<<<rows, 1>>>(out, in, ids, rows, cols);
  cudaDeviceSynchronize();
}

void gather_rows_bf16(__nv_bfloat16 *out, const __nv_bfloat16 *in,
                      const int *ids, int rows, int cols) {
  gather_rows_kernel_t<<<rows, 1>>>(out, in, ids, rows, cols);
  cudaDeviceSynchronize();
}

void row_mean_var(const float *in, float *mean, float *var, int rows,
                  int cols) {
  row_mean_var_kernel_t<float><<<rows, 1>>>(in, mean, var, rows, cols);
  cudaDeviceSynchronize();
}

void row_mean_var_fp16(const __half *in, float *mean, float *var, int rows,
                       int cols) {
  row_mean_var_kernel_t<<<rows, 1>>>(in, mean, var, rows, cols);
  cudaDeviceSynchronize();
}

void row_mean_var_bf16(const __nv_bfloat16 *in, float *mean, float *var,
                       int rows, int cols) {
  row_mean_var_kernel_t<<<rows, 1>>>(in, mean, var, rows, cols);
  cudaDeviceSynchronize();
}

void row_mean_var_f32(const float *in, float *mean, float *var, int rows,
                      int cols) {
  row_mean_var_kernel_t<<<rows, 1>>>(in, mean, var, rows, cols);
  cudaDeviceSynchronize();
}

void apply_layer_norm(float *out, const float *in, const float *mean,
                      const float *var, int rows, int cols, float epsilon) {
  apply_layer_norm_kernel_t<float><<<rows, 1>>>(out, in, mean, var, rows, cols,
                                               epsilon);
  cudaDeviceSynchronize();
}

void apply_layer_norm_fp16(__half *out, const __half *in, const float *mean,
                           const float *var, int rows, int cols,
                           float epsilon) {
  apply_layer_norm_kernel_t<<<rows, 1>>>(out, in, mean, var, rows, cols,
                                         epsilon);
  cudaDeviceSynchronize();
}

void apply_layer_norm_bf16(__nv_bfloat16 *out, const __nv_bfloat16 *in,
                           const float *mean, const float *var, int rows,
                           int cols, float epsilon) {
  apply_layer_norm_kernel_t<<<rows, 1>>>(out, in, mean, var, rows, cols,
                                         epsilon);
  cudaDeviceSynchronize();
}

void apply_layer_norm_f32(float *out, const float *in, const float *mean,
                          const float *var, int rows, int cols, float epsilon) {
  apply_layer_norm_kernel_t<<<rows, 1>>>(out, in, mean, var, rows, cols,
                                         epsilon);
  cudaDeviceSynchronize();
}


void slice_cols(float *out, const float *in, int rows, int src_cols,
                int start, int len) {
  slice_cols_kernel_t<float><<<rows, len>>>(out, in, rows, src_cols, start, len);
  cudaDeviceSynchronize();
}

void slice_cols_fp16(__half *out, const __half *in, int rows, int src_cols,
                     int start, int len) {
  slice_cols_kernel_t<<<rows, len>>>(out, in, rows, src_cols, start, len);
  cudaDeviceSynchronize();
}

void slice_cols_bf16(__nv_bfloat16 *out, const __nv_bfloat16 *in, int rows,
                     int src_cols, int start, int len) {
  slice_cols_kernel_t<<<rows, len>>>(out, in, rows, src_cols, start, len);
  cudaDeviceSynchronize();
}


void set_cols(float *out, const float *in, int rows, int dst_cols, int start,
              int len) {
  set_cols_kernel_t<float><<<rows, len>>>(out, in, rows, dst_cols, start, len);
  cudaDeviceSynchronize();
}

void set_cols_fp16(__half *out, const __half *in, int rows, int dst_cols,
                   int start, int len) {
  set_cols_kernel_t<<<rows, len>>>(out, in, rows, dst_cols, start, len);
  cudaDeviceSynchronize();
}

void set_cols_bf16(__nv_bfloat16 *out, const __nv_bfloat16 *in, int rows,
                   int dst_cols, int start, int len) {
  set_cols_kernel_t<<<rows, len>>>(out, in, rows, dst_cols, start, len);
  cudaDeviceSynchronize();
}

__global__ void count_token_pairs_kernel(const int *a, const int *b,
                                         const int *freq, int pair_count,
                                         int vocab_size, int *counts) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= pair_count)
    return;
  int offset = a[idx] * vocab_size + b[idx];
  atomicAdd(&counts[offset], freq[idx]);
}

void count_token_pairs(const int *a, const int *b, const int *freq,
                       int pair_count, int vocab_size, int *counts) {
  int blocks = (pair_count + 255) / 256;
  count_token_pairs_kernel<<<blocks, 256>>>(a, b, freq, pair_count, vocab_size,
                                            counts);
  cudaDeviceSynchronize();
}

__global__ void layer_norm_backward_kernel(
    float *d_x, float *d_gamma, float *d_beta, const float *d_out,
    const float *x, const float *gamma, const float *mean, const float *var,
    const float *norm, int rows, int cols, float epsilon) {
  int row = blockIdx.x;
  if (row >= rows)
    return;

  const float *x_row = x + row * cols;
  const float *dout_row = d_out + row * cols;
  const float *norm_row = norm + row * cols;
  float *dx_row = d_x + row * cols;

  float m = mean[row];
  float v = var[row];
  float denom = sqrtf(v + epsilon);
  float inv = 1.0f / denom;
  float col_f = (float)cols;

  // Compute sum_dout_gamma and sum_dout_gamma_norm
  float sum_dout_gamma = 0.0f;
  float sum_dout_gamma_norm = 0.0f;
  for (int j = 0; j < cols; ++j) {
    float doutg = dout_row[j] * gamma[j];
    sum_dout_gamma += doutg;
    sum_dout_gamma_norm += doutg * (x_row[j] - m);

    // Accumulate gradients for gamma and beta
    atomicAdd(&d_gamma[j], dout_row[j] * norm_row[j]);
    atomicAdd(&d_beta[j], dout_row[j]);
  }

  // Compute d_x
  for (int j = 0; j < cols; ++j) {
    float xm = x_row[j] - m;
    float doutg = dout_row[j] * gamma[j];
    dx_row[j] = inv * (doutg - sum_dout_gamma / col_f -
                       xm * inv * inv / col_f * sum_dout_gamma_norm);
  }
}

void layer_norm_backward(float *d_x, float *d_gamma, float *d_beta,
                         const float *d_out, const float *x,
                         const float *gamma, const float *mean,
                         const float *var, const float *norm, int rows,
                         int cols, float epsilon) {
  layer_norm_backward_kernel<<<rows, 1>>>(d_x, d_gamma, d_beta, d_out, x, gamma,
                                          mean, var, norm, rows, cols, epsilon);
  cudaDeviceSynchronize();
}

void sum_cols(float *out, const float *in, int rows, int cols) {
  sum_cols_kernel_t<float><<<cols, 1>>>(out, in, rows, cols);
  cudaDeviceSynchronize();
}

void sum_cols_fp16(__half *out, const __half *in, int rows, int cols) {
  sum_cols_kernel_t<<<cols, 1>>>(out, in, rows, cols);
  cudaDeviceSynchronize();
}

void sum_cols_bf16(__nv_bfloat16 *out, const __nv_bfloat16 *in, int rows,
                   int cols) {
  sum_cols_kernel_t<<<cols, 1>>>(out, in, rows, cols);
  cudaDeviceSynchronize();
}

void sum_cols_f32(float *out, const float *in, int rows, int cols) {
  sum_cols_kernel_t<<<cols, 1>>>(out, in, rows, cols);
  cudaDeviceSynchronize();
}

__global__ void mul_row_vector_kernel(float *matrix, const float *vec,
                                      int rows, int cols) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= rows * cols)
    return;

  int col = idx % cols;
  matrix[idx] *= vec[col];
}

void mul_row_vector(float *matrix, const float *vec, int rows, int cols) {
  int threads_per_block = 256;
  int blocks = (rows * cols + threads_per_block - 1) / threads_per_block;

  mul_row_vector_kernel<<<blocks, threads_per_block>>>(matrix, vec, rows, cols);
  cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    printf("CUDA Error in mul_row_vector: %s\n", cudaGetErrorString(err));
  }
}

void transpose(float *out, const float *in, int rows, int cols) {
  int threads_per_block = 256;
  int blocks = (rows * cols + threads_per_block - 1) / threads_per_block;

  transpose_kernel_t<float><<<blocks, threads_per_block>>>(out, in, rows, cols);
  cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    printf("CUDA Error in transpose: %s\n", cudaGetErrorString(err));
  }
}

void transpose_fp32(float *out, const float *in, int rows, int cols) {
  int threads_per_block = 256;
  int blocks = (rows * cols + threads_per_block - 1) / threads_per_block;

  transpose_kernel_t<<<blocks, threads_per_block>>>(out, in, rows, cols);
  cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    printf("CUDA Error in transpose_fp32: %s\n", cudaGetErrorString(err));
  }
}

void transpose_fp16(__half *out, const __half *in, int rows, int cols) {
  int threads_per_block = 256;
  int blocks = (rows * cols + threads_per_block - 1) / threads_per_block;

  transpose_kernel_t<<<blocks, threads_per_block>>>(out, in, rows, cols);
  cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    printf("CUDA Error in transpose_fp16: %s\n", cudaGetErrorString(err));
  }
}

void transpose_bf16(__nv_bfloat16 *out, const __nv_bfloat16 *in, int rows,
                    int cols) {
  int threads_per_block = 256;
  int blocks = (rows * cols + threads_per_block - 1) / threads_per_block;

  transpose_kernel_t<<<blocks, threads_per_block>>>(out, in, rows, cols);
  cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    printf("CUDA Error in transpose_bf16: %s\n", cudaGetErrorString(err));
  }
}

__global__ void sigmoid_forward_kernel(float *activations, float *derivatives,
                                       const float *linear, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= size)
    return;

  float val = linear[idx];
  float exp_neg_val = expf(-val);
  float sigmoid_val = 1.0f / (1.0f + exp_neg_val);

  activations[idx] = sigmoid_val;
  // Sigmoid derivative: σ(x) * (1 - σ(x))
  derivatives[idx] = sigmoid_val * (1.0f - sigmoid_val);
}

__global__ void gelu_forward_kernel(float *activations, float *derivatives,
                                    const float *linear, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= size)
    return;

  float x = linear[idx];
  float cdf = 0.5f * erfcf(-x / sqrtf(2.0f));
  activations[idx] = x * cdf;
  derivatives[idx] = cdf + x * expf(-0.5f * x * x) / sqrtf(2.0f * M_PI);
}

void gelu_forward(float *activations, float *derivatives,
                  const float *linear, int size) {
  int threads_per_block = 256;
  int blocks = (size + threads_per_block - 1) / threads_per_block;

  gelu_forward_kernel<<<blocks, threads_per_block>>>(activations, derivatives,
                                                     linear, size);
  cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    printf("CUDA Error in gelu_forward: %s\n", cudaGetErrorString(err));
  }
}

__global__ void sigmoid_forward_kernel_f32(float *activations,
                                           float *derivatives,
                                           const float *linear, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= size)
    return;

  float val = linear[idx];
  float exp_neg_val = expf(-val);
  float sigmoid_val = 1.0f / (1.0f + exp_neg_val);

  activations[idx] = sigmoid_val;
  derivatives[idx] = sigmoid_val * (1.0f - sigmoid_val);
}

__global__ void gelu_forward_kernel_f32(float *activations, float *derivatives,
                                        const float *linear, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= size)
    return;

  float x = linear[idx];
  // Use erfcf for better parity with double precision and CPU implementation
  float cdf = 0.5f * erfcf(-x / sqrtf(2.0f));
  activations[idx] = x * cdf;
  derivatives[idx] = cdf + x * expf(-0.5f * x * x) / sqrtf(2.0f * M_PI);
}

void gelu_forward_f32(float *activations, float *derivatives,
                      const float *linear, int size) {
  int threads_per_block = 256;
  int blocks = (size + threads_per_block - 1) / threads_per_block;

  gelu_forward_kernel_f32<<<blocks, threads_per_block>>>(
      activations, derivatives, linear, size);
  cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    printf("CUDA Error in gelu_forward_f32: %s\n", cudaGetErrorString(err));
  }
}

void sigmoid_forward_f32(float *activations, float *derivatives,
                         const float *linear, int size) {
  int threads_per_block = 256;
  int blocks = (size + threads_per_block - 1) / threads_per_block;

  sigmoid_forward_kernel_f32<<<blocks, threads_per_block>>>(
      activations, derivatives, linear, size);
  cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    printf("CUDA Error in sigmoid_forward_f32: %s\n", cudaGetErrorString(err));
  }
}

void sigmoid_forward(float *activations, float *derivatives,
                     const float *linear, int size) {
  int threads_per_block = 256;
  int blocks = (size + threads_per_block - 1) / threads_per_block;

  sigmoid_forward_kernel<<<blocks, threads_per_block>>>(
      activations, derivatives, linear, size);
  cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    printf("CUDA Error in sigmoid_forward: %s\n", cudaGetErrorString(err));
  }
}

__global__ void apply_gradient_kernel(float *local_grad, const float *grad,
                                      const float *derivatives, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= size)
    return;

  local_grad[idx] = grad[idx] * derivatives[idx];
}

void apply_gradient(float *local_grad, const float *grad,
                    const float *derivatives, int size) {
  int threads_per_block = 256;
  int blocks = (size + threads_per_block - 1) / threads_per_block;

  apply_gradient_kernel<<<blocks, threads_per_block>>>(local_grad, grad,
                                                       derivatives, size);
  cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    printf("CUDA Error in apply_gradient: %s\n", cudaGetErrorString(err));
  }
}

__global__ void accumulate_bias_grad_kernel(float *bias_grad,
                                            const float *local_grad, int rows,
                                            int cols) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (col >= cols)
    return;

  float sum = 0.0f;
  for (int row = 0; row < rows; row++) {
    sum += local_grad[row * cols + col];
  }
  atomicAdd(&bias_grad[col], sum);
}

void accumulate_bias_grad(float *bias_grad, const float *local_grad, int rows,
                          int cols) {
  int threads_per_block = 256;
  int blocks = (cols + threads_per_block - 1) / threads_per_block;

  accumulate_bias_grad_kernel<<<blocks, threads_per_block>>>(
      bias_grad, local_grad, rows, cols);
  cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    printf("CUDA Error in accumulate_bias_grad: %s\n", cudaGetErrorString(err));
  }
}

void row_sum(float *dst, const float *src, int rows, int cols) {
  int threads_per_block = 256;
  int blocks = (cols + threads_per_block - 1) / threads_per_block;

  row_sum_kernel_t<float><<<blocks, threads_per_block>>>(dst, src, rows, cols);
  cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    printf("CUDA Error in row_sum: %s\n", cudaGetErrorString(err));
  }
}

void row_sum_fp16(__half *dst, const __half *src, int rows, int cols) {
  int threads_per_block = 256;
  int blocks = (cols + threads_per_block - 1) / threads_per_block;
  row_sum_kernel_t<<<blocks, threads_per_block>>>(dst, src, rows, cols);
  cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    printf("CUDA Error in row_sum_fp16: %s\n", cudaGetErrorString(err));
  }
}

void row_sum_bf16(__nv_bfloat16 *dst, const __nv_bfloat16 *src, int rows,
                  int cols) {
  int threads_per_block = 256;
  int blocks = (cols + threads_per_block - 1) / threads_per_block;
  row_sum_kernel_t<<<blocks, threads_per_block>>>(dst, src, rows, cols);
  cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    printf("CUDA Error in row_sum_bf16: %s\n", cudaGetErrorString(err));
  }
}

void row_sum_f32(float *dst, const float *src, int rows, int cols) {
  int threads_per_block = 256;
  int blocks = (cols + threads_per_block - 1) / threads_per_block;
  row_sum_kernel_t<<<blocks, threads_per_block>>>(dst, src, rows, cols);
  cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    printf("CUDA Error in row_sum_f32: %s\n", cudaGetErrorString(err));
  }
}

void add_bias_fp16(__half *mat, const __half *bias, int rows, int cols) {
  int threads_per_block = 256;
  int blocks = (rows * cols + threads_per_block - 1) / threads_per_block;
  add_bias_kernel_t<<<blocks, threads_per_block>>>(mat, bias, rows, cols);
  cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    printf("CUDA Error in add_bias_fp16: %s\n", cudaGetErrorString(err));
  }
}

void add_bias_bf16(__nv_bfloat16 *mat, const __nv_bfloat16 *bias, int rows,
                   int cols) {
  int threads_per_block = 256;
  int blocks = (rows * cols + threads_per_block - 1) / threads_per_block;
  add_bias_kernel_t<<<blocks, threads_per_block>>>(mat, bias, rows, cols);
  cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    printf("CUDA Error in add_bias_bf16: %s\n", cudaGetErrorString(err));
  }
}

void zero_matrix(float *matrix, int size) {
  int threads_per_block = 256;
  int blocks = (size + threads_per_block - 1) / threads_per_block;

  zero_matrix_kernel_t<float><<<blocks, threads_per_block>>>(matrix, size);
  cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    printf("CUDA Error in zero_matrix: %s\n", cudaGetErrorString(err));
  }
}

void zero_matrix_fp16(__half *matrix, int size) {
  int threads_per_block = 256;
  int blocks = (size + threads_per_block - 1) / threads_per_block;
  zero_matrix_kernel_t<<<blocks, threads_per_block>>>(matrix, size);
  cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    printf("CUDA Error in zero_matrix_fp16: %s\n", cudaGetErrorString(err));
  }
}

void zero_matrix_bf16(__nv_bfloat16 *matrix, int size) {
  int threads_per_block = 256;
  int blocks = (size + threads_per_block - 1) / threads_per_block;
  zero_matrix_kernel_t<<<blocks, threads_per_block>>>(matrix, size);
  cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    printf("CUDA Error in zero_matrix_bf16: %s\n", cudaGetErrorString(err));
  }
}

void zero_matrix_fp32(float *matrix, int size) {
  int threads_per_block = 256;
  int blocks = (size + threads_per_block - 1) / threads_per_block;
  zero_matrix_kernel_t<<<blocks, threads_per_block>>>(matrix, size);
  cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    printf("CUDA Error in zero_matrix_fp32: %s\n", cudaGetErrorString(err));
  }
}

__global__ void fill_matrix_kernel(float *matrix, float value, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= size)
    return;

  matrix[idx] = value;
}

void fill_matrix(float *matrix, float value, int size) {
  int threads_per_block = 256;
  int blocks = (size + threads_per_block - 1) / threads_per_block;

  fill_matrix_kernel<<<blocks, threads_per_block>>>(matrix, value, size);
  cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    printf("CUDA Error in fill_matrix: %s\n", cudaGetErrorString(err));
  }
}

__global__ void weight_update_fp16_kernel(__half *weights, const __half *grads,
                                          float lr, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= size)
    return;
  float w = __half2float(weights[idx]);
  float g = __half2float(grads[idx]);
  weights[idx] = __float2half(w + lr * g); // lr already has sign
}

void weight_update_fp16(__half *weights, const __half *grads, float lr,
                        int size) {
  int threads_per_block = 256;
  int blocks = (size + threads_per_block - 1) / threads_per_block;
  weight_update_fp16_kernel<<<blocks, threads_per_block>>>(weights, grads, lr,
                                                           size);
  cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    printf("CUDA Error in weight_update_fp16: %s\n", cudaGetErrorString(err));
  }
}

__global__ void weight_update_bf16_kernel(__nv_bfloat16 *weights,
                                          const __nv_bfloat16 *grads, float lr,
                                          int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= size)
    return;
  float w = __bfloat162float(weights[idx]);
  float g = __bfloat162float(grads[idx]);
  weights[idx] = __float2bfloat16(w + lr * g);
}

void weight_update_bf16(__nv_bfloat16 *weights, const __nv_bfloat16 *grads,
                        float lr, int size) {
  int threads_per_block = 256;
  int blocks = (size + threads_per_block - 1) / threads_per_block;
  weight_update_bf16_kernel<<<blocks, threads_per_block>>>(weights, grads, lr,
                                                           size);
  cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    printf("CUDA Error in weight_update_bf16: %s\n", cudaGetErrorString(err));
  }
}

__global__ void element_div_kernel(float *out, const float *a,
                                   const float *b, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= size)
    return;

  float denom = b[idx];
  out[idx] = denom == 0.0f ? 0.0f : a[idx] / denom;
}

void element_div(float *out, const float *a, const float *b, int size) {
  int threads_per_block = 256;
  int blocks = (size + threads_per_block - 1) / threads_per_block;

  element_div_kernel<<<blocks, threads_per_block>>>(out, a, b, size);
  cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    printf("CUDA Error in element_div: %s\n", cudaGetErrorString(err));
  }
}

void element_div_fp16(__half *out, const __half *a, const __half *b, int size) {
  int threads_per_block = 256;
  int blocks = (size + threads_per_block - 1) / threads_per_block;
  element_div_kernel_t<<<blocks, threads_per_block>>>(out, a, b, size);
  cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    printf("CUDA Error in element_div_fp16: %s\n", cudaGetErrorString(err));
  }
}

void element_div_bf16(__nv_bfloat16 *out, const __nv_bfloat16 *a,
                      const __nv_bfloat16 *b, int size) {
  int threads_per_block = 256;
  int blocks = (size + threads_per_block - 1) / threads_per_block;
  element_div_kernel_t<<<blocks, threads_per_block>>>(out, a, b, size);
  cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    printf("CUDA Error in element_div_bf16: %s\n", cudaGetErrorString(err));
  }
}

void element_div_f32(float *out, const float *a, const float *b, int size) {
  int threads_per_block = 256;
  int blocks = (size + threads_per_block - 1) / threads_per_block;
  element_div_kernel_t<<<blocks, threads_per_block>>>(out, a, b, size);
  cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    printf("CUDA Error in element_div_f32: %s\n", cudaGetErrorString(err));
  }
}

__global__ void element_mul_kernel(float *out, const float *a,
                                   const float *b, float alpha, float beta,
                                   int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= size)
    return;

  out[idx] = alpha * a[idx] * b[idx] + beta * out[idx];
}

void element_mul(float *out, const float *a, const float *b, float alpha,
                 float beta, int size) {
  int threads_per_block = 256;
  int blocks = (size + threads_per_block - 1) / threads_per_block;
  element_mul_kernel<<<blocks, threads_per_block>>>(out, a, b, alpha, beta,
                                                    size);
  cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    printf("CUDA Error in element_mul: %s\n", cudaGetErrorString(err));
  }
}

void element_mul_fp16(__half *out, const __half *a, const __half *b,
                      float alpha, float beta, int size) {
  int threads_per_block = 256;
  int blocks = (size + threads_per_block - 1) / threads_per_block;
  element_mul_kernel_t<<<blocks, threads_per_block>>>(out, a, b, (float)alpha,
                                                      (float)beta, size);
  cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    printf("CUDA Error in element_mul_fp16: %s\n", cudaGetErrorString(err));
  }
}

void element_mul_bf16(__nv_bfloat16 *out, const __nv_bfloat16 *a,
                      const __nv_bfloat16 *b, float alpha, float beta,
                      int size) {
  int threads_per_block = 256;
  int blocks = (size + threads_per_block - 1) / threads_per_block;
  element_mul_kernel_t<<<blocks, threads_per_block>>>(out, a, b, (float)alpha,
                                                      (float)beta, size);
  cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    printf("CUDA Error in element_mul_bf16: %s\n", cudaGetErrorString(err));
  }
}

void element_mul_f32(float *out, const float *a, const float *b, float alpha,
                     float beta, int size) {
  int threads_per_block = 256;
  int blocks = (size + threads_per_block - 1) / threads_per_block;
  element_mul_kernel_t<<<blocks, threads_per_block>>>(out, a, b, (float)alpha,
                                                      (float)beta, size);
  cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    printf("CUDA Error in element_mul_f32: %s\n", cudaGetErrorString(err));
  }
}

__global__ void softmax_backward_kernel(float *output, const float *grad,
                                        const float *softmax_out, int rows,
                                        int cols) {
  int row = blockIdx.x;
  if (row >= rows)
    return;

  const float *grad_row = grad + row * cols;
  const float *softmax_row = softmax_out + row * cols;
  float *output_row = output + row * cols;

  // Compute sum of softmax * grad for this row
  float sum = 0.0f;
  for (int j = 0; j < cols; j++) {
    sum += softmax_row[j] * grad_row[j];
  }

  // Compute softmax backward: softmax * (grad - sum)
  for (int j = 0; j < cols; j++) {
    output_row[j] = softmax_row[j] * (grad_row[j] - sum);
  }
}

void softmax_backward(float *output, const float *grad,
                      const float *softmax_out, int rows, int cols) {
  softmax_backward_kernel<<<rows, 1>>>(output, grad, softmax_out, rows, cols);
  cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    printf("CUDA Error in softmax_backward: %s\n", cudaGetErrorString(err));
  }
}

__global__ void element_log_kernel(float *out, const float *in, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= size)
    return;

  float val = in[idx];
  out[idx] = logf(val);
}

void element_log(float *out, const float *in, int size) {
  int threads_per_block = 256;
  int blocks = (size + threads_per_block - 1) / threads_per_block;

  element_log_kernel<<<blocks, threads_per_block>>>(out, in, size);
  cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    printf("CUDA Error in element_log: %s\n", cudaGetErrorString(err));
  }
}

void scale_fp16(__half *data, float alpha, int size) {
  int threads = 256;
  int blocks = (size + threads - 1) / threads;
  scale_kernel_t<<<blocks, threads>>>(data, alpha, size);
  cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    printf("CUDA Error in scale_fp16: %s\n", cudaGetErrorString(err));
  }
}

void scale_bf16(__nv_bfloat16 *data, float alpha, int size) {
  int threads = 256;
  int blocks = (size + threads - 1) / threads;
  scale_kernel_t<<<blocks, threads>>>(data, alpha, size);
  cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    printf("CUDA Error in scale_bf16: %s\n", cudaGetErrorString(err));
  }
}

void ger_fp16(const __half *x, const __half *y, __half *a,
              int m, int n, int lda, float alpha) {
  ger_t<__half>(x, y, a, m, n, lda, alpha);
}

void ger_bf16(const __nv_bfloat16 *x, const __nv_bfloat16 *y, __nv_bfloat16 *a,
               int m, int n, int lda, float alpha) {
  ger_t<__nv_bfloat16>(x, y, a, m, n, lda, alpha);
}

// Public C API wrappers
void cross_entropy_loss_gradient(float *pred, float *target, float *grad,
                                 float *loss, int rows, int cols) {
  cross_entropy_loss_gradient_t<float>(pred, target, grad, loss, rows, cols);
}

void cross_entropy_loss_gradient_f32(float *pred, float *target, float *grad,
                                     float *loss, int rows, int cols) {
  cross_entropy_loss_gradient_t<float>(pred, target, grad, loss, rows, cols);
}

void cross_entropy_loss_gradient_fp16(__half *pred, __half *target, __half *grad,
                                      float *loss, int rows, int cols) {
  cross_entropy_loss_gradient_t<__half>(pred, target, grad, loss, rows, cols);
}

void cross_entropy_loss_gradient_bf16(__nv_bfloat16 *pred, const __nv_bfloat16 *target,
                                      __nv_bfloat16 *grad, float *loss, int rows,
                                      int cols) {
  cross_entropy_loss_gradient_t<__nv_bfloat16>(pred, target, grad, loss, rows,
                                               cols);
}

void softmax_cross_entropy_label(float *pred, const int *labels, float *grad,
                                 float *loss, int rows, int cols) {
  softmax_cross_entropy_label_t<float>(pred, labels, grad, loss, rows, cols);
}

void softmax_cross_entropy_label_f32(float *pred, const int *labels,
                                     float *grad, float *loss, int rows,
                                     int cols) {
  softmax_cross_entropy_label_t<float>(pred, labels, grad, loss, rows, cols);
}

void softmax_cross_entropy_label_matrix(float *pred, const float *labels,
                                        float *grad, float *loss, int rows,
                                        int cols) {
  softmax_cross_entropy_label_matrix_t<float>(pred, labels, grad, loss, rows,
                                              cols);
}

void softmax_cross_entropy_label_matrix_f32(float *pred, const float *labels,
                                            float *grad, float *loss, int rows,
                                            int cols) {
  softmax_cross_entropy_label_matrix_t<float>(pred, labels, grad, loss, rows,
                                              cols);
}

void softmax_cross_entropy_label_matrix_fp16(const __half *pred,
                                             const __half *labels, __half *grad,
                                             float *loss, int rows, int cols) {
  softmax_cross_entropy_label_matrix_t<__half>(pred, labels, grad, loss, rows,
                                               cols);
}

void softmax_cross_entropy_label_matrix_bf16(const __nv_bfloat16 *pred,
                                             const __nv_bfloat16 *labels,
                                             __nv_bfloat16 *grad, float *loss,
                                             int rows, int cols) {
  softmax_cross_entropy_label_matrix_t<__nv_bfloat16>(pred, labels, grad, loss,
                                                      rows, cols);
}

void mse_loss_gradient(float *pred, float *target, float *grad, float *loss,
                       int rows, int cols) {
  mse_loss_gradient_t<float>(pred, target, grad, loss, rows, cols);
}

void mse_loss_gradient_f32(float *pred, float *target, float *grad,
                           float *loss, int rows, int cols) {
  mse_loss_gradient_t<float>(pred, target, grad, loss, rows, cols);
}

void mse_loss_gradient_fp16(const __half *pred, const __half *target,
                            __half *grad, float *loss, int rows, int cols) {
  mse_loss_gradient_t<__half>(pred, target, grad, loss, rows, cols);
}

void mse_loss_gradient_bf16(const __nv_bfloat16 *pred, const __nv_bfloat16 *target,
                            __nv_bfloat16 *grad, float *loss, int rows,
                            int cols) {
  mse_loss_gradient_t<__nv_bfloat16>(pred, target, grad, loss, rows, cols);
}

} // extern "C"
