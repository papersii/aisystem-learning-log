// ════════════════════════════════════════════════════════════════
//  公共工具：CUDA_CHECK / CUBLAS_CHECK / GpuTimer / rand_fill /
//            check_close / cublas_gemm_rowmajor
//  用法：#include "helpers.cuh"（Makefile 已加 -I../common）
// ════════════════════════════════════════════════════════════════
#pragma once
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>
#include <cublas_v2.h>

// ── 错误检查：每一个 CUDA 调用都包上，错误不静默 ──
#define CUDA_CHECK(call)                                                      \
  do {                                                                        \
    cudaError_t err_ = (call);                                                \
    if (err_ != cudaSuccess) {                                                \
      fprintf(stderr, "[CUDA] %s:%d %s\n", __FILE__, __LINE__,                \
              cudaGetErrorString(err_));                                      \
      exit(1);                                                                \
    }                                                                         \
  } while (0)

#define CUBLAS_CHECK(call)                                                    \
  do {                                                                        \
    cublasStatus_t st_ = (call);                                              \
    if (st_ != CUBLAS_STATUS_SUCCESS) {                                       \
      fprintf(stderr, "[cuBLAS] %s:%d status=%d\n", __FILE__, __LINE__,       \
              (int)st_);                                                      \
      exit(1);                                                                \
    }                                                                         \
  } while (0)

// ── GPU 秒表：cudaEvent 封装（kernel launch 是异步的，CPU 计时器量不准）──
struct GpuTimer {
  cudaEvent_t start_evt, stop_evt;
  GpuTimer() {
    CUDA_CHECK(cudaEventCreate(&start_evt));
    CUDA_CHECK(cudaEventCreate(&stop_evt));
  }
  ~GpuTimer() {
    cudaEventDestroy(start_evt);
    cudaEventDestroy(stop_evt);
  }
  void start() { CUDA_CHECK(cudaEventRecord(start_evt)); }
  // 返回毫秒。内部会同步等待 GPU 真正跑完。
  float stop_ms() {
    CUDA_CHECK(cudaEventRecord(stop_evt));
    CUDA_CHECK(cudaEventSynchronize(stop_evt));
    float ms = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start_evt, stop_evt));
    return ms;
  }
};

// ── 随机填充 [-1, 1]（固定种子，保证两次运行数据一致）──
inline void rand_fill(float* p, size_t n, unsigned seed = 42) {
  srand(seed);
  for (size_t i = 0; i < n; ++i)
    p[i] = 2.0f * (float)rand() / (float)RAND_MAX - 1.0f;
}

// ── 数值比对：混合误差（abs < atol 或 rel < rtol 即算过）──
//    不能要求逐位相等：浮点加法不结合，两边对 K 个乘积的求和顺序不同。
inline bool check_close(const float* ref, const float* got, size_t n,
                        float rtol = 1e-3f, float atol = 1e-4f,
                        const char* tag = "") {
  size_t bad = 0;
  for (size_t i = 0; i < n; ++i) {
    float a = ref[i], b = got[i];
    float diff = fabsf(a - b);
    if (diff > atol && diff > rtol * fabsf(a)) {
      if (bad < 5)
        fprintf(stderr, "  MISMATCH %s at %zu: ref=%.6f got=%.6f\n", tag, i, a, b);
      ++bad;
    }
  }
  if (bad == 0) {
    printf("  [PASS] %s (%zu elements)\n", tag, n);
    return true;
  }
  printf("  [FAIL] %s: %zu / %zu mismatched\n", tag, bad, n);
  return false;
}

// ── row-major 的 C = A·B，用列主序的 cuBLAS 算 ──────────────────
//  cuBLAS 是列主序（Fortran 传统）。恒等式：C = A·B  ⇔  Cᵀ = Bᵀ·Aᵀ。
//  row-major 的 C(M×N) 在 cuBLAS 眼里恰好就是列主序的 Cᵀ(N×M)，
//  同理 row-major 的 B 是列主序的 Bᵀ(N×K)、A 是 Aᵀ(K×M)。
//  所以「先传 B、再传 A」，不需要做任何真实转置拷贝：
//      Cᵀ(N×M) = Bᵀ(N×K) · Aᵀ(K×M)
//
inline void cublas_gemm_rowmajor(cublasHandle_t h, const float* dA,
                                 const float* dB, float* dC, int M, int N,
                                 int K) {
  const float alpha = 1.0f, beta = 0.0f;
  CUBLAS_CHECK(cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N,
                           /*m=*/N, /*n=*/M, /*k=*/K,
                           &alpha,
                           dB, /*ldb=*/N,
                           dA, /*lda=*/K,
                           &beta,
                           dC, /*ldc=*/N));
}

// ── GFLOPS 换算 ──
inline double gemm_gflops(int M, int N, int K, float ms) {
  return 2.0 * M * N * K / (ms * 1e-3) / 1e9;
}
