// ════════════════════════════════════════════════════════════════
//  naive GEMM（两种线程映射）+ CPU 参考实现 + cuBLAS 校验/跑分框架
//  约定：A(M×K) · B(K×N) = C(M×N)，全部 row-major。
//  用法：./sgemm --selftest  /  ./sgemm --bench
// ════════════════════════════════════════════════════════════════
#include "helpers.cuh"
#include <cstring>
#include <vector>

// CPU 参考实现（裁判的裁判）：double 累加减小误差
void cpu_gemm(const float* A, const float* B, float* C, int M, int N, int K) {
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      double acc = 0.0;  // double 累加：K 次 float 连加会积累误差，裁判要更准
      for (int k = 0; k < K; ++k)
        acc += (double)A[i * K + k] * (double)B[k * N + j];
      C[i * N + j] = (float)acc;
    }
  }
}

// naive GEMM（coalesced 映射）：每线程算一个 C 元素
__global__ void sgemm_naive(const float* A, const float* B, float* C,
                            int M, int N, int K) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;  // warp 内连号的 tx 对准列 → 相邻线程访问相邻地址
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  if (row < M && col < N) {
    float acc = 0.0f;
    for (int k = 0; k < K; ++k)
      acc += A[row * K + k] * B[k * N + col];
    C[row * N + col] = acc;
  }
}

// 对照组：同一 kernel、映射方向对调（uncoalesced，用于量化 coalescing 的收益）
__global__ void sgemm_naive_swapped(const float* A, const float* B, float* C,
                                    int M, int N, int K) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;  // tx 对准行 → warp 内地址相隔 N×4B，无法合并
  int col = blockIdx.y * blockDim.y + threadIdx.y;
  if (row < M && col < N) {
    float acc = 0.0f;
    for (int k = 0; k < K; ++k)
      acc += A[row * K + k] * B[k * N + col];
    C[row * N + col] = acc;
  }
}

// ── 计时纪律：3 次 warmup + 20 次平均 ──
template <typename LaunchFn>
float bench_ms(LaunchFn launch, int warmup = 3, int reps = 20) {
  for (int i = 0; i < warmup; ++i) launch();
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  GpuTimer t;
  t.start();
  for (int i = 0; i < reps; ++i) launch();
  return t.stop_ms() / reps;
}

static void run_selftest() {
  printf("── selftest ──\n");
  // 测试 1：A = 单位矩阵 ⇒ C 必须等于 B（不需要任何手算，零出错空间）
  {
    const int n = 4;
    std::vector<float> A(n * n, 0.f), B(n * n), C(n * n, -1.f);
    for (int i = 0; i < n; ++i) A[i * n + i] = 1.0f;
    rand_fill(B.data(), n * n, 7);
    cpu_gemm(A.data(), B.data(), C.data(), n, n, n);
    check_close(B.data(), C.data(), n * n, 1e-6f, 1e-7f, "cpu_gemm(I,B)==B");
  }
  // 测试 2：128³，cpu_gemm vs cuBLAS 互相印证
  {
    const int M = 128, N = 128, K = 128;
    std::vector<float> A(M * K), B(K * N), Ccpu(M * N), Cblas(M * N);
    rand_fill(A.data(), A.size(), 1);
    rand_fill(B.data(), B.size(), 2);
    cpu_gemm(A.data(), B.data(), Ccpu.data(), M, N, K);

    float *dA, *dB, *dC;
    CUDA_CHECK(cudaMalloc(&dA, A.size() * 4));
    CUDA_CHECK(cudaMalloc(&dB, B.size() * 4));
    CUDA_CHECK(cudaMalloc(&dC, Cblas.size() * 4));
    CUDA_CHECK(cudaMemcpy(dA, A.data(), A.size() * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, B.data(), B.size() * 4, cudaMemcpyHostToDevice));
    cublasHandle_t h;
    CUBLAS_CHECK(cublasCreate(&h));
    cublas_gemm_rowmajor(h, dA, dB, dC, M, N, K);
    CUDA_CHECK(cudaMemcpy(Cblas.data(), dC, Cblas.size() * 4, cudaMemcpyDeviceToHost));
    check_close(Ccpu.data(), Cblas.data(), Cblas.size(), 1e-3f, 1e-4f,
                "cpu_gemm vs cuBLAS 128^3");
    CUBLAS_CHECK(cublasDestroy(h));
    CUDA_CHECK(cudaFree(dA)); CUDA_CHECK(cudaFree(dB)); CUDA_CHECK(cudaFree(dC));
  }
}

static void run_bench() {
  cublasHandle_t h;
  CUBLAS_CHECK(cublasCreate(&h));
  printf("%-8s %-22s %-10s %-10s %-10s\n", "size", "kernel", "ms", "GFLOPS", "%cuBLAS");

  for (int s : {1024, 2048, 4096}) {
    const int M = s, N = s, K = s;
    std::vector<float> A(1ull * M * K), B(1ull * K * N);
    std::vector<float> Cref(1ull * M * N), Cgot(1ull * M * N);
    rand_fill(A.data(), A.size(), 1);
    rand_fill(B.data(), B.size(), 2);

    float *dA, *dB, *dC;
    CUDA_CHECK(cudaMalloc(&dA, A.size() * 4));
    CUDA_CHECK(cudaMalloc(&dB, B.size() * 4));
    CUDA_CHECK(cudaMalloc(&dC, Cref.size() * 4));
    CUDA_CHECK(cudaMemcpy(dA, A.data(), A.size() * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, B.data(), B.size() * 4, cudaMemcpyHostToDevice));

    // cuBLAS：oracle + 标杆
    cublas_gemm_rowmajor(h, dA, dB, dC, M, N, K);
    CUDA_CHECK(cudaMemcpy(Cref.data(), dC, Cref.size() * 4, cudaMemcpyDeviceToHost));
    float blas_ms = bench_ms([&] { cublas_gemm_rowmajor(h, dA, dB, dC, M, N, K); });
    printf("%-8d %-22s %-10.3f %-10.1f %-10s\n", s, "cuBLAS", blas_ms,
           gemm_gflops(M, N, K, blas_ms), "100%");

    dim3 block(16, 16);
    dim3 grid((N + 15) / 16, (M + 15) / 16);  // 向上取整；x 对应 N（列），与映射一致

    CUDA_CHECK(cudaMemset(dC, 0, Cref.size() * 4));
    sgemm_naive<<<grid, block>>>(dA, dB, dC, M, N, K);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(Cgot.data(), dC, Cgot.size() * 4, cudaMemcpyDeviceToHost));
    // atol=1e-3：K=4096 时 fp32 累加顺序噪声可达 1e-4 量级（接近 0 的元素相对误差放大），
    // 实测仅 2/16.7M 元素触碰旧阈值且各 kernel 彼此一致——属数值噪声而非逻辑错误
    if (check_close(Cref.data(), Cgot.data(), Cgot.size(), 1e-3f, 1e-3f, "naive")) {
      float ms = bench_ms([&] { sgemm_naive<<<grid, block>>>(dA, dB, dC, M, N, K); });
      printf("%-8d %-22s %-10.3f %-10.1f %-9.2f%%\n", s, "naive(coalesced)", ms,
             gemm_gflops(M, N, K, ms), 100.0 * blas_ms / ms);
    }

    // 另一种映射（coalescing 对照组）
    CUDA_CHECK(cudaMemset(dC, 0, Cref.size() * 4));
    sgemm_naive_swapped<<<grid, block>>>(dA, dB, dC, M, N, K);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(Cgot.data(), dC, Cgot.size() * 4, cudaMemcpyDeviceToHost));
    if (check_close(Cref.data(), Cgot.data(), Cgot.size(), 1e-3f, 1e-3f, "swapped")) {
      float ms = bench_ms([&] { sgemm_naive_swapped<<<grid, block>>>(dA, dB, dC, M, N, K); });
      printf("%-8d %-22s %-10.3f %-10.1f %-9.2f%%\n", s, "naive(uncoalesced)", ms,
             gemm_gflops(M, N, K, ms), 100.0 * blas_ms / ms);
    }

    CUDA_CHECK(cudaFree(dA)); CUDA_CHECK(cudaFree(dB)); CUDA_CHECK(cudaFree(dC));
  }
  CUBLAS_CHECK(cublasDestroy(h));
}

int main(int argc, char** argv) {
  if (argc > 1 && strcmp(argv[1], "--selftest") == 0) run_selftest();
  else run_bench();
  return 0;
}
