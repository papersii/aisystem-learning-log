// ════════════════════════════════════════════════════════════════
//  shared memory tiling GEMM（TILE 为模板参数，自动扫 8/16/32）
//  用法：./sgemm_smem   （校验 + 三尺寸跑分 + TILE sweep）
// ════════════════════════════════════════════════════════════════
#include "helpers.cuh"
#include <vector>

// 边界规则：越界的线程「搬 0」而不是提前 return——__syncthreads() 必须全员到场；
// 写回 C 时才做边界判断。
template <int TILE>
__global__ void sgemm_smem(const float* A, const float* B, float* C,
                           int M, int N, int K) {
  // ① 本线程坐标
  int tx = threadIdx.x, ty = threadIdx.y;
  int col = blockIdx.x * TILE + tx;   // x ↔ 列：保持 coalesced
  int row = blockIdx.y * TILE + ty;

  // ② 车间仓库：两块 TILE×TILE 的货架，整个 block 共用
  __shared__ float As[TILE][TILE];
  __shared__ float Bs[TILE][TILE];

  float acc = 0.0f;
  // ③ 阶段循环：每轮吃掉 K 方向上的一对 tile
  for (int t = 0; t < (K + TILE - 1) / TILE; ++t) {
    // 协作搬砖；越界的工人搬 0（不许 return——哨声必须全员到场）
    As[ty][tx] = (row < M && t * TILE + tx < K) ? A[row * K + t * TILE + tx] : 0.0f;
    Bs[ty][tx] = (t * TILE + ty < K && col < N) ? B[(t * TILE + ty) * N + col] : 0.0f;
    __syncthreads();  // 哨声①：上齐才开吃
    // ④ 从仓库取料，TILE 次乘加
#pragma unroll
    for (int k = 0; k < TILE; ++k)
      acc += As[ty][k] * Bs[k][tx];
    __syncthreads();  // 哨声②：吃完才撤盘（防下一轮覆盖）
  }
  // ⑥ 带边界保护写回
  if (row < M && col < N) C[row * N + col] = acc;
}

// ── 校验与跑分框架 ──────────────────────────────────────────────

template <typename LaunchFn>
static float bench_ms(LaunchFn launch, int warmup = 3, int reps = 20) {
  for (int i = 0; i < warmup; ++i) launch();
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  GpuTimer t;
  t.start();
  for (int i = 0; i < reps; ++i) launch();
  return t.stop_ms() / reps;
}

template <int TILE>
static void check_and_bench(cublasHandle_t h, int M, int N, int K, bool bench) {
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

  cublas_gemm_rowmajor(h, dA, dB, dC, M, N, K);
  CUDA_CHECK(cudaMemcpy(Cref.data(), dC, Cref.size() * 4, cudaMemcpyDeviceToHost));

  dim3 block(TILE, TILE);
  dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);
  CUDA_CHECK(cudaMemset(dC, 0, Cref.size() * 4));
  sgemm_smem<TILE><<<grid, block>>>(dA, dB, dC, M, N, K);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaMemcpy(Cgot.data(), dC, Cgot.size() * 4, cudaMemcpyDeviceToHost));

  char tag[64];
  snprintf(tag, sizeof(tag), "smem TILE=%d %dx%dx%d", TILE, M, N, K);
  bool ok = check_close(Cref.data(), Cgot.data(), Cgot.size(), 1e-3f, 1e-3f, tag);  // atol 放宽理由见 02_naive_gemm 内注释
  if (ok && bench) {
    float blas_ms = bench_ms([&] { cublas_gemm_rowmajor(h, dA, dB, dC, M, N, K); });
    float ms = bench_ms([&] { sgemm_smem<TILE><<<grid, block>>>(dA, dB, dC, M, N, K); });
    printf("  TILE=%-3d %-6d ms=%-9.3f GFLOPS=%-9.1f %%cuBLAS=%.2f%%\n", TILE, M,
           ms, gemm_gflops(M, N, K, ms), 100.0 * blas_ms / ms);
  }
  CUDA_CHECK(cudaFree(dA)); CUDA_CHECK(cudaFree(dB)); CUDA_CHECK(cudaFree(dC));
}

int main() {
  cublasHandle_t h;
  CUBLAS_CHECK(cublasCreate(&h));

  printf("── 边界尺寸校验 ──\n");
  check_and_bench<32>(h, 128, 128, 128, false);
  check_and_bench<32>(h, 100, 100, 100, false);
  check_and_bench<32>(h, 300, 200, 77, false);   // K 不整除 TILE！

  printf("── 三尺寸跑分 ──\n");
  for (int s : {1024, 2048, 4096}) check_and_bench<32>(h, s, s, s, true);

  printf("── TILE sweep（2048³）──\n");
  check_and_bench<8>(h, 2048, 2048, 2048, true);
  check_and_bench<16>(h, 2048, 2048, 2048, true);
  check_and_bench<32>(h, 2048, 2048, 2048, true);

  CUBLAS_CHECK(cublasDestroy(h));
  return 0;
}
