// ════════════════════════════════════════════════════════════════
//  bank conflict 对照实验（4096³，三个 kernel 数学等价）：
//    v0_normal    正常存取（无冲突基线）
//    v1_conflict  As 转置存放 As[tx][ty]——warp 内写入地址 stride=32 floats，
//                 全员挤同一 bank，32 路冲突
//    v2_padded    同 v1，仅把 As 声明为 [T][T+1]，padding 错峰
//  运行：./bank_conflict
// ════════════════════════════════════════════════════════════════
#include "helpers.cuh"
#include <vector>

constexpr int T = 32;

// v0：标准 tiled GEMM 基线
__global__ void v0_normal(const float* A, const float* B, float* C,
                          int M, int N, int K) {
  int tx = threadIdx.x, ty = threadIdx.y;
  int col = blockIdx.x * T + tx, row = blockIdx.y * T + ty;
  __shared__ float As[T][T], Bs[T][T];
  float acc = 0.f;
  for (int t = 0; t < (K + T - 1) / T; ++t) {
    As[ty][tx] = (row < M && t * T + tx < K) ? A[row * K + t * T + tx] : 0.f;
    Bs[ty][tx] = (t * T + ty < K && col < N) ? B[(t * T + ty) * N + col] : 0.f;
    __syncthreads();
    for (int k = 0; k < T; ++k) acc += As[ty][k] * Bs[k][tx];
    __syncthreads();
  }
  if (row < M && col < N) C[row * N + col] = acc;
}

// v1：As 转置存放 —— 存入时 warp 全员同 bank（32 路写冲突）。
//     读取 As[k][ty] 是 warp 内同地址 broadcast，不冲突；
//     数学结果与 v0 完全一致，慢的部分全部来自银行排队。
__global__ void v1_conflict(const float* A, const float* B, float* C,
                            int M, int N, int K) {
  int tx = threadIdx.x, ty = threadIdx.y;
  int col = blockIdx.x * T + tx, row = blockIdx.y * T + ty;
  __shared__ float As[T][T];   // As[c][r] = A_tile[r][c]（转置）
  __shared__ float Bs[T][T];
  float acc = 0.f;
  for (int t = 0; t < (K + T - 1) / T; ++t) {
    As[tx][ty] = (row < M && t * T + tx < K) ? A[row * K + t * T + tx] : 0.f;  // ← stride 32：冲突！
    Bs[ty][tx] = (t * T + ty < K && col < N) ? B[(t * T + ty) * N + col] : 0.f;
    __syncthreads();
    for (int k = 0; k < T; ++k) acc += As[k][ty] * Bs[k][tx];
    __syncthreads();
  }
  if (row < M && col < N) C[row * N + col] = acc;
}

// v2：只改一处声明（[T][T+1]）——对照实验一次只改一个变量
__global__ void v2_padded(const float* A, const float* B, float* C,
                          int M, int N, int K) {
  int tx = threadIdx.x, ty = threadIdx.y;
  int col = blockIdx.x * T + tx, row = blockIdx.y * T + ty;
  __shared__ float As[T][T + 1];  // ← 唯一改动：每行 padding 1 个空位，间隔 33，33 mod 32 = 1，错峰
  __shared__ float Bs[T][T];
  float acc = 0.f;
  for (int t = 0; t < (K + T - 1) / T; ++t) {
    As[tx][ty] = (row < M && t * T + tx < K) ? A[row * K + t * T + tx] : 0.f;
    Bs[ty][tx] = (t * T + ty < K && col < N) ? B[(t * T + ty) * N + col] : 0.f;
    __syncthreads();
    for (int k = 0; k < T; ++k) acc += As[k][ty] * Bs[k][tx];
    __syncthreads();
  }
  if (row < M && col < N) C[row * N + col] = acc;
}

template <typename LaunchFn>
static float bench_ms(LaunchFn launch, int warmup = 3, int reps = 10) {
  for (int i = 0; i < warmup; ++i) launch();
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  GpuTimer t;
  t.start();
  for (int i = 0; i < reps; ++i) launch();
  return t.stop_ms() / reps;
}

int main() {
  const int M = 4096, N = 4096, K = 4096;
  std::vector<float> A(1ull * M * K), B(1ull * K * N), Cref(1ull * M * N),
      Cgot(1ull * M * N);
  rand_fill(A.data(), A.size(), 1);
  rand_fill(B.data(), B.size(), 2);
  float *dA, *dB, *dC;
  CUDA_CHECK(cudaMalloc(&dA, A.size() * 4));
  CUDA_CHECK(cudaMalloc(&dB, B.size() * 4));
  CUDA_CHECK(cudaMalloc(&dC, Cref.size() * 4));
  CUDA_CHECK(cudaMemcpy(dA, A.data(), A.size() * 4, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dB, B.data(), B.size() * 4, cudaMemcpyHostToDevice));

  dim3 block(T, T), grid((N + T - 1) / T, (M + T - 1) / T);

  v0_normal<<<grid, block>>>(dA, dB, dC, M, N, K);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaMemcpy(Cref.data(), dC, Cref.size() * 4, cudaMemcpyDeviceToHost));

  struct Entry { const char* name; void (*k)(const float*, const float*, float*, int, int, int); };
  Entry entries[] = {{"v0_normal  ", v0_normal},
                     {"v1_conflict", v1_conflict},
                     {"v2_padded  ", v2_padded}};
  for (auto& e : entries) {
    CUDA_CHECK(cudaMemset(dC, 0, Cref.size() * 4));
    e.k<<<grid, block>>>(dA, dB, dC, M, N, K);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(Cgot.data(), dC, Cgot.size() * 4, cudaMemcpyDeviceToHost));
    if (!check_close(Cref.data(), Cgot.data(), Cgot.size(), 1e-3f, 1e-3f, e.name))
      continue;
    float ms = bench_ms([&] { e.k<<<grid, block>>>(dA, dB, dC, M, N, K); });
    printf("  %s  ms=%-9.3f GFLOPS=%-9.1f\n", e.name, ms, gemm_gflops(M, N, K, ms));
  }

  CUDA_CHECK(cudaFree(dA)); CUDA_CHECK(cudaFree(dB)); CUDA_CHECK(cudaFree(dC));
  return 0;
}
