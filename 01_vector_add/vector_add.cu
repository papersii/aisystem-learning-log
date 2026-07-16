// ════════════════════════════════════════════════════════════════
//  vector add：入门练习——正确性校验 + cudaEvent 计时 + 有效带宽
//  编译运行：make && ./vector_add [block_size]（默认 256）
// ════════════════════════════════════════════════════════════════
#include "helpers.cuh"
#include <vector>

__global__ void vector_add(const float* a, const float* b, float* c, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) c[i] = a[i] + b[i];  // 边界保护：n 不一定被 block 整除
}

int main(int argc, char** argv) {
  const int block = (argc > 1) ? atoi(argv[1]) : 256;
  printf("block size = %d\n", block);

  // ── 正确性测试：三个尺寸，缺一不可 ──
  const std::vector<int> check_sizes = {1000, 1000003, 1 << 24};
  for (int n : check_sizes) {
    std::vector<float> ha(n), hb(n), hc(n, 0.f), href(n);
    rand_fill(ha.data(), n, 1);
    rand_fill(hb.data(), n, 2);
    for (int i = 0; i < n; ++i) href[i] = ha[i] + hb[i];  // CPU 参考答案

    float *da, *db, *dc;
    CUDA_CHECK(cudaMalloc(&da, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&db, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dc, n * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(da, ha.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(db, hb.data(), n * sizeof(float), cudaMemcpyHostToDevice));

    int grid = (n + block - 1) / block;  // 向上取整，罩住不满一个 block 的尾巴

    vector_add<<<grid, block>>>(da, db, dc, n);
    CUDA_CHECK(cudaGetLastError());        // launch 参数错误在这里现形
    CUDA_CHECK(cudaDeviceSynchronize());   // 越界等执行期错误在这里现形

    CUDA_CHECK(cudaMemcpy(hc.data(), dc, n * sizeof(float), cudaMemcpyDeviceToHost));

    char tag[64];
    snprintf(tag, sizeof(tag), "vector_add N=%d", n);
    check_close(href.data(), hc.data(), n, 1e-5f, 1e-6f, tag);

    CUDA_CHECK(cudaFree(da));
    CUDA_CHECK(cudaFree(db));
    CUDA_CHECK(cudaFree(dc));
  }

  // ── 计时 + 有效带宽表 ──
  printf("\n%-12s %-12s %-12s\n", "N", "avg_ms", "GB/s");
  const std::vector<int> bench_sizes = {1 << 20, 1 << 22, 1 << 24, 1 << 26};
  for (int n : bench_sizes) {
    std::vector<float> ha(n), hb(n);
    rand_fill(ha.data(), n, 1);
    rand_fill(hb.data(), n, 2);
    float *da, *db, *dc;
    CUDA_CHECK(cudaMalloc(&da, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&db, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dc, n * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(da, ha.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(db, hb.data(), n * sizeof(float), cudaMemcpyHostToDevice));

    int grid = (n + block - 1) / block;

    float avg_ms = 0.f;
    // 计时纪律：3 次 warmup + 20 次平均，计时段内不做 memcpy
    for (int w = 0; w < 3; ++w) vector_add<<<grid, block>>>(da, db, dc, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    GpuTimer t;
    t.start();
    for (int r = 0; r < 20; ++r) vector_add<<<grid, block>>>(da, db, dc, n);
    avg_ms = t.stop_ms() / 20.0f;

    // 有效带宽：每元素读 8B 写 4B
    double gbs = (avg_ms > 0.f) ? (3.0 * n * 4.0) / (avg_ms * 1e-3) / 1e9 : 0.0;
    printf("%-12d %-12.4f %-12.1f\n", n, avg_ms, gbs);

    CUDA_CHECK(cudaFree(da));
    CUDA_CHECK(cudaFree(db));
    CUDA_CHECK(cudaFree(dc));
  }

  return 0;
}
