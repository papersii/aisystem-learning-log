# results.md — 原始数据账本

测量环境：RTX 4090（SM89，云实例）· CUDA 13.0 · `nvcc -O3 -arch=native`
计时纪律：3 次 warmup + 20 次平均（bank 消融为 3+10），只计 kernel，不含 memcpy
校验：全部 kernel 对拍 cuBLAS，混合误差 abs<1e-3 或 rel<1e-3，全部 PASS（含 300×200×77 刁钻尺寸）
容差说明：atol 由 1e-4 放宽至 1e-3——K=4096 时 fp32 累加顺序噪声使个别接近 0 的元素（实测 2/16.7M）触碰旧阈值，且各 kernel 彼此一致，属数值噪声而非逻辑错误

## naive GEMM（两种线程映射）

| size | kernel | ms | GFLOPS | %cuBLAS |
|---|---|---|---|---|
| 1024³ | cuBLAS | 0.047 | 45789 | 100% |
| 1024³ | naive(coalesced) | 0.428 | 5013 | 10.95% |
| 1024³ | naive(uncoalesced) | 1.770 | 1213 | 2.65% |
| 2048³ | cuBLAS | 0.308 | 55862 | 100% |
| 2048³ | naive(coalesced) | 3.342 | 5141 | 9.20% |
| 2048³ | naive(uncoalesced) | 13.653 | 1258 | 2.25% |
| 4096³ | cuBLAS | 2.319 | 59269 | 100% |
| 4096³ | naive(coalesced) | 26.749 | 5138 | 8.67% |
| 4096³ | naive(uncoalesced) | 108.045 | 1272 | 2.15% |

**coalescing 加速比（4096³）：5138 / 1272 ≈ 4.0×**。唯一改动是 row/col 与线程编号的映射方向——warp 内相邻线程访问相邻地址（可合并为 128B 事务）vs 相隔 N×4B（各自独立事务）。低于教科书常说的 5–10×，推断是 Ada 的 72MB L2 替 uncoalesced 版兜住了部分重复流量。

## shared memory tiling

| size | kernel | ms | GFLOPS | %cuBLAS |
|---|---|---|---|---|
| 1024³ | smem TILE=32 | 0.331 | 6479 | 13.92% |
| 2048³ | smem TILE=32 | 2.629 | 6535 | 11.69% |
| 4096³ | smem TILE=32 | 22.237 | 6181 | 11.74% |

**vs coalesced naive：4096³ 为 6181/5138 ≈ 1.20×（2048³ 为 1.27×）**。global 流量理论上 ÷32，但实测远小于 32×——coalesced naive 已被 L2 缓存部分挽救；缓存靠猜、shared memory 靠显式管理，收益是"确定性"而非倍数。仍是 memory bound（算术强度 ≈ 8 FLOP/B，未过 4090 的 ridge point），下一步收益点在寄存器层（register blocking）。

TILE sweep（2048³）：TILE=8 → 4922；TILE=16 → 6831；TILE=32 → 6784 GFLOPS。
TILE=8 掉队：复用次数少（÷8）且 block 仅 64 线程，喂不饱 SM；16 与 32 打平（16 甚至略优——复用×16 已够 L2 补齐，且更小 tile 提高 occupancy 灵活性）。

## bank conflict 消融（4096³，v0/v1/v2 数学等价）

| 版本 | ms | GFLOPS | 说明 |
|---|---|---|---|
| v0_normal（As 行主序存取） | 22.073 | 6227 | 基线（与上表 22.237 一致，复现性 ✓） |
| v1_conflict（As 转置存放，写入 stride=32） | 36.554 | 3760 | 全 warp 写同一 bank，串行 32 拍 → **慢 66%** |
| v2_padded（同 v1 + `[32][33]` padding） | 31.611 | 4348 | padding 消除写冲突，**回收约 1/3 差距** |

v2 未完全回到 v0 的解释（推断）：padding 只治 bank 排队；v1/v2 的转置布局还让 FMA 阶段对 As 的访问失去行连续性，编译器无法把相邻 k 的读合并成向量化 LDS——这部分代价 padding 治不了。彻底归因需 ncu 的 bank conflict 计数器（云容器无性能计数器权限，未验证，标注为推断）。

## 原始日志

全部数字可由 README §复现 的命令重新生成。
