# 论文 notes — Mirage: A Multi-Level Superoptimizer for Tensor Programs（OSDI 2025）

## 1. 要解决什么问题（现有方案为什么不行）

解决了“高级性能优化需要在GPU计算层级的kernel、thread block、thread三层之间协同地做变换， 并引入全新的kernel计算”需要手工实现的麻烦问题。

现有自动方法分两派：调度派只优化"怎么算"，算法要人给定；代数派只在kernel层换算法，kernel本身要人手写。而“把Kernel拆解，只融合其中某些计算”恰好落在两派的搜索空间之外。

## 2. 核心 idea

𝝁Graph把kernel/block/thread三层放在一起看，画成一张图。之前的“换算法、调调度、造新kernel”三种优化方式，都变成在同一个图空间中的搜索问题。

## 3. 系统怎么实现这个 idea（关键抽象）

先把输入程序切成LAX子程序，只含线性算子/除法/受限指数。生成器从输入出发一个算子一个算子地长出候选 µGraph，kernel/block两层穷举、thread层用规则融合，每长一步就给半成品算一个"抽象表达式"（忽略张量内部细节），用SMT检查它是不是目标表达式的子表达式，不是就剪掉。验证器对剩下的候选在有限域上做随机测试（取模运算），最后优化器对验证过的µGraph做布局、算子调度、内存规划，选出最快的。

## 4. eval 亮点

在A100/H100上用 6 个 LLM 常用基准（GQA、RMSNorm、LoRA、QKNorm、GatedMLP、nTrans，半精度）对比了TASO/PET、PyTorch（torch.compile + FlashAttention）、TensorRT/TensorRT-LLM、FlashAttention/FlashDecoding和Triton，所有基线都开了CUDA Graphs。微基准上相比最强基线最多快3.3×。

最有说服力的实验是GQA案例里Mirage自动重新发现了FlashAttention和FlashDecoding这两个专家手写kernel（然后在小batch场景下靠更好的grid选择和并行维度选择超过了它们），重新发现人类专家的设计甚至改进，是对方法有效性的绝佳论证。

## 5. 我没看懂的地方 + 我的问题（≥2 条，具体到章节号）

1. 理论上要跑多轮随机测试把错误率压到任意低，但实现里只跑单轮为什么就够了？（§5.2/§7）

2. 最终结果的候选µGraph有很大可能不止一条，这个时候该如何比较它们的效率（也就是选出最快的）？是直接在GPU上跑吗？这样如果最终匹配的µGraph很多，找到最优解的效率是不是很低？（§6/§7）
