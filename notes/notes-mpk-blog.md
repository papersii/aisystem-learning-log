# notes（博客版初读）— MPK: Mirage Persistent Kernel

> 这篇我目前只读了作者的博客版（Zhihao Jia，2025-06），论文本体还没精读，排在后面，读完会补上note。

## 1. 要解决什么问题

现在的LLM推理系统是一个算子一个kernel，跑matmul launch一次，跑attention又 launch一次，多卡通信还要另外调NCCL，一次decode下来要发射几百个kernel。这带来三个浪费：launch本身有固定开销；kernel 之间是硬边界，下一层没法提前加载数据；计算和通信只能串行等。我自己的仓库现在也是，每个版本都是一次launch跑一个算子。

MPK的方案是把整个模型的推理编译成**一个** kernel，只launch一次，从头跑到尾。

## 2. 怎么做

**编译器**：把计算图变成更细粒度的任务图，任务是分给单个SM的一小块计算或通信，事件是任务间的同步点。原先allreduce要等整个matmul跑完才能开始，但其实allreduce的每块数据只依赖matmul输出的一部分。任务图把依赖记到数据块的层级，通信就能在部分结果出来时提前开跑。每个任务的CUDA实现由Mirage自动生成。

**运行时**：整个任务图在一个megakernel内部执行。SM被静态分成两种角色：worker（每人一个任务队列，循环干活）和scheduler（跑在单个warp上，每个SM最多塞4个，管事件队列、把就绪的下游任务派发出去，完全去中心化）。任务之间的切换开销极低。

## 3. 效果 + 和Mirage的关系

A100上decode每token延迟从vLLM/SGLang的14.5ms降到12.5ms，接近理论下限10ms。多卡时因为计算和通信能重叠，收益随卡数增大。

和Mirage的关系：是同一个团队、同一个仓库。MPK任务图里每个任务的实现就是Mirage生成的。

博客结尾作者自己列的三个没解决的问题：
1. Blackwell上怎么把warp specialization塞进megakernel的执行模型。
2. 任务图是静态的，撑不住MoE这种动态workload。
3. 调度目前只是简单round-robin。

## 4. 我的问题

静态任务图加上事件计数器这个方案，如果batch大小或序列长度一变，是不是就要重新编译一遍？这个限制是不是后续工作可以切入的口子？
