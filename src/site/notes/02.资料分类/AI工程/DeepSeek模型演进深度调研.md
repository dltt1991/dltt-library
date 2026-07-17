---
{"dg-publish":true,"permalink":"/02.资料分类/AI工程/DeepSeek模型演进深度调研/","dg-note-properties":{}}
---

# DeepSeek模型演进深度调研：从67B到V4的架构创新与训练优化

## TL;DR

DeepSeek在2023年11月至2026年4月的约两年半时间里，完成了从**67B稠密模型到1.6T MoE模型**的跨越式发展。其核心创新可归纳为四大支柱：**Multi-head Latent Attention（MLA）** 将KV缓存压缩93.3%；**DeepSeekMoE** 实现671B总参数仅激活37B的稀疏计算；**FP8混合精度+DualPipe** 将训练成本降至**$5.58M**（仅为GPT-4的~1/1800）；**CSA+HCA混合注意力** 在1M上下文下将推理FLOPs降至V3.2的27%。这些架构与工程创新的叠加，使DeepSeek以极低成本持续逼近甚至超越闭源前沿模型的性能。

---

## 1. 模型演进总览与核心指标

### 1.1 版本演进时间线

![架构创新时间线.png](/img/user/%E9%99%84%E4%BB%B6/DeepSeek%E6%A8%A1%E5%9E%8B%E6%BC%94%E8%BF%9B%E6%B7%B1%E5%BA%A6%E8%B0%83%E7%A0%94_assets/%E6%9E%B6%E6%9E%84%E5%88%9B%E6%96%B0%E6%97%B6%E9%97%B4%E7%BA%BF.png)

DeepSeek的模型家族经历了从**稠密Transformer**到**稀疏MoE**、再到**混合注意力架构**的三阶段跃迁。下表汇总了从2023年11月DeepSeek 67B到2026年4月V4系列的完整技术规格：

| 模型 | 发布时间 | 总参数 | 激活参数/Token | 上下文长度 | 核心架构创新 | 训练成本 |
|------|----------|--------|----------------|------------|--------------|----------|
| DeepSeek 67B | 2023.11 | 67B | 67B | 4K | 稠密Transformer，中英双语 | — |
| DeepSeek-V2 | 2024.05 | 236B | 21B | 128K | MLA + DeepSeekMoE | — |
| DeepSeek-V3 | 2024.12 | 671B | 37B | 128K | FP8 + DualPipe + MTP | **$5.58M**  [(epoch.ai)](https://epoch.ai/gradient-updates/what-went-into-training-deepseek-r1)  |
| DeepSeek-R1 | 2025.01 | 671B | 37B | 128K | GRPO + RLVR推理训练 | $5.58M (base)  [(epoch.ai)](https://epoch.ai/gradient-updates/what-went-into-training-deepseek-r1)  |
| R1-0528 | 2025.05 | 671B | 37B | 128K | 增强后训练，更长推理链 | — |
| V3.1 | 2025.08 | 671B | 37B | 128K | 混合推理架构 | — |
| V3.2 | 2025.12 | 671B | 37B | 160K | DSA稀疏注意力 | — |
| **V4-Pro** | **2026.04** | **1.6T** | **49B** | **1M** | **CSA+HCA + mHC + Muon** | **未披露**  [(Framia)](https://framia.converge.ai/page/en-US/news/deepseek-v4-model-card)  |
| **V4-Flash** | **2026.04** | **284B** | **13B** | **1M** | **CSA+HCA + mHC + Muon** | **未披露**  [(Framia)](https://framia.converge.ai/page/en-US/news/deepseek-v4-model-card)  |

DeepSeek-V4系列的发布标志着开源模型首次在**1M token上下文窗口**下实现与闭源前沿模型（GPT-5.5、Claude Opus 4.7、Gemini 3.1 Pro）相当的经济可部署性。V4-Pro的1.6T总参数与49B激活参数之比（**3.1%激活率**）进一步将MoE的稀疏性推向极致，而V4-Flash的284B/13B配置则代表了面向高吞吐场景的效率优化极致。

![参数规模演进.png](/img/user/%E9%99%84%E4%BB%B6/DeepSeek%E6%A8%A1%E5%9E%8B%E6%BC%94%E8%BF%9B%E6%B7%B1%E5%BA%A6%E8%B0%83%E7%A0%94_assets/%E5%8F%82%E6%95%B0%E8%A7%84%E6%A8%A1%E6%BC%94%E8%BF%9B.png)

### 1.2 参数规模与稀疏化趋势

DeepSeek模型演进中最显著的趋势是**总参数规模与激活参数规模的持续解耦**。从V1时代的1:1（67B/67B）到V4-Pro的约33:1（1.6T/49B），稀疏激活比例提升了两个数量级。这种解耦的实现依赖于两大核心机制：

DeepSeekMoE架构将每层FFN划分为**256个路由专家+1个共享专家**，每个token仅激活**8个路由专家+共享专家**  [(Grand Linux Solution Co., Ltd.)](https://www.grandlinux.com/en/blogs/deepseek-moe-architecture.html) 。共享专家处理所有token的通用特征（类似"全科医生"），而路由专家则负责领域特定特征。这种细粒度分工使得专家数量可以远多于传统MoE（如Mixtral的8专家×2激活），从而获得更精细的专业化。V4-Pro进一步将路由专家扩展到**384个**，激活6个  [(Effloow)](https://effloow.com/articles/deepseek-v4-pro-mit-frontier-model-developer-guide-2026) 。

辅助损失无关的负载均衡策略（auxiliary-loss-free load balancing）是DeepSeekMoE的另一关键创新。传统MoE需要额外的辅助损失函数来防止"专家崩溃"（所有token路由到少数专家），但这会损害模型质量。DeepSeek采用**序列级平衡损失**（sequence-wise balance loss），其平衡因子α被赋予极小的值，在确保负载均衡的同时最小化对模型性能的影响  [(arXiv.org)](https://arxiv.org/pdf/2412.19437) 。V3训练期间实现了**零token丢弃**（no token-dropping），证明了负载均衡策略的有效性。

---

## 2. 注意力机制的三代演进

### 2.1 第一代：Multi-Head Latent Attention (MLA) —— V2/V3/R1

![[Pasted image 20260630141827.png\|Pasted image 20260630141827.png]]
MLA是DeepSeek-V2于2024年5月首次引入的注意力机制  [(arXiv.org)](https://arxiv.org/abs/2405.04434v2) ，其核心思想是**通过低秩投影将KV缓存压缩到潜在空间**，而非像GQA那样通过减少KV头来降低缓存。标准MHA为每个注意力头存储完整的K/V向量，而MLA将所有头的K/V联合压缩为一个共享的低秩潜在向量（latent vector）。

MLA的数学实现可概括为三个步骤。首先，通过下投影矩阵 $W_c$ 将高维K/V压缩为低维潜在向量 $c = W_c \cdot [K;V]$。其次，在注意力计算时通过上投影矩阵 $W_{uk}$、$W_{uv}$ 从潜在向量重建各头的K/V。最后，通过**权重吸收技巧**（weight absorption trick）进一步优化推理效率——将上投影矩阵吸收进查询投影中，使得注意力计算直接在潜在空间完成，避免每次解码时重建完整K/V  [(datacrunch.io)](https://datacrunch.io/blog/deepseek-sglang-multi-head-latent-attention) 。
![[Pasted image 20260630141926.png\|Pasted image 20260630141926.png]]

MLA带来了**93.3%的KV缓存压缩率**（从V1 67B的MHA到V2的MLA） [(arXiv.org)](https://arxiv.org/abs/2405.04434v2) ，同时实验表明MLA在多个基准上不仅未损失质量，反而**略优于标准MHA**  [(Sebastian Raschka)](https://sebastianraschka.com/llms-from-scratch/ch04/05_mla/) 。这使得MLA成为DeepSeek-V3和R1的标准注意力机制，并被Kimi K2系列等其他模型采用  [(LLMS3.com)](https://llms3.com/node/multi-head-latent-attention-mla) 。

然而，MLA也存在局限：其压缩沿**头维度**（head dimension）进行，而上下文长度增长带来的序列维度膨胀问题未被根本解决。当上下文扩展到1M token时，即使压缩后的KV缓存仍可能达到数十GB量级。

### 2.2 第二代：DeepSeek Sparse Attention (DSA) —— V3.2

2025年9月发布的V3.2-Exp首次引入了**DeepSeek Sparse Attention (DSA)** ，这是注意力机制从"头维度压缩"向"序列维度压缩"演进的关键一步。DSA的核心机制是**闪电索引器**（Lightning Indexer）：模型在处理长序列时，先快速建立索引，然后每个query仅关注最相关的top-k个token块，将计算复杂度从 $O(L^2)$ 降低到近似 $O(kL)$  。

DSA的工程实现使得V3.2在128K序列上实现了**推理成本降低60%+、速度提升3.5倍、内存占用减少70%+**。这一技术为V4的混合注意力架构奠定了基础，V3.2可被视为从MLA到CSA+HCA的过渡实验。

### 2.3 第三代：CSA + HCA 混合注意力 —— V4

2026年4月发布的DeepSeek-V4标志着注意力机制的第三次重大变革——**完全放弃MLA，采用CSA+HCA混合架构**  [(techjacksolutions.com)](https://techjacksolutions.com/ai-tools/deepseek/deepseek-v4-architecture/) 。这一决策的根本原因在于：当上下文扩展到1M token时，序列维度的压缩比头维度的压缩带来更显著的效率提升。

**Compressed Sparse Attention (CSA)** 是精细压缩层。它将每4个token的KV压缩为1个entry（4:1压缩比），然后通过softmax门控的**FP4 lightning indexer**选择每个query最相关的top-k个压缩entry进行注意力计算  [(Dr. Kaoutar El Maghraoui)](https://kaoutarelmaghraoui.com/blog/deepseek-v4-long-march-open-weight-ai/) 。同时保留**128 token的滑动窗口**确保局部精度。

**Heavily Compressed Attention (HCA)** 是粗粒度压缩层。它以**128:1的压缩比**将token块压缩为全局摘要，然后在压缩后的stream上执行**稠密注意力**  [(Pratham Patel)](https://prathamp.com/blog/deepseek-v4-csa-and-hca/) 。HCA为模型提供低成本的全局上下文视图，与CSA的精确局部分析形成互补。

V4-Pro的61层结构中，前2层使用HCA，之后CSA与HCA严格交替（30层CSA + 29层HCA） [(techjacksolutions.com)](https://techjacksolutions.com/ai-tools/deepseek/deepseek-v4-architecture/) 。V4-Flash的43层结构则前2层使用滑动窗口注意力，之后同样交替CSA/HCA  [(Framia)](https://framia.converge.ai/page/en-US/news/deepseek-v4-model-card) 。

![KV缓存压缩演进.png](/img/user/%E9%99%84%E4%BB%B6/DeepSeek%E6%A8%A1%E5%9E%8B%E6%BC%94%E8%BF%9B%E6%B7%B1%E5%BA%A6%E8%B0%83%E7%A0%94_assets/KV%E7%BC%93%E5%AD%98%E5%8E%8B%E7%BC%A9%E6%BC%94%E8%BF%9B.png)

在**1M token上下文**下，CSA+HCA架构的效率提升极为显著：V4-Pro的单token推理FLOPs仅为V3.2的**27%**，KV缓存仅为V3.2的**10%**；V4-Flash更进一步，FLOPs为V3.2的**10%**，KV缓存为**7%**  [(Framia)](https://framia.converge.ai/page/en-US/news/deepseek-v4-model-card) 。与标准GQA-bf16基线相比，V4的KV缓存仅为其**约2%**  [(techjacksolutions.com)](https://techjacksolutions.com/ai-tools/deepseek/deepseek-v4-architecture/) 。

| 注意力机制 | KV缓存(1M上下文) | 压缩维度 | 代表模型 | 压缩率 |
|-----------|-----------------|---------|---------|--------|
| 标准MHA | ~488 GB  [(LLMS3.com)](https://llms3.com/node/multi-head-latent-attention-mla)  | 无 | GPT-3, LLaMA-1 | 1x |
| GQA | ~122 GB | 头维度 | LLaMA-2/3 | 4x |
| MLA | ~32.7 GB | 头维度(低秩) | DeepSeek V2/V3/R1 | ~15x |
| DSA | ~146 GB | 序列维度(稀疏) | DeepSeek V3.2 | ~3.3x |
| CSA+HCA (V4-Pro) | ~48.8 GB | 序列+头维度 | DeepSeek V4-Pro | ~10x |
| CSA+HCA (V4-Flash) | ~34.2 GB | 序列+头维度 | DeepSeek V4-Flash | ~14x |

---

## 3. 训练基础设施与效率优化

### 3.1 FP8混合精度训练框架

DeepSeek-V3是业界首个在**超大规模模型上验证FP8训练可行性**的实践  [(arXiv.org)](https://arxiv.org/html/2412.19437v1) 。其混合精度框架的核心设计是：计算密集型操作（GEMM）使用FP8，而数值敏感操作保持BF16/FP32。

具体而言，所有三个GEMM操作（前向传播Fprop、激活反向Dgrad、权重反向Wgrad）均以FP8执行，理论上可实现**2倍于BF16的计算速度**  [(arXiv.org)](https://arxiv.org/html/2412.19437v1) 。同时，以下组件保持高精度：Embedding模块、输出头、MoE门控模块、归一化操作和注意力算子。为应对FP8的数值不稳定问题，DeepSeek引入了**细粒度量化策略**：激活值按1×128 tile分组缩放，权重按128×128块分组缩放  [(arXiv.org)](https://arxiv.org/html/2412.19437v1) 。

FP8训练使V3的激活内存占用大幅降低，结合FP8 Wgrad GEMM允许激活以FP8格式存储用于反向传播。实验验证表明，该FP8框架与BF16训练相比，相对误差保持在**0.25%以下**  [(arXiv.org)](https://arxiv.org/pdf/2507.09955?) 。

### 3.2 DualPipe双向流水线并行

DualPipe是DeepSeek-V3训练基础设施中最具原创性的算法创新之一  [(CSDN博客)](https://blog.csdn.net/gitblog_01026/article/details/153176075) 。传统流水线并行（如1F1B）存在两大问题：**流水线气泡**（pipeline bubble，设备空闲等待）和**通信开销**（跨设备数据传输）。

DualPipe通过**双向微批次调度**彻底改变了这一局面。模型被垂直分为两半，前向传播和反向传播在相反方向同时流动。每个设备在某一时刻可能同时处理一个前向微批次和一个反向微批次，实现**计算与通信的完全重叠**  [(python | DeepWiki)](https://deepwiki.com/deepseek-ai/DualPipe/2-dualpipe-architecture) 。

| 方法 | 流水线气泡 | 每设备参数 | 每设备激活 | 所需设备数 |
|------|-----------|-----------|-----------|-----------|
| 1F1B | (PP-1)(F+B) | 1× | PP | PP |
| ZB1P | (PP-1)(F+B-2W) | 1× | PP | PP |
| **DualPipe** | **(PP/2-1)(F&B+B-3W)** | **2×** | **PP+1** | **PP** |
| DualPipeV | (PP/2-1)(F&B+B-3W) | 2× | PP+1 | PP/2 |

DualPipe的代价是每设备需要存储2×参数（因双向执行），但其气泡大小从$O(PP)$降低到$O(PP/2)$，且通过完全重叠通信隐藏了延迟。在V3的2048 H800 GPU集群上，DualPipe实现了**近乎完全的计算-通信重叠**，使每万亿token的训练时间缩短至**3.7天**（约18万H800小时） [(epoch.ai)](https://epoch.ai/gradient-updates/what-went-into-training-deepseek-r1) 。

### 3.3 训练成本的经济学

DeepSeek-V3的完整预训练在**14.8T token**上消耗了**2.664M H800 GPU小时**，按$2/H800小时计算，直接成本约为**$5.58M**  [(epoch.ai)](https://epoch.ai/gradient-updates/what-went-into-training-deepseek-r1) 。后续训练阶段（SFT、RL）仅需约0.1M GPU小时。这一数字与西方前沿模型的训练成本形成鲜明对比：

![训练成本对比.png](/img/user/%E9%99%84%E4%BB%B6/DeepSeek%E6%A8%A1%E5%9E%8B%E6%BC%94%E8%BF%9B%E6%B7%B1%E5%BA%A6%E8%B0%83%E7%A0%94_assets/%E8%AE%AD%E7%BB%83%E6%88%90%E6%9C%AC%E5%AF%B9%E6%AF%94.png)

| 模型 | 估计训练成本 | 架构类型 | 数据来源 |
|------|-------------|---------|---------|
| **DeepSeek-V3** | **$5.58M**  [(epoch.ai)](https://epoch.ai/gradient-updates/what-went-into-training-deepseek-r1)  | MoE, 671B/37B | 官方技术报告 |
| **DeepSeek-R1 (base)** | **$5.58M**  [(VentureBeat)](https://venturebeat.com/ai/deepseek-r1s-bold-bet-on-reinforcement-learning-how-it-outpaced-openai-at-3-of-the-cost/)  | MoE + RL | 官方技术报告 |
| Kimi K2 | ~$460M  [(dataglobehub.com)](https://dataglobehub.com/china-ai-statistics-and-insights/)  | 未披露 | 第三方估计 |
| GPT-4 | ~$100M+  [(introl.com)](https://introl.com/blog/deepseek-v3-2-open-source-ai-cost-advantage)  | 稠密, ~1.7T | 行业估计 |
| Gemini Ultra | ~$191M  [(Allied Insight)](https://alliedinsight.com/blog/deepseeks-technological-innovations-a-deep-dive-into-the-v3-model/)  | MoE | 行业估计 |
| Llama 4 | ~$300M+  [(dataglobehub.com)](https://dataglobehub.com/china-ai-statistics-and-insights/)  | MoE | 行业估计 |

DeepSeek的成本优势来源于三个层面的叠加：**架构层面**（MoE稀疏激活减少计算量）、**工程层面**（FP8+DualPipe提升硬件利用率）、**基础设施层面**（中国地区的GPU和能源成本较低）。值得注意的是，$5.58M仅指V3的预训练成本，不包含前期研究、架构探索和失败的实验。但即使考虑完整研发成本，DeepSeek的效率优势依然显著  [(epoch.ai)](https://epoch.ai/gradient-updates/what-went-into-training-deepseek-r1) 。

---

## 4. 推理模型R1的训练方法论

### 4.1 GRPO：无需Critic模型的强化学习

DeepSeek-R1的核心训练创新是**Group Relative Policy Optimization (GRPO)**  [(arXiv.org)](https://arxiv.org/pdf/2503.09512) ，这是PPO的一个精简变体。传统PPO在LLM训练中需要维护四个模型：策略模型（policy）、价值模型（critic/critic）、参考模型（reference）和奖励模型（reward model）。GRPO的关键突破是**完全消除了critic模型**。

GRPO的数学机制如下：对于每个输入query，从旧策略生成一组响应 $\{o_1, o_2, ..., o_G\}$，通过预定义的奖励函数获得奖励 $\{r_1, r_2, ..., r_G\}$。每个响应的优势函数通过组内归一化计算  [(arXiv.org)](https://arxiv.org/pdf/2503.09512) ：

$$A_i = \frac{r_i - \text{mean}(\{r_1, ..., r_G\})}{\text{std}(\{r_1, ..., r_G\})}$$

GRPO的目标函数为  [(arXiv.org)](https://arxiv.org/pdf/2503.09512) ：

$$J_{GRPO}(\theta) = \frac{1}{N}\sum_{i=1}^{N}\left(\frac{\pi_\theta(o_i|q)}{\pi_{\theta_{old}}(o_i|q)} A_i - \beta \cdot KL(\pi_\theta \| \pi_{ref})\right)$$

这种设计的优势在于：**消除了critic模型的内存和计算开销**（在671B模型尺度上节省数百GB显存）；优势估计基于组内相对奖励，天然归一化且对奖励尺度不敏感；实现更简单，超参数更少。实验表明GRPO在数学和代码推理任务上有效，但也存在**探索受限**和**训练不稳定**的问题  [(arXiv.org)](https://arxiv.org/pdf/2506.22200?) 。

### 4.2 RLVR：可验证奖励的强化学习

DeepSeek-R1采用的训练范式是**Reinforcement Learning with Verifiable Rewards (RLVR)**  [(sebastianraschka.com)](https://magazine.sebastianraschka.com/p/technical-deepseek) 。与传统RLHF使用人类偏好训练的奖励模型不同，RLVR的奖励来自**可符号验证的来源**：数学问题的正确答案（通过计算器验证）、代码的正确性（通过编译器/单元测试验证）。

R1的训练流程包含三个阶段：首先，在V3 base模型上进行**冷启动SFT**（数千条高质量推理数据）；其次，进行大规模**GRPO-RLVR训练**（核心阶段，模型自发涌现出长思维链、自我验证和反思行为）；最后，通过**拒绝采样**生成SFT数据并进行第二轮RL。值得注意的是，R1在RL阶段观察到了**"顿悟时刻"**（Aha Moment）：模型在未明确训练的情况下自发学会了重新评估和修正自己的错误  [(Chat Deep Ai)](https://chat-deep.ai/models/deepseek-r1-0528/) 。

R1-0528在原始R1基础上进一步优化了后训练流程。关键改进包括：增加了推理时的计算投入（AIME问题平均思考长度从**12K token增加到23K token**）；引入**长度惩罚**（length penalty）防止模型过度生成；对一般任务采用**生成式奖励模型**（LLM-as-a-judge）替代纯规则验证  [(sebastianraschka.com)](https://magazine.sebastianraschka.com/p/technical-deepseek) 。这些改进使得R1-0528在AIME 2025上从**70.0%跃升至87.5%**，在HMMT 2025上从**41.7%跃升至79.4%**  [(Chat Deep Ai)](https://chat-deep.ai/models/deepseek-r1-0528/) 。

![R1-0528性能提升.png](/img/user/%E9%99%84%E4%BB%B6/DeepSeek%E6%A8%A1%E5%9E%8B%E6%BC%94%E8%BF%9B%E6%B7%B1%E5%BA%A6%E8%B0%83%E7%A0%94_assets/R1-0528%E6%80%A7%E8%83%BD%E6%8F%90%E5%8D%87.png)

### 4.3 知识蒸馏：R1到小型模型

DeepSeek将R1的推理能力通过**知识蒸馏**传递给小型模型，发布了1.5B至70B参数的蒸馏系列  [(博客园)](https://www.cnblogs.com/ChanYanny/p/18901906) 。蒸馏过程使用R1生成的推理链（Chain-of-Thought）作为监督数据，在Qwen和LLaMA等开源底座上微调。这一策略使得**7B参数的蒸馏模型**在多项推理基准上接近甚至超越原始32B模型的性能  [(artificialanalysis.ai)](https://artificialanalysis.ai/articles/deepseek-r1-update) 。

R1-0528-Qwen3-8B的发布进一步推动了端侧推理能力：8B模型在Artificial Analysis Intelligence Index上达到**52分**，与Qwen3 8B（Reasoning）相当，但比1月的R1-Qwen2.5-32B蒸馏版本更轻量  [(artificialanalysis.ai)](https://artificialanalysis.ai/articles/deepseek-r1-update) 。这意味着在5个月内，实现相同推理能力所需的参数量从32B降至8B（**4倍压缩**）。

---

## 5. 多Token预测 (MTP) 与推理加速

### 5.1 MTP的训练与推理双重价值

DeepSeek-V3引入的**Multi-Token Prediction (MTP)** 目标同时服务于训练效率和推理加速  [(arXiv.org)](https://arxiv.org/pdf/2412.19437) 。与传统自回归模型每步仅预测下一个token不同，MTP通过D个串行的MTP模块，在每个位置额外预测D个未来token（V3中D=2）。

MTP的实现采用**完整的因果链**设计  [(arXiv.org)](https://arxiv.org/pdf/2412.19437) ：第k个MTP模块接收第(k-1)个模块的表示和第(i+k)个token的embedding，通过投影矩阵和共享的Transformer块生成第k深度的表示。所有MTP模块共享embedding层和输出头，与主模型物理共享参数。

MTP的训练收益体现在两方面：一是** densifies训练信号**，每个位置产生多个监督信号，提高数据效率；二是**促进表示预规划**，模型需要为未来token预规划更好的表示。消融实验表明，在小规模（15.7B总参数）和大规模（228.7B总参数）MoE模型上，MTP一致提升了大多数评估基准的性能  [(arXiv.org)](https://arxiv.org/pdf/2412.19437) 。

### 5.2 MTP在推理中的投机解码

MTP模块在推理时可被重新用于**投机解码**（speculative decoding）。V3的第二个token预测接受率在**85%-90%**之间，这使解码速度提升至**1.8倍TPS**（tokens per second） [(arXiv.org)](https://arxiv.org/pdf/2412.19437) 。MTP-Eagle变体进一步优化了投机解码：通过共享KV缓存和链式解码策略，消除了历史状态存储的需求  [(arXiv.org)](https://arxiv.org/pdf/2510.14702) 。

---

## 6. V4的全新架构组件

### 6.1 Manifold-Constrained Hyper-Connections (mHC)

V4引入了**mHC**替代标准残差连接，以稳定深层网络的信号传播  [(ThePlanetTools.ai)](https://theplanettools.ai/blog/deepseek-v4-launch-flash-pro-hybrid-attention-1m-context-april-2026) 。mHC将层间通路宽度扩展4倍，并将通道混合矩阵约束在Birkhoff多面体上（双随机矩阵），使其谱范数上界为1。这一约束防止了信号在极深网络（V4-Pro达61层）中的爆炸或消失。

### 6.2 Muon Optimizer

V4训练将优化器从AdamW切换为**Muon**  [(Effloow)](https://effloow.com/articles/deepseek-v4-pro-mit-frontier-model-developer-guide-2026) ，这是一种近似二阶的优化器族，在2025年的研究中已显示出比AdamW更快的收敛速度和更好的训练稳定性。对于MoE模型的稀疏梯度模式，Muon表现出更强的适应能力。

### 6.3 推理效率与API定价

V4系列的经济性是其在市场上最具竞争力的维度之一。在**1M token上下文**下，V4-Pro的推理成本仅为同等能力闭源模型的**1/6至1/20**：

![API定价对比.png](/img/user/%E9%99%84%E4%BB%B6/DeepSeek%E6%A8%A1%E5%9E%8B%E6%BC%94%E8%BF%9B%E6%B7%B1%E5%BA%A6%E8%B0%83%E7%A0%94_assets/API%E5%AE%9A%E4%BB%B7%E5%AF%B9%E6%AF%94.png)

| 模型 | 输入 ($/1M tokens) | 输出 ($/1M tokens) | 上下文窗口 |
|------|-------------------|-------------------|-----------|
| **DeepSeek V4-Flash** | **$0.14**  [(teamai.com)](https://teamai.com/blog/large-language-models-llms/understanding-the-different-deepseek-models/)  | **$0.28**  [(teamai.com)](https://teamai.com/blog/large-language-models-llms/understanding-the-different-deepseek-models/)  | 1M |
| **DeepSeek V4-Pro** | **$1.74**  [(teamai.com)](https://teamai.com/blog/large-language-models-llms/understanding-the-different-deepseek-models/)  | **$3.48**  [(teamai.com)](https://teamai.com/blog/large-language-models-llms/understanding-the-different-deepseek-models/)  | 1M |
| DeepSeek V3.2 | $0.28  [(introl.com)](https://introl.com/blog/deepseek-v3-2-open-source-ai-cost-advantage)  | $0.42  [(introl.com)](https://introl.com/blog/deepseek-v3-2-open-source-ai-cost-advantage)  | 160K |
| Gemini 3.1 Pro | $2.00  [(Mashable)](https://mashable.com/article/deepseek-v4-preview-comparison-chatgpt-claude-gemini)  | $12.00  [(Mashable)](https://mashable.com/article/deepseek-v4-preview-comparison-chatgpt-claude-gemini)  | 1M |
| GPT-5.5 | $5.00  [(Mashable)](https://mashable.com/article/deepseek-v4-preview-comparison-chatgpt-claude-gemini)  | $30.00  [(Mashable)](https://mashable.com/article/deepseek-v4-preview-comparison-chatgpt-claude-gemini)  | 1M |
| Claude Opus 4.7 | $5.00  [(totalum.app)](https://www.totalum.app/blog/deepseek-v4-pro-vs-claude-2026)  | $25.00  [(totalum.app)](https://www.totalum.app/blog/deepseek-v4-pro-vs-claude-2026)  | 1M |

V4-Pro的输入定价比GPT-5.5和Claude Opus 4.7便宜约**2.9倍**，输出定价便宜约**7-8倍**；V4-Flash更是将输出成本降至Claude Opus 4.7的**约1/90**  [(totalum.app)](https://www.totalum.app/blog/deepseek-v4-pro-vs-claude-2026) 。这种定价策略使DeepSeek在Vercel AI Gateway的生产流量中占比已达**17%**（截至2026年6月） [(totalum.app)](https://www.totalum.app/blog/deepseek-v4-pro-vs-claude-2026) 。

---

## 7. 与前沿模型的性能对比

### 7.1 综合能力对比

DeepSeek V4-Pro在多项基准上已接近或达到闭源前沿模型的水平  [(Mashable)](https://mashable.com/article/deepseek-v4-preview-comparison-chatgpt-claude-gemini) ：

| 基准测试 | DeepSeek V4-Pro | GPT-5.5 | Claude Opus 4.7 | Gemini 3.1 Pro |
|---------|-----------------|---------|-----------------|----------------|
| MMLU-Redux | ~93.4 | ~94 | ~94 | ~93 |
| GPQA-Diamond | **90.1** | ~88 | ~89 | ~87 |
| LiveCodeBench | **93.5** | ~92 | ~91 | ~90 |
| SWE-bench Verified | ~70 | ~72 | ~71 | ~69 |
| AIME 2025 | ~88 | ~90 | ~87 | ~89 |
| MRCR 1M | **83.5** | ~80 | ~82 | ~81 |

V4-Pro在**GPQA-Diamond**（科学推理）、**LiveCodeBench**（代码生成）和**MRCR 1M**（1M上下文长文本召回）上取得了领先或并列第一的成绩。V4-Flash虽然参数规模较小（284B/13B），在LiveCodeBench上仍达到**91.6%**，接近V4-Pro水平  [(Framia)](https://framia.converge.ai/page/en-US/news/deepseek-v4-model-card) 。

### 7.2 上下文窗口对比

![上下文窗口演进.png](/img/user/%E9%99%84%E4%BB%B6/DeepSeek%E6%A8%A1%E5%9E%8B%E6%BC%94%E8%BF%9B%E6%B7%B1%E5%BA%A6%E8%B0%83%E7%A0%94_assets/%E4%B8%8A%E4%B8%8B%E6%96%87%E7%AA%97%E5%8F%A3%E6%BC%94%E8%BF%9B.png)

DeepSeek在上下文长度方面经历了快速追赶：从V1的4K到V2/V3的128K（追平GPT-4o），再到V3.2的160K，最终V4实现**1M token**（与Gemini 1.5 Pro持平） [(ThePlanetTools.ai)](https://theplanettools.ai/blog/deepseek-v4-launch-flash-pro-hybrid-attention-1m-context-april-2026) 。值得注意的是，V4的1M上下文不仅是"技术上支持"，更是"经济上可部署"——在1M上下文的实际推理中，V4-Pro的KV缓存仅为V3.2的10%  [(techjacksolutions.com)](https://techjacksolutions.com/ai-tools/deepseek/deepseek-v4-architecture/) 。

---

## 8. 关键技术创新总结

DeepSeek从67B到V4的演进中，以下七项创新构成了其技术护城河：

| 创新 | 引入版本 | 核心贡献 | 影响 |
|------|---------|---------|------|
| **MLA** | V2 (2024.05) | KV缓存低秩压缩，93.3% reduction  [(arXiv.org)](https://arxiv.org/abs/2405.04434v2)  | 定义了DeepSeek式注意力范式 |
| **DeepSeekMoE** | V2 (2024.05) | 256路由专家+共享专家，细粒度稀疏  [(kanerika.com)](https://kanerika.com/blogs/moe-architecture/)  | 实现总/激活参数的大幅解耦 |
| **FP8训练** | V3 (2024.12) | 超大规模FP8混合精度验证  [(arXiv.org)](https://arxiv.org/html/2412.19437v1)  | 训练成本降至$5.58M |
| **DualPipe** | V3 (2024.12) | 双向流水线，完全计算-通信重叠  [(CSDN博客)](https://blog.csdn.net/gitblog_01026/article/details/153176075)  | 集群利用率接近理论上限 |
| **GRPO/RLVR** | R1 (2025.01) | 无需critic的可验证奖励强化学习  [(arXiv.org)](https://arxiv.org/pdf/2503.09512)  | 推理能力涌现，逼近o1 |
| **DSA** | V3.2 (2025.12) | 序列维度的稀疏注意力  [(博客园)](https://www.cnblogs.com/xtkyxnx/p/19571916)  | 长上下文推理成本降低60%+ |
| **CSA+HCA** | V4 (2026.04) | 混合压缩注意力，1M上下文经济部署  [(techjacksolutions.com)](https://techjacksolutions.com/ai-tools/deepseek/deepseek-v4-architecture/)  | 开源模型首次实现1M实用上下文 |

---

## 9. 开源策略与生态影响

DeepSeek自成立之初就采取了激进的开源策略：所有模型权重（从V3开始采用MIT许可证）、技术报告、训练代码和CUDA内核优化均公开发布  [(teamai.com)](https://teamai.com/blog/large-language-models-llms/understanding-the-different-deepseek-models/) 。V4-Pro和V4-Flash的1.6T/284B参数模型以MIT许可证发布在HuggingFace上，允许商业使用、修改和再分发  [(Framia)](https://framia.converge.ai/page/en-US/news/deepseek-v4-model-card) 。

这种开放性产生了深远的生态影响：vLLM、SGLang、TensorRT-LLM等主流推理框架均在V4发布后数日内完成适配；Megatron-LM社区在4月24日当天就启动了V4训练支持PR  [(Github)](https://github.com/NVIDIA/Megatron-LM/issues/4468) 。独立研究者能够在消费级硬件（4×RTX Pro 6000 Blackwell，96GB×4=384GB）上部署V4-Flash，实现200-400并发用户@32K上下文  [(Codersera)](https://codersera.com/blog/deepseek-v4-flash-rtx-pro-6000-blackwell-benchmarks-2026/) 。

DeepSeek的定价策略进一步放大了其影响力：V4-Pro在保持前沿性能的同时，API价格比GPT-5.5和Claude Opus便宜**5-10倍**  [(Mashable)](https://mashable.com/article/deepseek-v4-preview-comparison-chatgpt-claude-gemini) 。这种"前沿性能+开源权重+极低价格"的三重组合，正在重新定义AI行业的竞争规则。
