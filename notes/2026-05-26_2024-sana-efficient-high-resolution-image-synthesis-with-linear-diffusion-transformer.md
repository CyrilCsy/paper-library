---
type: paper-note
aliases:
  - "SANA Efficient High-Resolution Image Synthesis with Linear Diffusion Transformers"
paper_id: "2024-sana-efficient-high-resolution-image-synthesis-with-linear-diffusion-transformer"
title: "SANA Efficient High-Resolution Image Synthesis with Linear Diffusion Transformers"
year: 2024
venue: ""
subfield: "Diffusion / Flow Models"
topics:
  - "diffusion-flow"
  - "text-to-image"
importance: 5
read_status: "unread"
note_status: "generated"
selected_on: "2026-05-26"
last_reviewed_on: ""
paper: "[[literature/papers/2024-sana-efficient-high-resolution-image-synthesis-with-linear-diffusion-transformer]]"
pdf: "[[papers/2024_SANA Efficient High-Resolution Image Synthesis with Linear Diffusion Transformers.pdf]]"
tags:
  - paper
  - paper/diffusion_flow_models
  - topic/diffusion_flow
  - topic/text_to_image
  - year/2024
---
# 论文信息

- 标题：SANA: Efficient High-Resolution Image Synthesis with Linear Diffusion Transformers
- 年份：2024
- Venue：-（预印本 / Technical Report；文中标注 arXiv:2410.10629v3, 2024-10-20）
- 论文页：[[literature/papers/2024-sana-efficient-high-resolution-image-synthesis-with-linear-diffusion-transformer]]
- PDF：[[papers/2024_SANA Efficient High-Resolution Image Synthesis with Linear Diffusion Transformers.pdf]]
- 关键词：diffusion-flow, text-to-image；高分辨率（最高 4096×4096）生成；Linear Attention；DiT；Gemma 文本编码器；Flow-DPM-Solver；端侧量化

# 细分领域与重要程度

- 细分领域：Diffusion / Flow Models（高分辨率 T2I 的“系统 + 算法共设计”路线）
- 重要程度：5/5（把 4K 这条“token 太长导致 attention 爆炸”的矛盾拆到可落地：更强压缩 AE + 线性注意力 DiT + 更快采样 + 端侧部署）

# 一句话总结

Sana 用 **AE-F32 深压缩把 token 数降到传统 LDM 的 1/16**，再用 **线性注意力 + Mix-FFN 的 Linear DiT 把注意力复杂度从 O(N²) 变成 O(N)**，配合 **Gemma 作为文本编码器 + “复杂人类指令(CHI)”提示改写**与 **Flow-DPM-Solver 减步采样**，让 0.6B 级模型可以“质量不掉太多”的前提下直接跑到 4K，并能进一步做 W8A8 量化上端侧。

# 背景问题（作者要解决什么）

高分辨率（2K/4K）文生图的主要瓶颈不是“模型参数”，而是 **token 序列长度**：

- LDM / DiT 通常在 latent 空间做扩散，但主流 AE 压缩因子 F=8（AE-F8）时，4K 图像对应的 latent token 数仍然巨大。
- DiT 的 self-attention 成本随 token 数 **二次增长**，导致 4K 推理延迟与训练成本不可接受。

所以关键矛盾是：**如何在不显著牺牲图像质量/对齐的情况下，把 token budget 和 attention 复杂度压下来**，并让训练/推理链路真正“跑得动”。

# 核心贡献（按论文主线）

1. **Deep Compression Autoencoder（AE-F32C32P1）**：把下采样因子从 8 提到 32，latent token 数直接减少 16×，为 2K/4K 的可训练性与可推理性“清场”。
2. **Efficient Linear DiT**：用 **ReLU Linear Attention** 全面替换 DiT 的二次 attention，并用 **Mix-FFN（DWConv + GLU 等）**补足线性注意力的局部信息与收敛速度问题；额外发现 **NoPE（不加位置编码）**仍可不掉点。
3. **Decoder-only 小 LLM 作为文本编码器（Gemma）+ 复杂人类指令（CHI）**：提升 prompt 理解/指令跟随与对齐，并给出稳定训练的归一化与缩放 trick。
4. **训练/推理共设计**：多 VLM 自动标注多 caption、基于 CLIP-Score 的 caption 采样、级联分辨率训练；以及面向 rectified flow 的 **Flow-DPM-Solver**，把步数从 28–50 降到 14–20。
5. **端侧部署**：W8A8（INT8 权重与激活）量化 + kernel fusion，在消费级 GPU 上实现亚秒级 1024² 生成。

# 方法详解

## 1) Deep Compression Autoencoder：为什么是 F=32？

作者的策略很直接：**不要在 DiT 里用更大 patch 去“偷 token”，而是让 AE 负责压缩，让扩散模型专注去噪**。

- 传统：AE-F8（下采样 8×）+ DiT patch P=2 往往被写成 AE-F8C4P2 / AE-F8C16P2。
- Sana：把 AE 压到 **F=32**，并在 latent 上用 **P=1**（AE-F32C32P1）。
- 直觉：对 4K 来说，token 数是第一矛盾；先砍 token，再谈注意力/FFN 的复杂度才有意义。

文中也强调了一个常见误区：**重建更好（rFID 更低）的 AE，不一定导致生成更好（FID 更低）**；他们观察到“让 AE 专注高压缩、让 DiT 专注去噪”反而更优。

## 2) Linear DiT：线性注意力 + Mix-FFN + NoPE

### 2.1 线性注意力（ReLU linear attention）

将 softmax attention 换成 ReLU 线性注意力，核心是把对每个 query 都要算的项，改成“全局共享项先累积一次，再复用”：

- 共享项：\n
  - \(\sum_j \mathrm{ReLU}(K_j)^T\)\n
  - \(\sum_j \mathrm{ReLU}(K_j)^T V_j\)
- 然后对每个 \(Q_i\) 做一次线性组合即可，整体复杂度从 O(N²) 变成 O(N)。

### 2.2 Mix-FFN：补足线性注意力的“局部信息与收敛”

作者指出线性注意力缺少 softmax 的非线性相似度，容易导致收敛慢/局部建模弱，因此把原先的 MLP-FFN 替换为 Mix-FFN：

- 组成：inverted residual + **3×3 depth-wise conv** + **GLU**。
- 作用：DWConv 提供局部聚合，抵消线性注意力对局部结构的弱建模。

### 2.3 NoPE：为什么可以不加位置编码？

他们发现移除 DiT 的 position embedding 不掉点（NoPE）。论文给出的解释线索是：**Mix-FFN 里的 3×3 卷积（含 padding）会隐式注入位置信息**，从而让显式 PE 变得“不那么必要”。

## 3) 文本侧：Gemma 作为文本编码器 + CHI（复杂人类指令）

### 3.1 为什么不用 T5，而用 decoder-only 小 LLM？

作者认为主流 T2I 仍依赖 CLIP/T5，文本理解与指令跟随偏弱；而 decoder-only LLM（如 Gemma）在推理、ICL、CoT 等能力上更强，且小模型推理速度可接受。

### 3.2 稳定训练的关键 trick

直接把 LLM 的 embedding 喂给 cross-attention 会出现 NaN。作者的解决方案包括：

- 在 LLM 输出后加 **RMSNorm**，把 embedding 方差归一到 1；
- 乘一个 **小的可学习缩放因子**，并用很小的初值（例如 0.01）启动，改善早期训练稳定性与收敛。

### 3.3 CHI：把“提示改写”变成可训练的条件增强

CHI 的形式是给 LLM 一段固定模板，让它把用户简短 prompt 改写成更具体的“增强提示”，从而提升细节与对齐（尤其是短 prompt 场景）。

## 4) 训练/推理：数据标注 + 采样步数优化

### 4.1 多 caption 自动标注 + clipscore 采样

对每张图用多个 VLM 生成多条 caption，然后用 CLIP-Score 做温度采样，倾向抽到更高质量文本：

- 概率形式（论文给出温度 \(\tau\)）：\n
  \[
  P(c_i)=\frac{\exp(c_i/\tau)}{\sum_j\exp(c_j/\tau)}
  \]

### 4.2 Flow-based Training & Flow-DPM-Solver

训练侧采用 rectified flow / velocity 预测范式（与 [[knowledge/flow-matching|Flow Matching]] 这条线高度一致），推理侧把 DPM-Solver++ 改到 rectified flow 的时间参数化与“velocity→data”的换算上，形成 **Flow-DPM-Solver**：

- 经验结果：在更少步数（约 14–20）达到更好的 FID/CLIP，对比 Flow-Euler 往往要 28–50 步。

## 5) 端侧部署：W8A8 + kernel fusion

他们报告在端侧做 INT8 量化（activation per-token、weight per-channel），并保留部分层全精度，同时把线性注意力的关键矩阵乘与投影层融合进 kernel，减少访存与调度开销。

# 关键公式或算法直觉

把 Sana 的“快”拆成三层（你读图 2 时的心智模型）：

1. **token 数降维（AE-F32）**：4K 时 token 数先降 16×，否则任何 attention 优化都只是杯水车薪。
2. **attention 复杂度降阶（Linear Attention）**：把 DiT 的 O(N²) 改成 O(N)，高分辨率收益远大于 1K。
3. **采样步数减少（Flow-DPM-Solver）**：在更少步数达到同等/更好质量，直接影响端到端延迟。

# 关键原图讲解

说明：本论文部分图是矢量/组合对象，`extract-images` 可能无法完整导出整张 Figure；下面选取能完整提取、且最能支撑“核心贡献链条”的 5 张关键图/表。

## 图 1：4K 生成延迟的“分解式加速”（从 1023s 到 9.6s）

![图：4K 生成延迟的分解式加速](../figures/2024-sana-efficient-high-resolution-image-synthesis-with-linear-diffusion-transformer/page_002_01_img-002-013.png)

- 来源：PDF 第 2 页（Figure 2 的一部分）
- 展示内容：以 FLUX 为参照（1023s），把 Sana 的系统设计按模块逐步叠加：Baseline（469s）→ +AE（41s）→ +Linear DiT（24s）→ +Kernel Fusion（21s）→ +Flow-DPM-Solver（9.6s）。
- 如何解读：这是“系统 + 算法共设计”的核心证据链：**先砍 token（AE）得到数量级收益，再用线性 attention 与 kernel fusion 榨出常数，最后用更快 solver 把步数砍半**。
- 与核心贡献关系：把全文的 4 个设计点（AE / Linear DiT / Kernel / Solver）用同一指标（4K 延迟）串起来，解释了为何 0.6B 也能做“真 4K”。

## 图 2：性能-效率的帕累托（GenEval vs Throughput）

![图：性能-效率权衡（GenEval vs Throughput）](../figures/2024-sana-efficient-high-resolution-image-synthesis-with-linear-diffusion-transformer/page_004_01_img-004-017.jpg)

- 来源：PDF 第 4 页（Figure 4 的一部分）
- 展示内容：横轴是吞吐（samples/s），纵轴是 GenEval overall；气泡大小标注不同参数规模（灰色一组是更大参数的对比线，图中可见 0.6B / 4B / 8B / 12B）。
- 如何解读：作者想表达“**不靠堆参数，而是靠 token/复杂度优化**”，把点推到右上角：在较高 GenEval 的同时显著提高吞吐。
- 与核心贡献关系：支撑“Linear DiT + AE”带来的整体帕累托改善，而不是只在某一个指标上投机。

## 图 3：用 LLM 做文本编码器时，为什么必须做归一化与缩放（否则 NaN）

![图：LLM 文本编码器稳定性消融（Norm + scale）](../figures/2024-sana-efficient-high-resolution-image-synthesis-with-linear-diffusion-transformer/page_006_01_img-006-032.jpg)

- 来源：PDF 第 6 页（Table 3）
- 展示内容：不做 Text Embed Norm 会直接 NaN；做 Norm 后再配合 scale factor（如 0.01）能显著改善早期训练与指标（表中按训练步数列出 FID/CLIP 的变化）。
- 如何解读：这张表给了“LLM embedding 数值尺度不匹配”的工程性证据：**把 LLM 当 text encoder 不是换个模型名就完事，数值稳定性是第一坑**。
- 与核心贡献关系：这是 Sana 能把 Gemma 真正用进 cross-attention 的关键技术细节之一。

## 图 4：Flow-DPM-Solver 用更少步数达到更好 FID/CLIP

![图：Flow-Euler vs Flow-DPM-Solver（步数—FID/CLIP）](../figures/2024-sana-efficient-high-resolution-image-synthesis-with-linear-diffusion-transformer/page_007_01_img-007-084.png)

- 来源：PDF 第 7 页（Figure 8 的一部分）
- 展示内容：列出不同采样步数下 Flow-Euler 与 Flow-DPM-Solver 的 FID/CLIP；在 14–20 steps 区间，Flow-DPM-Solver 能达到更优的 FID 且 CLIP 不掉。
- 如何解读：如果你关心“端到端延迟”，这张表的意义是：**质量达到可用阈值的步数越少，latency 越低**；且对高分辨率尤其敏感（每一步都很贵）。
- 与核心贡献关系：把“推理加速”从 kernel 层扩展到“采样算法层”，是图 2 那条 9.6s 的最后一跳。

## 图 5：CHI（复杂人类指令）模板示例：把短 prompt 变成可对齐的“增强提示”

![图：Complex Human Instruction（CHI）模板示例](../figures/2024-sana-efficient-high-resolution-image-synthesis-with-linear-diffusion-transformer/page_005_02_img-005-029.png)

- 来源：PDF 第 5 页（Figure 5/相关说明的局部截图）
- 展示内容：CHI 的提示模板把“用户 prompt”包装进更长的指令，让 LLM 输出更细节化的 enhanced prompt（颜色、材质、空间关系等）。
- 如何解读：这不是“后处理提示工程”，而是把提示增强做成模型训练流程的一部分：**训练时让模型见到更明确的条件描述，推理时也更容易对齐**。
- 与核心贡献关系：解释了为什么作者强调“decoder-only LLM + CHI”是系统级设计，而不仅是换 encoder。

# 实验与评价

## 1) 512² 与 1024² 的综合对比（Table 7 摘要）

论文在 MJHQ-30K、GenEval、DPG-Bench 等指标上对比了主流模型，给出的核心结论是：

- 在 **512×512**：Sana-0.6B（590M 参数）对比 PixArt-Σ（0.6B）给出 **约 5× 吞吐**（6.7 vs 1.5 samples/s），并且在 FID/CLIP/GenEval/DPG 上全面更好（例如 GenEval 0.64 vs 0.52）。
- 在 **1024×1024**：Sana 的延迟与吞吐优势更明显，并宣称与更大的系统（如 FLUX）在对齐指标上接近，但速度快一个数量级。

## 2) 设计消融：LinearAttn / Mix-FFN / kernel fusion / AE-F32

论文还做了模块级消融来说明：

- 只换 linear attention 会更快但可能掉质量；加入 Mix-FFN 能补回质量（但牺牲部分效率）。
- Triton/kernel fusion 带来额外的常数级加速（文中提到约 10%）。
- 从 AE-F8 升级到 AE-F32，MACs 与吞吐可以进一步显著改善（本质是 token 数下降）。

## 3) 端侧：W8A8 量化

论文报告 1024² 生成在端侧从 **0.88s（FP16）→ 0.37s（W8A8）**，代价是 CLIP-Score 与 Image-Reward 轻微下降但基本保持。

# 局限性（读完你应该保留的怀疑点）

1. **“系统共设计”的可复现性**：论文把收益拆到 AE/Linear DiT/Kernel/Solver，但不同实现细节（kernel fusion、Triton、量化）对复现门槛很高。
2. **线性注意力的训练动态**：作者也承认线性 attention 可能收敛慢，需要 Mix-FFN 等补丁；在更复杂数据/更大模型上是否仍稳定，需要外部验证。
3. **高压缩 AE 的细节损失**：AE-F32 再怎么调优，仍可能在细粒度纹理/小目标上形成上限；论文主要强调“生成质量不掉太多”，但对失败模式讨论有限。
4. **文本侧依赖 LLM/提示模板**：CHI 的收益可能与数据、指令模板质量强耦合；模板迁移到其他语言/域时的鲁棒性不明确。

# 与库中相关论文的关系

- 与 [[literature/papers/2022-high-resolution-image-synthesis-with-latent-diffusion-models|Latent Diffusion Models]]：Sana 仍是 latent diffusion/flow 范式，但把“高分辨率”从级联/超分路线推进到“原生 4K 生成”。
- 与 [[literature/papers/2023-scalable-diffusion-models-with-transformers|DiT]]：Sana 属于 DiT 系谱，但围绕“序列长度爆炸”的痛点，把 attention 改成线性并配套局部聚合与 NoPE。
- 与 [[literature/papers/2024-scaling-rectified-flow-transformers-for-high-resolution-image-synthesis|Scaling Rectified Flow Transformers for High-Resolution Image Synthesis]]：共享 rectified flow/velocity 预测这条训练范式；Sana 更强调线性 attention 与端侧落地。
- 与 [[literature/papers/2025-deep-compression-autoencoder-for-efficient-high-resolution-diffusion-models|Deep Compression Autoencoder for Efficient High-Resolution Diffusion Models]]：Sana 的 AE-F32 是“高压缩 AE”路线的代表之一，可对照其在重建/生成间权衡的经验结论。
- 与视频方向的系统论文（如 [[notes/2026-05-14_2025-open-sora-2-0-training-a-commercial-level-video-generation-model-in-200k|Open-Sora 2.0]]、[[notes/2026-05-10_2025-wan-open-and-advanced-large-scale-video-generative-models|Wan]]）：共同点是都把“token 降维 + 训练/推理系统优化”当作决定性瓶颈，只是 Sana 把这套体系落在 T2I/4K 上。

# 后续阅读建议

按“先补底座→再看扩展”的顺序：

1. [[literature/papers/2022-high-resolution-image-synthesis-with-latent-diffusion-models|Latent Diffusion Models]]：理解 latent AE + diffusion 的基础分工。
2. [[literature/papers/2023-scalable-diffusion-models-with-transformers|DiT]]：理解为什么注意力二次复杂度会成为高分辨率瓶颈。
3. [[knowledge/flow-matching|Flow Matching]]：把 rectified flow / velocity 预测与采样器的改法串起来（读 Sana 的 Flow-DPM-Solver 更顺）。
4. [[literature/papers/2024-scaling-rectified-flow-transformers-for-high-resolution-image-synthesis|Scaling Rectified Flow Transformers for High-Resolution Image Synthesis]]：对照“flow 范式 + 高分辨率”的另一条实现路线。
5. 如果你更关心部署/压缩：[[literature/papers/2025-vq4dit-efficient-post-training-vector-quantization-for-diffusion-transformers|VQ4DiT]]（DiT 量化视角）与 Sana 的 W8A8 kernel fusion 思路互补。

