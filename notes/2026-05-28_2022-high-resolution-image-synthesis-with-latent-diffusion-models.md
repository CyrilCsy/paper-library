---
type: paper-note
aliases:
  - "High-Resolution Image Synthesis with Latent Diffusion Models"
paper_id: "2022-high-resolution-image-synthesis-with-latent-diffusion-models"
title: "High-Resolution Image Synthesis with Latent Diffusion Models"
year: 2022
venue: "CVPR"
subfield: "Diffusion / Flow Models"
topics:
  - "diffusion-flow"
  - "text-to-image"
importance: 5
read_status: "unread"
note_status: "generated"
selected_on: "2026-05-28"
last_reviewed_on: ""
paper: "[[literature/papers/2022-high-resolution-image-synthesis-with-latent-diffusion-models]]"
pdf: "[[papers/2022_High-Resolution Image Synthesis with Latent Diffusion Models_CVPR.pdf]]"
tags:
  - paper
  - paper/diffusion_flow_models
  - topic/diffusion_flow
  - topic/text_to_image
  - venue/cvpr
  - year/2022
---
# 论文信息

- 标题：High-Resolution Image Synthesis with Latent Diffusion Models
- 年份：2022
- Venue：CVPR 2022
- 论文页：[[literature/papers/2022-high-resolution-image-synthesis-with-latent-diffusion-models]]
- PDF：[[papers/2022_High-Resolution Image Synthesis with Latent Diffusion Models_CVPR.pdf]]
- 关键词：latent diffusion，perceptual compression（VQGAN/KL-AE），cross-attention conditioning，text-to-image

# 细分领域与重要程度

- 细分领域：Diffusion / Flow Models（也是 [[literature/papers/2024-sana-efficient-high-resolution-image-synthesis-with-linear-diffusion-transformer|SANA]]、[[literature/papers/2024-scaling-rectified-flow-transformers-for-high-resolution-image-synthesis|Rectified Flow Transformer scaling]] 等高分辨率路线的重要“祖先节点”）
- 重要程度：5/5（把“扩散=像素空间慢且贵”的痛点，通过“先压缩到潜空间 + 再在潜空间扩散”系统性解决；后续 Stable Diffusion 系列范式基本都沿这条管线）

# 一句话总结

LDM 的核心是把“细节的像素级生成”拆成两段：先用感知损失/对抗训练的自编码器把图像压到低维潜空间（保留可感知细节、丢掉不重要高频），再在潜空间做扩散去噪；同时用 cross-attention 把文本/语义图等条件注入 UNet，使扩散模型成为通用条件生成器。

# 背景问题（作者要解决什么）

扩散模型（像 DDPM/DDIM）在像素空间训练与采样都很重：

- 计算与显存压力大：高分辨率下，UNet 每一步在大张量上做卷积/注意力，训练慢、采样更慢（要很多步）。
- “该学什么细节”不清晰：像素空间把大量不可感知的高频/噪声细节也当作目标，等于把算力浪费在“人眼不在乎”的部分。
- 条件控制的通用性不足：早期扩散主要做类条件或简单图像条件；文本等更复杂条件需要更通用的机制。

# 核心贡献（按论文主线）

1. **潜空间扩散（Latent Diffusion Models）**：在固定的第一阶段自编码器潜空间里训练扩散模型，显著提升训练/采样效率，同时保持质量。
2. **“感知压缩 vs 语义压缩”的分工视角**：第一阶段负责“感知压缩”（perceptual fidelity），扩散模型负责“语义建模”（semantic / conceptual bits），并系统分析下采样因子 f 的权衡（LDM-4/8 通常更合适）。
3. **通用条件注入：cross-attention conditioning**：在 UNet 中间特征上做 cross-attention，把文本/语义图/其它模态条件统一为“可插拔条件编码器 + 注意力注入”。
4. **多任务适配展示**：除无条件/类条件生成，还演示 text-to-image、超分辨率、inpainting 等（同一套“潜扩散 + 条件机制”模板）。

# 方法详解

## 1) 两阶段管线：先压缩，再扩散

把图像生成拆成两段：

- **第一阶段（perceptual autoencoder）**：训练编码器/解码器 `E/D`，把图像 `x` 压到潜变量 `z = E(x)`，再用 `D(z)` 还原图像。这里的训练目标强调“感知相似”（perceptual loss）并可加入 GAN（VQGAN 风格）或轻量 KL 正则（连续潜空间）。
- **第二阶段（diffusion in latent）**：固定住第一阶段的 `E/D`，只在 `z` 空间训练扩散 UNet：前向过程往 `z` 里逐步加噪，反向过程由时间条件 UNet 预测噪声并迭代去噪；最后用 `D(z0)` 一次性解码回像素空间。

直觉：**把“人眼看不出来的高频细节”交给第一阶段丢掉/重建，把算力集中在“语义结构与可感知细节的分布建模”**。

## 2) LDM 的训练目标（噪声预测式）

扩散模型常用简化目标是预测噪声（论文中对应 Eq. (1) / (2) 的形式）：

- 像素空间：`L_DM = E_{x,ε,t} || ε - ε_θ(x_t, t) ||^2`
- 潜空间：把 `x_t` 换成 `z_t`，即 `L_LDM = E_{E(x),ε,t} || ε - ε_θ(z_t, t) ||^2`

这里 `t` 在 `1..T` 均匀采样，`ε_θ` 是时间条件 UNet。

## 3) 条件生成：用 cross-attention 注入条件

论文给了两类条件注入思路：

- **拼接（concatenation）**：把与空间对齐的条件（如低分辨率图、语义图）直接 concat 到 UNet 输入通道。
- **cross-attention（更通用）**：用条件编码器 `τ_θ(y)` 把条件 `y`（如文本）编码成 token 序列，作为 K/V；UNet 中间特征展平成 Q，在多层插入 cross-attention：`softmax(QK^T/√d) V`。

备注：论文正文里的 Figure 3 是“concat vs cross-attention”结构示意，但本次 `extract-images` 自动提取未得到完整结构图（很可能是 PDF 的矢量/组合对象导致的提取缺失）；下面用文字把结构要点补齐。

## 4) 关键权衡：下采样因子 f（压得太狠/太轻都不行）

潜空间的空间分辨率通常是原图的 `1/f`（例如 `f=4/8/16/32` 等）。这带来两端问题：

- `f` 太小：潜空间还很大，扩散仍然很贵（“感知压缩”没做够）。
- `f` 太大：第一阶段丢信息太多，扩散再强也学不回（“感知压缩”过头）。

所以论文强调要在“感知压缩（第一阶段）”和“语义压缩（扩散建模）”之间找到平衡点，经验上 LDM-4/8 往往是更好的折中。

# 关键公式或算法直觉

把 LDM 看成三个可替换模块的组合（也是后续工业实践最常用的“拼装点”）：

1. **Tokenizer / Autoencoder（E/D）**：决定“压缩率与可逆性”，也决定潜空间里哪些细节被视为“可感知”。
2. **Diffusion UNet（ε_θ）**：决定“在潜空间里如何建模分布”；同等算力下潜空间越小越占便宜。
3. **Conditioner（τ_θ + cross-attention / concat）**：决定“如何把条件信息注入生成过程”，并影响可控性与泛化到新条件模态的难易度。

直觉上：**第一阶段越像“语义级压缩”（丢掉大量可感知信息），第二阶段越难；第一阶段越像“无损压缩”，第二阶段越贵。**

# 关键原图讲解

## 图 1：感知压缩 vs 语义压缩的分工（Rate-Distortion 视角）

![图：感知压缩 vs 语义压缩](../figures/2022-high-resolution-image-synthesis-with-latent-diffusion-models/page_002_01_img-002-012.jpg)

- 来源：PDF 第 2 页（`page_002_01_img-002-012.jpg`）
- 展示内容：横轴是码率（bits/dim），纵轴是失真（RMSE）；曲线上标出两段区域：右下角是 **Perceptual Compression（Autoencoder+GAN）**，左侧陡峭区域是 **Semantic Compression（Generative Model: LDM）**。
- 如何解读：
  - **右下角区域**：通过自编码器把“像素细节”压缩到一个更小但仍能感知还原的潜空间（失真低、码率下降）。
  - **左侧陡峭区域**：进一步压缩会迅速增大失真；这部分不再适合靠自编码器硬压，而更适合交给生成模型学习“语义分布”，通过采样来补回合理细节。
- 与核心贡献的关系：这张图把 LDM 的核心动机讲得非常直观——**生成模型不该被迫去做“感知压缩”这种低效工作；应当把感知压缩交给专门的 autoencoder，把生成模型算力留给语义建模。**

## 图 2：同等算力下的质量-训练效率轨迹（FID vs V100 days）

![图：FID vs V100 days](../figures/2022-high-resolution-image-synthesis-with-latent-diffusion-models/page_022_01_img-022-169.jpg)

- 来源：PDF 第 22 页（`page_022_01_img-022-169.jpg`）
- 展示内容：不同配置的 LDM（LDM-1/2/4/8/16/32）在训练进度（V100 days）上的 FID 变化曲线。
- 如何解读：
  - 曲线越低越好，代表在同等算力投入下更快达到更好的生成质量。
  - 对比能看到：过小的压缩（如 LDM-1/2）训练更慢；过大的压缩（如 LDM-32）质量更难继续提升；中间区间（如 LDM-4/8/16）更像“甜点区”。
- 与核心贡献的关系：对应论文对下采样因子 `f` 的系统分析，支持“**LDM-4/8 是经验上更稳的折中点**”这一结论。

## 图 3：质量指标的另一侧（Inception Score vs V100 days）

![图：Inception Score vs V100 days](../figures/2022-high-resolution-image-synthesis-with-latent-diffusion-models/page_022_02_img-022-170.jpg)

- 来源：PDF 第 22 页（`page_022_02_img-022-170.jpg`）
- 展示内容：相同一组模型配置下，Inception Score（越高越好）随训练进度的变化。
- 如何解读：与图 2 结合看，可以把“训练效率—质量”的趋势从另一个指标侧面验证：**潜空间扩散让同样的训练预算更快拉高样本质量/可分性指标**。
- 与核心贡献的关系：强调 LDM 的价值不仅是“跑得快”，而是在质量指标上也能达到/接近更强的水平。

# 实验与评价（读者应带走什么）

- LDM 在多个数据集上展示了“效率提升 + 质量不降”的趋势；论文里还报告在 CelebA-HQ 256×256 上达到很强的 FID（作者宣称当时 SOTA 量级）。
- 实验主线很清晰：先做 `f` 的权衡（为什么 LDM-4/8 更好），再展示条件生成与下游任务（text-to-image、SR、inpainting）。
- 对工程实践的启发非常直接：只要有一个足够强的 autoencoder（后续常见的是 KL-AE / VAE），就能把扩散模型从像素空间“搬家”到潜空间，立刻得到可扩展性。

# 局限性（论文没解决/仍然麻烦的点）

- **第一阶段是瓶颈**：autoencoder 的压缩-失真曲线决定上限；压得不够省不了算力，压得太狠会产生不可逆信息损失（尤其是高频纹理与小物体）。
- **结构图/矢量图的可复现性问题**：不少关键结构示意在 PDF 中是矢量/组合对象，自动提取图片时可能不完整（本次就缺失了 Figure 3 的完整结构图）；阅读时需要结合正文描述。
- **采样仍需多步**：潜空间降低了单步开销，但“扩散需要迭代”的基本形态仍在；后续工作会继续从采样步数、蒸馏、ODE/flow 等方向提速。

# 与库中相关论文的关系

- 与视觉 tokenizer / VQGAN 路线：第一阶段与 [[literature/papers/2021-taming-transformers-for-high-resolution-image-synthesis|Taming Transformers for High-Resolution Image Synthesis]]、[[literature/papers/2022-vector-quantized-image-modeling-with-improved-vqgan|Improved VQGAN]] 等强相关（同样强调“先把像素压到更好建模的潜空间”）。
- 与扩散“打败 GAN”叙事：质量指标与讨论可对照 [[literature/papers/2021-diffusion-models-beat-gans-on-image-synthesis|Diffusion models beat GANs on image synthesis]]。
- 与后续高分辨率扩散/流模型加速：[[literature/papers/2024-sana-efficient-high-resolution-image-synthesis-with-linear-diffusion-transformer|SANA]]、[[literature/papers/2024-scaling-rectified-flow-transformers-for-high-resolution-image-synthesis|Scaling Rectified Flow Transformers]]、以及更偏工程的 [[literature/papers/2025-deep-compression-autoencoder-for-efficient-high-resolution-diffusion-models|Deep Compression Autoencoder]]，都可以视为在 LDM 的“压缩 + 潜生成”范式上继续扩展（结构从 UNet 到 Transformer/Flow、压缩器更强、采样更快）。

# 后续阅读建议（按学习路径）

1. 先读懂“为什么要潜空间扩散”：围绕图 1 的 rate-distortion 直觉 + `f` 的折中结论。
2. 再把条件机制吃透：cross-attention 的注入位置（UNet 各尺度特征）与 `τ_θ(y)` 的职责（把任意条件变成 token 序列）。
3. 然后读与之配套的 tokenizer / autoencoder：VQGAN/KL-AE 训练目标如何影响潜空间的“可建模性”。
4. 最后沿着“更快采样/更强 backbone/更好压缩器”三条线看后续：从 LDM 到更大规模的文本条件潜扩散系统（例如 Stable Diffusion 系列及其改进工作）。

