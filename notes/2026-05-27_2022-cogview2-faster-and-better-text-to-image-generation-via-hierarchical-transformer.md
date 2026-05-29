---
type: paper-note
aliases:
  - "CogView2 Faster and Better Text-to-Image Generation via Hierarchical Transformers"
paper_id: "2022-cogview2-faster-and-better-text-to-image-generation-via-hierarchical-transformer"
title: "CogView2 Faster and Better Text-to-Image Generation via Hierarchical Transformers"
year: 2022
venue: ""
subfield: "Text-to-Image Generation"
topics:
  - "text-to-image"
importance: 5
read_status: "unread"
note_status: "generated"
selected_on: "2026-05-27"
last_reviewed_on: ""
paper: "[[literature/papers/2022-cogview2-faster-and-better-text-to-image-generation-via-hierarchical-transformer]]"
pdf: "[[papers/2022_CogView2 Faster and Better Text-to-Image Generation via Hierarchical Transformers.pdf]]"
tags:
  - paper
  - paper/text_to_image_generation
  - topic/text_to_image
  - year/2022
---
# 论文信息

- 标题：CogView2: Faster and Better Text-to-Image Generation via Hierarchical Transformers
- 年份：2022
- Venue：预印本（arXiv:2204.14217v2, 2022-05-27；后续 NeurIPS 2022 版本也可见）
- 论文页：[[literature/papers/2022-cogview2-faster-and-better-text-to-image-generation-via-hierarchical-transformer]]
- PDF：[[papers/2022_CogView2 Faster and Better Text-to-Image Generation via Hierarchical Transformers.pdf]]
- 关键词：hierarchical transformers；CogLM；LoPAR（local parallel autoregressive）；CUDA local attention；VQ-VAE / VQVAE；text-guided editing / infilling

# 细分领域与重要程度

- 细分领域：Text-to-Image Generation（**大规模离散 token 自回归**路线下的“加速 + 高分辨率 + 可编辑”系统化方案）
- 重要程度：5/5（把“**先低分辨率生成** + **两级超分** + **局部并行自回归**”串成可落地管线，并给出与同期 diffusion 系方法的明确对照）

# 一句话总结

CogView2 先用预训练的跨模态通用语言模型 **CogLM** 在 20×20 离散图像 token 上快速出图，再用“**直接超分** + **LoPAR 迭代超分**”把分辨率抬到 60×60，并通过 **局部注意力 + 局部并行生成**把自回归的速度劣势大幅缓解，同时自然支持文本引导的局部编辑。

# 背景问题（作者要解决什么）

Transformer 自回归（AR）式文生图（如 DALL·E、CogView）在通用域取得突破，但存在三类结构性瓶颈：

1. **慢**：token-by-token 的生成顺序难以吃满 GPU 并行，端到端延迟高。
2. **高分辨率训练昂贵**：self-attention 的时间/显存复杂度为 O(n²)，序列越长越难扩到高分辨率。
3. **单向性带来能力缺口**：raster-scan 的单向 AR 在生成时看不到“右/下”token，天然不擅长 **infilling / 编辑**；也与 MAE/SimMIM 等双向视觉表征学习范式存在鸿沟。

# 核心贡献（按论文主线）

1. **CogLM：跨模态通用语言模型（Cross-modal general Language Model）**  
   用“文本 token + 图像 token 串联成一个序列”，通过**多种掩码策略**统一训练：既能做文生图（mask 全部图像 token），也能做图像补全（mask 随机 patch），还能做图像描述（mask 文本 token）。
2. **层级式生成（hierarchical generation）**  
   把高分辨率生成拆成：  
   (a) 20×20 低分辨率生成（CogLM 直接生成）；  
   (b) 20×20 → 60×60 的**直接超分**（encoder-decoder 化的 CogLM）；  
   (c) 60×60 的**迭代超分**：通过 **LoPAR（local parallel autoregressive）**在局部窗口内保持自回归，在窗口间并行生成，兼顾一致性与速度。
3. **工程级加速与可训练性细节**  
   通过 **CUDA local attention** 支持跨分辨率的局部注意力（SR 阶段）与更高效的推理/训练。
4. **更强的离散视觉 tokenizer**  
   统一 tokenizer `icetk` 覆盖中英文本与图像；图像侧采用 VQ-VAE 并引入感知损失、以及**多压缩率（multi-compression-rate）**设计，为不同阶段的生成/超分提供合适的 token 粒度。

# 方法详解

## 1) CogLM：把“文生图/补全/描述”统一为一个掩码预测框架

CogLM 的关键点不是把 `[MASK]` 写进输入序列，而是：**输入 token 不变，仅通过 attention mask 控制可见性**，让模型在同一套 Transformer 参数下同时具备：

- **文本→图像**：mask 掉图像 token，按自回归方式预测。
- **图像补全 / infilling**：mask 掉图像 token 的随机 patch（块），利用双向上下文完成局部恢复。
- **图像描述（captioning）**：mask 掉文本 token，让图像 token 提供条件。

## 2) 三段式层级生成：从 20×20 快速出图到 60×60 可控细化

### 2.1 第 1 段：20×20 token 的低分辨率生成

低分辨率序列短（20×20=400 个图像 token），可以显著降低 AR 的延迟；并且作者还提出可选的 **post-selection**：用 CogLM 的 captioning 困惑度过滤低质量样本（与 [[literature/papers/2021-cogview-mastering-text-to-image-generation-via-transformers|CogView]] 一脉相承）。

### 2.2 第 2 段：直接超分（20×20 → 60×60）

作者把预训练的 CogLM 改造成 encoder-decoder：  
- encoder 输入 20×20 的低分辨率 token；  
- decoder 输入全 `[MASK]` 的 60×60 token；  
decoder 通过**跨分辨率局部注意力**同时看 encoder 与 decoder 的局部邻域，并用交叉熵 + multinomial 采样逐 token 生成 60×60。

关键观察：如果只做“逐 token 的独立映射”（类似传统 SR），会更像纹理变换而不是生成；因此需要下一段“迭代超分”显式建模 token 依赖。

### 2.3 第 3 段：迭代超分 LoPAR（在局部窗口自回归，在窗口间并行）

LoPAR 的直觉是：层级化后“全局结构”已经由低分辨率阶段确定，因此在超分细化阶段，可以只保证**局部一致性**而不必做全局 token-by-token。

- 做法：在 60×60 上**保留约 25% token 作为上下文**，mask 其余 75% token；在一个局部窗口内按自回归因子分解，但让**不同局部窗口并行生成**。  
- 为减少“相邻 token 同步生成导致的不一致”，作者用对角线式的迭代日程（同色 token 同步生成），在并行与一致性间折中。

# 关键公式或算法直觉

这里把 CogView2 的“速度提升”拆成两层可解释的因子：

1. **序列长度先天变短**：20×20 的第 1 段直接把 AR 代价压到可控范围；之后的高分辨率阶段主要靠 SR，而不是从头长序列 AR。
2. **LoPAR 把全局 AR 改成“局部 AR + 全局并行”**：  
   在局部窗口内保持因果依赖（避免完全独立采样导致的破碎纹理），而窗口间并行生成（让 GPU 并行真正发挥出来）。

# 关键原图讲解

> 说明：本笔记中的部分关键 Figure 属于矢量/组合对象，`extract-images` 直接抽图时不一定能得到完整结构图；因此我对关键页面做了 `pdftocairo` 渲染并裁剪得到下列图（图像文件名以 `fig_*.png` 标记）。

## 图 1：CogLM 的统一训练任务 + attention mask（PDF 第 3 页，Figure 2）

![图：CogLM 的 token 序列与 attention mask](../figures/2022-cogview2-faster-and-better-text-to-image-generation-via-hierarchical-transformer/fig_02_coglm.png)

- 来自 PDF 第 3 页（Figure 2）。
- 展示内容：左侧是“文本 token + `[BOI]` + 图像 token”的拼接序列，以及“采样 mask region 后只预测 mask region 内（接近末尾）的 token”；右侧给出 attention mask 矩阵示意：**mask 通过 attention mask 约束可见性**，而不是把输入 token 替换成 `[MASK]`。
- 如何解读：把它看成“跨模态版 GLM / MAE”——同一模型通过不同 mask 任务覆盖生成、补全、理解（caption）。
- 与核心贡献关系：这是 CogView2 能在后续 SR 阶段“用同一个预训练骨干适配 encoder-decoder / infilling”的基础。

## 图 2：两级超分 + LoPAR 的并行日程（PDF 第 6 页，Figure 4）

![图：直接超分 + 迭代超分（LoPAR）](../figures/2022-cogview2-faster-and-better-text-to-image-generation-via-hierarchical-transformer/fig_04_superres.png)

- 来自 PDF 第 6 页（Figure 4）。
- 展示内容：从 20×20 token 生成到 60×60 token 的直接超分；以及在 60×60 上做迭代超分时，按“同色块同步生成”的方式并行更新多个局部窗口。
- 如何解读：上半部分更像“把低分辨率 token 当条件的条件生成”；下半部分才是 LoPAR 的关键：**保证局部因果一致性的同时尽可能并行**。
- 与核心贡献关系：这是论文 “faster” 的主要来源（把全局 AR 的串行瓶颈推到最短序列上，其余阶段靠局部并行）。

## 图 3：CUDA local attention 的内存/时间收益（PDF 第 7 页，Figure 6）

![图：不同 local attention 实现的显存/耗时对比](../figures/2022-cogview2-faster-and-better-text-to-image-generation-via-hierarchical-transformer/page_007_01_img-007-132.png)

- 来自 PDF 第 7 页（Figure 6）。
- 展示内容：full attention 与不同 receptive field（RF=5/11）下的 PyTorch vs CUDA local attention 的显存与耗时。
- 如何解读：当序列长度随分辨率上升（size²）时，full attention 显存/耗时迅速爆炸；而 CUDA local attention 把局部注意力做成高效 kernel 后，SR 阶段才“跑得动”。
- 与核心贡献关系：LoPAR 与跨分辨率 SR 都依赖局部注意力；没有工程级实现，层级设计很难落地。

## 图 4：人评结果（PDF 第 9 页，Figure 7）

![图：人类偏好评测（清晰度/纹理/相关性等）](../figures/2022-cogview2-faster-and-better-text-to-image-generation-via-hierarchical-transformer/page_009_01_img-009-133.png)

- 来自 PDF 第 9 页（Figure 7）。
- 展示内容：对 DF-GAN、LAFITE、CogView-v1、CogView-v2 等进行大规模 COCO caption 人评，包含“最佳占比”和多维打分。
- 如何解读：作者强调一个“指标错配”现象：在 COCO 上 fine-tune 可以明显改善 FID，但人评偏好可能变差（更贴近 COCO 风格但主体不够讨好）。
- 与核心贡献关系：论文的目标并不只是追逐单一 machine metric，而是希望在速度/质量/可编辑性之间给出更均衡的系统方案。

## 图 5：multi-compression-rate VQVAE 设计（PDF 第 13 页，Figure 9）

![图：多压缩率 VQVAE（多尺度离散 token）](../figures/2022-cogview2-faster-and-better-text-to-image-generation-via-hierarchical-transformer/fig_09_multi_compression.png)

- 来自 PDF 第 13 页（Figure 9）。
- 展示内容：同一个图像 tokenizer 产生多种压缩率（不同分辨率的 latent 网格），并配合不同 decoder 复原。
- 如何解读：这相当于把“层级生成”往 tokenizer 侧也推进了一步：**不同阶段使用合适粒度的 token**，既保全局结构又留局部细节空间。
- 与核心贡献关系：层级生成不仅是“多级 SR”，还需要“多粒度 token 表示”来支撑计算与表达的折中。

## 图 6：文本引导局部编辑 / infilling 示例（PDF 第 14 页，Figure 10）

![图：text-guided infilling（局部补全/编辑）](../figures/2022-cogview2-faster-and-better-text-to-image-generation-via-hierarchical-transformer/page_014_01_img-014-136.png)

- 来自 PDF 第 14 页（Figure 10）。
- 展示内容：给定原图与目标文本，在指定框区域内进行内容替换；对比“只填右下角”等受限填充策略，展示更强的上下文一致性。
- 如何解读：核心在于 CogLM/LoPAR 的“（近似）双向上下文能力”——编辑区域的生成可以同时利用来自各方向的上下文信息，而不是严格从左上到右下的单向因果链。
- 与核心贡献关系：这是 “naturally supports interactive text-guided editing” 的直接证据。

# 实验与评价

1. **机器指标（如 FID）与人评可能背离**：作者观察到在 MS-COCO 上 fine-tune 能显著降低 FID，但人类偏好不一定更好（Figure 7 附近讨论）。
2. **人评维度更贴近使用体验**：清晰度、纹理质量、与 caption 的相关性等维度，能更好反映文生图系统的综合质量上限。
3. **工程指标（显存/耗时）是系统能否扩分辨率的硬约束**：local attention 的 kernel 级优化是层级 SR 能跑通的关键（Figure 6）。

# 局限性

- **第 1 段仍是全局 AR**：虽然序列较短（20×20），但“从零到一”的生成仍存在串行瓶颈；要进一步加速可能还需要更多层级或不同范式（作者在 Discussion 中与 diffusion 对比）。
- **局部窗口假设带来折中**：LoPAR 以局部一致性换并行度，理论上可能在长程依赖（大物体全局结构）上受限，需要靠“保留 token 上下文 + 层级结构”兜底。
- **系统复杂度较高**：tokenizer + 三段式模型 + CUDA kernel，使得复现和工程落地门槛更高。

# 与库中相关论文的关系

- 与上一代：[[literature/papers/2021-cogview-mastering-text-to-image-generation-via-transformers|CogView]]  
  CogView2 延续“离散 token + 大 Transformer”的路线，但通过层级化 + LoPAR 主要解决“慢 + 高分辨率难”的问题，并显式支持编辑。
- 与并行 masked 生成：[[literature/papers/2022-maskgit-masked-generative-image-transformer|MaskGIT]]  
  二者都在做“并行化”的尝试：MaskGIT 用迭代 mask 预测并行生成；CogView2 用 LoPAR 在局部窗口并行。
- 与 tokenizer 路线：[[literature/papers/2021-taming-transformers-for-high-resolution-image-synthesis|Taming Transformers / VQGAN]]、[[literature/papers/2022-vector-quantized-image-modeling-with-improved-vqgan|Improved VQGAN]]  
  CogView2 的贡献更多在“如何用 tokenizer + 层级生成把系统跑快”，而不是纯粹的 tokenizer 提升，但其 multi-compression-rate VQVAE 是对 token 表示的进一步工程化。
- 与 diffusion 系层级化：[[literature/papers/2022-hierarchical-text-conditional-image-generation-with-clip-latent|Hierarchical Text-Conditional Image Generation with CLIP Latent]]、以及库内 diffusion note：[[notes/2026-05-26_2024-sana-efficient-high-resolution-image-synthesis-with-linear-diffusion-transformer|Sana]]  
  CogView2 强调“AR 也能通过层级 + 并行超分接近 diffusion 的交互速度”，与 diffusion 的“多步迭代但每步并行”形成对照。

# 后续阅读建议

1. 回读“并行生成”的两条分支：[[literature/papers/2022-maskgit-masked-generative-image-transformer|MaskGIT]]（mask-predict 并行） vs CogView2（LoPAR 局部并行 AR），对比它们对一致性与速度的取舍。
2. 深入 tokenizer：从 [[literature/papers/2021-taming-transformers-for-high-resolution-image-synthesis|VQGAN]] 到 multi-compression-rate VQVAE（Figure 9），理解“token 粒度”如何决定生成系统的计算形态。
3. 对照 diffusion 的采样并行性：对比 CogView2 Discussion 中对 Glide / DALL·E 2 的分析，再结合库里更近期的高分辨率 T2I（如 Sana）理解路线演化。



