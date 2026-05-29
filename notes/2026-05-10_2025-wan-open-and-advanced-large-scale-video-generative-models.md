---
type: paper-note
aliases:
  - "Wan Open and Advanced Large-Scale Video Generative Models"
paper_id: "2025-wan-open-and-advanced-large-scale-video-generative-models"
title: "Wan Open and Advanced Large-Scale Video Generative Models"
year: 2025
venue: "CVPR"
subfield: "Video Generation"
topics:
  - "text-to-video"
importance: 5
read_status: "unread"
note_status: "generated"
selected_on: "2026-05-10"
last_reviewed_on: ""
paper: "[[literature/papers/2025-wan-open-and-advanced-large-scale-video-generative-models]]"
pdf: "[[papers/2025_Wan Open and Advanced Large-Scale Video Generative Models_CVPR.pdf]]"
tags:
  - paper
  - paper/video_generation
  - topic/text_to_video
  - venue/cvpr
  - year/2025
---
# Wan: Open and Advanced Large-Scale Video Generative Models 精讲笔记

- 论文: 2025_Wan Open and Advanced Large-Scale Video Generative Models_CVPR.pdf
- 领域: Video Generation
- 重要程度: 5/5
- 阅读状态: 未读；已生成精讲笔记
- 生成日期: 2026-05-10

## 一句话总结

Wan 是阿里开源的一套大规模视频生成基础模型系统。它的核心价值不只是提出一个 T2V 模型，而是把视频生成所需的关键工程链条系统化：数据清洗与稠密 caption、3D causal VAE、基于 DiT 的视频扩散/[[knowledge/flow-matching|Flow Matching]] 主干、训练并行、推理加速、Prompt 改写、自动评价基准，以及图生视频、视频编辑、个性化等下游任务。

如果只把它当作“一个更强的视频生成模型”，会漏掉论文的重点。它更像是一份开放视频基础模型的系统设计报告：哪些模块真的卡住了可扩展训练、哪些优化能让模型在消费级显卡上跑起来、以及如何用 benchmark 和人评证明开放模型已经接近甚至超过商业闭源模型。

## 背景问题

视频生成比图像生成难，主要有三层瓶颈：

1. 数据比图像更脏：网页视频常见水印、黑边、抖动、低清、过曝、NSFW、合成图污染、弱 caption 等问题。
2. 序列太长：视频 latent 的 token 数可以达到几十万甚至百万级，DiT 的 attention 成本随序列长度平方增长。
3. 评价更复杂：FID/FVD 不能很好反映人类对动态质量、物理合理性、镜头控制、文本跟随、ID 一致性等维度的偏好。

Wan 的路线是工程上很务实的 “全栈解法”：先把数据、caption 和 VAE 做扎实，再用可扩展的 DiT/[[knowledge/flow-matching|Flow Matching]] 训练主模型，最后用缓存、量化和多 GPU 并行解决推理成本。

## 核心贡献

1. 开源 1.3B 和 14B 两个规模的视频基础模型。1.3B 版本强调消费级可用，论文称只需约 8.19 GB VRAM；14B 版本强调性能上限。
2. 设计 Wan-VAE：一个 3D causal VAE，把视频压缩到时间 4 倍、空间 8×8 倍的 latent 空间，latent channel 为 16，并用 feature cache 支持长视频分块推理。
3. 使用 DiT + [[knowledge/flow-matching|Flow Matching]] 作为主生成框架：视频经 Wan-VAE 编码成 latent，DiT 在 latent token 上预测 velocity，并用 umT5 文本编码器注入条件。
4. 构建大规模数据处理链路：预训练阶段做基础质量过滤、视觉质量/运动质量筛选、视觉文本数据合成；后训练阶段用更高质量的图像和视频数据提升保真度和运动表现。
5. 自建 dense video caption 模型：用 LLaVA 风格的视觉编码器 + Qwen LLM，对图像/视频生成稠密描述，使训练 caption 更接近高质量提示词。
6. 提出 Wan-Bench：从动态质量、图像质量、指令跟随三大类和 14 个细粒度指标评价视频模型。
7. 系统做训练和推理优化：2D context parallel、FSDP、activation offloading、diffusion cache、FP8 GEMM、8-bit FlashAttention 等。

## 方法详解

### 1. 数据管线：先解决“学什么”

论文把数据分成预训练数据和后训练数据。

预训练数据目标是“大而干净”。过滤维度包括 OCR 文本覆盖率、审美分、NSFW、水印/Logo、黑边、过曝、合成图污染、模糊、视频时长和分辨率。作者提到基础过滤会移除约 50% 初始数据，之后再做语义层面的质量选择。

后训练数据目标是“高质量和高运动”。图像侧选高质量构图、细节和类别覆盖；视频侧按视觉质量和运动质量选出简单运动与复杂运动视频，并保持 12 个大类的数据均衡。

这里值得注意的是，Wan 特别强调视觉文本生成：一方面合成大量含中文字符的白底文字图，另一方面从真实数据中用 OCR 抽取中英文文本，再让 Qwen2-VL 生成包含精确文字内容的描述。这解释了为什么论文把“中英文视觉文字生成”作为重要卖点。

### 2. Dense Caption：把训练 caption 拉近用户提示词

原始网页 caption 通常太短，不能描述动作、镜头、数量、OCR、风格、场景等细节。Wan 因此训练内部 caption 模型。

模型结构是 LLaVA 风格：ViT 提取图像/视频帧特征，两层 MLP 对齐到 Qwen LLM。视频侧每秒采样 3 帧，上限 129 帧，并用 slow-fast 编码：每 4 帧保留一次原分辨率，其余帧做全局平均池化，减少 visual token 数。

这个设计的意义是：视频生成模型最终学到的不是“短标签到视频”，而是“稠密视觉描述到视频”。这也解释了后面的 Prompt Alignment：推理时用户短 prompt 需要被 LLM 改写成更像训练 caption 的长描述。

### 3. Wan-VAE：压缩视频，同时保留时间因果性

Wan-VAE 把输入视频从像素空间映射到 latent 空间。给定视频形状约为 `(1+T, H, W, 3)`，编码后时间维压缩为 `1+T/4`，空间维压缩为 `H/8, W/8`，latent channel 为 16。

它是 3D causal VAE，关键点有三条：

1. 第一帧只做空间压缩，便于同时处理图像数据。
2. 用 RMSNorm 替代 GroupNorm，以保持时间因果性。
3. 在 causal convolution 中引入 feature cache，使长视频可以 chunk-wise 编码/解码，每个 chunk 只处理与单个 latent 对应的少量帧，同时缓存前序帧特征来保持跨 chunk 连续性。

训练分三阶段：先训练 2D image VAE，再 inflate 成 3D causal VAE 并在低分辨率短视频上训练，最后在高质量多分辨率视频上 fine-tune，并加入 3D discriminator 的 GAN loss。损失包括 L1 reconstruction、KL、LPIPS，后期再加入 GAN loss。

论文报告 Wan-VAE 只有 127M 参数，在 720×720、25 帧重建评估中兼顾 PSNR 和速度，并声称重建速度比 Hunyuan Video 的 VAE 快约 2.5 倍。

### 4. 主模型：DiT + [[knowledge/flow-matching|Flow Matching]]

Wan 主体由三部分组成：

1. Wan-VAE：把视频变成 latent token。
2. umT5 text encoder：编码文本条件，强调中英文和视觉文字理解能力。
3. Video Diffusion Transformer：在 latent token 序列上建模时空关系。

DiT 的 patchify 模块使用 kernel 为 `(1, 2, 2)` 的 3D convolution，把 latent 转成序列。每个 block 里有 self-attention 建模视频 token 间关系，cross-attention 注入文本条件，time embedding 通过 MLP 产生调制参数。

一个重要工程细节是 AdaLN 参数共享。论文说共享调制 MLP 能减少约 25% 参数，并且 ablation 显示在相同参数规模下，把参数用在增加网络深度上，比把参数堆在不共享 AdaLN 上更有效。

训练目标使用 [[knowledge/flow-matching|Flow Matching]]。可以把它理解为：不直接预测噪声，而是在噪声 `x0` 与真实 latent `x1` 之间的连续路径上，让模型学习从当前点 `xt` 指向目标分布的 velocity。论文中损失形式是预测 velocity 与目标 velocity 的 MSE。

训练 curriculum 也很关键：先用 256px 文生图预训练 14B 模型，建立文本语义对齐和几何结构能力；再做图像-视频联合训练，分阶段提升空间分辨率和视频时长，最后到 720px、5 秒视频。

## 关键原图讲解

说明：这部分图片来自 PDF 内部可提取的原始嵌入图。Wan 论文里有些结构图和曲线图是矢量对象或组合对象，自动提取时不一定能得到完整 Figure；下面优先选用已经成功提取的关键图辅助理解。

![图：Wan 数据训练流程](../figures/2025-wan-open-and-advanced-large-scale-video-generative-models/page_007_01_Im3.png)

这张图来自 PDF 第 7 页，展示了 Wan 从原始数据池到不同训练阶段的数据供给流程。左侧把数据分为 image、video、textual data，之后经过去重、fundamental filter、visual filter、motion filter、resolution filter 等步骤，最终进入 192P、480P、720P 以及后训练阶段。它对应论文最核心的系统观点：视频生成效果不是单靠模型结构堆出来的，数据质量、分辨率 curriculum、运动质量筛选和视觉筛选共同决定最终上限。

![图：Wan 生成样例 1](../figures/2025-wan-open-and-advanced-large-scale-video-generative-models/page_004_07_Im7.jpg)

这张图来自 PDF 第 4 页，是论文展示的生成样例之一。它体现了 Wan 对场景构图、主体关系和浅景深质感的控制能力。阅读时不要只看“图像是否好看”，还要注意视频模型在多帧生成中需要维持主体身份、动作连续性和背景一致性，这些静态帧只是动态质量的一个切片。

![图：Wan 生成样例 2](../figures/2025-wan-open-and-advanced-large-scale-video-generative-models/page_004_08_Im8.jpg)

这张图同样来自 PDF 第 4 页，重点展示复杂室内场景、多个物体、光效和主体动作。它和论文中的 Prompt Alignment 关系很强：用户短 prompt 通常不足以描述这么多视觉细节，因此 Wan 使用 LLM 把用户提示词改写成更接近训练 caption 分布的长描述，从而提升画面丰富度和动作合理性。

![图：细节重建样例](../figures/2025-wan-open-and-advanced-large-scale-video-generative-models/page_013_01_Image11.jpg)

这张图来自 PDF 第 13 页附近的 VAE 重建对比区域，展示了细粒度纹理场景。Wan-VAE 的目标不是单纯压缩视频，而是在较高压缩率下尽量保留纹理、文字、人脸和高运动场景中的细节。对视频扩散模型来说，VAE 重建能力会直接影响 DiT 学到的 latent 分布；如果 VAE 已经损失大量细节，后续生成主干很难完全补回来。

## 训练与推理加速

训练瓶颈主要在 DiT。论文指出 text encoder 和 VAE encoder 的计算占比低，DiT 占训练总计算超过 85%。对长视频，序列长度 $s$ 很大，attention 计算成本按 $s^2$ 增长。

训练侧的策略：

1. VAE 用普通数据并行。
2. text encoder 和 DiT 使用 FSDP 做参数/梯度/优化器状态切分。
3. DiT activation 用 context parallel 切分序列维，结合 Ulysses 和 Ring Attention 做 2D CP。
4. 对长序列优先用 activation offloading，并结合 gradient checkpointing 控制显存。

推理侧的策略：

1. 多 GPU context parallel 降低单视频延迟。
2. Diffusion cache：复用相邻采样步中高度相似的 attention 输出；在 CFG 后期复用 conditional/unconditional 结果并做 residual compensation。论文报告 14B T2V 推理提升 1.62×。
3. FP8 GEMM：对 DiT block 里的 GEMM 使用 FP8，论文报告 DiT 模块 1.13× 加速。
4. 8-bit FlashAttention：针对原生 FP8 attention 在视频生成中的质量退化，使用混合 8-bit 策略和 FP32 accumulation 处理跨 block reduction，论文称可带来超过 1.27× 效率提升。

## Prompt Alignment

训练 caption 通常很详细，但用户输入往往很短。Wan 用 Qwen2.5-Plus 做 prompt rewriting，把短 prompt 改写为更接近训练 caption 分布的长描述。

改写原则包括：

1. 增加细节但不改变原意。
2. 为主体补充自然运动属性，让视频更流畅。
3. 结构上先描述风格，再给内容摘要，最后给细节描述。

这部分对实际使用非常重要：Wan 的效果不是只来自模型本身，也来自“用户 prompt -> 训练 caption 风格 prompt”的对齐层。

## 评价

Wan-Bench 覆盖三大维度：

1. Dynamic quality：大幅运动、物理合理性、平滑度、像素稳定性、ID 一致性等。
2. Image quality：综合画质、场景质量、风格化能力等。
3. Instruction following：单/多物体、空间位置、镜头控制、动作指令跟随等。

论文在 Wan-Bench 上报告 Wan14B weighted score 为 0.724，高于 Sora 的 0.700、Hunyuan 的 0.673、Mochi 的 0.639 等。VBench 上 Wan14B 总分 86.22%，高于 Sora 的 84.28% 和 Hunyuan 开源版的 83.24%。人评部分也显示 Wan14B 在整体质量、匹配度、动态质量等维度相对多个商业模型有优势。

需要谨慎的是，Wan-Bench 是作者自建基准，虽然维度很细，但仍要关注 prompt 选择、自动打分器偏好和商业模型版本差异。它适合作为强证据之一，而不是唯一结论。

## 和相关论文的关系

- 相比 Latent Diffusion：Wan 继承 latent-space generation 的思路，但把 autoencoder、text encoder、DiT 和长序列训练全面视频化。
- 相比 DiT：Wan 使用 DiT/[[knowledge/flow-matching|Flow Matching]] 作为视频生成主干，重点在百万级 token 场景下的并行和推理优化。
- 相比 HunyuanVideo/Mochi/CogVideoX：Wan 的论文更像系统报告，强调开放权重、VAE 速度、推理效率和 benchmark。
- 相比 DALL-E 3 式 captioning：Wan 把“稠密 caption + prompt rewrite”迁移到视频生成，并加入动作、镜头、OCR、计数等视频特有维度。

## 读这篇时应重点抓住什么

1. Wan 的创新大多不是单点算法，而是系统组合：数据、caption、VAE、DiT、训练并行、推理加速和评价闭环。
2. VAE 是视频生成的关键底座。压缩率、重建质量、时间因果性、长视频 chunk 推理都会直接影响后续 DiT 的可训练性和生成质量。
3. 对视频 DiT，序列长度是核心矛盾。训练吞吐、显存、attention 成本和推理延迟都围绕 token 数展开。
4. Prompt rewriting 不是小技巧，而是训练分布与用户输入分布之间的桥梁。
5. 评价视频模型不能只看 FVD/FID，需要分解到动态、画质、指令、物理、镜头和一致性。

## 可复现/可借鉴点

如果你后续做视频生成或文献综述，Wan 最值得借鉴的是这些工程原则：

1. 先定义数据质量维度，再训练模型。
2. caption 模型和生成模型要联动设计。
3. VAE 的压缩率和重建质量要和 DiT token budget 一起考虑。
4. 高分辨率/长视频训练要 curriculum，而不是一开始直接上最长序列。
5. 大模型推理优化应同时看 attention cache、CFG cache、量化和多 GPU 分片。
6. benchmark 要贴近真实用户关心的能力，而不是只用单一分布距离指标。

## 可能的疑问

1. Wan-Bench 是否会偏向 Wan 的能力分布？需要用第三方 benchmark 和用户盲评进一步验证。
2. 视觉文本能力依赖大量合成文字图和 OCR 数据，这种能力在复杂背景、非标准字体和长文本场景下是否稳定？
3. Prompt rewriting 带来的提升中，有多少来自模型能力，有多少来自更强的提示词工程？
4. 1.3B 版本的消费级显存优势是否覆盖高分辨率、长视频和多任务场景，还是主要针对标准 T2V 设置？
5. 开源模型和训练数据之间的版权、安全过滤与评测可复现性仍需进一步检查。

## 结论

Wan 是当前视频生成方向必须读的系统型论文之一。它的学术贡献不在于提出全新的生成范式，而在于把已有的 latent diffusion、DiT、[[knowledge/flow-matching|Flow Matching]]、captioning、prompt rewriting、并行训练和推理加速整合成一条可规模化、可开放、可评估的视频基础模型路线。

对你的文献库来说，这篇应放在“视频生成系统 / 开源大模型 / DiT 视频化 / 长序列训练工程”的核心阅读列表中。



