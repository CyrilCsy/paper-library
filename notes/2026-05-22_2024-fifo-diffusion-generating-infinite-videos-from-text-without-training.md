---
type: paper-note
aliases:
  - "FIFO-Diffusion Generating Infinite Videos from Text without Training"
paper_id: "2024-fifo-diffusion-generating-infinite-videos-from-text-without-training"
title: "FIFO-Diffusion Generating Infinite Videos from Text without Training"
year: 2024
venue: "NEURIPS"
subfield: "Long Video Generation"
topics:
  - "diffusion-flow"
  - "long-video"
importance: 5
read_status: "unread"
note_status: "generated"
selected_on: "2026-05-22"
last_reviewed_on: ""
paper: "[[literature/papers/2024-fifo-diffusion-generating-infinite-videos-from-text-without-training]]"
pdf: "[[papers/2024_FIFO-Diffusion Generating Infinite Videos from Text without Training_NeurIPS.pdf]]"
tags:
  - paper
  - paper/long_video_generation
  - topic/diffusion_flow
  - topic/long_video
  - venue/neurips
  - year/2024
---
# FIFO-Diffusion：不用训练，把短 clip 扩成“无限长”文生视频

## 论文信息

- 标题：FIFO-Diffusion: Generating Infinite Videos from Text without Training
- 作者：Jihwan Kim, Junoh Kang, Jinyoung Choi, Bohyung Han（SNU）
- 会议：NeurIPS 2024
- arXiv：2405.11473（v4: 2024-11-03）
- PDF：[[papers/2024_FIFO-Diffusion Generating Infinite Videos from Text without Training_NeurIPS.pdf]]
- 项目页：jjihwan.github.io/projects/FIFO-Diffusion

## 细分领域 / 重要程度

- 细分领域：Long Video Generation（长视频生成 / 无限长度生成）
- 重要程度：5/5（原因：NeurIPS + 训练-free 的“推理层”长视频方案，容易迁移到现有 T2V 模型）

## 一句话总结

把“扩散时间步 t”和“帧索引 i”耦合成一条对角线，用一个 **FIFO 队列**在每次推理只保留固定窗口 `f` 帧的 latent，却能 **逐帧出队生成任意长视频**；再用 **latent partitioning** 缩小噪声差带来的训练-推理分布差、用 **lookahead denoising** 让待去噪帧看见更“干净”的前文，从而在不训练的前提下获得更稳的长程一致性与运动。

## 背景问题：为什么“长视频”难？

扩散式视频模型（VDM）通常把整段视频当成一个 4D 张量去做多步去噪；要生成长视频，要么：

- **chunked autoregressive（分块自回归）**：一次预测一小段（如 16 帧），再用少量上一段的帧做条件继续往后生成。问题是跨 chunk 的上下文很短，容易出现运动不连续、语义漂移。
- **training-based 长视频模型**：针对长视频额外训练/蒸馏/加子网络，成本高、可迁移性差。

这篇论文的核心诉求是：**在不做额外训练的条件下**，把任意一个“短 clip 文生视频扩散模型”改造成能生成极长视频的推理算法，并且内存不随目标时长增长。

## 核心贡献（作者自述 + 归纳）

- 提出 **FIFO-Diffusion**：一种训练-free 的推理算法，通过 *diagonal denoising* 实现任意长视频生成。
- 提出两项关键改进：
  - **Latent partitioning**：缩小对角去噪时同一窗口内不同帧的噪声等级差，缓解训练-推理 gap，同时支持多 GPU 并行。
  - **Lookahead denoising**：让要去噪的帧显式“向前看”更干净的帧，提高噪声预测准确性与时间一致性（代价是更高计算，但可并行摊薄）。
- 证明/分析：对角去噪引入的误差上界与窗口内噪声差有关，因此 partitioning 可以从原理上降低 gap。
- 实证：在多个开源基座（VideoCrafter1/2、zeroscope、Open-Sora Plan 等）上展示 10K 级别长视频质量不明显退化；并在 UCF-101 的 FVD/IS、以及内存/速度上给出对比。

## 方法详解

### 0) 预备：视频 latent 扩散的标准训练/推理

常见做法是在 VAE latent 空间做扩散。令视频 latent 为 `z0 = Enc(v) = [z0^1; ...; z0^f]`，训练目标是让噪声预测网络 `ε_θ(·)` 复原加入到 `z0` 的噪声：

- 扰动：`z_t = s_t z0 + σ_t ε`（`ε ~ N(0, I)`）
- 损失：`E[ || ε_θ(z_t; c, t) - ε || ]`

推理时按照调度器从高噪声逐步去噪到 `t=0`，得到每一帧的 latent，再 Dec 还原成像素帧。

### 1) Diagonal denoising：把“扩散步”和“帧序列”绑成一条对角线

关键观察：**标准 VDM 推理**对一个 `f` 帧短 clip，会在同一个扩散步 `t` 上并行去噪所有帧；而 FIFO-Diffusion 则构造一个包含 `f` 帧的队列 `Q`，但这 `f` 帧处于**不同的噪声等级**（从更干净到更嘈杂单调递增），并在每一次迭代沿“对角线”推进：

- 队列 `Q` 中存的是一组对角 latent：`[z_{τ1}^i, z_{τ2}^{i+1}, ..., z_{τf}^{i+f-1}]`（`τ1 < ... < τf`）
- 一次对角去噪：对这 `f` 个不同噪声等级的 latent，调用同一个基座去噪网络做一步更新
- 更新后：
  - 最“干净”的那一帧到达 `τ0=0`，**出队成为最终帧**（dequeued）
  - 在队尾追加一个新的随机噪声帧（enqueued），继续下一轮

直觉：这相当于用一个固定大小的滑动窗口，把上下文从“跨 chunk 的最后 1 帧”升级成“窗口内多帧连续前文”，从而更容易保持长程一致性。

### 2) 训练-推理 gap 从哪来？

对角去噪的双刃剑在于：窗口内同一轮的各帧噪声等级不同，而基座模型训练时一般假设“同一 batch 内帧噪声等级一致”。于是模型输入分布发生偏移，带来质量损失。

### 3) Latent partitioning：把“大噪声跨度”切碎

做法：把扩散过程分成 `n` 段，并把队列扩展为 `n*f` 长度后分块处理。每一块内部的噪声跨度更小，从理论上让对角去噪误差的上界更低（论文给出“误差与噪声差成正比”的界）。

同时，分块结构天然支持并行：不同 partition 的块可以在多 GPU 上并行推进，从而把额外计算部分摊薄（见 Table 2 的 8 GPU 加速）。

### 4) Lookahead denoising：让“后半段帧”看到更干净的前文

对角去噪的优势是“噪声更大的帧能参考更干净的帧”；作者进一步强化这一点：在每步只更新队列的**后半段**（stride 约为 `⌊f/2⌋`），确保每个被更新的帧都能看到足够数量的更干净前文，从而提升噪声预测精度、减少闪烁。

代价：理论上计算量约翻倍，但可配合 partitioning 用多 GPU 并行抵消一部分开销。

## 关键公式 / 算法直觉（抓住最关键的 3 个点）

1) **对角队列 = 把 2D（帧×扩散步）压成 1D 队列**：每次只维护窗口 `f`（或 `n*f`）大小，内存不随目标帧数 `N` 增长。
2) **误差上界与窗口噪声跨度有关**：噪声差越大，训练-推理 gap 越大；因此 partitioning 通过缩小跨度来改善。
3) **“向前看”让噪声预测更准**：lookahead 把“参考更干净帧”的好处制度化，实证上能把相对 MSE 拉到 <1。

## 关键原图讲解（5 张图，优先讲算法与实证）

> 说明：`extract-images` 只会抽取 PDF 的嵌入位图，对矢量/组合图（方法示意图、表格）往往不完整；下面的 Figure 2/3/4/9/10 采用对 PDF 页面渲染后裁剪得到，以保证图内容完整可读。

### 图 1：10K 帧长视频示例（Figure 1，PDF 第 1 页）

![图：10K 帧长视频示例](../figures/2024-fifo-diffusion-generating-infinite-videos-from-text-without-training/fig1_long_video_illustration.png)

- 来自 PDF 第 1 页（Figure 1）
- 展示内容：同一个 prompt 下，从 0 / 2500 / 5000 / 7500 / 10000 帧抽样出的帧图，展示“很长时间跨度下”的语义/风格保持与运动延续。
- 如何解读：如果模型在长程生成中出现语义漂移或纹理崩坏，通常会在后期帧显著恶化；这张图用极稀疏的抽样说明 **质量不随时间明显退化**。
- 与贡献关系：直接证明“只改推理、不训练也能生成极长视频”在视觉上成立。

### 图 2：对角去噪 + FIFO 出队（Figure 2，PDF 第 3 页）

![图：Diagonal denoising（对角去噪）](../figures/2024-fifo-diffusion-generating-infinite-videos-from-text-without-training/fig2_diagonal_denoising.png)

- 来自 PDF 第 3 页（Figure 2）
- 展示内容：队列里 `f` 帧处于不同时间步（噪声等级），红框为输入、虚线框为输出；最右上角 fully denoised 的帧出队，新噪声帧入队。
- 如何解读：它把“生成无限长视频”变成一个稳定的循环：**一次只跑一个固定窗口**，每次输出 1 帧并滚动窗口。
- 与贡献关系：这是 FIFO-Diffusion 的核心机制，也是常数内存的根源。

### 图 3：与 chunked autoregressive 的本质差异（Figure 3，PDF 第 4 页）

![图：分块自回归 vs FIFO-Diffusion](../figures/2024-fifo-diffusion-generating-infinite-videos-from-text-without-training/fig3_chunk_vs_fifo.png)

- 来自 PDF 第 4 页（Figure 3）
- 展示内容：左边是分块自回归（按块去噪/预测），右边是 FIFO 的对角推进；红框代表基座去噪网络被调用的位置（“推理的形状”不同）。
- 如何解读：chunked 方法跨块只传递很少帧，容易在块边界出现跳变；FIFO 的对角推进让每一帧都能引用足够多的前文，从而更平滑。
- 与贡献关系：解释了为什么 FIFO 在“运动连续性”和“长程一致性”上更有优势。

### 图 4：Latent partitioning + Lookahead（Figure 4，PDF 第 5 页）

![图：Latent partitioning 与 Lookahead denoising](../figures/2024-fifo-diffusion-generating-infinite-videos-from-text-without-training/fig4_latent_partition_lookahead.png)

- 来自 PDF 第 5 页（Figure 4）
- 展示内容：(a) 把扩散过程切成 `n` 段，降低同一轮窗口内的最大噪声差；(b) lookahead 让待更新帧能看见更多更干净的帧（计算更重）。
- 如何解读：
  - partitioning 的目标是“把训练-推理 gap 的来源（噪声跨度）变小”
  - lookahead 的目标是“把对角去噪的好处（参考更干净帧）用机制保证住”
- 与贡献关系：这是作者解决对角去噪“gap vs 参考优势”的两把关键工具。

### 图 5：消融：LP/LD 的实际收益（Figure 9 + Table 3，PDF 第 9 页）

![图：消融对比（DD vs DD+LP vs DD+LP+LD）](../figures/2024-fifo-diffusion-generating-infinite-videos-from-text-without-training/fig9_ablation.png)

- 来自 PDF 第 9 页（Figure 9 + Table 3）
- 展示内容：同一 prompt 下，比较 DD（仅对角去噪）、DD+LP、DD+LP+LD 的视觉效果与相对 MSE（噪声预测误差）。
- 如何解读：仅 DD 可能出现模糊/不稳定；加 LP 后明显更稳；再加 LD 往往进一步减少闪烁、提升一致性。Table 3 里相对 MSE 从 1.09（无 LP 无 LD）降到 0.98（LP=4 且有 LD）。
- 与贡献关系：用视觉 + 定量两条证据说明两个改进确实在修复对角去噪的关键问题。

### 图 6：更多定性长视频样例（Figure 10，PDF 第 17 页）

![图：更多 FIFO-Diffusion 样例网格](../figures/2024-fifo-diffusion-generating-infinite-videos-from-text-without-training/fig10_qualitative_grid.png)

- 来自 PDF 第 17 页（Figure 10）
- 展示内容：不同 prompt 下，从 0/20/40/60/80 帧抽样的网格，突出运动与语义保持。
- 如何解读：长视频质量评估的关键不是单帧清晰度，而是“跨时间的一致性 + 运动自然性”；这些网格能快速看出是否存在漂移或运动断裂。
- 与贡献关系：补充说明该推理范式在多个场景下具有普适性，而非只对个别 prompt 生效。

## 实验与评价（关心三件事：质量、成本、用户偏好）

### 1) 质量（UCF-101 定量：FVD128 / IS）

Table 1（PDF 第 8 页）给出了在 UCF-101 上的对比（Latte 基座）：

- FIFO-Diffusion（ours）：**FVD128 = 596.64**（越低越好），**IS = 74.44 ± 1.17**（越高越好）
- 对比方法：
  - PVDM：FVD128 = 648.4，IS = 74.40 ± 1.25
  - StyleGAN-V / VIDM 等显著更差（FVD 更高或缺失）

结论：在不额外训练（或少训练）假设下，FIFO 的推理改造在该基准上给出很强的质量指标。

### 2) 成本（内存随帧长是否增长？多 GPU 是否有效？）

Table 2（PDF 第 8 页）展示了 VideoCrafter2 基座下，128/256/512 帧的内存与速度：

- FreeNoise：内存随帧数增长（128: 26163MB，256: 44683MB，512: OOM）
- Gen-L-Video：内存接近常数（约 10.9GB），但速度最慢（22.07 s/frame）
- FIFO-Diffusion（1 GPU）：内存常数（约 11.25GB），速度 12.37 s/frame
- FIFO-Diffusion（8 GPUs）：内存常数（约 13.50GB），速度 **1.84 s/frame**

结论：FIFO 的“常数内存”特性在长视频场景非常关键；并行性是把额外计算成本摊薄的关键路径。

### 3) 用户研究（对 motion 等指标偏好更强）

论文报告用户更偏好 FIFO-Diffusion，尤其在“与运动相关”的维度优势明显（Figure 8，PDF 第 8 页附近）。

## 局限性（读完后你应该保持怀疑的点）

- **训练-推理 gap 仍未完全消失**：作者也承认，尽管 LP 降低了 gap，但输入分布变化仍在；最根本的解决可能是把“对角范式”融入训练阶段。
- **计算量/吞吐依赖并行**：lookahead 会增加计算，单卡场景下可能偏慢；真正想把它落地到生产推理，需要良好的多 GPU 调度与工程实现。
- **适配性与超参**：窗口 `f`、partition 数 `n`、lookahead stride 等会影响质量/速度；不同基座模型的最佳设置可能不同。

## 与库中相关论文的关系（建议对照阅读）

- 长视频/系统类：[[2026-05-10_2025-wan-open-and-advanced-large-scale-video-generative-models]]、[[2026-05-12_2025-hunyuanvideo-a-systematic-framework-for-large-video-generative-models]]、[[2026-05-14_2025-open-sora-2-0-training-a-commercial-level-video-generation-model-in-200k]]
- 自回归/推理改造方向：[[2026-05-11_2025-autoregressive-video-generation-without-vector-quantization]]、[[2026-05-17_2025-taming-teacher-forcing-for-masked-autoregressive-video-generation]]
- 训练/效率与采样范式：[[2026-05-16_2025-pyramidal-flow-matching-for-efficient-video-generative-modeling]]

对照角度建议：

- 这篇（FIFO）是“**不训练，改推理循环**”把短模型拉长；而系统类（Open-Sora/Wan 等）更多靠数据/训练/系统工程实现规模化。
- 与 AR/Masked AR 的对照重点是：FIFO 把扩散的计算“摊到时间轴上逐帧产出”，但仍保留扩散多步去噪的代价；AR 往往单步但需要不同训练范式。

## 后续阅读建议（按目的选）

- 想复现/落地：先看 Appendix C 的算法伪代码（对角去噪 / LP / LD），再对照项目页代码实现。
- 想理解“为什么可行”：看 Theorem 3.3（误差与噪声跨度关系）+ Table 3（相对 MSE 的实证）。
- 想做更强：考虑把 diagonal denoising 的队列范式纳入训练（作者未来工作），或研究更低成本的 lookahead/并行调度。




















