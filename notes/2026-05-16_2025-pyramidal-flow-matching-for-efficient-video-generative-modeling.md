---
type: paper-note
aliases:
  - "Pyramidal Flow Matching for Efficient Video Generative Modeling"
paper_id: "2025-pyramidal-flow-matching-for-efficient-video-generative-modeling"
title: "Pyramidal Flow Matching for Efficient Video Generative Modeling"
year: 2025
venue: "ICLR"
subfield: "Video Generation"
topics:
  - "diffusion-flow"
  - "text-to-video"
importance: 5
read_status: "unread"
note_status: "generated"
selected_on: "2026-05-16"
last_reviewed_on: ""
paper: "[[literature/papers/2025-pyramidal-flow-matching-for-efficient-video-generative-modeling]]"
pdf: "[[papers/2025_Pyramidal Flow Matching for Efficient Video Generative Modeling_ICLR.pdf]]"
tags:
  - paper
  - paper/video_generation
  - topic/diffusion_flow
  - topic/text_to_video
  - venue/iclr
  - year/2025
---
# Pyramidal Flow Matching：把“生成 + 超分”统一到一个 DiT 里

## 论文信息
- 标题：Pyramidal Flow Matching for Efficient Video Generative Modeling
- 年份：2025（arXiv:2410.05954v2, 2025-03-15）
- 会议：ICLR 2025
- PDF：[[papers/2025_Pyramidal Flow Matching for Efficient Video Generative Modeling_ICLR.pdf]]
- 代码/项目页：论文给出（本库不额外收录链接）

## 细分领域
- Video Generation / Text-to-Video（扩散/[[knowledge/flow-matching|Flow Matching]] + DiT）

## 重要程度
- 5/5（把多阶段 cascade 的“生成→超分”拆分，变成**单模型端到端**的分辨率分段 [[knowledge/flow-matching|flow matching]]；同时给出**temporal pyramid**让自回归视频训练可扩展）

## 一句话总结
把扩散/flow 的去噪轨迹按分辨率切成多个“金字塔 stage”，**绝大多数步在低分辨率 latent 上做**，只在最后少数步升到全分辨率；通过一个统一的 [[knowledge/flow-matching|flow matching]] 目标用同一个 DiT 同时学“生成 + 解压/超分”，并用 temporal pyramid 压缩长历史条件，让 768p、24fps、5–10s 的 T2V 训练更省算力。

## 背景问题（Why）
视频生成难点不止“模型大”，而是 token 爆炸：
- 空间：高分辨率视频 latent token 数巨大（即便 VAE 8× 压缩也很可怕）。
- 时间：全序列扩散需要一次性输入全部帧（训练复杂度随序列长度/分辨率急剧上升），且推理长度难以超过训练长度。
- 现有 cascade 思路（先低清生成，再上采样/超分）能省算力，但代价是：多模型分别训练，知识难共享、端到端不优雅、推理管线更复杂。

## 核心贡献（What）
1. **Pyramidal Flow Matching（空间金字塔）**：把从噪声到数据的轨迹分成多个分辨率 stage，在每个 stage 内做 [[knowledge/flow-matching|flow matching]]，stage 之间通过“跳点 renoising”保证概率路径连续。
2. **Unified objective + single DiT**：用一个统一的 [[knowledge/flow-matching|flow matching]] 目标把“生成 + 解压/超分”揉在同一个模型/同一次训练里，而不是每个分辨率单独训练。
3. **Temporal pyramid for autoregressive video**：用自回归方式逐段预测未来 latent，但把长历史条件按时间远近做多尺度压缩（越远越低分辨率），显著降低历史条件 token。
4. **可扩展的工程实现**：直接用标准 Transformer/DiT（论文用 SD3 Medium 风格的 MM-DiT, 2B 参数）配合 3D VAE（8×8×8 下采样）完成图像+视频混合训练。

## 方法详解（How）
### 1) 空间金字塔：把去噪轨迹分段到不同分辨率
核心直觉来自 Fig.1：扩散早期 latent 很“脏”，在全分辨率上做大量计算性价比很低。

做法：假设有 K 个分辨率（每一级宽高减半），把时间区间 [0,1] 切成 K 个 time window `[s_k, e_k]`，每个 window 内在某一分辨率上做 [[knowledge/flow-matching|flow matching]]：
- 直观：**越靠近噪声端的 stage 分辨率越低**；只在最后 stage 回到全分辨率。
- 文中把跨分辨率插值写成 `⊕`，并将其分解为分段（piecewise）插值（对应 Fig.2a）。

### 2) Unified training：用同一噪声方向“拉直”分段路径
为了把“生成 + 解压/超分”统一起来，论文构造了一个从“更噪、更像素化的 latent”到“更干净、更高分辨率 latent”的概率路径：
- stage 末端（更干净、更低噪、低分辨率）：`x̂_{e_k}`
- stage 起点（更噪、由更低分辨率上采样到当前分辨率）：`x̂_{s_k}`

关键小设计：端点噪声采用**同一方向**（同一个 `n ~ N(0, I)`），让轨迹更“直”，有利于 [[knowledge/flow-matching|flow matching]] 学习（文本里在 Eq.(9)(10) 一起定义端点）。

### 3) Inference with renoising：跨 stage 跳点如何保持连续（Algorithm 1）
问题：stage 之间分辨率不同，直接上采样会改变分布的均值/协方差，导致概率路径不连续。

论文在跳点处做两件事（Fig.2b / Eq.(15)）：
1) 先 `Up(x̂_{e_{k+1}})`（最近邻或双线性上采样）。
2) 再做**重缩放 + 纠正噪声（renoising）**，匹配下一个 stage 起点 `x̂_{s_k}` 的高斯分布均值与协方差。

直觉理解：
- 上采样会引入强相关（一个像素复制成 2×2 block），于是论文专门设计带负相关的纠正噪声，让 block 内相关性降低，从而让“跳点”更像真实的下一 stage 起点分布。

### 4) 自回归视频 + Temporal pyramid：压缩长历史条件
自回归生成支持灵活长度，但训练时历史条件太长仍然贵。

论文观察：远处历史更多提供语义/场景级约束，对细节影响没那么大，于是用 Fig.3 的 temporal pyramid：
- 越早的历史帧（距离当前越远）用**更低分辨率**的 latent 作为条件；
- 靠近当前帧的历史用更高分辨率 latent；
- 训练时还对历史 latent 加一点小噪声（强度从 [0, 1/3] 采样），缓解自回归误差累积。

位置编码也做了配套：
- 空间金字塔：位置编码外推（extrapolate）以支持更细粒度细节。
- 时间金字塔：位置编码插值（interpolate）以对齐不同分辨率的历史条件（Fig.3b）。

## 关键公式或算法直觉
- **分段跨分辨率插值**：把 [0,1] 划窗后，在每个窗口内做“当前分辨率上的线性插值”，从而把一个跨分辨率的整体路径拆成若干个同形状的子问题（Eq.(6) 的思路）。
- **同向噪声耦合**：用同一个 `n` 构造 stage 的起点/终点，等价于让端点差 `x̂_{e_k} - x̂_{s_k}` 更稳定、方向更一致，降低学习难度。
- **跳点 renoising 的本质**：不是“随便再加点噪声”，而是为匹配上采样后协方差结构而设计的纠正项（block 内去相关）。

## 关键原图讲解
> 说明：本论文的部分关键示意图（Fig.1/2/3/7/8）在 PDF 中是矢量/组合对象，`extract-images` 往往只能抽到局部 raster 或抽不到。因此这里对这些图使用了“PDF 页面渲染截图”（`render_page_*.png` + 自动裁剪）来保证图完整；生成示例（Fig.5）则直接使用 `extract-images` 的结果。

### 图1：为什么需要金字塔式 [[knowledge/flow-matching|flow matching]]（Fig.1，PDF 第2页）
![图1：动机（全分辨率去噪 vs 金字塔分段）](../figures/2025-pyramidal-flow-matching-for-efficient-video-generative-modeling/fig_01_motivation.png)
- 来自：PDF 第2页（Fig.1）
- 展示了什么：左边是典型全分辨率视频扩散/DiT 在整条去噪轨迹上都用高分辨率 token；右边是本文的“分段金字塔”轨迹：早期在低分辨率，最后才升到全分辨率；并在自回归设置下用压缩历史作为条件（图中蓝色箭头）。
- 如何解读：把“高噪声阶段”视为只需要粗尺度结构即可；将高分辨率计算留给末尾少数步，才能把算力用在“更接近数据、更有信息量”的阶段。
- 与核心贡献关系：这是全文的总论点：**在轨迹上做分辨率调度 + 用一个统一目标连起不同 stage**，避免 cascade 多模型。

### 图2：空间金字塔与跳点 renoising（Fig.2，PDF 第4页）
![图2：空间金字塔与跨 stage 连续性](../figures/2025-pyramidal-flow-matching-for-efficient-video-generative-modeling/fig_02_spatial_pyramid.png)
- 来自：PDF 第4页（Fig.2）
- 展示了什么：(a) 把去噪轨迹分成多个 stage：每个 stage 从“更像素化、更噪”的起点到“更清晰”的终点；(b) 说明为什么跨 stage 的跳点需要额外处理，并给出 renoising 的直观示意。
- 如何解读：关键不是“降分辨率”本身，而是**如何把不同分辨率 stage 串成一条连续概率路径**；否则 stage 之间会断裂，训练/推理就退化回“多模型/多次重启”。
- 与核心贡献关系：对应本文最关键的两点：Unified objective（端到端单模型）+ Jump-point renoising（保证 stage 链接）。

### 图3：Temporal pyramid 如何压缩长历史（Fig.3，PDF 第6页）
![图3：Temporal pyramid（远处历史低分辨率）](../figures/2025-pyramidal-flow-matching-for-efficient-video-generative-modeling/fig_03_temporal_pyramid.png)
- 来自：PDF 第6页（Fig.3）
- 展示了什么：(a) 历史条件按时间远近使用不同分辨率（越远越低分辨率）；(b) 位置编码在空间金字塔与时间金字塔里分别用外推/插值以适配多尺度条件对齐。
- 如何解读：远处历史更多是“语义/场景先验”，细节可以被压缩掉；把历史 token 变少后，才能在不牺牲全注意力建模能力的情况下训练自回归视频模型。
- 与核心贡献关系：这是本文把“视频变长”从工程上做可扩展的关键：把历史条件从 `T*N` 压缩到近似 `T*N/4^K` 量级。

### 图4：空间/时间金字塔的消融（Fig.7/8，PDF 第10页）
![图4：Ablation（空间金字塔 vs 全分辨率；时间金字塔 vs 全序列）](../figures/2025-pyramidal-flow-matching-for-efficient-video-generative-modeling/fig_04_ablations.png)
- 来自：PDF 第10页（Fig.7/8）
- 展示了什么：Fig.7 对比“标准 [[knowledge/flow-matching|flow matching]]（全分辨率）”与“金字塔 [[knowledge/flow-matching|flow matching]]”在图像训练上的收敛速度（以 FID 曲线为例）；Fig.8 对比 temporal pyramid 的自回归训练与 full-sequence diffusion baseline 的视频收敛/质量差距。
- 如何解读：金字塔的收益主要体现在**同算力下更快收敛**，并且 full-sequence baseline 在相同训练步下更难学到一致运动/连贯细节。
- 与核心贡献关系：用消融直接证明：省下来的 token/算力不是“偷工减料”，而是把模型容量集中在更关键的阶段，训练更有效率。

### 图5：生成视频的定性结果（Fig.5，PDF 第9页）
![图5：T2V 生成示例（论文截图）](../figures/2025-pyramidal-flow-matching-for-efficient-video-generative-modeling/page_009_01_img-009-145.jpg)
- 来自：PDF 第9页（Fig.5 的部分截图；`extract-images` 提取）
- 展示了什么：作者展示 5s/10s、768p、24fps 的生成结果截图（完整视频需看项目页）。
- 如何解读：这类图只能辅助判断“画面细节/风格/主体一致性”；运动质量仍需结合 VBench/EvalCrafter 指标与用户偏好实验一起看。
- 与核心贡献关系：给出“能跑到 768p、24fps、5–10s”的证据点，支撑“token 节省 → 训练/推理可扩展”的主张。

## 实验与评价
### 1) 训练/工程配置（论文给出的关键信息）
- Base model：MM-DiT（SD3 Medium 风格），约 2B 参数。
- Tokenizer：3D VAE，空间/时间下采样比 8×8×8（结构类似 MAGVIT-v2，作者在 WebVid-10M 上从零训练）。
- Pyramid stage 数：K=3（文中实验均如此）。
- 评测：VBench（16 个维度综合评估运动与语义对齐）+ EvalCrafter（多指标）。

### 2) 效率（Efficiency）
- 论文给出复杂度对比：全序列扩散输入 token 为 `T*N`，计算量随 `T^2*N^2` 增长；金字塔后 token 近似降到 `T*N/4^K`，计算量到 `T^2*N^2/16^K`（至少在最终 stage 也如此量级）。
- 训练成本：作者报告训练一个 10s（241 帧）、768p、24fps 的模型约 **20.7k A100 GPU-hours**。
- 推理速度：报告生成 5s、384p 的视频约 **56 秒**（与全序列扩散同量级）。

### 3) 主结果（Main results）
VBench（Table 1，5s/121 帧评测设定）中，论文给出：
- Ours（公有数据训练）：Total **81.72**，Quality **84.74**，Semantic **69.62**，Motion smoothness **99.12**，Dynamic degree **64.63**。
- 论文声称：在 Total/Quality 上超过一些开源基线，并在 Quality 上接近/优于部分商业系统对比项（例如文中提到 Quality 84.74 vs Gen-3 Alpha 84.11）。

EvalCrafter（Table 2）中，论文给出：
- Ours（公有数据训练）：Final sum **244**，Visual quality **67.94**，Text-video alignment **57.01**，Motion quality **55.31**，Temporal consistency **63.41**。
- 讨论：作者认为语义分数相对更低的原因之一是使用了较粗粒度的合成 caption（可通过更强视频 captioning 改善）。

### 4) 用户偏好（User study）
- 设置：50 个 VBench prompt，20+ 参与者，从审美、运动平滑、语义对齐三个维度对多模型排序（Fig.4 给出偏好对比）。
- 结论方向：作者强调在运动平滑上相对开源基线更受偏好，解释为 token 节省使其能在 24fps 下训练/生成，而不少对比基线常见设置是 8fps。

## 局限性（Limits）
- **语义对齐仍受 caption 质量影响**：作者明确提到语义分数偏低与 caption 粗糙有关；这意味着方法本身不是“自动变强对齐”，仍依赖数据与标注质量。
- **更多是“高效训练范式”而非新 backbone**：贡献重心在“分辨率/历史条件的金字塔化 + unified objective”，对模型结构创新相对克制；效果上仍依赖强 DiT + 强 VAE。
- **图像/视频两类任务的联合训练细节复杂**：虽然论文强调可用标准 pipeline，但实际复现涉及数据混合、token 打包（Patch n’ Pack）、多 GPU tokenization 等工程门槛不低。

## 与库中相关论文的关系
- 与 cascade T2V（例如 [[literature/papers/2024-lavie-high-quality-video-generation-with-cascaded-latent-diffusion-models|LaVie]]）：本文想解决 cascade 的“多模型分阶段训练”问题，用统一目标把多分辨率连到一个模型里。
- 与 DiT/[[knowledge/flow-matching|Flow Matching]] 系统化路线（例如 [[notes/2026-05-10_2025-wan-open-and-advanced-large-scale-video-generative-models|Wan]]）：都强调规模化训练与工程；本文更聚焦“分辨率与历史条件”的 token 经济学。
- 与强开源 T2V baseline（例如 [[literature/papers/2025-cogvideox-text-to-video-diffusion-models-with-an-expert-transformer|CogVideoX]]）：论文在 VBench 质量分上给出对比，并把自己定位为“更高效的高分辨率/高 fps 训练方案”。
- 与长视频/推理加速（例如 [[literature/papers/2024-fifo-diffusion-generating-infinite-videos-from-text-without-training|FIFO-Diffusion]]、[[literature/papers/2024-freenoise-tuning-free-longer-video-diffusion-via-noise-rescheduling|FreeNoise]]）：这些工作更偏推理侧或无需训练的延长；本文主要从训练侧降低成本，让“原生 5–10s、24fps、768p”更可行。

## 后续阅读建议
1. 先读本文方法段落（Fig.1–3 + Algorithm 1）：把“空间金字塔 + 跳点 renoising + temporal pyramid”三件套吃透。
2. 对照阅读：[[literature/papers/2025-cogvideox-text-to-video-diffusion-models-with-an-expert-transformer|CogVideoX]]（同样是强 T2V DiT/扩散路线，但更强调 expert transformer 等结构），理解“效率提升”与“模型结构提升”的取舍。
3. 扩展：如果你关心 [[knowledge/flow-matching|flow matching]]/[[knowledge/flow-matching|rectified flow]] 在图像方向的规模化，回头看 [[literature/papers/2024-scaling-rectified-flow-transformers-for-high-resolution-image-synthesis|Scaling Rectified Flow Transformers for High-Resolution Image Synthesis]]（本库未读），把 flow/[[knowledge/flow-matching|rectified flow]] 与 DiT 的训练细节串起来。
