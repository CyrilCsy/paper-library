---
type: paper-note
aliases:
  - "Autoregressive Video Generation without Vector Quantization"
paper_id: "2025-autoregressive-video-generation-without-vector-quantization"
title: "Autoregressive Video Generation without Vector Quantization"
year: 2025
venue: "ICLR"
subfield: "Video Generation"
topics:
  - "autoregressive"
  - "text-to-video"
  - "visual-tokenizer-vq"
importance: 5
read_status: "unread"
note_status: "generated"
selected_on: "2026-05-11"
last_reviewed_on: ""
paper: "[[literature/papers/2025-autoregressive-video-generation-without-vector-quantization]]"
pdf: "[[papers/2025_Autoregressive Video Generation without Vector Quantization_ICLR.pdf]]"
tags:
  - paper
  - paper/video_generation
  - topic/autoregressive
  - topic/text_to_video
  - topic/visual_tokenizer_vq
  - venue/iclr
  - year/2025
---
# Autoregressive Video Generation without Vector Quantization（NOVA）精讲笔记

## 论文信息

- 论文页：[[literature/papers/2025-autoregressive-video-generation-without-vector-quantization]]
- PDF：[[papers/2025_Autoregressive Video Generation without Vector Quantization_ICLR.pdf]]
- 会议：ICLR 2025
- 领域：Video Generation
- 重要程度：5 / 5
- 阅读状态：未读（注意：生成精讲笔记 ≠ 已读）

## 一句话总结

NOVA 把“视频生成”改写成**非量化（continuous）视觉 token 的自回归建模**：时间上按帧因果生成、帧内用 set-by-set 的并行/双向建模提速，再用一个 diffusion denoising 头在连续空间里把 token 细化，从而在保持 GPT 式“可变长度 + in-context”框架的同时，做到更高效率与更强统一多任务能力。

## 背景问题

作者认为两条主流路线各有硬伤：

1. **VQ + AR（离散 token）**：为了高保真通常要更多 token（更低压缩），分辨率/时长一上去训练与推理代价陡增；而 VQ tokenizer 还会引入细节损失或量化误差。
2. **Video Diffusion（连续 latent）**：常见训练目标更偏“固定长度片段的联合分布”，对**可变长度生成**与“把任意上下文塞进提示里当条件”的 GPT 式能力支持弱。

NOVA 的目标是把两者的优点合并：既要连续 token 的高压缩与保真，又要 AR 的因果可扩展与统一多任务。

## 核心贡献

1. **提出非量化视频 AR：时间帧级因果 + 帧内 set-by-set** 的因果分解方式（避免 raster-scan 的超长序列与高延迟）。
2. **统一多任务**：把 T2V 作为核心任务，借助同一套 AR 框架自然覆盖 T2I、I2V、V2V、条件补全等（论文图 1/2 的表达）。
3. **效率与性价比**：论文声称 0.6B 参数即可在 VBench 与 GenEval 等指标上取得强结果，并给出训练成本（A100-days）与推理速度（FPS）。

## 方法详解

NOVA 可以按“编码 → 时间 AR → 空间 set-by-set → 连续去噪”理解：

### 1) 输入与压缩表示

- **文本条件**：用一个预训练语言模型编码 prompt（文中提到使用开源 LM）。
- **运动条件（可选但很关键）**：用 OpenCV 计算视频帧 optical flow，并把平均 flow magnitude 作为 motion score 融合进条件（用于更好控制动态）。
- **视频编码器**：用开源 3D-VAE，把视频压到 latent（文中提到 temporal stride=4、spatial stride=8），再接一个 learnable patch embedding 做通道/步幅对齐。

### 2) 时间维：Frame-by-Frame 的“块级因果注意力”

核心直觉：视频的自然因果单位不是“像素/patch token”，而是“帧”。因此它把生成顺序改成：

- **按帧自回归**：第 f 帧只能看文本、运动条件、以及前面 1..f-1 的帧；
- **帧内 token 互相可见**：同一帧里允许 token 之间双向交互，避免 per-token raster-scan 的延迟。

论文在图 3(a) 用一个注意力可见性矩阵示意这种“块级因果”。

形式上可以写成（论文 Eq.(2) 的语义）：

- 令 $S_f$ 表示第 $f$ 帧的全部 latent token，则

$$
p(\mathrm{video}\mid\mathrm{cond}) = \prod_f p(S_f\mid\mathrm{cond}, S_{<f})
$$

### 3) 空间维：Set-by-Set 的“随机顺序 masked 预测”

在单帧内部，NOVA 不是一个个 token 预测，而是把 token 分成若干 **token set**，再按随机顺序逐个 set 预测：

- 每一步只把“当前要预测的 set”mask 掉；
- 其余已知 token 作为上下文，使用**双向 transformer**一次性回归该 set 的所有 token；
- 这样能在保证自回归可控性的同时，把并行度做起来（论文图 3(b) 的示意）。

对应地（论文 Eq.(3) 的语义），在给定某个“indicator / anchor feature”后，把 set 序列因果分解：

$$
p(S_{(f,1:K)}\mid S'_f,\mathrm{cond}) = \prod_k p(S_{(f,k)}\mid S'_f,\mathrm{cond}, S_{(f,<k)})
$$

### 4) 连续空间的 diffusion denoising 头

由于 token 是连续值向量，作者在训练/推理时引入一个 diffusion-style denoising 头（论文在图 2 中标为 Diffusion MLP，并在后续给出 denoising 形式的损失），直觉上相当于：

- transformer 给出“结构化的预测/初值”
- diffusion MLP 在连续空间做“细化与去噪”，提升保真与稳定性

## 关键公式或算法直觉

- **帧级因果**把复杂度从“分辨率×帧数 的超长序列”变为“以帧为 chunk 的序列”，KV-cache 也更友好。
- **帧内双向 + set-by-set**把“必须逐 token 串行”变为“逐 set 串行、set 内并行”，是速度与质量的折中点。
- **continuous token + 去噪头**试图绕开 VQ 的两难：既不过度离散化损失细节，也不让纯回归发散/模糊。

## 关键原图讲解

说明：本论文的关键结构图/表格在 PDF 中大量以矢量或组合对象形式存在，`extract-images` 自动提取到的多为定性样例小图；因此下面这些关键图使用“PDF 页面渲染图”来保证结构与表格完整，属于“原图自动提取不完整”的补救方式。

### 图：NOVA 总体框架与推理流程（Figure 2）

![图：NOVA 框架（Figure 2）](../figures/2025-autoregressive-video-generation-without-vector-quantization/fig2.png)

- 来自 PDF 第 4 页。
- 展示内容：文本/运动条件进入 Temporal Layers（按帧因果），每帧再进入 Spatial Layers（set-by-set），之后接 Diffusion MLP 在连续 token 上做去噪细化。
- 解读要点：把“时间因果”和“空间并行”拆开，是 NOVA 的效率来源；Diffusion MLP 则是连续 token 的质量护栏。
- 与核心贡献的关系：这是“非量化视频 AR”的整体落地形态。

### 图：块级时间因果 + 空间 set-by-set 注意力（Figure 3）

![图：块级因果与 set-by-set（Figure 3）](../figures/2025-autoregressive-video-generation-without-vector-quantization/fig3.png)

- 来自 PDF 第 5 页。
- 展示内容：(a) 时间维对比 per-token vs per-frame；(b) 空间维对比 per-token vs per-set，并用“当前 indicator token / 下一组 token set”解释生成顺序。
- 解读要点：NOVA 的“自回归”不是对所有 token 串行，而是对帧、对 set 做更粗粒度的因果；帧内/未 mask token 允许双向，减少延迟并提高一致性。
- 与核心贡献的关系：这是 NOVA 既“像 GPT（因果）”又“像 BERT（帧内双向）”的关键技巧。

### 图：T2I / T2V 定量结果与成本（Table 2 & Table 3）

![图：Table 2/3（评测与成本）](../figures/2025-autoregressive-video-generation-without-vector-quantization/tables_2_3.png)

- 来自 PDF 第 7 页。
- 展示内容：Table 2 给出 T2I（如 GenEval 等）对比与 A100-days；Table 3 给出 VBench 细项对比、模型规模与延迟。
- 解读要点：论文强调 NOVA 0.6B 的性价比：在 autoregressive 体系里显著强于 CogVideo（9B），并在总分上接近/对齐更大体量的强基线；同时把训练成本显式写出来，方便和 diffusion 体系做“钱与效果”的对账。
- 与核心贡献的关系：直接支撑“更高效率/更低成本仍能达到强质量”的主张。

## 实验与评价（读者视角的抓手）

- **T2V**：论文宣称在 VBench 上达到约 80 分量级，并给出推理速度（文中提到约 2.75 FPS 的测量设置）与训练成本（A100-days）。
- **T2I**：同一模型体系在 GenEval 等指标上也能做到强结果，且训练成本相对更低（表 2 给出对比）。
- **重要提醒**：表格里既有 “NOVA base” 也有 “+ Rewriter / + Videos”等设置；读指标时要先确认是不是同一训练阶段与同一数据规模。

## 局限性（基于本文可见信息的推断）

1. **方法链条较长**：3D-VAE、optical flow 条件、set schedule、diffusion MLP 等组件较多，工程与复现成本不低。
2. **连续 token 的训练稳定性**：作者引入 post-norm、去噪头等来对抗不稳定，暗示纯回归并不容易训练。
3. **评测口径与任务泛化**：统一多任务很吸引人，但不同任务（T2I vs T2V vs I2V）的最佳条件组织、token set 策略可能需要额外调参。

## 与库中相关论文的关系

（建议先按“同范式→同任务→同组件”对照阅读）

- 同范式（非量化 AR）：[[literature/papers/2024-autoregressive-image-generation-without-vector-quantization|Autoregressive Image Generation without Vector Quantization]]（把 MAR 式思想扩展到更复杂条件/视频）
- 同任务（AR 视频）：[[literature/papers/2025-magi-1-autoregressive-video-generation-at-scale|MAGI-1]]；[[literature/papers/2025-taming-teacher-forcing-for-masked-autoregressive-video-generation|Taming Teacher Forcing]]；[[literature/papers/2023-cogvideo-large-scale-pretraining-for-text-to-video-generation-via-transformers|CogVideo]]
- 对照路线（Diffusion 视频）：[[literature/papers/2025-cogvideox-text-to-video-diffusion-models-with-an-expert-transformer|CogVideoX]]；[[literature/papers/2025-pyramidal-flow-matching-for-efficient-video-generative-modeling|Pyramidal Flow Matching]]；基础概念见 [[knowledge/flow-matching|Flow Matching]]
- 统一模型/多模态 AR：[[literature/papers/2023-make-a-video-text-to-video-generation-without-text-video-data|Make-A-Video]]；以及本库中与 “Emu3” 相关条目（见论文页的 Related Papers）

## 后续阅读建议（按优先级）

1. 先把 Figure 2 / Figure 3 的因果分解吃透：到底哪些 token 在同一步里并行、哪些必须串行。
2. 对照非量化 AR 图像论文：理解 “set-by-set” 在图像里是怎么做 schedule 的，以及搬到视频后哪里改了。
3. 复核表 2 / 表 3：对齐数据量、分辨率、是否使用 rewriter/视频再训练阶段，再判断“性价比”结论是否成立。




































