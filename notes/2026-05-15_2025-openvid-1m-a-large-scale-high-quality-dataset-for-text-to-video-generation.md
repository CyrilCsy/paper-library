---
type: paper-note
aliases:
  - "OpenVid-1M A Large-Scale High-Quality Dataset for Text-to-video Generation"
paper_id: "2025-openvid-1m-a-large-scale-high-quality-dataset-for-text-to-video-generation"
title: "OpenVid-1M A Large-Scale High-Quality Dataset for Text-to-video Generation"
year: 2025
venue: "ICLR"
subfield: "Dataset / Data Curation"
topics:
  - "dataset"
  - "text-to-video"
importance: 5
read_status: "unread"
note_status: "generated"
selected_on: "2026-05-15"
last_reviewed_on: ""
paper: "[[literature/papers/2025-openvid-1m-a-large-scale-high-quality-dataset-for-text-to-video-generation]]"
pdf: "[[papers/2025_OpenVid-1M_A Large-Scale High-Quality Dataset for Text-to-video Generation_ICLR.pdf]]"
tags:
  - paper
  - paper/dataset_data_curation
  - topic/dataset
  - topic/text_to_video
  - venue/iclr
  - year/2025
---
# OpenVid-1M（ICLR 2025）中文带图精讲

## 论文信息
- 标题：OpenVid-1M: A Large-Scale High-Quality Dataset for Text-to-video Generation
- 会议/年份：ICLR 2025
- 任务：Text-to-Video（T2V）生成
- 论文核心产出：
  - 数据集：OpenVid-1M（>100万高质量 text-video pairs，至少 512×512）、OpenVidHD-0.4M（433K 1080p）
  - 模型：MVDiT（Multi-modal Video Diffusion Transformer，一种并行视觉-文本结构的 DiT 变体）

## 细分领域
Dataset / Data Curation（面向 T2V 训练数据的质量筛选 + 高信息量 caption 重写），并附带一个用于验证数据价值的生成模型结构（MVDiT）。

## 重要程度（importance=5）
这篇的价值主要在“**如何把百万级开源数据做得可用**”：给出了可复现的筛选流水线（美学/时序/运动/清晰度/切分/重写 caption）+ 用统一训练设置验证数据质量对 T2V 的直接影响，还额外给出一个更重视文本语义利用的骨干（MVDiT）。

## 一句话总结
作者提出 OpenVid-1M / OpenVidHD-0.4M，用系统化“质量筛选 + 长 caption 重写”把 T2V 训练数据做得更干净、更可控；并用并行视觉-文本的 MVDiT 验证：**更高质量 + 更长更准的描述**能稳定提升生成视频的清晰度、美学、文本对齐与时序一致性。

## 背景问题（Why）
T2V 近年受 Sora 等推动快速发展，但公开可用数据/训练策略常被两类问题卡住：
1) 数据集“量大但不精”：例如 WebVid/Panda 等更强调规模，混入水印、糊、闪烁、静帧、过度运动等低质样本，caption 也偏短/不精确，导致训练不稳定或上限偏低。  
2) 文本信息利用不足：不少 DiT 系列更偏“视觉 token 建模 + 简单 cross-attn”，难以充分利用文本 token 的语义细节。

## 核心贡献（What）
1) **OpenVid-1M**：百万级开源（open-scenario）高质量 T2V 数据集，保证分辨率下限并更强调可训练性（清晰度/时序一致性/运动幅度/美学）。  
2) **OpenVidHD-0.4M**：从 OpenVid-1M 中挑出 1080p 子集，面向高分辨率生成/微调。  
3) **MVDiT**：并行视觉-文本结构 + 多模态自注意力/时序注意力/多头 cross-attn，用于更充分挖掘文本语义对视频生成的帮助；并通过消融展示关键模块有效性。

## 方法详解（How）

### A. OpenVid-1M 数据处理流水线（表 1 的核心逻辑）
作者从多个来源汇总原始视频（重点描述对 Panda-50M 的处理），流水线要点：
1) **美学筛选（Aesthetics）**：用 LAION Aesthetics Predictor 过滤低美学视频；在 Panda-50M 上保留最高 20% 得到集合 `S_A`。  
2) **时序一致性筛选（Temporal consistency）**：用 CLIP 特征衡量相邻帧余弦相似度；过滤“几乎静止（过高）/频繁闪烁（过低）”的片段得到 `S_T`。  
3) **运动幅度筛选（Motion difference）**：用 UniMatch 光流估计衡量运动幅度；过滤“过大/过小”运动得到 `S_M`（仅靠时序一致性不足以剔除高速运动）。  
4) **交集得到稳定高质候选**：`S_I = S_A ∩ S_T ∩ S_M`。  
5) **清晰度筛选（Clarity）**：用 DOVER-Technical 给 `S_I` 打技术质量分，取最高 30% 得到清晰纹理更好的集合 `S`（并在文中用 Figure 3 展示清晰度分布与样例）。  
6) **单场景切分（Clip extraction）**：用 Cascaded Cut Detector 将多场景片段切成单场景，得到 `S_e`。  
7) **长 caption 重写（Video caption）**：用 LLaVA-v1.6-34b 对视频重新生成更长、更细致的描述；作者强调这类“高信息量 prompt”对生成至关重要，并用统计对比展示 caption 长度显著提升。  

最终得到 OpenVid-1M（并额外整理出 1080p 的 OpenVidHD-0.4M）。

### B. MVDiT：并行视觉-文本的 Video DiT
作者认为“只把文本当 conditioning”不够，提出并行分支并把两种 token 的交互做成**结构化模块**：
- 输入侧：视觉 token（VAE 编码后）+ 文本 token（T5 编码）并行进入每层。
- 关键模块（每层迭代）：
  1) **MMSA（Multi-Modal Self-Attention）**：把视觉 token 与文本 token 拼接做 self-attention，让两类 token 在每帧内直接交互；同时引入缩放系数 `α`（论文指出能加速收敛，类似 DiT 中的实践）。  
  2) **MMTA（Multi-Modal Temporal-Attention）**：沿时间维做注意力，而且是“多模态”的——同时考虑视觉与文本 token 的时序通信，用来提升时序一致性与语义一致性。  
  3) **MHCA（Multi-Head Cross-Attention）**：显式把文本语义灌入视觉 token（Query=视觉、Key/Value=文本），弥补仅靠拼接 self-attn 可能仍不足的问题。  
- 消融结论（来自文中描述 + Table 7 线索）：
  - 去掉 MHCA 会同时伤害视频质量与文本对齐；
  - 去掉 `α` 会让 loss 下降很慢（训练更难）；
  - 去掉 MMTA 会“直接生成失败”（无法形成连贯视频而变成不相关图像序列），说明时序模块是视频生成的刚需。

注：论文中的 **Figure 4（MVDiT 结构总览）在本次 `extract-images` 的结果里未出现**，很可能是矢量/组合对象导致“原图自动提取不完整”。本笔记因此用文字把模块逻辑补齐，并用后面的实验图（Figure 5/6/8/9 等）辅助理解其设计动机与效果。

## 关键公式或算法直觉
- 数据筛选本质是把“不可训练/会拖后腿”的样本从训练分布里剔除：  
  - 时序一致性过滤掉“静帧/闪烁”；  
  - 运动幅度过滤掉“过度运动/过于静止”；  
  - 清晰度过滤掉“糊/纹理差”；  
  - 最终用交集约束，得到更“干净、稳定、可对齐”的训练对。  
- MVDiT 的直觉：把文本 token 当作与视觉 token 等价的重要信息流，通过“帧内交互（MMSA）+ 跨帧一致性（MMTA）+ 显式语义注入（MHCA）”三件套把语义真正写进生成过程，而不是仅仅作为条件提示。

## 关键原图讲解（3–6 张）

### 图 1：OpenVid-1M 的“长而精确”caption 对比（PDF 第 2 页）
![图：OpenVid-1M 对比既有数据集（caption 更长更细）](../figures/2025-openvid-1m-a-large-scale-high-quality-dataset-for-text-to-video-generation/page_002_01_img-002-009.jpg)
- 来自 PDF：第 2 页（Figure 1 的组成部分）
- 图展示了什么：把 UCF-101（类别标签式 caption）、WebVid/Panda（短且不精确 caption）与 OpenVid-1M 的更长描述放在一起对比。
- 如何解读：这张图强调“caption 质量”不仅是长度，更是细节密度（实体/动作/场景细节），直接决定模型能否学到可控语义映射。
- 与核心贡献关系：支撑“OpenVid-1M 的 caption 更 expressive”这一卖点，也是后续作者用人评（Table 8）与指标验证的前提。

### 图 2：多维质量统计对比（PDF 第 4 页）
![图：OpenVid-1M vs Panda 的质量统计对比（美学/清晰度/时序/运动/caption 等）](../figures/2025-openvid-1m-a-large-scale-high-quality-dataset-for-text-to-video-generation/page_004_01_img-004-024.jpg)
- 来自 PDF：第 4 页（Figure 2 的组成部分）
- 图展示了什么：用多种统计分布对比 OpenVid-1M 与 Panda（文中也提到包含美学分、清晰度、运动/时序相关统计、caption 长度等维度）。
- 如何解读：不是“再堆数据量”，而是把训练最敏感的质量维度显式量化，并通过筛选把分布推向更可训练区域。
- 与核心贡献关系：对应表 1 的流水线设计动机；也解释了为什么作者后面做 step-wise ablation（Table 9）能看到一致的指标提升。

### 图 3：训练规模/收敛/一致性曲线（PDF 第 7 页）
![图：SOTA 对比与训练曲线（FVD、时序一致性等）](../figures/2025-openvid-1m-a-large-scale-high-quality-dataset-for-text-to-video-generation/page_007_01_img-007-077.png)
- 来自 PDF：第 7 页（Figure 5 的组成部分）
- 图展示了什么：左侧给出若干 SOTA T2V 模型在质量指标与资源（GPU/分辨率）上的对比；中/右展示训练过程中的 FVD、Clip_temp_score、warping_error 等曲线。
- 如何解读：作者在强调两点：一是他们在既定算力下能把指标“训练到位”；二是时序一致性相关指标需要足够训练步数才会稳定。
- 与核心贡献关系：为“数据集 + 模型设计”能在标准指标上跑出竞争力提供证据，也为后续消融（Table 6/7）提供参照。

### 图 4：定性结果对比（PDF 第 9 页）
![图：不同模型的生成视频定性对比（清晰度/细节/运动/文本理解）](../figures/2025-openvid-1m-a-large-scale-high-quality-dataset-for-text-to-video-generation/page_009_01_img-009-103.jpg)
- 来自 PDF：第 9 页（Figure 6/7 的组成部分，文中在“Qualitative Evaluation”处引用）
- 图展示了什么：展示作者方法与其他模型在具体 prompt 下的生成差异（文中提到清晰度、美学、细节与运动质量、以及对特定语义（如“android”“kicking up dust”）的理解）。
- 如何解读：定性图通常最能暴露数据与文本对齐的短板：若数据 caption 不精确/噪声大，模型常出现“理解不到位/细节糊/运动不自然”等现象。
- 与核心贡献关系：与 Table 4 的“同架构 STDiT 但换数据集”对比结论互相印证：数据质量与 caption 信息量会反映到可见的生成质量差异上。

### 图 5：数据处理步骤的效果可视化（PDF 第 15 页）
![图：不同清晰度/美学/运动等维度的可视化示例](../figures/2025-openvid-1m-a-large-scale-high-quality-dataset-for-text-to-video-generation/page_015_01_img-015-145.jpg)
- 来自 PDF：第 15 页（Figure 8 的组成部分）
- 图展示了什么：把视频质量分解成多个可观察维度（清晰度、美学、运动等）并给出可视化示例。
- 如何解读：这图的价值在于把“为什么要做那么多筛选”讲清楚：这些维度彼此独立，单一指标不足以保证可训练性，需要组合约束。
- 与核心贡献关系：对应 Table 9 的 step-wise ablation 结论：分步筛选在不同指标上各有贡献，组合后收益最大。

## 实验与评价（Evidence）
### 1) 数据集对比：同一模型，换数据集（Table 4 的结论）
作者用 OpenSora 的 STDiT 固定训练设置做对比：在 256×256 分辨率时，OpenVid-1M 训练的模型在除 VQAT 外的大多数指标最好；在更高分辨率（如 1024×1024）结论类似。作者解释：低分辨率下很难充分体现高质量数据的优势；而 OpenVidHD-0.4M 可以直接用于 HD 训练，避免对低质数据做超分带来的副作用。

### 2) 模块消融：MVDiT 的关键组件（Table 7 + 文中描述）
- MHCA：提升视频质量与文本对齐（语义显式注入是必要的）。  
- `α` scaling：显著改善收敛速度（去掉后 loss 降得很慢）。  
- MMTA：对“视频性”至关重要（去掉会生成失败，只剩无关图像）。  

### 3) 数据处理消融：每个筛选环节的贡献（Table 9 文中总结）
作者总结了流水线的 step-wise 影响：  
1) Temporal screening 明显改善 Clip_temp_score / warping_error（时序一致性）；  
2) Aesthetics+temporal+motion 提升 VQAA/VQAT/Blip_bleu（美学与文本理解）；  
3) Clarity 筛选显著提升 VQAA/VQAT（清晰度与整体质量）；  
4) 四步组合整体最好。

## 局限性（Limits）
1) **版权/许可与可持续性**：大规模 in-the-wild 视频的开源/再分发始终是潜在风险点（即便提供索引/脚本，也会受平台可用性影响）。  
2) **筛选带来的偏置**：过度偏好“高美学/高清晰/适中运动”的分布，可能降低长尾场景与极端运动/低光等真实分布覆盖。  
3) **caption 重写的幻觉风险**：LLaVA 重写虽然更长，但可能引入少量幻觉；作者用人评（Table 8）在一定程度上检查了 omission/hallucination 等问题，但规模与覆盖仍有限。  
4) **高算力的数据清洗成本**：表 1 显示流水线需要多种模型与大量 GPU 资源，复现门槛较高。

## 与库中相关论文的关系（Links）
- 与复现/工程路线更强相关：[[2026-05-14_2025-open-sora-2-0-training-a-commercial-level-video-generation-model-in-200k]]（关注训练成本、工程与大规模训练策略；本篇更偏“数据质量 + caption + 结构化多模态注意力”）。  
- 其它可能的对照阅读：
  - [[papers/2024_VideoCrafter2 Overcoming Data Limitations for High-Quality Video Diffusion Models_CVPR.pdf]]
  - [[papers/2025_CogVideoX Text-to-Video Diffusion Models with An Expert Transformer_ICLR.pdf]]
  - [[papers/2024_From Sora What We Can See A Survey of Text-to-Video Generation.pdf]]
  - [[papers/2023_CogVideo Large-scale Pretraining for Text-to-Video Generation via Transformers_ICLR.pdf]]

## 后续阅读建议（Next）
1) 如果你在做“开源复现/训练”：先读 [[2026-05-14_2025-open-sora-2-0-training-a-commercial-level-video-generation-model-in-200k]] 把训练系统和成本框架搭起来，再用本篇的流水线补齐数据质量与 caption。  
2) 如果你要做“数据构建/评估”：重点复刻表 1 的筛选逻辑 + Table 9 的 step-wise 评估方式，把每一步对指标的边际贡献算清楚。  
3) 如果你要做“模型结构”：把 MVDiT 里 **并行文本分支 + MMTA + 显式 cross-attn（MHCA）** 当作“文本语义利用的工程化模版”，再对比 CogVideoX 等架构路线。



