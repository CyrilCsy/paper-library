---
type: paper-note
aliases:
  - "VideoREPA Learning Physics for Video Generation through Relational Alignment with Foundation Models"
paper_id: "2025-videorepa-learning-physics-for-video-generation-through-relational-alignment-wit"
title: "VideoREPA Learning Physics for Video Generation through Relational Alignment with Foundation Models"
year: 2025
venue: "NEURIPS"
subfield: "Video Generation"
topics:
  - "text-to-video"
importance: 5
read_status: "unread"
note_status: "generated"
selected_on: "2026-05-19"
last_reviewed_on: ""
paper: "[[literature/papers/2025-videorepa-learning-physics-for-video-generation-through-relational-alignment-wit]]"
pdf: "[[papers/2025_VideoREPA_Learning Physics for Video Generation through Relational Alignment with Foundation Models_NeurIPS.pdf]]"
tags:
  - paper
  - paper/video_generation
  - topic/text_to_video
  - venue/neurips
  - year/2025
---
## 一句话总结
用**“关系蒸馏”而不是“特征硬对齐”**把视频理解基础模型（如 VideoMAEv2）里隐含的物理时空结构，迁移进预训练 T2V 扩散模型（CogVideoX）里，从而提升生成视频的物理合理性。

## 论文信息
- 标题：VideoREPA: Learning Physics for Video Generation through Relational Alignment with Foundation Models
- 年份/会议：2025 / NEURIPS（预印本显示 under review）
- arXiv：2505.23656v1（2025-05-29）
- PDF：[[papers/2025_VideoREPA_Learning Physics for Video Generation through Relational Alignment with Foundation Models_NeurIPS.pdf]]
- 主要任务：Text-to-Video（T2V）扩散模型的**物理常识/物理一致性**提升
- 关键词：physics commonsense、representation alignment、distillation、CogVideoX、VideoMAEv2、VideoPhy / VideoPhy2 / Physion

## 细分领域与重要程度
- 细分领域：Video Generation（偏“物理一致性/物理常识”方向）
- 重要程度：5/5
- 为什么重要：
  - 不引入外部物理模拟器、也不依赖“显式物理现象”小数据集（对比 WISA-32K），而是尝试**从视频理解模型蒸馏物理知识**，更贴近“开域 + 规模化”的路线。
  - 关键点是提出 **TRD（Token Relation Distillation）**：把“物理”视作 token 间**时空关系结构**，从而在 finetune 预训练 VDM 时更稳定。

## 背景问题（作者在解决什么）
当前 T2V 扩散模型虽然画面质量越来越高，但经常出现明显违反直觉物理的现象（刚体滚动不对、接触/支撑关系不对、碎裂/流体不对等）。常见提升路径：
- **扩大数据/改架构**：成本高、且不一定显式建模物理。
- **引入物理模拟器**：适用范围受限，开域复杂现象难覆盖。
- **非模拟训练策略**（本文关注）：希望在开域数据上让模型“更懂物理”，从而生成更合理。

作者的关键观察：在 Physion 的物理理解评测上，大型 T2V 模型（CogVideoX 2B）**物理理解能力明显弱于**更小的自监督视频理解模型（VideoMAEv2 86M）。因此提出：把 VFM（Video Foundation Model）的“物理理解”迁移到 VDM（Video Diffusion Model）里。

## 核心贡献（Contributions）
1) 指出并量化了 VFM 与 VDM 的**物理理解鸿沟**（Physion/OCP）。
2) 提出 VideoREPA：面向**预训练 VDM 的微调（finetune）**，用 TRD 损失做“关系蒸馏”，同时覆盖**帧内空间关系**与**跨帧时间关系**。
3) 在 VideoPhy / VideoPhy2 上显著提升物理常识评分，并给出消融与失败案例分析（说明“直接套用 REPA 会不稳定/甚至变差”）。

## 方法详解（VideoREPA / TRD）
### 1) 学生-教师设定
- 学生（要被增强的生成模型）：CogVideoX（视频 latent diffusion，内部是 denoising transformer）。
- 教师（物理理解更强的视觉基础模型）：VideoMAEv2（以及其它 VFM 作为消融对比）。

训练时，对同一段视频：
- 走一遍 VFM encoder 得到 token 表示 `y_v`。
- 走一遍 VDM（VAE 压缩到 latent，再加噪、过 denoiser）得到中间层 token 表示 `h_t`，再经一个 MLP `h_φ` 做维度对齐。

### 2) 关键思想：对齐“token 两两关系”，而不是对齐“token 特征”
直接把学生 token 特征去“硬对齐”教师特征（类似 REPA 的 cosine 相似度最大化）在 finetune 预训练 VDM 时容易：
- 破坏原有特征空间（稳定性差，甚至语义退化）
- 忽视视频的时间维（只做空间对齐不够）

TRD 改为对齐**相似度矩阵/张量**：
- 空间关系：同一帧内，所有空间 token 的两两 cosine 相似度（`hw × hw`）。
- 时间关系：跨帧，某帧某 token 与其它帧所有 token 的 cosine 相似度（`hw × hw × (f-1)`）。

这相当于让学生学习教师的“结构”（谁和谁应该更相关），作为更“软”的约束，适合 finetune。

### 3) 关键公式/直觉（TRD loss）
论文把 TRD 写成空间项 + 时间项的平均 L1 距离（并与扩散的噪声预测损失加权求和）：

$$
\\mathcal{L} = \\mathcal{L}_{diff} + \\lambda \\mathcal{L}_{TRD}
$$

直觉：
- `L_diff` 保证“我还是在学扩散去噪/生成”；
- `L_TRD` 像一个“结构老师”，把物理相关的时空关系迁移进来；
- `λ` 太大可能过度约束、影响语义；太小则蒸馏不够（论文在附录里给了经验最优范围）。

### 4) 实现层面的坑与作者的处理
- **token 尺度不匹配**：VDM 的 3D VAE 往往时间压缩更强，而 VFM token 时间分辨率更高；作者倾向于**插值 VDM latent/token 尺度去对齐 VFM**，以保留教师信息。
- **算力限制**：VFM 直接吃高分辨率 + 多帧会很贵，作者采用折中策略（文中提到做过对比）。
- **稳定微调**：用“关系蒸馏 + L1 差异”替代 REPA 的“硬相似度对齐”，并引入 margin（受 VA-VAE 启发）避免对齐无意义噪声。

## 关键原图讲解（3–6 张）
> 说明：以下图片均来自论文 PDF 的页渲染/裁剪或自动提取。若遇到矢量图拆分不完整，会以“整页渲染 + 裁剪”方式保留关键信息。

### 图 1：物理理解差距与生成效果对比（Physion + 可视化）
![图：Physion 物理理解差距与生成对比](../figures/2025-videorepa-learning-physics-for-video-generation-through-relational-alignment-wit/fig1_physion_gap.png)
- 来源：PDF 第 1 页（Figure 1）
- 展示内容：
  - 左侧：CogVideoX（baseline）、CogVideoX+REPA、VideoREPA 三者的生成对比（红框标出违反物理常识的现象）。
  - 右侧：Physion（OCP）物理理解评测曲线/对比，显示 VFM（VideoMAEv2）和 VDM（CogVideoX）之间的理解差距，以及 VideoREPA 在一定程度上缩小差距。
- 解读方式：
  - 这张图把本文“为什么做”说清楚：不是单纯让视频更清晰，而是要**让接触/支撑/运动更合理**。
  - 也点出 REPA 在 finetune 场景会出问题（作者在后文专门分析）。
- 与核心贡献的关系：给出“物理理解差距”证据 + “生成侧更物理”的初步可视化佐证。

### 图 2：VideoREPA 框架总览（TRD 作用位置）
![图：VideoREPA 总览与 TRD 对齐对象](../figures/2025-videorepa-learning-physics-for-video-generation-through-relational-alignment-wit/fig2_overview.png)
- 来源：PDF 第 3 页（Figure 2）
- 展示内容：
  - 上支路：视频经 VAE/文本条件进入扩散 transformer（MM-DiT blocks），正常优化 diffusion loss。
  - 下支路：同一视频喂给预训练视频编码器（VFM）产出 token 表示。
  - 右侧：TRD（Token Relation Distillation）对齐的是“空间关系 + 时间关系”的结构，而不是直接对齐特征向量。
- 解读方式：
  - 把“物理”抽象为**时空关系约束**：同一帧内哪些区域应保持几何/接触一致；跨帧哪些 token 应随时间保持合理的相关性变化。
- 与核心贡献的关系：对应本文方法主张——用关系蒸馏把 VFM 的物理结构迁移进 VDM。

### 表 1：VideoPhy 定量结果（物理常识 PC 显著提升）
![图：VideoPhy 结果（SA/PC）](../figures/2025-videorepa-learning-physics-for-video-generation-through-relational-alignment-wit/table1_videophy.png)
- 来源：PDF 第 7 页（Table 1）
- 展示内容：
  - 对多个 T2V 模型与方法在 VideoPhy 上的 **Semantic Adherence（SA）** 与 **Physical Commonsense（PC）** 评分对比。
  - 重点：VideoREPA-5B 在 Overall PC 上达到 40.1，相比 CogVideoX-5B baseline 明显提升（文中强调 24.1% 提升）。
- 解读方式：
  - SA 更像“视频与文本是否对得上”，PC 更像“物理是否合理”；本文主要拉升 PC，同时尽量不牺牲 SA。
  - 这张表也体现了作者的论点：WISA 在“物理显式小数据集”上有效，但开域泛化不一定好；VideoREPA 用开域 OpenVid 训练仍能提升 PC。
- 与核心贡献的关系：给出主要 benchmark 的 SOTA 级别量化证据。

### 图 4：为什么“直接 REPA”不行（微调稳定性）
![图：REPA loss vs TRD loss 的消融可视化](../figures/2025-videorepa-learning-physics-for-video-generation-through-relational-alignment-wit/fig4_repa_vs_trd.png)
- 来源：PDF 第 9 页（Figure 4）
- 展示内容：对比用不同对齐损失微调后的生成片段（REPA loss 的两种教师特征 vs TRD loss）。
- 解读方式：
  - 作者把 REPA 称为更“硬”的对齐（强迫特征空间直接靠近），在预训练 VDM 上容易破坏原有表示，出现语义/一致性退化。
  - TRD 通过“关系结构”提供更“软”的引导，更适合 finetune。
- 与核心贡献的关系：这是 TRD 设计合理性的关键证据之一。

### 图 5：对齐层深度与权重 λ 的影响（实践可用性）
![图：对齐深度与 λ 的消融](../figures/2025-videorepa-learning-physics-for-video-generation-through-relational-alignment-wit/fig5_depth_lambda.png)
- 来源：PDF 第 14 页（Figure 5）
- 展示内容：VideoPhy 的 PC 随对齐深度（对齐哪一层/多深）与 λ 的变化曲线。
- 解读方式：
  - 说明“把 TRD 加到哪里、加多大”会影响效果；存在一个折中点（λ 过大/过小都不理想）。
- 与核心贡献的关系：把方法落到可复现的工程超参上，解释“为什么能稳定提升”。

## 实验与评价（我认为最关键的点）
### 训练设定（把结论放回训练细节里看）
- 基座：CogVideoX（2B / 5B）
- 教师：默认 VideoMAEv2
- 数据：OpenVid（开域视频-文本数据集），而不是专门的“物理现象小数据集”
- 微调：
  - 2B：32k OpenVid、4000 steps（全参）
  - 5B：64k OpenVid、2000 steps（LoRA）
  - 默认对齐深度 18

### 指标与 benchmark
- VideoPhy：SA（语义贴合）与 PC（物理常识）二指标，覆盖不同材料交互类型。
- VideoPhy2：更偏 action-centric 的物理常识评测（含更多人-物交互），文中报告 VideoREPA 相对 CogVideoX 的 PC 提升（Table 2：PC 67.97 → 72.54）。
- Physion/OCP：更像“物理理解”侧的代理任务，展示 VFM 与 VDM 的差距与蒸馏后的变化（Figure 1）。

### 消融结论（为什么 TRD 的两个分量都需要）
- TRD 同时包含空间项与时间项，去掉任意一项都会让 PC 下降；只做空间或只做时间对齐也会伤 SA（说明可能破坏生成模型已学到的表示完整性）。

## 局限性（作者明确承认 + 我补充的理解）
- 作者承认：目前主要验证在“微调预训练 VDM”场景；**用于从头预训练 VDM** 的潜力未验证（算力限制）。
- 我认为还需要关注：
  - TRD 需要同时跑 VFM encoder，训练成本会上升；不同 VFM 的选择与 token 尺度适配是工程关键。
  - “关系蒸馏”可能更偏“结构一致性”，对极端复杂物理（细粒度流体/刚体接触）是否足够仍需更多样例验证。

## 与库中相关论文的关系（建议从库里怎么串起来读）
- 基座模型与对比：
  - CogVideoX：[[papers/2025_CogVideoX Text-to-Video Diffusion Models with An Expert Transformer_ICLR.pdf]]
  - VideoCrafter2：[[papers/2024_VideoCrafter2 Overcoming Data Limitations for High-Quality Video Diffusion Models_CVPR.pdf]]
  - HunyuanVideo（同为对比模型之一）：[[2026-05-12_2025-hunyuanvideo-a-systematic-framework-for-large-video-generative-models]]
- 训练数据（本文微调依赖）：
  - OpenVid-1M：[[2026-05-15_2025-openvid-1m-a-large-scale-high-quality-dataset-for-text-to-video-generation]]
- 同期/同方向（物理一致性/评测/可控）：
  - Wan：[[2026-05-10_2025-wan-open-and-advanced-large-scale-video-generative-models]]
  - Open-Sora 2.0：[[2026-05-14_2025-open-sora-2-0-training-a-commercial-level-video-generation-model-in-200k]]
  - T2V-CompBench（更偏组合泛化评测）：[[2026-05-14_2025-t2v-compbench-a-comprehensive-benchmark-for-compositional-text-to-video-generati]]

## 后续阅读建议（按“最少成本获得最大增益”的顺序）
1) 先回看 CogVideoX（理解其 VAE 压缩/denoiser 结构），再看本文 TRD 在哪些层对齐（对照 Figure 2/5）。
2) 把 OpenVid-1M 的数据构成与偏差补上（理解“开域训练为何能泛化到物理”）。
3) 进一步找：REPA（原始方法）、以及视频理解基础模型（VideoMAEv2 / V-JEPA）在物理理解任务上的表现与差异（本文附录有消融结论）。



