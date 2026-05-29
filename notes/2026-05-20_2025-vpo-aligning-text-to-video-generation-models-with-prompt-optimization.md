---
type: paper-note
aliases:
  - "VPO Aligning Text-to-Video Generation Models with Prompt Optimization"
paper_id: "2025-vpo-aligning-text-to-video-generation-models-with-prompt-optimization"
title: "VPO Aligning Text-to-Video Generation Models with Prompt Optimization"
year: 2025
venue: "ICCV"
subfield: "Video Generation"
topics:
  - "prompt-alignment"
  - "text-to-video"
importance: 5
read_status: "unread"
note_status: "generated"
selected_on: "2026-05-20"
last_reviewed_on: ""
paper: "[[literature/papers/2025-vpo-aligning-text-to-video-generation-models-with-prompt-optimization]]"
pdf: "[[papers/2025_VPO_Aligning Text-to-Video Generation Models with Prompt Optimization_ICCV.pdf]]"
tags:
  - paper
  - paper/video_generation
  - topic/prompt_alignment
  - topic/text_to_video
  - venue/iccv
  - year/2025
---
# VPO：用“提示词优化器”对齐文生视频模型

## 论文信息
- 标题：VPO: Aligning Text-to-Video Generation Models with Prompt Optimization
- 年份/会议：2025 / ICCV（论文同时有 arXiv:2503.20491v2，2025-08-30）
- 任务：将真实用户“短、糙、含糊/潜在不安全”的输入，改写成更适合视频模型生成、且更安全/更对齐的 prompt
- PDF：[[papers/2025_VPO_Aligning Text-to-Video Generation Models with Prompt Optimization_ICCV.pdf]]

## 细分领域
Video Generation / Prompt Optimization / Alignment（把“对齐原则 + 偏好优化”引入视频提示词改写）

## 重要程度
5/5：直接解决“训练 caption vs 推理用户输入”落差，把 prompt 改写从纯 LLM ICL 推到“可训练的对齐模块”，并显式把最终视频质量纳入优化目标。

## 一句话总结
VPO 把“提示词改写”变成一个可训练的策略模型：先用三原则（Harmless / Accurate / Helpful）做 SFT 打底，再用**文本层面的原则判定** + **视频层面的 reward** 双反馈做 DPO，让改写 prompt 同时更安全、对齐且更能产出高质量视频。

## 背景问题
文生视频模型训练时用的是“长、细、结构化”的 caption；但推理时用户输入常常：
1) 太短/缺信息；2) 容易把意图改坏（misalignment）；3) 引入安全风险；4) **即使语义更丰富，也未必能让最终视频更好**。

## 核心贡献
1) 提出视频 prompt 优化器的三原则：**Harmless / Accurate / Helpful**，把“对齐”写成可操作的判定与训练信号。
2) 两阶段框架：**Principle-Based SFT + Multi-Feedback Preference Optimization(DPO)**，把文本层面与视频层面反馈融合到同一个偏好学习目标里。
3) 在 CogVideoX、Open-Sora 1.2 等模型上验证：提升对齐/质量并降低不安全率；人评相对原始 query 与官方改写有明显 win-rate 提升。

## 方法详解

### Step 1：Principle-Based SFT（原则驱动的监督微调）
目标：先训练一个“会写好 prompt”的基础模型（SFT model）。

数据构建（抽象版）：
1) **Query Curation**：从真实用户查询集合中筛选/去重（文中提到基于 VidProM 的去重版本并做规则过滤），同时补充安全相关 query。
2) **Initial prompt generation**：用强 LLM（文中示例为 GPT-4o ICL）把 `user query x` 写成初版 prompt `p`（更长、更结构化）。
3) **Principle-based refinement**：LLM-as-a-judge 对 `p` 做三原则审查并给出 critique `c`，若有问题则根据 critique 产出 `p_refined`。
4) **SFT 目标**：得到监督对 `(x, s)`，其中 `s = p` 或 `p_refined`，用标准 NLL 做监督训练。

关键点：SFT 阶段“只保证原则层面正确”，但还没有把“最终视频质量”作为训练信号显式纳入。

### Step 2：Multi-Feedback Preference Optimization（双反馈偏好优化 / DPO）
目标：在 SFT 模型基础上进一步提升——不仅 prompt 文本更对齐/安全，而且**更能产出好视频**。

对每个 query `x`：
1) 从 SFT 模型采样 K 个候选 prompt：`p1..pK`
2) 构造两类偏好对：
   - **Text-level preference（原则/意图对齐）**：LLM-as-a-judge 找出违反原则或丢关键信息的 prompt，并产出修正版 `p_refined`，形成偏好对 `(x, p_bad < p_refined)`。
   - **Video-level preference（最终视频质量）**：对通过文本检查的 prompt，用目标视频生成模型生成视频，再用视频 reward 模型（文中用 VisionReward）打分 `r`，用分数排序形成偏好对 `(x, p_low < p_high)`。
3) 用 DPO 在 SFT 模型上训练，训练集为两类偏好对的并集。

直觉：文本层面对齐/安全是“硬门槛”，视频层面 reward 是“有效性/可生成性”的排序信号；两者合并避免“写得好看但生成更差”的 prompt。

## 关键公式/算法直觉
1) **SFT**：最常规的条件语言模型 NLL（对目标 prompt token 做监督）。
2) **DPO**：把偏好 `(p_w, p_l)` 变成一个二分类/排序学习目标：提高 `p_w` 相对 `p_l` 的对数似然差（相对 reference policy），用 sigmoid/温度系数 `β` 控制更新强度。

## 关键原图讲解

### 图1：VPO 总体框架（两阶段 + 双反馈）
![图：VPO总体框架](../figures/2025-vpo-aligning-text-to-video-generation-models-with-prompt-optimization/fig2_framework.png)
- 来自 PDF 第 4 页（Figure 2）。
- 展示什么：左侧是“原则驱动的 SFT”——从用户 query 生成初版 prompt，再做 critique/refine；右侧是“多反馈偏好优化”——对候选 prompt 既做文本层面的对齐审查（Text-level alignment critique），也用视频 reward 做质量排序（Video-level）。
- 如何解读：把 prompt 改写拆成“先学会写 + 再学会写得更有用”；尤其是右侧把文本层面对齐与视频层面质量统一进 DPO 数据构造。
- 和核心贡献的关系：这是 VPO 的方法核心，解释了为什么它不止是 LLM 改写，而是一个“可训练的 prompt optimizer”。

### 图2：Open-Sora 1.2 的 MonetBench 定量结果（示例）
![图：MonetBench表格结果](../figures/2025-vpo-aligning-text-to-video-generation-models-with-prompt-optimization/table4_monetbench.png)
- 来自 PDF 第 8 页（Table 4）。
- 展示什么：在 Open-Sora 1.2 上，VPO 相比 Original Query / GPT-4o Few-Shot / VPO-SFT 的 Alignment、Stability、Preservation、Physics 与 Overall。
- 如何解读：VPO 在 Overall 上最高（3.18），并在多个子项上相对更优；这表明“仅做 SFT（VPO-SFT）”能带来提升，但加入双反馈的 DPO 后还能再涨。
- 和核心贡献的关系：直接支持“多反馈偏好优化是必要的”这一主张。

### 图3：安全等级分布（Level 1~4）
![图：安全等级分布](../figures/2025-vpo-aligning-text-to-video-generation-models-with-prompt-optimization/page_007_02_img-007-046.png)
- 来自 PDF 第 7 页（安全评估相关图表）。
- 展示什么：不同 prompt 优化方法下，生成内容的安全等级分布（Level 1 最安全，Level 4 最不安全）。
- 如何解读：关注各方法 Level 1/2 的占比是否上升、Level 3/4 是否下降；VPO 在更安全等级上的占比更高，说明文本层面对齐与安全约束确实传导到了最终输出。
- 和核心贡献的关系：对应“Harmless 原则 + text-level feedback”的有效性。

### 图4：安全任务 case study（同一危险 query 的不同改写与生成效果）
![图：安全任务案例](../figures/2025-vpo-aligning-text-to-video-generation-models-with-prompt-optimization/fig10_safety_case.png)
- 来自 PDF 第 14 页（Figure 10）。
- 展示什么：同一条包含潜在血腥/危险细节的查询，在 Original Query、Few-shot 改写、VPO-SFT、VPO(ours) 等设置下的生成差异（配合文字描述标出危险细节是否被保留/缓和）。
- 如何解读：这类图的重点不是“画质好不好”，而是 VPO 是否能把危险细节改写为更安全表达，同时仍保留事件结构与可生成性（即 Accurate + Harmless 的折中）。
- 和核心贡献的关系：解释 VPO 的“原则对齐”在真实样例上的行为边界：不是简单删除，而是安全地重写并补足细节。

## 实验与评价（读后要点）
1) 多基座验证：在 CogVideoX 与 Open-Sora 1.2 等模型上做实验，说明它更像“可迁移的 prompt optimizer”，而不是某个模型的特例。
2) 人评与安全：文中报告人评 win-rate 相对原始 query 与官方改写有明显提升，并显著降低不安全率（对应上面的安全等级分布与 case study）。
3) 诊断结论：对比 VPO-SFT 与 VPO，支持“只做原则 SFT 不够，必须把视频层面反馈纳入偏好优化”。

## 局限性
1) 训练/数据成本：需要 LLM-as-a-judge 的 critique/refine，以及用目标视频模型生成视频再跑 reward 模型，整体代价不小。
2) reward 偏差：视频 reward 模型（如 VisionReward）的偏好会影响 prompt optimizer 的优化方向，可能导致“迎合 reward”而非真实用户偏好。
3) 原则折中：Harmless vs Accurate 的边界需要设计得很细；不同应用场景（教育/医疗/恐怖片等）对“安全”定义不同。

## 与库中相关论文的关系
- 与生成模型本体：[[2026-05-10_2025-wan-open-and-advanced-large-scale-video-generative-models]]、[[2026-05-12_2025-hunyuanvideo-a-systematic-framework-for-large-video-generative-models]]、[[2026-05-14_2025-open-sora-2-0-training-a-commercial-level-video-generation-model-in-200k]]（VPO 属于“前置的对齐/提示词层”，可作为这些系统的可插拔模块）。
- 与评测基准：[[papers/2024_VBench Comprehensive Benchmark Suite for Video Generative Models_CVPR.pdf]]（文中使用 VBench / MonetBench 做定量评测）。
- 与“从视频侧反馈优化”：VPO 的 video-level DPO 属于“把最终生成质量回流到 prompt 改写策略”的范式，可对照阅读 Prompt-A-Video、Diffusion DPO 等路线（若库中后续补齐可互相链接）。

## 后续阅读建议
1) 先读评测：VBench / MonetBench，明确各指标到底测什么、怎么避免刷分。
2) 再读对齐方法：DPO / Diffusion DPO / RLHF-for-generation（理解“偏好学习如何作用在生成策略上”）。
3) 最后做工程落地：把 prompt optimizer 作为推理前置模块，观测不同用户域（安全敏感 vs 创意写作）下 Harmless/Accurate 的权衡策略。
























