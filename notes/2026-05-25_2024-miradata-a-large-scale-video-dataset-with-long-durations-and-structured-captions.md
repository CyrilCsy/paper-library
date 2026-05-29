---
type: paper-note
aliases:
  - "MiraData A Large-Scale Video Dataset with Long Durations and Structured Captions"
paper_id: "2024-miradata-a-large-scale-video-dataset-with-long-durations-and-structured-captions"
title: "MiraData A Large-Scale Video Dataset with Long Durations and Structured Captions"
year: 2024
venue: "NEURIPS"
subfield: "Dataset / Data Curation"
topics:
  - "dataset"
importance: 5
read_status: "unread"
note_status: "generated"
selected_on: "2026-05-25"
last_reviewed_on: ""
paper: "[[literature/papers/2024-miradata-a-large-scale-video-dataset-with-long-durations-and-structured-captions]]"
pdf: "[[papers/2024_MiraData_A Large-Scale Video Dataset with Long Durations and Structured Captions_NeurIPS.pdf]]"
tags:
  - paper
  - paper/dataset_data_curation
  - topic/dataset
  - venue/neurips
  - year/2024
---
# 论文信息

- 标题：MiraData: A Large-Scale Video Dataset with Long Durations and Structured Captions
- 年份：2024
- Venue：NEURIPS（Preprint / Under review 版本，arXiv:2407.06358v1）
- PDF：[[papers/2024_MiraData_A Large-Scale Video Dataset with Long Durations and Structured Captions_NeurIPS.pdf]]
- 代码/项目：论文首页给出 GitHub（mira-space/MiraData）

# 细分领域与重要程度

- 细分领域：Dataset / Data Curation（面向 T2V 的长时长 + 强运动数据集）
- 重要程度：5/5（对“数据→模型运动强度/一致性”的因果链条给了较完整证据）

# 一句话总结

MiraData 通过“多源长视频采集 + 分割/拼接保证语义连续 + 颜色/美学/运动/NSFW 四维过滤 + GPT-4V 结构化字幕”构建长时长、强运动、长文本的视频数据集，并配套 MiraBench（150 prompts、17 指标）来更系统评测长视频生成的运动强度与 3D/时序一致性。

# 背景问题（作者要解决什么）

作者观察到：公开数据集普遍 **时长短、运动弱、caption 短且不准**，难以支撑 “Sora-like 长时长/高运动/强一致性” 的视频生成训练与评测；同时现有 benchmark 往往 **缺少对运动强度与 3D 一致性** 的全面测量。

# 核心贡献（按论文主线）

1. **MiraData 数据集**：强调长时长（文中提到平均 72.1s）、更强运动、以及更长更细的 caption（dense/structured）。
2. **结构化字幕（Structured Captions）**：除 short/dense 外，再从多视角输出（主物体、背景、镜头、风格等）以增强可控性与对齐信号。
3. **MiraBench**：150 条评测 prompts（短/稠密/结构化三种版本）+ 17 个指标，覆盖时序一致性、运动强度、3D 一致性、视觉质量、文图对齐、分布相似度。
4. **MiraDiT 验证**：用一个 DiT-based 训练管线对比（WebVid-10M vs MiraData），展示数据带来的 motion strength 提升，并做 caption 粒度消融。

# 方法详解

## 1) 数据来源与采集（Sec. 3.1）

- 多源：YouTube + Videvo + Pixabay + Pexels（论文强调为生成任务挑源，尤其追求美学质量、时长与运动）。
- YouTube 部分：手工挑选 156 个高质量频道，覆盖 3D 渲染、城市/风景游览、电影、第一视角、物理规律展示、延时、人类运动等类别；示例性统计写到约 68K videos（720p），经后续处理得到约 34K videos / 173K clips（此处是 YouTube 子集的过程性数据）。

## 2) 分割与拼接：把“长视频”变成“语义连续的长 clip”（Sec. 3.2）

- 分割：shot change detection 将视频切成短片段，避免跨镜头的突变影响一致性。
- 拼接：将“本该连续但被误切”的相邻短 clip 重新 stitch 回来；是否可拼接同时依赖：
  - 视觉语言模型：Qwen-VL-Chat、LLaVA（判断场景/内容是否一致）
  - 图像特征模型：ImageBind、DINOv2（用相似度捕捉误切/相近内容）
  - 只有当两类模型都认可，才连接，降低误拼风险。
- 时长阈值：YouTube 侧保留 >40s 的 clip；其他来源（天然更接近 clip 形态）保留 >10s 的 clip。

## 3) 质量过滤与多版本数据（Sec. 3.3）

MiraData 提供 5 个数据版本，通过四个维度过滤：

- **颜色**：过滤过亮/过暗（用平均颜色与亮/暗分位统计做判别）
- **美学**：Laion-Aesthetic predictor
- **运动强度**：RAFT 光流估计（作为 motion strength 的基础信号）
- **NSFW**：Stable Diffusion Safety Checker（从视频中均匀抽 8 帧检测）

论文给出的 5 个过滤版本 clip 规模：**788K / 330K / 93K / 42K / 9K**（阈值逐步更严格）。

## 4) 视频字幕：用 GPT-4V 生成 dense + structured（Sec. 3.4）

- 为适配“只吃图片”的 GPT-4V：从视频均匀采样 8 帧，拼成 2×4 grid 单张图输入（降成本、利于整体理解）。
- 先用 Panda-70M 生成 short caption，作为提示让 GPT-4V 更聚焦“对生成有用”的描述。
- GPT-4V 输出：
  - Dense caption：覆盖主体、运动、风格、背景、镜头等（更长、更全）
  - Structured captions：把描述拆成多个视角（论文文本列出 Main Object / Background / Camera Movements / Video Style；Figure 1 的示例还展示了 short + dense + 多条结构化字段）
- 文中提到 dense/structured 的平均长度提升到约 **90 / 214 words**。

## 5) MiraBench：prompt 与指标（Sec. 4）

- Prompt：按 human/animal/object/landscape 四类，从候选 caption 里筛 50 个高质量 video-text pairs；每个 pair 生成 short/dense/structured 三种 prompt → 共 **150 prompts**。
- 17 指标来自 6 个视角：
  - 运动强度：Dynamic Degree（RAFT 光流距离均值）；Tracking Strength（用 CoTracker 追踪点路径长度均值）
  - 时序一致性：DINO(结构)/CLIP(语义) 相邻帧特征相似度；再乘 Tracking Strength 让“强运动却一致”更被认可；以及 AMT 的 Temporal Motion Smoothness
  - 3D 一致性：沿用 GVGC 思路用 MAE / RMSE 做 3D 重建误差
  - 视觉质量：LAION aesthetic；MUSIQ imaging quality
  - 文本对齐：用 ViCLIP 分 5 个方面（Camera / Main Object / Background / Style / Overall）
  - 分布相似：FVD/FID/KID

# 关键公式或算法直觉（抓住“度量设计”要点）

1. **光流 Dynamic Degree 的盲点**：光流更像“局部瞬时速度场”；当存在来回抖动或短程复杂运动时，平均光流距离可能误判运动大小。
2. **Tracking Strength 的直觉**：跟踪点跨时间的累计位移更接近“真实位移/移动距离”，更能区分“相机抖动 vs 物体长距离移动”。
3. **一致性 × 运动强度 的耦合**：作者把 DINO/CLIP 的相邻帧相似度乘上 Tracking Strength，意图避免“低运动天然高一致”的模型在一致性指标上占便宜。

# 关键原图讲解

> 说明：该 PDF 中很多图表是矢量对象，`extract-images` 往往只能提到少量 raster 图。为保证可读性，以下部分关键图来自“页面渲染后裁剪”的截图（同样存放在 `figures/<paper_id>/`）。

## 图 1：数据收集与标注流水线（Figure 1）

![图 1：Video collection and annotation pipeline](../figures/2024-miradata-a-large-scale-video-dataset-with-long-durations-and-structured-captions/fig1_pipeline.png)

- 来自 PDF：第 1 页（Figure 1）
- 展示内容：从“手工挑选 YouTube 长视频”开始，依次经历 **Splitting → Stitching（用 Qwen-VL-Chat / LLaVA 与 ImageBind / DINOv2 双重判别）→ Selection（颜色/美学/运动/NSFW）→ Captioning（Panda-70M short caption + GPT-4V dense + structured）**。
- 如何解读：这张图把 MiraData 的关键点（“长时长 + 强运动 + 可控的长 caption”）对应到可执行的数据工程步骤；也是理解论文贡献最直接的入口。
- 与核心贡献关系：对应贡献 #1（数据集构建）与 #2（结构化字幕）。

## 图 3：不同来源 clip 时长分布（Figure 3）

![图 3：Distribution of video clip duration](../figures/2024-miradata-a-large-scale-video-dataset-with-long-durations-and-structured-captions/fig3_duration.png)

- 来自 PDF：第 5 页（Figure 3）
- 展示内容：YouTube 与其他来源（Others）的 clip 时长分布对比。
- 如何解读：对 MiraData 来说，“长时长”不是只靠阈值筛出来的，还需要分割/拼接把“语义连续”片段尽量拉长；该分布也暗示不同来源的时长模式差异。
- 与核心贡献关系：对应贡献 #1（强调长时长数据）与 Sec. 3.2 的分割/拼接策略。

## 图 4：short / dense / structured 的 caption 长度分布（Figure 4）

![图 4：Distribution of caption length](../figures/2024-miradata-a-large-scale-video-dataset-with-long-durations-and-structured-captions/fig4_caption_length.png)

- 来自 PDF：第 6 页（Figure 4）
- 展示内容：三类 caption 的长度分布（short 最短，dense 居中，structured 显著更长）。
- 如何解读：structured captions 并不是简单“更长”，而是把可控要素拆成字段；这对训练时的对齐信号与评测 prompt 的可解释性更重要。
- 与核心贡献关系：对应贡献 #2（结构化字幕）与后续 caption 粒度消融（Table 4）。

## 图 5：Tracking Strength vs 光流 Dynamic Degree（Figure 5）

![图 5：Tracking Strength vs Optical Flow](../figures/2024-miradata-a-large-scale-video-dataset-with-long-durations-and-structured-captions/page_007_01_img-007-133.jpg)

- 来自 PDF：第 7 页（Figure 5）
- 展示内容：同一段视频在“光流强度”与“跟踪点路径长度”两种运动度量上的差异示例。
- 如何解读：光流可能把局部抖动/小范围变化当成“强运动”，而 tracking 更能反映跨帧的实际位移规模；作者据此在 MiraBench 中新增 Tracking Strength。
- 与核心贡献关系：对应贡献 #3（MiraBench 的运动强度指标设计）。

## 图 6：MiraDiT 的训练/推理管线（Figure 6）

![图 6：MiraDiT pipeline](../figures/2024-miradata-a-large-scale-video-dataset-with-long-durations-and-structured-captions/fig6_miradit_pipeline.png)

- 来自 PDF：第 8 页（Figure 6）
- 展示内容：用 2D encoder + 3D decoder 的 VAE 做视频 latent；文本侧用（Flan-）T5 编码 long captions；主干是拆开的空间/时间注意力与 cross-attention，并通过调制（AdaLN 类）注入 timestep 与 fps 条件。
- 如何解读：作者用一个相对“可复用的 DiT 骨架”来做控制变量实验，主旨是验证“数据与 caption”对 motion strength / consistency 的影响，而不是追求最强模型。
- 与核心贡献关系：对应贡献 #4（用 MiraDiT 证明 MiraData 的有效性）。

# 实验与评价（抓住能复用的结论）

## 1) 数值统计（Table 2）

- 论文用 **光流强度** 与 **美学分数** 对比多数据集：MiraData（filtered）在光流强度上最高（表中为 6.93），美学分数也保持在较高水平（约 5.02），支持“更强运动 + 更高视觉质量”的定位。

## 2) 训练数据的影响（Table 3）

- 同一 MiraDiT 架构：WebVid-10M 训练 vs MiraData 训练，对运动强度提升显著：
  - Dynamic Degree：7.12 → 15.46
  - Tracking Strength：22.36 → 49.47
- 同时在 DINO/CLIP temporal consistency、3D consistency、文本对齐等指标上也整体更好或相当（表中多项更优）。

## 3) caption 粒度的影响（Table 4）

- short → dense → structural：视觉质量不一定单调提升，但运动强度/一致性/对齐显著受益：
  - Dynamic Degree：9.45 → 17.39 → 19.53
  - Tracking Strength：27.03 → 52.53 → 68.85
  - Overall Alignment：7.73 → 14.88 → 15.36
- 这条结论对“训练时到底要不要长 caption / 结构化字段”非常有参考价值：**更细粒度的语义约束更像是在给模型提供可控维度，而不仅是增加 token。**

# 局限性（论文明确提到的）

- 数据偏差与标注误差：多源采集与自动标注不可避免引入偏差/幻觉。
- 覆盖不充分：仍可能对某些场景/风格不够全面。
- 指标的适用性：在少见场景（抖动、过曝等）上，MiraBench 指标可能不稳定或不准确。
- 潜在社会影响：更强视频生成能力可能带来 deepfake、隐私与有害内容风险（论文在讨论中明确提示）。

# 与库中相关论文的关系（建议怎么串起来读）

- 与通用评测：[[literature/papers/2024-vbench-comprehensive-benchmark-suite-for-video-generative-models|VBench]]（MiraBench 继承并扩展了 VBench 的部分思想，重点补“运动强度/3D 一致性”）。
- 与开源系统训练：[[literature/papers/2025-open-sora-2-0-training-a-commercial-level-video-generation-model-in-200k|Open-Sora 2.0]]（更偏系统工程与训练配方；MiraData/MiraBench 可作为数据与评测侧的补强）。
- 与模型/数据限制：[[literature/papers/2024-videocrafter2-overcoming-data-limitations-for-high-quality-video-diffusion-model|VideoCrafter2]]（强调数据限制下的训练与模型策略；可对照“数据过滤 + caption 质量”对结果的影响）。
- 与趋势综述：[[literature/papers/2024-from-sora-what-we-can-see-a-survey-of-text-to-video-generation|From Sora… Survey]]（把 MiraData 放到 Sora 后的“数据/评测缺口”脉络里理解更顺）。

# 后续阅读建议（按优先级）

1. 先读：[[literature/papers/2024-vbench-comprehensive-benchmark-suite-for-video-generative-models|VBench]]，理解视频生成评测常用维度与局限。
2. 再读：[[notes/2026-05-14_2025-open-sora-2-0-training-a-commercial-level-video-generation-model-in-200k|Open-Sora 2.0 笔记]]，把“数据/评测”与“系统训练”两条线合起来看。
3. 想深挖指标：顺着 MiraBench 的 Tracking Strength（CoTracker）与 3D consistency（GVGC）把原论文/实现细节补齐。
4. 想做复现/迁移：重点复用 Sec. 3.2/3.3 的“拼接 + 四维过滤”范式，再针对你的目标域重新设阈值与 prompt。












