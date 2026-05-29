---
type: paper-note
aliases:
  - "TIP-I2V A Million-Scale Real Text and Image Prompt Dataset for Image-to-Video Generation"
paper_id: "2025-tip-i2v-a-million-scale-real-text-and-image-prompt-dataset-for-image-to-video-ge"
title: "TIP-I2V A Million-Scale Real Text and Image Prompt Dataset for Image-to-Video Generation"
year: 2025
venue: "ICCV"
subfield: "Dataset / Data Curation"
topics:
  - "dataset"
  - "image-to-video"
  - "prompt-alignment"
  - "text-to-video"
importance: 5
read_status: "unread"
note_status: "generated"
selected_on: "2026-05-18"
last_reviewed_on: ""
paper: "[[literature/papers/2025-tip-i2v-a-million-scale-real-text-and-image-prompt-dataset-for-image-to-video-ge]]"
pdf: "[[papers/2025_TIP-I2V A Million-Scale Real Text and Image Prompt Dataset for Image-to-Video Generation_ICCV.pdf]]"
tags:
  - paper
  - paper/dataset_data_curation
  - topic/dataset
  - topic/image_to_video
  - topic/prompt_alignment
  - topic/text_to_video
  - venue/iccv
  - year/2025
---
# TIP-I2V：面向 I2V 的“真实用户提示词 + 首帧”百万级数据集

## 论文信息
- 标题：TIP-I2V: A Million-Scale Real Text and Image Prompt Dataset for Image-to-Video Generation
- 年份：2025（arXiv:2411.04709v2，2025-07-09）
- 会议：ICCV 2025
- PDF：[[papers/2025_TIP-I2V A Million-Scale Real Text and Image Prompt Dataset for Image-to-Video Generation_ICCV.pdf]]
- 项目页：论文中给出（tip-i2v.github.io）

## 细分领域
- Dataset / Data Curation（面向 Image-to-Video 的提示词与输入图像数据）

## 重要程度
- 5/5（对 I2V 来说“用户怎么写提示词、想让图里什么动起来”是核心现实问题；该数据集把 prompts / subject / direction / 多模型生成结果打包在一起，同时还提出 TIP-Eval 做更贴近真实使用的评测）

## 一句话总结
TIP-I2V 把真实 I2V 用户的“文本提示词 + 输入图像（首帧）”规模化收集到 **170 万级**，并为其中一部分 prompts 生成多种 I2V 模型输出，进一步用这些真实 prompts 构建了更“全面 + 实用”的评测集 TIP-Eval，用来揭示当前 I2V 模型在一致性与对齐上的真实短板（尤其是 video-text alignment）。

## 背景问题（Why）
- I2V（image-to-video）在应用层很受欢迎：相比 T2V，**更可控**、**更一致**、更适合社媒的“围绕一个主体”叙事。
- 但研究侧长期缺口是：我们很少有“从用户视角”出发的数据——真实用户会：
  - 提供一张图（首帧/参考图），期待对图中对象做“**动作/运动方向**”控制；
  - 写的文本往往不是“描述新画面”，而是“让图里的某些东西按某方式动起来”。
- 现有 prompt-gallery 数据集更多面向 T2V / T2I（例如 VidProM、DiffusionDB），语义重心与 I2V 不同，因此需要专门的 I2V prompt 数据集。

## 核心贡献（What）
1. **TIP-I2V 数据集**：来自真实 I2V 用户的 **1,701,935** 条（text, image）prompt，并包含丰富元信息（时间戳、匿名用户、embedding、NSFW、subject/direction 等）。
2. **多模型生成结果**：为一部分 prompts 生成 5 个 I2V 扩散模型的输出视频，方便研究者做对比评估/误用分析等。
3. **与类似数据集对比与语义分析**：系统对比 TIP-I2V vs VidProM vs DiffusionDB（规模、prompt 形态、语义分布）。
4. **TIP-Eval**：用真实用户 prompts 构建更贴近实际使用的 I2V 评测集，覆盖更多 subject 与 prompt 多样性，并给出基准测试观察。

## 方法详解（How）
### 1) 一个数据点都包含哪些字段？
TIP-I2V 的核心不是“只有 prompts”，而是把 prompts 变成可研究对象：**谁（匿名）在什么时候，用什么图+什么文本，让什么主体以什么方式运动**，以及不同模型生成的结果如何。

### 2) 数据是怎么采集与加工的？
- **源数据**：从 Pika 的公开 Discord 渠道导出聊天记录（HTML）。
- **抽取与清洗**：正则解析 text prompt 与视频链接、去重、校验链接；得到 170 万级唯一 prompt 对应的（约 3 秒）视频。
- **图像 prompt 的来源**：原始 image prompt 不可直接访问时，从抓取视频中解析出“首帧/输入图像”（论文称其质量较高）。
- **结构化标注与安全信息**：
  - 生成 UUID、匿名 UserID、时间戳；
  - 用 LLM 推断 **subject**（主体）与 **direction**（运动方向/动作类型）；
  - 生成 text/image embedding；
  - 给 text/image 打 NSFW/有害性标记（用于研究安全与过滤）。
- **扩展：其他 I2V 模型生成**：出于算力开销，论文只对 100k 随机 prompts/每个模型生成视频，但鼓励后续研究者用新模型继续扩展。

## 关键公式或算法直觉
- 这篇工作几乎没有“新模型公式”，核心直觉是把 I2V 的交互拆成结构化问题：
  - I2V prompt 的语义重点是“**对既定图像的运动控制**”，而不是“从零描述一个场景”；
  - 用 **subject**（图里谁/什么）与 **direction**（怎么动）把真实用户需求拆出可统计、可采样、可评测的维度；
  - 用真实 prompts 构建 TIP-Eval 时，把“覆盖面（subject 多）”和“每个 subject 的 prompt 多样性（每类配多条真实 prompts）”同时纳入，避免只用少量专家 prompt 造成偏差。

## 关键原图讲解
> 说明：本次笔记的“关键图”来自对 PDF 页面栅格化截图后裁剪（用以补全矢量图/表在 `extract-images` 中不易完整提取的问题）。

### 图 1：TIP-I2V 的整体直观（Figure 1，PDF 第 1 页）
![图：TIP-I2V 概览（示例 prompts + 图像多样性）](../figures/2025-tip-i2v-a-million-scale-real-text-and-image-prompt-dataset-for-image-to-video-ge/fig1_tip_i2v_overview.png)
- 来自：PDF 第 1 页 Figure 1
- 展示：真实用户的“文本提示词 + 输入图像”示例，以及图像 prompt 的大规模多样性拼图。
- 解读：I2V 的文本往往是“让图里某物动起来”的指令句式（例如强调某个局部对象、某种运动方式/相机运动）。
- 与贡献关系：对应贡献 1（TIP-I2V 数据集的规模与语义特征）。

### 图 2：一个数据点的字段长什么样（Figure 2，PDF 第 3 页）
![图：TIP-I2V 单条数据的结构字段（UUID/UserID/Prompt/Subject/Direction/Embedding/多模型视频）](../figures/2025-tip-i2v-a-million-scale-real-text-and-image-prompt-dataset-for-image-to-video-ge/fig2_data_point_box.png)
- 来自：PDF 第 3 页 Figure 2
- 展示：TIP-I2V 把每条记录结构化为：匿名身份、文本/图像 prompts、subject/direction、NSFW、embedding，以及不同 I2V 模型生成视频。
- 解读：这张图告诉你 TIP-I2V 的“研究接口”——你可以按 subject/direction 做分层统计、按 embedding 做聚类/检索、按 NSFW 做安全研究、按多模型输出做评测与诊断。
- 与贡献关系：对应贡献 1（数据集字段设计）与贡献 4（为 TIP-Eval / 后续研究提供抽样维度）。

### 图 3：TIP-Eval 与现有 I2V benchmark 的差异（Table 2，PDF 第 6 页）
![表：TIP-Eval 更全面也更贴近真实使用](../figures/2025-tip-i2v-a-million-scale-real-text-and-image-prompt-dataset-for-image-to-video-ge/fig_table2_tip_eval.png)
- 来自：PDF 第 6 页 Table 2
- 展示：TIP-Eval 在 subject 数量、prompt 数量，以及 prompt 来源（真实用户 vs 生成/抽帧）上对比现有 I2V benchmark。
- 解读：把 prompt 来源换成“真实用户”会显著改变评测结论（见下方图 5 雷达图的观察）。
- 与贡献关系：对应贡献 4（TIP-Eval）。

### 图 4：用户到底想让什么动起来？（Figure 4，PDF 第 6 页）
![图：Top-25 subject 与 direction 的长尾偏好](../figures/2025-tip-i2v-a-million-scale-real-text-and-image-prompt-dataset-for-image-to-video-ge/fig4_subject_direction.png)
- 来自：PDF 第 6 页 Figure 4
- 展示：最常见 subject 与 direction 的频次分布（明显头部集中）。
- 解读：
  - subject 侧：人相关类（person / portrait 等）在头部占比极高；
  - direction 侧：除了泛化的 move，用户偏好更具体的运动（zoom / walk / blink 等）。
- 与贡献关系：对应贡献 4（TIP-Eval 的 subject 构成依据）与“面向用户偏好的数据/训练策略”启发。

### 图 5：用真实用户 prompts 做评测，会暴露什么问题？（Figure 6，PDF 第 6 页）
![图：TIP-Eval 10 维雷达对比（多模型各维表现不一）](../figures/2025-tip-i2v-a-million-scale-real-text-and-image-prompt-dataset-for-image-to-video-ge/fig6_tip_eval_radar.png)
- 来自：PDF 第 6 页 Figure 6
- 展示：基于 10k prompts 的 TIP-Eval，比较五个 I2V 模型在 10 个维度的归一化表现（一致性、动态、对齐等）。
- 解读（论文给出的关键观察）：
  - 没有任何模型在所有维度都最强：不同维度之间存在明显 trade-off；
  - video-text alignment 整体偏弱（论文提到最高分也只有 0.26 量级），说明“听懂用户动作指令”仍是短板；
  - 早期商业系统在“用户视角评测”上可能优于最新开源系统，提示基准与真实体验间的差异。
- 与贡献关系：对应贡献 4（TIP-Eval）并为模型诊断提供依据。

### 图 6：I2V 的安全风险示例（Figure 7，PDF 第 7 页）
![图：从“友好合影”到“打斗视频”的误用示例（连续帧）](../figures/2025-tip-i2v-a-million-scale-real-text-and-image-prompt-dataset-for-image-to-video-ge/fig7_misinformation_frames.png)
- 来自：PDF 第 7 页 Figure 7
- 展示：给定一张“看似正常”的输入图像，简单的动作指令即可生成具有误导性的事件视频（论文用公众人物场景举例）。
- 解读：I2V 的“输入图像真实性”会强烈影响观众判断；即便视频内容是生成的，首帧的“真实感”也会放大误导风险。
- 与贡献关系：对应贡献 1（包含安全相关字段）与贡献 4（为安全研究提供真实用户 prompts 与多模型输出）。

## 实验与评价
- 数据规模与覆盖：
  - 170 万级唯一（text, image）prompt，覆盖时间跨度约 Jul 2023–Oct 2024；
  - 附带 subject/direction、embedding、NSFW 等结构化字段，便于研究。
- TIP-Eval 的设计：
  - 用 1,000 个热门 subject，每个 subject 配 10 条真实 prompts，形成 10,000 prompts 的评测集；
  - 用多维度指标对五个 I2V 模型进行对比（论文可视化为雷达图）。
- 评测结论（从论文文字描述提炼）：
  - 没有模型在所有维度占优；
  - 对“动作指令”的对齐（video-text alignment）仍很弱；
  - 用真实用户 prompts 的结论可能与“专家 prompt”基准不一致，强调了评测的现实性。

## 局限性
- 图像 prompt 在原始系统不可直接获得时，需从视频解析首帧；这会引入与真实输入图像的潜在偏差（尽管论文称检查后质量较高）。
- subject/direction 等字段依赖自动推断，存在语义重叠与噪声（例如 person/people/man 的重叠），可能影响后续分层统计/采样。
- “多模型生成视频”目前只覆盖部分 prompts（受算力限制），后续扩展依赖社区/研究者持续生成。

## 与库中相关论文的关系
- 与“训练数据集”互补：TIP-I2V 是 **prompt-gallery（真实用户交互）**，而不是 caption-(real)-video 的训练对。
  - [[2026-05-15_2025-openvid-1m-a-large-scale-high-quality-dataset-for-text-to-video-generation|OpenVid-1M（T2V 训练数据集）笔记]]
- 与“评测基准”互补：TIP-Eval 更贴近“真实用户 prompt + 首帧”，可作为现有 video generation benchmark 的补充视角。
  - [[2026-05-14_2025-t2v-compbench-a-comprehensive-benchmark-for-compositional-text-to-video-generati|T2V-CompBench 笔记]]
  - [[papers/2024_VBench Comprehensive Benchmark Suite for Video Generative Models_CVPR.pdf]]
- 与 I2V 系统/模型阅读互补：TIP-I2V 提供真实 prompts，可用于诊断 I2V 系统“到底卡在什么 subject/direction”。
  - [[2026-05-13_2025-livephoto-real-image-animation-with-text-guided-motion-control|LivePhoto（I2V 可控运动）笔记]]

## 后续阅读建议
- 如果你要“用它做评测/诊断”：
  1. 先把 TIP-Eval 的 subject/direction 采样逻辑吃透（这决定了你评测的覆盖面与偏差）。
  2. 再对照 VBench 的维度与实现，理解各维度指标的定义与局限。
- 如果你要“用它做训练/对齐”：
  1. 先做 prompt 分析（按 subject/direction 聚类，找头部需求与长尾失败模式）。
  2. 再做 targeted finetune：针对 TIP-Eval 暴露的弱项 subject/direction 做数据补齐与微调。
- 如果你要“用它做安全研究”：
  1. 把“首帧真实感 + 动作指令”作为误用放大器来建模；
  2. 结合 NSFW/有害性标注与检测器泛化实验，设计更贴近 I2V 的检测/溯源基线。




























