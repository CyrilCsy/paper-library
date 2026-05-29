---
type: paper-note
aliases:
  - "HunyuanVideo A Systematic Framework For Large Video Generative Models"
paper_id: "2025-hunyuanvideo-a-systematic-framework-for-large-video-generative-models"
title: "HunyuanVideo A Systematic Framework For Large Video Generative Models"
year: 2025
venue: ""
subfield: "Video Generation"
topics:
  - "text-to-video"
importance: 5
read_status: "unread"
note_status: "generated"
selected_on: "2026-05-12"
last_reviewed_on: ""
paper: "[[literature/papers/2025-hunyuanvideo-a-systematic-framework-for-large-video-generative-models]]"
pdf: "[[papers/2025_HunyuanVideo A Systematic Framework For Large Video Generative Models.pdf]]"
tags:
  - paper
  - paper/video_generation
  - topic/text_to_video
  - year/2025
---
# HunyuanVideo A Systematic Framework For Large Video Generative Models 精讲笔记

## 论文信息

- 论文页：[[literature/papers/2025-hunyuanvideo-a-systematic-framework-for-large-video-generative-models]]
- PDF：[[papers/2025_HunyuanVideo A Systematic Framework For Large Video Generative Models.pdf]]
- arXiv：2412.03603v6（11 Mar 2025）
- 领域：Video Generation（开放视频基础模型 / 系统报告）
- 重要程度：5 / 5
- 阅读状态：未读（注意：生成精讲笔记 ≠ 已读）

## 细分领域

这是一篇**视频生成（Video Generation）**方向的系统性技术报告：围绕一个大规模开源视频基础模型 **HunyuanVideo**，从数据、训练基础设施、模型结构、训练/推理策略到下游应用（I2V、V2A 等）给出整套方案与经验总结。

## 重要程度

重要性高（5/5）：论文声称训练了 **13B 参数**的开源视频生成模型，并通过系统工程与训练策略把性能拉到与多家闭源系统同档对比，且明确以“缩小开源/闭源差距”为目标（并给出代码开源地址）。

## 一句话总结

HunyuanVideo 把视频生成当作“**压缩到时空 latent 的 [[knowledge/flow-matching|Flow-Matching]] DiT**”来做：用因果 3D-VAE 做表征压缩，用结构化字幕与强数据过滤保证训练信号质量，再用分阶段训练与架构细节（多分辨率/多时长、3D RoPE、双流→单流融合等）把大模型训练推到可用、可复现、可开源的水平。

## 背景问题

- 视频生成的 SOTA 多为闭源系统，导致开源社区很难基于强基础模型做算法创新与应用落地。
- 直接“堆数据/堆算力/堆参数”训练一个简单的 Flow-Matching Transformer 并不高效：作者指出需要更有效的 scaling 与训练策略，才能在可承受的资源下达到目标效果。

## 核心贡献（按系统链路拆解）

1. **数据侧**：分层数据过滤 + 去重/重采样 + 结构化字幕（Structured Captioning），提升训练数据的美学质量、运动性、概念覆盖与文本对齐信号。
2. **模型侧**：在**因果 3D-VAE 压缩 latent**上训练视频生成模型；文本条件由大语言模型编码；在 Transformer 中处理多分辨率、多比例、可变时长，并引入**3D RoPE**等关键细节。
3. **训练侧**：以 [[knowledge/flow-matching|Flow Matching]] 为核心训练框架，配合多阶段训练（先图像预训练，再图像/视频联合训练，再面向下游做渐进式微调）。
4. **工程侧**：强调训练/推理基础设施与加速，使得 13B 级视频模型在实际资源约束下可训练。
5. **评测侧**：给出面向真实用户偏好的专业人评对比，展示整体满意度与运动表现上的优势。

## 方法详解

### 1) 数据处理与标注：先把“训练信号”做好

**(a) 去重与概念均衡**

- 以内部 VideoCLIP embedding 做相似片段去重（余弦距离），并用 k-means 得到约 10K 概念中心用于重采样与均衡。

**(b) 分层过滤（hierarchical filtering）**

论文给出一套多视角过滤组合（对应 Figure 4，但本次原图未被自动完整提取），包括：

- Dover：从审美与技术维度评估视觉美学；
- 清晰度/模糊检测：剔除明显模糊片段；
- 运动速度：用 optical flow 估计运动，过滤静态或低运动视频；
- 场景切分：结合 PySceneDetect 与 Transnet v2 获取 scene boundary；
- OCR：剔除过多文本，并定位/裁剪字幕；
- 水印/边框/Logo：YOLOX-like 检测并清理遮挡或敏感信息。

**(c) 结构化字幕（Structured Captioning）**

作者训练自研 VLM 生成 JSON 格式结构化描述，包含：

- Short Description（主内容）
- Dense Description（含镜头/转场/相机运动等）
- Background（环境）
- Style（纪录片/电影感/写实/科幻等）
- Shot Type（航拍/近景/中景/远景等）
- Lighting（光照）
- Atmosphere（氛围）
- 以及 metadata 衍生的 source/quality tags 等

并通过 dropout + 重排/组合，拼装出不同长度/模式的 caption，以增强泛化并减少过拟合。

### 2) 表征与条件：在“时空压缩 latent”上建模

整体架构（论文 Figure 5）要点如下（本次自动提取未得到对应结构图，以下根据正文描述补足）：

- 训练空间：先用 **Causal 3D-VAE** 把图像/视频压缩到时空 latent；
- 文本条件：prompt 由**大语言模型**编码作为条件；
- 生成器输入：高斯噪声 + 条件；
- 输出：预测 latent，再由 3D-VAE decoder 解码回图像/视频。

此外，模型采用“**双流（视觉 latent / 文本 token）→ 单流融合**”的 Transformer 设计：双流阶段分别调制各自的 token 表示，单流阶段再拼接做全注意力融合，以捕获视觉-语义交互。

### 3) [[knowledge/flow-matching|Flow Matching]]：目标函数与采样直觉

训练时，给定数据 latent 记为 $\mathbf{x}_1$，采样 $t\in[0,1]$（论文中提到用 logit-normal），并从高斯采样噪声 $\mathbf{x}_0\sim\mathcal{N}(0,\mathbf{I})$，用线性插值得到中间状态 $\mathbf{x}_t$。模型学习预测速度场（velocity）把 $\mathbf{x}_t$ 推向 $\mathbf{x}_1$：

$$
\mathcal{L}_{\text{generation}}
=\mathbb{E}_{t,\mathbf{x}_0,\mathbf{x}_1}\left\|\mathbf{v}_t-\mathbf{u}_t\right\|^2,
\quad \mathbf{u}_t = \frac{d\mathbf{x}_t}{dt}.
$$

推理时，从噪声 $\mathbf{x}_0$ 出发，把模型给出的 $d\mathbf{x}_t/dt$ 当作 ODE 的右端，用**一阶 Euler ODE solver**积分到 $\mathbf{x}_1$ 得到生成样本（这也是 Flow/Rectified-Flow 系列常见的“从噪声沿速度场走到数据分布”的直觉）。

### 4) 位置编码：把 RoPE 扩展到时空 3D

为支持多分辨率/多比例/可变时长，论文使用 RoPE，并把它扩展到三维：分别对时间 (T)、高度 (H)、宽度 (W) 计算旋转频率矩阵，把 Q/K 的通道分成三段（$d_t,d_h,d_w$）分别旋转后再拼接，从而同时编码时空位置关系。

### 5) 训练配方：分阶段与渐进式策略

论文强调分阶段训练能显著改善收敛与效果：

- 先做图像预训练（含 256px 等低分辨率阶段，帮助多比例与文本对齐）
- 再做图像 + 视频联合训练（覆盖 256px 到 960px 的多尺度）
- 对下游任务（例如 I2V、肖像 I2V）采用渐进式微调（逐步解冻层等）以兼顾域内质量与泛化

## 关键原图讲解（精选 4 张）

> 说明：本次 `extract-images` 能提取到部分图表/结构图，但也出现“矢量/组合对象导致提取不完整”的情况。例如 Figure 2 左侧（开源/闭源算力对比）在自动提取中为空白图；Figure 4/5 等结构示意图未在提取结果中出现。以下选取能支撑核心论点的关键图片，并用文字补足缺失部分。

### 图 1：人评总体得分对比（HunyuanVideo 排名第一）

![图：Overall Score 对比（Figure 2 的右侧）](../figures/2025-hunyuanvideo-a-systematic-framework-for-large-video-generative-models/page_002_02_img-002-004.png)

- 来自 PDF 第 2 页。
- 展示内容：以“Overall Score(%)”汇总对比多个强基线（含 Gen-3 alpha、Luma 1.6、若干中国商用系统），HunyuanVideo（Ours）得分最高。
- 解读要点：论文不只强调“开源最大”，而是用人评把目标对齐到“真实观感/满意度”，并突出 motion dynamics 等维度优势。
- 与核心贡献关系：支撑“开源模型与闭源系统同档甚至更强”的主叙述，是论文选题动机与成果展示的关键证据。

### 图 2：Video-to-Audio（V2A）模型结构：三流→单流的 DiT 融合

![图：V2A Diffusion Backbone（Figure 18 相关）](../figures/2025-hunyuanvideo-a-systematic-framework-for-large-video-generative-models/page_019_01_img-019-149.png)

- 来自 PDF 第 19 页。
- 展示内容：把视觉（CLIP）、文本（T5-XXL）与音频（mel-spectrogram 的 2D VAE latent）分别编码到 token；先用 Triple-stream DiT block 分别处理，再进入 Single-stream block 做跨模态融合，最后输出并经 VAE/HifiGAN 还原音频。
- 解读要点：这张图把论文里“先分开调制、再融合对齐”的思路具象化；三流阶段让每种模态先在自己的表征空间里“站稳”，再单流对齐到共同语义空间，减少直接早融合带来的干扰。
- 与核心贡献关系：说明 HunyuanVideo 不只做 T2V，还提供多模态扩展（V2A）的系统方案，体现其“基础模型 + 应用”框架化设计。

### 图 3：Image-to-Video（I2V）扩展：Token Replace + 语义图像注入

![图：HunyuanVideo Image-to-Video Backbone（Figure 19）](../figures/2025-hunyuanvideo-a-systematic-framework-for-large-video-generative-models/page_020_02_img-020-150.png)

- 来自 PDF 第 20 页。
- 展示内容：I2V 在 T2V 基础上加入 token replace（将参考图像 latent 作为首帧 latent，timestep 设为 0），并引入语义图像注入（用 MLLM 得到语义 token，与视频 latent 拼接做 full-attention）。
- 解读要点：token replace 强化“首帧一致性”，语义注入提升模型对参考图像语义的理解与文本条件融合能力；两者都指向 I2V 常见痛点：保持身份/外观一致与动作合理过渡。
- 与核心贡献关系：展示“同一套 backbone 通过条件组织即可扩展任务”的可扩展性，与论文强调的系统性/框架性一致。

### 图 4：肖像 I2V 样例：外观一致性 + 动作连贯

![图：Portrait I2V 样例（Figure 21）](../figures/2025-hunyuanvideo-a-systematic-framework-for-large-video-generative-models/page_021_02_img-021-153.png)

- 来自 PDF 第 21 页。
- 展示内容：给定参考图（Ref. Img.）与文本提示（Prompt），生成多帧视频序列（Gen Video）。
- 解读要点：例子强调“主体外观保持 + 动作/交互连贯”（如手持烟花、喝水、双人靠近/远离等），对应论文中“针对肖像域的监督微调 + 渐进式解冻”等训练策略。
- 与核心贡献关系：把“下游应用可落地”的效果直观呈现，补强论文主线：不仅训练出大模型，还能用工程化流程把能力迁移到具体应用。

## 实验与评价（抓重点）

- 论文强调对比对象包括 Gen-3、Luma 1.6 与若干高水平商用系统；并采用约 1,500 条提示词、60 人参与的人评流程，HunyuanVideo 在总体满意度上领先，尤其在运动表现上更突出。
- 由于本次图表自动提取不包含完整评测细表（多为矢量/组合对象或未被抽取），建议结合原 PDF 的评测章节对：人评维度定义、提示词覆盖范围、以及是否有对齐推理设置（分辨率/时长/采样步数）进行复核。

## 局限性（基于正文可见信息的推断）

1. **系统链路很长**：高质量数据过滤、结构化字幕、VAE、训练配方与基础设施一起“打包”才成立，复现门槛较高。
2. **依赖强数据与标注**：自研 VLM 结构化字幕与多种内部模型（VideoCLIP、OCR、检测器等）是关键组成，开源复刻可能受限于数据/标注/算力。
3. **可解释性与可控性仍是挑战**：模型强调视觉质量、运动与对齐，但对长视频一致性、精细可控镜头语言、复杂因果一致性等仍需更多验证与工具链支持。

## 与库中相关论文的关系（建议对照阅读）

- 同题材开源系统：[[literature/papers/2025-open-sora-2-0-training-a-commercial-level-video-generation-model-in-200k|Open-Sora 2.0]]（更“开源训练成本/工程”视角）
- T2V 的大规模预训练传统线：[[literature/papers/2023-cogvideo-large-scale-pretraining-for-text-to-video-generation-via-transformers|CogVideo]] → [[literature/papers/2025-cogvideox-text-to-video-diffusion-models-with-an-expert-transformer|CogVideoX]]
- [[knowledge/flow-matching|Flow Matching]] / Flow 系路线索：[[literature/papers/2025-pyramidal-flow-matching-for-efficient-video-generative-modeling|Pyramidal Flow Matching]]（更强调高效视频生成的 flow 设计）
- 不同范式对照（AR）：[[literature/papers/2025-autoregressive-video-generation-without-vector-quantization|Autoregressive Video Generation without Vector Quantization]]（把生成顺序与 token 化方式走到另一条路径）

## 后续阅读建议

1. 先把论文的“数据链路”吃透：Figure 4/结构化字幕的 JSON 字段与组合策略，决定了训练信号质量与对齐能力的上限。
2. 对照 [[knowledge/flow-matching|Flow Matching]] 目标与采样：理解 $\mathbf{x}_t$ 的构造、速度场学习与 Euler 采样的取舍（步数、稳定性、速度场质量）。
3. 读 I2V/V2A 扩展章节：看同一 backbone 在“条件组织/输入拼接”上如何变化，能为你在库里做统一化笔记与任务迁移提供模板。






