---
type: paper-note
aliases:
  - "MEVG Multi-event Video Generation with Text-to-Video Models"
paper_id: "2024-mevg-multi-event-video-generation-with-text-to-video-models"
title: "MEVG Multi-event Video Generation with Text-to-Video Models"
year: 2024
venue: "ECCV"
subfield: "Video Generation"
topics:
  - "text-to-video"
importance: 5
read_status: "unread"
note_status: "generated"
selected_on: "2026-05-24"
last_reviewed_on: ""
paper: "[[literature/papers/2024-mevg-multi-event-video-generation-with-text-to-video-models]]"
pdf: "[[papers/2024_MEVG_Multi-event Video Generation with Text-to-Video Models_ECCV.pdf]]"
tags:
  - paper
  - paper/video_generation
  - topic/text_to_video
  - venue/eccv
  - year/2024
---
# MEVG：用预训练 T2V 做“多事件连续视频”生成（训练/微调全免）

## 论文信息
- 标题：MEVG: Multi-event Video Generation with Text-to-Video Models
- 作者：Gyeongrok Oh, Jaehwan Jeong, Sieun Kim, Wonmin Byeon, Jinkyu Kim, Sungwoong Kim, Sangpil Kim
- 单位：Korea University；NVIDIA
- 发表：ECCV 2024（arXiv:2312.04086v2，2024-07-16）
- 论文页：[[literature/papers/2024-mevg-multi-event-video-generation-with-text-to-video-models]]
- PDF：[[papers/2024_MEVG_Multi-event Video Generation with Text-to-Video Models_ECCV.pdf]]
- 项目页（文中给出）：https://kuai-lab.github.io/eccv2024mevg

## 细分领域
视频生成（Video Generation）里的 **multi-event / multi-prompt 叙事视频生成**：给定一段故事（或多句事件描述），希望模型输出“事件有衔接、身份与背景连贯、同时又有足够运动与变化”的长视频。

## 重要程度
5/5：它把“多段 prompt 的连续视频”拆成 **训练/微调之外** 仍可落地的三件事：跨段衔接（LFAI）、段内连续（SGS）、以及把故事拆成可用的 prompt（LLM prompt generator）；并用系统性的消融解释每个组件的作用与权衡。

## 一句话总结
MEVG 在不训练/不微调基础 T2V 的前提下，用 **“上一段最后一帧驱动的反演 + 带时间调度的动态噪声 + 结构引导采样”**，把多个事件段落拼成一个在视觉上更连贯、运动更自然的多事件视频；同时用 LLM 把复杂故事自动拆成“一句一事件”的可用 prompt 序列。

## 背景问题（为什么多事件难）
多事件视频生成至少同时满足三点：
1) **段与段之间过渡要平滑**：下一段开头不能“换人换景”。  
2) **语义要对齐**：每段 prompt 的动作/场景变化要被体现。  
3) **运动与多样性要够**：否则容易出现“重复动作 / 机械循环 / 镜头不动”。  

现有多 prompt 方法常见两类问题：  
- 直接把多段 prompt 在采样中“重叠共去噪”（如 Gen-L-Video 类思路）容易引入不该出现的内容并破坏一致性。  
- 纯粹把上一段 latent 直接拿来续（偏保守）又容易造成“动作/镜头模式重复”，缺少变化。

## 核心贡献（按模块拆解）
1) **训练/微调全免的 multi-event T2V 框架**：直接利用已发布的预训练 T2V（文中用的是 LVDM）生成多段事件视频。  
2) **Last Frame-aware Latent Initialization（LFAI）**：用“上一段最后一帧”构造下一段的反演初始化，并配合动态噪声，在“继承结构”与“允许变化”间做可控权衡。  
3) **Structure-guided Sampling（SGS）**：在采样过程中对“去噪观测（denoised observation）”做迭代更新，增强段内帧与帧的结构一致性，减少纹理/布局漂移。  
4) **Prompt Generator（LLM）**：把长句故事转成多条“一句一事件”的 optimal prompts，便于直接喂给 T2V。

## 方法详解

### 0) 整体流程（多段视频如何串起来）
MEVG 把多事件视频看作由多个短 clip 组成：`prompt #1 -> clip #1 -> prompt #2 -> clip #2 -> ...`。关键是：**clip 之间要“接得上”，clip 内要“稳且动”。**

实现上分两段：
- **初始化阶段（LFAI）**：用上一段最后一帧（复制成一段视频）作为“视觉锚点”，做 DDIM inversion 得到下一段的初始 latent，同时注入按帧调度的动态噪声以增加变化。  
- **采样阶段（SGS）**：在反向扩散采样时，逐帧地对 latent 的“去噪观测”施加结构约束，鼓励相邻帧结构一致，从而降低段内漂移。

### 1) LFAI：既要继承上一段，又要允许变化

#### 1.1 Dynamic Noise：越往后越“放开”
直觉：下一段开头应该更像上一段结尾（衔接），而段尾可以更自由地完成新事件。  
做法：对反演时预测的噪声加入一个随帧序号变化的噪声强度调度：
$$
\\kappa_n = \\mathcal{F}(n),\\quad \\mathcal{F}(n)=\\exp(-n),\\quad 0\\le n < N
$$
并把动态噪声混入每帧的噪声预测（符号按原文）：
$$
\\epsilon_t^{inv_p}[n] = \\frac{\\kappa_n}{\\sqrt{1+\\kappa_n^2}}\\epsilon_t^{inv_p}[n] + \\epsilon_t^{dyn}.
$$
含义：用 `\\kappa_n` 控制“这一帧能偏离上一段结构多少”。`\\kappa` 小时更自由（变化更大），`\\kappa` 大时更保守（更像上一段）。

#### 1.2 Last Frame-aware Inversion：用“去噪观测”对齐跨段结构
仅靠动态噪声会造成跨段不连续，因此他们在 inversion 中用 **去噪观测 `\\hat{x}_t`** 做对齐：  
让“上一段采样得到的最后一帧去噪观测”与“下一段 inversion 得到的第一帧去噪观测”尽量一致：
$$
\\mathcal{L}_{\\text{LFAI}} = \\|\\hat{x}_{t}^{sam_{p-1}}[-1]-\\hat{x}_{t}^{inv_p}[0]\\|_2^2.
$$
然后按梯度方向更新 `\\hat{x}_t`（强度由 `\\delta_{LFAI}` 控制）。  
直觉：`\\hat{x}_t` 像是“这一扩散步的粗结构草图”，跨段对齐它，比对齐像素或 raw latent 更稳定，也更贴近“结构一致”的目标。

### 2) SGS：让段内相邻帧“结构上”更连贯
即便跨段接上了，段内采样仍可能产生纹理/布局抖动。SGS 用一个简单的目标鼓励相邻帧结构一致：
$$
\\mathcal{L}_{\\text{SGS}} = \\|\\hat{x}_{t}^{sam_p}[1:n]-\\hat{x}_{t}^{sam_p}[:n-1]\\|_2^2,
$$
并在每个扩散步对 `\\hat{x}_t` 做迭代更新：
$$
\\hat{x}_{t}^{sam_p}\\leftarrow \\hat{x}_{t}^{sam_p}-\\delta_{\\text{SGS}}\\nabla_{\\hat{x}_t}\\mathcal{L}_{\\text{SGS}}.
$$
直觉：把段内“去噪观测”的几何结构拉近，相当于在不改模型参数的前提下，给采样加了一个**结构先验**，减少漂移与闪烁。

### 3) Prompt Generator：把故事拆成“一句一事件”
现实输入常是一个长句/段落，包含多个动作（例如“跑→停→躺下”）。作者用 LLM（文中写明用 ChatGPT）把故事拆成 K 个 prompt，并通过一组约束让每个 prompt **只描述一个主事件（一个 main verb）**，同时保留必要的主体/背景信息，避免额外动作干扰。

你可以把它理解成：把“故事脚本”转换成 **T2V 更擅长的短指令序列**，再配合 MEVG 的跨段对齐机制来串成连续视频。

## 关键公式或算法直觉（把它当成可调的“续写器”）
把 MEVG 看作一个“可控续写器”会更好记：
- **续写的锚点**：上一段的最后一帧（先复制成一段，再反演）。  
- **续写的自由度曲线**：`\\kappa_n`（越靠后越自由/越能变化）。  
- **续写的硬约束**：跨段用 `\\mathcal{L}_{LFAI}` 对齐“结构草图”；段内用 `\\mathcal{L}_{SGS}` 抑制漂移。  
- **权衡旋钮**：`\\delta_{LFAI}` 与 `\\delta_{SGS}`。文中也强调：值越大越稳，但语义对齐可能下降（更“像上一段”而非更“像新 prompt”）。

## 关键原图讲解（精选 5 张）
> 说明：论文中部分 Figure 是矢量/组合对象，`extract-images` 会只抽到零散小图或抽不全（本次就出现了 page 14 的空白提取）。下面选用可用的原图/渲染图，并在需要时用文字补足。

### 图 1：多事件视频示例（Figure 1）
![图：多事件连续示例](../figures/2024-mevg-multi-event-video-generation-with-text-to-video-models/page_002_01_img-002-011.jpg)
- 来源：PDF 第 2 页（Figure 1 的一帧/示例图之一；自动抽图为局部缩略图）
- 展示内容：同一主体（Santa Claus）在连续场景中的一段片段，体现“主体/背景连贯”的目标
- 如何解读：把它当成“任务定义图”：下一段事件要变化，但角色身份、整体场景语义要接得上
- 与核心贡献关系：解释了为什么需要 LFAI（跨段一致）+ 动态噪声（允许变化）这对组合

### 图 2：MEVG 总体管线（Figure 2）
![图：MEVG overall pipeline](../figures/2024-mevg-multi-event-video-generation-with-text-to-video-models/fig2_pipeline.png)
- 来源：PDF 第 5 页（Figure 2；由 PDF 渲染并裁剪得到）
- 展示内容：`prompt #p-1` 生成 clip 后，用其最后一帧驱动下一段的初始化（LFAI），再用 SGS 保证段内连续
- 如何解读：上半部分是“段与段如何串”，下半部分两个框分别对应“跨段一致（LFAI）”与“段内一致（SGS）”
- 与核心贡献关系：这张图基本覆盖了论文方法的全部骨架，后续公式都是在补齐两个框里的细节

### 图 3：LFAI 细节（Figure 3）
![图：LFAI 细节](../figures/2024-mevg-multi-event-video-generation-with-text-to-video-models/fig3_pipeline.png)
- 来源：PDF 第 6 页（Figure 3；由 PDF 渲染并裁剪得到）
- 展示内容：把“上一段最后一帧的结构”带到下一段：REPEAT → inversion（含动态噪声与 last-frame 约束）→ 得到下一段初始 latent
- 如何解读：红色虚线框是关键：它表示 inversion 的过程中既引入噪声（给变化空间），又用 last-frame 约束拉回结构一致
- 与核心贡献关系：解释了 MEVG 为什么能在“不过度重叠去噪”的情况下，仍然实现跨段衔接

### 图 4：多 prompt 生成的定性对比（Figure 4）
![图：定性对比示例帧](../figures/2024-mevg-multi-event-video-generation-with-text-to-video-models/page_010_01_img-010-110.jpg)
- 来源：PDF 第 10 页（Figure 4 的局部/单格示例帧；自动抽图未能抽出完整网格对比）
- 展示内容：相同故事被拆成多个 prompts（骑车→走路→读书）时，MEVG 相比基线更强调“段间身份与场景稳定”
- 如何解读：单帧看不出“过渡是否顺滑”，更适合结合论文视频/项目页看；这里主要用来提示 Figure 4 的对比设定
- 与核心贡献关系：对应论文的主张：多 prompt 的连续性不是单段质量能解决的，需要跨段机制（LFAI/SGS）

### 图 5：额外开销分析（Figure 2 in Appendix）
![图：额外开销（内存/推理时间）](../figures/2024-mevg-multi-event-video-generation-with-text-to-video-models/page_021_01_img-021-255.png)
- 来源：PDF 第 21 页（Appendix Figure 2）
- 展示内容：相对基础模型 LVDM，MEVG 内存基本不涨（约 ×1.02），推理时间约 ×1.41；而 Gen-L-Video 的内存/时间更高（约 ×3.05 / ×1.81）
- 如何解读：这张图说明 MEVG 虽然做了 inversion/latent 更新等“采样侧技巧”，但资源增量仍可控
- 与核心贡献关系：支撑“training-free 可落地”：不靠训练堆算力，而是把代价放在推理侧可控的优化上

## 实验与评价（怎么读结论）
### 设置
- 基础模型：LVDM（预训练 T2V，不做微调）
- 每段 clip：16 帧，分辨率 256×256
- LLM：ChatGPT 用于把复杂场景拆成多 prompt
- 超参：`\\delta_{LFAI}=1000`、`\\delta_{SGS}=7`（文中默认）
- 资源：单张 RTX 3090

### 指标与结果要点
- 自动指标：  
  - **CLIP-Text**：prompt 与生成结果语义对齐  
  - **CLIP-Image**：相邻帧相似度（用作时序一致性 proxy）  
  结论：MEVG 在综合上优于对比方法；T2V-Zero 更偏语义但时序差，DirecT2V 时序更好但语义下降；与同基础模型的方法相比，MEVG 更平衡“语义变化”与“结构稳定”。  
- 人评（AMT 100 人）：Temporal / Semantic / Realism / Preference 四项均最好；作者强调“身份与背景的稳定性”对人类偏好非常关键。

### 消融（为什么每个组件都需要）
- 仅 DDIM inversion：能接上大结构，但细节纹理/背景变化明显。  
- 加 DN：变化更自然，但跨段容易断开。  
- 加 LFAI：跨段结构接上了，但段内仍有轻微抖动。  
- 再加 SGS：段内更稳、观感更真实。  

## 局限性（读完你应该警惕什么）
1) **依赖基础模型能力**：方法不训练，质量上限主要由预训练 T2V 决定。  
2) **权衡难免**：`\\delta_{LFAI}` / `\\delta_{SGS}` 过大可能牺牲语义变化（更“像上一段”）。  
3) **评测仍不完美**：CLIP-Image 只是时序一致性的 proxy；真正的“过渡自然程度”更依赖人评与视频级检查。  
4) **LLM prompt generator 的不确定性**：拆 prompt 的质量会直接影响最终视频的事件表达；不同 LLM/提示词会导致风格差异。

## 与库中相关论文的关系
- 多 prompt / 长视频的“重叠去噪”路线：[[literature/papers/2023-gen-l-video-multi-text-to-long-video-generation-via-temporal-co-denoising|Gen-L-Video]]（MEVG 的对照基线之一）。  
- 同为“推理侧、无需训练”的长视频/一致性技巧：[[notes/2026-05-23_2024-freenoise-tuning-free-longer-video-diffusion-via-noise-rescheduling|FreeNoise]]、[[notes/2026-05-22_2024-fifo-diffusion-generating-infinite-videos-from-text-without-training|FIFO-Diffusion]]（更聚焦长视频推理与延长；MEVG 更聚焦多事件跨段过渡）。  
- 相关纸面关联（见该论文页的 Related Papers）：[[literature/papers/2025-mint-mind-the-time-temporally-controlled-multi-event-video-generation|MinT]]、[[literature/papers/2025-instancecap-improving-text-to-video-generation-via-instance-aware-structured-cap|InstanceCap]]（分别对应“多事件控制”与“结构化文本条件”的方向）。

## 后续阅读建议（带着问题读）
1) 想复用到你的 pipeline：优先读 Sec. 3.3/3.4，把 `\\kappa_n` 与 `\\delta` 两个“旋钮”当成可调参数去理解。  
2) 关心“更长、更稳”：对照 [[notes/2026-05-23_2024-freenoise-tuning-free-longer-video-diffusion-via-noise-rescheduling|FreeNoise]] 看它们如何在推理侧延长时序一致性。  
3) 关心“多事件文本控制”：接着读 [[literature/papers/2025-mint-mind-the-time-temporally-controlled-multi-event-video-generation|MinT]]，看它如何显式建模事件时间/节奏，而不仅仅是把 prompt 串起来。













