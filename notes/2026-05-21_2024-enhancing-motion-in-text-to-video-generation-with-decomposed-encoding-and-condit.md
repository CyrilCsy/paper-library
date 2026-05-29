---
type: paper-note
aliases:
  - "Enhancing Motion in Text-to-Video Generation with Decomposed Encoding and Conditioning"
paper_id: "2024-enhancing-motion-in-text-to-video-generation-with-decomposed-encoding-and-condit"
title: "Enhancing Motion in Text-to-Video Generation with Decomposed Encoding and Conditioning"
year: 2024
venue: "NEURIPS"
subfield: "Video Generation"
topics:
  - "text-to-video"
importance: 5
read_status: "unread"
note_status: "generated"
selected_on: "2026-05-21"
last_reviewed_on: ""
paper: "[[literature/papers/2024-enhancing-motion-in-text-to-video-generation-with-decomposed-encoding-and-condit]]"
pdf: "[[papers/2024_Enhancing Motion in Text-to-Video Generation with Decomposed Encoding and Conditioning_NeurIPS.pdf]]"
tags:
  - paper
  - paper/video_generation
  - topic/text_to_video
  - venue/neurips
  - year/2024
---
# 论文信息
- 标题：Enhancing Motion in Text-to-Video Generation with Decomposed Encoding and Conditioning
- 会议：NeurIPS 2024
- 关键词：text-to-video
- PDF：[[papers/2024_Enhancing Motion in Text-to-Video Generation with Decomposed Encoding and Conditioning_NeurIPS.pdf]]

# 细分领域 / 重要程度
- 细分领域：Video Generation
- 重要程度：5/5（动机清晰：直指 T2V “动作不够 / 过于静态”的核心痛点，并给出可迁移的监督信号设计）

# 一句话总结
把“文本里对动作的理解”和“扩散模型里对动作的注入”都拆成 **内容(content)** 与 **运动(motion)** 两条路，并用两个简单但关键的监督（`L_text-motion`、`L_video-motion`）把“动作信息”真正拉进模型。

# 背景问题（这篇在解决什么）
很多 T2V 直接复用 T2I 的做法：文本编码偏向名词/物体（静态语义强、动词/动作弱），再用逐帧的空间 cross-attention 做条件注入，结果常见现象是：
- 画面好看但“像动图”：有主体、无明显动作；
- 动作词写得很具体（run / fall / walk left）但生成只体现了轻微的镜头运动或局部抖动。

作者认为根因主要有两类偏置：
1) **文本编码偏置**：CLIP 类 VLM 更擅长静态语义，动作词在 embedding 里“存在但不敏感”；  
2) **条件注入偏置**：空间维度逐帧注入对视频足够，但对跨时间的动作一致性不足。

# 核心贡献（我认为最关键的 3 点）
1) **Decomposed Text Encoding**：保留原 text encoder 作为内容编码器 `E_c`，新增运动编码器 `E_m` 专门学动作语义。  
2) **Decomposed Conditioning**：保留原有内容条件注入（空间），新增运动条件注入模块（时间）让“动作 embedding”沿时间轴起作用。  
3) **两种 motion supervision 设计**：  
   - `L_text-motion`：让 cross-attention 的时间变化去“模仿真实视频的运动模式”（用光流作为 motion 表征）。  
   - `L_video-motion`：让预测的去噪视频 latent 的“帧间差分”接近真实视频 latent 的帧间差分（直接在 latent 空间约束运动）。

# 方法详解（DEMO：Decomposed Motion）
作者把问题拆成两层：**文本侧**要学会“动作词怎么影响视频”，**生成侧**要学会“动作信息怎么沿时间传播”。

## 1) Decomposed Text Encoding：`E_c` + `E_m`
- `E_c`（Content Encoder）：保留原始 CLIP text encoder（更擅长物体/场景/属性），主要用于静态内容条件。  
- `E_m`（Motion Encoder）：从 CLIP text encoder 初始化，但用专门的监督把注意力重心拉向 motion token/全句动作语义。

### `L_text-motion`（文本-运动监督）：让 cross-attn “像光流一样随时间变化”
关键观察：在扩散 UNet 的 cross-attention 里，**[eot]** token 的 attention map 聚合了全句语义，并对“整体动作”很关键。  
做法：对每层 cross-attn，取与 `[eot]` 相关的 attention map，经过一个 motion 提取函数 `ϕ(·)`（论文用光流），与真实视频 `x0` 的 motion 表征做 cosine 相似度约束：
- 直觉：如果真实视频里“向左走”，那 attention map 在时间维度也应该体现出一致的位移/变化模式。

### `L_reg`（正则）：避免 catastrophic forgetting
直接用新目标 fine-tune CLIP text encoder 很容易“忘掉”对内容语义的泛化。作者用一个轻量正则约束 `E_m(p)` 不要偏离对应视频中间帧的 CLIP image embedding（保持 text-image 对齐）。

## 2) Decomposed Conditioning：内容走空间，运动走时间
- Content conditioning：沿用原 T2V 模型的空间 cross-attn，把 `E_c(p)` 注入到每帧的空间结构里。  
- Motion conditioning：新增一个 temporal transformer（时间维度的 cross-attn / self-attn），把 `E_m(p)` 作为“动作条件”沿时间传播。

### `L_video-motion`（视频-运动监督）：在 latent 空间直接约束帧间差分
作者不直接用“高层 motion embedding”去约束（可能与像素/latent 去噪目标冲突），而是定义：
- `Φ(z) = z_{2:F} - z_{1:F-1}`（连续帧差分，仍在 latent 空间）
然后让预测的 `\hat z_{0,t}` 与真实 `z0` 在 `Φ(·)` 上接近（L2），鼓励“运动幅度/方向/一致性”更贴近真实视频。

## 3) Joint Training：把上述 loss 叠加
最终目标（省略系数细节）：
`L = L_diffusion + α L_text-motion + β L_reg + γ L_video-motion`

# 关键公式 / 算法直觉（怎么把 motion 拉进来）
- `L_text-motion` 的本质：**让“文本→视频结构”的投影（cross-attn map）在时间维度学到 motion 规律**；相比“直接监督文本 embedding”，它更接近 UNet 实际使用文本的方式。  
- `L_video-motion` 的本质：**用与去噪目标同空间的（latent）运动代理**，避免“表示空间不一致”导致训练互相拉扯。  
- `E_c / E_m` 拆分的本质：既不丢掉 CLIP 的内容语义优势（冻结/保留 `E_c`），又能用专门损失把 `E_m` 调成“对动作敏感”的编码器。

# 关键原图讲解（3–6 张）
> 说明：本论文大量图表为矢量/组合对象，`extract-images` 在 PDF 第 4 页的 Figure 2 上**自动提取不完整**（得到接近全黑的碎片）。这里改用整页渲染（`pdftocairo`）后再裁剪，保证图的可读性。

## 图 1：DEMO 训练总览（Figure 2）
![图：DEMO 训练总览](../figures/2024-enhancing-motion-in-text-to-video-generation-with-decomposed-encoding-and-condit/fig2_demo_training.png)
- 来源：PDF 第 4 页（Figure 2）
- 展示内容：左侧是 **双文本编码器**（Motion Encoder / Content Encoder）与对应的条件注入；右侧是训练时三条关键约束：`L_text-motion`（红）、`L_reg`（绿）、`L_video-motion`（黄）。
- 怎么读：把它当作“这篇论文所有设计的依赖图”。一眼看清：动作信息从文本侧怎么来（`E_m`），在生成侧怎么走时间模块（motion conditioning）。
- 与核心贡献关系：这张图基本覆盖了 3 个贡献点（拆编码、拆条件、两类监督）。

## 图 2：CLIP 对不同词性敏感度（Figure 1，可视化的一部分）
![图：不同词性敏感度对比](../figures/2024-enhancing-motion-in-text-to-video-generation-with-decomposed-encoding-and-condit/page_003_01_img-003-000.png)
- 来源：PDF 第 3 页（Figure 1 的柱状图子图）
- 展示内容：不同 POS（名词/动词/副词等）下，CLIP text encoder 的“敏感度”对比，以及不同 fine-tune 组合（`L_text-motion`、`L_reg`、两者一起）对敏感度的影响。
- 怎么读：重点看趋势而非绝对值：原始/单独 `L_reg` 对 motion 提升有限；只用 `L_text-motion` 会灾难性遗忘；两者结合能在不丢内容的前提下提升 motion token 敏感度。
- 与核心贡献关系：这是“为什么要加 `L_reg`”和“为什么 motion encoder 需要专门监督”的直接证据。

## 图 3：定性对比（Figure 3 的一个例子）
![图：定性对比样例](../figures/2024-enhancing-motion-in-text-to-video-generation-with-decomposed-encoding-and-condit/page_006_01_img-006-037.png)
- 来源：PDF 第 6 页（Figure 3）
- 展示内容：与 LaVie / VideoCrafter2 / ModelScopeT2V 等对比的 16 帧可视化（这里截取其中一个 prompt 的样例网格）。
- 怎么读：观察“相邻帧变化是否有方向性/一致性”，以及是否出现“动作相关的细节变化”（例如姿态、肢体、相对位置）而不是单纯的噪声或镜头抖动。
- 与核心贡献关系：验证 decomposed conditioning + `L_video-motion` 是否真的把动作带进视频，而不仅仅是提升单帧画质。

## 图 4：局限性（Figure 4）
![图：顺序动作失败示例](../figures/2024-enhancing-motion-in-text-to-video-generation-with-decomposed-encoding-and-condit/page_010_01_img-010-040.png)
- 来源：PDF 第 10 页（Figure 4）
- 展示内容：文本包含两个**顺序发生**的动作/镜头描述，但模型倾向于把它们“同时出现/混在一起”。
- 怎么读：把它当作“动作时序理解”失败的反例：模型增强了 motion dynamics，但对“先后顺序/分段叙事”的建模仍弱。
- 与核心贡献关系：明确 DEMO 的改进点主要在“动作强度与一致性”，而不是“复杂时序叙事”。

# 实验与评价（抓重点）
作者在多个基准上报告了 **画质 + 动作** 两类指标，整体结论是：**动作动态性显著提升**，同时画质不退步或小幅提升；但“更强动作”会与某些稳定性指标存在权衡。

- MSR-VTT（zero-shot）：FID 11.77 vs 14.89（更好），FVD 422 vs 557（更好）。  
- WebVid-10M（Val）：FID 9.86 vs 11.14，FVD 351 vs 508，CLIPSIM 0.3083（也更好）。  
- VBench（motion 相关）：Motion Dynamics 62.50 → 68.90（显著提升）；Temporal Flickering / Motion Smoothness 略降（更大动作带来稳定性冲突）。

# 局限性
- **顺序动作/分段叙事弱**：文本里要求“先 A 后 B”的动作，容易被模型混合成同时发生（见图 4）。  
- **指标权衡**：提高 motion dynamics 可能会牺牲 flickering / smoothness（作者在 VBench 上观察到该现象）。  

# 与库中相关论文的关系（建议的 Obsidian 连接）
- 与“对齐/可控性”相关：[[notes/2026-05-20_2025-vpo-aligning-text-to-video-generation-models-with-prompt-optimization|VPO (2025)]] 更偏推理时 prompt 优化；DEMO 更偏训练侧把动作信息学进模型。  
- 与“视频生成架构/效率”相关：[[notes/2026-05-16_2025-pyramidal-flow-matching-for-efficient-video-generative-modeling|Pyramidal Flow Matching (2025)]] 更关注生成框架与效率；DEMO 更关注 motion 表征与监督。  
- 与“更强先验/物理一致性”相关：[[notes/2026-05-19_2025-videorepa-learning-physics-for-video-generation-through-relational-alignment-wit|VideoRePa (2025)]] 尝试引入物理/关系先验；DEMO 的切入点是文本侧动作语义与时间注入模块。

# 后续阅读建议（按优先级）
1) 复现/对照：把 DEMO 的两个监督拆开做 ablation（只加 `E_m`、只加 `L_video-motion`、只加 motion conditioning）看哪一环对你的基座模型最有效。  
2) 扩展到“顺序动作”：参考作者建议，把每帧或时间段绑定子 prompt（类似时间条件/分段 caption），再训练/蒸馏。  
3) 更强 motion 表征：`L_text-motion` 里 `ϕ(·)` 用光流是个起点；可考虑更鲁棒的时序特征（但要注意与去噪目标的表示空间冲突）。






















