---
type: paper-note
aliases:
  - "LivePhoto Real Image Animation with Text-guided Motion Control"
paper_id: "2025-livephoto-real-image-animation-with-text-guided-motion-control"
title: "LivePhoto Real Image Animation with Text-guided Motion Control"
year: 2025
venue: "ECCV"
subfield: "Video Generation"
topics:
  - "image-to-video"
importance: 5
read_status: "unread"
note_status: "generated"
selected_on: "2026-05-13"
last_reviewed_on: ""
paper: "[[literature/papers/2025-livephoto-real-image-animation-with-text-guided-motion-control]]"
pdf: "[[papers/2025_LivePhoto Real Image Animation with Text-guided Motion Control_ECCV.pdf]]"
tags:
  - paper
  - paper/video_generation
  - topic/image_to_video
  - venue/eccv
  - year/2025
---
# LivePhoto：用文本精细控制“真实图片动起来”

## 论文信息
- 标题：LivePhoto: Real Image Animation with Text-guided Motion Control
- 年份：2025（arXiv v1: 2023-12-05）
- 会议：ECCV
- PDF：[[papers/2025_LivePhoto Real Image Animation with Text-guided Motion Control_ECCV.pdf]]
- 代码/项目页：论文中给出（本库未收录链接）

## 细分领域
- Video Generation / Image-to-Video（单张真实图片 → 多帧视频）

## 重要程度
- 5/5（主流扩散基座 + 针对“文本控运动”的可控性设计：强 baseline + 两个关键模块）

## 一句话总结
把 Stable Diffusion v1.5 改造成“图像控内容、文本控运动”的 I2V 系统：在 AnimateDiff 式时序模块之上，用“运动强度”降低文本→运动的歧义，并用“文本重加权”把 prompt 里真正的动作词从内容词里分离出来。

## 背景问题（Why）
- 现有 T2V / I2V 往往出现：
  - 文本更多影响“画面内容/风格”，对“运动幅度、速度、是否真的动起来”控制弱。
  - prompt 同时包含内容+动作描述：当内容词与参考图冲突时，模型会整体压制文本，导致动作词也被抑制。

## 核心贡献（What）
1. **强 I2V baseline**：在 SD v1.5 上加入对参考图的多层内容约束（局部潜变量拼接 + 全局内容编码器 + 先验反演）。
2. **Motion Intensity Guidance（运动强度条件）**：给训练样本估计一个 1~10 的运动强度等级，作为额外条件输入 UNet，减少同一文本对应多种“动法”的歧义，并在推理时可调。
3. **Text Re-weighting（文本重加权）**：对 CLIP 文本 token 学习 0~1 权重，突出动作/运动相关 token，抑制与参考图冲突的内容 token，从而提升“听懂动作词”的能力。

## 方法详解（How）
### 1) 总体框架：把 SD 变成视频扩散
- 表示：视频噪声潜变量 `z` 形状为 `B×F×C×H×W`（论文中 C=4）。
- UNet：冻结 SD v1.5 的大部分模块，在各 stage 插入可训练的 **Motion Module**（结构与 AnimateDiff 对齐）以建模帧间关系。

### 2) 内容保持：参考图三重约束
- **Reference latent 拼接（局部约束）**：用 VAE 编码参考图得到 `r`，与视频噪声潜变量在通道维拼接输入 UNet，直接提供像素级外观先验。
- **Content encoder（全局语义约束）**：用冻结的 DINOv2 提取参考图 patch tokens，经线性层投影后，通过新增 cross-attention 注入 UNet，帮助后续帧在语义层面保持 identity。
- **Prior inversion（先验反演注入）**：推理初始步 T 不用纯高斯，而混入参考图潜变量的反演 `Inv(r0)`：
  - 论文形式：$\tilde{z}_T^n = \alpha_n \cdot \operatorname{Inv}(r_0) + (1-\alpha_n)\cdot z_T^n$（$\alpha_n$ 从首帧到末帧递减，默认约 0.033 → 0.016），使首帧更“锁定参考图”，后续帧逐步放松。

### 3) 运动强度：用 SSIM 给视频打“动感等级”
- 观察：同一句动作文本可能对应不同速度/幅度，导致优化目标不一致。
- 做法：在训练数据上用相邻帧 SSIM 的平均值估计运动强度（SSIM 越低通常意味着变化越大），再把强度分桶成 10 个 level，作为 1 通道 embedding 拼入 UNet 输入。
- 推理：默认 level=5，可从 1~10 调节；level 越大通常运动越强，但可能引入更明显的 motion blur。

### 4) 文本重加权：让模型“听动作词、别被内容词带偏”
- 在 CLIP text embedding 后接 **3 层 Transformer + 线性投影** 预测每个 token 的权重，sigmoid 归一到 [0,1]。
- 用权重逐 token 乘回原 embedding，不破坏 CLIP 特征空间，同时强化 motion-related token。

### 5) 训练与评测设置（Implementation）
- 基座：Stable Diffusion v1.5。
- 训练集：WebVID。
- 训练：8×A100，16 帧，中心裁剪并缩放到 256×256；CFG 训练时以 0.5 概率丢弃文本；仅用 MSE loss。
- 定量指标：相邻帧 DINO/CLIP 相似度衡量帧一致性。

## 关键公式或算法直觉
- **内容保持的“渐进放松”**：$\alpha_n$ 随帧编号递减，让首帧强对齐参考图、后续帧允许合理变化。
- **强度条件的本质**：把“同一句话的多解”显式拆成（文本，强度）→ 运动，降低训练时的 label 噪声。
- **文本重加权的本质**：把 prompt 拆成“内容词”和“动作词”两股信号，避免内容冲突时把动作也一起压没。

## 关键原图讲解（来自 PDF）
> 说明：`extract-images` 对矢量/组合对象的 Figure 提取不完整（只抓到了部分示例帧，甚至出现空白图）。下列关键图通过将 PDF 页面渲染为位图后裁剪得到，信息来自原 PDF，不做臆测。

### 图 1：整体管线（Figure 2，PDF 第 3 页）
![图：LivePhoto 总体管线](../figures/2025-livephoto-real-image-animation-with-text-guided-motion-control/fig2_pipeline.png)
- 来自：PDF 第 3 页 Figure 2
- 展示：参考图 → reference latent（局部内容）；DINOv2 content encoder（全局内容）；UNet 内插入 motion modules；额外输入 motion intensity；以及 text re-weighting。
- 解读：这是“内容由图控制、运动由文控制”的核心结构图：把 motion intensity 当成显式条件，把文本的动作信息通过 re-weighting 强化后再用 cross-attention 注入。
- 与贡献关系：对应贡献 1（baseline）+ 2（强度条件）+ 3（文本重加权）。

### 图 2：文本重加权如何突出动作词（Figure 3，PDF 第 5 页）
![图：Text re-weighting 模块与 token 权重示例](../figures/2025-livephoto-real-image-animation-with-text-guided-motion-control/fig3_text_reweight_module.png)
- 来自：PDF 第 5 页 Figure 3
- 展示：3 层 Transformer 预测 token 权重；右侧示例中动作相关词（如 dancing / flies / opens）权重更高。
- 解读：模型学到“把动作词当主要条件”，从而在参考图内容强约束下仍能把 motion 指令传进去。
- 与贡献关系：对应贡献 3（Text Re-weighting）。

### 图 3：内容保持三件套的消融（Figure 4 + Table 1，PDF 第 5 页）
![图：内容保持消融（Reference latent / +Content encoder / +Prior inversion）](../figures/2025-livephoto-real-image-animation-with-text-guided-motion-control/fig4_content_guidance_ablation.png)
![表：内容保持定量（DINO/CLIP 相邻帧一致性）](../figures/2025-livephoto-real-image-animation-with-text-guided-motion-control/tab1_content_guidance.png)
- 来自：PDF 第 5 页 Figure 4、Table 1
- 展示：仅拼接 reference latent 容易后续帧 identity 漂移；加入 content encoder 与 prior inversion 后逐步改善；Table 1 显示 DINO/CLIP 一致性显著提升。
- 解读：
  - reference latent：更像“局部贴图约束”，对远帧约束变弱。
  - content encoder：补上全局语义/身份特征，改善后续帧。
  - prior inversion：从初始噪声就把外观先验注入，细节更稳。
- 与贡献关系：对应贡献 1（强 baseline）。

### 图 4：运动强度可控（Figure 5，PDF 第 6 页）
![图：运动强度从静到强（level 2/5/7/10）](../figures/2025-livephoto-real-image-animation-with-text-guided-motion-control/fig5_motion_intensity_only.png)
- 来自：PDF 第 6 页 Figure 5
- 展示：同一 prompt 下，不同强度 level 导致运动幅度/速度不同；无强度条件时容易静止或发糊。
- 解读：强度条件提供了“第二个旋钮”，把动作的幅度/速度从文本里剥离出来；但强度过高可能带来模糊。
- 与贡献关系：对应贡献 2（Motion Intensity Guidance）。

### 图 5：文本重加权消融（Figure 6，PDF 第 6 页）
![图：无重加权时会被内容词带跑（baby/dinosaur），有重加权时突出 waving）](../figures/2025-livephoto-real-image-animation-with-text-guided-motion-control/fig6_text_reweight.png)
- 来自：PDF 第 6 页 Figure 6
- 展示：无重加权时，模型可能忽略动作词或被内容描述误导（甚至改写主体）；重加权后能更强调动作（waving）。
- 解读：当 prompt 同时含“内容+动作”时，内容词与参考图冲突会导致整体抑制文本；重加权等价于把动作词从冲突中“救出来”。
- 与贡献关系：对应贡献 3（Text Re-weighting）。

### 图 6：模块的定量收益（Table 2，PDF 第 6 页）
![表：去掉强度/去掉重加权都会降分](../figures/2025-livephoto-real-image-animation-with-text-guided-motion-control/tab2_quant.png)
- 来自：PDF 第 6 页 Table 2
- 展示：完整 LivePhoto 的 DINO/CLIP 一致性最好；去掉 motion intensity 或 text re-weighting 都会下降。
- 解读：两个模块分别解决“强度歧义”和“动作词被压制”，对帧一致性/文本一致性均有贡献。
- 与贡献关系：支撑贡献 2 与 3 的有效性。

## 实验与评价
- 优点：
  - 在 SD 生态上做 I2V：工程上可落地（冻结大模型 + 插 motion module）。
  - 提供可解释的控制接口：文本 + 强度 level。
  - 文本重加权对“内容冲突”场景很关键，避免动作信息被一起压掉。
- 需要谨慎：
  - 强度 level 过高会引入模糊/不稳定。
  - 训练分辨率 256×256，真实应用可能依赖更高分辨率基座/后处理（论文也讨论了 GEN-2/Pika 等可能用更强数据与 SR）。

## 局限性
- 强度是离散 10 档且由 SSIM 估计，未必对所有类型运动都等价（比如局部运动 vs 全局相机运动）。
- 仍受限于训练数据与分辨率；高质量商业系统可能依赖更强数据/更大基座。

## 与库中相关论文的关系
- 与“更大规模 T2V 系统”互补：LivePhoto 更像可控 I2V 子任务，可作为更大视频生成系统的可控组件或交互前端。
  - [[2026-05-10_2025-wan-open-and-advanced-large-scale-video-generative-models|WAN 系统笔记]]
  - [[2026-05-12_2025-hunyuanvideo-a-systematic-framework-for-large-video-generative-models|HunyuanVideo 系统笔记]]
- 与扩散基础：LivePhoto 直接建立在 SD / LDM 体系上。
  - [[papers/2022_High-Resolution Image Synthesis with Latent Diffusion Models.pdf]]
  - [[papers/2020_Denoising diffusion probabilistic models.pdf]]

## 后续阅读建议
- 从基础到系统：先读 LDM/SD（理解 latent diffusion + cross-attention 条件注入），再回到 LivePhoto 看它如何“加条件 + 加模块”。
- 如果你关心“更强视频质量/更大模型”：对照本库的 WAN/HunyuanVideo 笔记，关注它们的数据与训练规模如何影响运动质量与分辨率。




































