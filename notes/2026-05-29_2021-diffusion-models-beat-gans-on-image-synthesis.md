---
type: paper-note
aliases:
  - "Diffusion models beat gans on image synthesis"
paper_id: "2021-diffusion-models-beat-gans-on-image-synthesis"
title: "Diffusion models beat gans on image synthesis"
year: 2021
venue: "NIPS"
subfield: "Diffusion / Flow Models"
topics:
  - "diffusion-flow"
  - "text-to-image"
importance: 5
read_status: "unread"
note_status: "generated"
selected_on: "2026-05-29"
last_reviewed_on: ""
paper: "[[literature/papers/2021-diffusion-models-beat-gans-on-image-synthesis]]"
pdf: "[[papers/2021_Diffusion models beat gans on image synthesis_NIPS.pdf]]"
tags:
  - paper
  - paper/diffusion_flow_models
  - topic/diffusion_flow
  - topic/text_to_image
  - venue/nips
  - year/2021
---
# 论文信息

- 标题：Diffusion Models Beat GANs on Image Synthesis
- 年份：2021
- Venue：NeurIPS 2021（库内字段：NIPS）
- 论文页：[[literature/papers/2021-diffusion-models-beat-gans-on-image-synthesis]]
- PDF：[[papers/2021_Diffusion models beat gans on image synthesis_NIPS.pdf]]
- 关键词：diffusion-flow，text-to-image；classifier guidance；architecture ablations；(precision/recall) 多样性/保真度权衡

# 细分领域与重要程度

- 细分领域：Diffusion / Flow Models
- 重要程度：5/5  
  这篇是“扩散模型在 ImageNet 上系统性打赢 GAN（指标与样例层面）”的标志性节点：一手把 **UNet 架构工程化到更像 GAN 的强 backbone**，一手提出 **classifier guidance** 作为“保真度 vs 多样性”的可调旋钮。

# 一句话总结

通过一系列 UNet 架构改造 + 用分类器梯度在采样时引导（classifier guidance），作者把扩散模型在 ImageNet 多分辨率上推到当时的 SOTA，并且首次把“多样性/覆盖度 vs 画面保真度”变成可调参数。

# 背景问题（作者要解决什么）

1. **扩散模型当时在 ImageNet/高分辨率上仍落后强 GAN**：指标（FID/IS）与视觉保真度上不占优。  
2. **GAN 有天然的“调 fidelity、牺牲 diversity”的旋钮**（例如 truncation trick），扩散模型缺少类似手段。  
3. 既有扩散实现的 **架构探索不充分**：相比 GAN 社区多年打磨，扩散 UNet 的结构/注意力/残差等还有大量空间。

# 核心贡献（按论文主线）

1. **架构改造带来显著 FID 提升**：在 UNet 上系统 ablation，得到作者后续默认的“更强扩散 UNet”（文中 ADM 相关设置）。  
2. **Classifier Guidance**：训练一个在噪声图像上工作的分类器 `p(y|x_t,t)`，在采样每一步用 `∇_{x_t} log p(y|x_t,t)` 进行引导，让扩散在条件生成上获得显著提升，并提供可调的 trade-off。  
3. **高分辨率与两阶段 upsampling 可叠加**：guidance 与 upsampling diffusion stack 在不同维度提升质量，组合可进一步提升指标（如 512×512 FID 3.85）。

# 方法详解

## 1) 扩散模型回顾（噪声预测范式）

论文用 DDPM 的直观表述解释扩散：从高斯噪声 `x_T` 出发，逐步去噪得到 `x_0`。模型用 `ε_θ(x_t,t)` 预测噪声分量，用 MSE 训练（文中描述为 `||ε_θ(x_t,t) - ε||^2`）。

在采样端，单步反向过程 `p_θ(x_{t-1}|x_t)` 可写成对角高斯 `N(μ_θ(x_t,t), Σ_θ(x_t,t))`；论文沿用并采用 Nichol & Dhariwal 2021 的一些改进（如学习/插值反向方差、混合目标），以支持更少步数采样时的质量。

## 2) 架构改造：把“扩散 UNet”做得更像强生成 backbone

论文明确列出的关键改动（并在 ImageNet 128×128 上做 ablation）包括：

- **注意力更密、更细尺度**：不仅在 16×16 用注意力，还扩展到 32×32、16×16、8×8。  
- **更多注意力头/更合适的 head 维度**：更贴近 Transformer 的常用配置（作者默认 64 channels per head）。  
- **BigGAN 风格的 up/down residual block**：用于上下采样路径。  
- **残差连接重标定**（rescale residual connections）。  
- **Adaptive GroupNorm**：用于把 timestep / class embedding 注入到残差块中（作者指出去掉会显著变差）。

直觉上：扩散在每一步都要“修正细节”，UNet 的表征能力上限非常关键；把注意力下沉到更高分辨率、增加 head 与更合理的归一化/残差设计，会直接提升样本保真度。

## 3) Classifier Guidance：用分类器梯度当“可控旋钮”

核心做法：

- 训练分类器在噪声图像上预测类别：`p(y|x_t,t)`。  
- 采样时，每一步不仅按扩散模型给出的 `μ, Σ` 采样，还加上一个沿着分类器梯度的偏移项（论文 Algorithm 1 的核心形态）：

`x_{t-1} ~ N( μ + s · Σ · ∇_{x_t} log p(y|x_t,t),  Σ )`

其中 `s` 是 guidance scale。作者观察到在大规模任务上 `s=1` 往往不够，需要更大的 `s` 才能得到视觉上更“像该类”的样本；并给出解释：`s · ∇ log p(y|x)` 等价于对 `p(y|x)^s` 的对数梯度，使分布变“更尖”，从而提升 fidelity、牺牲 diversity。

# 关键公式或算法直觉

1. **噪声预测损失**：把学习难题转为“预测加入的噪声”，MSE 训练稳定、可扩展。  
2. **引导项的意义**：`∇_{x_t} log p(y|x_t,t)` 是“朝着更像类别 y 的方向改动当前噪声图像”的最直接信号；乘上 `Σ` 相当于按该步的不确定性/尺度进行合适步长的修正。  
3. **Scale `s` 就是 trade-off knob**：`s↑` 会更强调分类器高置信的模式，通常 precision/IS 更好，但 recall（覆盖度）下降；FID 往往在中等 `s` 处最好（论文 Figure 3 的结论）。

# 关键原图讲解

本次 `extract-images` 仅提取到 6 张位图（主要是样例对比）。论文中一些关键曲线/结构图（如 Figure 3 的 trade-off 曲线）可能是矢量/组合对象，自动提取可能不完整；下面以可提取图片 + 正文描述为主。

## 图 1：512×512 的高质量无条件样例（Figure 1）

![图：512×512 无条件样例](../figures/2021-diffusion-models-beat-gans-on-image-synthesis/page_001_01_img-001-000.jpg)

- 来源：PDF 第 1 页（`page_001_01_img-001-000.jpg`）
- 展示内容：作者“最佳 ImageNet 512×512 模型”的随机样例拼图（caption 提到 FID 3.85）。
- 如何解读：这张图要证明的不仅是“能生成清晰图”，更是扩散在高分辨率上已经能达到很强的视觉保真度。
- 与核心贡献关系：作为定性证据支撑“扩散 > GAN”的主张，并为后面的 guidance/upsampling 组合结果做铺垫。

## 图 2：同一个无条件模型，用不同 scale 做 classifier guidance（Figure 2 的两个子图）

![图：classifier scale=1.0](../figures/2021-diffusion-models-beat-gans-on-image-synthesis/page_005_02_img-005-001.jpg)

- 来源：PDF 第 5 页（`page_005_02_img-005-001.jpg`）
- 展示内容：以某类别（caption 里是 “Pembroke Welsh corgi”）为目标时，scale 较小（如 1.0）的生成样例。
- 如何解读：视觉上往往“像普通 ImageNet 图片”，但类一致性不强；对应作者的观察：分类器概率看起来不差，但人眼并不觉得“真的像该类”。
- 与核心贡献关系：解释为什么要把 `s` 当旋钮，并且在大规模任务上 `s>1` 很关键。

![图：classifier scale=10.0](../figures/2021-diffusion-models-beat-gans-on-image-synthesis/page_005_01_img-005-002.jpg)

- 来源：PDF 第 5 页（`page_005_01_img-005-002.jpg`）
- 展示内容：同类别目标下 scale 较大（如 10.0）的样例。
- 如何解读：类别一致性显著增强，但通常也意味着多样性会下降（更集中在分类器的高置信模式）。
- 与核心贡献关系：这是 classifier guidance 机制的“最直观演示”，也是后来 CFG（classifier-free guidance）思想的前身之一（但实现方式不同）。

## 图 3：BigGAN truncation vs guidance vs 训练集对照（Figure 4 的三列）

![图：BigGAN-deep（truncation=1.0）样例](../figures/2021-diffusion-models-beat-gans-on-image-synthesis/page_008_01_img-008-005.jpg)

- 来源：PDF 第 8 页（`page_008_01_img-008-005.jpg`）
- 展示内容：BigGAN-deep 在 truncation=1.0 下的随机样例（caption 提到 FID 6.95）。
- 如何解读：GAN 的样例往往很“像”，但模式覆盖（作者后文用 recall 指标刻画）可能不足。
- 与核心贡献关系：作为强 GAN baseline，明确比较对象。

![图：扩散 + classifier guidance 样例](../figures/2021-diffusion-models-beat-gans-on-image-synthesis/page_008_02_img-008-004.jpg)

- 来源：PDF 第 8 页（`page_008_02_img-008-004.jpg`）
- 展示内容：扩散模型在 guidance 下的随机样例（caption 提到 FID 4.59）。
- 如何解读：作者希望读者看到“同等甚至更强的感知质量”，并结合表格指标看到更好的 FID/precision 以及可控的 recall。
- 与核心贡献关系：把论文主张落到“定性样例 + 定量指标”双证据。

![图：训练集样例对照](../figures/2021-diffusion-models-beat-gans-on-image-synthesis/page_008_03_img-008-003.jpg)

- 来源：PDF 第 8 页（`page_008_03_img-008-003.jpg`）
- 展示内容：训练集真实图像样例对照。
- 如何解读：用于校准“真实分布的多样性/构图范围”，也帮助理解作者提到的一些模式（例如不同视角/局部裁剪的物体等）是否覆盖得更好。
- 与核心贡献关系：支撑论文强调的“coverage / recall”叙事，而不只盯着单一 fidelity 指标。

# 实验与评价（读者应带走什么）

- **指标体系**：不仅报告 FID/IS，也用 Precision/Recall 拆分 fidelity 与 diversity，强调“扩散覆盖更好”这一点。  
- **主要结论**：架构改造后，无条件生成已很强；加上 classifier guidance 后，条件生成在多个分辨率上显著超过当时最强 GAN。  
- **可控性**：`s` 提供平滑的 trade-off 曲线（论文 Figure 3 的要点）：`s↑` 通常 precision/IS ↑、recall ↓，FID 常在中等 `s` 最优。  
- **可组合性**：guidance 与两阶段 upsampling diffusion stack 的提升维度不同，可组合拿到更强指标（例如 512×512 FID 3.85）。

# 局限性

1. **采样仍慢**：即使能用 DDIM 等减少步数，扩散依然需要多次前向；论文也指出单步蒸馏等方向但当时效果不够。  
2. **需要额外分类器**：classifier guidance 需要训练/维护噪声条件分类器；这在后续被 CFG 等方法在工程上部分替代（不需要单独分类器）。  
3. **Trade-off 并非“免费”**：fidelity 提升往往伴随 diversity/recall 下降；如何在不同应用中选 `s` 仍是实践问题。

# 与库中相关论文的关系

- 与扩散基础/后续体系：[[literature/papers/2022-high-resolution-image-synthesis-with-latent-diffusion-models|LDM]] 把扩散推向“潜空间 + 文本条件”的工程化路线；本论文更像“扩散在像素空间打赢 GAN 的里程碑”。  
- 与 Transformer-based diffusion：[[literature/papers/2023-scalable-diffusion-models-with-transformers|DiT]] 把 backbone 进一步推向 Transformer；但“高保真 vs 多样性”的调参逻辑仍沿用 guidance 思想。  
- 与 GAN/离散表征路线：[[literature/papers/2021-taming-transformers-for-high-resolution-image-synthesis|Taming Transformers / VQGAN]] 代表当时另一条强生成路线（离散 token + AR/Transformer）；本论文用扩散路线在指标上正面对抗 GAN。

# 后续阅读建议

1. 先把 guidance 的“旋钮直觉”吃透：为什么 `s` 会提升 fidelity、牺牲 diversity，以及 precision/recall 的拆解方式。  
2. 对比后续更常用的 **Classifier-Free Guidance (CFG)**：它解决了“需要额外分类器”的工程痛点，但 trade-off 本质相同。  
3. 想理解“扩散 backbone 进化”的：从本论文的 UNet 工程化出发，继续看 DiT、以及后续更大规模/更快采样的工作（蒸馏、ODE/flow、rectified flow 等）。



