---
type: paper-note
aliases:
  - "HunyuanVideo A Systematic Framework For Large Video Generative Models"
paper_id: "2024-hunyuanvideo-a-systematic-framework-for-large-video-generative-models"
title: "HunyuanVideo A Systematic Framework For Large Video Generative Models"
year: 2024
venue: ""
subfield: "Video Generation"
topics:
  - "text-to-video"
importance: 5
read_status: "unread"
note_status: "generated"
selected_on: "2026-05-23"
last_reviewed_on: ""
paper: "[[literature/papers/2024-hunyuanvideo-a-systematic-framework-for-large-video-generative-models]]"
pdf: "[[papers/2024_HunyuanVideo A Systematic Framework For Large Video Generative Models.pdf]]"
tags:
  - paper
  - paper/video_generation
  - topic/text_to_video
  - year/2024
---
# HunyuanVideo: A Systematic Framework For Large Video Generative Models

## 论文信息
- 标题：HunyuanVideo: A Systematic Framework For Large Video Generative Models
- 团队：Hunyuan Foundation Model Team（作者列表在文末）
- 类型：Technical Report / arXiv（arXiv:2412.03603v6，日期 2025-03-11）
- 论文页：[[literature/papers/2024-hunyuanvideo-a-systematic-framework-for-large-video-generative-models]]
- PDF：[[papers/2024_HunyuanVideo A Systematic Framework For Large Video Generative Models.pdf]]
- 代码（文中给出）：https://github.com/Tencent/HunyuanVideo

## 细分领域
视频生成（Video Generation），偏“**开放视频基础模型的系统性报告**”：从数据、结构、训练、推理加速、基础设施与下游应用（I2V、V2A 等）给出一套可复现的工程化框架。

## 重要程度
5/5：它把“大视频生成模型”拆成一套可落地的系统问题（数据→VAE→主干→训练→推理→分布式基础设施），并给出 13B 级别开源模型的关键取舍与经验（含 scaling law、Flow Matching 训练、极低步数推理、5D 并行等）。

## 一句话总结
HunyuanVideo 用 **Flow-Matching 的 DiT 主干 + Causal 3D-VAE 压缩空间 + 逐级数据/分辨率/时长课程学习 + 推理/训练系统优化**，把一个 13B 级开源视频基础模型从“能训出来”推进到“能高质量可用、可扩展、可复现”。

## 背景问题
1) 视频生成的 SOTA 往往闭源，社区缺少“足够强且可复现”的开源基础模型，导致算法迭代受限。  
2) 视频比图像更吃数据、算力与系统工程：长序列注意力、VAE 编解码 OOM、训练稳定性、推理速度等都是硬门槛。  
3) 单纯“更大数据/更大模型/更大算力”并不一定高效，需要 scaling 的定量指导与分阶段训练策略。

## 核心贡献（按系统链路拆解）
1) **数据策划与结构化标注**：分层过滤 + 分阶段构建多分辨率训练集；用结构化 JSON caption（含镜头运动、风格、光照等维度）提升可控性与泛化。  
2) **Causal 3D-VAE**：把高分辨率长视频压缩到时空 latent，配合 tiling 编解码与训练-推理一致性微调，解决长视频 VAE OOM 与拼接伪影。  
3) **统一的图像/视频生成架构**：在同一 DiT 主干上做图像预训练热身 + 图像/视频联合训练，提升视频训练收敛与质量。  
4) **Scaling Law + 训练策略**：用 scaling law 估计最优 N/D/C，结合分阶段 progressive training，把算力用在“更划算”的地方（文中提到可把计算需求降低到约 1/5）。  
5) **推理与训练系统优化**：时间步 shifting 支持极低步数推理；CFG guidance distillation 约 1.9× 加速；5D 并行（TP/SP/CP/DP+ZeRO 等）与自动容错提升大规模训练稳定性。

## 方法详解

### 1) 数据与文本条件：把 caption 当成“控制接口”
文中把 prompt 拆成多维结构（subject / dense description / background / style / shot type / lighting / atmosphere 等），并把来源标签、质量标签、相机运动分类器结果等一起写入 JSON 结构化 caption；再用 dropout + permutation/combination 组合出不同长度/模式的文本，缓解过拟合并提升泛化与可控性（例如镜头运动控制）。

### 2) Causal 3D-VAE：视频生成的“tokenizer”
目标：把像素空间视频压到更短的时空 token 序列，以便 DiT 主干可训练。

形状与压缩率（文中给出一组典型配置）：
- 输入视频：`(T+1) × 3 × H × W`
- 输出 latent：`(T/ct + 1) × C × (H/cs) × (W/cs)`，其中 `ct=4, cs=8, C=16`

训练损失（Equation (1)）：
$$
\\text{Loss}=L_1 + 0.1 L_{lpips} + 0.05 L_{adv} + 10^{-6} L_{kl}
$$

关键工程点：
- **课程学习**：从低分辨率/短视频逐步到高分辨率/长视频；随机采样间隔（1～8）帮助高运动片段重建。  
- **Tiling 编解码**：把长视频切成时空重叠块分别编解码并融合，避免单卡 OOM；同时加一个“tiling 随机开关”的微调阶段降低拼接伪影。

### 3) Flow Matching 训练：从扩散到 ODE 的同一条线
他们用 Flow Matching 训练生成模型。给定数据 latent `x1`，采样 `t∈[0,1]`（logit-normal），再采样噪声 `x0~N(0,I)`，用线性插值构造 `xt`，训练网络预测速度场（velocity）：
$$
\\mathcal{L}_{gen}=\\mathbb{E}_{t,\\mathbf{x}_0,\\mathbf{x}_1}\\|\\mathbf{v}_t-\\mathbf{u}_t\\|^2
$$
推理时用一阶 Euler ODE solver 积分得到 `x1`。

直觉：
- 把“去噪”理解成“沿着速度场把样本从噪声搬运到数据流形”，天然对应连续时间与可变步数推理；后面“时间步 shifting”就是在这种连续时间参数化下做更合适的时间采样/映射。

### 4) Scaling Law：决定 13B 不是拍脑袋
文中拟合了最优模型规模与 token 数随计算预算的幂律（Equation (2)）：
$$
N_{opt}=a_1 C^{b_1},\\quad D_{opt}=a_2 C^{b_2}
$$
并分别给出图像/视频模型族的拟合系数与曲线（Figure 10）。他们最终综合训练/推理成本选了 **13B** 作为 foundation model 尺度。

### 5) 推理加速：时间步 shifting + 文本引导蒸馏
时间步 shifting（Section 5.1）：把原始时间条件 `t` 映射到
$$
t' = \\frac{s\\,t}{1+(s-1)t}
$$
当推理步数更少时用更大的 `s`（文中示例：50 步 `s≈7`，<20 步 `s≈17`），让模型“更关注早期时间段的变化”，从而在 **10 步**这类极低步数下仍保持质量（Figure 11）。

CFG 文本引导蒸馏（Section 5.2）：把“有/无条件两次前向合成”的输出蒸馏到一个带 guidance-scale 条件的 student，文中报告约 **1.9×** 加速。

## 关键公式或算法直觉（浓缩）
1) 3D-VAE 的核心不是“重建好看”，而是给 DiT 提供一个**可扩展的时空 token 化接口**；tiling 与一致性微调是把它从论文落到工程的关键。  
2) Flow Matching 让生成过程天然变成 ODE 积分；步数缩减问题变成“时间参数化/采样是否合理”，因此出现时间步 shifting 这类技巧。  
3) scaling law 把“该加模型还是加数据”变成可计算的 trade-off，避免无效堆算力。

## 关键原图讲解（选 6 张）
> 说明：部分 Figure 在 PDF 中是矢量/组合对象时，自动抽图可能会拆分子图或顺序不稳定；若你需要逐格精确对照，请直接打开 PDF 查看原图。

### 图 1：开源 vs 闭源算力与效果差距（Figure 2）
![图：开源/闭源对比与性能对比](../figures/2024-hunyuanvideo-a-systematic-framework-for-large-video-generative-models/page_002_02_img-002-004.png)
- 来源：PDF 第 2 页（Figure 2）
- 展示内容：左侧对比开源/闭源视频生成模型在计算资源上的差距；右侧对比 HunyuanVideo 与强基线的性能/主观满意度
- 如何解读：这是全文的“问题定义 + 目标指标”图——他们要证明开源可以追上甚至超过部分闭源模型，并且要给出一整套可复现路径
- 与核心贡献关系：把贡献从“一个模型结构”提升为“系统性工程方案（数据/训练/推理/基础设施）”

### 图 2：极低步数推理的时间步调度（Figure 11，子图之一）
![图：时间步调度与 10-step 对比（子图 1）](../figures/2024-hunyuanvideo-a-systematic-framework-for-large-video-generative-models/page_012_01_img-012-128.jpg)
- 来源：PDF 第 12 页（Figure 11 的子图块之一，自动抽取可能把 (a)/(b) 拆开）
- 展示内容：不同 time-step scheduler 的曲线或 10-step 生成质量对比
- 如何解读：核心不是“哪条曲线更漂亮”，而是：步数越少越需要把采样重心放在更关键的早期阶段（对应本文的 shifting）
- 与核心贡献关系：支撑“推理步数缩减”这条系统优化链路，直接影响可用性与成本

### 图 3：极低步数推理的时间步调度（Figure 11，另一子图之一）
![图：时间步调度与 10-step 对比（子图 2）](../figures/2024-hunyuanvideo-a-systematic-framework-for-large-video-generative-models/page_012_02_img-012-127.jpg)
- 来源：PDF 第 12 页（Figure 11 的子图块之一）
- 展示内容：与上图互补的 scheduler/样例对比
- 如何解读：如果你在复现/部署里要做“10～20 步视频生成”，应优先把这种 **time re-parameterization** 放进推理管线，而不是只调 CFG 或分辨率
- 与核心贡献关系：把“Flow/扩散推理加速”落成可执行的策略

### 图 4：高运动样例展示（Figure 14）
![图：高运动动力学样例](../figures/2024-hunyuanvideo-a-systematic-framework-for-large-video-generative-models/page_016_01_img-016-141.jpg)
- 来源：PDF 第 16 页（Figure 14 的子图块之一）
- 展示内容：多种场景与运动类型（车辆、跑动、游泳、球场等）的生成样例
- 如何解读：视频生成里“看起来像真的”不够，还要能保持运动连贯与镜头语言；这类定性图能快速暴露模型的运动崩坏/物理不自然
- 与核心贡献关系：呼应他们在数据筛选、caption（镜头运动）、训练课程学习与基础设施投入上的目标：提升运动表现与稳定性

### 图 5：音效/音乐生成模型结构（Figure 18）
![图：V2A 声音生成结构](../figures/2024-hunyuanvideo-a-systematic-framework-for-large-video-generative-models/page_019_01_img-019-149.png)
- 来源：PDF 第 19 页（Figure 18）
- 展示内容：从“三流（visual/audio/text）”到“单流 DiT”的融合结构；结合 VAE（mel 频谱 latent）与 vocoder（HifiGAN）重建音频
- 如何解读：它展示了一个通用套路：不同模态先各自编码并投影到同一 latent space，再通过 transformer block 逐步对齐与融合
- 与核心贡献关系：说明 HunyuanVideo 不只是 T2V，也把“视频生成系统”延伸到多模态应用（V2A）

### 图 6：I2V 扩展的扩散主干（Figure 19）
![图：I2V diffusion backbone](../figures/2024-hunyuanvideo-a-systematic-framework-for-large-video-generative-models/page_020_02_img-020-150.png)
- 来源：PDF 第 20 页（Figure 19）
- 展示内容：在 T2V 基座上扩展 I2V：token replace（把参考图像 latent 作为首帧 latent，timestep=0）+ semantic image injection（用 MLLM 产出的语义 token 注入）
- 如何解读：I2V 的难点是“既要严格保留输入图的外观/身份，又要按文本产生运动”；他们用“首帧硬条件 + 语义注入”来同时满足两端
- 与核心贡献关系：体现“基础模型 + 下游适配模块”的系统视角（同一主干上做不同任务）

## 实验与评价（读图与读结论的方式）
这篇 report 的实验部分更像“系统验证”而非单点算法 SOTA：
- **VAE 侧**：用 PSNR 等重建指标对比多个开源 VAE，并强调对文字、小脸、复杂纹理的优势（Figure 7/ Table 1）。  
- **基础模型侧**：用专业人评（60 人、1500+ prompts）对比多个商业/闭源系统与强基线，强调总体满意度与关键维度（视觉质量、运动、文本对齐、分镜/切景）。  
- **推理效率侧**：时间步 shifting + guidance distillation 直接回答“13B 视频模型怎么跑得动”。  
- **系统侧**：5D 并行与容错（99.5% 稳定性）把“能训出来”与“持续训得住”讲清楚。

## 局限性（从工程角度看）
1) 作为系统报告，它提供了大量组件与技巧，但每个子模块的“理论最优性”不一定是重点；复现时要考虑组件间耦合。  
2) 自动抽图对矢量/组合 Figure 可能不完整（尤其是结构总览图、复杂表格/网格）；需要时应直接对照 PDF。  
3) 公开报告中的对比结论依赖其评测协议与 prompts 集合；在你的目标分布（特定镜头语言/人物/动作）上仍需要二次评测。

## 与库中相关论文的关系
- 开源大模型系统报告对照：[[notes/2026-05-14_2025-open-sora-2-0-training-a-commercial-level-video-generation-model-in-200k|Open-Sora 2.0]]、[[notes/2026-05-10_2025-wan-open-and-advanced-large-scale-video-generative-models|WAN (Open and Advanced)]]（同类“系统复盘”，但训练配方与工程取舍不同）。  
- 推理侧提速/延长：[[notes/2026-05-23_2024-freenoise-tuning-free-longer-video-diffusion-via-noise-rescheduling|FreeNoise]]、[[notes/2026-05-22_2024-fifo-diffusion-generating-infinite-videos-from-text-without-training|FIFO-Diffusion]]（更聚焦推理技巧；HunyuanVideo 是全链路系统）。  
- 数据与评测补齐：[[notes/2026-05-15_2025-openvid-1m-a-large-scale-high-quality-dataset-for-text-to-video-generation|OpenVid-1M]]、[[notes/2026-05-14_2025-t2v-compbench-a-comprehensive-benchmark-for-compositional-text-to-video-generati|T2V-CompBench]]（分别对应数据与组合性评测维度）。  
- 结构/训练范式对照：[[literature/papers/2023-scalable-diffusion-models-with-transformers|DiT]]（DiT 范式来源）、[[literature/papers/2023-make-a-video-text-to-video-generation-without-text-video-data|Make-A-Video]]（早期 T2V 思路对照）。

## 后续阅读建议
1) 先用系统视角读：把本文按“数据→VAE→主干→训练→推理→系统”画成一页流程图，对照 Figure 3/4/5/8（需要直接开 PDF）。  
2) 若你关心“开源追上闭源”的路径：对照 [[notes/2026-05-14_2025-open-sora-2-0-training-a-commercial-level-video-generation-model-in-200k|Open-Sora 2.0]] 与本文在数据/并行/推理上的差异点。  
3) 若你关心“更长、更省”推理：沿着 shifting / distillation / long-video 方法串起来读 [[notes/2026-05-23_2024-freenoise-tuning-free-longer-video-diffusion-via-noise-rescheduling|FreeNoise]] 与 [[notes/2026-05-22_2024-fifo-diffusion-generating-infinite-videos-from-text-without-training|FIFO-Diffusion]]。  
4) 想做复现实验：优先复现 Figure 11（10-step 质量对比）与 Figure 19（I2V token replace + semantic injection），它们最能暴露“配方是否跑通”。
















