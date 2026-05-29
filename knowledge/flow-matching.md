---
type: knowledge
aliases:
  - Flow Matching
  - Rectified Flow
  - Velocity Prediction
  - 流匹配
title: "Flow Matching"
field: "Diffusion / Flow Models"
topics:
  - diffusion-flow
  - text-to-video
  - text-to-image
status: evergreen
created_on: 2026-05-16
updated_on: 2026-05-17
tags:
  - knowledge
  - knowledge/generative_modeling
  - topic/diffusion_flow
---
# Flow Matching

Flow Matching（流匹配）是一种训练连续生成模型的方法。它不直接让网络一次性输出图像、视频或 latent，而是让网络学习一个**速度场**：给定当前状态、时间和条件，预测这个状态下一瞬间应该往哪里移动。采样时从简单噪声分布出发，沿着这个速度场积分，最终得到数据样本。

在视觉生成论文中，Flow Matching 经常和 DiT、latent VAE、文本编码器一起出现。论文里还会看到 velocity prediction、rectified flow、flow matching objective 等说法。它们不完全等价，但工程直觉相近：模型学习的不是“这一步有多少噪声”，而是“当前位置应该朝数据分布移动的速度”。

## 一句话理解

扩散模型像是学会逐步去噪；Flow Matching 更像是直接学习一张从噪声分布流向数据分布的速度地图。

## 核心设定

设简单初始分布为 $p_0$，通常是标准高斯；目标数据分布为 $p_1$ 或 $p_{\text{data}}$。我们希望学习一个时间相关的向量场：

$$
\frac{d x_t}{dt} = v_\theta(t, x_t, c), \quad t \in [0,1]
$$

其中：

- $x_t$ 是时间 $t$ 的样本状态，可以是像素、图像 latent、视频 latent 或其他连续表示。
- $v_\theta(t, x_t, c)$ 是神经网络预测的速度场。
- $c$ 是可选条件，例如文本 embedding、类别、首帧、历史帧、分辨率或运动强度。

如果 $v_\theta$ 学得足够好，从 $x_0 \sim p_0$ 出发解这个 ODE，$t=1$ 时的 $x_1$ 就应当服从目标数据分布。

## ODE 直觉

ODE 描述的是连续时间动力系统：

$$
\frac{d x_t}{dt} = v_\theta(t, x_t)
$$

含义是：给定当前位置 $x_t$ 和时间 $t$，速度场告诉样本接下来应该怎么动。离散化以后，最简单的一阶 Euler 更新是：

$$
x_{t+\Delta t} \approx x_t + \Delta t \, v_\theta(t, x_t)
$$

所以模型训练时学的是局部运动规则，生成时得到的是整条轨迹；轨迹终点就是生成样本。对图像来说，网络输出通常和输入同形状。如果：

$$
x_t \in \mathbb{R}^{C \times H \times W}
$$

那么：

$$
v_\theta(t, x_t) \in \mathbb{R}^{C \times H \times W}
$$

每个像素或 latent 位置都有一个当前变化方向。

## 概率路径

Flow Matching 的关键是先人为定义一族从噪声到数据的中间分布，也叫概率路径。给定真实样本 $x_1$，常见的条件高斯路径写作：

$$
x_t = \mu_t(x_1) + \sigma_t \epsilon,
\qquad \epsilon \sim \mathcal{N}(0,I)
$$

其中：

- $\mu_t(x_1)$ 是条件路径中心。
- $\sigma_t$ 控制噪声尺度。
- $\epsilon$ 是一次采样得到的高斯噪声。

直观上，$t=0$ 时状态更接近噪声，$t=1$ 时更接近真实样本。不同论文会选择不同的 $\mu_t$、$\sigma_t$ 或时间方向，所以阅读公式时要先看清楚作者如何定义 $t=0$ 和 $t=1$。

## 目标速度从哪里来

Flow Matching 的训练标签不是拍脑袋设定的，而是概率路径对时间的导数。

从条件路径开始：

$$
x_t = \mu_t(x_1) + \sigma_t \epsilon
$$

固定 $x_1$ 和 $\epsilon$，对时间 $t$ 求导：

$$
\frac{d x_t}{dt}
= \dot{\mu}_t(x_1) + \dot{\sigma}_t \epsilon
$$

训练时网络输入通常是 $(t, x_t)$，因此把 $\epsilon$ 消去：

$$
\epsilon = \frac{x_t - \mu_t(x_1)}{\sigma_t}
$$

代回可得理想条件速度：

$$
u_t(x \mid x_1)
= \dot{\mu}_t(x_1)
+ \frac{\dot{\sigma}_t}{\sigma_t}\bigl(x - \mu_t(x_1)\bigr)
$$

这个式子由两部分组成：

- **中心移动项** $\dot{\mu}_t(x_1)$：描述条件分布中心如何移动。
- **尺度变化项** $\frac{\dot{\sigma}_t}{\sigma_t}(x - \mu_t(x_1))$：描述分布如何收缩或扩张。

如果 $\sigma_t$ 变小，点会整体向中心收缩；如果 $\sigma_t$ 变大，点会向外扩张。

## 线性路径特例

最常见、也最容易理解的路径是从噪声 $x_0$ 到数据 $x_1$ 的线性插值：

$$
x_t = (1-t)x_0 + t x_1
$$

直接求导得到：

$$
u_t = \frac{d x_t}{dt} = x_1 - x_0
$$

也可以写成条件高斯形式：

$$
x_t = t x_1 + (1-t)\epsilon
$$

此时：

$$
\mu_t(x_1) = t x_1,
\qquad
\sigma_t = 1-t
$$

代入通用公式后：

$$
u_t(x \mid x_1)
= x_1 - \frac{1}{1-t}(x - t x_1)
$$

又因为 $x = t x_1 + (1-t)\epsilon$，所以：

$$
u_t(x \mid x_1) = x_1 - \epsilon
$$

这和直接对线性路径求导完全一致。很多工程实现中的 velocity 目标就是这一类路径的变体。

## 训练目标

给定路径和理想条件速度后，训练就是一个监督回归问题：

$$
\mathcal{L}(\theta)
= \mathbb{E}_{t,x_1,\epsilon}
\left[
\left\|v_\theta(t,x_t,c)-u_t(x_t \mid x_1)\right\|_2^2
\right]
$$

典型训练流程：

1. 从数据集中采样真实样本 $x_1$。
2. 采样时间 $t \sim \mathrm{Uniform}(0,1)$，实际系统也可能使用 logit-normal 等非均匀时间采样。
3. 采样噪声 $\epsilon \sim \mathcal{N}(0,I)$。
4. 用概率路径构造中间状态 $x_t$。
5. 计算目标速度 $u_t(x_t \mid x_1)$。
6. 把 $(x_t,t,c)$ 输入网络，得到预测速度 $v_\theta(t,x_t,c)$。
7. 用 MSE 回归目标速度并更新参数。

伪代码如下：

```python
# x1: data sample, such as image/video latent
for x1, cond in dataloader:
    t = Uniform(0, 1).sample()
    eps = Normal(0, I).sample_like(x1)

    xt = mu_t(x1, t) + sigma_t(t) * eps
    target_u = dmu_dt(x1, t) + dsigma_dt(t) / sigma_t(t) * (xt - mu_t(x1, t))

    pred_v = model(xt, t, cond)
    loss = ((pred_v - target_u) ** 2).mean()

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

## 采样过程

训练好以后，生成不需要真实样本 $x_1$。采样只需要从噪声分布开始，然后沿学习到的速度场积分：

$$
x_0 \sim p_0,
\qquad
\frac{d x_t}{dt} = v_\theta(t,x_t,c),
\qquad
t:0 \to 1
$$

离散化后可以写成：

```python
x = randn(shape)  # x0 ~ N(0, I)

for t in time_grid:
    v = model(x, t, cond)
    x = x + dt * v

# x is the generated sample
```

实际系统会使用 Euler、Heun 或其他 ODE solver，并配合时间步调度、classifier-free guidance、VAE 解码、并行推理等工程细节。速度场越平滑、轨迹越接近直线，通常越容易用较少步数采样。

## 为什么训练依赖真实样本，采样却不需要

训练时我们知道真实样本 $x_1$，因此可以构造条件路径和条件速度 $u_t(x \mid x_1)$。采样时当然不知道最终会生成哪一个 $x_1$，看起来似乎无法使用这个标签。

关键在于平方误差回归的最优解。对目标：

$$
\mathbb{E}\left[\|f(Z)-Y\|^2\right]
$$

最优函数是条件期望：

$$
f^*(Z)=\mathbb{E}[Y \mid Z]
$$

在 Flow Matching 中：

$$
Z=(t,x_t),
\qquad
Y=u_t(x_t \mid x_1)
$$

所以理想网络满足：

$$
v_t^*(x)
= \mathbb{E}\left[u_t(x \mid x_1) \mid x_t=x\right]
$$

也就是说，网络从许多由真实样本提供的局部监督中，学到一个只依赖当前位置和时间的全局速度场。生成时只要沿这个全局速度场走，就可以把整个噪声分布推向数据分布。

## 和 CNF 的关系

Continuous Normalizing Flow（CNF，连续归一化流）也是用 ODE 定义从简单分布到复杂分布的连续可逆变换：

$$
\frac{d x_t}{dt}=v_\theta(t,x_t)
$$

传统 CNF 通常通过最大似然训练，需要跟踪 ODE 轨迹和密度变化，训练成本较高。Flow Matching 的改动是：仍然学习一个可用于 ODE 采样的向量场，但把训练改成监督式速度回归，不必在训练阶段显式优化整段 likelihood。

所以可以把 Flow Matching 理解为一种更直接的 CNF 训练方式：采样时仍然解 ODE，训练时主要回归路径导数。

## 和扩散模型的关系

Flow Matching 和扩散模型都在学习从简单分布到数据分布的变换。主要差别在训练目标和路径表述：

| 维度 | 扩散/去噪视角 | Flow Matching 视角 |
|---|---|---|
| 核心任务 | 从带噪样本恢复噪声、score 或干净样本 | 预测连续路径上的速度 |
| 常见预测量 | noise、score、x0、v | velocity / flow |
| 推理形式 | 反向扩散、SDE 或 probability-flow ODE | 沿速度场积分 |
| 工程直觉 | 每一步去掉一部分噪声 | 每一步朝数据分布移动 |
| 常见主干 | U-Net、DiT | U-Net、DiT |

现代视觉系统里两者边界经常很近：都可能在 latent 空间中训练，都可能使用文本条件、CFG、VAE 和采样调度器。区别往往体现在目标参数化、时间路径和 solver 设计上。

## Rectified Flow 是什么

Rectified Flow 可以看作 Flow Matching/velocity 训练的一类常见形式，重点是把噪声到数据的传输路径尽量“拉直”。如果模型学习到的轨迹更接近直线，采样时通常可以用更少 ODE 步数获得较好结果。

论文中提到 rectified flow 时，通常需要关注两件事：

- 训练目标是不是直接预测从当前状态到目标方向的 velocity。
- 采样轨迹是否被设计得更直、更稳定，从而减少高分辨率图像或视频生成的步数。

## 为什么适合视觉和视频生成

视频生成比图像生成更吃 token、算力和显存。Flow Matching 在图像和视频系统中常见，主要因为：

- **目标清晰**：直接在像素或 latent token 上回归速度，容易和 DiT 的 token 建模方式结合。
- **采样效率好**：速度场足够稳定时，可以用较少步数完成采样。
- **扩展性强**：可以放进 latent VAE + DiT + 文本编码器的标准大模型管线。
- **条件接口自然**：文本、首帧、历史帧、分辨率、运动强度、时间位置等都可以作为条件影响速度预测。
- **适合多尺度设计**：视频论文可以把 flow objective 和空间/时间金字塔、分段生成、历史条件压缩结合起来。

因此很多视频生成论文采用 DiT + Flow Matching：Flow Matching 给出训练目标，DiT 负责大规模 token 建模，VAE 负责把高维视频压到可训练的 latent 空间。

## 阅读论文时该看什么

读到一篇使用 Flow Matching 或 Rectified Flow 的论文，可以优先检查：

1. **时间方向**：$t=0$ 是噪声还是数据，velocity 的符号是否随之改变。
2. **路径定义**：直线路径、条件高斯路径、rectified path，还是多阶段/多尺度路径。
3. **预测量**：velocity、noise、score、x0，还是混合参数化。
4. **训练空间**：像素空间、图像 latent、视频 latent，还是连续视觉 token。
5. **条件注入**：文本、类别、首帧、历史帧、分辨率、运动控制等如何进入模型。
6. **时间采样**：均匀采样、logit-normal 采样，还是任务相关的 schedule。
7. **采样器**：Euler、Heun、DPM 类 solver，采样步数是多少。
8. **效率设计**：是否有多尺度、分阶段、token 压缩、KV cache 或并行推理。
9. **收益落点**：最终改进是质量、速度、长视频一致性，还是训练成本。

## 常见误区

**误区 1：Flow Matching 完全取代扩散模型。**  
更准确的说法是，它提供了另一种连续生成建模目标。很多系统仍然沿用扩散模型的 latent 表示、条件注入、CFG、采样调度和评估方式。

**误区 2：只要用了 Flow Matching，采样一定很快。**  
采样速度还取决于模型规模、分辨率、视频长度、VAE 压缩率、ODE 步数、solver 稳定性和并行实现。

**误区 3：Flow Matching 等于 DiT。**  
DiT 是 Transformer 结构，Flow Matching 是训练目标。二者经常一起出现，但不是同一层概念。

**误区 4：训练时回归速度，说明网络不会生成图像。**  
图像不是一步前向直接吐出来的，而是由速度场积分得到的轨迹终点。学速度和生成图像并不矛盾。

## 与本库论文的连接

- [[literature/topics/diffusion-flow|diffusion-flow]]：本知识点对应的主题轴。
- [[literature/fields/diffusion-flow-models|Diffusion / Flow Models]]：主要理论和方法归属。
- [[literature/papers/2025-pyramidal-flow-matching-for-efficient-video-generative-modeling|Pyramidal Flow Matching for Efficient Video Generative Modeling]]：把 Flow Matching 和空间/时间金字塔结合，用于更高效的视频训练。
- [[notes/2026-05-16_2025-pyramidal-flow-matching-for-efficient-video-generative-modeling|Pyramidal Flow Matching 精讲笔记]]：本库已有的带图讲解，重点是多分辨率 stage、renoising 和 temporal pyramid。
- [[literature/papers/2024-scaling-rectified-flow-transformers-for-high-resolution-image-synthesis|Scaling Rectified Flow Transformers for High-Resolution Image Synthesis]]：图像方向的 rectified flow + Transformer 扩展路线。
- [[literature/papers/2025-wan-open-and-advanced-large-scale-video-generative-models|Wan Open and Advanced Large-Scale Video Generative Models]]：视频生成系统中 DiT + Flow Matching 的工程化组合。
- [[literature/papers/2024-hunyuanvideo-a-systematic-framework-for-large-video-generative-models|HunyuanVideo A Systematic Framework for Large Video Generative Models]]：大规模视频模型中 flow matching、3D VAE 和文本条件训练的系统整合。
- [[literature/papers/2025-open-sora-2-0-training-a-commercial-level-video-generation-model-in-200k|Open-Sora 2.0 Training a Commercial-Level Video Generation Model in $200k]]：工程约束下使用 velocity/flow 目标训练商业级视频模型的案例。
- [[literature/papers/2023-scalable-diffusion-models-with-transformers|Scalable Diffusion Models with Transformers]]：理解 DiT 结构为什么适合和 flow/diffusion 目标结合。
- [[literature/papers/2020-denoising-diffusion-probabilistic-models|Denoising Diffusion Probabilistic Models]]：理解扩散模型基线，再对照 Flow Matching 的目标差异。

## 相关阅读笔记

- [[notes/2026-05-10_2025-wan-open-and-advanced-large-scale-video-generative-models|Wan 精讲笔记]]
- [[notes/2026-05-12_2025-hunyuanvideo-a-systematic-framework-for-large-video-generative-models|HunyuanVideo 精讲笔记]]
- [[notes/2026-05-14_2025-open-sora-2-0-training-a-commercial-level-video-generation-model-in-200k|Open-Sora 2.0 精讲笔记]]
- [[notes/2026-05-16_2025-pyramidal-flow-matching-for-efficient-video-generative-modeling|Pyramidal Flow Matching 精讲笔记]]
- [[notes/2026-05-11_2025-autoregressive-video-generation-without-vector-quantization|NOVA 精讲笔记]]

## 相关概念

- [[literature/topics/text-to-video|text-to-video]]
- [[literature/topics/text-to-image|text-to-image]]
- [[literature/topics/autoregressive|autoregressive]]
- [[knowledge/index|基础知识库]]

## 压缩记忆版

Flow Matching = 用路径导数提供“局部正确速度”的监督，训练一个连续时间生成动力系统。训练时模型回归速度，采样时从噪声出发沿速度场积分；最终图像或视频不是直接预测出来的，而是速度场推动分布演化后的终点。
