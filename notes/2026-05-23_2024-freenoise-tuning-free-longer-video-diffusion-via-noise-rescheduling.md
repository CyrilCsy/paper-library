---
type: paper-note
aliases:
  - "FreeNoise Tuning-Free Longer Video Diffusion via Noise Rescheduling"
paper_id: "2024-freenoise-tuning-free-longer-video-diffusion-via-noise-rescheduling"
title: "FreeNoise Tuning-Free Longer Video Diffusion via Noise Rescheduling"
year: 2024
venue: "ICLR"
subfield: "Long Video Generation"
topics:
  - "diffusion-flow"
  - "long-video"
importance: 5
read_status: "unread"
note_status: "generated"
selected_on: "2026-05-23"
last_reviewed_on: ""
paper: "[[literature/papers/2024-freenoise-tuning-free-longer-video-diffusion-via-noise-rescheduling]]"
pdf: "[[papers/2024_FreeNoise Tuning-Free Longer Video Diffusion via Noise Rescheduling_ICLR.pdf]]"
tags:
  - paper
  - paper/long_video_generation
  - topic/diffusion_flow
  - topic/long_video
  - venue/iclr
  - year/2024
---
# FreeNoise: Tuning-Free Longer Video Diffusion via Noise Rescheduling

## 论文信息
- 标题：FreeNoise: Tuning-Free Longer Video Diffusion via Noise Rescheduling
- 作者：Haonan Qiu, Menghan Xia, Yong Zhang, Yingqing He, Xintao Wang, Ying Shan, Ziwei Liu
- 会议：ICLR 2024（arXiv:2310.15169v3, 2024-01-30）
- PDF：[[papers/2024_FreeNoise Tuning-Free Longer Video Diffusion via Noise Rescheduling_ICLR.pdf]]
- 代码（论文中给出）：在 VideoCrafter / AnimateDiff / LaVie 上实现（见论文 Reproducibility Statement）

## 细分领域
长视频生成（Long Video Generation），更偏“推理时延长/增强”而非重新训练基座模型。

## 重要程度
5/5：不训练（tuning-free）就能把“训练帧长较短”的 T2V 扩展到更长帧数，并给出较清晰的机制解释（初始噪声、时序模块的作用差异）+ 较低额外推理开销。

## 一句话总结
通过“**噪声重排 + 时序注意力滑窗融合**”让预训练视频扩散模型在不训练的前提下生成更长视频，并用“**motion injection**”在保持场景一致的同时支持多 prompt 的动作/事件切换。

## 背景问题
1) 现有视频扩散模型常以较短帧数训练（如 16 帧），推理直接生成更长序列（如 64 帧）会出现明显质量坍塌（训练-推理帧长鸿沟）。  
2) 真实叙事需要多段文本条件（multi-prompt）：视频内容随时间变化；但多数模型训练时只见过单 prompt，推理时直接换 prompt 往往破坏一致性。

## 核心贡献
1) **Noise Rescheduling**：构造“既有随机性又有长程相关”的初始噪声序列，提升长视频的一致性与可生成性。  
2) **Window-based Attention Fusion**：把时序自注意力从“全局”改为“滑窗”，并对重叠窗口输出做平滑融合，避免 attentive-scope sensitivity（超出训练帧长导致注意力失效/分布外）。  
3) **Motion Injection for Multi-prompt**：在合适的时间步与网络层把目标 prompt 注入 cross-attention，让布局/外观尽量继承第一个 prompt，同时实现动作/姿态随 prompt 切换。

## 方法详解
### 1) 关键观察：初始噪声不仅“定外观”，其时序顺序还会影响内容演化
文中以 VideoLDM 的时序建模为例，指出两类跨帧算子性质不同：
- **Temporal Attention（时序注意力）**：更偏“顺序无关”的全局交互。
- **Temporal Convolution（时序卷积）**：顺序相关，负责时序连续性/内容演化的一部分。

因此，若为长视频直接拼接独立采样噪声，模型会面临“噪声间缺乏长程相关”与“注意力处理超出训练范围”的双重问题。

### 2) Noise Rescheduling：局部洗牌单元（Local Noise Shuffle Unit）
设训练帧长为 `N_train`（如 16），目标帧长为 `M`（如 64）。先采样 `N_train` 个独立噪声帧：
`[ε1, ε2, ..., εN_train]`。

然后用大小为 `S` 的“局部洗牌单元”反复重排来扩展到 `M` 帧（`S` 是 `N_train` 的因子，默认示例里 `S=4`）：

$$
[\\varepsilon_1,\\ldots,\\varepsilon_{N_{train}},\\; shuffle(\\varepsilon_1,\\ldots,\\varepsilon_S),\\;\\ldots]
\\tag{7}
$$

直觉：
- **重复使用**同一组噪声 ⇒ 给长视频引入“跨片段的一致基础”（长程相关）。
- **局部洗牌**顺序 ⇒ 在一致基础上仍能产生内容变化，避免完全复制粘贴式的静态视频。

### 3) Window-based Attention Fusion：让时序注意力“像训练时一样工作”
问题：时序注意力若对所有 `M` 帧做全局注意力，会引发“注意力范围敏感”（超出训练帧长分布）。  
做法：只在每个长度为 `U=N_train` 的滑动窗口里算注意力（stride 取 `S`，确保每个窗口覆盖的噪声集合仍近似“i.i.d + 洗牌”）：

$$
F_{i:i+U} = Attn_{temp}(Q_{i:i+U}, K_{i:i+U}, V_{i:i+U})
\\tag{8}
$$

由于帧会被多个重叠窗口覆盖，需要把这些窗口输出融合。文中给出一种“以到窗口中心距离为权重”的平滑加权融合（避免简单平均导致边界跳变）：

$$
F_i^o = \\frac{\\sum_j F_i^j \\cdot (U-\\lfloor |i-c_j| \\rfloor)}
{\\sum_j (U-\\lfloor |i-c_j| \\rfloor)}
\\tag{9}
$$

工程直觉：只改“时序注意力”的窗口切分/融合，其他模块不额外重复计算 ⇒ 额外开销小。

### 4) Multi-prompt 的 Motion Injection：何时、在哪些层换 prompt
设两段 prompt 为 `P1 -> P2`。核心思想：
- 多数去噪步（更决定布局/外观）用 `P1`；
- 仅在一段时间步区间 `[Tα, Tβ]`（更决定形状/姿态）和/或 U-Net 的后 `L` 个 cross-attn 层注入目标 prompt（逐帧插值过渡），以实现平滑事件切换。

$$
MotionInjection =
\\begin{cases}
Attn_{cross}(Q, K(P_e), V(P_e)), & \\text{if } T_\\alpha < t < T_\\beta \\;\\text{or}\\; l > L \\\\
Attn_{cross}(Q, K(P_1), V(P_1)), & \\text{otherwise}
\\end{cases}
\\tag{11}
$$

其中目标 prompt embedding `P_e` 随帧索引 `n` 在 `[N_\\gamma, N_\\tau]` 做线性插值以保证过渡平滑：

$$
P_e =
\\begin{cases}
P_1, & n < N_\\gamma \\\\
P_1 + \\frac{n-N_\\gamma}{N_\\tau-N_\\gamma}(P_2-P_1), & N_\\gamma \\le n < N_\\tau \\\\
P_2, & \\text{otherwise}
\\end{cases}
\\tag{12}
$$

## 关键公式或算法直觉
- **噪声重排**是在“初始条件空间”里制造长程相关：长视频的多个子片段共享同一组噪声原子，只是顺序不同，从而更容易保持主体/场景一致。  
- **注意力滑窗**是在“计算图”层面恢复训练分布：永远只让时序注意力看到 `N_train` 帧范围，避免超范围导致的注意力退化。  
- **motion injection**则是把“prompt 影响的语义层级”与“去噪阶段/网络层级”对齐：布局/外观尽量继承旧 prompt，动作/姿态在中期与解码端更强地受新 prompt 驱动。

## 关键原图讲解
说明：该 PDF 的部分 Figure 为矢量/组合对象，自动 `extract-images` 会把网格图拆成若干小块，且可能无法提取到“完整 Figure（含标题/标注/所有子图）”。下列图均为**从 PDF 中自动提取的子图块**，我会结合正文描述补足其语境，避免凭空脑补。

### 图 1：滑窗输出的平滑融合直觉（对应式 (9)）
![图：滑窗融合权重直觉](../figures/2024-freenoise-tuning-free-longer-video-diffusion-via-noise-rescheduling/page_005_01_img-005-056.png)
- 来源：PDF 第 5 页（Figure 3 附近的子图块，自动提取为局部示意）
- 展示内容：把重叠窗口输出按“离窗口中心越近权重越大”的方式融合（类似三角窗）
- 如何解读：如果直接平均，窗口边界处不同窗口的输出会产生突变；用中心加权可让边界过渡更平滑
- 与核心贡献关系：这是 Window-based Attention Fusion 能在“训练长度窗口”内计算注意力、又能拼接成长序列且保持平滑的关键

### 图 2：长视频定性对比（Figure 4 的一个子块：panda prompt）
![图：长视频定性样例（panda chef）](../figures/2024-freenoise-tuning-free-longer-video-diffusion-via-noise-rescheduling/page_007_01_img-007-079.jpg)
- 来源：PDF 第 7 页（Figure 4 的子图块之一）
- 展示内容：prompt 为“披萨+熊猫厨师+纽约街头餐车”的某帧（图中出现 watermark/伪影也可能反映模型训练数据偏好）
- 如何解读：论文的主张是：Direct（直接拉长）会崩，Sliding 会丢长程一致性，GenL 仍会变异；FreeNoise 借助噪声重排让远距离片段共享主体与场景基础，从而更一致
- 与核心贡献关系：直接体现“噪声重排 + 注意力滑窗融合”对长视频一致性的提升（定性层面）

### 图 3：多 prompt 生成样例（Figure 5 子块：camel）
![图：多 prompt 样例（camel in snow）](../figures/2024-freenoise-tuning-free-longer-video-diffusion-via-noise-rescheduling/page_008_01_img-008-105.jpg)
- 来源：PDF 第 8 页（Figure 5 的子图块之一）
- 展示内容：多 prompt 从“camel running on snow field”切到“camel standing on snow field”的某帧
- 如何解读：多 prompt 的难点是“换 prompt 会把主体/场景也换掉”；motion injection 的目标是让场景/主体尽量不变，主要改变动作/姿态并平滑过渡
- 与核心贡献关系：对应 Motion Injection（式 (11)(12)）解决 multi-prompt 的一致性问题

### 图 4：多 prompt 样例（Figure 5/7 的子块：astronaut + horse）
![图：多 prompt 样例（astronaut + horse）](../figures/2024-freenoise-tuning-free-longer-video-diffusion-via-noise-rescheduling/page_009_01_img-009-138.jpg)
- 来源：PDF 第 9 页（Figure 5 与 Figure 7 附近的子图块之一，自动提取无法保证它属于哪一行/哪一方法）
- 展示内容：宇航员与马的场景某帧（论文中用它讨论 multi-prompt 切换与 motion injection 的消融）
- 如何解读：文中消融指出：如果只让 P1 控制解码端/或控制范围不当，会出现“动作被抑制”或“外观变化明显/局部缺失”等问题；因此需要在层与时间步上做合理注入
- 与核心贡献关系：体现 motion injection 的设计空间（哪些层、哪些去噪步对动作/外观更敏感）

### 图 5：显著运动的三种模式（Figure 8 子块之一）
![图：显著运动案例（running subject）](../figures/2024-freenoise-tuning-free-longer-video-diffusion-via-noise-rescheduling/page_014_01_img-014-165.jpg)
- 来源：PDF 第 14 页（Figure 8 的子图块之一）
- 展示内容：显著运动（significant movement）下的一类视频模式示例帧
- 如何解读：附录把显著运动分成“镜头跟随主体 / 主体移出画面 / 主体在画面内移动”三类；并指出基座模型对后两类仍不自然，FreeNoise 的上限受基座限制
- 与核心贡献关系：明确 FreeNoise 的适用边界与失败模式，避免把一致性提升误解为“能产生任意长程位移/复杂运动”

## 实验与评价
### 设置
- 基座：VideoCrafter（论文中：训练 16 帧，推理到 64 帧）
- 默认：窗口 `U=16`，stride `S=4`
- 指标：FVD / KVD（越低越好），CLIP-SIM（越高越好，衡量相邻帧一致性），以及推理耗时

### 主要结果（Table 1）
在 64 帧推理任务上（基于文中表格数值）：
- Direct：FVD 737.61，KVD 359.11，CLIP-SIM 0.9104，耗时 21.97s
- Sliding：FVD 224.55，KVD 44.09，CLIP-SIM 0.9438，耗时 36.76s
- GenL：FVD 177.63，KVD 21.06，CLIP-SIM 0.9370，耗时 77.89s
- **FreeNoise**：**FVD 85.83，KVD 7.06，CLIP-SIM 0.9732，耗时 25.75s**

解读：
- Direct 主要死于训练-推理帧长鸿沟；
- Sliding 缓解鸿沟但牺牲长程一致性；
- GenL 有融合但仍有内容突变且时间开销很大；
- FreeNoise 以较小额外时间把一致性与质量都拉起来（文中强调约 17% 额外耗时）。

### 用户研究（Table 2）
三项主观维度（内容一致性 / 视频质量 / 文本对齐）里，FreeNoise 均获得最高被选比例（约 51%～58%）。

## 局限性
- 因为使用了重复且局部洗牌的噪声，随着视频变长，**引入“全新内容/大幅位移”会变弱**，主体位移可能被限制（附录 Limitation Discussion）。  
- 基座模型本身对“主体移出画面/主体在画面内大范围移动”就不擅长，FreeNoise 上限受基座能力约束。  
- 自动抽图可能不完整：若你需要对 Figure 3 总览或消融网格做精确逐格解读，建议直接打开 PDF 对照阅读。

## 与库中相关论文的关系
- 与无需训练的长视频推理扩展：[[literature/papers/2024-fifo-diffusion-generating-infinite-videos-from-text-without-training|FIFO-Diffusion]]（同为推理侧延长，但机制不同：FIFO 更偏“流式/窗口化生成到极长序列”，并系统讨论内存/速度）。  
- 与基座模型：[[literature/papers/2024-videocrafter2-overcoming-data-limitations-for-high-quality-video-diffusion-model|VideoCrafter2]]（数据与训练改进提升基础质量；FreeNoise 更像“在同一基座上做推理增强”）。  
- 与训练侧提升效率/分辨率：[[notes/2026-05-16_2025-pyramidal-flow-matching-for-efficient-video-generative-modeling|Pyramidal Flow Matching for Efficient Video Generative Modeling]]（训练目标与结构设计减少成本；FreeNoise 不改训练）。  
- 与 I2V / 运动控制：[[notes/2026-05-13_2025-livephoto-real-image-animation-with-text-guided-motion-control|LivePhoto]]（对“文本控运动”的歧义拆解；FreeNoise 的 motion injection 是 prompt 注入时序/层级控制的一种思路）。

## 后续阅读建议
1) 先读并对照：[[literature/papers/2024-fifo-diffusion-generating-infinite-videos-from-text-without-training|FIFO-Diffusion]]，比较两者对“极长序列、内存/速度、长程一致性”的取舍。  
2) 回到基座：[[literature/papers/2024-videocrafter2-overcoming-data-limitations-for-high-quality-video-diffusion-model|VideoCrafter2]]，理解训练/数据改进如何影响“推理侧延长”效果上限。  
3) 如果你关心“prompt 分段控制”，可继续沿着 motion control / prompt editing 系列（如 MasaCtrl、I2V 运动模块）整理一条“语义层级×去噪阶段×网络层”的统一视角。


















