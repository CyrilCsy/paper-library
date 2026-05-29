---
type: paper-note
aliases:
  - "T2V-CompBench A Comprehensive Benchmark for Compositional Text-to-video Generation"
paper_id: "2025-t2v-compbench-a-comprehensive-benchmark-for-compositional-text-to-video-generati"
title: "T2V-CompBench A Comprehensive Benchmark for Compositional Text-to-video Generation"
year: 2025
venue: "CVPR"
subfield: "Benchmark / Evaluation"
topics:
  - "benchmark"
  - "text-to-video"
importance: 5
read_status: "read"
note_status: "generated"
selected_on: ""
last_reviewed_on: "2026-05-15"
paper: "[[literature/papers/2025-t2v-compbench-a-comprehensive-benchmark-for-compositional-text-to-video-generati]]"
pdf: "[[papers/2025_T2V-CompBench A Comprehensive Benchmark for Compositional Text-to-video Generation_CVPR.pdf]]"
tags:
  - paper
  - paper/benchmark_evaluation
  - topic/benchmark
  - topic/text_to_video
  - venue/cvpr
  - year/2025
---
# 论文信息

- 标题：T2V-CompBench: A Comprehensive Benchmark for Compositional Text-to-video Generation
- 作者：Kaiyue Sun, Kaiyi Huang, Xian Liu, Yue Wu, Zihan Xu, Zhenguo Li, Xihui Liu
- 机构：The University of Hong Kong, The Chinese University of Hong Kong, Huawei Noah's Ark Lab
- 年份/会议：CVPR 2025
- arXiv：2407.14505
- PDF：[[papers/2025_T2V-CompBench A Comprehensive Benchmark for Compositional Text-to-video Generation_CVPR.pdf]]
- Project：<https://t2v-compbench-2025.github.io/>
- Code：<https://github.com/KaiyueSun98/T2V-CompBench>
- 相关库内论文：[[literature/papers/2024-vbench-comprehensive-benchmark-suite-for-video-generative-models]]、[[literature/papers/2024-videotetris-towards-compositional-text-to-video-generation]]、[[literature/papers/2025-cogvideox-text-to-video-diffusion-models-with-an-expert-transformer]]、[[literature/papers/2025-open-sora-2-0-training-a-commercial-level-video-generation-model-in-200k]]

# 细分领域与重要程度

- 细分领域：Benchmark / Evaluation
- 重要程度：5/5
- 原因：这是视频生成里专门面向组合性（composition）的系统 benchmark。它不只问“视频好不好看、动不动”，而是问模型能不能把多个对象、属性、动作、空间关系、运动方向和数量正确绑定到一起。

# 一句话总结

T2V-CompBench 把文本到视频生成的“组合理解”拆成 7 类能力、1400 条 prompts，并针对不同类别设计 MLLM、检测和跟踪三类评测指标；实验说明当前 T2V 模型在动态属性、空间关系、运动方向和数量上仍然很弱，常规 CLIP/FVD 这类指标不足以衡量这些失败模式。

# 背景问题

已有视频生成评测通常关注三件事：

- 视觉质量：画面清晰度、美学、artifact。
- 运动质量：有没有明显运动、是否稳定。
- 粗粒度 text-video alignment：整体是否像 prompt。

但真实 prompt 往往不是单对象描述，而是多对象、多属性、多动作、多关系的组合。例如“一个蓝色汽车经过白色栅栏”“狗在田野里跑，同时猫在爬树”“机器人向左走，而背景里的车向右行驶”。这些 prompt 的关键不是“生成了车/狗/猫”这么简单，而是要把**属性绑定到正确对象**、把**动作绑定到正确对象**、把**方向绑定到正确实体**，还要在时间维度上保持变化过程。

这篇论文的切入点就是：视频生成模型已经有很多 benchmark，但组合性 T2V 还缺一个系统、细粒度、带自动评测指标且经过人评验证的 benchmark。

# 核心贡献

1. **提出 T2V-CompBench**：7 个组合性类别，每类 200 条 prompt，总共 1400 条。
2. **把 T2V 组合性分成空间维度和时间维度**：空间侧继承 T2I 组合评测里的属性、空间关系、数量；时间侧新增动态属性、运动绑定、动作绑定、对象交互。
3. **为不同类别设计不同指标**：不是用一个通用 CLIPScore 打天下，而是按问题类型选择 MLLM、GroundingDINO/SAM/Depth Anything、DOT tracking 等工具。
4. **用人类标注验证自动指标**：651 个视频做人评，对自动指标与人评分数计算 Kendall tau 和 Spearman rho。
5. **评测 23 个 T2V 模型**：17 个开源/公开模型 + 6 个商业模型，并给出按类别的失败模式分析。

# 总览图讲解

![T2V-CompBench 总览](../figures/2025-t2v-compbench-a-comprehensive-benchmark-for-compositional-text-to-video-generati/fig1_overview.png)

这张图把整篇论文压缩成三块：

- 左边是 prompt suite：七个能力类别，每类还有子类，比如 consistent attribute binding 下分 color/shape/texture/human，spatial relationships 下分 left/right/above/below/in front/behind。
- 中间是 evaluation metrics：MLLM-based、Detection-based、Tracking-based 三种评测路线。
- 右边是模型雷达图：不同模型在七类组合能力上表现不均衡，说明“一个总分”很容易掩盖具体短板。

直觉上，这篇论文的关键不是“又做了一个 benchmark”，而是把 T2V 组合失败拆到足够细，让模型改进有明确靶点。

# Benchmark 怎么构造

![Prompt 类别与生成流程](../figures/2025-t2v-compbench-a-comprehensive-benchmark-for-compositional-text-to-video-generati/fig2_prompt_categories.png)

作者先从真实用户 prompt 里抽词，再构造包含两个对象、动作动词和目标组合关系的 prompt。为了避免 benchmark 变成静态图片评测，所有 prompt 都要求至少有一个 active verb。

七类任务可以按“检查什么”来理解：

| 类别 | 要检查的能力 | 典型失败 |
|---|---|---|
| Consistent attribute binding | 两个对象各自绑定固定属性 | 蓝车/白栅栏变成白车/蓝栅栏 |
| Dynamic attribute binding | 属性随时间变化 | 叶子应该由绿变棕，但整段视频都不变 |
| Spatial relationships | 两个对象的空间关系 | left/right、above/below 混淆 |
| Motion binding | 对象运动方向 | prompt 说向左，生成物体不动或向右 |
| Action binding | 不同对象执行各自动作 | 狗跑、猫爬树变成两者都跑 |
| Object interactions | 物理/社交交互过程 | 两车碰撞只生成并排静止车 |
| Generative numeracy | 对象数量 | 数量大于 3 后常常数不准 |

这里的一个设计点很重要：T2V-CompBench 不是直接把 T2I-CompBench 搬到视频上，而是加入 temporal dynamics。视频的组合性不仅要求每帧“东西对”，还要求变化过程、运动方向、交互过程对。

# Prompt Suite 的来源

作者分析了 VidProM 中来自 Pika Discord 的 167 万条去重 T2V prompt，用 WordNet 统计高频名词和动词的 metaclass。筛选原则有两个：

- 选真实用户常提到的对象、动作和属性，避免 benchmark 脱离实际使用场景。
- 选可检测的 thing 类对象，比如 car、dog，少用 sky 这类边界不清的 stuff 类概念，方便自动评测。

最后得到大约 260 个对象名词、200 个 active verbs、80 个属性词，再组合生成 7 类 prompt。每类 200 条，其中不少类别还保留一部分 uncommon/challenging prompt，用来测试泛化而不是只测常识搭配。

# 评测指标详解

这篇的评测设计可以概括为一句话：**不要期待一个通用语义相似度指标解决所有组合问题。**

## 1) MLLM-based metrics

用于 consistent attribute binding、dynamic attribute binding、action binding、object interactions。

### Grid-LLaVA

作者把视频均匀采样 6 帧，拼成 image grid 输入 LLaVA。然后先让 MLLM 描述视频，再让它按拆解后的问题打分。

它主要用于：

- consistent attribute binding：GPT-4 先把 prompt 拆成对象-属性短语，例如 “a blue car” 和 “a white picket fence”，LLaVA 再检查每个短语是否在视频网格里成立。
- action binding：GPT-4 拆出对象和动作，例如 “a dog runs through a field” 与 “a cat climbs a tree”，LLaVA 检查对象是否存在、动作是否绑定正确。
- object interactions：先判断对象是否存在，再判断交互过程是否发生、发展是否符合 prompt。

为什么不用单帧 LLaVA？因为 action 和 interaction 经常要看多帧才能判断。Grid-LLaVA 的优势是同时看到多帧，能捕捉一定时间变化。

### D-LLaVA

用于 dynamic attribute binding。作者发现视频 LLM 对“属性变化”判断不好，于是改成逐帧评估：

- GPT-4 解析初始状态和最终状态，比如 “bright green leaf” 和 “brown leaf”。
- LLaVA 给每一帧分别打初始状态/最终状态的匹配分。
- 评分函数鼓励前几帧接近初始状态、后几帧接近最终状态，中间帧处于过渡。

这很贴合动态属性的本质：不是看某一帧有没有目标属性，而是看**变化轨迹**是否对。

## 2) Detection-based metrics

用于 spatial relationships 和 generative numeracy。

### Spatial relationships

2D 关系（left/right/above/below）用 GroundingDINO 检测目标框，再用中心点坐标写规则。例如 A 在 B 左边需要满足：

$$
x_A < x_B \quad \text{且} \quad |x_A - x_B| > |y_A - y_B|
$$

3D 关系（in front of/behind）不能只靠 2D 框，所以作者在 GroundingDINO 检测框基础上引入 SAM 分割 mask，再用 Depth Anything 估计深度，通过相对深度判断前后关系。

### Generative numeracy

数量评测直接数每一帧检测到的目标类别数量。如果检测数量与 prompt 中数量一致，该对象类别得 1，否则得 0；帧级分数再平均成视频级分数。

这个指标虽然简单，但很有针对性：数量失败不是 CLIP 分数能稳定捕捉的，必须显式数对象。

## 3) Tracking-based metrics

用于 motion binding。

运动方向不能只看目标框起点和终点，因为镜头运动会干扰。作者用 GroundingSAM 分出前景对象和背景，再用 DOT 分别跟踪前景点和背景点。实际对象运动向量定义为：

$$
v_{\text{actual}} = v_{\text{foreground}} - v_{\text{background}}
$$

这样可以把相机运动扣掉，更接近“对象相对背景到底往哪儿动”。最终分数检查这个方向是否符合 prompt 里的 left/right/up/down。

# 人评相关性：为什么这些指标可信

![自动指标与人评相关性](../figures/2025-t2v-compbench-a-comprehensive-benchmark-for-compositional-text-to-video-generati/table1_human_correlation.png)

作者随机抽每类 15 条 prompt，用 6 个 T2V 模型生成 90 个视频；dynamic attribute 和 object interaction 额外加入 ground-truth video，总人评视频数为 651。每个 text-video pair 由 3 个 MTurk 标注者评分，再和自动指标算 Kendall tau 与 Spearman rho。

关键结论：

- Grid-LLaVA 在 consistent attribute、action、interaction 上最好。
- D-LLaVA 在 dynamic attribute 上最好。
- G-Dino 在 spatial relationship 和 numeracy 上最好。
- DOT 在 motion binding 上最好。
- 常规 CLIP、BLIP-CLIP、BLIP-BLEU、ViCLIP 等指标在很多组合类别上相关性明显不足。

这说明自动评测不是“越通用越好”。组合能力越细，越需要针对类别设计 evaluator。

# 模型评测结果怎么读

![模型评测结果](../figures/2025-t2v-compbench-a-comprehensive-benchmark-for-compositional-text-to-video-generati/table2_benchmark_results.png)

表 2 的分数都归一化到 0 到 1，越高越好。按类别看，比看平均分更有意义。

几个重点：

- **商业模型总体领先**：PixVerse-V3 在 consistent attribute、spatial、motion、action、interaction、numeracy 上都很强；Gen-3 在 dynamic attribute 上得分最高。
- **DiT-based 开源模型有进展**：CogVideoX-5B 在 motion、action、interaction 上相对强；Mochi 在 dynamic attribute 和 spatial 上表现突出。
- **VideoCrafter2 系列适配模型常有提升**：VideoTetris、Vico、T2V-Turbo-V2 在多个类别上比基础模型更好，说明数据/训练策略/结构改造会实际影响组合能力。
- **动态属性整体分数极低**：即使最好模型也只有 0.0687 级别，说明“属性随时间变化”仍是当前 T2V 的硬短板。
- **运动绑定分数也偏低**：多数模型难以稳定按 prompt 方向移动对象，尤其镜头运动和对象运动混在一起时更明显。

读表时要避免一个误区：某模型在视觉质量上强，不代表组合性强。T2V-CompBench 的价值就是把“看起来不错但没按 prompt 做”的失败暴露出来。

# 论文给出的三条模型洞察

## 1) T2V 模型在进化，但能力不是均匀进化

早期模型更偏单帧视觉质量；后来的模型开始更重视跨帧动态和运动质量。比如 CogVideoX-5B、T2V-Turbo-V2 在 motion binding 上相对更好，但这不等于它们在 dynamic attribute 或 numeracy 上同样强。

## 2) 动态属性是最难类别

很多模型会抓住 prompt 里的关键词，但忽略“变化”这个要求。比如 prompt 要求叶子从绿色变棕色，模型可能只生成一片绿色叶子，或者直接生成棕色叶子，但没有过渡过程。

这是视频生成相对图像生成的关键难点：正确的最终状态不够，**变化路径**也要正确。

## 3) 空间、运动、数量仍然薄弱

模型常常混淆 left/right，运动方向也经常不对；数量在小于 3 时还可以，一旦数量更大就容易失败。作者认为这需要更细致的 caption、专门的数据，以及更可控的 motion module。

# 这篇论文的真正价值

我认为它最大的价值不在 benchmark 排名，而在评测问题的拆解方式：

- 把组合性从“一个模糊概念”拆成 7 个类别。
- 对每个类别选合适 evaluator，而不是依赖单一多模态相似度。
- 用人评相关性证明 evaluator 的合理性。
- 把模型失败归因到具体能力短板。

这对后续读视频生成论文很有用。凡是论文声称 “better text alignment” 或 “better compositional generation”，都可以追问：它到底改善的是属性绑定、动作绑定、空间关系、运动方向、交互，还是数量？有没有分项指标？有没有人评相关性？

# 局限性与读者需要警惕的点

- **自动指标仍依赖外部模型能力**：GroundingDINO、SAM、Depth Anything、LLaVA、DOT 都会出错，因此 benchmark 分数包含 evaluator bias。
- **Prompt 覆盖有限**：1400 条 prompt 覆盖了核心组合类型，但真实用户 prompt 的语义复杂度更高，比如因果、长程事件、多镜头叙事、风格与动作耦合。
- **视频时长较短**：论文主要面向短视频生成，长视频里的跨场景一致性、剧情组合、事件顺序不在核心评测范围内。
- **MLLM 评测成本较高**：如果要大规模跑 Grid-LLaVA/D-LLaVA，成本和环境配置会比 CLIP/FVD 复杂。
- **数值结果会随模型版本快速变化**：商业模型和开源模型迭代很快，表 2 更适合作为方法学参考，而不是长期静态排行榜。

# 和库中已有论文的关系

- [[literature/papers/2024-vbench-comprehensive-benchmark-suite-for-video-generative-models]]：VBench 更像通用视频生成质量/运动/text alignment 评测；T2V-CompBench 专门补组合性短板。
- [[literature/papers/2024-videotetris-towards-compositional-text-to-video-generation]]：VideoTetris 是提升组合 T2V 的方法论文；T2V-CompBench 是评测这种能力的工具。
- [[literature/papers/2025-cogvideox-text-to-video-diffusion-models-with-an-expert-transformer]]：CogVideoX-5B 在本文表 2 中属于 DiT-based 模型，多个组合类别表现相对较好，可结合模型结构读。
- [[literature/papers/2025-open-sora-2-0-training-a-commercial-level-video-generation-model-in-200k]]：Open-Sora 类系统论文关注训练、数据、架构和成本；T2V-CompBench 可作为评估这类系统 prompt-following 细粒度能力的补充。

# 后续阅读建议

1. 先和 [[literature/papers/2024-vbench-comprehensive-benchmark-suite-for-video-generative-models]] 对照：理解通用视频质量 benchmark 与组合性 benchmark 的边界。
2. 再读 [[literature/papers/2024-videotetris-towards-compositional-text-to-video-generation]]：看方法论文如何试图解决组合生成问题。
3. 如果要复现实验，优先看官方 GitHub 的 `prompts/`、`meta_data/` 和 7 个类别的 evaluation scripts；不要只看论文表格。
4. 如果要设计自己的 T2V benchmark，可以直接借鉴它的“按能力类别拆分 + 为类别定制 evaluator + 做人评相关性验证”框架。




