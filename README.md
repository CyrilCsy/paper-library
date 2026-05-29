# Paper Library 管理说明

这个目录是一个 Obsidian 友好的本地论文库，用来维护论文细分领域、阅读状态、重要程度，以及每日未读论文带图精讲笔记。

## 项目结构

- 核心资料: `papers/` 保存原始 PDF，`notes/` 保存精讲笔记。
- 基础知识: `knowledge/` 保存论文中反复出现的基础概念、方法和背景知识。
- 索引数据: `papers.csv` 是主索引表，`paper_library.xlsx` 是表格视图。
- Obsidian 输出: `literature/` 保存自动生成的论文页、索引页和图谱入口。
- 派生资源: `figures/` 保存论文图表和截图，`extracted_text/` 保存 PDF 文本抽取结果。
- 自动化: `scripts/paper_manager.py` 是稳定命令入口，内部实现位于 `scripts/paperlib/`；`scripts/paper_manager.ps1` 是备用实现。
- 项目文档: `docs/ARCHITECTURE.md` 说明目录职责、数据流和维护约定。
- 工作区配置: `.obsidian/` 保存 Obsidian vault 配置。

更完整的架构说明见 `docs/ARCHITECTURE.md`。

## 索引字段

- `subfield`: 细分领域。
- `importance`: 1 到 5，越高越重要。
- `read_status`: 阅读状态，默认 `unread`；读完后可改为 `read`。
- `note_status`: 精讲笔记状态，`generated` 表示已生成笔记。
- `note_path`: 对应笔记路径。
- `selected_on`: 被每日任务选中的日期。

## 基础知识库

`knowledge/` 用来沉淀跨论文复用的基础知识。它和 `notes/` 的区别是：

- `notes/`: 围绕单篇论文写精讲和阅读结论。
- `knowledge/`: 围绕一个概念写长期维护的解释、公式、误区、阅读路径和相关论文链接。

当前已有知识点：

- [[knowledge/flow-matching|Flow Matching]]

## 常用命令

扫描新增 PDF 并更新索引：

```powershell
python scripts/paper_manager.py scan
```

重新计算细分领域和重要程度：

```powershell
python scripts/paper_manager.py scan --refresh-classification
```

导出 Excel：

```powershell
python scripts/paper_manager.py export-xlsx
```

查看统计：

```powershell
python scripts/paper_manager.py stats
```

手动选择下一篇未读且未生成笔记的论文：

```powershell
python scripts/paper_manager.py pick --reserve
```

提取论文文本：

```powershell
python scripts/paper_manager.py extract --paper-id <paper_id> --output extracted_text/<paper_id>.txt
```

提取论文原图：

```powershell
python scripts/paper_manager.py extract-images --paper-id <paper_id>
```

同步 Obsidian 索引页和笔记 frontmatter：

```powershell
python scripts/paper_manager.py sync-obsidian
```

标记某篇论文已读：

```powershell
python scripts/paper_manager.py mark-read --paper-id <paper_id>
```

标记某篇论文已有笔记：

```powershell
python scripts/paper_manager.py mark-note --paper-id <paper_id> --note-path notes/<note_file>.md
```

## 带图精讲

每日自动任务会优先从 PDF 中提取原始嵌入图片，存入 `figures/<paper_id>/`，并在笔记中用 Markdown 引用关键图。

注意：部分论文图表是矢量绘制或由多个对象组合而成，PDF 内部不一定以单张图片保存。这种情况下脚本可能无法完整提取“整张 Figure”，自动笔记会说明缺失，并使用能提取到的原始图片辅助讲解。

## Obsidian 协同

可以把 `C:\Documents\Share\Paper Library` 直接作为 Obsidian vault 打开。

Codex 负责维护结构化数据和自动生成笔记，Obsidian 负责阅读、双链、标签、搜索和图谱视图。`sync-obsidian` 会生成：

- `literature/index.md`: 论文库入口。
- `literature/unread.md`: 未读论文列表。
- `literature/read.md`: 已读论文列表。
- `literature/high-priority.md`: 高优先级论文列表。
- `literature/notes.md`: 已生成精讲笔记列表。
- `literature/papers/*.md`: 每篇论文的 Obsidian 页面，包含元数据、PDF、笔记和精选相关论文。
- `literature/maps/*.md`: 面向关系图谱的地图页，包括总览、跨主题桥接和重点阅读路线。
- `literature/fields/*.md`: 按细分领域拆分的索引页。
- `literature/topics/*.md`: 按主题关键词拆分的横向索引页。
- `literature/years/*.md`、`literature/venues/*.md`: 年份和会场索引页，主要用于检索，不默认进入全局关系图谱。
- `literature/dataview.md`: Dataview 插件查询示例。

每篇论文页和精讲笔记会带 YAML frontmatter，包含 `aliases`、`paper_id`、`title`、`year`、`venue`、`subfield`、`topics`、`importance`、`read_status`、`note_status`、`pdf` 和 `tags`。这样 Obsidian 可以用属性、标签和 Dataview 查询组织论文库。

## Obsidian 关系图谱

默认全局图谱配置在 `.obsidian/graph.json`。当前策略是保留真正表达语义关系的节点，隐藏 `index`、`overview`、`unread`、`read`、`high-priority`、`notes` 列表、年份、会场、PDF、图片、提取文本和脚本文档等噪声节点。

默认图谱会呈现四类主要节点：

1. `literature/papers`: 论文页，显示精选相关论文边。
2. `literature/fields` 与 `literature/topics`: 领域负责纵向分层，主题负责横向连接。
3. `notes`: 精讲笔记，作为人工阅读沉淀连接到对应论文。
4. `knowledge`: 基础知识详解，连接概念、论文、主题和精讲笔记。

颜色分组也在 `.obsidian/graph.json` 中配置：论文页为蓝色，领域为绿色，主题为橙色，精讲笔记为紫色，基础知识为红色。`literature/maps`、`literature/index.md`、`literature/unread.md`、`knowledge/index.md` 等页面仍然保留用于阅读和导航，只是不参与默认全局关系图谱。

如果你在 Obsidian 的笔记属性里修改了 `read_status`、`importance`、`subfield`、`venue`、`selected_on` 或 `last_reviewed_on`，运行 `sync-obsidian` 会先把这些修改回写到 `papers.csv`，再刷新所有索引页。

## 使用建议

1. 平时主要维护 `papers.csv`、`paper_library.xlsx` 或 Obsidian 笔记属性。
2. 读完论文后，把 `read_status` 改成 `read`，或运行 `mark-read` 命令。
3. `note_status` 和 `read_status` 是分开的：生成精讲笔记不代表你本人已经读完。
4. 新增 PDF 时放入 `papers/`，再运行 `scan`、`sync-obsidian` 和 `export-xlsx`。
5. 如果你手动修改了 `subfield` 或 `importance`，普通 `scan` 会保留你的修改；只有 `scan --refresh-classification` 会覆盖分类和重要程度。
