---
type: paper-library-help
generated_on: 2026-05-29
tags:
  - paper-library
---
# Dataview 查询示例

如果 Obsidian 安装了 Dataview 插件，可以使用下面的查询。

## 未读论文笔记

```dataview
TABLE year, venue, subfield, importance, pdf
FROM "literature/papers"
WHERE type = "paper" AND read_status != "read"
SORT importance DESC, year DESC
```

## 视频生成方向

```dataview
TABLE year, venue, importance, read_status
FROM "literature/papers"
WHERE type = "paper" AND subfield = "Video Generation"
SORT year DESC
```

