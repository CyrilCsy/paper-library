# Scripts

`paper_manager.py` 是稳定的 Python CLI 入口，内部实现位于 `paperlib/cli.py`。

常用命令保持不变：

```powershell
python scripts/paper_manager.py scan
python scripts/paper_manager.py sync-obsidian
python scripts/paper_manager.py export-xlsx
```

`paper_manager.ps1` 是 PowerShell 备用实现，主要用于 Python 不可用的环境。`crop_image.ps1` 只负责图片裁剪，不参与论文索引同步。

新增脚本时优先遵守这个边界：

- 论文库状态和 Obsidian 同步逻辑放入 `paperlib/`。
- 一次性辅助工具可以放在 `scripts/` 根部。
- 不直接改写 `literature/` 的生成结果，除非对应逻辑也写入管理器。
