from __future__ import annotations

from pathlib import Path


def load_text_file(path: str | Path) -> str:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"找不到文件: {file_path}")
    return file_path.read_text(encoding="utf-8")
