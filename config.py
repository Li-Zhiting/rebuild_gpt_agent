from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
EVAL_DIR = BASE_DIR / "eval"


@dataclass(slots=True)
class Settings:
    max_context_chars: int = 12000
    chunk_size: int = 1200
    chunk_overlap: int = 150
    history_turn_limit: int = 6
    benchmark_path: Path = EVAL_DIR / "benchmark.json"


settings = Settings()
