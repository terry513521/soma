"""Thin local-test wrapper around `openclaw_algorithmic_base_miner.py`.

Contract:
    main(task: str, compression_ratio: float | None = None) -> str
"""

from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_openclaw_module():
    path = Path(__file__).with_name("openclaw_algorithmic_base_miner.py")
    spec = importlib.util.spec_from_file_location("openclaw_algorithmic_base_miner", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import algorithmic OpenClaw miner: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main(task: str, compression_ratio: float | None = None) -> str:
    ratio = 0.45 if compression_ratio is None else compression_ratio
    module = _load_openclaw_module()
    return module.compress_task_text(
        task,
        compression_ratio=ratio,
        plugin_dir=Path(__file__).resolve().parent,
    )
