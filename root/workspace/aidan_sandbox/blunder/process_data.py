#!/usr/bin/env python3
"""CLI entry point for extracting Lichess eval-game features.

This file can be run directly from the project root without installing the
package first, because it adds ``src`` to ``sys.path``.
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from chess_eval_features.cli import main  # noqa: E402


if __name__ == "__main__":
  main()
