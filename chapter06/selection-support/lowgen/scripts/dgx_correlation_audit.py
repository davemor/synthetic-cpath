#!/usr/bin/env python
from __future__ import annotations

import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root / "src"))

from lowgen.__main__ import main


if __name__ == "__main__":
    sys.argv = [sys.argv[0], "dgx", "correlation-audit", *sys.argv[1:]]
    main()
