from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

try:
    from shejianplus.main import main
except ModuleNotFoundError as exc:
    if exc.name and exc.name.startswith("PySide6"):
        print("缺少依赖 PySide6，请先执行: pip install -r requirements.txt")
        raise SystemExit(1)
    raise


if __name__ == "__main__":
    raise SystemExit(main())
