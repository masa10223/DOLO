"""``python -m dolo.gui`` の入口。"""

from __future__ import annotations

import sys

from dolo.cli import main

if __name__ == "__main__":  # pragma: no cover
    sys.exit(main(["gui", *sys.argv[1:]]))
