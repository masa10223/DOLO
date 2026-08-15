"""DOLO — Drosophila tracking with YOLO Pose.

このトップレベルは **意図的に軽く保たれている**。torch / ultralytics /
opencv を import しないので、``import dolo`` は一瞬で終わり、GPU の無い環境
でも失敗しない。重い依存が要るものは明示的にサブモジュールから読むこと::

    from dolo.tracker import FixedIDTracker      # numpy + scipy のみ
    from dolo.tracking import track_video_legacy # ultralytics が必要
"""

from __future__ import annotations

__version__ = "0.1.0.dev0"

__all__ = ["__version__"]
