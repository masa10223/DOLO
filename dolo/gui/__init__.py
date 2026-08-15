"""DOLO のブラウザ GUI。

NiceGUI は optional dependency なので、このモジュールを import しただけでは
NiceGUI・torch・OpenCV を読み込まない。GUI を起動するときだけ
``dolo.gui.app`` がそれらを遅延 import する。
"""

from __future__ import annotations

from .config import GUIPaths, ModelChoice, discover_default_model
from .jobs import InferenceConfig, JobManager, JobSnapshot, JobState

__all__ = [
    "GUIPaths",
    "InferenceConfig",
    "JobManager",
    "JobSnapshot",
    "JobState",
    "ModelChoice",
    "discover_default_model",
]
