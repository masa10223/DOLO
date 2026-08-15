"""起動前診断。重いパッケージは import せず存在だけを調べる。"""

from __future__ import annotations

import importlib.util
import platform
import sys
from dataclasses import dataclass


@dataclass(frozen=True)
class RuntimeStatus:
    python_ok: bool
    nicegui: bool
    opencv: bool
    torch: bool
    ultralytics: bool
    ffmpeg: bool

    @property
    def inference_ready(self) -> bool:
        return self.python_ok and self.opencv and self.torch and self.ultralytics and self.ffmpeg

    @property
    def missing_inference(self) -> tuple[str, ...]:
        checks = {
            "opencv-python": self.opencv,
            "torch": self.torch,
            "ultralytics": self.ultralytics,
            "imageio-ffmpeg": self.ffmpeg,
        }
        return tuple(name for name, present in checks.items() if not present)


def inspect_runtime() -> RuntimeStatus:
    def has(name: str) -> bool:
        return importlib.util.find_spec(name) is not None

    return RuntimeStatus(
        python_ok=sys.version_info >= (3, 10),
        nicegui=has("nicegui"),
        opencv=has("cv2"),
        torch=has("torch"),
        ultralytics=has("ultralytics"),
        ffmpeg=has("imageio_ffmpeg"),
    )


def doctor_report(model=None, data_dir=None) -> tuple[str, bool]:
    from .config import GUIPaths, discover_default_model

    runtime = inspect_runtime()
    paths = GUIPaths.from_environment(data_dir).ensure()
    choice = discover_default_model(model, data_root=paths.root)
    rows = [
        "DOLO environment doctor",
        f"  OS             {platform.platform()}",
        f"  Python         {platform.python_version()}  {'OK' if runtime.python_ok else '3.10+ required'}",
        f"  NiceGUI        {'OK' if runtime.nicegui else 'missing (install dolo[gui])'}",
        f"  OpenCV         {'OK' if runtime.opencv else 'missing'}",
        f"  PyTorch        {'OK' if runtime.torch else 'missing'}",
        f"  Ultralytics    {'OK' if runtime.ultralytics else 'missing'}",
        f"  Video encoder  {'OK (imageio-ffmpeg)' if runtime.ffmpeg else 'missing (imageio-ffmpeg)'}",
        f"  Data directory {paths.root}",
        f"  Default model  {choice.path if choice.available else choice.warning}",
    ]
    ready = runtime.python_ok and runtime.nicegui and runtime.inference_ready and choice.available
    rows.append(f"  Result         {'READY' if ready else 'SETUP REQUIRED'}")
    return "\n".join(rows), ready
