"""NiceGUIのアップロードオブジェクトを安全に保存する小さな境界。"""

from __future__ import annotations

import inspect
import uuid
from pathlib import Path

from .config import safe_filename


async def save_uploaded_video(file, upload_dir: str | Path) -> Path:
    """アップロードされた動画を衝突しない安全な名前で保存する。

    NiceGUI 3 の ``FileUpload.save`` はasyncだが、古い実装やテストダブルの同期saveも
    扱えるよう、戻り値がawaitableのときだけ待機する。
    """
    original = safe_filename(getattr(file, "name", "video.mov"), "video.mov")
    suffix = Path(original).suffix.lower() or ".mov"
    stem = Path(original).stem
    destination = Path(upload_dir) / f"{uuid.uuid4().hex[:10]}-{stem}{suffix}"
    destination.parent.mkdir(parents=True, exist_ok=True)
    saved = file.save(destination)
    if inspect.isawaitable(saved):
        await saved
    if not destination.is_file():
        raise OSError(f"アップロードを保存できませんでした: {destination}")
    return destination
