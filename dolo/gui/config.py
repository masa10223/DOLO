"""GUI のパス解決と default モデル検出。

環境固有の絶対パスをアプリ本体へ書かないための境界。OSS 利用者は環境変数で
上書きでき、リポジトリ内で開発している間はルートの ``best.pt`` をそのまま使える。
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

VIDEO_SUFFIXES = frozenset({".avi", ".m4v", ".mkv", ".mov", ".mp4", ".mpeg", ".mpg", ".webm"})
MODEL_SUFFIXES = frozenset({".pt", ".onnx", ".engine"})


def project_root() -> Path:
    """ソース checkout または editable install のプロジェクトルート。"""
    return Path(__file__).resolve().parents[2]


def safe_filename(name: str, fallback: str = "upload") -> str:
    """クライアント由来の名前を単一の安全なファイル名へ変換する。

    Windows の ``\\`` も区切りとして扱い、パストラバーサルを防ぐ。日本語などの
    Unicode 文字は保持する。
    """
    leaf = Path(str(name).replace("\\", "/")).name
    leaf = re.sub(r"[^\w.()-]+", "_", leaf, flags=re.UNICODE).strip(" ._")
    if not leaf or leaf in {".", ".."}:
        return fallback
    return leaf[:180]


@dataclass(frozen=True)
class GUIPaths:
    """GUI が管理する書き込み先。"""

    root: Path
    uploads: Path
    runs: Path
    thumbnails: Path

    @classmethod
    def from_environment(
        cls,
        data_dir: str | Path | None = None,
        *,
        root: str | Path | None = None,
    ) -> GUIPaths:
        repo = Path(root).expanduser().resolve() if root else project_root()
        configured = data_dir or os.environ.get("DOLO_DATA_DIR")
        base = Path(configured).expanduser() if configured else repo / ".dolo"
        base = base.resolve()
        return cls(
            root=base,
            uploads=base / "uploads",
            runs=base / "runs",
            thumbnails=base / "thumbnails",
        )

    def ensure(self) -> GUIPaths:
        for path in (self.root, self.uploads, self.runs, self.thumbnails):
            path.mkdir(parents=True, exist_ok=True)
        return self


@dataclass(frozen=True)
class ModelChoice:
    """自動検出したモデルと、その採用理由。"""

    path: Path | None
    source: str
    searched: tuple[Path, ...]
    warning: str | None = None

    @property
    def available(self) -> bool:
        return self.path is not None and self.path.is_file()


def _model_candidates(repo: Path, data_root: Path | None) -> list[Path]:
    candidates = [
        repo / "best.pt",
        repo / "models" / "default.pt",
        repo / "models" / "best.pt",
        repo / "scripts" / "runs" / "pose" / "train" / "weights" / "best.pt",
    ]
    if data_root is not None:
        candidates += [data_root / "models" / "default.pt", data_root / "models" / "best.pt"]
    candidates.append(Path.home() / ".cache" / "dolo" / "models" / "default.pt")
    return candidates


def discover_default_model(
    explicit: str | Path | None = None,
    *,
    root: str | Path | None = None,
    data_root: str | Path | None = None,
) -> ModelChoice:
    """default 重みを決定する。

    優先順位は明示引数、``DOLO_MODEL_PATH``、既知のローカル候補。明示されたパスが
    間違っている場合に別モデルへ黙って切り替えると再現性を損なうため、その場合は
    warning 付きで未解決として返す。
    """
    repo = Path(root).expanduser().resolve() if root else project_root()
    data = Path(data_root).expanduser().resolve() if data_root else None
    candidates = _model_candidates(repo, data)

    requested = explicit or os.environ.get("DOLO_MODEL_PATH")
    if requested:
        path = Path(requested).expanduser().resolve()
        source = "--model" if explicit else "DOLO_MODEL_PATH"
        if path.is_file():
            return ModelChoice(path=path, source=source, searched=(path,))
        return ModelChoice(
            path=None,
            source=source,
            searched=(path,),
            warning=f"{source} で指定されたモデルが見つかりません: {path}",
        )

    for candidate in candidates:
        resolved = candidate.expanduser().resolve()
        if resolved.is_file():
            return ModelChoice(path=resolved, source="auto", searched=tuple(candidates))

    return ModelChoice(
        path=None,
        source="auto",
        searched=tuple(candidates),
        warning="default モデルが見つかりません。モデルパスを指定してください。",
    )


def validate_video_path(path: str | Path) -> Path:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"動画が見つかりません: {resolved}")
    if resolved.suffix.lower() not in VIDEO_SUFFIXES:
        allowed = ", ".join(sorted(VIDEO_SUFFIXES))
        raise ValueError(
            f"対応していない動画形式です: {resolved.suffix or '(拡張子なし)'}（{allowed}）"
        )
    return resolved


def validate_model_path(path: str | Path) -> Path:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"モデル重みが見つかりません: {resolved}")
    if resolved.suffix.lower() not in MODEL_SUFFIXES:
        allowed = ", ".join(sorted(MODEL_SUFFIXES))
        raise ValueError(
            f"対応していないモデル形式です: {resolved.suffix or '(拡張子なし)'}（{allowed}）"
        )
    return resolved
