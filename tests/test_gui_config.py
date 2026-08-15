from __future__ import annotations

import asyncio

import pytest

from dolo.gui.config import GUIPaths, discover_default_model, safe_filename, validate_model_path
from dolo.gui.uploads import save_uploaded_video


@pytest.mark.parametrize(
    "name,expected",
    [
        ("movie.mov", "movie.mov"),
        ("../../best.pt", "best.pt"),
        (r"..\..\escape.mov", "escape.mov"),
        ("実験 01.mov", "実験_01.mov"),
        ("...", "upload"),
    ],
)
def test_safe_filename(name, expected):
    assert safe_filename(name) == expected


def test_gui_paths_can_be_relocated(tmp_path):
    paths = GUIPaths.from_environment(tmp_path / "data", root=tmp_path).ensure()
    assert paths.root == (tmp_path / "data").resolve()
    assert paths.uploads.is_dir()
    assert paths.runs.is_dir()
    assert paths.thumbnails.is_dir()


def test_repository_best_pt_is_the_auto_default(tmp_path):
    model = tmp_path / "best.pt"
    model.write_bytes(b"weights")
    choice = discover_default_model(root=tmp_path)
    assert choice.available
    assert choice.path == model.resolve()
    assert choice.source == "auto"


def test_explicit_missing_model_does_not_silently_fallback(tmp_path):
    (tmp_path / "best.pt").write_bytes(b"weights")
    choice = discover_default_model(tmp_path / "missing.pt", root=tmp_path)
    assert not choice.available
    assert "missing.pt" in (choice.warning or "")


def test_environment_model_takes_priority(tmp_path, monkeypatch):
    environment_model = tmp_path / "env.pt"
    environment_model.write_bytes(b"weights")
    (tmp_path / "best.pt").write_bytes(b"other")
    monkeypatch.setenv("DOLO_MODEL_PATH", str(environment_model))
    choice = discover_default_model(root=tmp_path)
    assert choice.path == environment_model.resolve()
    assert choice.source == "DOLO_MODEL_PATH"


def test_model_extension_is_validated(tmp_path):
    path = tmp_path / "weights.txt"
    path.write_text("no")
    with pytest.raises(ValueError, match="モデル形式"):
        validate_model_path(path)


def test_uploaded_video_is_saved_with_safe_unique_name(tmp_path):
    class FakeUpload:
        name = r"..\..\experiment 01.mov"

        async def save(self, path):
            path.write_bytes(b"video-data")

    saved = asyncio.run(save_uploaded_video(FakeUpload(), tmp_path))
    assert saved.parent == tmp_path
    assert saved.name.endswith("-experiment_01.mov")
    assert saved.read_bytes() == b"video-data"
