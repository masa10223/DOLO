"""追跡のゴールデン回帰テスト。

`tests/data/` に動画・重み・ゴールデンCSVが揃っているときだけ実行される。
揃っていなければ skip されるので、CI や torch 無しの環境でも支障はない。

ゴールデンCSVの作り方は `tests/data/README.md` を参照。
生成には `tools/make_golden.py` が使える。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pd = pytest.importorskip("pandas")

DATA = Path(__file__).parent / "data"
VIDEO = DATA / "sample.mov"
MODEL = DATA / "model.pt"
GOLDEN = DATA / "golden_trajectory.csv"
META = DATA / "meta.json"

# 追跡が出力する CSV のスキーマ（列の順序も含めて固定する）
EXPECTED_COLUMNS = [
    "Frame",
    "ID",
    "Head_X",
    "Head_Y",
    "Middle_X",
    "Middle_Y",
    "Tail_X",
    "Tail_Y",
    "Angle",
    "DistMoved",
    "Confidence",
    "TimeSinceUpdate",
]

# GPU / cuDNN のバージョン差で下位ビットが揺れるため、完全一致ではなく許容誤差で比較する。
# ID の割り当てだけは完全一致を要求する（ここが変わったら本物の回帰）。
COORD_TOL = 0.5  # ピクセル
ANGLE_TOL = 1.0  # 度


def _missing():
    return [p.name for p in (VIDEO, MODEL, GOLDEN) if not p.exists()]


needs_data = pytest.mark.skipif(
    bool(_missing()),
    reason=f"tests/data/ に必要なファイルがありません: {_missing()}（tests/data/README.md 参照）",
)

needs_torch = pytest.mark.skipif(
    __import__("importlib").util.find_spec("ultralytics") is None,
    reason="ultralytics 未インストール（pip install 'dolo[torch]'）",
)


def _runtime_matches_golden() -> tuple[bool, str]:
    """ゴールデン生成時と推論ライブラリが一致するか。

    新しいtorchでも推論はできるが、NMSや下位ビットの差がID割り当てへ伝播するため、
    別バージョン同士のCSV比較は回帰テストにならない。
    """
    if not META.exists() or __import__("importlib").util.find_spec("ultralytics") is None:
        return False, "推論ランタイムまたはmeta.jsonがありません"
    import torch
    import ultralytics

    meta = json.loads(META.read_text())
    torch_version = torch.__version__.split("+")[0]
    expected_torch = meta.get("golden_torch_versions", [])
    expected_ultralytics = meta.get("golden_ultralytics_version")
    matches = torch_version in expected_torch and ultralytics.__version__ == expected_ultralytics
    reason = (
        f"ゴールデンは torch={expected_torch}, ultralytics={expected_ultralytics} 用。"
        f"現在は torch={torch_version}, ultralytics={ultralytics.__version__}"
    )
    return matches, reason


_matches_golden, _golden_runtime_reason = _runtime_matches_golden()
needs_golden_runtime = pytest.mark.skipif(not _matches_golden, reason=_golden_runtime_reason)


# --------------------------------------------------------------------------
# ゴールデンCSV そのものの健全性（torch 不要）
# --------------------------------------------------------------------------
@pytest.mark.skipif(not GOLDEN.exists(), reason="ゴールデンCSV未作成")
def test_golden_has_expected_schema():
    df = pd.read_csv(GOLDEN)
    assert list(df.columns) == EXPECTED_COLUMNS


@pytest.mark.skipif(not GOLDEN.exists(), reason="ゴールデンCSV未作成")
def test_golden_is_not_empty():
    assert len(pd.read_csv(GOLDEN)) > 0


@pytest.mark.skipif(not GOLDEN.exists(), reason="ゴールデンCSV未作成")
def test_golden_ids_are_within_max_id():
    df = pd.read_csv(GOLDEN)
    max_id = json.loads(META.read_text())["max_id"] if META.exists() else df["ID"].max()
    assert df["ID"].min() >= 1
    assert df["ID"].max() <= max_id


@pytest.mark.skipif(not GOLDEN.exists(), reason="ゴールデンCSV未作成")
def test_golden_has_one_row_per_id_per_frame():
    """同一フレームに同じ ID が2回出てはいけない。"""
    df = pd.read_csv(GOLDEN)
    assert not df.duplicated(subset=["Frame", "ID"]).any()


@pytest.mark.skipif(not GOLDEN.exists(), reason="ゴールデンCSV未作成")
def test_golden_angle_is_in_signed_range():
    """Angle 列は符号あり [-180, 180]。符号なしに変わったら気づけるようにする。"""
    df = pd.read_csv(GOLDEN)
    assert df["Angle"].between(-180.0, 180.0).all()


@pytest.mark.skipif(not GOLDEN.exists(), reason="ゴールデンCSV未作成")
def test_golden_coordinates_are_inside_the_frame():
    pytest.importorskip("cv2")
    import cv2

    cap = cv2.VideoCapture(str(VIDEO))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    df = pd.read_csv(GOLDEN)
    for axis, limit in (("X", width), ("Y", height)):
        for part in ("Head", "Middle", "Tail"):
            col = df[f"{part}_{axis}"]
            assert col.between(-limit * 0.1, limit * 1.1).all(), f"{part}_{axis} が画面外"


# --------------------------------------------------------------------------
# エンドツーエンド回帰（torch とデータの両方が必要）
# --------------------------------------------------------------------------
@needs_data
@needs_torch
@needs_golden_runtime
@pytest.mark.needs_weights
def test_tracking_output_matches_golden(tmp_path):
    """現行コードの出力がゴールデンから乖離していないことを確認する。

    リファクタで挙動が変わっていないことの最終的な保証。
    """
    from dolo.tracking import track_to_csv  # noqa: PLC0415

    meta = json.loads(META.read_text()) if META.exists() else {}
    out_csv = tmp_path / "trajectory.csv"

    track_to_csv(
        video_path=VIDEO,
        model_path=MODEL,
        output_csv_path=out_csv,
        max_ids=meta.get("max_id", 6),
        conf_thres=meta.get("conf", 1e-3),
        iou_thres=meta.get("iou_thr", 0.45),
        max_age=meta.get("max_missing_frames", 15),
        dist_thresh=meta.get("dist_thresh", 30),
        head_tail_jump_thresh=meta.get("head_tail_jump_thresh", 50),
        overlap_thresh=meta.get("overlap_thresh", 5),
        start_frame=meta.get("start_frame", 0),
        end_frame=meta.get("end_frame"),
        device=meta.get("device", "auto"),
    )

    got = pd.read_csv(out_csv)
    want = pd.read_csv(GOLDEN)

    assert list(got.columns) == list(want.columns), "列構成が変わった"
    assert len(got) == len(want), f"行数が変わった: {len(want)} → {len(got)}"

    # ID の割り当ては完全一致でなければならない
    pd.testing.assert_series_equal(
        got["ID"].reset_index(drop=True),
        want["ID"].reset_index(drop=True),
        check_names=False,
    )
    pd.testing.assert_series_equal(
        got["Frame"].reset_index(drop=True),
        want["Frame"].reset_index(drop=True),
        check_names=False,
    )

    for col in ("Head_X", "Head_Y", "Middle_X", "Middle_Y", "Tail_X", "Tail_Y", "DistMoved"):
        diff = got[col].to_numpy() - want[col].to_numpy()
        assert abs(diff).max() <= COORD_TOL, f"{col} が {abs(diff).max():.3f} px ずれた"

    angle_diff = got["Angle"].to_numpy() - want["Angle"].to_numpy()
    assert abs(angle_diff).max() <= ANGLE_TOL, f"Angle が {abs(angle_diff).max():.3f} 度ずれた"
