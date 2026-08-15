# テスト用データ

このフォルダには、追跡のゴールデン回帰テスト（`tests/test_regression_tracking.py`）に
使う実データを置く。**中身は git に入らない**（`.gitignore` 済み）。

ファイルが無い場合、該当テストは自動的に skip される。ユニットテスト
（`test_geometry.py` / `test_kalman.py` / `test_tracker.py` / `test_legacy_equivalence.py`）は
これらが無くても動く。

## 置くファイル

| ファイル名 | 内容 |
|---|---|
| `sample.avi` | 数百フレームの短い動画。個体数が既知のもの |
| `model.pt` | その動画に対して十分な性能が出る学習済み重み（`best.pt` をリネーム） |
| `golden_trajectory.csv` | 上記2つを現行コードで処理して得た追跡結果。**これが正解** |
| `meta.json` | 実行時のパラメータ（下記） |

## `meta.json` の例

```json
{
  "max_id": 6,
  "conf": 0.001,
  "iou_thr": 0.45,
  "max_missing_frames": 15,
  "dist_thresh": 30,
  "head_tail_jump_thresh": 50,
  "overlap_thresh": 5,
  "start_frame": 0,
  "end_frame": 300,
  "device": "cuda:0"
}
```

## ゴールデンファイルの作り方

**リファクタを進める前の**現行コードで生成すること。これが「壊していないこと」の基準になる。

```bash
cd scripts
python3 drosophila_predict.py \
  --video_path      ../tests/data/sample.avi \
  --model_path      ../tests/data/model.pt \
  --output_csv_path ../tests/data/golden_trajectory.csv \
  --output_gif_path /tmp/throwaway.gif \
  --output_mov_path /tmp/throwaway.mov \
  --start_frame 0 --end_frame 300 --max_id 6
```

## 注意: 再現性について

YOLO の推論は GPU/ドライバ/cuDNN のバージョンによって浮動小数点の下位ビットが
変わりうる。したがってゴールデン比較は完全一致ではなく **許容誤差付き** で行う。
座標は 0.5 px、角度は 1 度を目安とし、ID の割り当ては完全一致を要求する。
ID が変わったら、それは本物の回帰。
