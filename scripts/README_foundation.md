# Foundation パイプライン：できることと実行例

## 何ができるか

このパイプラインは **全アノテーションCSVをまとめて、動画単位で train/val を分け、YOLO Pose の基盤モデルを一括で学習**するためのものです。

| できること | 説明 |
|------------|------|
| **1. マニフェスト作成** | `annotations` 配下の `annotation_*_manual.csv` を走査し、全フレームの索引 `all_samples.csv` と動画ID一覧（`all_videos.txt`）を生成する。 |
| **2. 動画単位の train/val 分割** | 同一動画が train と val に混在しないよう、動画ID単位で `train_videos.txt` / `val_videos.txt` に分割する（`--val_ratio` と `--seed` で再現可能）。 |
| **3. YOLO用データ生成** | **TIFF のみ**を使用（動画は不要）。`--tiffs_root`（既定: `annotations/overfittings/tiffs`）配下の `<video_id>/<frame_idx>.tif` と CSV から、YOLO Pose 用の画像（jpg）とラベル（txt）を生成。発見した全アノテーションCSV・全TIFFを利用。train は水増し（augment）、val は水増しなし。動画ごとに一時出力してから `train_annotations/<run_name>/` と `val_annotations/<run_name>/` にマージする。 |
| **4. YAML 生成** | 上記 train/val ディレクトリを指す `yamls/<run_name>.yaml` を生成（絶対パスで記述）。 |
| **5. 学習** | 指定した YOLO Pose の重みを読み、上記 YAML で学習。重みとログは `weights/<run_name>/` に保存。 |

**モード**でどこまでやるかを選べます。

- **manifest** … 1 と 2 だけ（CSV・動画IDリスト作成）
- **yolo** … 3 と 4 だけ（YOLOデータ＋YAML。manifests が既にある前提）
- **train** … 5 だけ（既存 YAML で学習）
- **all** … 1 → 2 → 3 → 4 → 5 を一括で実行

---

## 実行例

いずれも **`scripts/` をカレントにした状態**で実行してください。

```bash
cd /inthdd/tsutsumi/drosophila/DOLO/scripts
```

### シェルで実行（バックグラウンド・ログ付き）

- ログは `../annotations/foundation/logs/train_<run_name>_<日時>.out` に出力されます。
- 環境変数でパラメータを上書きできます。

#### 例1: 一括実行（manifest → YOLOデータ＋YAML → 学習）

```bash
./foundation_train.sh
```

- デフォルト: `MODE=all`, `RUN_NAME=foundation_v1`, `EPOCHS=20`, `GPU_IDS="0 1"` など。

#### 例2: スモーク（5 epoch だけ回す）

```bash
MODE=all EPOCHS=5 RUN_NAME=foundation_v1_smoke ./foundation_train.sh
```

#### 例3: manifest だけ（索引と train/val 動画IDだけ作る）

```bash
MODE=manifest RUN_NAME=foundation_v1 ./foundation_train.sh
```

#### 例4: YOLOデータ＋YAMLだけ（manifests は既にある前提）

```bash
MODE=yolo RUN_NAME=foundation_v1 ./foundation_train.sh
```

#### 例5: TIFF の場所を指定（既定: overfittings/tiffs）

```bash
TIFFS_ROOT=/path/to/other/tiffs ./foundation_train.sh
```

#### 例6: 学習だけ（既に YAML とデータがあるとき）

```bash
MODE=train RUN_NAME=foundation_v1 EPOCHS=50 ./foundation_train.sh
```

#### 例7: GPU を 1 枚だけ、検証比率 0.2

```bash
GPU_IDS="0" VAL_RATIO=0.2 ./foundation_train.sh
```

---

### Python で直接実行

細かくオプションを指定したいときは、`foundation_pipeline.py` を直接呼びます。

#### 例1: 一括（all）

```bash
python3 foundation_pipeline.py --mode all --run_name foundation_v1 \
  --load_model_path ./yolo11x-pose.pt \
  --tiffs_root /inthdd/tsutsumi/drosophila/DOLO/annotations/overfittings/tiffs \
  --val_ratio 0.2 --seed 42 --epochs 20 --gpu 0 1
```

#### 例2: manifest だけ

```bash
python3 foundation_pipeline.py --mode manifest --run_name foundation_v1 \
  --annotations_root ../annotations --foundation_root ../annotations/foundation \
  --val_ratio 0.2 --seed 42
```

#### 例3: YOLOデータ＋YAMLだけ

```bash
python3 foundation_pipeline.py --mode yolo --run_name foundation_v1 \
  --foundation_root ../annotations/foundation \
  --tiffs_root /inthdd/tsutsumi/drosophila/DOLO/annotations/overfittings/tiffs \
  --target_size 360
```

#### 例4: 学習だけ

```bash
python3 foundation_pipeline.py --mode train --run_name foundation_v1 \
  --load_model_path ./yolo11x-pose.pt --epochs 50 --gpu 2 3
```

---

## 前提条件

- **アノテーションCSV**: `annotations_root` 配下の `**/csvs/annotation_*_manual.csv` を**すべて**使用。列 `frame_idx`, `Head.x`, `Head.y`, `mid.x`, `mid.y`, `Tail.x`, `Tail.y` があること。
- **TIFF**: `--tiffs_root`（既定: `/inthdd/tsutsumi/drosophila/DOLO/annotations/overfittings/tiffs`）配下に `<video_id>/<frame_idx>.tif` を配置。**動画は不要**（TIFF がそのまま学習用画像として使われる）。
- **学習時**: `yolo` または `all` で YAML と train/val 画像が生成済みであること（`MODE=train` のときは既存 YAML を参照）。

---

## 生成物の場所（run_name = foundation_v1 の例）

| 中身 | パス |
|------|------|
| 全サンプル索引 | `annotations/foundation/manifests/all_samples.csv` |
| 動画ID一覧 | `annotations/foundation/manifests/all_videos.txt` |
| 学習用動画ID | `annotations/foundation/manifests/train_videos.txt` |
| 検証用動画ID | `annotations/foundation/manifests/val_videos.txt` |
| 学習用画像・ラベル | `annotations/foundation/train_annotations/foundation_v1/` |
| 検証用画像・ラベル | `annotations/foundation/val_annotations/foundation_v1/` |
| データ設定YAML | `annotations/foundation/yamls/foundation_v1.yaml` |
| 学習済み重み・ログ | `annotations/foundation/weights/foundation_v1/` |
| 実行ログ（シェル利用時） | `annotations/foundation/logs/train_foundation_v1_<日時>.out` |
