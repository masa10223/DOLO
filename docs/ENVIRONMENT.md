# 環境構築

DOLO は **3種類の環境**で動く必要がある。用途によって必要な依存が違うので、
インストール方法も分けてある。

| 環境 | 用途 | GPU | ロックファイル |
|---|---|---|---|
| **サーバー** | 学習・大量推論 | CUDA | `requirements/lock-linux.txt` |
| **Mac** | 開発・解析・少量推論 | MPS | `requirements/lock-macos.txt` |
| **CI / 軽量** | ユニットテストのみ | 不要 | `requirements/core.in` |

パッケージマネージャは [uv](https://docs.astral.sh/uv/) を推奨する。pip より
桁違いに速く、ロックファイルの生成も uv で行っている。もちろん pip でも入る。

---

## 1. uv のインストール

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## 2. 環境を作る

### Mac（開発機）

```bash
cd /path/to/DOLO
uv venv --python 3.10
source .venv/bin/activate
uv pip install -r requirements/lock-macos.txt
uv pip install -r requirements/gui.in
uv pip install --no-deps -e .
```

### GPU サーバー

```bash
cd /path/to/DOLO
uv venv --python 3.10
source .venv/bin/activate
uv pip install -r requirements/lock-linux.txt
uv pip install -r requirements/gui.in
uv pip install --no-deps -e .
```

`lock-linux.txt` は torch 2.0.0（CUDA 11.7 系の nvidia-*-cu11 を同梱）を固定している。
サーバーのドライバがこれより新しい CUDA を要求する場合は、下記「torch の入れ替え」を参照。

### テストだけ動かしたい（GPU も torch も不要）

```bash
uv venv --python 3.10
source .venv/bin/activate
uv pip install -r requirements/core.in pytest
uv pip install -e .
python -m pytest tests/ -q
```

追跡コアは torch に依存していないので、これだけでユニットテストは全て通る。

## 3. 動作確認

```bash
python -m pytest tests/ -q          # ユニットテスト
python -c "from dolo.device import describe_devices; print(describe_devices())"
dolo doctor
```

---

## Docker（サーバー向け・任意）

サーバー環境をそのまま再現したい場合。

```bash
docker build -t dolo:dev .
docker run --gpus all -it --rm \
  -v "$PWD":/workspace \
  -v /path/to/annotations:/data/annotations \
  -v /path/to/video:/data/video \
  dolo:dev bash
```

---

## ロックファイルの更新

依存を変えたら `.in` を編集して再生成する。

```bash
uv pip compile requirements/server.in --python-version 3.10 \
  --python-platform x86_64-unknown-linux-gnu -o requirements/lock-linux.txt

uv pip compile requirements/macos.in  --python-version 3.10 \
  --python-platform aarch64-apple-darwin  -o requirements/lock-macos.txt
```

`--python-platform` を指定しているので、Mac 上にいても Linux 用のロックを生成できる。

---

## 実機環境で見つかった問題

2026-08-02 に両機の `pip freeze` を突き合わせて判明したもの。

### ultralytics のバージョンが機械間でずれていた（対応済み）

| | 旧状態 | ロック後 |
|---|---|---|
| Mac | 8.3.202 | **8.4.19** |
| サーバー | 8.4.19 | 8.4.19 |

推論結果は ultralytics のバージョンに依存する。ずれていると、同じ動画・同じ重みでも
機械によって出力が変わり、**ゴールデン回帰テストが機械ごとに違う答えを出す**。
新しい方（8.4.19）に揃えた。

他にも `pandas` (2.2.3 / 2.2.2)、`scikit-learn` (1.6.1 / 1.5.0)、
`matplotlib` (3.6系 / 3.9.0)、`opencv` (4.8.0.76 / 4.9.0.80) がずれていた。
すべてサーバー側の値に揃えてある。

### opencv-python と opencv-python-headless の同居（要対応）

**両機とも**、両方のパッケージがインストールされている。同じ `cv2` という名前空間を
提供するため、どちらが読み込まれるかはインストール順に依存し不定になる。
headless 版には GUI 表示機能（`cv2.imshow` 等）が無いので、GUI を作る段階で
表示が突然動かなくなる典型的な原因になる。

ロックファイルでは `opencv-python` のみを指定している。既存環境では明示的に片方を消すこと:

```bash
pip uninstall -y opencv-python-headless
```

### imgaug が実質的に終わっている（Phase 2 以降で対応）

`imgaug==0.4.0` は 2020 年 2 月が最後のリリースで、以後メンテナンスされていない。

- numpy 2.x では動作しない（削除された `np.bool` 等を使用）
- 現在は numpy 1.26.4 に固定しているので問題は顕在化していない
- **OSS 公開後にユーザーが新しい numpy で入れると壊れる**

両機に既に `albumentations==2.0.8` が入っている。学習データ生成の augmentation を
albumentations へ移すのが現実的な解決策。`functions.py` の `apply_augmentation()` が
対象で、キーポイント変換のAPIは albumentations にも同等のものがある。

### matplotlib 3.10 で削除された API を使用（要対応）

`functions_deepsort.py` の `annotate_frame_with_keypoints()` が
`fig.canvas.tostring_rgb()` を使っている。これは matplotlib 3.8 で非推奨、
**3.10 で削除**された。現在は 3.9.0 に固定しているので動くが、移植時に
`buffer_rgba()` ベースへ書き換える必要がある。

なお同関数は**1フレームごとに matplotlib の Figure を生成・破棄している**ため
極端に遅い。同じファイルに cv2 だけで描く `annotate_frame_with_keypoints_and_angle()`
が定義されているが呼ばれていない。そちらへ寄せれば描画は大幅に速くなる。

### torch 2.0.0 は古い（将来の課題）

サーバーの torch 2.0.0 は 2023 年 3 月リリース。新しい GPU（Ada / Hopper 以降）では
性能が出ないか動かない。OSS 公開時には torch 2.4 以降を推奨したいが、
**モデルの出力が変わりうる**ため、ゴールデン回帰テストが整うまでは上げない。
