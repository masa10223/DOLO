# DOLO GUI 化 設計書

作成日: 2026-08-02 / 対象ブランチ: `dev`

---

## 0. 決定事項

| 項目 | 決定 |
|---|---|
| 実行環境 | ローカルPC と リモートGPUサーバー の両方 |
| 配布対象 | 論文公開と同時に OSS として一般公開 |
| v1 スコープ | 推論 + トラッキング + 出力選択ダイアログ + 結果ビューア + 手動ID修正 |
| v2 以降 | 解析(contacts/interactions/RL)、アノテーション作成UI、転移学習UI |
| GUI スタック | **NiceGUI**（内部が FastAPI のブラウザUI） |

### なぜ NiceGUI か

- **ブラウザUI一択の理由**: リモートGPUサーバー上で動かす要件がある。デスクトップアプリ(PySide6/Qt)も napari プラグインも SSH 越しには実用にならない。ブラウザUIなら `dolo gui --port 8080` してポートフォワードするだけで、ローカルでもリモートでも同じものが動く。
- **NiceGUI を選ぶ理由**:
  1. 実体が FastAPI。将来 React フロントに差し替えてもバックエンドが生き残る（技術的な後戻りが効く）。
  2. Python のみ。HTML/JS/npm ビルド工程の学習コストがゼロ。
  3. `ui.interactive_image` が画像上の SVG オーバーレイとクリックイベントに対応しており、手動ID修正UIが素直に作れる。Gradio はここが作れないため不採用。
- **不採用**: Gradio(レイアウト自由度不足)、Streamlit(状態管理が長時間ジョブに不向き)、PySide6/napari(リモート不可)、React 自前実装(初手としては学習コスト過大)。

---

## 1. 現状のアーキテクチャ

論理的には5層あるが、**層の境界がコードではなく「スクリプトの実行順序」として存在**している。

| 層 | 実体 | インターフェース |
|---|---|---|
| データ準備 | `functions.py` の `create_yolo_annotations_with_mask` / `_from_images` / `apply_augmentation` / `create_yolo_pose_yaml` | import |
| 学習 | `overfitting_pipeline.py`（単一動画）<br>`foundation_pipeline.py`（複数動画統合, mode=manifest/yolo/train/all） | argparse CLI + `.sh` |
| 推論・追跡 | `drosophila_predict.py` → `functions_deepsort.py`<br>`FixedIDTracker`(Kalman + ID固定割当 + head/tail反転補正) | argparse CLI + `.sh` |
| 解析 | `excecute.py` → `summarize_and_plot_{contacts,interactions,reinforcement_learn}.py` | argparse CLI |
| 手動修正 | `notebooks/Manual_correction.ipynb` ほか | Jupyter 手動実行 |

**追跡の出力スキーマ**（`functions_deepsort.py:559`）:

```
Frame, ID, Head_X, Head_Y, Middle_X, Middle_Y, Tail_X, Tail_Y,
Angle, DistMoved, Confidence, TimeSinceUpdate
```

このスキーマは素直で、GUI の中核データ構造としてそのまま採用できる。設計上の資産。

---

## 2. GUI化のブロッカー

### A. パッケージになっていない

`pyproject.toml` / `requirements.txt` / テスト / CI が一切ない。`cd scripts` 前提の相対パス実行。

→ GUI から `import` できない。**最初にやるべきはここ。**

### B. 絶対パス・環境のハードコード

| 場所 | 内容 |
|---|---|
| `functions.py:439-440` | YAML の train/val を `/cellpose/scripts/...` で生成 |
| `overfitting_pipeline.py:85`, `drosophila_train.py:6` | `settings.update({"datasets_dir": "/cellpose/scripts"})` |
| `foundation_pipeline.py:623` | `/inthdd/tsutsumi/drosophila/DOLO/annotations/overfittings/tiffs` |
| `overfitting_pipeline.py:14,32` | `../video/whi-DM/{unique_name}.avi` |
| `functions_deepsort.py:509`, `drosophila_predict.py:102` | `device="cuda:3"` |

→ 他人のPCでは1行も動かない。設定オブジェクトに置換必須。

### C. ロジック・I/O・描画が単一関数に一体化

`process_video_to_gif_with_angles()` が **推論 + 追跡 + 描画 + GIF書き + MOV書き + CSV書き** を1つのループで同時実行し、`print()` でログを出す。GIF/MOV/CSV は全て必須引数。

→ 「出力をダイアログで選択」を実現するには、**追跡結果を返すこと**と**書き出すこと**の分離が必須。副次的に、可視出力を選ばなければ描画処理をスキップでき、CSVのみ出力時は高速化する。

### D. 重複実装

- `process_video_to_gif_with_angles` … `functions.py:896` / `functions_gpt.py:224` / `functions_deepsort.py:500` の3実装（現用は deepsort 版のみ）
- `calculate_angle_between_vectors` … 4箇所
- `annotate_frame_with_keypoints` … 3箇所

→ 外部から「どれが正か」が判別できない。GUI から呼ぶ正典を1つ決めて残りを削除する。

### E. 長時間処理の実行基盤がない

数千フレームの推論も学習も同期実行前提。進捗通知・中断・ログ取得の口がない。

→ **別プロセス実行 + 進捗コールバック + キャンセル + ログのストリーミング**は完全な新規開発。GUIの成否を左右する最重要コンポーネント。

### F. 解析層が自分の実験デザイン専用

`select_frames()` / `select_frames_of_contact_events()` が control/mutant のファイル命名規則とディレクトリ構造を前提にしており、出力先は `Fig_paper/` 固定。

→ 汎用化には「グループをGUIで定義する」層が必要。v2 送り。

### G. 転移学習の入口がない

学習には CSVアノテーション + TIFFフレームのペアが必要（foundation は TIFF のみ使用）。それを作るUIが存在しない。

→ 「動画からフレーム抽出 → head/mid/tail の3点をクリックしてラベル」のアノテータが実質必須。単独で最大の開発量。v2 送り。

### H. モデル重みの配布手段がない

`yolo11x-pose.pt` も `best.pt` も `.gitignore` 対象。

→ 「学習済みデータがあってそれをもとに動く」という要件の前提が、現状では満たせない。**Zenodo で重みを公開して DOI を取り、初回起動時にチェックサム付きで自動DL**するのが定石（論文からも引用でき一石二鳥）。

### I. テスト0件

リファクタで追跡結果が変わっても検知できない。**リファクタ前に回帰テストを用意する**のが順序。

---

## 3. 目標アーキテクチャ

```
dolo/
├── core/
│   ├── config.py        # Project 設定（パス解決の一元化）← B を解消
│   ├── project.py       # プロジェクトフォルダの概念
│   ├── device.py        # GPU 検出・選択
│   └── registry.py      # モデル重みの取得/キャッシュ/検証 ← H
├── data/                # アノテーション→YOLO変換, augmentation
├── train/               # overfit / foundation（進捗コールバック対応）
├── track/
│   ├── tracker.py       # FixedIDTracker（現行を移設）
│   └── session.py       # track_video() … 結果を流すだけ。書き出さない ← C
├── export/              # Sink 群 ← 「出力選択ダイアログ」の受け皿
│   ├── csv.py  gif.py  mov.py  json.py  hdf5.py
├── correct/             # 手動修正の操作ログとその適用 ← 手動ID修正UIの土台
├── analyze/             # contacts / interactions / RL（グループ定義を引数化）← F
├── jobs/                # ジョブ実行・進捗・キャンセル・ログ ← E
├── cli/                 # 既存CLI互換（今の .sh を壊さない）
└── gui/                 # NiceGUI。薄いガワに保つ
```

**原則: GUI は薄く保つ。処理を `gui/` の中に書いた時点で詰む。**
CLI と GUI が同じ関数を呼ぶ状態を維持すれば、テストも書けるし技術選択も後から変えられる。

### 3.1 追跡の Sink 設計（出力選択ダイアログの実体）

10万フレームの動画では、描画済みフレームを全てメモリに保持できない。したがって「結果をまとめて返してから書く」設計は不可。**ストリーミング + Sink** にする。

```python
@dataclass(frozen=True)
class TrackRecord:
    track_id: int
    head: tuple[float, float]
    mid: tuple[float, float]
    tail: tuple[float, float]
    angle: float
    dist_moved: float
    confidence: float
    time_since_update: int

@dataclass
class FrameResult:
    frame_idx: int
    tracks: list[TrackRecord]

class Sink(Protocol):
    needs_image: bool           # 描画が必要か（GIF/MOV=True, CSV=False）
    def open(self, meta: VideoMeta) -> None: ...
    def write(self, fr: FrameResult, image: np.ndarray | None) -> None: ...
    def close(self) -> None: ...

def track_video(
    video_path: Path,
    model_path: Path,
    params: TrackParams,
    sinks: list[Sink],
    device: str,
    progress: Callable[[int, int], None] | None = None,
    cancel: threading.Event | None = None,
) -> TrackingSummary: ...
```

- GUI の出力選択ダイアログは、**チェックされた項目に対応する Sink を組み立てて渡すだけ**になる。
- `any(s.needs_image for s in sinks)` が False なら描画処理を丸ごとスキップ（CSVのみ出力が大幅に高速化）。
- 新しい出力形式（HDF5, SLEAP互換, DeepLabCut互換）は Sink を1つ足すだけで追加できる。

### 3.2 手動修正の設計（非破壊・操作ログ方式）

追跡CSVを直接書き換えず、**操作の列**として保存し、必要時に元データへ適用する。

```yaml
# results/<video_id>/corrections.yaml
- {op: swap_ids,       ids: [2, 3],   frames: [1200, 1830]}
- {op: flip_head_tail, id: 4,         frames: [500, 520]}
- {op: set_keypoint,   id: 1, frame: 903, point: head, xy: [123.0, 45.5]}
- {op: delete_track,   id: 5,         frames: [0, 3000]}
```

```python
def apply_corrections(raw: pd.DataFrame, ops: list[Op]) -> pd.DataFrame
```

- **Undo/Redo が操作リストの push/pop で自動的に手に入る。** GUI で最も面倒な機能がタダになる。
- 生データが常に残るので、修正のやり直し・修正内容の再現性・査読対応が容易。
- 既存の `manual_correction/*_change_log.csv` は事実上これを手作業でやっているので、その置き換えになる。

### 3.3 ジョブ実行層

```python
class Job:
    id: str
    kind: Literal["track", "train", "analyze"]
    state: Literal["queued", "running", "done", "failed", "cancelled"]
    progress: float
    log: deque[str]
```

- 実処理は `multiprocessing.Process` で別プロセス実行（CUDA コンテキストとGIL回避、強制終了が可能）。
- 進捗・ログは `multiprocessing.Queue` 経由で親へ。
- キャンセルは `multiprocessing.Event`。`track_video` のループ先頭で確認する。
- NiceGUI 側は `ui.timer` でポーリングして進捗バーとログを更新。

---

## 4. ロードマップ

### Phase 0 — 基礎工事（GUIコードは1行も書かない）

1. `pyproject.toml` 作成、`dolo` パッケージ化、`pip install -e .` を通す
2. 依存バージョンの固定（`ultralytics`, `torch`, `opencv-python`, `imageio`, `pandas`, `imgaug`…）
3. **回帰テストの作成** — 小さな合成動画（数十フレーム）で `process_video_to_gif_with_angles` を実行し、出力CSVをゴールデンファイルとして固定。以降の全リファクタはこれを壊さないことで検証する
4. 重複実装の削除 — `functions_gpt.py` と `functions.py:896` の追跡実装を削除し、`functions_deepsort.py` 版を正典とする
5. `functions.py`(3151行) の分割 — データ準備 / 描画 / 解析ユーティリティへ

### Phase 1 — コアAPI分離

6. `core/config.py`: Project 概念の導入。全ハードコードパスを置換
   ```
   my_project/
   ├── dolo.yaml
   ├── videos/  annotations/  models/  results/
   ```
7. `core/device.py`: GPU 自動検出（`cuda:3` 固定を廃止、CPU/MPS フォールバック）
8. `track/session.py` + `export/`: 3.1 の Sink 設計へ移行
9. `jobs/`: 3.3 のジョブ層
10. `core/registry.py` + Zenodo への重み公開、自動DL

### Phase 2 — GUI v1（推論・追跡）

11. NiceGUI アプリの骨格、`dolo gui` コマンド
12. プロジェクトを開く / 動画を選ぶ / モデルを選ぶ
13. パラメータ設定パネル（`max_id`, `conf`, `dist_thresh`, `max_missing_frames`, `head_tail_jump_thresh`, `overlap_thresh`, 開始/終了フレーム）
14. **出力選択ダイアログ**（CSV / GIF / MOV / JSON をチェックボックスで選択 → Sink 組み立て）
15. 実行・進捗バー・ログ表示・キャンセル

### Phase 3 — GUI v1（ビューア + 手動ID修正）

16. フレームスクラバ + キーポイント重畳表示（`ui.interactive_image` + SVGオーバーレイ）
17. 軌跡プロット、ID別の色分け、ID消失区間のハイライト
18. `correct/` 実装 + 修正操作UI（ID入替 / head-tail反転 / キーポイント移動 / トラック削除）
19. Undo/Redo、修正後CSVのエクスポート

### Phase 4 — v2

20. 解析層の汎用化（グループ定義UI）
21. アノテーション作成UI（3点クリック）
22. 転移学習UI（データセット構築 → 学習ジョブ → 学習曲線表示）

### Phase 5 — OSS公開整備

23. ドキュメント（インストール、チュートリアル、サンプルデータ）
24. GitHub Actions CI、Linux/macOS/Windows での動作確認
25. サンプル動画 + 学習済み重みの Zenodo 公開（DOI取得）

---

## 5. 参考にすべき先行実装

| プロジェクト | 参考になる点 |
|---|---|
| **SLEAP** | 同じ動物ポーズ追跡。アノテーションUI、ID修正UI、学習UIの設計。GUIとコアの分離が綺麗 |
| **DeepLabCut** | プロジェクトフォルダ + `config.yaml` 方式。Phase 1 の設計はこれに倣うのが安全 |
| **idtracker.ai** | 多個体ID維持の問題設定が最も近い |
| **napari** | レイヤーモデル（画像 / 点 / トラック）の抽象化。ビューアの概念設計 |

特に **DeepLabCut のプロジェクト構成** と **SLEAP の GUI/コア分離** の2点は、着手前に実際にコードを読む価値がある。

---

## 6. 最初の3コミット（具体的な着手点）

1. `pyproject.toml` + `dolo/` へのファイル移動 + `pip install -e .` が通る状態
2. `tests/test_tracking_regression.py` — 合成動画でのゴールデンテスト
3. `functions_gpt.py` 削除 + `functions.py` 内の重複追跡実装削除（テストが守る）

この3つが終われば、以降のリファクタは安全に進められる。
