# DOLO GUI

DOLO GUI はローカルPCとリモートGPUサーバーの両方で同じように動くブラウザUIです。
動画の受け付け、default重みの選択、追跡ジョブ、進捗・中断、結果集計、成果物の取得までを
`dolo` パッケージの公開APIへ接続しています。

## 1. インストール

Python 3.10–3.12 を推奨します。先にOS別の推論環境を入れ、GUIを追加します。

```bash
# macOS (Apple Silicon)
uv venv --python 3.10
source .venv/bin/activate
uv pip install -r requirements/lock-macos.txt

# NVIDIA Linux の場合は上のlockを requirements/lock-linux.txt に変更

uv pip install -r requirements/gui.in
uv pip install --no-deps -e .
dolo doctor
```

`dolo doctor` の `Result` が `READY` になれば、動画推論に必要なものが揃っています。
`SETUP REQUIRED` の場合は表示された missing 項目かモデルパスを直してください。

## 2. 起動

```bash
dolo gui
```

ブラウザが自動で開かない場合は `http://127.0.0.1:8080` を開きます。

```bash
# ポート、モデル、データ保存先を明示する例
dolo gui --port 8090 --model /models/dolo-default.pt --data-dir /data/dolo
```

### `address already in use` と表示された場合

次のエラーは、別のプロセスがすでに8080番ポートを使用していることを示します。

```text
ERROR: [Errno 48] error while attempting to bind on address ('127.0.0.1', 8080): address already in use
```

もっとも簡単な対処は、空いている別のポートで起動することです。

```bash
dolo gui --port 8090
```

この場合は `http://127.0.0.1:8090` を開きます。以前起動したDOLO GUIのターミナルが残っている
場合は、そのターミナルで `Ctrl+C` を押してください。macOSまたはLinuxでは、8080番を使用中の
プロセスを次のコマンドで確認できます。

```bash
lsof -nP -iTCP:8080 -sTCP:LISTEN
```

表示されたPIDが終了してよいプロセスのものだと確認してから、通常終了させます。

```bash
kill <PID>
```

ポートが解放されたら、もう一度 `dolo gui` を実行します。不明なプロセスは終了せず、別のポートを
使用してください。

環境変数でも指定できます。

| 変数 | 用途 | 既定 |
|---|---|---|
| `DOLO_MODEL_PATH` | defaultモデル重み | 自動探索 |
| `DOLO_DATA_DIR` | upload、thumbnail、runの保存先 | リポジトリの `.dolo/` |
| `DOLO_MAX_JOBS` | 同時ワーカー数 | `1` |

GPUメモリ競合を避けるため、`DOLO_MAX_JOBS=1` を推奨します。複数の依頼は順番に実行されます。

## 3. リモートGPUサーバー

GUIをサーバーのloopbackへbindし、SSHで転送します。認証なしで外部公開しないでください。

```bash
# GPUサーバー側
dolo gui --host 127.0.0.1 --port 8080 --no-open

# 手元PC側
ssh -L 8080:127.0.0.1:8080 user@gpu-server
```

手元PCの `http://127.0.0.1:8080` から操作できます。「サーバーパス」タブへ入力するパスは
GPUサーバー側から見える絶対パスです。手元の動画を渡す場合は「アップロード」タブを使います。

## 4. Docker

```bash
docker build -t dolo:gui .
docker run --rm --gpus all -p 127.0.0.1:8080:8080 \
  -v "$PWD/.dolo:/data" \
  -v "/absolute/path/to/best.pt:/models/default.pt:ro" \
  -e DOLO_MODEL_PATH=/models/default.pt \
  dolo:gui
```

ホストの `127.0.0.1` にだけ公開しています。リモートDockerの場合もSSH転送を併用してください。

## 5. 1回の推論で作られるもの

各実行は `<DOLO_DATA_DIR>/runs/<timestamp>-<video>-<id>/` へ分離されます。

| ファイル | 内容 |
|---|---|
| `<video>.csv` | 既存解析コードと互換な軌跡 |
| `<video>.jsonl` | 検出ゼロのフレームを含むフレーム単位結果 |
| `<video>_pose.mp4` | Head ◯  Middle ×  Triangle △ とIDを重ねた姿勢動画 |
| `<video>_angle.mp4` | 姿勢と角度値を重ねた角度確認動画（選択時のみ） |
| `<video>_center_track.mp4` | 中部の軌跡動画（選択時のみ、残像frame数を指定可能） |
| `<video>.gif` | 短い共有用プレビュー（選択時のみ） |
| `run.json` | 入力、モデル、全パラメータ、進捗、要約、成果物一覧 |

MP4はH.264で作成され、推論完了後はGUI内の「VIDEO PREVIEW」からそのまま再生できます。
中心軌跡の残像frame数は「詳細設定」で指定し、`0` は全履歴を残す設定です。中断時も、
それまでに書かれたファイルを正しく閉じて `run.json` を残します。失敗時はエラーとログを
GUIに表示し、次のジョブを投入できます。

## 6. defaultモデル

探索順は次の通りです。

1. `dolo gui --model ...`
2. `DOLO_MODEL_PATH`
3. リポジトリ直下の `best.pt`
4. `models/default.pt` / `models/best.pt`
5. `<DOLO_DATA_DIR>/models/default.pt`
6. `~/.cache/dolo/models/default.pt`

明示パスが間違っている場合、別の重みへ黙ってフォールバックしません。これは実験の再現性を
守るためです。公開用の重みURLとSHA-256が決まった段階で、チェックサム付き自動取得を追加できます。

## 7. 現在のスコープ

この版は推論・追跡・結果確認が対象です。manual annotationと再学習UIは次の段階で、現在の
`JobManager` に `annotate` / `train` ジョブを追加する形で実装できます。

## 8. GUIの文言を変更する

画面に表示する見出し、説明、ボタン名は `dolo/gui/app.py` にまとまっています。
出力チェック欄の名称と説明だけは `dolo/export.py` の `AVAILABLE_FORMATS` を編集します。
変更後に開発用の `dolo gui --reload` で起動すると、保存時に画面へ反映されます。
